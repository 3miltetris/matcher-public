"""
browser_agent.py — Claude driving a headless Chromium over one funding-source
site (Stage 10).

Streamlit-free, shared by ``jobs/deep_research_job.py`` and (for the single-site
test button) ``views/funding_sources.py``.

Why a browser and not ``web_fetch``: most of the master list is OTA consortium
sites whose solicitations sit one or two clicks behind a JS-rendered listing
page, and several rows carry free-form instructions like "click through each
challenge and copy the text". A fetch tool cannot follow those; a real browser
with a numbered link list can.

Design notes that are load-bearing:

* **Manual agentic loop**, not ``client.beta.messages.tool_runner``. The repo
  pins ``anthropic>=1.3.0,<2`` precisely because an unpinned major silently
  broke production once (see CLAUDE.md, SDK version pins); the tool runner is a
  beta surface and is not worth that exposure here. For the same reason this
  module never passes ``temperature`` / ``top_p`` / ``top_k``.
* **Every site is budgeted** — tool calls, pages, and wall clock. A site that
  blows a budget is not an exception: the loop forces one final
  ``report_findings`` call so whatever was found so far is still returned, with
  a ``stopped_early`` reason attached.
* **Navigation is href-first.** ``click`` with a link number navigates to that
  link's resolved href rather than dispatching a DOM click, which is far more
  reliable on listing pages. A ``text`` argument is available for genuine
  JS-only buttons.
"""

import asyncio
import re
import time
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

MODEL = 'claude-sonnet-4-6'

# Per-site budgets. Deliberately tight: one pathological site must not be able
# to spend the whole run's time or token budget.
DEFAULT_MAX_TOOL_CALLS = 25
DEFAULT_MAX_PAGES      = 12
DEFAULT_SITE_TIMEOUT_S = 240

PAGE_CHAR_CAP  = 12_000
MAX_LINKS      = 120
NAV_TIMEOUT_MS = 30_000
MAX_TOKENS     = 16_000

# How long get_email_code waits for a forwarded 2FA code to land, and how often
# it re-checks the mailbox. A code email usually arrives in seconds, but a
# filter-forward can lag — 90s covers it without stalling the site's budget.
_OTP_POLL_TOTAL_S    = 90
_OTP_POLL_INTERVAL_S = 6

# Per-MTok list prices, for the cost figure reported in status.json.
_PRICING = {
    'claude-sonnet-4-6': (3.00, 15.00),
    'claude-haiku-4-5':  (1.00, 5.00),
    'claude-opus-4-8':   (5.00, 25.00),
}

# Cached input is billed at a fraction of the normal input rate; writing an
# entry costs a premium. Getting these wrong in the cost figure would hide
# whether caching is working at all.
_CACHE_READ_MULT  = 0.10
_CACHE_WRITE_MULT = 1.25

_SKIP_SCHEMES = ('mailto:', 'tel:', 'javascript:', 'data:', '#')
_STRIP_TAGS   = ('script', 'style', 'noscript', 'svg', 'iframe')

# Hints that a site exposes a machine-readable interface. The agent is asked to
# judge this too; these are the deterministic backstop.
_API_HINTS = re.compile(
    r'\b(api\s*(?:docs?|documentation|portal|key|endpoint)|developer\s*portal'
    r'|swagger|openapi|graphql|rest\s*api|json\s*feed|rss\s*feed)\b', re.I
)
_LOGIN_HINTS = re.compile(
    r'\b(sign\s*in|log\s*in|login required|please (?:sign|log) in|create an account'
    r'|you (?:do not|don.t) have (?:an )?organization|member(?:s)?\s*only'
    r'|access denied|unauthorized)\b', re.I
)


# -- Tool schemas -----------------------------------------------------------

TOOLS = [
    {
        'name': 'open_page',
        'description': (
            'Open a URL in the browser and read it. Returns the page title, its '
            'visible text, and a numbered list of the links on it. Use this to '
            'start, and to jump to a URL you already know.'
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'url': {'type': 'string', 'description': 'Absolute URL to open.'},
            },
            'required': ['url'],
        },
    },
    {
        'name': 'click',
        'description': (
            'Follow a link on the current page. Prefer link_number, taken from the '
            'numbered link list. Use text only for buttons that are not links '
            '(accordions, "load more", tabs).'
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'link_number': {'type': 'integer', 'description': 'Number from the link list.'},
                'text':        {'type': 'string',  'description': 'Visible text of a button to click.'},
            },
        },
    },
    {
        'name': 'go_back',
        'description': 'Return to the previous page — use it after reading one '
                       'opportunity to get back to the listing.',
        'input_schema': {'type': 'object', 'properties': {}},
    },
    {
        'name': 'find_on_page',
        'description': (
            'Search the current page text for a phrase and return the matching '
            'lines with context. Useful on long listing pages where the text was '
            'truncated.'
        ),
        'input_schema': {
            'type': 'object',
            'properties': {'query': {'type': 'string'}},
            'required': ['query'],
        },
    },
    {
        'name': 'log_in',
        'description': (
            "Log in to this site using the team's stored credentials. Call this "
            'when a login form or a members-only wall is blocking the listing. '
            'The username and password are filled in automatically from secure '
            'storage — you never see, receive, or type them. Only available when '
            'the site has stored credentials (the task message will say so); if it '
            'does not, this returns an error. If the login form is on a separate '
            'page, open that page first, then call this.'
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'username_field_hint': {
                    'type': 'string',
                    'description': 'Optional: the visible label or placeholder of '
                                   'the username/email field, if the form is unusual.',
                },
                'password_field_hint': {
                    'type': 'string',
                    'description': 'Optional: the visible label or placeholder of '
                                   'the password field, if the form is unusual.',
                },
            },
        },
    },
    {
        'name': 'get_email_code',
        'description': (
            'Fetch a one-time 2FA code that the site emailed, and enter it. Call '
            'this when, after logging in, the site asks for a verification / '
            'security code sent to email. The code is read from the team mailbox '
            'and typed in automatically — you never see or type it. Only works '
            'when the task message says a 2FA mailbox is configured.'
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'otp_field_hint': {
                    'type': 'string',
                    'description': 'Optional: the visible label or placeholder of '
                                   'the code field, if the form is unusual.',
                },
            },
        },
    },
    {
        'name': 'report_findings',
        'description': (
            'Report everything you found and finish. Call this exactly once, at '
            'the end, even if you found nothing (pass an empty opportunities list).'
        ),
        'input_schema': {
            'type': 'object',
            'properties': {
                'opportunities': {
                    'type': 'array',
                    'description': 'Open funding opportunities found on this site.',
                    'items': {
                        'type': 'object',
                        'properties': {
                            'title': {
                                'type': 'string',
                                'description': 'The opportunity title, as published.',
                            },
                            'description': {
                                'type': 'string',
                                'description': (
                                    'The substantive body of the opportunity: what is '
                                    'being funded, the technical scope, eligibility, '
                                    'objectives. Copy the published text closely rather '
                                    'than paraphrasing into one line — this text is '
                                    'embedded and matched against company capabilities, '
                                    'so detail is what makes it useful. Aim for 400-3000 '
                                    'characters.'
                                ),
                            },
                            'topic_number': {
                                'type': 'string',
                                'description': 'Solicitation / topic / reference number if shown.',
                            },
                            'solicitation_title': {
                                'type': 'string',
                                'description': (
                                    'The official name of the solicitation or funding '
                                    'call this opportunity belongs to, copied exactly as '
                                    'the page publishes it (e.g. the BAA, CSO, RPP, '
                                    'challenge or program-announcement name). When the '
                                    'opportunity is one topic inside a larger call, give '
                                    "the larger call's name. Empty if the page never "
                                    'names it — never invent one.'
                                ),
                            },
                            'url':            {'type': 'string', 'description': 'Direct link to this opportunity.'},
                            'open_date':      {'type': 'string', 'description': 'Publication or open date, as shown.'},
                            'close_date':     {'type': 'string', 'description': 'Response deadline, as shown.'},
                            'funding_amount': {'type': 'string', 'description': 'Award value or ceiling, as shown.'},
                            'is_rolling': {
                                'type': 'boolean',
                                'description': (
                                    'True if the opportunity accepts submissions '
                                    'continuously rather than by a single fixed '
                                    'deadline — "open until filled", "rolling basis", '
                                    '"accepted on an ongoing basis", a standing/open '
                                    'BAA or CSO, or a call with no published deadline. '
                                    'False if it has one specific closing date.'
                                ),
                            },
                        },
                        'required': ['title', 'description'],
                    },
                },
                'has_api': {
                    'type': 'boolean',
                    'description': 'True if the site exposes an API, developer portal, '
                                   'or machine-readable feed (JSON/RSS) for opportunities.',
                },
                'api_evidence': {
                    'type': 'string',
                    'description': 'Where you saw it — URL or link text. Empty if has_api is false.',
                },
                'requires_login': {
                    'type': 'boolean',
                    'description': 'True if a login or membership wall blocked the listing.',
                },
                'notes': {
                    'type': 'string',
                    'description': 'Anything the team should know: site reorganised, '
                                   'listing moved, page broken, nothing posted yet.',
                },
            },
            'required': ['opportunities', 'has_api', 'requires_login'],
        },
    },
]

_TERMINAL_TOOL = 'report_findings'


# -- Prompt -----------------------------------------------------------------

_SYSTEM = """\
You are checking one website for OPEN funding opportunities on behalf of a firm \
that writes federal grant and OTA proposals for small technology companies.

What counts as an opportunity: solicitations, RFPs, RFIs, RPPs, project calls, \
BAAs, topic calls, challenges, prize competitions, white-paper calls, and grant \
programs that an organisation could apply to. Consortium "requests for project \
proposals" and state matching-fund programs count.

What does NOT count, and must never be reported: news or press releases, awards \
that have already been made, member spotlights, events and webinars, past \
opportunities whose deadline has clearly passed, and generic "about us" or \
"how to join" pages that describe no specific funding action. A continuously-open \
or rolling opportunity that has NO deadline still counts and should be reported \
(set is_rolling true) — only a clearly-passed fixed deadline disqualifies one.

How to work:
1. Open the starting URL and read it.
2. If it is a listing, open the individual opportunities to read their real \
descriptions. A one-line title from a listing page is not enough — the \
description you report is embedded and matched against company capabilities, so \
it needs the actual scope and technical content. Go back to the listing between \
items.
3. If the page shows nothing fundable, say so and stop. That is a normal, \
useful answer — do not pad the list with near-misses to look productive.
4. Work within your budget. You will be told how many tool calls remain; leave \
yourself one for report_findings. Depth on a few real opportunities beats a \
shallow pass over many.

Always finish by calling report_findings exactly once, including when you found \
nothing.

Also report two things about the site itself:
- has_api: whether it offers an API, a developer portal, or a JSON/RSS feed of \
opportunities. The team wants to integrate directly with those instead of \
browsing them.
- requires_login: whether a login or membership wall stopped you from seeing the \
listing.
"""

# Sent as a content block so it can carry a cache breakpoint. Render order is
# tools -> system -> messages, so a breakpoint here caches the tool definitions
# too. Both are identical for every site, so this entry is also reused across
# sites within the cache TTL.
_SYSTEM_BLOCKS = [{
    'type': 'text',
    'text': _SYSTEM,
    'cache_control': {'type': 'ephemeral'},
}]


def _build_user_message(site: dict, known: list, max_tool_calls: int,
                        credentials: dict = None,
                        has_code_fetcher: bool = False) -> str:
    parts = [
        f"Site: {site.get('name') or site.get('url')}",
        f"Starting URL: {site.get('url')}",
        f"Budget: {max_tool_calls} tool calls, {site.get('max_pages')} pages.",
    ]
    instructions = str(site.get('instructions') or '').strip()
    if instructions:
        parts.append(
            '\nSite-specific instructions from the team — follow these closely:\n'
            + instructions
        )
    if credentials:
        # Login guidance lives in the per-site user message, never in _SYSTEM, so
        # the system/tools cache entry stays byte-identical across every site.
        login_url = str(credentials.get('login_url') or '').strip()
        parts.append(
            '\nThis site has stored login credentials. If a login form or a '
            'members-only wall blocks the listing, call log_in to sign in — the '
            'username and password are filled automatically.'
            + (f' The login form is at {login_url} — open it first if you are not '
               'already on a page with a login form.' if login_url else '')
            + (' If the site then requires a one-time code to continue, call '
               'get_email_code to complete that step.' if has_code_fetcher else '')
        )
    if known:
        listed = '\n'.join(f'- {t}' for t in known)
        parts.append(
            '\nAlready stored from this site in previous runs. Do NOT report these '
            'again; report only opportunities that are not on this list:\n' + listed
        )
    else:
        parts.append('\nNothing has been stored from this site before, so report '
                     'every open opportunity you find.')
    parts.append('\nBegin by opening the starting URL.')
    return '\n'.join(parts)


def parse_page(html: str, base_url: str):
    """HTML -> (visible text, numbered link list).

    Pure, so it can be exercised against fixture HTML without a browser.
    Links are resolved against `base_url` and de-duplicated; a link with no
    visible label and no title/aria-label is dropped, since the model has no way
    to tell what it is.
    """
    soup = BeautifulSoup(html, 'html.parser')
    for tag in soup(list(_STRIP_TAGS)):
        tag.decompose()

    lines = [ln.strip() for ln in soup.get_text('\n').splitlines()]
    text  = '\n'.join(ln for ln in lines if ln)

    links = []
    seen  = set()
    for a in soup.find_all('a', href=True):
        href = a['href'].strip()
        if not href or href.lower().startswith(_SKIP_SCHEMES):
            continue
        absolute = urljoin(base_url, href)
        if urlparse(absolute).scheme not in ('http', 'https'):
            continue
        if absolute in seen:
            continue
        seen.add(absolute)
        label = ' '.join(a.get_text(' ').split())[:110]
        if not label:
            label = (a.get('title') or a.get('aria-label') or '').strip()[:110]
        if not label:
            continue
        links.append({'n': len(links) + 1, 'text': label, 'url': absolute})
        if len(links) >= MAX_LINKS:
            break
    return text, links


# -- Browser session --------------------------------------------------------

class PageSession:
    """One browser context bound to one site, with a small navigation history."""

    def __init__(self, context, max_pages: int, credentials: dict = None,
                 code_fetcher=None):
        self.context    = context
        self.page       = None
        self.max_pages  = max_pages
        self.history    = []
        self.pages_seen = 0
        self.text       = ''
        self.links      = []
        self.url        = ''
        self.title      = ''
        # Bound out-of-band: the password is read only inside log_in() and typed
        # straight into the page. It is never placed in `messages`, a tool
        # result, a log line, or the registry.
        credentials     = credentials or {}
        self.username   = str(credentials.get('username') or '')
        self.password   = str(credentials.get('password') or '')
        self.login_url  = str(credentials.get('login_url') or '').strip()
        # Returns a fresh 2FA code (or None) given the login-attempt timestamp;
        # bound to this site's sender/regex + the shared mailbox by the job. The
        # code, like the password, never enters the conversation or logs.
        self.code_fetcher = code_fetcher
        self._login_ts    = 0.0

    async def _ensure_page(self):
        if self.page is None:
            self.page = await self.context.new_page()
            self.page.set_default_timeout(NAV_TIMEOUT_MS)
        return self.page

    async def _capture(self) -> str:
        """Read the live DOM into text + a numbered link list."""
        page = self.page
        try:
            html = await page.content()
        except Exception as e:
            return f'ERROR: could not read the page ({e}).'

        text, links = parse_page(html, page.url)
        self.text  = text
        self.links = links
        self.url   = page.url
        self.title = (await page.title()) or ''
        return self._render(text, links)

    def _render(self, text: str, links: list) -> str:
        body = _truncate_middle(text, PAGE_CHAR_CAP)
        out = [f'URL: {self.url}', f'TITLE: {self.title}', '', 'PAGE TEXT:', body]
        if links:
            out.append('')
            out.append(f'LINKS ({len(links)}):')
            out.extend(f"[{l['n']}] {l['text']} -> {l['url']}" for l in links)
        else:
            out.append('\nLINKS: none found.')
        return '\n'.join(out)

    async def open_page(self, url: str, count: bool = True) -> str:
        """Open a URL. `count=False` for revisits, which are free.

        Going back to a listing between opportunities is the whole navigation
        pattern here; charging the page budget for it would halve the number of
        opportunities the agent can actually read.
        """
        if count and self.pages_seen >= self.max_pages:
            return ('Page budget spent — no more new pages may be opened. Report '
                    'what you have found so far with report_findings.')
        page = await self._ensure_page()
        try:
            await page.goto(url, timeout=NAV_TIMEOUT_MS, wait_until='domcontentloaded')
            # Give client-rendered listings a moment to populate.
            try:
                await page.wait_for_load_state('networkidle', timeout=5_000)
            except Exception:
                pass
        except Exception as e:
            return f'ERROR: could not open {url} ({type(e).__name__}: {e}).'
        if count:
            self.pages_seen += 1
        if self.url and self.url != url:
            self.history.append(self.url)
        return await self._capture()

    async def click(self, link_number=None, text: str = '') -> str:
        if self.page is None:
            return 'ERROR: no page is open yet — call open_page first.'
        if link_number is not None:
            match = next((l for l in self.links if l['n'] == int(link_number)), None)
            if match is None:
                return (f'ERROR: there is no link [{link_number}] on this page. '
                        f'Valid numbers are 1-{len(self.links)}.')
            return await self.open_page(match['url'])
        if text:
            if self.pages_seen >= self.max_pages:
                return ('Page budget spent — report what you have found so far with '
                        'report_findings.')
            try:
                await self.page.get_by_text(text, exact=False).first.click(timeout=10_000)
                await self.page.wait_for_load_state('domcontentloaded', timeout=NAV_TIMEOUT_MS)
            except Exception as e:
                return f'ERROR: could not click "{text}" ({type(e).__name__}).'
            self.pages_seen += 1
            return await self._capture()
        return 'ERROR: click needs either link_number or text.'

    async def go_back(self) -> str:
        if not self.history:
            return 'ERROR: nothing to go back to.'
        return await self.open_page(self.history.pop(), count=False)

    def find_on_page(self, query: str) -> str:
        if not self.text:
            return 'ERROR: no page is open yet.'
        needle = (query or '').lower().strip()
        if not needle:
            return 'ERROR: find_on_page needs a query.'
        lines = self.text.splitlines()
        hits  = []
        for i, line in enumerate(lines):
            if needle in line.lower():
                start = max(0, i - 1)
                end   = min(len(lines), i + 3)
                hits.append('\n'.join(lines[start:end]))
            if len(hits) >= 20:
                break
        if not hits:
            return f'No matches for "{query}" on this page.'
        return f'{len(hits)} match(es) for "{query}":\n\n' + '\n---\n'.join(hits)

    async def log_in(self, username_field_hint: str = '',
                     password_field_hint: str = '') -> str:
        """Fill and submit the login form with the bound credentials.

        The password is read only here and typed straight into the field; it
        never enters the returned string (parse_page reads visible text + hrefs,
        not input values), the conversation, or any log. Returns the post-login
        page so the agent can carry on.
        """
        if not self.password:
            return 'ERROR: no credentials are configured for this site.'

        if self.page is None:
            if self.login_url:
                await self.open_page(self.login_url)
            else:
                return ('ERROR: no page is open yet — open the login page first, '
                        'then call log_in.')
        page = self.page

        async def _first(*locators):
            for loc in locators:
                try:
                    if await loc.count():
                        return loc
                except Exception:
                    continue
            return None

        # Password field — the anchor of a login form.
        pw_locator = None
        if password_field_hint:
            pw_locator = await _first(
                page.get_by_label(password_field_hint, exact=False).first,
                page.get_by_placeholder(password_field_hint, exact=False).first,
            )
        if pw_locator is None:
            pw_locator = await _first(page.locator('input[type="password"]').first)
        if pw_locator is None:
            return ('ERROR: could not find a password field on this page. If the '
                    'login form is on another page, open that page first.')

        # Username / email field.
        user_locator = None
        if username_field_hint:
            user_locator = await _first(
                page.get_by_label(username_field_hint, exact=False).first,
                page.get_by_placeholder(username_field_hint, exact=False).first,
            )
        if user_locator is None:
            user_locator = await _first(*[
                page.locator(sel).first for sel in (
                    'input[type="email"]',
                    'input[autocomplete="username"]',
                    'input[name*="user" i]', 'input[name*="email" i]',
                    'input[id*="user" i]', 'input[id*="email" i]',
                    'input[type="text"]',
                )
            ])

        try:
            if user_locator is not None and self.username:
                await user_locator.fill(self.username, timeout=10_000)
            await pw_locator.fill(self.password, timeout=10_000)
        except Exception as e:                                # noqa: BLE001
            return f'ERROR: could not fill the login form ({type(e).__name__}).'

        # Mark the attempt so a 2FA code email that arrives after this is the one
        # get_email_code accepts (not a stale code from an earlier attempt).
        self._login_ts = time.time()

        # Submit: a submit control, a button labelled log in / sign in, else Enter.
        submitted = await _first(
            page.locator('button[type="submit"]').first,
            page.locator('input[type="submit"]').first,
        )
        try:
            if submitted is not None:
                await submitted.click(timeout=10_000)
            else:
                btn = page.get_by_role(
                    'button', name=re.compile(r'log\s*in|sign\s*in|continue', re.I)
                ).first
                if await btn.count():
                    await btn.click(timeout=10_000)
                    submitted = btn
                else:
                    await pw_locator.press('Enter', timeout=10_000)
                    submitted = pw_locator
        except Exception as e:                                # noqa: BLE001
            return f'ERROR: filled the login form but could not submit it ({type(e).__name__}).'

        try:
            await page.wait_for_load_state('domcontentloaded', timeout=NAV_TIMEOUT_MS)
            try:
                await page.wait_for_load_state('networkidle', timeout=5_000)
            except Exception:
                pass
        except Exception:
            pass

        if self.pages_seen < self.max_pages:
            self.pages_seen += 1
        return 'Submitted the login form.\n\n' + await self._capture()

    async def get_email_code(self, otp_field_hint: str = '') -> str:
        """Fetch a forwarded 2FA code from the team mailbox and enter it.

        The code is read via the bound code_fetcher (Gmail-backed, built by the
        job), typed into the code field, and submitted. It is never returned in
        the string the model sees or written to any log.
        """
        if self.code_fetcher is None:
            return 'ERROR: no 2FA mailbox is configured for this site.'
        if self.page is None:
            return 'ERROR: no page is open — log in first, then call get_email_code.'
        page = self.page

        async def _first(*locators):
            for loc in locators:
                try:
                    if await loc.count():
                        return loc
                except Exception:
                    continue
            return None

        otp = None
        if otp_field_hint:
            otp = await _first(
                page.get_by_label(otp_field_hint, exact=False).first,
                page.get_by_placeholder(otp_field_hint, exact=False).first,
            )
        if otp is None:
            otp = await _first(*[
                page.locator(sel).first for sel in (
                    'input[autocomplete="one-time-code"]',
                    'input[name*="otp" i]', 'input[name*="code" i]',
                    'input[name*="token" i]', 'input[id*="otp" i]',
                    'input[id*="code" i]', 'input[inputmode="numeric"]',
                    'input[type="tel"]', 'input[type="text"]',
                )
            ])
        if otp is None:
            return ('ERROR: could not find a code field on this page. If the code '
                    'is entered on another page, open that page first.')

        # Poll the mailbox for a code newer than the login attempt. The fetcher
        # is a sync Gmail call, so run it off the event loop while other sites
        # keep browsing.
        since    = self._login_ts or (time.time() - 180)
        loop     = asyncio.get_running_loop()
        code     = None
        deadline = time.monotonic() + _OTP_POLL_TOTAL_S
        while True:
            try:
                code = await loop.run_in_executor(None, self.code_fetcher, since)
            except Exception as e:                            # noqa: BLE001
                return f'ERROR: could not read the 2FA mailbox ({type(e).__name__}).'
            if code or time.monotonic() >= deadline:
                break
            await asyncio.sleep(_OTP_POLL_INTERVAL_S)
        if not code:
            return (f'ERROR: no 2FA code arrived within {_OTP_POLL_TOTAL_S}s — the '
                    'email may not have forwarded to the team mailbox, or the '
                    'sender/pattern is misconfigured.')

        try:
            await otp.fill(str(code), timeout=10_000)
        except Exception as e:                                # noqa: BLE001
            return f'ERROR: found a code but could not enter it ({type(e).__name__}).'

        submitted = await _first(
            page.locator('button[type="submit"]').first,
            page.locator('input[type="submit"]').first,
        )
        try:
            if submitted is not None:
                await submitted.click(timeout=10_000)
            else:
                btn = page.get_by_role(
                    'button',
                    name=re.compile(r'verif|submit|continue|log\s*in|sign\s*in', re.I),
                ).first
                if await btn.count():
                    await btn.click(timeout=10_000)
                else:
                    await otp.press('Enter', timeout=10_000)
        except Exception as e:                                # noqa: BLE001
            return f'ERROR: entered the code but could not submit it ({type(e).__name__}).'

        try:
            await page.wait_for_load_state('domcontentloaded', timeout=NAV_TIMEOUT_MS)
            try:
                await page.wait_for_load_state('networkidle', timeout=5_000)
            except Exception:
                pass
        except Exception:
            pass

        if self.pages_seen < self.max_pages:
            self.pages_seen += 1
        return 'Entered the emailed 2FA code.\n\n' + await self._capture()

    async def close(self):
        try:
            if self.page is not None:
                await self.page.close()
        except Exception:
            pass


def _truncate_middle(text: str, cap: int) -> str:
    """Keep the head and tail, drop the middle.

    Listing pages front-load the opportunities and back-load footer boilerplate,
    but detail pages often put the deadline last — so both ends matter more than
    the middle. Same reasoning as fathom_client.transcript_text.
    """
    if len(text) <= cap:
        return text
    head = int(cap * 0.7)
    tail = cap - head
    return text[:head] + '\n\n[... middle of page omitted ...]\n\n' + text[-tail:]



def roll_cache_breakpoint(messages: list) -> None:
    """Move the conversation cache breakpoint to the newest tool result.

    The API allows at most 4 cache breakpoints per request, so this moves a
    single marker forward rather than accumulating one per turn. Each turn then
    writes a cache entry covering the whole conversation so far and reads the
    one written last turn.

    Assistant turns hold SDK block objects rather than dicts and are skipped —
    the marker always lands on a tool result we constructed ourselves.
    """
    for msg in messages:
        content = msg.get('content')
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    block.pop('cache_control', None)

    for msg in reversed(messages):
        content = msg.get('content')
        if isinstance(content, list) and content and isinstance(content[-1], dict):
            content[-1]['cache_control'] = {'type': 'ephemeral'}
            return


# -- Cost -------------------------------------------------------------------

def _usage_cost(model: str, usage) -> float:
    """Actual spend for one turn, with cached input billed at its real rate."""
    if usage is None:
        return 0.0
    rate_in, rate_out = _PRICING.get(model, _PRICING[MODEL])
    tin    = getattr(usage, 'input_tokens', 0) or 0
    tout   = getattr(usage, 'output_tokens', 0) or 0
    tread  = getattr(usage, 'cache_read_input_tokens', 0) or 0
    twrite = getattr(usage, 'cache_creation_input_tokens', 0) or 0
    return (
        (tin / 1_000_000) * rate_in
        + (tread / 1_000_000) * rate_in * _CACHE_READ_MULT
        + (twrite / 1_000_000) * rate_in * _CACHE_WRITE_MULT
        + (tout / 1_000_000) * rate_out
    )


# -- The loop ---------------------------------------------------------------

def _blank_result(site: dict) -> dict:
    return {
        'source_id':      site.get('source_id', ''),
        'name':           site.get('name', ''),
        'url':            site.get('url', ''),
        'ok':             False,
        'opportunities':  [],
        'has_api':        False,
        'api_evidence':   '',
        'requires_login': False,
        'notes':          '',
        'pages_visited':  0,
        'tool_calls':     0,
        'stopped_early':  None,
        'error':          '',
        'cost_usd':       0.0,
        'input_tokens':   0,
        'output_tokens':  0,
        # Caching fails silently — a varying prefix just means you keep paying
        # full price with no error. These make that visible.
        'cache_read_tokens':  0,
        'cache_write_tokens': 0,
    }


def _clean_opportunities(raw, base_url: str) -> list:
    """Drop junk the model may emit and normalise the fields we store."""
    out = []
    if not isinstance(raw, list):
        return out
    for item in raw:
        if not isinstance(item, dict):
            continue
        title = str(item.get('title') or '').strip()
        desc  = str(item.get('description') or '').strip()
        if not title or len(desc) < 80:
            # A title with no real body cannot be embedded usefully, and a
            # near-empty description is the signature of a listing row the model
            # never actually opened.
            continue
        url = str(item.get('url') or '').strip()
        if url:
            url = urljoin(base_url, url)
        out.append({
            'title':          title[:500],
            'description':    desc,
            'topic_number':   str(item.get('topic_number') or '').strip()[:120],
            'solicitation_title': str(item.get('solicitation_title') or '').strip()[:300],
            'url':            url,
            'open_date':      str(item.get('open_date') or '').strip()[:60],
            'close_date':     str(item.get('close_date') or '').strip()[:60],
            'funding_amount': str(item.get('funding_amount') or '').strip()[:120],
            'is_rolling':     bool(item.get('is_rolling')),
        })
    return out


async def research_site(
    anth,
    browser,
    site: dict,
    known: list = None,
    model: str = MODEL,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
    site_timeout_s: int = DEFAULT_SITE_TIMEOUT_S,
    credentials: dict = None,
    code_fetcher=None,
) -> dict:
    """Walk one site and return what was found. Never raises.

    `site` is a registry row as a dict (url, name, instructions, max_pages,
    source_id). `known` is the list of already-stored titles for this source.
    `credentials`, when given, is `{username, password, login_url}` for a
    login-walled site — bound to the browser session out-of-band and used only
    by the `log_in` tool; the password never enters the conversation or logs.
    `code_fetcher`, when given, is a callable `(login_ts) -> code|None` for a
    site with email 2FA — used only by `get_email_code`; the code never enters
    the conversation or logs either.
    `browser` is a live Playwright Browser; a fresh context is created and
    disposed of here so sites cannot share cookies or state.
    """
    result   = _blank_result(site)
    known    = known or []
    deadline = time.monotonic() + site_timeout_s
    start_url = site.get('url') or ''
    if not start_url:
        result['error'] = 'no url'
        return result

    context = None
    session = None
    try:
        context = await browser.new_context(
            user_agent=('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                        '(KHTML, like Gecko) Chrome/124.0 Safari/537.36 MatcherBot/1.0'),
            viewport={'width': 1440, 'height': 900},
            ignore_https_errors=True,
        )
        session = PageSession(context, int(site.get('max_pages') or DEFAULT_MAX_PAGES),
                              credentials=credentials, code_fetcher=code_fetcher)

        messages = [{
            'role': 'user',
            'content': _build_user_message(site, known, max_tool_calls, credentials,
                                           has_code_fetcher=code_fetcher is not None),
        }]

        calls_used = 0
        while True:
            remaining = max_tool_calls - calls_used
            out_of_budget = (
                remaining <= 1
                or time.monotonic() > deadline
                or session.pages_seen >= session.max_pages
            )
            if out_of_budget and result['stopped_early'] is None:
                result['stopped_early'] = (
                    'timeout' if time.monotonic() > deadline else
                    'page_budget' if session.pages_seen >= session.max_pages else
                    'tool_budget'
                )

            roll_cache_breakpoint(messages)
            kwargs = {
                'model':      model,
                'max_tokens': MAX_TOKENS,
                'system':     _SYSTEM_BLOCKS,
                'tools':      TOOLS,
                'messages':   messages,
            }
            if out_of_budget:
                # Force the report rather than letting the loop die with the
                # findings still inside the model's head.
                kwargs['tool_choice'] = {'type': 'tool', 'name': _TERMINAL_TOOL}

            response = await anth.messages.create(**kwargs)

            result['cost_usd']      += _usage_cost(model, response.usage)
            result['input_tokens']  += getattr(response.usage, 'input_tokens', 0) or 0
            result['output_tokens'] += getattr(response.usage, 'output_tokens', 0) or 0
            result['cache_read_tokens'] += (
                getattr(response.usage, 'cache_read_input_tokens', 0) or 0)
            result['cache_write_tokens'] += (
                getattr(response.usage, 'cache_creation_input_tokens', 0) or 0)

            tool_uses = [b for b in response.content if getattr(b, 'type', '') == 'tool_use']
            if not tool_uses:
                # Model answered in prose without reporting. Nudge once, then
                # the budget check above will force the terminal tool.
                if result['stopped_early'] is None:
                    result['stopped_early'] = 'no_report'
                messages.append({'role': 'assistant', 'content': response.content})
                messages.append({
                    'role': 'user',
                    'content': 'Call report_findings now with whatever you found.',
                })
                calls_used += 1
                if calls_used >= max_tool_calls + 2:
                    result['error'] = 'model never called report_findings'
                    break
                continue

            messages.append({'role': 'assistant', 'content': response.content})

            terminal = next((t for t in tool_uses if t.name == _TERMINAL_TOOL), None)
            if terminal is not None:
                payload = terminal.input if isinstance(terminal.input, dict) else {}
                result['opportunities']  = _clean_opportunities(
                    payload.get('opportunities'), session.url or start_url)
                result['has_api']        = bool(payload.get('has_api'))
                result['api_evidence']   = str(payload.get('api_evidence') or '')[:500]
                result['requires_login'] = bool(payload.get('requires_login'))
                result['notes']          = str(payload.get('notes') or '')[:1000]
                result['ok']             = True
                break

            tool_results = []
            for call in tool_uses:
                calls_used += 1
                args = call.input if isinstance(call.input, dict) else {}
                try:
                    if call.name == 'open_page':
                        out = await session.open_page(str(args.get('url') or ''))
                    elif call.name == 'click':
                        out = await session.click(
                            link_number=args.get('link_number'),
                            text=str(args.get('text') or ''),
                        )
                    elif call.name == 'go_back':
                        out = await session.go_back()
                    elif call.name == 'find_on_page':
                        out = session.find_on_page(str(args.get('query') or ''))
                    elif call.name == 'log_in':
                        out = await session.log_in(
                            username_field_hint=str(args.get('username_field_hint') or ''),
                            password_field_hint=str(args.get('password_field_hint') or ''),
                        )
                    elif call.name == 'get_email_code':
                        out = await session.get_email_code(
                            otp_field_hint=str(args.get('otp_field_hint') or ''),
                        )
                    else:
                        out = f'ERROR: unknown tool {call.name}.'
                except Exception as e:                      # noqa: BLE001
                    out = f'ERROR: {type(e).__name__}: {e}'

                left = max(0, max_tool_calls - calls_used)
                out  = f'{out}\n\n[{left} tool call(s) left]'
                tool_results.append({
                    'type':        'tool_result',
                    'tool_use_id': call.id,
                    'content':     out,
                })

            messages.append({'role': 'user', 'content': tool_results})

        # Deterministic backstop for the two site-level flags: the agent is
        # asked to judge them, but a page it could not read still leaves
        # evidence in the text.
        page_text = session.text or ''
        if not result['has_api'] and _API_HINTS.search(page_text):
            result['has_api']      = True
            result['api_evidence'] = (result['api_evidence']
                                      or 'API/feed wording found in page text')
        if not result['requires_login'] and not result['opportunities'] \
                and _LOGIN_HINTS.search(page_text):
            result['requires_login'] = True

        result['pages_visited'] = session.pages_seen

    except asyncio.CancelledError:
        raise
    except Exception as e:                                   # noqa: BLE001
        result['error'] = f'{type(e).__name__}: {e}'
        if session is not None:
            result['pages_visited'] = session.pages_seen
    finally:
        if session is not None:
            await session.close()
        if context is not None:
            try:
                await context.close()
            except Exception:
                pass

    return result


async def launch_browser(playwright):
    """Chromium with the flags a container needs.

    `--no-sandbox` and `--disable-dev-shm-usage` are required under Cloud Run —
    this mirrors jobs/contact_import_job.py, not lead_importer, which omits both.
    """
    return await playwright.chromium.launch(
        headless=True,
        args=['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu'],
    )
