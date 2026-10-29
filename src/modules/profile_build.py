"""
profile_build.py — build a multi-aspect capability profile from raw text.

Streamlit-free. This is the orchestration that was inline in
views/grant_search.py (`_claude_json` + `_build_profile_from_notes`): drive the
Claude aspect call, embed each aspect/market, run the merges and normalisation,
optionally run the unexplored pass, and assemble a `build_profile_record` row.

It is extracted here so there is ONE implementation shared by:
  * the Grant Search view (a profile built on the spot from pasted notes), and
  * the roadmap pipeline (a profile built from an event form's description +
    optional pitch-deck text),

rather than a second copy. The external clients are injected — `anth` (an
anthropic.Anthropic) and `embed` (a str -> list[float] callable, e.g.
TextProcessor.get_embedding) — so nothing here reads st.secrets or os.environ,
and tests can pass fakes.

The prompt, parsing, merges and record shape are aspect_profile's; a profile
built here is indistinguishable from one client-profile-job writes.
"""

import src.modules.anthropic_utils as au
import src.modules.aspect_profile as ap


def claude_json(anth, model: str, system: str, user_msg: str, parse):
    """One Claude call with one strict-JSON retry, parsed by `parse`.
    Returns (parsed, model_that_answered).

    Same contract as client_profile_job._claude_json and the Grant Search view:
    a truncated response is a hard error rather than a half-parsed profile, and
    a **refusal falls back to the other supported model** — some entirely
    legitimate companies trip a false-positive refusal (measured on a real
    biopharma client), and a refusal is deterministic for the same model and
    material, so the strict-JSON retry can only burn a second call. No
    temperature: the anthropic 1.x SDK removed it, and passing it raises a local
    TypeError (see CLAUDE.md "SDK version pins")."""
    last_err   = None
    others     = [m for m in ap.ASPECT_MODELS if m != model]
    candidates = [model] + others[:1]

    for current in candidates:
        refused = False
        for attempt in range(2):
            content = user_msg if attempt == 0 else (
                user_msg + '\n\nYour previous response was not valid JSON. '
                           'Return ONLY the valid JSON object.'
            )
            resp = anth.messages.create(
                model=current,
                max_tokens=6000,
                system=system,
                messages=[{'role': 'user', 'content': content}],
            )
            if resp.stop_reason == 'max_tokens':
                raise ValueError('Claude hit the output token limit')
            if resp.stop_reason == 'refusal':
                last_err = ValueError(f'{current} declined to answer (stop_reason=refusal)')
                refused  = True
                break          # retrying the same model cannot change a refusal
            try:
                return parse(au.response_text(resp)), current
            except ValueError as e:
                last_err = e
        if not refused:
            # Bad JSON twice is a prompt/material problem, not a model-policy
            # one — another model is no more likely to parse, so stop here.
            break
    raise ValueError(f'invalid response twice: {last_err}')


def build_profile_from_texts(
    anth,
    embed,
    *,
    company_name: str,
    website: str,
    state: str,
    texts: dict[str, str],
    target_aspects: int,
    max_markets: int,
    assess_defense: bool,
    assess_unexplored: bool,
    max_unexplored: int,
    model: str,
) -> tuple[dict, list[str]]:
    """(profile record, warnings). Mirrors client_profile_job's per-client build
    so a profile made here is the same shape as one the job wrote.

    `texts` is {source_key: text}, already capped (e.g. ap.notes_source_texts()
    for pasted notes, or any SOURCES key). `embed(str) -> list[float]`."""
    warnings: list[str] = []
    if not texts:
        raise ValueError('nothing to profile')

    parsed, built_by = claude_json(
        anth, model,
        ap.build_aspect_system(target_aspects, max_markets, assess_defense),
        ap.build_aspect_user_message(
            {'company_name': company_name, 'website': website, 'state': state}, texts
        ),
        ap.parse_aspect_response,
    )
    if built_by != model:
        warnings.append(
            f'{model} declined to answer (refusal) — the profile was built '
            f'with {built_by} instead.'
        )
    summary = parsed['profile_summary']
    aspects = parsed['aspects']
    markets = parsed['markets']

    vectors = [embed(ap.aspect_embed_text(a)) for a in aspects]
    # Fold near-identical aspects together BEFORE market membership is re-derived,
    # so no market is left pointing at an aspect that was merged away.
    aspects, vectors, merges = ap.merge_similar_aspects(
        aspects, vectors, ap.ASPECT_MERGE_THRESHOLD
    )
    warnings += [f'Merged near-identical aspects: {m}' for m in merges]

    market_vectors = [embed(ap.market_embed_text(m)) for m in markets]
    if markets:
        aspects, markets, market_vectors = ap.merge_similar_markets(
            aspects, markets, market_vectors, ap.MARKET_MERGE_THRESHOLD
        )
        by_name = {m['market']: v for m, v in zip(markets, market_vectors)}
        aspects, markets = ap.normalize_markets(aspects, markets, max_markets)
        market_vectors = [by_name[m['market']] for m in markets]

    # Pass 2 — markets the company does NOT serve. Additive and separately
    # guarded: a failure here must never discard the pass-1 work above, which
    # cost a Claude call and every embedding.
    unexplored, unexplored_vectors = [], []
    if assess_unexplored:
        try:
            unexplored, _ = claude_json(
                anth, built_by,
                ap.build_unexplored_system(max_unexplored),
                ap.build_unexplored_user_message(
                    {'company_name': company_name, 'website': website, 'state': state},
                    summary, aspects, markets,
                    # stated_intentions() reads notable_updates off client rows;
                    # ad-hoc text has no such structure, so there are none.
                    [],
                ),
                lambda raw: ap.parse_unexplored_response(
                    raw, markets, aspects, max_unexplored
                ),
            )
            unexplored_vectors = [
                embed(ap.market_embed_text(m)) for m in unexplored
            ]
        except Exception as e:
            unexplored, unexplored_vectors = [], []
            warnings.append(
                f'Unexplored-market pass failed ({type(e).__name__}: {e}) — the '
                'profile is complete apart from its unexplored markets.'
            )

    record = ap.build_profile_record(
        company_key     = ap.company_key(
            {'company_name': company_name, 'companyWebsite': website}
        ),
        company_name    = company_name,
        website         = website,
        profile_summary = summary,
        aspects         = aspects,
        vectors         = vectors,
        sources_used    = list(texts.keys()),
        fingerprint     = ap.source_fingerprint(texts),
        model           = built_by,
        markets         = markets,
        market_vectors  = market_vectors,
        unexplored         = unexplored,
        unexplored_vectors = unexplored_vectors,
        dod_assessment  = parsed['dod_assessment'],
    )
    return record, warnings
