// DD intake wizard — rendered entirely from GET /api/schema.
// isVisible() must stay in step with src/modules/intake/validation.py.
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  // Drafts used to be found through localStorage; the server now keys them to
  // the signed-in email. The old key is read once so an in-flight draft is kept.
  const LEGACY_LS_KEY = 'bwco_intake_session';
  const UPLOAD_SECTION = '7_uploads';
  const CONTACT_EMAIL = 'contact_email';

  const state = {
    config: null, sections: [], fieldsById: {},
    sessionId: null, answers: {}, confirmed: new Set(), current: 0,
    turnstileId: null, saveTimer: null, email: null,
  };

  // ── API ────────────────────────────────────────────────────────────────
  async function api(method, path, body) {
    const res = await fetch(path, {
      method, credentials: 'same-origin',
      headers: body ? { 'Content-Type': 'application/json' } : {},
      body: body ? JSON.stringify(body) : undefined,
    });
    let data = {};
    try { data = await res.json(); } catch (_) { /* empty body */ }
    if (!res.ok) {
      const e = new Error(data.detail || `Request failed (${res.status})`); e.status = res.status; e.data = data;
      if (res.status === 401 && path.startsWith('/api/sessions')) signedOut(e.message);
      throw e;
    }
    return data;
  }

  function notice(msg) { const n = $('notice'); n.textContent = msg || ''; n.hidden = !msg; }

  function show(view) {
    for (const v of ['signin', 'sent', 'start', 'uploads', 'section', 'done']) $(`view-${v}`).hidden = v !== view;
    $('progress').hidden = view !== 'section';
    window.scrollTo(0, 0);
  }

  // ── Conditions ─────────────────────────────────────────────────────────
  const asList = (v) => Array.isArray(v) ? v : (v == null || v === '' ? [] : [String(v)]);
  function isVisible(f) {
    if (!f.condition || !f.condition.length) return true;
    return f.condition.some((c) => asList(state.answers[c.field]).some((v) => c.any_of.includes(v)));
  }
  const visibleFields = (s) => s.fields.filter((f) => f.type !== 'file' && isVisible(f));
  const activeSections = () => state.sections.filter((s) => visibleFields(s).length);

  // ── Sign in ────────────────────────────────────────────────────────────
  // The Cloudflare script loads async; until it does, window.turnstile may be
  // undefined (or, with an id="turnstile" element, that element), so check
  // for the API itself rather than the name.
  const turnstileReady = () => !!(window.turnstile && typeof window.turnstile.render === 'function');

  function renderTurnstile() {
    const key = state.config.turnstile_site_key;
    if (!key || state.turnstileId !== null) return;   // local dev: server skips verification
    const tryRender = () => {
      if (turnstileReady()) state.turnstileId = window.turnstile.render('#turnstile-box', { sitekey: key });
      else setTimeout(tryRender, 200);
    };
    tryRender();
  }

  function resetTurnstile() {
    if (turnstileReady() && state.turnstileId !== null) window.turnstile.reset(state.turnstileId);
  }

  function showSignin() {
    setWho(null);
    show('signin');                 // first, so a Turnstile hiccup can never blank the page
    try { renderTurnstile(); } catch (e) { notice('The verification check could not load. Please refresh the page.'); }
  }

  function signedOut(msg) {
    clearTimeout(state.saveTimer);
    state.sessionId = null;
    showSignin();
    notice(msg || 'Please sign in again to continue.');
  }

  function setWho(email) {
    state.email = email;
    $('who').hidden = !email;
    $('who-email').textContent = email || '';
  }

  async function onSignin(ev) {
    ev.preventDefault();
    notice('');
    const form = ev.target;
    if (!form.reportValidity()) return;
    const email = form.elements.email.value.trim();
    const token = state.turnstileId !== null && turnstileReady()
      ? window.turnstile.getResponse(state.turnstileId) || '' : '';
    form.querySelector('button').disabled = true;
    try {
      await api('POST', '/api/auth/request', { email, turnstile_token: token });
      $('sent-email').textContent = email;
      show('sent');
    } catch (e) {
      notice(e.message);
    } finally { form.querySelector('button').disabled = false; resetTurnstile(); }
  }

  async function onSignout() {
    try { await api('POST', '/api/auth/logout'); } catch (_) { /* cookie cleared or not, start over */ }
    state.sessionId = null; state.answers = {}; state.confirmed = new Set();
    notice('');
    showSignin();
  }

  // ── Start ──────────────────────────────────────────────────────────────
  async function onStart(ev) {
    ev.preventDefault();
    notice('');
    const form = ev.target;
    if (!form.reportValidity()) return;
    const body = Object.fromEntries(new FormData(form).entries());
    form.querySelector('button').disabled = true;
    try {
      const { session_id, resumed } = await api('POST', '/api/sessions', body);
      if (resumed && await resume(session_id)) return;   // a draft opened in another tab
      state.sessionId = session_id;
      state.answers = { company_legal_name: body.company_legal_name, website: body.website, [CONTACT_EMAIL]: state.email };
      show('uploads');
    } catch (e) {
      notice(e.message);
    } finally { form.querySelector('button').disabled = false; }
  }

  // ── Uploads ────────────────────────────────────────────────────────────
  async function onFiles(ev) {
    const files = Array.from(ev.target.files || []);
    ev.target.value = '';
    if (!files.length) return;
    const list = $('file-list');
    const okExt = state.config.extensions;
    const maxBytes = state.config.max_mb * 1024 * 1024;
    const bad = files.find((f) => !okExt.some((x) => f.name.toLowerCase().endsWith(x)) || f.size > maxBytes);
    if (bad) { notice(`${bad.name}: only PDF, PPTX or DOCX files under ${state.config.max_mb} MB.`); return; }
    notice('');
    let issued;
    try {
      issued = (await api('POST', `/api/sessions/${state.sessionId}/upload-urls`,
        { files: files.map((f) => ({ filename: f.name, size: f.size })) })).uploads;
    } catch (e) { notice(e.message); return; }
    $('done-uploads').disabled = true;
    await Promise.all(issued.map(async (u, i) => {
      const li = document.createElement('li');
      li.textContent = `${u.filename} — uploading…`;
      list.appendChild(li);
      try {
        const r = await fetch(u.url, { method: 'PUT', headers: u.headers, body: files[i] });
        if (!r.ok) throw new Error(String(r.status));
        li.textContent = `${u.filename} — uploaded`;
      } catch (_) { li.textContent = `${u.filename} — upload failed`; li.className = 'bad'; }
    }));
    $('done-uploads').disabled = false;
  }

  // ── Sections ───────────────────────────────────────────────────────────
  function renderProgress() {
    const ol = $('progress');
    ol.innerHTML = '';
    activeSections().forEach((s, i) => {
      const li = document.createElement('li');
      li.textContent = s.title;
      li.className = i === state.current ? 'current' : (state.confirmed.has(s.id) ? 'done' : '');
      ol.appendChild(li);
    });
  }

  function fieldNode(f) {
    const val = state.answers[f.id];
    const help = f.help_text ? `<span class="help">${esc(f.help_text)}</span>` : '';
    const req = f.required ? ' <span class="req">*</span>' : '';
    const wrap = document.createElement('div');
    wrap.dataset.field = f.id;
    if (f.type === 'single_select' || f.type === 'multi_select') {
      const multi = f.type === 'multi_select';
      const chosen = asList(val);
      wrap.innerHTML = `<fieldset><legend>${esc(f.label)}${req}</legend>${help}` +
        f.options.map((o, i) => `<label class="choice"><input type="${multi ? 'checkbox' : 'radio'}" name="${f.id}" value="${esc(o)}" id="${f.id}_${i}" ${chosen.includes(o) ? 'checked' : ''}> ${esc(o)}</label>`).join('') +
        `</fieldset>`;
    } else {
      const tag = f.type === 'textarea' ? 'textarea' : 'input';
      const type = { email: 'email', url: 'url', phone: 'tel' }[f.type] || 'text';
      wrap.innerHTML = `<label>${esc(f.label)}${req}${help}` +
        (tag === 'textarea'
          ? `<textarea name="${f.id}" maxlength="8000">${esc(val || '')}</textarea>`
          : `<input type="${type}" name="${f.id}" maxlength="500" value="${esc(val || '')}"${f.id === CONTACT_EMAIL ? ' readonly title="Your sign-in address"' : ''}>`) +
        `</label>`;
    }
    return wrap;
  }

  function readField(f) {
    const form = $('section-form');
    if (f.type === 'multi_select') return Array.from(form.querySelectorAll(`input[name="${f.id}"]:checked`)).map((x) => x.value);
    if (f.type === 'single_select') { const x = form.querySelector(`input[name="${f.id}"]:checked`); return x ? x.value : ''; }
    const x = form.querySelector(`[name="${f.id}"]`);
    return x ? x.value : '';
  }

  function renderSection() {
    const secs = activeSections();
    state.current = Math.min(state.current, secs.length - 1);
    const s = secs[state.current];
    $('section-title').textContent = s.title;
    const form = $('section-form');
    form.innerHTML = '';
    visibleFields(s).forEach((f) => form.appendChild(fieldNode(f)));
    $('section-confirm').checked = state.confirmed.has(s.id);
    $('prev-section').disabled = state.current === 0;
    $('next-section').textContent = state.current === secs.length - 1 ? 'Submit' : 'Next';
    renderProgress();
  }

  function onSectionInput() {
    const s = activeSections()[state.current];
    const shape = () => activeSections().map((x) => x.id).join() + '|' + visibleFields(s).map((f) => f.id).join();
    const before = shape();
    s.fields.forEach((f) => { if (isVisible(f) && f.type !== 'file') state.answers[f.id] = readField(f); });
    state.confirmed.delete(s.id);
    $('section-confirm').checked = false;
    // A changed sector can reveal/hide whole sections (2A) or fields; re-render only then.
    if (shape() !== before) {
      const focused = document.activeElement && document.activeElement.name;
      renderSection();
      if (focused) { const el = $('section-form').querySelector(`[name="${focused}"]`); if (el) el.focus(); }
    }
    scheduleSave();
  }

  function scheduleSave() {
    clearTimeout(state.saveTimer);
    $('saved-indicator').textContent = 'Saving…';
    state.saveTimer = setTimeout(save, 800);
  }

  async function save() {
    clearTimeout(state.saveTimer);
    try {
      await api('PUT', `/api/sessions/${state.sessionId}/answers`,
        { answers: state.answers, confirmed_sections: [...state.confirmed] });
      $('saved-indicator').textContent = 'Saved';
    } catch (e) { $('saved-indicator').textContent = ''; notice(e.message); }
  }

  function clientErrors(s) {
    const errs = {};
    visibleFields(s).forEach((f) => {
      const v = state.answers[f.id];
      if (f.required && (v == null || v === '' || (Array.isArray(v) && !v.length))) errs[f.id] = 'Required';
    });
    return errs;
  }

  function markErrors(errs) {
    document.querySelectorAll('#section-form [data-field]').forEach((w) => {
      w.classList.toggle('invalid', !!errs[w.dataset.field]);
      const old = w.querySelector('.error'); if (old) old.remove();
      if (errs[w.dataset.field]) {
        const sp = document.createElement('span'); sp.className = 'error'; sp.textContent = errs[w.dataset.field];
        w.firstElementChild.appendChild(sp);
      }
    });
  }

  async function onNext() {
    notice('');
    const secs = activeSections();
    const s = secs[state.current];
    const errs = clientErrors(s);
    markErrors(errs);
    if (Object.keys(errs).length) { notice('Please complete the required questions.'); return; }
    if (!$('section-confirm').checked) { notice('Please confirm these answers look right before continuing.'); return; }
    state.confirmed.add(s.id);
    if (state.current < secs.length - 1) { state.current += 1; await save(); renderSection(); window.scrollTo(0, 0); return; }
    await submit();
  }

  async function submit() {
    const btn = $('next-section');
    btn.disabled = true;
    try {
      await api('POST', `/api/sessions/${state.sessionId}/submit`,
        { answers: state.answers, confirmed_sections: [...state.confirmed] });
      show('done');
    } catch (e) {
      if (e.status === 422 && e.data.errors) {
        const ids = Object.keys(e.data.errors);
        const idx = activeSections().findIndex((sec) => sec.fields.some((f) => ids.includes(f.id)));
        if (idx >= 0) { state.current = idx; renderSection(); markErrors(e.data.errors); }
      }
      notice(e.message);
    } finally { btn.disabled = false; }
  }

  function esc(s) { return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }

  // ── Boot ───────────────────────────────────────────────────────────────
  async function boot() {
    try {
      const [config, schema] = await Promise.all([api('GET', '/api/config'), api('GET', '/api/schema')]);
      state.config = config;
      state.sections = schema.sections.filter((s) => s.id !== UPLOAD_SECTION);
      $('max-files').textContent = config.max_files;
      $('max-mb').textContent = config.max_mb;
    } catch (e) { notice('The form could not be loaded. Please refresh the page.'); return; }

    $('link-ttl').textContent = state.config.link_ttl_min;
    $('signin-form').addEventListener('submit', onSignin);
    $('resend').addEventListener('click', () => { notice(''); showSignin(); });
    $('sign-out').addEventListener('click', onSignout);
    $('start-form').addEventListener('submit', onStart);
    $('file-input').addEventListener('change', onFiles);
    const toSections = () => { state.current = 0; renderSection(); show('section'); };
    $('skip-uploads').addEventListener('click', toSections);
    $('done-uploads').addEventListener('click', toSections);
    $('section-form').addEventListener('input', onSectionInput);
    $('section-form').addEventListener('change', onSectionInput);
    $('section-confirm').addEventListener('change', (ev) => {
      const s = activeSections()[state.current];
      if (ev.target.checked) state.confirmed.add(s.id); else state.confirmed.delete(s.id);
      scheduleSave();
    });
    $('prev-section').addEventListener('click', () => { state.current = Math.max(0, state.current - 1); renderSection(); });
    $('next-section').addEventListener('click', onNext);

    // A sign-in link: exchange the token by POST (mail scanners only GET),
    // then drop it from the address bar so it never lands in history.
    const token = new URLSearchParams(window.location.search).get('t');
    if (token) {
      history.replaceState(null, '', window.location.pathname);
      try { await api('POST', '/api/auth/verify', { token }); }
      catch (e) { showSignin(); notice(e.message); return; }
    }

    let me;
    try { me = await api('GET', '/api/auth/me'); }
    catch (_) { showSignin(); return; }
    setWho(me.email);

    if (me.session_id && await resume(me.session_id)) return;
    const legacy = safeLocal(() => localStorage.getItem(LEGACY_LS_KEY));
    safeLocal(() => localStorage.removeItem(LEGACY_LS_KEY));
    if (legacy && await resume(legacy)) return;
    show('start');
  }

  // Open a draft at its first unconfirmed section; false if it is not one.
  async function resume(sessionId) {
    try {
      const s = await api('GET', `/api/sessions/${sessionId}`);
      if (s.status !== 'draft') return false;
      state.sessionId = sessionId; state.answers = s.answers || {};
      state.confirmed = new Set(s.confirmed_sections || []);
      const idx = activeSections().findIndex((x) => !state.confirmed.has(x.id));
      state.current = idx < 0 ? 0 : idx;
      renderSection(); show('section');
      return true;
    } catch (_) { return false; }
  }

  function safeLocal(fn) { try { return fn(); } catch (_) { return null; } }

  boot();
})();
