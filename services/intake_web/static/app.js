// DD intake wizard — rendered entirely from GET /api/schema.
// isVisible() must stay in step with src/modules/intake/validation.py.
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const LS_KEY = 'bwco_intake_session';
  const UPLOAD_SECTION = '7_uploads';

  const state = {
    config: null, sections: [], fieldsById: {},
    sessionId: null, answers: {}, confirmed: new Set(), current: 0,
    turnstileId: null, saveTimer: null,
  };

  // ── API ────────────────────────────────────────────────────────────────
  async function api(method, path, body) {
    const res = await fetch(path, {
      method, headers: body ? { 'Content-Type': 'application/json' } : {},
      body: body ? JSON.stringify(body) : undefined,
    });
    let data = {};
    try { data = await res.json(); } catch (_) { /* empty body */ }
    if (!res.ok) { const e = new Error(data.detail || `Request failed (${res.status})`); e.status = res.status; e.data = data; throw e; }
    return data;
  }

  function notice(msg) { const n = $('notice'); n.textContent = msg || ''; n.hidden = !msg; }

  function show(view) {
    for (const v of ['start', 'uploads', 'section', 'done']) $(`view-${v}`).hidden = v !== view;
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

  // ── Start ──────────────────────────────────────────────────────────────
  function renderTurnstile() {
    const key = state.config.turnstile_site_key;
    if (!key) return;                       // local dev: server skips verification
    const tryRender = () => {
      if (window.turnstile) state.turnstileId = window.turnstile.render('#turnstile', { sitekey: key });
      else setTimeout(tryRender, 200);
    };
    tryRender();
  }

  async function onStart(ev) {
    ev.preventDefault();
    notice('');
    const form = ev.target;
    if (!form.reportValidity()) return;
    const fd = new FormData(form);
    const body = Object.fromEntries(fd.entries());
    body.turnstile_token = state.turnstileId !== null && window.turnstile
      ? window.turnstile.getResponse(state.turnstileId) || '' : '';
    form.querySelector('button').disabled = true;
    try {
      const { session_id } = await api('POST', '/api/sessions', body);
      state.sessionId = session_id;
      localStorage.setItem(LS_KEY, session_id);
      state.answers = { company_legal_name: body.company_legal_name, website: body.website, contact_email: body.contact_email };
      show('uploads');
    } catch (e) {
      notice(e.message);
      if (window.turnstile && state.turnstileId !== null) window.turnstile.reset(state.turnstileId);
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
          : `<input type="${type}" name="${f.id}" maxlength="500" value="${esc(val || '')}">`) +
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
      localStorage.removeItem(LS_KEY);
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

    const saved = localStorage.getItem(LS_KEY);
    if (saved) {
      try {
        const s = await api('GET', `/api/sessions/${saved}`);
        if (s.status === 'draft') {
          state.sessionId = saved; state.answers = s.answers || {};
          state.confirmed = new Set(s.confirmed_sections || []);
          const idx = activeSections().findIndex((x) => !state.confirmed.has(x.id));
          state.current = idx < 0 ? 0 : idx;
          renderSection(); show('section');
          return;
        }
      } catch (_) { /* expired */ }
      localStorage.removeItem(LS_KEY);
    }
    renderTurnstile();
    show('start');
  }

  boot();
})();
