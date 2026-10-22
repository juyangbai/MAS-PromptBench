/* MAS-PromptBench Docs: code bars and copy, tables, theme toggle, mobile nav,
   "On this page" highlighting and full-text search over search/search_index.json. */
(function () {
  'use strict';
  var doc = document, root = doc.documentElement;
  var siteEl = doc.querySelector('[data-base-url]');
  var base = ((siteEl && siteEl.getAttribute('data-base-url')) || '.').replace(/\/$/, '');
  var reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function el(tag, cls, text) {
    var n = doc.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }
  function svgIcon(name) {
    var s = doc.createElementNS('http://www.w3.org/2000/svg', 'svg');
    s.setAttribute('class', 'i'); s.setAttribute('aria-hidden', 'true');
    var u = doc.createElementNS('http://www.w3.org/2000/svg', 'use');
    u.setAttribute('href', '#i-' + name); s.appendChild(u);
    return s;
  }

  /* ---------- code blocks: title bar + copy ---------- */
  function copyText(text, btn) {
    function done() {
      var label = btn.lastChild;
      btn.classList.add('done'); btn.replaceChild(svgIcon('check'), btn.firstChild); label.textContent = 'Copied';
      setTimeout(function () { btn.classList.remove('done'); btn.replaceChild(svgIcon('copy'), btn.firstChild); label.textContent = 'Copy'; }, 1600);
    }
    function fallback() {
      var ta = el('textarea'); ta.value = text; ta.setAttribute('readonly', '');
      ta.style.position = 'fixed'; ta.style.opacity = '0'; doc.body.appendChild(ta); ta.select();
      try { doc.execCommand('copy'); done(); } catch (e) {}
      doc.body.removeChild(ta);
    }
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(done, fallback);
      else fallback();
    } catch (e) { fallback(); }
  }
  doc.querySelectorAll('.prose .highlight').forEach(function (hl) {
    var code = hl.querySelector('pre code') || hl.querySelector('pre');
    if (!code) return;
    var bar = el('div', 'code-bar');
    var m = ((hl.className || '') + ' ' + (code.className || '')).match(/language-([\w+-]+)/);
    var lang = m ? m[1] : '';
    if (lang && lang !== 'text') bar.appendChild(el('span', 'code-lang', lang === 'bibtex' ? 'bib' : lang));
    var fname = hl.querySelector('.filename');
    if (fname) bar.appendChild(el('span', 'code-title', fname.textContent));
    var btn = el('button', 'copy-btn'); btn.type = 'button';
    btn.setAttribute('aria-label', 'Copy code');
    btn.appendChild(svgIcon('copy')); btn.appendChild(el('span', null, 'Copy'));
    btn.addEventListener('click', function () {
      var text = code.textContent.replace(/\n$/, '');
      copyText(text, btn);
    });
    bar.appendChild(btn);
    hl.insertBefore(bar, hl.firstChild);
  });

  /* ---------- tables scroll inside their own box ---------- */
  doc.querySelectorAll('.prose table:not(.dgrid):not(.dbars)').forEach(function (t) {
    if (t.parentElement.classList.contains('table-wrap')) return;
    var w = el('div', 'table-wrap'); t.parentNode.insertBefore(w, t); w.appendChild(t);
  });

  /* ---------- keep short inline code (flags, names) on one line ---------- */
  doc.querySelectorAll('.prose :not(pre) > code').forEach(function (c) {
    if (c.textContent.length <= 28) c.style.whiteSpace = 'nowrap';
  });

  /* ---------- external links open in a new tab ---------- */
  doc.querySelectorAll('a[href^="http"]').forEach(function (a) {
    if (a.host !== location.host) { a.target = '_blank'; a.rel = 'noopener'; }
  });

  /* ---------- result tooltips (hover only; every value is also printed in the cell) ---------- */
  var tipEl = null, tipFor = null;
  function placeTip(x, y) {
    var w = tipEl.offsetWidth, h = tipEl.offsetHeight, vw = window.innerWidth, vh = window.innerHeight;
    var left = Math.min(x + 14, vw - w - 8), top = y + 18;
    if (top + h > vh - 8) top = y - h - 12;
    tipEl.style.left = Math.max(8, left) + 'px'; tipEl.style.top = Math.max(8, top) + 'px';
  }
  doc.addEventListener('pointermove', function (e) {
    if (e.pointerType && e.pointerType !== 'mouse') return;
    var t = e.target.closest && e.target.closest('[data-tip]');
    if (!t) { if (tipEl) { tipEl.classList.remove('on'); tipFor = null; } return; }
    if (!tipEl) { tipEl = el('div', 'dtip'); tipEl.setAttribute('role', 'tooltip'); doc.body.appendChild(tipEl); }
    if (tipFor !== t) {
      tipFor = t; tipEl.textContent = '';
      var parts = t.getAttribute('data-tip').split(' · ');
      tipEl.appendChild(el('b', null, parts.shift()));
      if (parts.length) tipEl.appendChild(el('span', null, parts.join(' · ')));
      if (t.matches('td')) {
        var bEl = t.querySelector('b'), sEl = t.querySelector('span');
        var line = (bEl ? bEl.textContent + ' points' : '') + (sEl ? ' · ' + sEl.textContent : '');
        if (line) tipEl.appendChild(el('span', null, line));
      }
    }
    placeTip(e.clientX, e.clientY);
    tipEl.classList.add('on');
  }, { passive: true });
  window.addEventListener('scroll', function () { if (tipEl) { tipEl.classList.remove('on'); tipFor = null; } }, { passive: true });

  /* ---------- theme toggle ---------- */
  var mq = window.matchMedia('(prefers-color-scheme: dark)');
  function effectiveTheme() {
    var t = root.getAttribute('data-theme');
    return t === 'dark' || t === 'light' ? t : (mq.matches ? 'dark' : 'light');
  }
  var toggle = doc.querySelector('.theme-toggle');
  if (toggle) {
    function label() { toggle.setAttribute('aria-label', effectiveTheme() === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'); }
    label();
    toggle.addEventListener('click', function () {
      var next = effectiveTheme() === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem('mpb-theme', next); } catch (e) {
        /* Storage is blocked in a sandboxed page: keep the choice in the tab's window.name. */
        if (!window.name || /^mpb-theme=/.test(window.name)) window.name = 'mpb-theme=' + next;
      }
      label();
    });
    if (mq.addEventListener) mq.addEventListener('change', label);
  }

  /* ---------- mobile navigation drawer ---------- */
  var sidebar = doc.getElementById('sidebar');
  var scrim = doc.querySelector('.scrim');
  var navToggle = doc.querySelector('.nav-toggle');
  function setNav(open) {
    if (!sidebar) return;
    sidebar.classList.toggle('is-open', open);
    if (scrim) scrim.hidden = !open;
    if (navToggle) navToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    if (open) { var a = sidebar.querySelector('.is-active') || sidebar.querySelector('a'); if (a) a.focus({ preventScroll: true }); }
  }
  if (navToggle) navToggle.addEventListener('click', function () { setNav(!sidebar.classList.contains('is-open')); });
  var navClose = doc.querySelector('.nav-close');
  if (navClose) navClose.addEventListener('click', function () { setNav(false); if (navToggle) navToggle.focus(); });
  if (scrim) scrim.addEventListener('click', function () { setNav(false); });
  if (sidebar) {
    var active = sidebar.querySelector('.nav-link.is-active');
    if (active && active.scrollIntoView) {
      var r = active.getBoundingClientRect();
      if (r.bottom > window.innerHeight - 40) sidebar.scrollTop = active.offsetTop - 120;
    }
  }

  /* ---------- "On this page" highlighting ---------- */
  var tocLinks = Array.prototype.slice.call(doc.querySelectorAll('.toc a'));
  if (tocLinks.length) {
    var targets = tocLinks.map(function (a) {
      var id = decodeURIComponent(a.getAttribute('href').slice(1));
      return doc.getElementById(id);
    });
    var ticking = false;
    function spy() {
      ticking = false;
      var y = 96, current = 0;
      for (var i = 0; i < targets.length; i++) {
        if (targets[i] && targets[i].getBoundingClientRect().top <= y) current = i;
      }
      if (window.innerHeight + window.scrollY >= doc.body.scrollHeight - 4) current = targets.length - 1;
      tocLinks.forEach(function (a, i) { a.classList.toggle('is-active', i === current); });
    }
    window.addEventListener('scroll', function () { if (!ticking) { ticking = true; requestAnimationFrame(spy); } }, { passive: true });
    spy();
  }

  /* ---------- search ---------- */
  var dialog = doc.querySelector('.search-dialog');
  var input = doc.getElementById('search-input');
  var list = doc.getElementById('search-results');
  var index = null, loading = null, pageTitles = {}, results = [], activeIdx = -1, lastFocus = null;

  function loadIndex() {
    if (index || loading) return loading;
    loading = fetch(base + '/search/search_index.json?v=' + ((siteEl && siteEl.getAttribute('data-version')) || ''))
      .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(function (data) {
        index = (data.docs || []).map(function (d) {
          var page = d.location.split('#')[0];
          if (d.location.indexOf('#') < 0) pageTitles[page] = d.title;
          var text = (d.text || '').replace(/\s+/g, ' ').replace(/ ([,.;:)\]}])/g, '$1').replace(/([(\[{]) /g, '$1').trim();
          return { loc: d.location, page: page, title: d.title || '', titleL: (d.title || '').toLowerCase(), text: text, textL: text.toLowerCase() };
        });
      })
      .catch(function () { index = []; loading = null; list.innerHTML = '<li class="r-empty">Search is unavailable here. Use the navigation instead.</li>'; });
    return loading;
  }
  function escapeHtml(s) { return s.replace(/[&<>"']/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]; }); }
  function mark(s, terms) {
    var out = escapeHtml(s);
    terms.forEach(function (t) {
      var re = new RegExp('(' + t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + ')', 'ig');
      out = out.replace(re, '<mark>$1</mark>');
    });
    return out;
  }
  function snippet(d, terms) {
    var pos = -1;
    for (var i = 0; i < terms.length && pos < 0; i++) pos = d.textL.indexOf(terms[i]);
    if (pos < 0) return d.text.slice(0, 150);
    var start = Math.max(0, pos - 60);
    return (start > 0 ? '… ' : '') + d.text.slice(start, start + 170);
  }
  function search(q) {
    var terms = q.toLowerCase().split(/\s+/).filter(function (t) { return t.length >= 2; });
    if (!terms.length || !index) return [];
    var scored = [];
    index.forEach(function (d) {
      var score = 0;
      for (var i = 0; i < terms.length; i++) {
        var t = terms[i], inTitle = d.titleL.indexOf(t) >= 0, at = d.textL.indexOf(t);
        if (!inTitle && at < 0) return;
        if (inTitle) score += d.titleL === t ? 30 : 12;
        if (at >= 0) { var n = d.textL.split(t).length - 1; score += Math.min(n, 6); }
      }
      if (d.loc.indexOf('#') < 0) score += 2;
      scored.push({ d: d, s: score });
    });
    scored.sort(function (a, b) { return b.s - a.s; });
    return scored.slice(0, 12).map(function (x) { return x.d; });
  }
  function render(q) {
    var terms = q.toLowerCase().split(/\s+/).filter(function (t) { return t.length >= 2; });
    list.innerHTML = '';
    activeIdx = -1;
    if (!terms.length) return;
    results = search(q);
    if (!results.length) { list.innerHTML = '<li class="r-empty">No results for “' + escapeHtml(q) + '”.</li>'; return; }
    results.forEach(function (d, i) {
      var li = el('li'); li.setAttribute('role', 'option'); li.id = 'sr-' + i;
      var a = el('a'); a.href = base + '/' + d.loc;
      var pageTitle = pageTitles[d.page] && d.loc.indexOf('#') >= 0 ? pageTitles[d.page] : '';
      a.innerHTML = (pageTitle ? '<span class="r-path">' + escapeHtml(pageTitle) + '</span>' : '') +
        '<span class="r-title">' + mark(d.title, terms) + '</span>' +
        '<span class="r-text">' + mark(snippet(d, terms), terms) + '</span>';
      a.addEventListener('click', closeSearch);
      li.appendChild(a); list.appendChild(li);
    });
    setActive(0);
  }
  function setActive(i) {
    var items = list.querySelectorAll('li[role="option"]');
    if (!items.length) return;
    activeIdx = (i + items.length) % items.length;
    items.forEach(function (li, j) { li.classList.toggle('is-active', j === activeIdx); });
    input.setAttribute('aria-activedescendant', 'sr-' + activeIdx);
    items[activeIdx].scrollIntoView({ block: 'nearest' });
  }
  function openSearch() {
    if (!dialog) return;
    lastFocus = doc.activeElement;
    dialog.hidden = false;
    loadIndex();
    input.value = ''; list.innerHTML = '';
    setTimeout(function () { input.focus(); }, 10);
  }
  function closeSearch() {
    if (!dialog || dialog.hidden) return;
    dialog.hidden = true;
    if (lastFocus && lastFocus.focus) lastFocus.focus();
  }
  if (dialog) {
    doc.querySelectorAll('.search-trigger').forEach(function (b) { b.addEventListener('click', openSearch); });
    dialog.addEventListener('mousedown', function (e) { if (e.target === dialog) closeSearch(); });
    var debounce;
    input.addEventListener('input', function () {
      clearTimeout(debounce);
      var q = input.value;
      debounce = setTimeout(function () {
        var p = loadIndex();
        if (index) render(q); else if (p) p.then(function () { render(input.value); });
      }, 80);
    });
    input.addEventListener('keydown', function (e) {
      if (e.key === 'ArrowDown') { e.preventDefault(); setActive(activeIdx + 1); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); setActive(activeIdx - 1); }
      else if (e.key === 'Enter') {
        var a = list.querySelector('li.is-active a');
        if (a) { e.preventDefault(); closeSearch(); location.href = a.href; }
      }
    });
  }
  doc.addEventListener('keydown', function (e) {
    var typing = /^(INPUT|TEXTAREA|SELECT)$/.test((e.target.tagName || '')) || e.target.isContentEditable;
    if (e.key === 'Escape') { closeSearch(); setNav(false); return; }
    if ((e.key === '/' && !typing) || ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k')) {
      e.preventDefault(); openSearch();
    }
  });

  /* ---------- topology diagrams and fact strips: build in on first view, then
     send a dot along every solid edge, left to right, to show messages moving ---------- */
  (function () {
    if (reduceMotion || !('IntersectionObserver' in window)) return;
    var NS = 'http://www.w3.org/2000/svg';
    var SPREAD = 1600, DUR = 900, REST = 1000, CYCLE = SPREAD + DUR + REST;

    function ease(t) { return t < .5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2; }

    function setup(svg) {
      var box = svg.viewBox.baseVal, W = (box && box.width) || 620;
      svg.querySelectorAll('rect, text, path').forEach(function (p) {
        var x = 0;
        try { x = p.getBBox().x; } catch (e) {}
        p.style.setProperty('--d', Math.round(Math.max(0, x) / W * 700) + 'ms');
        if (p.matches('.edge, .edge-io')) p.style.setProperty('--len', p.getTotalLength().toFixed(1));
      });
      var flows = Array.prototype.map.call(svg.querySelectorAll('.edge, .edge-io'), function (edge) {
        var dot = doc.createElementNS(NS, 'circle');
        dot.setAttribute('r', '3.4');
        dot.setAttribute('class', 'flow-dot' + (edge.classList.contains('edge-io') ? ' flow-dot-io' : ''));
        svg.appendChild(dot);
        return { edge: edge, dot: dot, len: edge.getTotalLength(), t0: Math.max(0, edge.getPointAtLength(0).x) / W * SPREAD };
      });
      svg.classList.add('is-anim');
      return { svg: svg, flows: flows, raf: 0, start: 0, visible: false };
    }

    function frame(st, now) {
      if (!st.start) st.start = now;
      var t = (now - st.start) % CYCLE;
      st.flows.forEach(function (f) {
        var local = t - f.t0;
        if (local < 0 || local > DUR) { f.dot.style.opacity = 0; return; }
        var p = local / DUR, pt = f.edge.getPointAtLength(ease(p) * f.len);
        f.dot.setAttribute('cx', pt.x.toFixed(1)); f.dot.setAttribute('cy', pt.y.toFixed(1));
        f.dot.style.opacity = Math.min(1, p * 6, (1 - p) * 6).toFixed(2);
      });
      st.raf = requestAnimationFrame(function (n) { frame(st, n); });
    }
    function play(st) { if (!st.raf && st.visible && !doc.hidden) st.raf = requestAnimationFrame(function (n) { frame(st, n); }); }
    function pause(st) { if (st.raf) { cancelAnimationFrame(st.raf); st.raf = 0; } }

    var states = Array.prototype.map.call(doc.querySelectorAll('.topo-diagram'), setup);
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        var st = states.filter(function (s) { return s.svg === en.target; })[0];
        if (en.isIntersecting) en.target.classList.add('is-in');
        if (!st) return;
        st.visible = en.isIntersecting;
        if (st.visible) setTimeout(function () { play(st); }, st.start ? 0 : 1100); else pause(st);
      });
    }, { threshold: 0.35 });
    states.forEach(function (st) { io.observe(st.svg); });
    doc.addEventListener('visibilitychange', function () { states.forEach(function (st) { doc.hidden ? pause(st) : play(st); }); });

    doc.querySelectorAll('.facts').forEach(function (strip) {
      Array.prototype.forEach.call(strip.children, function (c, i) { c.style.setProperty('--d', (i * 80) + 'ms'); });
      strip.classList.add('is-anim');
      io.observe(strip);
    });
  })();
})();
