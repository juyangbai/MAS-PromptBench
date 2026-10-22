/* MAS-PromptBench page transition: "Short glide, ordered"
   ---------------------------------------------------------
   Load in <head> as a plain script (no async, no defer) so it runs before the
   first frame. It marks each navigation between the project page and the docs
   as forward (into the docs) or back (to the project page), and skips the
   transition for every other navigation, such as one docs page to another.
   When the title wraps onto two lines (narrow screens), it fades with the page
   instead of gliding, because two-line text can't glide cleanly. */
(function () {
  'use strict';
  var FROM = 'mpb-vt-from', PLAIN = 'mpb-vt-plain';
  /* Any path containing /docs/ is the docs; everything else is the project side. */
  var DOCS = /\/docs(\/|$)/;
  function side(path) { return DOCS.test(path) ? 'docs' : 'project'; }
  function read(k) { try { return sessionStorage.getItem(k); } catch (e) { return null; } }
  function write(k, v) { try { if (v) sessionStorage.setItem(k, v); else sessionStorage.removeItem(k); } catch (e) {} }
  function title() { return document.getElementById('mpb-title'); }
  /* Fallback when session storage is blocked: the page we came from, if it was on this site. */
  function referrerSide() {
    try { var r = new URL(document.referrer); return r.origin === location.origin ? side(r.pathname) : null; } catch (e) { return null; }
  }
  /* true when the title sits on one line */
  function oneLine(el) {
    var lh = parseFloat(getComputedStyle(el).lineHeight);
    return !lh || el.getBoundingClientRect().height < lh * 1.5;
  }

  /* Leaving a page: remember which side it was on, and whether its title can glide. */
  window.addEventListener('pageswap', function (e) {
    write(FROM, side(location.pathname));
    var t = title(), plain = !!t && !oneLine(t);
    if (plain && e.viewTransition) t.style.viewTransitionName = 'none';
    write(PLAIN, plain ? '1' : null);
  });

  /* Arriving: glide only when crossing between the two sides. */
  window.addEventListener('pagereveal', function (e) {
    var vt = e.viewTransition, from = read(FROM) || referrerSide(), plain = read(PLAIN) === '1', t = title();
    write(FROM, null); write(PLAIN, null);
    if (t) t.style.viewTransitionName = '';
    if (!vt) return;
    var to = side(location.pathname);
    if (!from || from === to || !vt.types) { vt.skipTransition(); return; }
    vt.types.add(to === 'docs' ? 'mpb-forward' : 'mpb-back');
    if (t && (plain || !oneLine(t))) { t.style.viewTransitionName = 'none'; vt.types.add('mpb-plain'); }
    document.documentElement.classList.add('mpb-vt');
  });
})();
