/**
 * Light/dark theme toggle.
 *
 * Starts from the OS preference and only pins a choice once the user makes one,
 * so someone who has set "dark" system-wide is never shown a white page first.
 * The choice is stored under one key and applied to `data-theme` on <html>, which
 * the stylesheet reads. Loaded in <head>, so the theme is set before first paint;
 * the toggle button is wired once the document has loaded.
 *
 * The image viewer keeps its own dark surround in both themes (micrographs are
 * read on dark), so only the page chrome changes here.
 */
(() => {
  'use strict';

  const KEY = 'phasecv-theme';
  const root = document.documentElement;
  const media = window.matchMedia('(prefers-color-scheme: dark)');

  /** 'light' | 'dark' | null (null = follow the system) */
  function stored() {
    try {
      const value = localStorage.getItem(KEY);
      return value === 'light' || value === 'dark' ? value : null;
    } catch {
      return null; // private browsing
    }
  }

  function active() {
    return stored() || (media.matches ? 'dark' : 'light');
  }

  function apply(theme) {
    root.dataset.theme = theme;
    document.dispatchEvent(new CustomEvent('themechange', { detail: theme }));
    const button = document.getElementById('themeToggle');
    if (!button) return;
    const toDark = theme === 'light';
    // The button says what it will DO, which is what a screen reader should read.
    button.setAttribute('aria-label', toDark ? 'Switch to dark theme' : 'Switch to light theme');
    button.setAttribute('title', button.getAttribute('aria-label'));
    button.querySelector('.theme-icon').textContent = toDark ? '☾' : '☀';
  }

  // Before first paint, so a light-OS visitor never sees a dark flash.
  root.dataset.theme = active();

  function init() {
    apply(active());

    const button = document.getElementById('themeToggle');
    if (button) {
      button.addEventListener('click', () => {
        const next = active() === 'dark' ? 'light' : 'dark';
        try {
          localStorage.setItem(KEY, next);
        } catch { /* ignore */ }
        apply(next);
      });
    }

    // Follow the OS while the user has not chosen for themselves.
    media.addEventListener('change', () => {
      if (!stored()) apply(active());
    });
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
