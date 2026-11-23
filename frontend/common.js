// Shared by both pages: config, small helpers and the API call with its "waking up" notice.
(() => {
  "use strict";
  const cfg = window.RR_CONFIG || {};
  const API = (new URLSearchParams(location.search).get("api") || cfg.API_URL || "").replace(/\/$/, "");
  const $ = (s) => document.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const title = (s) => String(s || "").toLowerCase().replace(/\b([a-z])/g, (m) => m.toUpperCase()).replace(/\b(\d+)(St|Nd|Rd|Th)\b/gi, (m) => m.toLowerCase());
  const pct = (x, d = 0) => `${(x * 100).toFixed(d)}%`;
  const fmt = (n) => Number(n).toLocaleString("en-US");

  document.querySelectorAll("[data-repo]").forEach((a) => (a.href = cfg.REPO_URL || "#"));
  // keep ?api=... when moving between the two pages (testing against a local API)
  document.querySelectorAll("a[data-keep-api]").forEach((a) => { if (location.search) a.href += location.search; });

  // ---------- API calls ----------
  // Each request owns its own "waking up" notice and only removes that notice, never someone else's message.
  const WAKING = "Waking up the model server. The first request after a quiet spell can take about 20 seconds.";
  async function call(path, opts = {}) {
    let shown = false;
    const slow = setTimeout(() => { if (!$("#status").textContent) { setStatus(WAKING); shown = true; } }, 2500);
    const ctrl = new AbortController();
    const kill = setTimeout(() => ctrl.abort(), 60000);
    try {
      const t0 = performance.now();
      const r = await fetch(API + path, { ...opts, signal: ctrl.signal, headers: { "Content-Type": "application/json", ...(opts.headers || {}) } });
      const ms = performance.now() - t0;
      const body = await r.json().catch(() => ({}));
      if (!r.ok) throw Object.assign(new Error(body.detail ? JSON.stringify(body.detail) : `HTTP ${r.status}`), { status: r.status });
      return { body, ms };
    } catch (e) {
      if (e.name === "AbortError") throw new Error("The model server did not answer within 60 seconds.");
      if (e instanceof TypeError) throw new Error("Could not reach the model API. Check your connection and try again.");
      throw e;
    } finally {
      clearTimeout(kill);
      clearTimeout(slow);
      if (shown && $("#status").textContent === WAKING) setStatus("");
    }
  }
  function setStatus(msg, err = false) {
    const s = $("#status");
    s.textContent = msg || "";
    s.classList.toggle("err", err);
  }

  window.RR = { cfg, API, $, esc, title, pct, fmt, call, setStatus };
})();
