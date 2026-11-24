// Renter page: search -> GET /buildings/{id} -> POST /predict -> show the forecast. Helpers come from common.js.
(() => {
  "use strict";
  const { cfg, API, $, esc, title, pct, fmt, call, setStatus } = window.RR;

  // ---------- search ----------
  const input = $("#q"), list = $("#results");
  let items = [], active = -1, debounce;
  async function search(q) {
    if (q.trim().length < 2) return hideList();
    try {
      const { body } = await call(`/buildings/search?q=${encodeURIComponent(q)}&limit=8`);
      if ($("#status").classList.contains("err")) setStatus("");  // a new successful search clears an old error
      items = body.results || [];
      renderList(q);
      return items;
    } catch (e) {
      setStatus(e.message, true);
      return [];
    }
  }
  function renderList(q, note = "") {
    active = -1;
    if (!items.length) {
      // Nothing in our own records: offer the OpenStreetMap lookup, but only when the user asks for it.
      list.innerHTML = `<li class="empty"${note ? ' aria-disabled="true"' : ""}><span class="ad">${note ? esc(note) : `No NYC building with 3+ units matches “${esc(q)}”. Try the street address, or:`}</span>
        ${note ? "" : `<button type="button" class="btn ghost small lookup" data-q="${esc(q)}">Look up this name online</button>`}</li>`;
    } else {
      list.innerHTML = items.map((b, i) => `<li role="option" id="opt${i}" data-i="${i}" aria-selected="false">
        <div><div class="nm">${esc(b.name ? b.name.split(" / ")[0] : title(b.address))}</div>
        <div class="ad">${esc(title(b.address))}, ${esc(title(b.borough))} ${esc(b.zip || "")}</div>
        ${b.source ? `<a class="src" href="${esc(b.source_url)}" target="_blank" rel="noopener">via OpenStreetMap · ${b.match === "nearby" ? "nearby, please confirm the address" : esc(b.match || "") + " in HPD records"}</a>` : ""}</div>
        <span class="u">${b.units ? fmt(b.units) + " units" : ""}</span></li>`).join("");
    }
    list.hidden = false;
    input.setAttribute("aria-expanded", "true");
  }
  function hideList() {
    list.hidden = true;
    input.setAttribute("aria-expanded", "false");
    input.removeAttribute("aria-activedescendant");
  }
  function move(d) {
    if (!items.length) return;
    active = (active + d + items.length) % items.length;
    list.querySelectorAll("li").forEach((li, i) => li.setAttribute("aria-selected", String(i === active)));
    input.setAttribute("aria-activedescendant", "opt" + active);
    list.querySelector(`#opt${active}`)?.scrollIntoView({ block: "nearest" });
  }
  input.addEventListener("input", () => { clearTimeout(debounce); debounce = setTimeout(() => search(input.value), 220); });
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { e.preventDefault(); move(1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); move(-1); }
    else if (e.key === "Escape") hideList();
    else if (e.key === "Enter" && active >= 0 && !list.hidden) { e.preventDefault(); choose(items[active].bid); }
  });
  // Fallback: ask the API to find the name through OpenStreetMap. One request per click, never per keystroke.
  async function lookupOnline(q, btn) {
    btn.disabled = true;
    btn.textContent = "Looking it up…";
    try {
      const { body } = await call(`/buildings/lookup?q=${encodeURIComponent(q)}`);
      if (input.value !== q) return;  // the user typed something else meanwhile: drop this answer
      items = body.results || [];
      renderList(q, body.reason || "OpenStreetMap did not find this name in New York City.");
    } catch (e) {
      if (input.value === q) renderList(q, e.message);
    }
  }
  list.addEventListener("click", (e) => {
    if (e.target.closest("a.src")) return;  // the source link opens OpenStreetMap; it does not pick the building
    const btn = e.target.closest("button.lookup");
    if (btn) { e.preventDefault(); lookupOnline(btn.dataset.q, btn); return; }
    const li = e.target.closest("li[data-i]");
    if (li) choose(items[+li.dataset.i].bid);
  });
  document.addEventListener("click", (e) => { if (!e.target.closest(".search")) hideList(); });
  $("#searchForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const res = await search(input.value);
    if (res && res.length) choose(res[0].bid);
  });
  document.querySelectorAll(".ex").forEach((b) => b.addEventListener("click", async () => {
    input.value = b.dataset.q;
    const res = await search(b.dataset.q);
    if (res && res.length) choose(res[0].bid);
  }));
  $("#again").addEventListener("click", () => { window.scrollTo({ top: 0 }); input.focus(); input.select(); });

  // ---------- forecast ----------
  async function choose(bid) {
    hideList();
    setStatus("Asking the model…");
    const out = $("#result");
    out.hidden = false;
    $("#forecast").classList.add("skeleton");
    try {
      const { body: b } = await call(`/buildings/${bid}`);
      renderBuilding(b.building);
      const { body: p } = await call("/predict", { method: "POST", body: JSON.stringify(b.features) });
      setStatus("");  // removes "Asking the model…", which this function set
      renderForecast(p, b.building);
      history.replaceState(null, "", `#b${bid}`);
      out.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
    } catch (e) {
      setStatus(e.status === 404 ? "That building is not in the dataset." : e.message, true);
      if (!$("#pct").dataset.done) out.hidden = true;
    } finally {
      $("#forecast").classList.remove("skeleton");
    }
  }

  function renderBuilding(b) {
    const name = b.name ? b.name.split(" / ")[0] : title(b.address);
    const age = b.year_built ? new Date().getFullYear() - b.year_built : null;
    const facts = [
      b.units && [fmt(b.units), "units"],
      b.stories && [b.stories, "stories"],
      b.year_built && [b.year_built, `built (${age} yrs)`],
      b.borough && [title(b.borough), "borough"],
    ].filter(Boolean);
    $("#bhead").innerHTML = `<div><h2>${esc(name)}</h2><p class="addr">${esc(title(b.address))}, ${esc(title(b.borough))}, NY ${esc(b.zip || "")}</p></div>
      <div class="facts">${facts.map(([v, l]) => `<div class="fact"><b>${esc(v)}</b><span>${esc(l)}</span></div>`).join("")}</div>`;
    const report = $("#fullReport");
    report.href = `${(cfg.REPORT_URL || "").replace(/\/$/, "")}/#b${b.bid}`;
    report.hidden = !cfg.REPORT_URL;
  }

  const CHIP = {
    age: (v) => [Math.round(v), "years old"],
    units: (v) => [fmt(Math.round(v)), "units in the building"],
    stories: (v) => [Math.round(v), "stories"],
    has_mgmt: (v) => [v ? "Yes" : "No", "management company registered"],
  };
  function chip(d, i) {
    const up = d.effect === "raises";
    let num, label;
    if (d.feature === "location") { num = ""; label = "where it is in the city"; }
    else if (CHIP[d.feature]) [num, label] = CHIP[d.feature](d.value);
    else { num = d.value == null ? "" : Number.isInteger(d.value) ? fmt(d.value) : d.value.toFixed(d.value < 1 ? 2 : 1); label = d.label; }
    return `<span class="chipb ${up ? "up" : "dn"}" style="animation-delay:${0.25 + i * 0.08}s" title="${up ? "Raises" : "Lowers"} the risk">
      <i aria-hidden="true">${up ? "▲" : "▼"}</i>${num !== "" ? `<b>${esc(num)}</b>` : ""}<span>${esc(label)}<span class="sr">, ${up ? "raises" : "lowers"} the risk</span></span></span>`;
  }

  // One question to ask at the viewing, picked from the strongest driver that raises the risk.
  const ASK = [
    [/mold|leak/, "Any leaks or mold in this line of apartments last year? How were they fixed?"],
    [/pest|bb24/, "When was the last pest or bedbug treatment, and which apartments were treated?"],
    [/heat/, "Did the heat or hot water go out last winter? For how long?"],
    [/noise|street/, "Which apartments get noise complaints, and what does management do about them?"],
    [/v12_c|v3_c|v12_b|v3_b|vbc|v12_ri/, "Which violations did the city find last year, and are they fixed now?"],
    [/c12|c3/, "What do tenants complain about most, and how fast do repairs get done?"],
    [/units|stories|age|v12_a/, "Who handles repairs here, and how do tenants request them?"],
  ];
  function ask(drivers, p) {
    const ups = drivers.filter((d) => d.effect === "raises" && d.feature !== "location");
    if (p.probability < p.base_rate * 0.6 || !ups.length) return "Can I see the most recent repair log? Low risk is not no risk.";
    // most specific topic first (mold, pests, heat...), among the drivers that push the risk up
    for (const [re, q] of ASK) if (ups.some((d) => re.test(d.feature))) return q;
    return ASK[5][1];
  }

  function renderForecast(p, b) {
    const prob = p.probability, top = 1 - p.percentile;
    const st = $("#sticker");
    if (p.percentile >= 0.5) { st.className = "sticker"; st.innerHTML = `Top <span>${Math.max(1, Math.round(top * 100))}%</span> riskiest in NYC`; }
    else { st.className = "sticker calm"; st.innerHTML = `Calmer than <span>${Math.round(top * 100)}%</span> of NYC`; }

    const ratio = prob / p.base_rate;
    $("#sideWord").textContent = p.label ? "High: above the alert line" : ratio >= 1.15 ? `${ratio.toFixed(1)}× the NYC average` : ratio <= 0.85 ? "Lower than most" : "About average";
    $("#sideSub").textContent = `NYC average is ${pct(p.base_rate)} · alert line ${pct(p.threshold)}`;

    const el = $("#pct"), target = Math.round(prob * 100);
    el.classList.remove("go"); void el.offsetWidth; el.classList.add("go");
    el.dataset.done = "1";
    // count up over ~0.9 s; a timer guarantees the final number even if animation frames are paused
    const t0 = performance.now(), show = (v) => (el.innerHTML = `${v}<small>%</small>`);
    const step = (now) => { const k = Math.min(1, (now - t0) / 900); show(Math.round(target * (1 - Math.pow(1 - k, 3)))); if (k < 1) requestAnimationFrame(step); };
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) show(target);
    else { requestAnimationFrame(step); setTimeout(() => show(target), 1000); }

    // gauge
    const arc = $("#arc"), needle = $("#needle");
    arc.style.strokeDashoffset = 100; needle.style.transform = "rotate(-90deg)";
    requestAnimationFrame(() => requestAnimationFrame(() => {
      arc.style.strokeDashoffset = 100 - prob * 100;
      needle.style.transform = `rotate(${-90 + 180 * prob}deg)`;
    }));
    $("#mouth").setAttribute("d", prob >= p.threshold ? "M105 121 q5 -5 10 0" : prob >= p.base_rate ? "M105 119 h10" : "M105 118 q5 5 10 0");
    const a = Math.PI * p.base_rate, x = 110 - 90 * Math.cos(a), y = 115 - 90 * Math.sin(a);
    $("#avgMark").innerHTML = `<circle cx="${x}" cy="${y}" r="9" fill="#fff" stroke="#1E1B17" stroke-width="3"/>
      <text x="${x - 16}" y="${y - 14}" text-anchor="end" class="ml">NYC avg</text><text x="${x - 16}" y="${y + 4}" text-anchor="end" class="mv">${pct(p.base_rate)}</text>`;
    $("#gaugeLabel").textContent = `Gauge: ${target}% against a NYC average of ${pct(p.base_rate)}`;

    $("#why").innerHTML = p.drivers.slice(0, 5).map(chip).join("");
    $("#ask").textContent = ask(p.drivers.slice(0, 5), p);  // only from the reasons shown on screen
    const band = p.backtest_band;
    $("#backtest").innerHTML = band
      ? `Backtest: of <b>${fmt(band.n)}</b> buildings scored ${Math.round(band.low * 100)}–${Math.round(band.high * 100)}%, <b>${pct(band.observed_rate)}</b> really got a B/C violation.`
      : "";
    $("#fcMeta").textContent = `Model ${p.model_version} · a forecast, not an inspection`;
  }

  const m = location.hash.match(/^#b(\d+)$/);
  if (m) choose(m[1]);
})();
