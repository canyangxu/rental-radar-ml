// Project page (for graders): links, the model card from GET /model, and a form that sends
// the user's own input to POST /predict on the deployed API. Helpers come from common.js.
(() => {
  "use strict";
  const { cfg, API, $, esc, pct, fmt, call, setStatus } = window.RR;

  // ---------- links ----------
  $("#lkDocs").href = `${API}/docs`;
  $("#lkApi").textContent = API.replace(/^https?:\/\//, "");
  $("#apiUrl").textContent = `${API}/predict`;

  // ---------- model card (from GET /model) ----------
  async function loadModel() {
    try {
      const { body: m } = await call("/model");
      const t = m.test_metrics, rule = m.baselines.baseline_bc_last_year, no = m.baselines.baseline_always_no;
      const rows = [["Always say no", no.pr_auc], ["Had a B/C violation last year", rule.pr_auc], [`Our model (${m.model_type === "xgboost" ? "XGBoost" : "logistic regression"})`, t.pr_auc, true]];
      $("#bars").innerHTML = rows.map(([l, v, me]) => `<div class="bar${me ? " me" : ""}"><div class="lab"><span>${esc(l)}</span><b>${v.toFixed(3)}</b></div><div class="tr"><div class="fill" data-w="${v * 100}"></div></div></div>`).join("");
      $("#thr").textContent = t.threshold.toFixed(2);
      $("#cm").innerHTML = `<span></span><span class="h">said no</span><span class="h">said yes</span>
        <span class="r">no violation</span><div class="c good"><b>${fmt(t.tn)}</b><span>correct</span></div><div class="c bad"><b>${fmt(t.fp)}</b><span>false alarm</span></div>
        <span class="r">got one</span><div class="c bad"><b>${fmt(t.fn)}</b><span>missed</span></div><div class="c good"><b>${fmt(t.tp)}</b><span>caught</span></div>`;
      $("#cmNote").textContent = `Catches ${pct(t.recall)} of the buildings that got a violation; ${pct(t.precision)} of its alerts were right. The "last year" rule: ${pct(rule.recall)} caught, ${pct(rule.precision)} right.`;
      $("#calib").innerHTML = m.calibration_bands.map((b) => `<div class="col" title="${Math.round(b.low * 100)}–${Math.round(b.high * 100)}% band: ${pct(b.observed_rate)} observed, ${fmt(b.n)} buildings">
        <div class="obs" data-h="${b.observed_rate * 100}"></div><div class="tick" style="bottom:${b.mean_predicted * 100}%"></div><span class="bl">${Math.round(b.low * 100)}</span></div>`).join("");
      $("#calib").classList.add("calib-wrap");
      $("#modelMeta").textContent = `Model ${m.version} · trained ${m.trained_at.slice(0, 10)} · ${m.n_features} features · threshold ${m.threshold_rule} · ROC-AUC ${t.roc_auc.toFixed(3)} · top-10% precision ${pct(t.precision_top_10pct)}`;
      const io = new IntersectionObserver((es) => es.forEach((e) => {
        if (!e.isIntersecting) return;
        e.target.querySelectorAll("[data-w]").forEach((f) => (f.style.width = f.dataset.w + "%"));
        e.target.querySelectorAll("[data-h]").forEach((f) => (f.style.height = f.dataset.h + "%"));
        io.unobserve(e.target);
      }), { threshold: 0.25 });
      io.observe($("#model"));
    } catch (e) {
      $("#modelMeta").textContent = `Model details unavailable: ${e.message}`;
    }
  }

  // ---------- try the API: form -> POST /predict ----------
  const form = $("#pform");
  const PRESETS = {
    quiet: { borough: "BROOKLYN", units: 12, stories: 3, year_built: 1931, has_mgmt: true, c12_all: 1, c12_heat: 0, c12_mold: 0, c12_pests: 0, v12_a: 0, v12_b: 0, v12_c: 0, bb24_infested: 0 },
    busy: { borough: "BRONX", units: 40, stories: 5, year_built: 1925, has_mgmt: false, c12_all: 25, c12_heat: 8, c12_mold: 4, c12_pests: 3, v12_a: 2, v12_b: 6, v12_c: 3, bb24_infested: 1 },
  };
  let base = {};       // the full record of a real building, when one is loaded; the form edits it
  let lastReq = null;

  function fill(values) {
    for (const el of form.elements) {
      if (!el.name) continue;
      const v = values[el.name];
      if (el.type === "checkbox") el.checked = !!v;
      else el.value = v == null ? "" : String(v);
    }
  }
  function readForm() {
    const out = { ...base };
    for (const el of form.elements) {
      if (!el.name) continue;
      if (el.type === "checkbox") out[el.name] = el.checked;
      else if (el.value === "") delete out[el.name];  // empty: let the model use the training median (or 0 for counts)
      else out[el.name] = el.tagName === "SELECT" ? el.value : Number(el.value);
    }
    return out;
  }

  document.querySelectorAll("[data-preset]").forEach((b) => b.addEventListener("click", () => {
    base = {};
    $("#realNote").textContent = "";
    fill(PRESETS[b.dataset.preset]);
    predict();
  }));

  $("#loadReal").addEventListener("click", async () => {
    const q = $("#realQ").value.trim();
    if (q.length < 2) return $("#realQ").focus();
    $("#realNote").textContent = "Looking it up…";
    try {
      const { body: s } = await call(`/buildings/search?q=${encodeURIComponent(q)}&limit=1`);
      if (!s.results.length) { $("#realNote").textContent = `No NYC building with 3+ units matches “${q}”.`; return; }
      const { body: b } = await call(`/buildings/${s.results[0].bid}`);
      base = { ...b.features };
      fill(base);
      const name = b.building.name ? b.building.name.split(" / ")[0] : b.building.address;
      $("#realNote").innerHTML = `Loaded <b>${esc(name)}</b> (${esc(b.building.borough || "")}) from <code>GET /buildings/${b.building.bid}</code>: all ${Object.keys(base).length} fields are sent; the form changes the ones shown.`;
      predict();
    } catch (e) {
      $("#realNote").textContent = e.message;
    }
  });
  $("#realQ").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); $("#loadReal").click(); } });

  form.addEventListener("submit", (e) => { e.preventDefault(); predict(); });

  async function predict() {
    const req = readForm();
    lastReq = req;
    $("#apiReq").textContent = JSON.stringify(req, null, 2);
    setStatus("Asking the model…");
    $("#pout").classList.add("skeleton");
    try {
      const { body: p, ms } = await call("/predict", { method: "POST", body: JSON.stringify(req) });
      setStatus("");
      renderPrediction(p);
      $("#apiRes").textContent = JSON.stringify(p, null, 2);
      $("#apiLat").textContent = `200 OK · ${Math.round(ms)} ms round trip`;
    } catch (e) {
      setStatus(e.status === 422 ? `The API rejected the input (422): ${e.message}` : e.message, true);
      $("#apiRes").textContent = e.message;
      $("#apiLat").textContent = e.status ? `HTTP ${e.status}` : "";
    } finally {
      $("#pout").classList.remove("skeleton");
    }
  }

  function renderPrediction(p) {
    $("#pProb").innerHTML = `${Math.round(p.probability * 100)}<small>%</small>`;
    const risk = $("#pRisk");
    risk.hidden = false;
    risk.className = `pill ${p.risk_level}`;
    risk.textContent = `${p.risk_level} risk`;
    $("#pLine").innerHTML = `Chance of at least one class B or C violation in the next 6 months. ` +
      `${p.label ? "<b>Above</b>" : "Below"} the alert line of ${pct(p.threshold)}; riskier than <b>${pct(p.percentile)}</b> of NYC buildings (average ${pct(p.base_rate)}).`;
    const band = p.backtest_band;
    $("#pBand").innerHTML = band
      ? `Backtest: of <b>${fmt(band.n)}</b> test-set buildings scored ${Math.round(band.low * 100)}–${Math.round(band.high * 100)}%, <b>${pct(band.observed_rate)}</b> really got a B/C violation.`
      : "";
    $("#pWhy").innerHTML = p.drivers.slice(0, 5).map((d, i) => {
      const up = d.effect === "raises";
      const v = d.value == null ? "" : Number.isInteger(d.value) ? fmt(d.value) : d.value.toFixed(d.value < 1 ? 2 : 1);
      return `<span class="chipb ${up ? "up" : "dn"}" style="animation-delay:${i * 0.06}s"><i aria-hidden="true">${up ? "▲" : "▼"}</i>${v !== "" ? `<b>${esc(v)}</b>` : ""}<span>${esc(d.label)}<span class="sr">, ${up ? "raises" : "lowers"} the risk</span></span></span>`;
    }).join("");
  }

  $("#copyCurl").addEventListener("click", async () => {
    const body = JSON.stringify(lastReq || readForm());
    const cmd = `curl -X POST ${API}/predict -H "Content-Type: application/json" -d '${body}'`;
    try { await navigator.clipboard.writeText(cmd); $("#copyCurl").textContent = "Copied"; }
    catch { window.prompt("Copy this command", cmd); }
    setTimeout(() => ($("#copyCurl").textContent = "Copy as curl"), 1600);
  });

  fill(PRESETS.busy);
  loadModel();
})();
