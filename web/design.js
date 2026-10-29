/* Design page behaviour: control wiring, submit/poll against the Modal
   worker, and rendering.

   CONFIGURE ME -----------------------------------------------------------
   After `modal deploy worker/modal_app.py`, Modal prints one URL per
   endpoint. Paste them here. While they are empty the page runs in preview
   mode: the controls all work and the results are clearly-labelled
   placeholders, so the layout can be designed without a backend.
   ----------------------------------------------------------------------- */
const SUBMIT_URL = "";
const RESULT_URL = "";

const POLL_MS = 3000;
const POLL_TIMEOUT_MS = 30 * 60 * 1000;
const MAX_LEN = 380;
const MIN_LEN = 20;
const AA = "ACDEFGHIKLMNPQRSTVWY";

/* Measured cost of one interaction map, in ms, for a 292-residue sequence on
   a few CPU cores. Cost grows with the square of the window count, so these
   are rescaled below. Rough: used only for the time estimate. */
const MAP_MS = { mpipi: { 17: 227, 31: 509 }, calvados: { 17: 167, 31: 418 } };
const REF_WINDOWS = { 17: 276, 31: 262 };
const CANDIDATES_PER_ITER = 5;

const EXAMPLE = {
  name: "RUNX2 C-terminal IDR",
  seq:
    "HRQKLDDSKPSLFSDRLSDLGRIPHPSMRVGVPPQNPRPSLNSAPSPFNPQGQSQITDPRQAQSSPPWSY" +
    "DQSYPSYLSQMTSPSIHSTTPLSSTRGTGLPAITDVPRRISDDDTATSDFCLWPSTLSKKSQAGASELGP" +
    "FSDPRQFPSISSLTESRFSNPRMHYPATFTYTPPVTSGMSLGMSATTHYHTYLPPPYPGSSQSQSGPFQT" +
    "SSTPYLYYGTSSGSYQFPMVPGGDRSPSRMLPPCTTTSNGSTLLNPNLPNQNDGVDADGSHSSSPTVLNS" +
    "SGRMDESVWRPY",
};

const $ = (id) => document.getElementById(id);

const form = $("design-form");
const seqBox = $("sequence");
const statusEl = $("status");
const runBtn = $("run");
const results = $("results");

const PREVIEW = !SUBMIT_URL || !RESULT_URL;

/* --- Sequence hygiene --------------------------------------------------- */

/** Mirror of the worker's clean_sequence, for the counter and estimate only.
 *  The worker revalidates everything it is sent. */
function cleanSequence(raw) {
  return raw
    .split("\n")
    .filter((ln) => !ln.trim().startsWith(">"))
    .join("")
    .replace(/[\s\d]/g, "")
    .replace(/[-*]/g, "")
    .toUpperCase();
}

function updateCounter() {
  const n = cleanSequence(seqBox.value).length;
  const el = $("len-readout");
  el.textContent = `${n} / ${MAX_LEN}`;
  el.classList.toggle("over", n > MAX_LEN);
}

/* --- Reading the form --------------------------------------------------- */

const selectedForcefields = () =>
  [...form.querySelectorAll('input[name="forcefield"]:checked')].map((e) => e.value);

const windowSize = () =>
  Number(form.querySelector('input[name="window_size"]:checked').value);

function collect() {
  return {
    sequence: cleanSequence(seqBox.value),
    composition_mode: form.querySelector('input[name="composition_mode"]:checked').value,
    forcefields: selectedForcefields(),
    window_size: windowSize(),
    max_iterations: Number($("max_iterations").value),
    render_maps: $("render_maps").checked,
    seed: Number($("seed").value || 0),
  };
}

/* --- Live time estimate ------------------------------------------------- */

function updateEstimate() {
  const n = cleanSequence(seqBox.value).length;
  const w = windowSize();
  const ffs = selectedForcefields();
  const el = $("time-estimate");

  if (!n || !ffs.length || n < w * 2) {
    el.textContent = "";
    return;
  }

  // Map cost scales with the square of the number of windows.
  const windows = n - w + 1;
  const scale = (windows / REF_WINDOWS[w]) ** 2;
  const perCandidate = ffs.reduce((s, ff) => s + MAP_MS[ff][w] * scale, 0);
  const seconds =
    (perCandidate * CANDIDATES_PER_ITER * Number($("max_iterations").value)) / 1000;

  const txt =
    seconds < 90
      ? `about ${Math.round(seconds / 10) * 10} seconds`
      : `about ${Math.round(seconds / 60)} minutes`;
  el.textContent = `At these settings, a full run takes ${txt} — often less, since it stops early.`;
}

/* --- Control readouts --------------------------------------------------- */

$("max_iterations").addEventListener("input", () => {
  $("out-iters").textContent = $("max_iterations").value;
  updateEstimate();
});
$("out-iters").textContent = $("max_iterations").value;

form.querySelectorAll('input[name="window_size"], input[name="forcefield"]').forEach((el) =>
  el.addEventListener("change", updateEstimate)
);

seqBox.addEventListener("input", () => {
  updateCounter();
  updateEstimate();
});
updateCounter();
updateEstimate();

$("load-example").addEventListener("click", () => {
  seqBox.value = `>${EXAMPLE.name}\n${EXAMPLE.seq}`;
  updateCounter();
  updateEstimate();
  seqBox.focus();
});

/* --- Rendering ---------------------------------------------------------- */

/** Break a sequence into blocks of ten so it can be counted by eye. */
const blocked = (seq) => (seq.match(/.{1,10}/g) || []).join(" ");

function setStatus(msg, isError = false) {
  statusEl.textContent = msg;
  statusEl.classList.toggle("error", isError);
}

const FF_LABEL = { mpipi: "Mpipi", calvados: "CALVADOS" };

let lastResult = null;

function render(data) {
  lastResult = data;

  const list = $("design-list");
  list.textContent = "";

  const d = data.design;
  const status =
    d.convergence_status === "converged"
      ? "converged"
      : `stopped: ${d.convergence_status.replace(/_/g, " ")}`;

  for (const [name, seq, stats, cls] of [
    ["Input", data.target.sequence, `${data.target.length} residues`, "seq target"],
    [
      "Design",
      d.sequence,
      `${d.iterations} iterations · ${d.wall_seconds}s · ${status}`,
      "seq",
    ],
  ]) {
    const row = document.createElement("div");
    row.className = "design-row";
    row.innerHTML =
      `<div class="row-head"><span class="row-name"></span>` +
      `<span class="row-stats"></span></div><p class="${cls}"></p>`;
    row.querySelector(".row-name").textContent = name;
    row.querySelector(".row-stats").textContent = stats;
    row.querySelector("p").textContent = blocked(seq);
    list.appendChild(row);
  }

  renderScores(data.scores);
  renderMaps(data.maps);
  results.hidden = false;
}

function renderScores(scores) {
  const wrap = $("scores-wrap");
  if (!scores || !Object.keys(scores).length) {
    wrap.hidden = true;
    return;
  }
  const head =
    "<thead><tr><th>Forcefield</th><th>RMSE at start</th>" +
    "<th>RMSE final</th><th>Improvement</th><th>Pearson r</th></tr></thead>";
  const body = Object.entries(scores)
    .map(([ff, s]) => {
      const pct = s.rmse_start > 0 ? (1 - s.rmse / s.rmse_start) * 100 : 0;
      return (
        `<tr><td>${FF_LABEL[ff] || ff}</td>` +
        `<td>${s.rmse_start.toFixed(3)}</td>` +
        `<td>${s.rmse.toFixed(3)}</td>` +
        `<td>${pct.toFixed(0)}%</td>` +
        `<td>${Number.isFinite(s.pearson) ? s.pearson.toFixed(3) : "—"}</td></tr>`
      );
    })
    .join("");
  $("scores-table").innerHTML = head + `<tbody>${body}</tbody>`;
  wrap.hidden = false;
}

function renderMaps(maps) {
  const wrap = $("maps-wrap");
  const host = $("map-figures");
  host.textContent = "";
  if (!maps || !Object.keys(maps).length) {
    wrap.hidden = true;
    return;
  }
  for (const [ff, b64] of Object.entries(maps)) {
    const fig = document.createElement("figure");
    const img = document.createElement("img");
    img.src = `data:image/png;base64,${b64}`;
    img.alt =
      `${FF_LABEL[ff] || ff} interaction maps: input, design, and their difference`;
    img.loading = "lazy";
    fig.appendChild(img);
    host.appendChild(fig);
  }
  wrap.hidden = false;
}

/* --- Download ----------------------------------------------------------- */

$("download").addEventListener("click", () => {
  if (!lastResult) return;
  const d = lastResult.design;
  const scores = Object.entries(lastResult.scores)
    .map(([ff, s]) => `${ff}_rmse=${s.rmse};${ff}_r=${s.pearson}`)
    .join("|");
  const lines = [
    `>input|length=${lastResult.target.length}`,
    lastResult.target.sequence,
    `>design|composition=${d.composition_mode}|window=${d.window_size}` +
      `|iterations=${d.iterations}|${scores}`,
    d.sequence,
  ];
  const blob = new Blob([lines.join("\n") + "\n"], { type: "text/plain" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "idrdesign.fasta";
  a.click();
  URL.revokeObjectURL(a.href);
});

/* --- Submit and poll ---------------------------------------------------- */

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function poll(jobId, startedAt) {
  while (Date.now() - startedAt < POLL_TIMEOUT_MS) {
    const elapsed = Math.round((Date.now() - startedAt) / 1000);
    const shown =
      elapsed < 90 ? `${elapsed}s` : `${Math.floor(elapsed / 60)}m ${elapsed % 60}s`;
    setStatus(`Running — ${shown} elapsed. This page can stay open.`);
    await sleep(POLL_MS);

    const res = await fetch(`${RESULT_URL}?job_id=${encodeURIComponent(jobId)}`);
    const data = await res
      .json()
      .catch(() => ({ status: "error", error: `worker returned ${res.status}` }));

    if (data.status === "done") return data;
    if (data.status === "error") throw new Error(data.error || "the run failed");
  }
  throw new Error("timed out waiting for the worker");
}

form.addEventListener("submit", async (ev) => {
  ev.preventDefault();

  const payload = collect();
  const n = payload.sequence.length;

  if (!payload.forcefields.length) {
    setStatus("Select at least one forcefield to match.", true);
    return;
  }
  if (n < MIN_LEN || n > MAX_LEN) {
    setStatus(
      n < MIN_LEN
        ? `Sequence is ${n} residues; needs at least ${MIN_LEN}.`
        : `Sequence is ${n} residues; the limit is ${MAX_LEN}.`,
      true
    );
    return;
  }
  if (n < payload.window_size * 2) {
    setStatus(
      `Sequence is ${n} residues, too short for window ${payload.window_size}; ` +
        `needs at least ${payload.window_size * 2}.`,
      true
    );
    return;
  }
  const bad = [...new Set(payload.sequence.split(""))]
    .filter((c) => !AA.includes(c))
    .sort();
  if (bad.length) {
    setStatus(`Unrecognized residue(s): ${bad.join(", ")}.`, true);
    return;
  }

  runBtn.disabled = true;
  results.hidden = true;

  try {
    if (PREVIEW) {
      setStatus("Preview mode — no worker configured, showing placeholder output.");
      await sleep(600);
      render(previewResult(payload));
      setStatus("Preview mode — this is a placeholder, not a design.", true);
      return;
    }

    const started = Date.now();
    setStatus("Submitting…");
    const res = await fetch(SUBMIT_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const sub = await res
      .json()
      .catch(() => ({ status: "error", error: `worker returned ${res.status}` }));
    if (sub.status === "error" || !sub.job_id) {
      throw new Error(sub.error || "could not start the run");
    }

    const data = await poll(sub.job_id, started);
    render(data);
    setStatus(`Done in ${Math.round((Date.now() - started) / 1000)}s.`);
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    runBtn.disabled = false;
  }
});

/* --- Preview mode ------------------------------------------------------- */

/** Placeholder output for designing the page without a backend.
 *  The sequence is a GS repeat so it can never be mistaken for a real design,
 *  and no maps are produced. */
function previewResult(payload) {
  const L = payload.sequence.length;
  const gs = "GS".repeat(Math.ceil(L / 2)).slice(0, L);

  const scores = {};
  for (const ff of payload.forcefields) {
    scores[ff] = { rmse: 0, rmse_start: 0, pearson: 0 };
  }

  return {
    status: "done",
    target: { sequence: payload.sequence, length: L },
    params: payload,
    design: {
      sequence: gs,
      starting_sequence: gs,
      iterations: 0,
      wall_seconds: 0,
      convergence_status: "preview_placeholder",
      composition_mode: payload.composition_mode,
      window_size: payload.window_size,
    },
    scores,
    maps: null,
  };
}
