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

const POLL_MS = 2000;
const POLL_TIMEOUT_MS = 10 * 60 * 1000;
const MAX_LEN = 380;
const MIN_LEN = 10;

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

/** Mirror of the worker's clean_sequence, for the live counter only.
 *  The worker revalidates; this is just so the count tracks what will be sent. */
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

/* --- Control readouts --------------------------------------------------- */

const readouts = [
  ["target_similarity", "out-sim", (v) => Number(v).toFixed(3)],
  ["ionic_strength", "out-ionic", (v) => `${v} mM`],
  ["n_designs", "out-n", (v) => v],
  ["n_conformers", "out-conf", (v) => v],
];

for (const [input, out, fmt] of readouts) {
  const el = $(input);
  const sync = () => ($(out).textContent = fmt(el.value));
  el.addEventListener("input", sync);
  sync();
}

/* Conformers only mean something when the ensembles are being measured. */
function syncMetricsToggle() {
  const on = $("compute_metrics").checked;
  $("conformers-row").classList.toggle("disabled", !on);
  $("conformers-note").classList.toggle("disabled", !on);
  $("n_conformers").disabled = !on;
}
$("compute_metrics").addEventListener("change", syncMetricsToggle);
syncMetricsToggle();

seqBox.addEventListener("input", updateCounter);
updateCounter();

$("load-example").addEventListener("click", () => {
  seqBox.value = `>${EXAMPLE.name}\n${EXAMPLE.seq}`;
  updateCounter();
  seqBox.focus();
});

/* --- Rendering ---------------------------------------------------------- */

/** Break a sequence into blocks of ten so it can be counted by eye. */
function blocked(seq) {
  return (seq.match(/.{1,10}/g) || []).join(" ");
}

function setStatus(msg, isError = false) {
  statusEl.textContent = msg;
  statusEl.classList.toggle("error", isError);
}

let lastResult = null;

function render(data) {
  lastResult = data;

  const list = $("design-list");
  list.textContent = "";

  const target = document.createElement("div");
  target.className = "design-row";
  target.innerHTML =
    `<div class="row-head"><span class="row-name">Input</span>` +
    `<span class="row-stats">${data.target.length} residues</span></div>` +
    `<p class="seq target"></p>`;
  target.querySelector(".seq").textContent = blocked(data.target.sequence);
  list.appendChild(target);

  data.designs.forEach((d, i) => {
    const row = document.createElement("div");
    row.className = "design-row";
    const status = d.convergence_status === "converged"
      ? "converged"
      : `stopped: ${d.convergence_status.replace(/_/g, " ")}`;
    row.innerHTML =
      `<div class="row-head"><span class="row-name">Design ${String(i + 1).padStart(2, "0")}</span>` +
      `<span class="row-stats">cosine ${d.cosine_similarity.toFixed(3)} · ` +
      `${d.iterations} iterations · ${d.wall_seconds}s · ${status}</span></div>` +
      `<p class="seq"></p>`;
    row.querySelector(".seq").textContent = blocked(d.sequence);
    list.appendChild(row);
  });

  const wrap = $("metrics-wrap");
  if (data.metrics) {
    renderMetrics(data.metrics);
    wrap.hidden = false;
  } else {
    wrap.hidden = true;
  }

  results.hidden = false;
}

function renderMetrics(metrics) {
  const rows = Object.keys(metrics).sort((a, b) =>
    a === "target" ? -1 : b === "target" ? 1 : a.localeCompare(b)
  );

  const label = (k) =>
    k === "target" ? "Input" : `Design ${String(Number(k.split("_")[1]) + 1).padStart(2, "0")}`;

  const head =
    "<thead><tr><th>Sequence</th><th>Rg (Å)</th>" +
    "<th>End-to-end (Å)</th><th>Asphericity</th></tr></thead>";

  const body = rows
    .map((k) => {
      const m = metrics[k];
      return (
        `<tr${k === "target" ? ' class="is-target"' : ""}>` +
        `<td>${label(k)}</td>` +
        `<td>${m.rg_mean.toFixed(1)} ± ${m.rg_std.toFixed(1)}</td>` +
        `<td>${m.end_to_end_mean.toFixed(1)} ± ${m.end_to_end_std.toFixed(1)}</td>` +
        `<td>${m.asphericity_mean.toFixed(3)}</td>` +
        `</tr>`
      );
    })
    .join("");

  $("metrics-table").innerHTML = head + `<tbody>${body}</tbody>`;
}

/* --- Download ----------------------------------------------------------- */

$("download").addEventListener("click", () => {
  if (!lastResult) return;
  const lines = [`>input|length=${lastResult.target.length}`, lastResult.target.sequence];
  lastResult.designs.forEach((d, i) => {
    lines.push(
      `>design_${String(i + 1).padStart(2, "0")}|cosine=${d.cosine_similarity}` +
        `|composition=${d.composition_mode}|seed=${d.seed}`
    );
    lines.push(d.sequence);
  });
  const blob = new Blob([lines.join("\n") + "\n"], { type: "text/plain" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "idrdesign.fasta";
  a.click();
  URL.revokeObjectURL(a.href);
});

/* --- Submit and poll ---------------------------------------------------- */

function collect() {
  return {
    sequence: cleanSequence(seqBox.value),
    composition_mode: form.querySelector('input[name="composition_mode"]:checked').value,
    target_similarity: Number($("target_similarity").value),
    ionic_strength: Number($("ionic_strength").value),
    n_designs: Number($("n_designs").value),
    n_conformers: Number($("n_conformers").value),
    compute_metrics: $("compute_metrics").checked,
    seed: Number($("seed").value || 0),
  };
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function poll(jobId, startedAt) {
  while (Date.now() - startedAt < POLL_TIMEOUT_MS) {
    const elapsed = Math.round((Date.now() - startedAt) / 1000);
    setStatus(`Running — ${elapsed}s elapsed. This page can stay open.`);
    await sleep(POLL_MS);

    const res = await fetch(`${RESULT_URL}?job_id=${encodeURIComponent(jobId)}`);
    const data = await res.json().catch(() => ({
      status: "error",
      error: `worker returned ${res.status}`,
    }));

    if (data.status === "done") return data;
    if (data.status === "error") throw new Error(data.error || "the run failed");
  }
  throw new Error("timed out waiting for the worker");
}

form.addEventListener("submit", async (ev) => {
  ev.preventDefault();

  const payload = collect();
  const n = payload.sequence.length;
  if (n < MIN_LEN || n > MAX_LEN) {
    setStatus(
      n < MIN_LEN
        ? `Sequence is ${n} residues; needs at least ${MIN_LEN}.`
        : `Sequence is ${n} residues; the limit is ${MAX_LEN}.`,
      true
    );
    return;
  }
  const bad = [...new Set(payload.sequence.split(""))]
    .filter((c) => !"ACDEFGHIKLMNPQRSTVWY".includes(c))
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
      setStatus("Preview mode — these sequences are placeholders, not designs.", true);
      return;
    }

    const started = Date.now();
    setStatus("Submitting…");
    const res = await fetch(SUBMIT_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const sub = await res.json().catch(() => ({
      status: "error",
      error: `worker returned ${res.status}`,
    }));
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
 *  Sequences are a GS repeat so they can never be mistaken for real designs. */
function previewResult(payload) {
  const L = payload.sequence.length;
  const gs = "GS".repeat(Math.ceil(L / 2)).slice(0, L);

  const designs = Array.from({ length: payload.n_designs }, (_, i) => ({
    sequence: gs,
    cosine_similarity: payload.target_similarity,
    iterations: 0,
    wall_seconds: 0,
    convergence_status: "preview_placeholder",
    seed: payload.seed + i,
    composition_mode: payload.composition_mode,
  }));

  let metrics = null;
  if (payload.compute_metrics) {
    metrics = { target: { rg_mean: 0, rg_std: 0, end_to_end_mean: 0, end_to_end_std: 0, asphericity_mean: 0 } };
    designs.forEach((_, i) => {
      metrics[`design_${String(i).padStart(2, "0")}`] = {
        rg_mean: 0, rg_std: 0, end_to_end_mean: 0, end_to_end_std: 0, asphericity_mean: 0,
      };
    });
  }

  return {
    status: "done",
    target: { sequence: payload.sequence, length: L },
    params: payload,
    designs,
    metrics,
  };
}
