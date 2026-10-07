// TruthCheck popup: circular gauge UI + TEXT/IMAGE/VIDEO analysis via the Flask backend.
const DEFAULT_API = "http://127.0.0.1:5000";
const $ = (id) => document.getElementById(id);

let apiBase = DEFAULT_API;
const RING_CIRCUMFERENCE = 339.292; // 2 * PI * 54

function esc(value) {
  return String(value).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function setSubtitle(text) {
  const el = $("subtitle");
  if (el) el.textContent = text;
}

function tierOf(verdict, pct) {
  if (verdict === "FAKE" || verdict === "ERROR") return "high";
  if (verdict === "REAL") return "low";
  if (pct >= 75) return "high";
  if (pct >= 45) return "medium";
  return "low";
}

function resetRing() {
  const ring = $("circleProgress");
  ring.style.strokeDashoffset = String(RING_CIRCUMFERENCE);
  ring.className = "circle-progress";
  $("confidenceValue").textContent = "0";
  $("confidenceLabel").textContent = "—";
  $("confidenceLabel").className = "confidence-label";
  const badge = $("badge");
  badge.textContent = "—";
  badge.className = "verdict-text";
  $("signalCount").textContent = "0";
}

async function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  await chrome.storage.local.set({ theme });
}

function toggleTheme() {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  applyTheme(next);
}

async function loadTheme() {
  const { theme } = await chrome.storage.local.get("theme");
  document.documentElement.dataset.theme = theme === "dark" ? "dark" : "light";
}

async function loadSettings() {
  const { apiBase: stored } = await chrome.storage.local.get("apiBase");
  apiBase = (stored || DEFAULT_API).replace(/\/+$/, "");
}

async function checkHealth() {
  const dot = $("health-dot");
  try {
    const resp = await fetch(`${apiBase}/api/health`, { signal: AbortSignal.timeout(5000) });
    const body = await resp.json();
    if (body.status === "ok") {
      dot.classList.add("ok");
      dot.classList.remove("down");
      try {
        const st = await (await fetch(`${apiBase}/api/model-status`)).json();
        let title = `Backend online · text: ${st.isot ? "ready" : "missing"} · image: ${st.cifake ? "ready" : "missing"} · video: ${st.mesonet ? "ready" : "missing"}`;
        try {
          const h = await (await fetch("http://127.0.0.1:8000/health", { signal: AbortSignal.timeout(3000) })).json();
          title += h.models && h.models.text ? " · ISOT text backend: trained model" : " · ISOT text backend: online (heuristic — train text_model.joblib)";
        } catch (_) { title += " · ISOT text backend: offline (using fallback)"; }
        dot.title = title;
      } catch (_) { dot.title = "Backend online"; }
      if ($("subtitle").textContent.startsWith("Backend")) setSubtitle("Verify before you trust.");
      return;
    }
    throw new Error("bad status");
  } catch (_) {
    dot.classList.remove("ok");
    dot.classList.add("down");
    dot.title = "Backend unreachable — start it with: python app.py (port 5000)";
    setSubtitle("Backend unreachable - start app.py on port 5000.");
  }
}

function metaLines(result) {
  const d = (result && result.details) || {};
  const lines = [];
  if (d.model) lines.push(`Model: ${d.model}`);
  if (typeof d.p_fake === "number") lines.push(`P(fake) = ${Math.round(d.p_fake * 100)}% · P(real) = ${Math.round((1 - d.p_fake) * 100)}%`);
  if (typeof d.risk === "string") lines.push(`Risk level: ${d.risk}`);
  if (typeof d.frames_analyzed === "number") lines.push(`Frames analyzed: ${d.frames_analyzed} (suspicious: ${d.suspicious_frames})`);
  if (d.error) lines.push(`Error: ${d.error}`);
  if (d.reason) lines.push(d.reason);
  return lines;
}

function signalItems(result) {
  const d = (result && result.details) || {};
  if (Array.isArray(d.signals) && d.signals.length) return d.signals;
  const items = [];
  if (typeof d.p_fake === "number") {
    items.push(`fake probability ${(d.p_fake * 100).toFixed(1)}%`);
    items.push(`real probability ${((1 - d.p_fake) * 100).toFixed(1)}%`);
  }
  if (d.risk) items.push(`risk level ${d.risk}`);
  if (typeof d.suspicious_frames === "number" && d.suspicious_frames > 0)
    items.push(`${d.suspicious_frames} of ${d.frames_analyzed} frames scored as suspicious`);
  return items;
}

function showHint(show) {
  const hint = $("hint");
  if (hint) hint.classList.toggle("hidden", !show);
}

function renderResult(result) {
  const box = $("result");
  if (!result) {
    box.classList.add("hidden");
    resetRing();
    setSubtitle("Verify before you trust.");
    showHint(true);
    return;
  }
  const verdict = result.verdict || "UNKNOWN";
  const pct = Math.round((result.confidence || 0) * 100);

  const badge = $("badge");
  badge.textContent = verdict;
  badge.className = `verdict-text ${verdict}`;

  $("confidenceValue").textContent = String(pct);
  const ring = $("circleProgress");
  ring.style.strokeDashoffset = String(RING_CIRCUMFERENCE * (1 - pct / 100));
  const tier = tierOf(verdict, pct);
  ring.className = `circle-progress ${tier}`;

  const label = $("confidenceLabel");
  label.textContent = tier === "high" ? "High" : tier === "medium" ? "Medium" : "Low";
  label.className = `confidence-label ${tier}`;

  $("meta").innerHTML = metaLines(result).map((l) => `<div>${esc(l)}</div>`).join("");

  const items = signalItems(result);
  $("signals").innerHTML = items.map((s) => `<li>${esc(s)}</li>`).join("");
  $("signalCount").textContent = String(items.length);

  box.classList.remove("hidden");
  showHint(false);
  setSubtitle(
    verdict === "FAKE"
      ? "Manipulation detected - review the signals below."
      : verdict === "REAL"
      ? "No manipulation found in this content."
      : verdict === "ERROR"
      ? "Analysis failed - see the details below."
      : "Inconclusive - not enough evidence either way."
  );
}

// --- Map backend /api/detect/* response into the popup result shape ---
function toResult(data) {
  const prediction = data.prediction || "";
  const verdict =
    prediction === "REAL" ? "REAL"
    : prediction === "SUSPICIOUS" ? "SUSPICIOUS"
    : prediction === "ERROR" ? "ERROR"
    : "FAKE"; // FAKE / AI_GENERATED / DEEPFAKE
  return {
    verdict,
    confidence: data.confidence || 0,
    details: {
      model: data.model,
      p_fake: data.fake_probability,
      risk: data.risk,
      frames_analyzed: data.frames_analyzed,
      suspicious_frames: data.suspicious_frames,
      type: data.type,
    },
  };
}

async function saveAndRender(data) {
  const result = toResult(data);
  await chrome.storage.local.set({ lastResult: { at: Date.now(), result } });
  renderResult(result);
}

async function restoreLastResult() {
  const { lastResult } = await chrome.storage.local.get("lastResult");
  if (lastResult && Date.now() - lastResult.at < 10 * 60 * 1000) {
    renderResult(lastResult.result);
  } else {
    renderResult(null);
  }
}

chrome.storage.onChanged.addListener((changes, area) => {
  if (area !== "local" || !changes.lastResult) return;
  const lastResult = changes.lastResult.newValue;
  renderResult(
    lastResult && Date.now() - lastResult.at < 10 * 60 * 1000 ? lastResult.result : null
  );
});

function openSettings() {
  if (chrome.runtime.openOptionsPage) chrome.runtime.openOptionsPage();
}

// --- Tabs ---
document.querySelectorAll(".tab").forEach((btn) =>
  btn.addEventListener("click", () => switchTab(btn.dataset.tab))
);
function switchTab(name) {
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  ["text", "image", "video", "audio"].forEach((t) =>
    $("tab-" + t).classList.toggle("hidden", t !== name)
  );
}

// --- Loading ---
function showLoading(steps) {
  const el = $("loading");
  el.classList.remove("hidden");
  let i = 0;
  $("loadingText").textContent = steps[0];
  const t = setInterval(() => {
    i = (i + 1) % steps.length;
    $("loadingText").textContent = steps[i];
  }, 1500);
  el.dataset.timer = t;
}
function hideLoading() {
  clearInterval(parseInt($("loading").dataset.timer || "0", 10));
  $("loading").classList.add("hidden");
}

async function post(path, body) {
  const res = await fetch(apiBase + path, { method: "POST", body });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

// --- Analyze handlers ---
let currentImage = null, currentVideo = null, currentAudio = null;

document.getElementById("analyzeTextBtn").addEventListener("click", async () => {
  const text = document.getElementById("textInput").value.trim();
  if (!text) return alert("Please paste some text.");
  showLoading(["Analyzing...", "Running text model...", "Calculating confidence..."]);
  try {
    let result;
    try {
      // Primary: TruthCheck text backend (ISOT TF-IDF model + verification)
      const r = await fetch("http://127.0.0.1:8000/analyze/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, verify: true }),
        signal: AbortSignal.timeout(20000),
      });
      const body = await r.json();
      if (!r.ok) throw new Error(body.detail || `HTTP ${r.status}`);
      result = body; // already {verdict, confidence, details}
    } catch (_) {
      // Fallback: local Flask text detector
      const fd = new FormData(); fd.append("text", text);
      result = toResult(await post("/api/detect/text", fd));
    }
    await chrome.storage.local.set({ lastResult: { at: Date.now(), result } });
    renderResult(result);
  } catch (e) { alert("Text analysis failed.\n\n" + e.message); }
  finally { hideLoading(); }
});

document.getElementById("imageInput").addEventListener("change", (e) => {
  currentImage = e.target.files[0];
  if (currentImage) {
    const img = $("imagePreview");
    img.src = URL.createObjectURL(currentImage);
    img.classList.remove("hidden");
  }
});
document.getElementById("analyzeImageBtn").addEventListener("click", async () => {
  if (!currentImage) return alert("Choose an image first.");
  showLoading(["Uploading...", "Running image model...", "Calculating confidence..."]);
  try {
    const fd = new FormData(); fd.append("file", currentImage);
    await saveAndRender(await post("/api/detect/image", fd));
  } catch (e) { alert("Error: " + e.message); } finally { hideLoading(); }
});

document.getElementById("videoInput").addEventListener("change", (e) => {
  currentVideo = e.target.files[0];
  if (currentVideo) {
    const v = $("videoPreview");
    v.src = URL.createObjectURL(currentVideo);
    v.classList.remove("hidden");
  }
});
document.getElementById("analyzeVideoBtn").addEventListener("click", async () => {
  if (!currentVideo) return alert("Choose a video first.");
  showLoading(["Extracting frames...", "Detecting faces...", "Running model...", "Aggregating results..."]);
  try {
    const fd = new FormData(); fd.append("file", currentVideo);
    await saveAndRender(await post("/api/detect/video", fd));
  } catch (e) { alert("Error: " + e.message); } finally { hideLoading(); }
});

document.getElementById("audioInput").addEventListener("change", (e) => {
  currentAudio = e.target.files[0];
  if (currentAudio) {
    const a = $("audioPreview");
    a.src = URL.createObjectURL(currentAudio);
    a.classList.remove("hidden");
  }
});
document.getElementById("analyzeAudioBtn").addEventListener("click", async () => {
  if (!currentAudio) return alert("Choose an audio file first.");
  showLoading(["Extracting waveform...", "Running RawNet model...", "Calculating confidence..."]);
  try {
    const fd = new FormData(); fd.append("file", currentAudio);
    await saveAndRender(await post("/api/detect/audio", fd));
  } catch (e) { alert("Error: " + e.message); } finally { hideLoading(); }
});

// Drag & drop
function wireDrop(dropId, inputId, kind) {
  const zone = $(dropId);
  ["dragover", "dragenter"].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.style.borderColor = "#5b86e5"; }));
  ["dragleave", "drop"].forEach((ev) => zone.addEventListener(ev, (e) => { e.preventDefault(); zone.style.borderColor = ""; }));
  zone.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files[0];
    if (!f) return;
    const ok = kind === "video" ? (f.type.startsWith("video/") || /\.(mp4|avi|mov|webm)$/i.test(f.name))
      : kind === "audio" ? (f.type.startsWith("audio/") || /\.(wav|mp3|flac|ogg|m4a|aac|opus)$/i.test(f.name))
      : f.type.startsWith("image/");
    if (!ok) return alert(kind === "video" ? "Use MP4, AVI, MOV or WEBM."
      : kind === "audio" ? "Use WAV, MP3, FLAC, OGG, M4A or AAC."
      : "Use PNG, JPG, JPEG or WEBP.");
    const input = $(inputId);
    const dt = new DataTransfer();
    dt.items.add(f);
    input.files = dt.files;
    input.dispatchEvent(new Event("change"));
  });
}
wireDrop("imageDrop", "imageInput", "image");
wireDrop("videoDrop", "videoInput", "video");
wireDrop("audioDrop", "audioInput", "audio");

document.addEventListener("DOMContentLoaded", async () => {
  await loadTheme();
  await loadSettings();
  await checkHealth();
  await restoreLastResult();

  $("refreshButton").addEventListener("click", checkHealth);
  $("themeToggle").addEventListener("click", toggleTheme);
  $("optionsButton").addEventListener("click", openSettings);
  $("footerSettings").addEventListener("click", openSettings);

  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const tag = (e.target && e.target.tagName) || "";
    if (tag === "INPUT" || tag === "TEXTAREA") return;
    const key = e.key.toLowerCase();
    if (key === "t") toggleTheme();
    else if (key === "r") checkHealth();
    else if (key === "s") openSettings();
  });
});
