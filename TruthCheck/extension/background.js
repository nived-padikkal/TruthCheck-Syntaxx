// TruthCheck background: right-click menus for text / image / video -> Flask backend.
const API_DEFAULT = "http://127.0.0.1:5000";

function registerMenus() {
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({ id: "tc-text", title: "TruthCheck: analyze text", contexts: ["selection"] });
    chrome.contextMenus.create({ id: "tc-image", title: "TruthCheck: analyze image", contexts: ["image"] });
    chrome.contextMenus.create({ id: "tc-video", title: "TruthCheck: analyze video", contexts: ["video"] });
  });
}
// Runs on every service-worker load — covers install, browser start and reload.
registerMenus();

async function apiBase() {
  const { apiBase } = await chrome.storage.local.get("apiBase");
  return String(apiBase || API_DEFAULT).replace(/\/+$/, "");
}

function toResult(data) {
  if (data && data.verdict && typeof data.confidence === "number") return data; // already in popup shape
  const p = data.prediction || "";
  const verdict = p === "REAL" ? "REAL" : p === "SUSPICIOUS" ? "SUSPICIOUS" : p === "ERROR" ? "ERROR" : "FAKE";
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
      error: data.error,
    },
  };
}

async function post(path, body) {
  const base = await apiBase();
  const resp = await fetch(base + path, { method: "POST", body });
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
  return data;
}

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  const kind = info.menuItemId === "tc-text" ? "text" : info.menuItemId === "tc-image" ? "image" : "video";
  try {
    let data;
    if (kind === "text") {
      // Primary: ISOT text backend (FastAPI, port 8000)
      try {
        const r = await fetch("http://127.0.0.1:8000/analyze/text", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: info.selectionText || "", verify: true }),
          signal: AbortSignal.timeout(20000),
        });
        data = await r.json();
        if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
      } catch (_) {
        const fd = new FormData();
        fd.append("text", info.selectionText || "");
        data = await post("/api/detect/text", fd);
        data = toResult(data);
      }
    } else {
      const resp = await fetch(info.srcUrl);
      const blob = await resp.blob();
      const fd = new FormData();
      fd.append("file", blob, kind === "image" ? "selection.jpg" : "selection.mp4");
      data = await post(`/api/detect/${kind}`, fd);
    }
    await chrome.storage.local.set({ lastResult: { at: Date.now(), result: toResult(data) } });
  } catch (err) {
    await chrome.storage.local.set({
      lastResult: {
        at: Date.now(),
        result: { verdict: "ERROR", confidence: 0, details: { model: "-", p_fake: 0, risk: "HIGH", error: String(err) } },
      },
    });
  }
});
