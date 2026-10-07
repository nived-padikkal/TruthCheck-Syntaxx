chrome.runtime.onInstalled.addListener(ensureMenu);
chrome.runtime.onStartup.addListener(ensureMenu);
ensureMenu(); // run on every service-worker load so the menu exists after reload
function ensureMenu() {
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({
      id: "tc-text",
      title: "🔍 Scan selected text",
      contexts: ["selection"]
    });
    chrome.contextMenus.create({
      id: "scan-deepfake",
      title: "🔍 Scan with TruthCheck",
      contexts: ["image", "video"]
    });
  });
}

function showOverlay(tabId, title, text) {
  chrome.scripting.executeScript({
    target: { tabId },
    func: (title, text) => {
      const old = document.getElementById("truthcheck-overlay");
      if (old) old.remove();
      const box = document.createElement("div");
      box.id = "truthcheck-overlay";
      box.style.cssText =
        "position:fixed;top:20px;right:20px;z-index:2147483647;max-width:400px;" +
        "background:#111;color:#fff;padding:16px;border-radius:10px;" +
        "box-shadow:0 4px 20px rgba(0,0,0,.5);font-family:Arial,sans-serif;" +
        "white-space:pre-wrap;font-size:13px;";
      box.innerHTML = "<b>✅ " + title + "</b><hr>" + text.replace(/</g, "&lt;");
      const close = document.createElement("button");
      close.textContent = "✕ Close";
      close.style.cssText = "margin-top:10px;padding:4px 10px;cursor:pointer;";
      close.onclick = () => box.remove();
      box.appendChild(document.createElement("br"));
      box.appendChild(close);
      document.body.appendChild(box);
    },
    args: [title, text]
  });
}

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === "tc-text") {
    const text = (info.selectionText || "").trim();
    if (!text) return;
    try {
      showOverlay(tab.id, "TruthCheck", "⏳ Analyzing text...");
      const r = await fetch("http://127.0.0.1:8000/analyze/text", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, verify: true }),
        signal: AbortSignal.timeout(30000),
      });
      const data = await r.json();
      if (!r.ok) throw new Error(data.detail || "HTTP " + r.status);
      const pct = Math.round((data.confidence || 0) * 100);
      const verdict = data.verdict || "UNKNOWN";
      const icon = verdict === "REAL" ? "✅" : verdict === "FAKE" ? "🚨" : "⚠️";
      const sigs = (data.details && data.details.signals || []).slice(0, 5)
        .map(s => "• " + s).join("\n");
      showOverlay(tab.id, `Text: ${verdict} (${pct}%)`,
        `${icon} ${verdict}\nConfidence: ${pct}%\nModel: ${(data.details && data.details.model) || "-"}\n` +
        (sigs ? "\nSignals:\n" + sigs : ""));
      // Save in the popup's shape so the dashboard shows it too.
      chrome.storage.local.set({ lastResult: { at: Date.now(), result: data } });
    } catch (err) {
      showOverlay(tab.id, "Text analysis error", String(err));
    }
    return;
  }
  if (info.menuItemId !== "scan-deepfake" || !info.srcUrl) return;

  try {
    showOverlay(tab.id, "TruthCheck", "⏳ Analyzing media...");

    const response = await fetch(info.srcUrl);
    const blob = await response.blob();
    const extMap = {
      "image/jpeg": ".jpg", "image/jpg": ".jpg", "image/png": ".png", "image/gif": ".gif",
      "image/bmp": ".bmp", "image/webp": ".webp",
      "video/mp4": ".mp4", "video/webm": ".webm", "video/quicktime": ".mov"
    };
    const videoExtSet = [".mp4", ".avi", ".mov", ".webm"];
    let filename = info.srcUrl.split("/").pop().split("?")[0] || "file";
    const isVideoFile = videoExtSet.some(e => filename.toLowerCase().endsWith(e)) || extMap[blob.type]?.startsWith(".mp4") || blob.type?.startsWith("video/");
    if (!/\.(jpg|jpeg|png|gif|bmp|webp|mp4|avi|mov|webm)$/i.test(filename)) {
      filename += extMap[blob.type] || ".jpg";
    }

    const formData = new FormData();
    formData.append("file", blob, filename);

    const url = isVideoFile ? "http://127.0.0.1:5000/api/detect/video" : "http://127.0.0.1:5000/api/detect/image";
    const res = await fetch(url, { method: "POST", body: formData });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.error || ("HTTP " + res.status));

    const isReal = data.prediction === "REAL";
    const verdict = isReal ? "✅ RESULT: LIKELY REAL"
      : data.prediction === "DEEPFAKE" ? "🚨 RESULT: DEEPFAKE VIDEO"
      : data.prediction === "AI_GENERATED" ? "🚨 RESULT: FAKE / AI-GENERATED"
      : "⚠️ RESULT: SUSPICIOUS";
    const text =
      `${verdict}\nPrediction: ${data.prediction}\nConfidence: ${(data.confidence * 100).toFixed(1)}%\n` +
      (data.frames_analyzed ? `Frames analyzed: ${data.frames_analyzed} (suspicious: ${data.suspicious_frames})\n` : "") +
      `Risk: ${data.risk}\nModel: ${data.model}`;

    showOverlay(tab.id, "TruthCheck Result", text);

    // Save in the popup's shape: { at, result: {verdict, confidence, details} }
    const v = data.prediction === "REAL" ? "REAL"
      : data.prediction === "SUSPICIOUS" ? "SUSPICIOUS"
      : "FAKE"; // AI_GENERATED / DEEPFAKE / FAKE
    await chrome.storage.local.set({
      lastResult: {
        at: Date.now(),
        result: {
          verdict: v,
          confidence: data.confidence || 0,
          details: {
            model: data.model,
            p_fake: data.fake_probability,
            risk: data.risk,
            frames_analyzed: data.frames_analyzed,
            suspicious_frames: data.suspicious_frames,
            type: data.type,
          },
        },
      },
      lastScanned: info.srcUrl,
    });
  } catch (err) {
    showOverlay(tab.id, "TruthCheck Error", String(err));
    chrome.storage.local.set({
      lastResult: {
        at: Date.now(),
        result: { verdict: "ERROR", confidence: 0, details: { model: "-", p_fake: 0, risk: "HIGH", error: String(err) } },
      },
    });
  }
});
