// Placeholder content script: lets the service worker message the active tab if needed.
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg && msg.type === "ping") sendResponse({ ok: true });
});
