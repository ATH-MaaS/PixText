import { api } from "../../scripts/api.js";

// One status request/poll loop for every Detect Words node in this browser.
const subscribers = new Set();
let current = null, timer = null, pending = null, prepareQueued = false;
const el = (tag, className, text) => {
  const item = document.createElement(tag);
  item.className = className;
  if (text !== undefined) item.textContent = text;
  return item;
};
const publish = (state) => { current = state; for (const render of subscribers) render(state); };
const megabytes = (bytes) => `${(bytes / 1048576).toFixed(1)} MB`;

async function refresh(prepare = false) {
  if (pending) {
    if (!prepare || prepareQueued) return pending;
    prepareQueued = true;
    publish({ ...current, busy: true, state: "checking", message: "Waiting for local model check…" });
    await pending;
    prepareQueued = false;
    return refresh(true);
  }
  clearTimeout(timer);
  if (prepare) publish({ ...current, busy: true, state: "checking", message: "Checking local files…" });
  pending = (async () => {
    try {
      const response = await api.fetchApi(`/pixtext/ocr/models${prepare ? "/prepare" : ""}`, {
        method: prepare ? "POST" : "GET", cache: "no-store",
      });
      if (!response.ok) throw new Error(`OCR setup unavailable (HTTP ${response.status}). Retry when connected.`);
      publish(await response.json());
    } catch (error) {
      publish({ ...current, state: "error", busy: false, message: error.message });
    } finally {
      pending = null;
      if (subscribers.size) timer = setTimeout(() => refresh(), current?.busy ? 800 : 10000);
    }
  })();
  return pending;
}

export function setupOCR(node) {
  if (node.ptOCRSetup) return;
  const card = el("section", "pt-ocr-setup");
  card.setAttribute("aria-label", "OCR model setup");
  const header = el("div", "pt-ocr-header");
  const status = el("span", "pt-ocr-status", "Checking");
  header.append(el("strong", "", "OCR models"), status);
  const files = el("div", "pt-ocr-files");
  const track = el("div", "pt-ocr-track");
  track.setAttribute("role", "progressbar");
  track.setAttribute("aria-label", "OCR model preparation");
  track.setAttribute("aria-valuemin", "0"); track.setAttribute("aria-valuemax", "100");
  const fill = el("div", "pt-ocr-fill"); track.append(fill);
  const detail = el("div", "pt-ocr-detail", "Checking local models…");
  detail.setAttribute("role", "status");
  const action = el("button", "pt-ocr-action", "Prepare OCR models");
  action.type = "button"; action.disabled = true;
  action.addEventListener("click", () => {
    if (!action.disabled && !current?.busy) refresh(true);
  });
  const caption = el("small", "pt-ocr-caption", "Stored on this ComfyUI server");
  card.append(header, files, track, detail, action, caption);
  const chips = ["CRAFT", "PP-OCRv6", "Dictionary"].map((label) => {
    const chip = el("span", "pt-ocr-chip", label); files.append(chip); return chip;
  });
  const render = (data) => {
    const ready = data.state === "ready", busy = Boolean(data.busy);
    card.dataset.state = data.state;
    card.setAttribute("aria-busy", String(busy));
    status.textContent = ready ? "Verified" : data.state === "error" ? "Needs attention" : busy ? "Preparing" : "Setup needed";
    const percent = Math.max(0, Math.min(100, data.progress || 0));
    fill.style.width = `${percent}%`;
    track.setAttribute("aria-valuenow", String(percent));
    chips.forEach((chip, index) => {
      const row = data.files?.[index];
      chip.dataset.state = row?.state || "pending";
      chip.textContent = `${row?.state === "verified" ? "✓ " : ""}${["CRAFT", "PP-OCRv6", "Dictionary"][index]}`;
      chip.title = `${row?.name || ""} · ${row?.state || "pending"}`;
    });
    const transfer = data.state === "downloading" && data.received
      ? ` · ${megabytes(data.received)}${data.total ? ` / ${megabytes(data.total)}` : ""}` : "";
    detail.textContent = data.message + transfer;
    if (ready && data.missing_dependencies?.length) {
      detail.textContent = "Models verified. Install requirements-ocr.txt in ComfyUI’s Python to enable detection.";
    }
    detail.title = data.error_details || detail.textContent;
    action.disabled = busy || ready || prepareQueued;
    action.textContent = ready ? "✓  OCR models ready" : busy ? `Preparing · ${Math.round(percent)}%` : data.state === "error" ? "Retry OCR setup" : "↓  Prepare OCR models";
    caption.textContent = "models/pixtext_ocr" + (data.model_bytes ? ` · ${megabytes(data.model_bytes)}` : "");
    caption.title = data.directory || "ComfyUI model directory";
  };
  subscribers.add(render);
  if (current) render(current);
  const dom = node.addDOMWidget("pixtext_ocr_setup", "div", card, { serialize: false, hideOnZoom: false });
  dom.computeSize = () => [320, 200];
  node.setSize([Math.max(380, node.size[0]), Math.max(390, node.size[1])]);
  node.ptOCRSetup = card;
  const removed = node.onRemoved;
  node.onRemoved = function (...args) {
    subscribers.delete(render);
    if (!subscribers.size) clearTimeout(timer);
    removed?.apply(this, args);
  };
  refresh();
}
