import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const css = document.createElement("link");
css.rel = "stylesheet";
css.href = new URL("./comparison.css", import.meta.url).href;
document.head.append(css);
const make = (tag, className, text) => {
  const item = document.createElement(tag);
  if (className) item.className = className;
  if (text !== undefined) item.textContent = text;
  return item;
};
function buildCard(card) {
  const item = make("article", "pt-pair");
  item.dataset.region = String(card.index);
  const heading = make("div", "pt-pair-title");
  heading.append(make("span", "pt-pair-index", String(card.index).padStart(2, "0")), make("span", "pt-pair-target", card.text));
  const columns = make("div", "pt-pair-columns");
  for (const [key, label] of [["before", "BEFORE"], ["after", "AFTER"]]) {
    const figure = make("figure", "pt-pair-image");
    const image = make("img");
    image.src = api.apiURL(`/view?${new URLSearchParams(card[key])}`);
    image.alt = `${label} · Region ${card.index} · ${card.text}`;
    image.loading = "lazy";
    figure.append(make("figcaption", "", label), image);
    columns.append(figure);
  }
  item.append(heading, columns);
  return item;
}
function board(cards, expanded = false) {
  const root = make("section", "pt-comparison");
  const header = make("header", "pt-comparison-header");
  header.append(make("strong", "", `PixText · ${cards.length} region${cards.length === 1 ? "" : "s"}`));
  const action = make("button", "", expanded ? "Close comparison" : "Open comparison");
  action.type = "button";
  const list = make("div", "pt-comparison-list");
  list.addEventListener("wheel", (event) => event.stopPropagation(), { passive: true });
  if (!cards.length) list.append(make("div", "pt-comparison-empty", "Connect the original image, repaired image and targets. Run to see each region before and after."));
  cards.forEach((card) => list.append(buildCard(card)));
  header.append(action); root.append(header, list);
  if (!expanded) action.addEventListener("click", () => {
    if (!cards.length) return;
    const dialog = make("dialog", "pt-comparison-dialog");
    dialog.setAttribute("aria-label", "PixText region comparison");
    const gallery = board(cards, true); dialog.append(gallery);
    gallery.querySelector("button").addEventListener("click", () => dialog.close());
    dialog.addEventListener("close", () => dialog.remove(), { once: true });
    dialog.addEventListener("keydown", (event) => event.stopPropagation());
    document.body.append(dialog); dialog.showModal();
  });
  return root;
}
function setupComparison(node) {
  if (node.ptComparisonHost) return;
  const host = make("div"); host.style.height = "100%";
  node.ptComparisonHost = host;
  const render = () => host.replaceChildren(board(node.properties?.pixtext_comparison?.cards || []));
  const dom = node.addDOMWidget("pixtext_comparison", "div", host, { serialize: false, hideOnZoom: false });
  dom.computeSize = () => [480, 560];
  node.setSize([Math.max(520, node.size[0]), Math.max(660, node.size[1])]);
  const before = node.onExecuted;
  node.onExecuted = function (message) {
    before?.call(this, message);
    if (message.pixtext_comparison?.[0]) {
      this.properties ??= {};
      this.properties.pixtext_comparison = message.pixtext_comparison[0];
      render();
    }
  };
  const configure = node.onConfigure;
  node.onConfigure = function (...args) { configure?.apply(this, args); render(); };
  render();
}

function recommendSteps(node) {
  const input = node.inputs?.find((item) => item.name === "model");
  if (input?.link == null) return;
  const link = app.graph.links[input.link];
  const loader = app.graph.getNodeById(link?.origin_id);
  const name = loader?.widgets?.find((item) => item.name === "model_name")?.value;
  if (typeof name !== "string") return;
  const suggested = name.includes("Teacher") ? 25 : name.includes("MeanFlow") ? 4 : null;
  const steps = node.widgets?.find((item) => item.name === "steps");
  if (!suggested || !steps) return;
  const previous = node.properties?.pixtext_recommended_steps ?? 4;
  if (steps.value === previous || steps.value === 0) {
    steps.value = suggested;
    steps.callback?.(steps.value);
  }
  node.properties ??= {};
  node.properties.pixtext_recommended_steps = suggested;
}
app.registerExtension({
  name: "PixText.RegionComparison",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const created = nodeType.prototype.onNodeCreated;
    if (nodeData.name === "PixTextCompare") {
      nodeType.prototype.onNodeCreated = function (...args) { const result = created?.apply(this, args); setupComparison(this); return result; };
    } else if (nodeData.name === "PixTextRefine") {
      const connected = nodeType.prototype.onConnectionsChange;
      nodeType.prototype.onConnectionsChange = function (...args) { const result = connected?.apply(this, args); queueMicrotask(() => recommendSteps(this)); return result; };
    } else if (nodeData.name === "PixTextModelLoader") {
      nodeType.prototype.onNodeCreated = function (...args) {
        const result = created?.apply(this, args);
        const loader = this;
        const model = this.widgets?.find((item) => item.name === "model_name");
        if (model) {
          const callback = model.callback;
          model.callback = function (...values) {
            callback?.apply(this, values);
            for (const id of loader.outputs?.[0]?.links || []) {
              const link = app.graph.links[id];
              const node = app.graph.getNodeById(link?.target_id);
              if (node?.comfyClass === "PixTextRefine") recommendSteps(node);
            }
            app.graph.setDirtyCanvas(true, true);
          };
        }
        return result;
      };
    }
  },
  loadedGraphNode(node) { if (node.comfyClass === "PixTextRefine") recommendSteps(node); },
});
