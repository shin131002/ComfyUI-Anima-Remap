/*
 * LoRA autocomplete for the Anima LoRA Tag Loader text box.
 *
 * Type 3+ characters of a LoRA filename and a candidate list appears (partial,
 * case-insensitive match). Highlighting a candidate shows its preview image and
 * trigger words; choosing one replaces the typed fragment with
 *   <lora:NAME:1>, trigger, words
 *
 * Scope is deliberately narrow: only the `text` widget of the two Anima tag
 * loader nodes, and only while it is a real textarea. When the widget is fed
 * from an upstream node there is no textarea, so nothing is attached and the
 * connected behaviour is unchanged.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET_NODES = new Set(["AnimaLoRARemapTagLoader", "AnimaLoRARemapExtendedTagLoader"]);
const MIN_CHARS = 3;
const MAX_RESULTS = 30;
const LIST_TTL_MS = 30000;
const LORA_EXTS = [".safetensors", ".pt", ".ckpt", ".sft"];

// ---------------------------------------------------------------------------
// Pure helpers (no DOM)
// ---------------------------------------------------------------------------

function stripExt(name) {
    const low = name.toLowerCase();
    for (const ext of LORA_EXTS) {
        if (low.endsWith(ext)) return name.slice(0, -ext.length);
    }
    return name;
}

/**
 * The fragment being typed right before the caret, or null when no list should open.
 *  - inside an unclosed "<...>" (e.g. editing an existing <lora:...> tag) -> null
 *  - fewer than MIN_CHARS characters -> null
 *  - purely numeric (typing a weight such as 0.85) -> null
 */
function extractToken(textBeforeCaret) {
    if (textBeforeCaret.lastIndexOf("<") > textBeforeCaret.lastIndexOf(">")) return null;
    const m = textBeforeCaret.match(/([^\s,<>()\[\]{}:|]+)$/);
    if (!m) return null;
    const token = m[1];
    if (token.length < MIN_CHARS) return null;
    if (/^[\d.]+$/.test(token)) return null;
    return token;
}

/** Prefix matches on the filename first, then substring matches, then subfolder-path matches. */
function rankMatches(list, token) {
    const q = token.toLowerCase();
    const scored = [];
    for (const item of list) {
        const n = item.n.toLowerCase();
        let score;
        if (n.startsWith(q)) score = 0;
        else if (n.includes(q)) score = 1;
        else if (item.p.toLowerCase().includes(q)) score = 2;
        else continue;
        scored.push([score, n, item]);
    }
    scored.sort((a, b) => a[0] - b[0] || (a[1] < b[1] ? -1 : a[1] > b[1] ? 1 : 0));
    return scored.slice(0, MAX_RESULTS).map((s) => s[2]);
}

/** Bare filename normally; "subfolder/name" only when the bare name is shared by several files. */
function insertNameFor(item) {
    return item.d ? stripExt(item.p) : item.n;
}

function buildSnippet(item, triggerWords) {
    let s = `<lora:${insertNameFor(item)}:1>`;
    if (triggerWords && triggerWords.length) s += ", " + triggerWords.join(", ");
    return s;
}

// ---------------------------------------------------------------------------
// Data (cached)
// ---------------------------------------------------------------------------

let loraList = null;
let loraListAt = 0;
let loraListPromise = null;

function getLoraList() {
    const fresh = loraList && Date.now() - loraListAt < LIST_TTL_MS;
    if (fresh) return Promise.resolve(loraList);
    if (!loraListPromise) {
        loraListPromise = api
            .fetchApi("/anima_remap/loras")
            .then((r) => (r.ok ? r.json() : []))
            .then((list) => {
                loraList = Array.isArray(list) ? list : [];
                loraListAt = Date.now();
                return loraList;
            })
            .catch(() => loraList || [])
            .finally(() => {
                loraListPromise = null;
            });
    }
    return loraListPromise;
}

const infoCache = new Map();

function getLoraInfo(path) {
    if (!infoCache.has(path)) {
        const p = api
            .fetchApi("/anima_remap/lora_info?path=" + encodeURIComponent(path))
            .then((r) => (r.ok ? r.json() : { trigger_words: [], preview_type: null }))
            .catch(() => ({ trigger_words: [], preview_type: null }));
        infoCache.set(path, p);
    }
    return infoCache.get(path);
}

// ---------------------------------------------------------------------------
// Dropdown (one shared instance)
// ---------------------------------------------------------------------------

const ui = {
    root: null,
    list: null,
    preview: null,
    img: null,
    video: null,
    placeholder: null,
    triggers: null,
    items: [],
    active: 0,
    textarea: null,
    token: null,
    tokenStart: 0,
    previewTimer: null,
    previewFor: null,
    lastMouse: null,
};

function ensureUI() {
    if (ui.root) return;
    const root = document.createElement("div");
    root.style.cssText = [
        "position:fixed", "z-index:10000", "display:none", "gap:6px",
        "background:var(--comfy-menu-bg,#202020)", "color:var(--fg-color,#ddd)",
        "border:1px solid var(--border-color,#555)", "border-radius:6px",
        "box-shadow:0 4px 16px rgba(0,0,0,.5)", "padding:4px",
        "font-size:12px", "font-family:sans-serif",
    ].join(";");

    const list = document.createElement("div");
    list.style.cssText = "max-height:280px;overflow-y:auto;min-width:220px;max-width:360px";

    const preview = document.createElement("div");
    preview.style.cssText = "width:200px;display:flex;flex-direction:column;gap:4px";
    const img = document.createElement("img");
    img.style.cssText = "max-width:200px;max-height:200px;object-fit:contain;border-radius:4px;display:none";
    // Video previews loop silently, the same way animated GIF/WebP previews already
    // animate in the <img>. `muted` is required: browsers refuse to autoplay video with sound.
    const video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    video.autoplay = true;
    video.loop = true;
    video.preload = "auto";
    video.style.cssText = "max-width:200px;max-height:200px;object-fit:contain;border-radius:4px;display:none";
    const placeholder = document.createElement("div");
    placeholder.textContent = "No preview";
    placeholder.style.cssText = "height:60px;display:flex;align-items:center;justify-content:center;opacity:.5;border:1px dashed var(--border-color,#555);border-radius:4px";
    const triggers = document.createElement("div");
    triggers.style.cssText = "opacity:.85;word-break:break-word;line-height:1.35";
    preview.append(img, video, placeholder, triggers);

    root.append(list, preview);

    // Keep pointer/wheel events from reaching the canvas underneath, and keep
    // mousedown from blurring the textarea before a click registers.
    for (const ev of ["pointerdown", "mousedown", "wheel", "contextmenu"]) {
        root.addEventListener(ev, (e) => {
            e.stopPropagation();
            if (ev === "mousedown") e.preventDefault();
        });
    }

    document.body.appendChild(root);
    Object.assign(ui, { root, list, preview, img, video, placeholder, triggers });
}

function isOpen() {
    return ui.root && ui.root.style.display !== "none";
}

function close() {
    if (!ui.root) return;
    ui.root.style.display = "none";
    clearVideo();
    ui.items = [];
    ui.textarea = null;
    ui.token = null;
    clearTimeout(ui.previewTimer);
}

const ROW_BG_ACTIVE = "var(--comfy-input-bg,#333)";

/**
 * Build the rows ONCE per candidate set. Moving the selection must not rebuild
 * them: replacing the element under a stationary cursor makes the browser treat
 * it as a fresh hover, which (a) kept restarting the preview timer so the preview
 * never updated until the cursor left the list, and (b) snatched the selection
 * back to the row under the mouse after every arrow-key press.
 */
function renderList() {
    ui.list.textContent = "";
    ui.items.forEach((item, i) => {
        const row = document.createElement("div");
        row.style.cssText = "padding:3px 6px;border-radius:4px;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis";

        const name = document.createElement("span");
        name.textContent = item.n;
        row.appendChild(name);
        const folder = item.p.includes("/") ? item.p.slice(0, item.p.lastIndexOf("/")) : "";
        if (folder) {
            const sub = document.createElement("span");
            sub.textContent = "  " + folder + (item.d ? "  (duplicate name)" : "");
            sub.style.opacity = ".5";
            row.appendChild(sub);
        }

        // mousemove + "did the pointer actually move" instead of mouseenter:
        // scrolling the list with the arrow keys slides a different row under a
        // cursor that hasn't moved, and that must not count as the user hovering it.
        row.addEventListener("mousemove", (e) => {
            const last = ui.lastMouse;
            if (last && last.x === e.clientX && last.y === e.clientY) return;
            ui.lastMouse = { x: e.clientX, y: e.clientY };
            if (i !== ui.active) setActive(i, false);
        });
        row.addEventListener("mousedown", () => accept(i));
        ui.list.appendChild(row);
    });
    highlight();
}

function highlight() {
    Array.from(ui.list.children).forEach((row, i) => {
        row.style.background = i === ui.active ? ROW_BG_ACTIVE : "";
    });
}

function setActive(i, scroll = true) {
    if (!ui.items.length) return;
    const next = (i + ui.items.length) % ui.items.length;
    if (next === ui.active) return;
    ui.active = next;
    highlight();
    if (scroll) ui.list.children[ui.active]?.scrollIntoView({ block: "nearest" });
    schedulePreview();
}

function schedulePreview() {
    clearTimeout(ui.previewTimer);
    const item = ui.items[ui.active];
    if (!item) return;
    ui.previewTimer = setTimeout(async () => {
        ui.previewFor = item.p;
        const info = await getLoraInfo(item.p);
        if (ui.previewFor !== item.p || !isOpen()) return; // highlight moved on meanwhile
        showPreview(item.p, info.preview_type);
        const tw = info.trigger_words || [];
        ui.triggers.textContent = tw.length ? tw.join(", ") : "(no trigger words)";
        position();
    }, 100);
}

function clearVideo() {
    if (ui.video.getAttribute("src")) {
        ui.video.removeAttribute("src");
        ui.video.load(); // aborts any in-flight download of the previous clip
    }
    ui.video.style.display = "none";
}

function showPreview(path, type) {
    const url = api.apiURL("/anima_remap/lora_preview?path=" + encodeURIComponent(path));
    ui.img.style.display = "none";
    ui.placeholder.style.display = "none";
    if (type === "image") {
        clearVideo();
        ui.img.src = url;
        ui.img.style.display = "block";
    } else if (type === "video") {
        ui.img.removeAttribute("src");
        ui.video.src = url;
        ui.video.style.display = "block";
    } else {
        clearVideo();
        ui.img.removeAttribute("src");
        ui.placeholder.style.display = "flex";
    }
}

/** Pixel position of the caret, accounting for the canvas zoom (CSS transform) on the textarea. */
function caretRect(el) {
    const style = getComputedStyle(el);
    const mirror = document.createElement("div");
    for (const p of [
        "boxSizing", "borderTopWidth", "borderRightWidth", "borderBottomWidth", "borderLeftWidth",
        "paddingTop", "paddingRight", "paddingBottom", "paddingLeft", "fontStyle", "fontVariant",
        "fontWeight", "fontStretch", "fontSize", "lineHeight", "fontFamily", "textAlign",
        "textTransform", "textIndent", "letterSpacing", "wordSpacing", "tabSize",
    ]) mirror.style[p] = style[p];
    mirror.style.cssText += ";position:absolute;visibility:hidden;top:0;left:-9999px;white-space:pre-wrap;word-wrap:break-word;overflow:hidden";
    mirror.style.width = el.offsetWidth + "px";
    mirror.textContent = el.value.slice(0, el.selectionStart);
    const marker = document.createElement("span");
    marker.textContent = "\u200b";
    mirror.appendChild(marker);
    document.body.appendChild(mirror);
    const x = marker.offsetLeft;
    const y = marker.offsetTop;
    const lh = marker.offsetHeight || parseFloat(style.fontSize) * 1.3;
    document.body.removeChild(mirror);

    const rect = el.getBoundingClientRect();
    const scale = el.offsetWidth ? rect.width / el.offsetWidth : 1;
    const left = rect.left + (x - el.scrollLeft) * scale;
    const top = rect.top + (y - el.scrollTop) * scale;
    return { left, top, bottom: top + lh * scale };
}

function position() {
    if (!ui.textarea) return;
    const c = caretRect(ui.textarea);
    const box = ui.root.getBoundingClientRect();
    let left = Math.min(c.left, window.innerWidth - box.width - 8);
    let top = c.bottom + 4;
    if (top + box.height > window.innerHeight - 8) top = c.top - box.height - 4;
    ui.root.style.left = Math.max(8, left) + "px";
    ui.root.style.top = Math.max(8, top) + "px";
}

async function update(el) {
    const caret = el.selectionStart;
    if (caret !== el.selectionEnd) return close();
    const token = extractToken(el.value.slice(0, caret));
    if (!token) return close();

    const list = await getLoraList();
    // The user may have kept typing while the list loaded.
    if (el.selectionStart !== caret || extractToken(el.value.slice(0, caret)) !== token) return;

    const items = rankMatches(list, token);
    if (!items.length) return close();

    ensureUI();
    ui.textarea = el;
    ui.token = token;
    ui.tokenStart = caret - token.length;
    ui.items = items;
    ui.active = 0;
    ui.lastMouse = null;
    ui.root.style.display = "flex";
    renderList();
    position();
    schedulePreview();
}

async function accept(i) {
    const el = ui.textarea;
    const item = ui.items[i];
    const token = ui.token;
    const start = ui.tokenStart;
    if (!el || !item) return close();
    close();

    const info = await getLoraInfo(item.p);
    // Only replace if the typed fragment is still exactly where we found it.
    if (el.value.slice(start, start + token.length) !== token) return;

    const snippet = buildSnippet(item, info.trigger_words);
    el.value = el.value.slice(0, start) + snippet + el.value.slice(start + token.length);
    const pos = start + snippet.length;
    el.focus();
    el.setSelectionRange(pos, pos);

    // Let the frontend pick up the new value (widget value / workflow-modified state)
    // without our own input handler reopening the list on the inserted trigger words.
    el.__animaSuppress = true;
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.__animaSuppress = false;
    app.graph?.setDirtyCanvas?.(true, true);
}

// ---------------------------------------------------------------------------
// Wiring
// ---------------------------------------------------------------------------

function attach(el) {
    if (!el || el.tagName !== "TEXTAREA" || el.__animaAutocomplete) return;
    el.__animaAutocomplete = true;

    el.addEventListener("focus", () => getLoraList());
    el.addEventListener("input", () => {
        if (!el.__animaSuppress) update(el);
    });
    el.addEventListener("blur", () => setTimeout(() => ui.textarea === el && close(), 150));
    el.addEventListener("scroll", () => ui.textarea === el && close());
    el.addEventListener("keyup", (e) => {
        if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key) && ui.textarea === el) close();
    });

    // Capture phase so these keys are handled before the canvas/global shortcuts see them.
    el.addEventListener(
        "keydown",
        (e) => {
            if (!isOpen() || ui.textarea !== el) return;
            if (e.ctrlKey || e.metaKey || e.altKey) return; // leave Ctrl+Enter (queue) etc. alone
            let handled = true;
            if (e.key === "ArrowDown") setActive(ui.active + 1);
            else if (e.key === "ArrowUp") setActive(ui.active - 1);
            else if ((e.key === "Enter" && !e.shiftKey) || e.key === "Tab") accept(ui.active);
            else if (e.key === "Escape") close();
            else handled = false;
            if (handled) {
                e.preventDefault();
                e.stopImmediatePropagation();
            }
        },
        true,
    );
}

function attachToNode(node, tries = 0) {
    if (!TARGET_NODES.has(node?.comfyClass)) return;
    const w = node.widgets?.find((w) => w.name === "text");
    const el = w?.inputEl || w?.element;
    if (el) return attach(el);
    // The DOM element can appear a frame or two after the node itself.
    if (tries < 10) requestAnimationFrame(() => attachToNode(node, tries + 1));
}

app.registerExtension({
    name: "AnimaRemap.LoraAutocomplete",
    nodeCreated(node) {
        attachToNode(node);
    },
    loadedGraphNode(node) {
        attachToNode(node);
    },
});
