#!/usr/bin/env node
"use strict";
/* Exercises ui.js rendering/tick paths headlessly across every screen/state. */
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const ROOT = path.join(__dirname, "..");
const files = ["js/core.js", "js/content.js", "js/game.js", "js/ui.js"].map(f =>
  fs.readFileSync(path.join(ROOT, f), "utf8"));

const grad = { addColorStop() {} };
const ctx = new Proxy(function () {}, {
  get(t, k) {
    if (k === "createLinearGradient" || k === "createRadialGradient") return () => grad;
    if (k === "measureText") return () => ({ width: 42 });
    return ctx;
  },
  set() { return true; },
  apply() { return ctx; },
});
const cv = () => ({ width: 960, height: 540, style: {}, getContext: () => ctx });
const store = new Map();
let rafCb = null;
const sb = {
  console, Math, JSON, Object, Array, Set, Map, parseInt, parseFloat, isNaN,
  setInterval: () => 0, setTimeout: (f) => { try { f(); } catch (e) { console.error("setTimeout cb err:", e.message); } return 0; },
  clearTimeout: () => {},
  performance: { now: () => Date.now() },
  requestAnimationFrame: cb => { rafCb = cb; return 1; },
  navigator: {},
  addEventListener() {},
  innerWidth: 1280, innerHeight: 720,
  document: { createElement: cv, addEventListener() {}, getElementById: () => cv() },
  localStorage: {
    getItem: k => (store.has(k) ? store.get(k) : null),
    setItem: (k, v) => store.set(k, String(v)),
    removeItem: k => store.delete(k),
  },
};
sb.window = sb;
vm.createContext(sb);

let now = 0;
function frames(n) {
  for (let i = 0; i < n; i++) {
    now += 16.7;
    const cb = rafCb; rafCb = null;
    if (!cb) throw new Error("rAF chain broken");
    cb(now);
  }
}

const script = files.join("\n;\n") + "\n;window.__test = { get G(){return G;}, get Input(){return Input;}, loadMap, initMenu, openShop, toast, banner, startTransition, startEnding, newGame };";
try {
  vm.runInContext(script, sb, { filename: "ui-bundle.js", timeout: 60000 });
} catch (e) {
  console.error("LOAD CRASH:", e.stack || e.message);
  process.exit(2);
}
const T = sb.__test;
if (!T) { console.error("no test export"); process.exit(2); }

// ---- drive the UI through every state ----
try {
  // title (boot registered the first rAF)
  frames(10);
  console.log("title frames ok");

  // start new game via simulated Enter
  T.Input.pressedQueue.add("Enter");
  frames(3);
  if (T.G.state !== "play") throw new Error("new game didn't start, state=" + T.G.state);
  console.log("new game ok");

  // play frames with dialog up (intro), then dismiss
  frames(30);
  T.Input.pressedQueue.add("KeyE"); frames(1); // complete line typing
  for (let i = 0; i < 6; i++) { T.Input.pressedQueue.add("KeyJ"); frames(1); } // advance all lines
  T.G.dialog = null;
  frames(120);
  console.log("play frames ok, pos", Math.round(T.G.player.x), Math.round(T.G.player.y));

  // wander between maps to exercise all bakes + HUD + lighting
  for (const [id, x, y] of [["meadow", 28, 3], ["forest", 2, 21], ["cave", 24, 32], ["ember", 28, 40], ["shrine", 22, 39], ["village", 20, 25]]) {
    T.loadMap(id, x, y);
    frames(20);
    console.log("rendered", id);
  }

  // combat visuals: melee + skills + projectiles
  T.G.player.invulnT = 0;
  T.Input.pressedQueue.add("KeyJ"); frames(20);
  T.Input.pressedQueue.add("Digit1"); frames(3);
  T.Input.pressedQueue.add("Digit2"); frames(3);
  T.Input.pressedQueue.add("Digit3"); frames(3);
  T.Input.pressedQueue.add("KeyK"); frames(12);
  console.log("combat visuals ok");

  // the combat dash may have carried the bot into a portal (legit!) — settle it
  if (T.G.state === "transition") { frames(80); }
  if (T.G.state === "transition") { T.G.trans = null; T.G.state = "play"; }
  console.log("settled into", T.G.mapId);

  // menu: all tabs
  T.initMenu(); frames(2);
  for (const tab of [0, 1, 2]) { T.G.menu.tab = tab; T.G.menu.sel = 0; frames(3); }
  T.Input.pressedQueue.add("KeyI"); frames(1); // close
  if (T.G.menuOpen) throw new Error("menu didn't close");
  console.log("menu ok");

  // shop
  T.openShop("village"); frames(3);
  T.G.gold = 9999; T.Input.pressedQueue.add("KeyJ"); frames(2); // buy first
  T.Input.pressedQueue.add("KeyI"); frames(1);
  console.log("shop ok");

  // toast + banner
  T.toast("A TOAST", 5); T.banner("A BANNER", "sub", 5); frames(5);
  console.log("toast/banner ok");

  // transition
  T.startTransition("meadow", 28, 3); frames(60);
  if (T.G.trans) throw new Error("transition stuck");
  console.log("transition ok");

  // pause
  T.Input.pressedQueue.add("Escape"); frames(2);
  if (!T.G.paused) throw new Error("pause didn't engage");
  T.Input.pressedQueue.add("Escape"); frames(2);
  console.log("pause ok");

  // dead screen
  T.G.state = "dead"; T.G.deadT = 2; frames(5);
  T.Input.pressedQueue.add("Enter"); frames(3);
  if (T.G.state !== "play") throw new Error("respawn failed");
  console.log("dead/respawn ok");

  // ending + victory
  T.startEnding(); frames(30);
  T.G.endingT = 20; frames(3);
  T.Input.pressedQueue.add("Enter"); frames(3);
  if (T.G.state !== "title") throw new Error("victory->title failed, state=" + T.G.state);
  console.log("ending/victory ok");

  // continue option on title (save exists from autosaves)
  frames(5);
  T.Input.pressedQueue.add("ArrowDown"); frames(2);
  T.Input.pressedQueue.add("Enter"); frames(3);
  console.log("continue ->", T.G.state, T.G.mapId);

  console.log("\nUI PASS — all screens rendered headless.");
} catch (e) {
  console.error("UI FAIL:", e.stack || e.message);
  process.exit(1);
}
