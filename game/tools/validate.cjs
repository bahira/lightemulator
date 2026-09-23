#!/usr/bin/env node
"use strict";
/* AETHERFALL headless validator — runs the game logic with a stub DOM.
   Checks content consistency + smoke-simulates the engine. Exit code 0 = PASS. */
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const ROOT = path.join(__dirname, "..");
const files = ["js/core.js", "js/content.js", "js/game.js"].map(f =>
  fs.readFileSync(path.join(ROOT, f), "utf8"));

/* ---------- DOM stubs ---------- */
const gradStub = { addColorStop() {} };
const ctxStub = new Proxy(function () {}, {
  get(_t, k) {
    if (k === "createLinearGradient" || k === "createRadialGradient") return () => gradStub;
    if (k === "measureText") return () => ({ width: 10 });
    return ctxStub;
  },
  set() { return true; },
  apply() { return ctxStub; },
});
const canvasStub = () => ({ width: 0, height: 0, style: {}, getContext: () => ctxStub });

const store = new Map();
const sandbox = {
  console, Math, JSON, Object, Array, Set, Map, String, Number, Boolean, Symbol,
  parseInt, parseFloat, isNaN,
  setInterval: () => 0, setTimeout: (f) => 0, clearTimeout: () => {},
  performance: { now: () => 0 },
  requestAnimationFrame: () => 0,
  navigator: {},
  document: { createElement: canvasStub, addEventListener() {}, getElementById: () => canvasStub() },
  localStorage: {
    getItem: k => (store.has(k) ? store.get(k) : null),
    setItem: (k, v) => store.set(k, String(v)),
    removeItem: k => store.delete(k),
  },
  __out: { errors: [], log: [] },
};
sandbox.window = sandbox;
sandbox.globalThis = sandbox;
vm.createContext(sandbox);

/* ---------- run engine + tests in one script scope ---------- */
const testSrc = `
;(function(){
  const out = __out;
  const err = m => out.errors.push(m);
  const log = m => out.log.push(m);

  /* == static content checks == */
  for (const [id, def] of Object.entries(MAPS)) {
    const { grid } = parseMapDef(def);
    if (grid.length !== def.h) err(id + ": grid height mismatch");
    for (let y = 0; y < def.h; y++) if (grid[y].length !== def.w) err(id + ": row width mismatch at y=" + y);
    const walk = (tx, ty) => tx >= 0 && ty >= 0 && tx < def.w && ty < def.h && !isSolidChar(grid[ty][tx]);

    for (const [i, por] of def.portals.entries()) {
      const tDef = MAPS[por.to];
      if (!tDef) { err(id + ": portal " + i + " -> unknown map " + por.to); continue; }
      // portal rect must contain at least one walkable tile
      let anyWalk = false;
      for (let yy = por.y; yy < por.y + por.h; yy++)
        for (let xx = por.x; xx < por.x + por.w; xx++)
          if (xx >= 0 && yy >= 0 && xx < def.w && yy < def.h && !isSolidChar(grid[yy][xx])) anyWalk = true;
      if (!anyWalk) err(id + ": portal " + i + " rect fully solid/blocked");
      // target must be walkable
      const tParsed = parseMapDef(tDef);
      const ttx = por.tx, tty = por.ty;
      if (ttx < 0 || tty < 0 || ttx >= tDef.w || tty >= tDef.h) err(id + ": portal " + i + " target OOB");
      else if (isSolidChar(tParsed.grid[tty][ttx])) err(id + ": portal " + i + " target tile solid at " + ttx + "," + tty);
      // target must not sit inside another portal that would instantly re-trigger back
      for (const rp of tDef.portals) {
        if (ttx >= rp.x && ttx < rp.x + rp.w && tty >= rp.y && tty < rp.y + rp.h)
          err(id + ": portal " + i + " target lands inside return-portal of " + rp.to);
      }
    }
    for (const [i, s] of def.spawns.entries()) {
      if (!ENEMIES[s.e]) err(id + ": spawn " + i + " unknown enemy " + s.e);
      if (!walk(s.x, s.y)) err(id + ": spawn " + i + " (" + s.e + ") on solid tile " + s.x + "," + s.y);
    }
    if (def.boss) {
      if (!BOSSES[def.boss.id]) err(id + ": unknown boss " + def.boss.id);
      if (!walk(def.boss.x, def.boss.y)) err(id + ": boss on solid tile");
    }
    for (const [i, n] of def.npcs.entries())
      if (!walk(Math.floor(n.x), Math.floor(n.y))) err(id + ": npc " + n.id + " on solid tile " + n.x + "," + n.y);
    for (const [i, c] of def.chests.entries()) {
      if (!walk(c.x, c.y)) err(id + ": chest " + i + " on solid tile " + c.x + "," + c.y);
      for (const it of c.items) if (!ITEMS[it.id]) err(id + ": chest item unknown: " + it.id);
    }
    // petals only where intended
    let petals = 0;
    for (let y = 0; y < def.h; y++) for (let x = 0; x < def.w; x++) if (grid[y][x] === "*") petals++;
    log(id + ": petals=" + petals);
  }
  let totalPetals = 0;
  for (const def of Object.values(MAPS)) {
    const { grid } = parseMapDef(def);
    for (const row of grid) for (const ch of row) if (ch === "*") totalPetals++;
  }
  if (totalPetals < 6) err("not enough moonpetal nodes for the quest (" + totalPetals + ")");
  for (const [shopId, list] of Object.entries(SHOPS))
    for (const id of list) {
      if (!ITEMS[id]) err("shop " + shopId + " sells unknown item " + id);
      else if (!(ITEMS[id].price > 0)) err("shop item " + id + " has no price");
    }
  log("static checks done, petals total=" + totalPetals);

  /* == runtime smoke == */
  newGame();
  log("newGame ok: map=" + G.mapId + " enemies=" + G.enemies.length + " npcs=" + G.npcs.length);
  G.dialog = null;

  // simulate 5 minutes of idle + walking right + attacks
  Input.held.add("KeyD");
  for (let i = 0; i < 60 * 20; i++) {
    if (i % 40 === 0) Input.pressedQueue.add("KeyJ");
    if (i === 300) Input.pressedQueue.add("Digit1");
    if (i === 320) Input.pressedQueue.add("Digit2");
    if (i === 340) Input.pressedQueue.add("Digit3");
    if (i === 360) Input.pressedQueue.add("KeyK");
    gameTick(1 / 60);
    Input.endFrame();
    if (G.state !== "play") break;
  }
  Input.held.delete("KeyD");
  log("sim done: state=" + G.state + " hp=" + Math.round(G.player.hp) + " pos=" + Math.round(G.player.x) + "," + Math.round(G.player.y));
  if (G.state !== "play") err("player died during idle sim (balance?)");

  // exercise every map + every portal destination
  for (const [id, def] of Object.entries(MAPS)) {
    loadMap(id, Math.floor(def.w / 2), Math.floor(def.h / 2));
    if (G.enemies.length === 0 && def.spawns.length > 0) err("map " + id + " spawned no enemies");
    for (let i = 0; i < 120; i++) { gameTick(1 / 60); Input.endFrame(); }
    for (const por of def.portals) {
      loadMap(por.to, por.tx, por.ty);
      for (let i = 0; i < 30; i++) { gameTick(1 / 60); Input.endFrame(); }
      loadMap(id, Math.floor(def.w / 2), Math.floor(def.h / 2));
    }
    log("map ok: " + id);
  }

  // save / load roundtrip
  gainXP(500); G.gold += 123; addItem("ether", 2);
  saveGame();
  const snap = Save.read();
  if (!snap) err("save produced nothing");
  loadSaveData(snap);
  if (G.gold !== snap.gold) err("save/load gold mismatch");
  if (G.player.lvl !== snap.lvl) err("save/load level mismatch");
  if (!countItem("ether")) err("save/load lost ether");
  log("save/load ok: lvl=" + G.player.lvl + " gold=" + G.gold);

  // shop sanity at runtime
  openShop("village");
  const before = G.gold;
  G.gold = 1000;
  buyItem("potion");
  if (G.gold !== 1000 - ITEMS.potion.price) err("buy math wrong");
  G.gold = before;
  G.shop = null;
  log("ALL DONE");
})();
`;

try {
  vm.runInContext(files.join("\n;\n") + "\n" + testSrc, sandbox, { filename: "bundle.js", timeout: 60000 });
} catch (e) {
  console.error("RUNTIME CRASH:", e.stack || e.message);
  process.exit(2);
}

const { errors, log } = sandbox.__out;
for (const l of log) console.log("  [log]", l);
if (errors.length) {
  console.error("\nFAIL — " + errors.length + " problem(s):");
  for (const e of errors) console.error("  ✗", e);
  process.exit(1);
}
console.log("\nPASS — content valid, engine smoke-tested headless.");
