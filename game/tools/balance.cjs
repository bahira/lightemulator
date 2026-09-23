#!/usr/bin/env node
"use strict";
/* Balance smoke: TTK vs slime, boss fight duration, full playthrough walkthrough. */
const fs = require("fs");
const path = require("path");
const vm = require("vm");
const ROOT = path.join(__dirname, "..");
const files = ["js/core.js", "js/content.js", "js/game.js"].map(f =>
  fs.readFileSync(path.join(ROOT, f), "utf8"));

const grad = { addColorStop() {} };
const ctx = new Proxy(function () {}, {
  get(t, k) {
    if (k === "createLinearGradient" || k === "createRadialGradient") return () => grad;
    if (k === "measureText") return () => ({ width: 10 });
    return ctx;
  },
  set() { return true; },
  apply() { return ctx; },
});
const cv = () => ({ width: 0, height: 0, style: {}, getContext: () => ctx });
const store = new Map();
const sb = {
  console, Math, JSON, Object, Array, Set, parseInt, parseFloat, isNaN,
  setInterval: () => 0, setTimeout: (f, ms) => { try { f(); } catch (e) {} return 0; }, clearTimeout: () => {},
  performance: { now: () => 0 }, requestAnimationFrame: () => 0, navigator: {},
  document: { createElement: cv, addEventListener() {}, getElementById: () => cv() },
  localStorage: {
    getItem: k => (store.has(k) ? store.get(k) : null),
    setItem: (k, v) => store.set(k, String(v)),
    removeItem: k => store.delete(k),
  },
};
sb.window = sb;
vm.createContext(sb);

const test = `
;(function(){
  const tickN = (n, keys) => {
    for (let i = 0; i < n; i++) {
      if (keys) keys(i);
      gameTick(1/60); Input.endFrame();
      if (G.state !== "play") return false;
    }
    return true;
  };

  newGame(); G.dialog = null;
  const p = G.player;

  // 1) slime TTK
  loadMap("meadow", 10, 15);
  let target = G.enemies.find(e => e.kind === "slime");
  let frames = 0;
  while (target.hp > 0 && frames < 600 && G.state === "play") {
    const a = Math.atan2(target.y - p.y, target.x - p.x);
    p.faceX = Math.cos(a); p.faceY = Math.sin(a); p.dir = a;
    // step toward
    p.x += Math.cos(a) * 2.6; p.y += Math.sin(a) * 2.6;
    Input.pressedQueue.add("KeyJ");
    gameTick(1/60); Input.endFrame(); frames++;
  }
  console.log("slime TTK:", frames, "frames (" + (frames/60).toFixed(2) + "s)");
  if (frames >= 600) console.log("  !! BALANCE: slime unkillable in 10s");
  if (frames < 12) console.log("  !! BALANCE: slime dies instantly");

  // 2) player takes contact damage?
  const hpBefore = p.hp;
  p.invulnT = 0;
  // stand inside slime swarm
  tickN(240);
  console.log("contact damage check: hp", Math.round(hpBefore), "->", Math.round(p.hp), G.state);

  // 3) boss fight: Verdant Guardian, player attacks + novae, no dodging
  loadMap("forest", 38, 10);
  const b = G.enemies.find(e => e.boss);
  frames = 0;
  let bossDead = false, playerDead = false;
  while (frames < 60 * 180) {
    if (b.hp <= 0) { bossDead = true; break; }
    if (G.state !== "play") { playerDead = true; break; }
    const a = Math.atan2(b.y - p.y, b.x - p.x);
    p.faceX = Math.cos(a); p.faceY = Math.sin(a); p.dir = a;
    const dd = Math.hypot(b.x - p.x, b.y - p.y);
    if (dd > 44) { p.x += Math.cos(a) * 2.9; p.y += Math.sin(a) * 2.9; }
    Input.pressedQueue.add("KeyJ");
    if (frames % 60 === 30) Input.pressedQueue.add("Digit1");
    if (frames % 200 === 100) Input.pressedQueue.add("Digit3");
    gameTick(1/60); Input.endFrame(); frames++;
  }
  console.log("guardian fight:", bossDead ? "BOSS DEAD in " + (frames/60).toFixed(1) + "s" : (playerDead ? "player died at " + (frames/60).toFixed(1) + "s (no dodging used)" : "timeout"));
  console.log("  flags.boss_forest =", G.flags.boss_forest, " shards =", G.flags.shards);
  if (bossDead && !G.flags.boss_forest) console.log("  !! boss flag not set");
  if (bossDead && !(G.flags.shards >= 1)) console.log("  !! shard not dropped");

  // 4) level curve: how many slimes to reach lvl 3 from lvl 1?
  const need = (l) => Math.round(45 * Math.pow(l, 1.5));
  console.log("xp to lvl2:", need(1), " lvl5:", need(1)+need(2)+need(3)+need(4), " slime xp=9");

  // 5) full boss gauntlet with a "competent" bot: approach, swing, dodge often, heal, nova
  // give the bot endgame-ish power so the gauntlet is about mechanics not raw stats
  p.lvl = 12; p.equip.weapon = "flamebrand"; p.equip.armor = "knight_plate"; p.equip.acc = "power_band";
  const stBot = pstats(); p.hp = stBot.maxhp; p.mp = stBot.maxmp;
  console.log("bot now lvl", p.lvl, "atk", stBot.atk, "hp", stBot.maxhp);
  const results = {};
  for (const [map, bx, by] of [["cave", 38, 12], ["ember", 28, 12], ["shrine", 22, 13]]) {
    // revive + heal bot before each round; bot skips dialogs instantly
    G.state = "play"; G.deadT = 0; G.dialog = null;
    { const s2 = pstats(); p.hp = s2.maxhp; p.mp = s2.maxmp; }
    loadMap(map, bx, by);
    const bb = G.enemies.find(e => e.boss);
    let dodgeT = 0;
    for (let i = 0; i < 60 * 120 && G.state === "play"; i++) {
      const a = Math.atan2(bb.y - p.y, bb.x - p.x);
      const dd = Math.hypot(bb.x - p.x, bb.y - p.y);
      p.faceX = Math.cos(a); p.faceY = Math.sin(a); p.dir = a;
      // kite: close to melee range, back off when boss is casting
      const want = (bb.state === "thorns" || bb.state === "slam" || bb.state === "leap" || bb.state === "meteors" || bb.state === "breath" || bb.state === "spiral") ? 150 : 46;
      if (dd > want) { p.x += Math.cos(a) * 2.9; p.y += Math.sin(a) * 2.9; }
      else if (dd < want - 30) { p.x -= Math.cos(a) * 2.2; p.y -= Math.sin(a) * 2.2; }
      // dodge incoming projectiles
      let danger = false;
      for (const pr of G.projectiles) if (pr.from === "enemy" && Math.hypot(pr.x - p.x, pr.y - p.y) < 55) danger = true;
      if (danger && dodgeT <= 0) { Input.pressedQueue.add("KeyK"); dodgeT = 40; }
      dodgeT--;
      if (p.hp < stBot.maxhp * 0.45) Input.pressedQueue.add("Digit3");
      if (i % 60 === 30) Input.pressedQueue.add("Digit1");
      Input.pressedQueue.add("KeyJ");
      gameTick(1/60);
      if (G.dialog) { G.dialog = null; } // bot skips story beats
      Input.endFrame();
      if (bb.hp <= 0) break;
    }
    results[map] = bb.hp <= 0 ? "DOWN" : "hp " + Math.round(bb.hp) + "/" + bb.maxhp + (G.state !== "play" ? " (bot fell)" : " timeout");
    console.log(map, "boss:", results[map], "| shards:", G.flags.shards || 0, "| flag:", !!bb.flag && !!G.flags[bb.flag]);
    if (G.state !== "play") { // revive bot for next round
      G.state = "play"; const s2 = pstats(); p.hp = s2.maxhp; p.mp = s2.maxmp;
    }
  }
})();
`;

try {
  vm.runInContext(files.join("\n;\n") + "\n" + test, sb, { filename: "balance.js", timeout: 300000 });
  console.log("BALANCE SIM DONE");
} catch (e) {
  console.error("CRASH:", e.stack || e.message);
  process.exit(2);
}
