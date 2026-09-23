"use strict";
/* ============================================================
   AETHERFALL — game.js
   world sim · entities · combat · boss AI · map baking
   ============================================================ */

const VIEW_W = 960, VIEW_H = 540;

/* ---------------- global state ---------------- */
const G = {
  state: "title",           // title | play | transition | dead | victory | ending
  paused: false, menuOpen: false, dialog: null, shop: null,
  t: 0, playT: 0,
  mapId: null, mapDef: null, mapGrid: null, mapBaked: null, mapMini: null,
  waterTiles: [], lavaTiles: [], glowTiles: [], petalTiles: [],
  player: null,
  enemies: [], projectiles: [], pickups: [], chests: [], npcs: [],
  flags: {},
  quests: { main: 0, pests: 0, pestsN: 0, petals: 0, petalsN: 0 },
  gold: 0, kills: 0,
  cam: new Camera(),
  trans: null, toast: null, banner: null,
  hitstop: 0, deadT: 0, endingT: 0, endingStep: 0,
  bossActive: null, portalLockToastT: 0,
  petalsCollectedTotal: 0,
  shardsGot: 0,
};

/* ---------------- player ---------------- */
function makePlayer() {
  return {
    x: 0, y: 0, vx: 0, vy: 0, r: 12,
    faceX: 0, faceY: 1, dir: Math.PI / 2,
    lvl: 1, xp: 0, hp: 60, mp: 30,
    inv: [{ id: "potion", n: 2 }],
    equip: { weapon: "trainee_sword", armor: "cloth_tunic", acc: null },
    state: "idle", atkT: 0, atkDur: 0, combo: 0, comboT: 0, hitDone: true,
    dashT: 0, dashCd: 0, dashX: 0, dashY: 0,
    hurtT: 0, invulnT: 0, walkT: 0, ghosts: [],
    cds: [0, 0, 0],
  };
}
function xpNeed(l) { return Math.round(45 * Math.pow(l, 1.5)); }

function pstats() {
  const p = G.player;
  const w = ITEMS[p.equip.weapon] || {}, a = ITEMS[p.equip.armor] || {}, c = p.equip.acc ? ITEMS[p.equip.acc] : {};
  const maxhp = 60 + (p.lvl - 1) * 8 + (a.hp || 0) + (c.hp || 0);
  const maxmp = 30 + (p.lvl - 1) * 4 + (a.mp || 0) + (c.mp || 0);
  return {
    atk: 10 + (p.lvl - 1) * 2 + (w.atk || 0) + (c.atk || 0),
    def: 4 + (p.lvl - 1) + (a.def || 0) + (c.def || 0),
    spd: 178 + (c.spd || 0),
    luck: 4 + (w.luck || 0) + (c.luck || 0),
    maxhp, maxmp,
  };
}

function gainXP(n) {
  const p = G.player;
  p.xp += n;
  let leveled = false;
  while (p.xp >= xpNeed(p.lvl) && p.lvl < 40) {
    p.xp -= xpNeed(p.lvl);
    p.lvl++;
    leveled = true;
  }
  if (leveled) {
    const st = pstats();
    p.hp = st.maxhp; p.mp = st.maxmp;
    AudioSys.sfx("level");
    FloatTexts.add(p.x, p.y - 30, "LEVEL UP!", "#ffe27a", 20, 1.4);
    Particles.ring(p.x, p.y, "#ffe27a", 26, 8, 260);
    Particles.burst(p.x, p.y, "#fff2b0", 20, 200, 0.8);
    G.cam.addShake(4);
  }
}

function addItem(id, n = 1) {
  const inv = G.player.inv;
  const it = inv.find(i => i.id === id);
  if (it) it.n += n; else inv.push({ id, n });
}
function countItem(id) {
  const it = G.player.inv.find(i => i.id === id);
  return it ? it.n : 0;
}
function removeItem(id, n = 1) {
  const inv = G.player.inv;
  const it = inv.find(i => i.id === id);
  if (!it) return false;
  it.n -= n;
  if (it.n <= 0) inv.splice(inv.indexOf(it), 1);
  return true;
}

/* ---------------- collision ---------------- */
function tileAt(tx, ty) {
  if (!G.mapGrid) return "#";
  if (ty < 0 || tx < 0 || ty >= G.mapGrid.length || tx >= G.mapGrid[0].length) return "#";
  return G.mapGrid[ty][tx];
}
function solidPx(px, py) { return isSolidChar(tileAt(Math.floor(px / TS), Math.floor(py / TS))); }

function collideMove(ent, dx, dy) {
  const hw = ent.r * 0.72, hh = ent.r * 0.6;
  if (dx !== 0) {
    ent.x += dx;
    const edge = dx > 0 ? ent.x + hw : ent.x - hw;
    if (solidPx(edge, ent.y - hh) || solidPx(edge, ent.y + hh) || solidPx(edge, ent.y)) {
      const t = Math.floor(edge / TS);
      ent.x = dx > 0 ? t * TS - hw - 0.01 : (t + 1) * TS + hw + 0.01;
    }
  }
  if (dy !== 0) {
    ent.y += dy;
    const edge = dy > 0 ? ent.y + hh : ent.y - hh;
    if (solidPx(ent.x - hw, edge) || solidPx(ent.x + hw, edge) || solidPx(ent.x, edge)) {
      const t = Math.floor(edge / TS);
      ent.y = dy > 0 ? t * TS - hh - 0.01 : (t + 1) * TS + hh + 0.01;
    }
  }
  ent.x = clamp(ent.x, hw, (G.mapGrid[0].length) * TS - hw);
  ent.y = clamp(ent.y, hh, (G.mapGrid.length) * TS - hh);
}

/* ---------------- map baking ---------------- */
function hash2(x, y, s) {
  let h = (x * 374761393 + y * 668265263 + s * 974634) | 0;
  h = (h ^ (h >> 13)) * 1274126177;
  return ((h ^ (h >> 16)) >>> 0) / 4294967296;
}

const THEME_PAL = {
  village: { ground: "#82c25e", ground2: "#79b957", road: "#d9b878", tree: "#3e7d3a", treeDark: "#2e5f2c", wall: "#b98a5e", roof: "#c0493b" },
  meadow:  { ground: "#8bcb63", ground2: "#81c25b", road: "#dcbd7c", tree: "#478a40", treeDark: "#356b30", wall: "#b98a5e", roof: "#c0493b" },
  forest:  { ground: "#5d9a52", ground2: "#54914b", road: "#c2a468", tree: "#2f6b34", treeDark: "#20512a", wall: "#2f6b34", roof: "#c0493b" },
  cave:    { ground: "#4d4761", ground2: "#464059", road: "#5d566f", tree: "#221d33", treeDark: "#1a1628", wall: "#2a2440", roof: "#332c4a" },
  ember:   { ground: "#6e5a4e", ground2: "#665347", road: "#8a6f52", tree: "#332722", treeDark: "#241a16", wall: "#3a2b25", roof: "#4a342c" },
  shrine:  { ground: "#5a5378", ground2: "#534c70", road: "#7d729c", tree: "#2c2547", treeDark: "#211b38", wall: "#332b52", roof: "#3d3462" },
};

function bakeMap(def, grid) {
  const w = def.w, h = def.h, pal = THEME_PAL[def.theme] || THEME_PAL.meadow;
  const cv = document.createElement("canvas");
  cv.width = w * TS; cv.height = h * TS;
  const c = cv.getContext("2d");
  const water = [], lava = [], glow = [];

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const ch = grid[y][x];
      const r1 = hash2(x, y, def.seed), r2 = hash2(x, y, def.seed + 7);
      const px = x * TS, py = y * TS;
      // base ground always
      c.fillStyle = (x + y) % 2 === 0 ? pal.ground : pal.ground2;
      c.fillRect(px, py, TS, TS);
      if (ch === "," || ch === "f" || ch === "*") {
        c.fillStyle = "rgba(0,60,0,0.10)";
        for (let i = 0; i < 3; i++) {
          const gx = px + 4 + hash2(x * 3 + i, y, 5) * 24, gy = py + 4 + hash2(x, y * 3 + i, 9) * 24;
          c.fillRect(gx, gy, 2, 5);
        }
      }
      switch (ch) {
        case ".": case ",": case "f": case "*": case "s": case "S": case "R": case "b":
          if (ch === "R") { c.fillStyle = pal.road; c.fillRect(px, py, TS, TS); c.fillStyle = "rgba(0,0,0,0.08)"; c.fillRect(px, py + TS - 4, TS, 4); }
          if (ch === "S") { c.fillStyle = "rgba(255,255,255,0.03)"; c.fillRect(px, py, TS, 1); c.fillStyle = "rgba(0,0,0,0.10)"; c.fillRect(px + TS - 1, py, 1, TS); c.fillRect(px, py + TS - 1, TS, 1); }
          if (ch === "s") { c.fillStyle = "rgba(0,0,0,0.06)"; if (r1 > 0.6) c.fillRect(px + r2 * 22, py + r1 * 22, 5, 3); }
          if (ch === "b") { c.fillStyle = "#6b4a2f"; c.fillRect(px, py, TS, TS); c.fillStyle = "rgba(0,0,0,0.25)"; for (let i = 0; i < 4; i++) c.fillRect(px, py + i * 8, TS, 1); }
          if (ch === "f") {
            const cols = ["#ff8ab8", "#ffd24c", "#c9a6ff", "#ff9a6b"];
            for (let i = 0; i < 3; i++) {
              c.fillStyle = cols[Math.floor(hash2(x + i, y - i, 3) * cols.length)];
              c.fillRect(px + 4 + hash2(x, y + i, 11) * 22, py + 4 + hash2(x - i, y, 13) * 22, 4, 4);
            }
          }
          if (ch === "*") {
            c.fillStyle = "#bfe6ff"; c.beginPath(); c.arc(px + 16, py + 16, 5, 0, TAU); c.fill();
            c.fillStyle = "#7ab8e8"; c.fillRect(px + 15, py + 20, 2, 8);
          }
          break;
        case "W":
          c.fillStyle = "#3f7fd4"; c.fillRect(px, py, TS, TS);
          c.fillStyle = "#366fb9"; c.fillRect(px, py + 16, TS, 16);
          water.push([x, y]);
          break;
        case "~":
          c.fillStyle = "#e8541f"; c.fillRect(px, py, TS, TS);
          c.fillStyle = "#ffb02e"; c.fillRect(px + r1 * 20, py + r2 * 20, 8, 8);
          lava.push([x, y]);
          break;
        case "#": {
          if (def.theme === "village" || def.theme === "meadow" || def.theme === "forest") {
            // tree
            c.fillStyle = pal.treeDark; c.fillRect(px + 12, py + 16, 8, 16);
            c.fillStyle = pal.tree;
            c.beginPath(); c.arc(px + 16, py + 12, 13 + r1 * 3, 0, TAU); c.fill();
            c.fillStyle = "rgba(255,255,255,0.12)";
            c.beginPath(); c.arc(px + 12, py + 8, 5, 0, TAU); c.fill();
          } else {
            c.fillStyle = pal.wall; c.fillRect(px, py, TS, TS);
            c.fillStyle = "rgba(255,255,255,0.06)"; c.fillRect(px, py, TS, 4);
            c.fillStyle = "rgba(0,0,0,0.25)"; c.fillRect(px + r1 * 20, py + 8 + r2 * 14, 10, 7);
          }
          break;
        }
        case "h": c.fillStyle = pal.wall; c.fillRect(px, py, TS, TS); c.fillStyle = "rgba(0,0,0,0.2)"; c.fillRect(px + 4, py + 8, 10, 12); break;
        case "r": c.fillStyle = pal.roof; c.fillRect(px, py, TS, TS); c.fillStyle = "rgba(0,0,0,0.15)"; c.fillRect(px, py + TS - 6, TS, 6); break;
        case "d": c.fillStyle = pal.wall; c.fillRect(px, py, TS, TS); c.fillStyle = "#5b3a24"; c.fillRect(px + 6, py + 4, 20, 28); c.fillStyle = "#ffd24c"; c.fillRect(px + 20, py + 16, 3, 3); break;
        case "F": c.fillStyle = "#9a7748"; c.fillRect(px + 2, py + 10, 28, 5); c.fillRect(px + 2, py + 20, 28, 5); c.fillRect(px + 4, py + 6, 4, 24); c.fillRect(px + 24, py + 6, 4, 24); break;
        case "O": {
          const isOrigin = tileAt(x - 1, y) !== "O" && tileAt(x, y - 1) !== "O";
          c.fillStyle = "#9aa3b8";
          c.fillRect(px, py, TS, TS);
          if (isOrigin) {
            c.fillStyle = "#7d879e";
            c.fillRect(px - 3, py - 3, 70, 70);
            c.fillStyle = "#5a8fd4";
            c.beginPath(); c.arc(px + 32, py + 32, 24, 0, TAU); c.fill();
            c.fillStyle = "#7fb2e8";
            c.beginPath(); c.arc(px + 32, py + 32, 17, 0, TAU); c.fill();
            c.fillStyle = "#bfe6ff";
            c.beginPath(); c.arc(px + 26, py + 26, 5, 0, TAU); c.fill();
          }
          break;
        }
        case "c":
          c.fillStyle = "#2a2440"; c.fillRect(px, py, TS, TS);
          c.fillStyle = "#7ee0ff";
          c.beginPath(); c.moveTo(px + 16, py + 3); c.lineTo(px + 26, py + 26); c.lineTo(px + 6, py + 26); c.closePath(); c.fill();
          glow.push([x, y, "#7ee0ff"]);
          break;
        case "T":
          if (def.theme === "shrine") {
            c.fillStyle = "#8f86ad"; c.fillRect(px + 8, py + 2, 16, 28); c.fillStyle = "rgba(0,0,0,0.25)"; c.fillRect(px + 8, py + 24, 16, 6);
          } else {
            c.fillStyle = "#6b4a2f"; c.fillRect(px + 13, py + 10, 6, 20);
            glow.push([x, y, "#ffb02e"]);
          }
          break;
      }
    }
  }
  return { canvas: cv, water, lava, glow };
}

function buildMini(def, grid) {
  const cv = document.createElement("canvas");
  cv.width = def.w * 2; cv.height = def.h * 2;
  const c = cv.getContext("2d");
  for (let y = 0; y < def.h; y++) for (let x = 0; x < def.w; x++) {
    const ch = grid[y][x];
    let col = "#20304a";
    if (".,".includes(ch) || ch === "f" || ch === "*") col = def.theme === "forest" ? "#2f6b34" : def.theme === "ember" ? "#5a4638" : "#4a7a3a";
    if (ch === "s") col = "#5a4638";
    if (ch === "S") col = "#565070";
    if (ch === "R" || ch === "b") col = "#8a7a4f";
    if (ch === "W") col = "#2f5fae";
    if (ch === "~") col = "#c04a1a";
    if (ch === "#" || "hrdFcTO".includes(ch)) col = "#12141f";
    c.fillStyle = col; c.fillRect(x * 2, y * 2, 2, 2);
  }
  return cv;
}

/* ---------------- map lifecycle ---------------- */
function loadMap(id, entryTX, entryTY) {
  const def = MAPS[id];
  const parsed = parseMapDef(def);
  G.mapId = id; G.mapDef = def; G.mapGrid = parsed.grid;
  const baked = bakeMap(def, parsed.grid);
  G.mapBaked = baked.canvas;
  G.waterTiles = baked.water; G.lavaTiles = baked.lava; G.glowTiles = baked.glow;
  G.mapMini = buildMini(def, parsed.grid);

  G.enemies = []; G.projectiles = []; G.pickups = []; G.chests = []; G.npcs = []; G.bossActive = null;
  Particles.list.length = 0; FloatTexts.list.length = 0;

  // petals
  G.petalTiles = [];
  for (let y = 0; y < def.h; y++) for (let x = 0; x < def.w; x++)
    if (parsed.grid[y][x] === "*") G.petalTiles.push({ x, y });

  for (const s of def.spawns) G.enemies.push(makeEnemy(s.e, s.x, s.y));
  if (def.boss && !G.flags[def.boss.flag]) {
    const b = makeBoss(def.boss.id, def.boss.x, def.boss.y, def.boss.flag);
    G.enemies.push(b);
  }
  for (const c of def.chests) if (!G.flags["chest_" + c.id]) G.chests.push({ ...c, opened: false, t: 0 });
  for (const n of def.npcs) G.npcs.push({ ...n, px: (n.x + 0.5) * TS, py: (n.y + 0.5) * TS, t: rand(9) });

  const p = G.player;
  p.x = (entryTX + 0.5) * TS; p.y = (entryTY + 0.5) * TS;
  p.vx = p.vy = 0; p.atkT = 0; p.dashT = 0;
  G.cam.snap(p.x, p.y, def.w * TS, def.h * TS, VIEW_W, VIEW_H);
  AudioSys.setTheme(def.music);
  toast(def.name.toUpperCase(), 2.4);
  saveGame();
}

function toast(txt, dur = 2) { G.toast = { txt, t: dur, dur }; }
function banner(txt, sub, dur = 2.6) { G.banner = { txt, sub, t: dur, dur }; }

/* ---------------- entities ---------------- */
function makeEnemy(kind, tx, ty) {
  const d = ENEMIES[kind];
  return {
    kind, def: d, boss: false, dormant: false,
    x: (tx + 0.5) * TS, y: (ty + 0.5) * TS, sx: (tx + 0.5) * TS, sy: (ty + 0.5) * TS,
    hp: d.hp, maxhp: d.hp, r: d.r,
    t: rand(10), hitT: 0, atkCd: 0, state: "idle", stateT: rand(2),
    dir: rand(TAU), dashX: 0, dashY: 0, seenPlayer: false,
  };
}

function makeBoss(kind, tx, ty, flag) {
  const d = BOSSES[kind];
  return {
    kind, def: d, boss: true, flag, dormant: true,
    x: (tx + 0.5) * TS, y: (ty + 0.5) * TS, sx: (tx + 0.5) * TS, sy: (ty + 0.5) * TS,
    hp: d.hp, maxhp: d.hp, r: d.r,
    t: 0, hitT: 0, atkCd: 0, state: "idle", stateT: 1.4, phase: 1,
    dir: Math.PI / 2, dashX: 0, dashY: 0, seenPlayer: false,
    extra: { spiralA: 0, tele: 0, px: 0, py: 0, notes: [] },
  };
}

/* ---------------- projectiles & pickups ---------------- */
function fireProjectile(opts) {
  G.projectiles.push(Object.assign({
    x: 0, y: 0, vx: 0, vy: 0, r: 5, dmg: 10, from: "enemy", kind: "shadow",
    life: 3, col: "#b06bff", trail: 0,
  }, opts));
}
function spawnPickup(x, y, type, val = 0, itemId = null) {
  G.pickups.push({ x, y, type, val, itemId, t: rand(9), vy: 0 });
}

/* ---------------- damage ---------------- */
function hitEnemy(e, dmg, kx, ky, crit) {
  e.hp -= dmg;
  e.hitT = 0.16;
  if (!e.boss) { e.x += kx; e.y += ky; }
  FloatTexts.add(e.x + rand(-6, 6), e.y - e.r - 8, String(dmg), crit ? "#ffd24c" : "#ffffff", crit ? 19 : 15);
  Particles.burst(e.x, e.y, e.boss ? "#ffd24c" : "#ffb0a0", crit ? 12 : 6, 160, 0.4);
  G.hitstop = Math.max(G.hitstop, crit ? 0.075 : 0.042);
  G.cam.addShake(crit ? 6 : 3);
  AudioSys.sfx(crit ? "crit" : "hit");
  if (e.hp <= 0) killEnemy(e);
}

function killEnemy(e) {
  const idx = G.enemies.indexOf(e);
  if (idx >= 0) G.enemies.splice(idx, 1);
  G.kills++;
  Particles.burst(e.x, e.y, e.def.col || "#fff", 16, 220, 0.6);
  Particles.ring(e.x, e.y, "#ffffff", 12, 4, 180);

  if (e.boss) { bossDefeated(e); return; }

  const d = e.def;
  gainXP(d.xp);
  const gold = irand(d.gold[0], d.gold[1]);
  G.gold += gold;
  FloatTexts.add(e.x, e.y - 20, `+${gold}G`, "#ffd24c", 13);
  AudioSys.sfx("gold");
  if (chance(0.16)) spawnPickup(e.x, e.y, "heart");
  if (chance(0.07)) spawnPickup(e.x + rand(-8, 8), e.y, "item", 0, "potion");

  if (e.kind === "slime" && G.quests.pests === 1) {
    G.quests.pestsN++;
    if (G.quests.pestsN === 6) toast("Pests culled! Report to Petra.");
  }
}

function bossDefeated(e) {
  const d = e.def;
  gainXP(d.xp);
  const gold = irand(d.gold[0], d.gold[1]);
  G.gold += gold;
  FloatTexts.add(e.x, e.y - 30, `+${gold}G`, "#ffd24c", 16);
  Particles.burst(e.x, e.y, "#ffd24c", 40, 320, 1.1, 4);
  Particles.ring(e.x, e.y, "#fff", 30, 8, 320);
  G.cam.addShake(12); G.hitstop = Math.max(G.hitstop, 0.35);
  AudioSys.sfx("roar");
  spawnPickup(e.x, e.y, "heart");
  spawnPickup(e.x - 16, e.y, "heart");
  G.flags[e.flag] = true;
  G.bossActive = null;

  if (e.kind === "serpent") {
    G.flags.shards = 3;
    banner("THE TWILIGHT IS BROKEN", "", 3.4);
    startEnding();
    return;
  }
  // grant shard directly (never lose progress by leaving before pickup)
  addItem("shard_dawn");
  G.flags.shards = (G.flags.shards || 0) + 1;
  AudioSys.sfx("shard");
  Particles.ring(G.player.x, G.player.y, "#ffe27a", 30, 10, 300);
  banner("GUARDIAN FELLED", `${d.name} is freed from the shard's madness`, 3);
  AudioSys.setTheme(G.mapDef.music);
  const nShards = G.flags.shards;
  dialogPush([["", `The Dawn Shard hums in your palm... (${nShards}/3)`]], () => {
    if (nShards >= 3) {
      G.quests.main = 2;
      toast("The seal on the SHRINE OF DAWN is broken!");
      AudioSys.sfx("level");
    }
  });
  saveGame();
}

function damagePlayer(raw, srcX, srcY) {
  const p = G.player;
  if (p.invulnT > 0 || p.dashT > 0 || G.state !== "play") return;
  const st = pstats();
  const dmg = Math.max(1, Math.round(raw - st.def * 0.55));
  p.hp -= dmg;
  p.invulnT = 0.95; p.hurtT = 0.35;
  const a = angleTo(srcX === undefined ? p.x + 1 : srcX, srcY === undefined ? p.y + 1 : srcY, p.x, p.y);
  collideMove(p, Math.cos(a) * 22, Math.sin(a) * 22);
  FloatTexts.add(p.x, p.y - 26, "-" + dmg, "#ff6b6b", 16);
  Particles.burst(p.x, p.y, "#ff6b6b", 10, 180, 0.5);
  G.cam.addShake(6); G.hitstop = Math.max(G.hitstop, 0.05);
  AudioSys.sfx("hurt");
  if (p.hp <= 0) {
    p.hp = 0;
    G.state = "dead"; G.deadT = 0;
    AudioSys.sfx("death");
    AudioSys.setTheme("title");
  }
}

/* ---------------- player update ---------------- */
function updatePlayer(dt) {
  const p = G.player;
  const st = pstats();
  p.walkT += dt;
  p.invulnT = Math.max(0, p.invulnT - dt);
  p.hurtT = Math.max(0, p.hurtT - dt);
  p.dashCd = Math.max(0, p.dashCd - dt);
  p.comboT = Math.max(0, p.comboT - dt);
  for (let i = 0; i < 3; i++) p.cds[i] = Math.max(0, p.cds[i] - dt);
  p.mp = Math.min(st.maxmp, p.mp + 2.2 * dt);
  p.hp = Math.min(st.maxhp, p.hp + 0.6 * dt);

  const mv = Input.axis();

  // dash
  if (Input.consume("dodge") && p.dashCd <= 0 && p.dashT <= 0) {
    const dx = mv.x || p.faceX, dy = mv.y || p.faceY;
    const m = Math.hypot(dx, dy) || 1;
    p.dashX = dx / m; p.dashY = dy / m;
    p.dashT = 0.26; p.dashCd = 0.62;
    AudioSys.sfx("dash");
    Particles.burst(p.x, p.y, "#bfe6ff", 8, 90, 0.35);
  }
  if (p.dashT > 0) {
    p.dashT -= dt;
    collideMove(p, p.dashX * st.spd * 3.1 * dt, p.dashY * st.spd * 3.1 * dt);
    p.ghosts.push({ x: p.x, y: p.y, t: 0.28 });
    if (p.ghosts.length > 8) p.ghosts.shift();
  } else {
    // walk
    if (p.atkT <= 0) {
      const spd = st.spd;
      if (mv.x !== 0 || mv.y !== 0) {
        collideMove(p, mv.x * spd * dt, mv.y * spd * dt);
        p.faceX = mv.x; p.faceY = mv.y;
        const m2 = Math.hypot(mv.x, mv.y);
        p.dir = Math.atan2(mv.y / m2, mv.x / m2);
        p.state = "move";
      } else p.state = "idle";
    } else p.state = "attack";
  }
  p.dir = Math.atan2(p.faceY, p.faceX || 0.0001);

  // attack
  if (Input.consume("attack") && p.atkT <= 0 && p.dashT <= 0) {
    p.combo = p.comboT > 0 ? (p.combo + 1) % 3 : 0;
    p.atkDur = p.combo === 2 ? 0.36 : 0.26;
    p.atkT = p.atkDur;
    p.hitDone = false;
    p.comboT = 0.85;
    AudioSys.sfx("sword");
    // soft aim-assist: snap facing to nearest foe within 130px
    let best = null, bd = 130;
    for (const e of G.enemies) {
      if (e.hp <= 0 || e.dormant) continue;
      const dd = dist(p.x, p.y, e.x, e.y);
      if (dd < bd) { bd = dd; best = e; }
    }
    if (best) {
      const a = angleTo(p.x, p.y, best.x, best.y);
      p.faceX = Math.cos(a); p.faceY = Math.sin(a);
    }
    p.atkDir = Math.atan2(p.faceY, p.faceX || 0.0001);
    p.dir = p.atkDir;
    collideMove(p, p.faceX * 10, p.faceY * 10);
  }
  if (p.atkT > 0) {
    p.atkT -= dt;
    const progress = 1 - p.atkT / p.atkDur;
    if (!p.hitDone && progress > 0.28) {
      p.hitDone = true;
      doPlayerMelee(st);
    }
  }

  // skills
  if (Input.consume("sk1") && p.cds[0] <= 0 && p.mp >= 8) skillNova(p, st);
  if (Input.consume("sk2") && p.cds[1] <= 0 && p.mp >= 5) skillLance(p, st);
  if (Input.consume("sk3") && p.cds[2] <= 0 && p.mp >= 14) skillHeal(p, st);

  // ghosts decay
  for (let i = p.ghosts.length - 1; i >= 0; i--) {
    p.ghosts[i].t -= dt;
    if (p.ghosts[i].t <= 0) p.ghosts.splice(i, 1);
  }
}

function doPlayerMelee(st) {
  const p = G.player;
  const reach = 46, arc = 1.25;
  const aimA = p.atkDir !== undefined ? p.atkDir : p.dir;
  let hitAny = false;
  for (const e of [...G.enemies]) {
    if (e.hp <= 0 || e.dormant) continue;
    const d = dist(p.x, p.y, e.x, e.y);
    if (d > reach + e.r) continue;
    let da = Math.atan2(e.y - p.y, e.x - p.x) - aimA;
    while (da > Math.PI) da -= TAU;
    while (da < -Math.PI) da += TAU;
    if (d > 30 && Math.abs(da) > arc) continue;
    const mult = [1.0, 1.0, 1.5][p.combo];
    let dmg = Math.max(1, Math.round(st.atk * mult * rand(0.92, 1.1) - e.def.def * 0.6));
    const crit = chance(0.05 + st.luck * 0.008);
    if (crit) dmg = Math.round(dmg * 1.7);
    hitEnemy(e, dmg, Math.cos(p.dir) * (e.boss ? 0 : 12), Math.sin(p.dir) * (e.boss ? 0 : 12), crit);
    hitAny = true;
  }
  if (hitAny && p.combo === 2) G.cam.addShake(5);
}

function skillNova(p, st) {
  p.cds[0] = 5; p.mp -= 8;
  AudioSys.sfx("nova");
  Particles.ring(p.x, p.y, "#ffe27a", 34, 10, 380);
  Particles.burst(p.x, p.y, "#fff2b0", 26, 260, 0.6);
  G.cam.addShake(8); G.hitstop = Math.max(G.hitstop, 0.05);
  for (const e of [...G.enemies]) {
    if (e.hp <= 0) continue;
    if (dist(p.x, p.y, e.x, e.y) <= 100 + e.r) {
      const dmg = Math.max(1, Math.round(st.atk * 1.35 * rand(0.92, 1.1) - e.def.def * 0.5));
      hitEnemy(e, dmg, 0, 0, false);
      if (!e.boss) {
        const a = angleTo(p.x, p.y, e.x, e.y);
        e.x += Math.cos(a) * 26; e.y += Math.sin(a) * 26;
      }
    }
  }
}
function skillLance(p, st) {
  p.cds[1] = 2.2; p.mp -= 5;
  AudioSys.sfx("lance");
  fireProjectile({
    x: p.x + p.faceX * 18, y: p.y + p.faceY * 18,
    vx: p.faceX * 470, vy: p.faceY * 470,
    dmg: Math.round(st.atk * 1.6), from: "player", kind: "lance", r: 6, col: "#ffe27a", life: 1.1,
  });
}
function skillHeal(p, st) {
  p.cds[2] = 9; p.mp -= 14;
  const amt = Math.round(st.maxhp * 0.35);
  p.hp = Math.min(st.maxhp, p.hp + amt);
  AudioSys.sfx("heal");
  FloatTexts.add(p.x, p.y - 30, "+" + amt, "#8aff8a", 17);
  Particles.burst(p.x, p.y, "#8aff8a", 18, 130, 0.7);
}

/* ---------------- enemy AI ---------------- */
function updateEnemy(e, dt) {
  const p = G.player;
  e.t += dt;
  e.hitT = Math.max(0, e.hitT - dt);
  e.atkCd = Math.max(0, e.atkCd - dt);
  const d = dist(e.x, e.y, p.x, p.y);

  if (e.boss) { updateBoss(e, dt, d); contactDamage(e); return; }
  if (e.dormant) return;

  const def = e.def;
  const aToP = angleTo(e.x, e.y, p.x, p.y);
  e.seenPlayer = d < def.sight || e.seenPlayer && d < def.sight * 1.6;

  switch (def.ai) {
    case "chase": {
      if (e.seenPlayer) {
        e.dir = aToP;
        collideMove(e, Math.cos(aToP) * def.spd * dt, Math.sin(aToP) * def.spd * dt);
      } else {
        e.stateT -= dt;
        if (e.stateT <= 0) { e.stateT = rand(1, 3); e.dir = rand(TAU); e.state = chance(0.5) ? "walk" : "idle"; }
        if (e.state === "walk") collideMove(e, Math.cos(e.dir) * def.spd * 0.35 * dt, Math.sin(e.dir) * def.spd * 0.35 * dt);
      }
      break;
    }
    case "flutter": {
      const wob = Math.sin(e.t * 5) * 40;
      if (e.seenPlayer) {
        collideMove(e, Math.cos(aToP) * def.spd * dt + Math.cos(aToP + Math.PI / 2) * wob * dt, Math.sin(aToP) * def.spd * dt + Math.sin(aToP + Math.PI / 2) * wob * dt);
      } else {
        e.stateT -= dt;
        if (e.stateT <= 0) { e.stateT = rand(0.8, 2); e.dir = rand(TAU); }
        collideMove(e, Math.cos(e.dir) * def.spd * 0.4 * dt, Math.sin(e.dir) * def.spd * 0.4 * dt);
      }
      break;
    }
    case "charger": {
      if (e.state === "windup") {
        e.stateT -= dt;
        e.dir = aToP;
        if (e.stateT <= 0) { e.state = "charge"; e.stateT = 0.42; e.dashX = Math.cos(aToP); e.dashY = Math.sin(aToP); AudioSys.sfx("dash"); }
      } else if (e.state === "charge") {
        e.stateT -= dt;
        collideMove(e, e.dashX * def.spd * 3.4 * dt, e.dashY * def.spd * 3.4 * dt);
        if (chance(0.5)) Particles.spawn({ x: e.x, y: e.y + 6, vx: rand(-20, 20), vy: rand(-10, 20), life: 0.3, col: "#cfd8e8", size: 2 });
        if (e.stateT <= 0) { e.state = "idle"; e.stateT = rand(1.2, 2); }
      } else if (e.seenPlayer && d < def.sight && e.atkCd <= 0 && d > 60) {
        e.state = "windup"; e.stateT = 0.5; e.atkCd = 2.6;
        Particles.burst(e.x, e.y, "#ff6b6b", 4, 60, 0.3);
      } else if (e.seenPlayer) {
        collideMove(e, Math.cos(aToP) * def.spd * 0.6 * dt, Math.sin(aToP) * def.spd * 0.6 * dt);
      } else {
        e.stateT -= dt;
        if (e.stateT <= 0) { e.stateT = rand(1, 3); e.dir = rand(TAU); }
        collideMove(e, Math.cos(e.dir) * def.spd * 0.3 * dt, Math.sin(e.dir) * def.spd * 0.3 * dt);
      }
      break;
    }
    case "ranged": {
      if (e.seenPlayer) {
        const band = 190;
        e.dir = aToP;
        if (d > band + 40) collideMove(e, Math.cos(aToP) * def.spd * dt, Math.sin(aToP) * def.spd * dt);
        else if (d < band - 60) collideMove(e, -Math.cos(aToP) * def.spd * dt, -Math.sin(aToP) * def.spd * dt);
        else collideMove(e, Math.cos(aToP + Math.PI / 2) * def.spd * 0.6 * dt * Math.sin(e.t * 1.3), Math.sin(aToP + Math.PI / 2) * def.spd * 0.6 * dt * Math.sin(e.t * 1.3));
        if (e.atkCd <= 0 && d < def.sight) {
          e.atkCd = rand(1.9, 2.6);
          const kind = def.proj || "shadow";
          const spd = kind === "arrow" ? 330 : 220;
          fireProjectile({
            x: e.x, y: e.y - 6, vx: Math.cos(aToP) * spd, vy: Math.sin(aToP) * spd,
            dmg: def.dmg, from: "enemy", kind,
            col: kind === "arrow" ? "#d9b878" : kind === "fire" ? "#ff8a4c" : "#b06bff",
            r: kind === "arrow" ? 4 : 6, life: 2.6,
          });
          AudioSys.sfx(kind === "fire" ? "fire" : "lance");
        }
      } else {
        e.stateT -= dt;
        if (e.stateT <= 0) { e.stateT = rand(1, 3); e.dir = rand(TAU); }
        collideMove(e, Math.cos(e.dir) * def.spd * 0.3 * dt, Math.sin(e.dir) * def.spd * 0.3 * dt);
      }
      break;
    }
  }
  contactDamage(e);
}

function contactDamage(e) {
  const p = G.player;
  const dmg = e.boss ? e.def.atk : e.def.dmg;
  if (e.atkCd > 0 && e.boss) return;
  if (dist(e.x, e.y, p.x, p.y) < e.r + p.r - 2) {
    if (e.boss) {
      if (e.atkCd <= 0) { e.atkCd = 0.8; damagePlayer(dmg, e.x, e.y); }
    } else if (e.t - (e.lastHitT || -9) > 0.85) {
      e.lastHitT = e.t;
      damagePlayer(dmg, e.x, e.y);
    }
  }
}

/* ---------------- boss AI ---------------- */
function aimedVolley(b, n, spread, spd, kind, col, dmg) {
  const p = G.player;
  const base = angleTo(b.x, b.y, p.x, p.y);
  for (let i = 0; i < n; i++) {
    const a = base + (i - (n - 1) / 2) * spread;
    fireProjectile({ x: b.x, y: b.y, vx: Math.cos(a) * spd, vy: Math.sin(a) * spd, dmg, from: "enemy", kind, col, r: 6, life: 3.4 });
  }
}
function ringVolley(b, n, spd, kind, col, dmg, off = 0) {
  for (let i = 0; i < n; i++) {
    const a = off + (i / n) * TAU;
    fireProjectile({ x: b.x, y: b.y, vx: Math.cos(a) * spd, vy: Math.sin(a) * spd, dmg, from: "enemy", kind, col, r: 6, life: 3.4 });
  }
}
function groundBlast(x, y, radius, dmg, col) {
  Particles.ring(x, y, col, 24, radius * 0.3, 300);
  Particles.burst(x, y, col, 20, 240, 0.6);
  G.cam.addShake(7);
  AudioSys.sfx("nova");
  const p = G.player;
  if (dist(x, y, p.x, p.y) <= radius) damagePlayer(dmg, x, y);
}

function updateBoss(b, dt, d) {
  const p = G.player;
  b.stateT -= dt;

  // aggro
  if (b.dormant) {
    if (d < 230 && G.state === "play") {
      b.dormant = false;
      G.bossActive = b;
      banner(b.def.name.toUpperCase(), b.def.title, 3);
      AudioSys.sfx("boss");
      AudioSys.setTheme(b.kind === "serpent" ? "final" : "boss");
      b.state = "idle"; b.stateT = 1.2;
    }
    return;
  }

  const frac = b.hp / b.maxhp;
  const ph2 = frac < 0.62, ph3 = frac < 0.32;
  b.phase = ph3 ? 3 : ph2 ? 2 : 1;
  const aToP = angleTo(b.x, b.y, p.x, p.y);

  switch (b.kind) {
    case "guardian": {
      if (b.state === "idle") {
        b.dir = aToP;
        if (b.stateT <= 0) {
          const roll = rand();
          if (roll < 0.4) { b.state = "thorns"; b.stateT = 0.7; b.extra.tele = 1; }
          else if (roll < 0.72) { b.state = "slam"; b.stateT = 0.85; b.extra.px = p.x; b.extra.py = p.y; }
          else { b.state = "summon"; b.stateT = 0.6; }
        }
      } else if (b.state === "thorns") {
        if (b.stateT <= 0) {
          b.state = "idle"; b.stateT = ph3 ? 0.7 : 1.3;
          const n = ph2 ? 14 : 10;
          ringVolley(b, n, 180 + b.phase * 20, "thorn", "#8aff6b", b.def.atk * 0.8, rand(TAU));
          if (ph3) setTimeout(() => { if (b.hp > 0) ringVolley(b, n, 240, "thorn", "#8aff6b", b.def.atk * 0.8, rand(TAU)); }, 350);
          AudioSys.sfx("fire"); G.cam.addShake(4);
        }
      } else if (b.state === "slam") {
        if (b.stateT <= 0) {
          b.state = "idle"; b.stateT = 1.2;
          groundBlast(b.extra.px, b.extra.py, 78, b.def.atk, "#8aff6b");
        }
      } else if (b.state === "summon") {
        if (b.stateT <= 0) {
          b.state = "idle"; b.stateT = 1.6;
          const alive = G.enemies.filter(e => !e.boss).length;
          if (alive < 5) {
            for (let i = 0; i < 3; i++) {
              const a = rand(TAU);
              const s = makeEnemy("slime", 0, 0);
              s.x = b.x + Math.cos(a) * 60; s.y = b.y + Math.sin(a) * 60;
              s.seenPlayer = true;
              G.enemies.push(s);
              Particles.burst(s.x, s.y, "#6fd44c", 8, 120, 0.5);
            }
            AudioSys.sfx("roar");
          }
        }
      }
      break;
    }
    case "wraith": {
      if (b.state === "idle") {
        // drift toward player
        if (d > 150) collideMove(b, Math.cos(aToP) * 90 * dt, Math.sin(aToP) * 90 * dt);
        if (b.stateT <= 0) {
          const roll = rand();
          if (roll < 0.34) { b.state = "teleport"; b.stateT = 0.45; Particles.burst(b.x, b.y, "#b06bff", 14, 160, 0.5); }
          else if (roll < 0.66) { b.state = "volley"; b.stateT = 0.55; }
          else { b.state = "spiral"; b.stateT = 1.5; b.extra.spiralA = rand(TAU); }
        }
      } else if (b.state === "teleport") {
        if (b.stateT <= 0) {
          const a = rand(TAU);
          b.x = clamp(p.x + Math.cos(a) * 150, 48, G.mapDef.w * TS - 48);
          b.y = clamp(p.y + Math.sin(a) * 150, 48, G.mapDef.h * TS - 48);
          Particles.burst(b.x, b.y, "#b06bff", 14, 160, 0.5);
          b.state = "idle"; b.stateT = ph3 ? 0.5 : 0.9;
        }
      } else if (b.state === "volley") {
        if (b.stateT <= 0) {
          aimedVolley(b, ph2 ? 5 : 3, 0.22, 260, "shadow", "#b06bff", b.def.atk * 0.85);
          AudioSys.sfx("lance");
          b.state = "idle"; b.stateT = ph3 ? 0.6 : 1.1;
        }
      } else if (b.state === "spiral") {
        b.extra.spiralA += dt * (ph3 ? 9 : 6);
        b.extra.tele += dt;
        if (b.extra.tele > 0.11) {
          b.extra.tele = 0;
          const arms = ph2 ? 2 : 1;
          for (let k = 0; k < arms; k++)
            fireProjectile({ x: b.x, y: b.y, vx: Math.cos(b.extra.spiralA + k * Math.PI) * 190, vy: Math.sin(b.extra.spiralA + k * Math.PI) * 190, dmg: b.def.atk * 0.7, from: "enemy", kind: "shadow", col: "#b06bff", r: 6, life: 3 });
        }
        if (b.stateT <= 0) { b.state = "idle"; b.stateT = 1; }
      }
      break;
    }
    case "tyrant": {
      if (b.state === "idle") {
        b.dir = aToP;
        if (d > 130) collideMove(b, Math.cos(aToP) * 74 * dt, Math.sin(aToP) * 74 * dt);
        if (b.stateT <= 0) {
          const roll = rand();
          if (roll < 0.36) {
            b.state = "meteors"; b.stateT = 1.2; b.extra.notes = [];
            for (let i = 0; i < (ph2 ? 7 : 5); i++)
              b.extra.notes.push({ x: p.x + rand(-130, 130), y: p.y + rand(-130, 130), t: 0.5 + i * 0.16, hit: false });
            AudioSys.sfx("roar");
          } else if (roll < 0.66) { b.state = "leap"; b.stateT = 0.7; b.extra.px = p.x; b.extra.py = p.y; }
          else { b.state = "breath"; b.stateT = 1.1; b.extra.tele = 0; }
        }
      } else if (b.state === "meteors") {
        for (const n of b.extra.notes) {
          if (!n.hit) {
            n.t -= dt;
            if (n.t <= 0) { n.hit = true; groundBlast(n.x, n.y, 55, b.def.atk * 0.9, "#ff8a4c"); }
          }
        }
        if (b.stateT <= 0 && b.extra.notes.every(n => n.hit)) { b.state = "idle"; b.stateT = ph3 ? 0.8 : 1.4; }
      } else if (b.state === "leap") {
        if (b.stateT <= 0) {
          b.x = b.extra.px; b.y = b.extra.py;
          groundBlast(b.x, b.y, 85, b.def.atk, "#ff8a4c");
          b.state = "idle"; b.stateT = 1.3;
        }
      } else if (b.state === "breath") {
        b.extra.tele += dt;
        if (b.extra.tele > 0.13) {
          b.extra.tele = 0;
          aimedVolley(b, ph2 ? 3 : 2, 0.3, 300, "fire", "#ff8a4c", b.def.atk * 0.75);
          AudioSys.sfx("fire");
        }
        if (b.stateT <= 0) { b.state = "idle"; b.stateT = 1.2; }
      }
      break;
    }
    case "serpent": {
      const transformed = b.phase >= 2 && !b.extra.form;
      if (b.hp / b.maxhp < 0.55 && !b.extra.form) {
        b.extra.form = true;
        banner("TRUE FORM", "The Twilight Serpent sheds its stolen shape", 3);
        AudioSys.sfx("roar");
        ringVolley(b, 22, 260, "dark", "#7a5cff", b.def.atk * 0.8);
        Particles.burst(b.x, b.y, "#7a5cff", 40, 300, 1);
        G.cam.addShake(10);
        b.state = "idle"; b.stateT = 1.4;
      }
      if (b.state === "idle") {
        b.dir = aToP;
        const spd = b.extra.form ? 120 : 96;
        if (d > 140) collideMove(b, Math.cos(aToP) * spd * dt, Math.sin(aToP) * spd * dt);
        if (b.stateT <= 0) {
          const roll = rand();
          if (!b.extra.form) {
            if (roll < 0.4) { b.state = "waves"; b.stateT = 0.6; }
            else if (roll < 0.75) { b.state = "dash"; b.stateT = 0.5; b.dashX = Math.cos(aToP); b.dashY = Math.sin(aToP); }
            else { b.state = "teleport"; b.stateT = 0.4; }
          } else {
            if (roll < 0.42) { b.state = "spiral"; b.stateT = 1.9; b.extra.spiralA = rand(TAU); }
            else if (roll < 0.72) { b.state = "waves"; b.stateT = 0.6; }
            else if (roll < 0.88) { b.state = "summon"; b.stateT = 0.7; }
            else { b.state = "dash"; b.stateT = 0.45; b.dashX = Math.cos(aToP); b.dashY = Math.sin(aToP); }
          }
        }
      } else if (b.state === "waves") {
        if (b.stateT <= 0) {
          aimedVolley(b, b.extra.form ? 7 : 5, 0.26, 280, "dark", "#7a5cff", b.def.atk * 0.85);
          AudioSys.sfx("lance");
          b.state = "idle"; b.stateT = ph3 ? 0.55 : 1;
        }
      } else if (b.state === "dash") {
        collideMove(b, b.dashX * 420 * dt, b.dashY * 420 * dt);
        Particles.spawn({ x: b.x, y: b.y, vx: rand(-30, 30), vy: rand(-30, 30), life: 0.35, col: "#7a5cff", size: 3 });
        if (b.stateT <= 0) { b.state = "idle"; b.stateT = 0.8; }
      } else if (b.state === "teleport") {
        if (b.stateT <= 0) {
          const a = rand(TAU);
          b.x = clamp(p.x + Math.cos(a) * 160, 48, G.mapDef.w * TS - 48);
          b.y = clamp(p.y + Math.sin(a) * 160, 48, G.mapDef.h * TS - 48);
          Particles.burst(b.x, b.y, "#7a5cff", 16, 180, 0.5);
          b.state = "waves"; b.stateT = 0.35;
        }
      } else if (b.state === "spiral") {
        b.extra.spiralA += dt * 10;
        b.extra.tele += dt;
        if (b.extra.tele > 0.1) {
          b.extra.tele = 0;
          for (let k = 0; k < 3; k++) {
            const a = b.extra.spiralA + k * (TAU / 3);
            fireProjectile({ x: b.x, y: b.y, vx: Math.cos(a) * 210, vy: Math.sin(a) * 210, dmg: b.def.atk * 0.7, from: "enemy", kind: "dark", col: "#7a5cff", r: 6, life: 3 });
          }
        }
        if (b.stateT <= 0) { b.state = "idle"; b.stateT = 0.9; }
      } else if (b.state === "summon") {
        if (b.stateT <= 0) {
          b.state = "idle"; b.stateT = 1.4;
          const alive = G.enemies.filter(e => !e.boss).length;
          if (alive < 4) {
            for (let i = 0; i < 2; i++) {
              const s = makeEnemy("skeleton", 0, 0);
              const a = rand(TAU);
              s.x = b.x + Math.cos(a) * 80; s.y = b.y + Math.sin(a) * 80;
              s.seenPlayer = true;
              G.enemies.push(s);
              Particles.burst(s.x, s.y, "#7a5cff", 10, 140, 0.5);
            }
          }
        }
      }
      break;
    }
  }

  // boss telegraph particles
  if ((b.state === "thorns" || b.state === "slam" || b.state === "leap") && chance(0.4))
    Particles.spawn({ x: b.x + rand(-20, 20), y: b.y + rand(-20, 20), vx: 0, vy: -40, life: 0.4, col: "#ffd24c", size: 2 });
  contactDamage(b);
}

/* ---------------- projectiles / pickups update ---------------- */
function updateProjectiles(dt) {
  const p = G.player;
  for (let i = G.projectiles.length - 1; i >= 0; i--) {
    const pr = G.projectiles[i];
    pr.life -= dt;
    pr.x += pr.vx * dt; pr.y += pr.vy * dt;
    if (chance(0.5)) Particles.spawn({ x: pr.x, y: pr.y, vx: 0, vy: 0, life: 0.22, col: pr.col, size: pr.r * 0.5 });
    if (pr.life <= 0 || solidPx(pr.x, pr.y)) {
      Particles.burst(pr.x, pr.y, pr.col, 5, 90, 0.3);
      G.projectiles.splice(i, 1);
      continue;
    }
    if (pr.from === "enemy") {
      if (dist(pr.x, pr.y, p.x, p.y) < pr.r + p.r - 3) {
        damagePlayer(pr.dmg, pr.x, pr.y);
        G.projectiles.splice(i, 1);
      }
    } else {
      for (const e of [...G.enemies]) {
        if (e.hp <= 0 || e.dormant) continue;
        if (dist(pr.x, pr.y, e.x, e.y) < pr.r + e.r) {
          const crit = chance(0.05 + pstats().luck * 0.008);
          hitEnemy(e, Math.round(pr.dmg * (crit ? 1.7 : 1)), 0, 0, crit);
          G.projectiles.splice(i, 1);
          break;
        }
      }
    }
  }
}

function updatePickups(dt) {
  const p = G.player;
  for (let i = G.pickups.length - 1; i >= 0; i--) {
    const u = G.pickups[i];
    u.t += dt;
    const d = dist(u.x, u.y, p.x, p.y);
    if ((u.type === "gold" || u.type === "heart") && d < 70) {
      const a = angleTo(u.x, u.y, p.x, p.y);
      u.x += Math.cos(a) * 260 * dt; u.y += Math.sin(a) * 260 * dt;
    }
    if (d < 22) {
      switch (u.type) {
        case "gold": G.gold += u.val; AudioSys.sfx("gold"); FloatTexts.add(u.x, u.y - 14, "+" + u.val + "G", "#ffd24c", 12); break;
        case "heart": {
          const st = pstats();
          const amt = Math.min(22, st.maxhp - p.hp);
          p.hp += amt; AudioSys.sfx("heart");
          if (amt > 0) FloatTexts.add(u.x, u.y - 14, "+" + amt, "#8aff8a", 13);
          break;
        }
        case "item": addItem(u.itemId); AudioSys.sfx("pickup"); FloatTexts.add(u.x, u.y - 14, ITEMS[u.itemId].name, "#bfe6ff", 13); break;
      }
      G.pickups.splice(i, 1);
    }
  }
}

/* ---------------- interactables ---------------- */
function nearestInteract() {
  const p = G.player;
  let best = null, bd = 46;
  for (const n of G.npcs) {
    const d = dist(p.x, p.y, n.px, n.py);
    if (d < bd) { bd = d; best = { type: "npc", o: n }; }
  }
  for (const c of G.chests) {
    if (c.opened) continue;
    const d = dist(p.x, p.y, (c.x + 0.5) * TS, (c.y + 0.5) * TS);
    if (d < bd) { bd = d; best = { type: "chest", o: c }; }
  }
  for (const pt of G.petalTiles) {
    const d = dist(p.x, p.y, (pt.x + 0.5) * TS, (pt.y + 0.5) * TS);
    if (d < bd) { bd = d; best = { type: "petal", o: pt }; }
  }
  return best;
}

function doInteract() {
  const it = nearestInteract();
  if (!it) return;
  if (it.type === "npc") interactNPC(it.o);
  else if (it.type === "chest") openChest(it.o);
  else if (it.type === "petal") collectPetal(it.o);
}

function openChest(c) {
  c.opened = true;
  G.flags["chest_" + c.id] = true;
  AudioSys.sfx("chest");
  const cx = (c.x + 0.5) * TS, cy = (c.y + 0.5) * TS;
  Particles.burst(cx, cy, "#ffd24c", 18, 180, 0.7);
  if (c.gold) { G.gold += c.gold; FloatTexts.add(cx, cy - 26, "+" + c.gold + "G", "#ffd24c", 14); }
  const names = [];
  for (const it of c.items) { addItem(it.id, it.n); names.push(ITEMS[it.id].name + (it.n > 1 ? " ×" + it.n : "")); }
  dialogPush([["", names.length ? `Found: ${names.join(", ")}${c.gold ? ` and ${c.gold} gold` : ""}.` : `Found ${c.gold} gold.`]]);
  saveGame();
}

function collectPetal(pt) {
  const idx = G.petalTiles.indexOf(pt);
  if (idx >= 0) G.petalTiles.splice(idx, 1);
  G.mapGrid[pt.y][pt.x] = G.mapDef.fill === "." ? "." : ",";
  addItem("moonpetal");
  G.quests.petalsN++;
  G.petalsCollectedTotal++;
  AudioSys.sfx("pickup");
  Particles.burst((pt.x + 0.5) * TS, (pt.y + 0.5) * TS, "#bfe6ff", 12, 130, 0.6);
  FloatTexts.add((pt.x + 0.5) * TS, (pt.y + 0.5) * TS - 20, "Moonpetal", "#bfe6ff", 13);
  if (G.quests.petals === 0) G.quests.petals = 0; // quest may not be accepted yet
  if (G.quests.petalsN === 6) toast("Collected all 6 Moonpetals! Return to Tillo.");
}

function substitute(text) {
  return text.replace("{petals}", String(Math.min(6, G.quests.petalsN)));
}

function dialogPush(lines, onEnd = null) {
  const cur = G.dialog;
  if (cur) { cur.queue.push({ lines, onEnd }); return; }
  G.dialog = { lines, i: 0, ch: 0, t: 0, onEnd, queue: [] };
}

function interactNPC(n) {
  AudioSys.sfx("talk");
  const q = G.quests;
  switch (n.id) {
    case "elder": {
      if (q.main === 0) {
        q.main = 1;
        dialogPush(DLG.elder_intro);
        toast("Quest updated: The Shattered Dawn");
      } else if ((G.flags.shards || 0) < 3) dialogPush([pick(DLG.elder_progress)]);
      else if (q.main === 1) { q.main = 2; dialogPush(DLG.elder_shards3); }
      else dialogPush([pick(DLG.elder_after)]);
      break;
    }
    case "petra": {
      if (q.pests === 1 && q.pestsN >= 6) {
        q.pests = 2;
        G.gold += 80; addItem("potion");
        dialogPush([
          ["Petra", "Oh, the meadow is quiet at last! You truly are the Order's finest."],
          ["Petra", "Here — 80 gold and a potion on the house. Spend it all, hear?"],
        ], () => openShop("village"));
        toast("Quest complete: Meadow Pests");
      } else if (q.pests === 0) {
        q.pests = 1; q.pestsN = 0;
        dialogPush([
          ["Petra", "Welcome, welcome! Potions for the brave, steel for the braver."],
          ["Petra", "Say, Kael — those meadow slimes keep eating my herb deliveries. Cull six of them and I'll pay you proper."],
        ], () => openShop("village"));
        toast("New quest: Meadow Pests");
      } else if (q.pests === 1) {
        dialogPush([["Petra", `Six slimes culled, remember! You've got ${q.pestsN} so far.`]], () => openShop("village"));
      } else {
        dialogPush(DLG.petra_hello, () => openShop("village"));
      }
      break;
    }
    case "tillo": {
      if (q.petals === 0) {
        q.petals = 1;
        dialogPush(DLG.tillo_intro);
        toast("New quest: Moonpetal Request");
      } else if (q.petals === 1 && q.petalsN >= 6) {
        q.petals = 2;
        addItem("lucky_charm");
        dialogPush(DLG.tillo_done);
        toast("Quest complete: Moonpetal Request — got Lucky Charm!");
      } else if (q.petals === 1) {
        dialogPush(DLG.tillo_progress.map(l => [l[0], substitute(l[1])]));
      } else dialogPush(DLG.tillo_after);
      break;
    }
    case "brennan": {
      if ((G.flags.shards || 0) < 3) dialogPush(DLG.brennan_locked);
      else if (!G.flags.brennan_open) { G.flags.brennan_open = true; dialogPush(DLG.brennan_open); }
      else dialogPush([DLG.brennan_open[1]]);
      break;
    }
    case "traveler": dialogPush([pick(DLG.traveler)]); break;
  }
}

function openShop(id) {
  G.shop = { id, sel: 0 };
}

function buyItem(id) {
  const it = ITEMS[id];
  if (G.gold < it.price) { AudioSys.sfx("deny"); toast("Not enough gold."); return; }
  G.gold -= it.price;
  addItem(id);
  AudioSys.sfx("buy");
  toast(`Bought ${it.name}.`);
}

/* ---------------- portals / transition ---------------- */
function checkPortals(dt) {
  const p = G.player;
  if (G.trans) return;
  for (const por of G.mapDef.portals) {
    const rx = por.x * TS, ry = por.y * TS, rw = por.w * TS, rh = por.h * TS;
    if (p.x > rx && p.x < rx + rw && p.y > ry && p.y < ry + rh) {
      if (por.needs && (G.flags.shards || 0) < por.needs.shards) {
        if (G.portalLockToastT <= 0) {
          toast(por.lockMsg || "The way is sealed.", 2.6);
          AudioSys.sfx("deny");
          G.portalLockToastT = 3;
        }
        // push player back a bit
        collideMove(p, 0, 24);
        continue;
      }
      if (G.bossActive) {
        if (G.portalLockToastT <= 0) { toast("You cannot flee a Guardian!", 2); AudioSys.sfx("deny"); G.portalLockToastT = 2.5; }
        continue;
      }
      startTransition(por.to, por.tx, por.ty);
      return;
    }
  }
  G.portalLockToastT = Math.max(0, G.portalLockToastT - dt);
}

function startTransition(to, tx, ty) {
  G.trans = { t: 0, phase: 0, to, tx, ty };
  G.state = "transition";
}

function updateTransition(dt) {
  const tr = G.trans;
  tr.t += dt * 2.6;
  if (tr.phase === 0 && tr.t >= 1) {
    loadMap(tr.to, tr.tx, tr.ty);
    tr.phase = 1; tr.t = 0;
  } else if (tr.phase === 1 && tr.t >= 1) {
    G.trans = null;
    G.state = "play";
  }
}

/* ---------------- save / load ---------------- */
function saveGame() {
  if (!G.player || !G.mapId) return;
  const p = G.player;
  Save.write({
    v: 1, mapId: G.mapId, tx: Math.floor(p.x / TS), ty: Math.floor(p.y / TS),
    lvl: p.lvl, xp: p.xp, hp: p.hp, mp: p.mp,
    inv: p.inv, equip: p.equip,
    flags: G.flags, quests: G.quests,
    gold: G.gold, kills: G.kills, playT: G.playT,
    petalsTotal: G.petalsCollectedTotal,
  });
}

function loadSaveData(d) {
  G.player = makePlayer();
  const p = G.player;
  p.lvl = d.lvl || 1; p.xp = d.xp || 0;
  p.inv = d.inv || [{ id: "potion", n: 2 }];
  p.equip = d.equip || p.equip;
  G.flags = d.flags || {};
  G.quests = Object.assign({ main: 0, pests: 0, pestsN: 0, petals: 0, petalsN: 0 }, d.quests);
  G.gold = d.gold || 0; G.kills = d.kills || 0; G.playT = d.playT || 0;
  G.petalsCollectedTotal = d.petalsTotal || 0;
  const st = pstats();
  p.hp = clamp(d.hp == null ? st.maxhp : d.hp, 1, st.maxhp);
  p.mp = clamp(d.mp == null ? st.maxmp : d.mp, 0, st.maxmp);
  loadMap(d.mapId || "village", d.tx == null ? 20 : d.tx, d.ty == null ? 25 : d.ty);
}

function newGame() {
  G.player = makePlayer();
  G.flags = {}; G.quests = { main: 0, pests: 0, pestsN: 0, petals: 0, petalsN: 0 };
  G.gold = 30; G.kills = 0; G.playT = 0; G.petalsCollectedTotal = 0;
  G.dialog = null; G.shop = null; G.bossActive = null;
  loadMap("village", 20, 25);
  G.state = "play";
  dialogPush([
    ["", "Dawnspire Village — morning after the Shattering."],
    ["", "Find ELDER MARO at the fountain. (Move: WASD · Talk: E)"],
  ]);
}

/* ---------------- ending ---------------- */
function startEnding() {
  G.state = "ending"; G.endingT = 0; G.endingStep = 0;
  AudioSys.setTheme("victory");
}

/* ---------------- master tick ---------------- */
function gameTick(dt) {
  G.t += dt;
  if (G.toast) { G.toast.t -= dt; if (G.toast.t <= 0) G.toast = null; }
  if (G.banner) { G.banner.t -= dt; if (G.banner.t <= 0) G.banner = null; }

  if (G.state === "transition") { updateTransition(dt); return; }
  if (G.state === "ending") { G.endingT += dt; return; }
  if (G.state === "dead") { G.deadT += dt; return; }
  if (G.state !== "play" || G.paused || G.menuOpen || G.dialog || G.shop) {
    if (G.state === "play" && !G.paused) { Particles.update(dt); FloatTexts.update(dt); }
    return;
  }

  G.playT += dt;

  // hitstop: freeze world briefly
  if (G.hitstop > 0) {
    G.hitstop -= dt;
    G.cam.update(dt, G.player.x, G.player.y, G.mapDef.w * TS, G.mapDef.h * TS, VIEW_W, VIEW_H);
    return;
  }

  updatePlayer(dt);
  for (const e of [...G.enemies]) updateEnemy(e, dt);
  updateProjectiles(dt);
  updatePickups(dt);
  checkPortals(dt);

  // interact
  if (Input.consume("interact")) doInteract();

  // ambient particles
  if (G.mapDef.theme === "ember" && chance(0.12)) {
    Particles.spawn({ x: G.cam.x + rand(VIEW_W), y: G.cam.y + VIEW_H + 10, vx: rand(-10, 10), vy: rand(-70, -30), life: 2, col: "#ff9a4c", size: 2 });
  }
  if (G.mapDef.theme === "forest" && chance(0.05)) {
    Particles.spawn({ x: G.cam.x + rand(VIEW_W), y: G.cam.y + rand(VIEW_H), vx: rand(-14, 14), vy: rand(-8, 8), life: 2.6, col: "#c8ffb0", size: 1.6 });
  }

  Particles.update(dt);
  FloatTexts.update(dt);
  G.cam.update(dt, G.player.x, G.player.y, G.mapDef.w * TS, G.mapDef.h * TS, VIEW_W, VIEW_H);
}
