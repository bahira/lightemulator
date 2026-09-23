"use strict";
/* ============================================================
   AETHERFALL — ui.js
   rendering · HUD · menus · dialogue · screens · main loop
   ============================================================ */

const canvas = document.getElementById("game");
const ctx = canvas.getContext("2d");
const lightCv = document.createElement("canvas");
lightCv.width = VIEW_W; lightCv.height = VIEW_H;
const lctx = lightCv.getContext("2d");

/* ---------- canvas fit ---------- */
function fitCanvas() {
  const s = Math.min(window.innerWidth / VIEW_W, window.innerHeight / VIEW_H);
  canvas.style.width = Math.floor(VIEW_W * s) + "px";
  canvas.style.height = Math.floor(VIEW_H * s) + "px";
}
window.addEventListener("resize", fitCanvas);
fitCanvas();

/* ---------- UI helpers ---------- */
function roundRect(c, x, y, w, h, r) {
  c.beginPath();
  c.moveTo(x + r, y);
  c.arcTo(x + w, y, x + w, y + h, r);
  c.arcTo(x + w, y + h, x, y + h, r);
  c.arcTo(x, y + h, x, y, r);
  c.arcTo(x, y, x + w, y, r);
  c.closePath();
}
function bar(c, x, y, w, h, frac, col1, col2) {
  c.fillStyle = "rgba(0,0,0,0.55)";
  roundRect(c, x, y, w, h, h / 2); c.fill();
  if (frac > 0) {
    const g = c.createLinearGradient(x, y, x, y + h);
    g.addColorStop(0, col1); g.addColorStop(1, col2);
    c.fillStyle = g;
    roundRect(c, x + 1.5, y + 1.5, Math.max(2, (w - 3) * clamp(frac, 0, 1)), h - 3, (h - 3) / 2); c.fill();
  }
}
function txt(c, s, x, y, size = 14, col = "#fff", align = "left", weight = 600) {
  c.font = `${weight} ${size}px "Segoe UI", system-ui, sans-serif`;
  c.fillStyle = col;
  c.textAlign = align;
  c.fillText(s, x, y);
}
function txtGlow(c, s, x, y, size, col, glow, align = "center") {
  c.save();
  c.font = `800 ${size}px "Segoe UI", system-ui, sans-serif`;
  c.textAlign = align;
  c.shadowColor = glow; c.shadowBlur = 18;
  c.fillStyle = col;
  c.fillText(s, x, y);
  c.restore();
}

/* ---------- UI nav (with repeat) ---------- */
const UI = {
  prevAx: 0, prevAy: 0, delay: 0, rep: 0,
  axis(dt) {
    const mv = Input.axis();
    const ax = Math.abs(mv.x) > 0.5 ? Math.sign(mv.x) : 0;
    const ay = Math.abs(mv.y) > 0.5 ? Math.sign(mv.y) : 0;
    let nx = 0, ny = 0;
    const freshX = ax !== 0 && ax !== this.prevAx;
    const freshY = ay !== 0 && ay !== this.prevAy;
    if (freshX || freshY) { this.delay = 0.34; this.rep = 0.14; nx = freshX ? ax : 0; ny = freshY ? ay : 0; }
    else if (ax !== 0 || ay !== 0) {
      this.delay -= dt;
      if (this.delay <= 0) {
        this.rep -= dt;
        if (this.rep <= 0) { this.rep = 0.14; nx = ax; ny = ay; }
      }
    }
    this.prevAx = ax; this.prevAy = ay;
    return { x: nx, y: ny };
  },
  confirm() { return Input.consume("attack") || Input.consume("interact"); },
  back() { return Input.consume("menu") || Input.consume("pause"); },
};

/* ============================================================
   WORLD RENDERING
   ============================================================ */
function drawDynamicTiles(t) {
  const camX = G.cam.x - G.cam.sx, camY = G.cam.y - G.cam.sy;
  // water shimmer
  ctx.save();
  ctx.globalAlpha = 0.35;
  ctx.fillStyle = "#bfe6ff";
  for (const [x, y] of G.waterTiles) {
    const px = x * TS - camX, py = y * TS - camY;
    if (px < -TS || py < -TS || px > VIEW_W || py > VIEW_H) continue;
    const ph = Math.sin(t * 2 + x * 1.7 + y * 2.3);
    ctx.fillRect(px + 4 + ph * 4, py + 8, 10, 2);
    ctx.fillRect(px + 16 - ph * 4, py + 22, 8, 2);
  }
  ctx.restore();
  // lava glow
  for (const [x, y] of G.lavaTiles) {
    const px = x * TS - camX + 16, py = y * TS - camY + 16;
    if (px < -40 || py < -40 || px > VIEW_W + 40 || py > VIEW_H + 40) continue;
    const ph = 0.5 + 0.5 * Math.sin(t * 3 + x * 2.1 + y);
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    ctx.globalAlpha = 0.10 + ph * 0.12;
    const g = ctx.createRadialGradient(px, py, 2, px, py, 34);
    g.addColorStop(0, "#ffd24c"); g.addColorStop(1, "rgba(255,100,20,0)");
    ctx.fillStyle = g;
    ctx.fillRect(px - 34, py - 34, 68, 68);
    ctx.restore();
    if (chance(0.004)) Particles.spawn({ x: x * TS + rand(28) + 2, y: y * TS + rand(28) + 2, vx: rand(-6, 6), vy: rand(-46, -20), life: 0.9, col: "#ffb02e", size: 2 });
  }
  // glow tiles (torches, crystals)
  for (const [x, y, col] of G.glowTiles) {
    const px = x * TS - camX + 16, py = y * TS - camY + 12;
    if (px < -60 || py < -60 || px > VIEW_W + 60 || py > VIEW_H + 60) continue;
    const ph = 0.6 + 0.4 * Math.sin(t * 6 + x * 3 + y * 7);
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    ctx.globalAlpha = 0.16 * ph + 0.08;
    const g = ctx.createRadialGradient(px, py, 2, px, py, 52);
    g.addColorStop(0, col); g.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = g;
    ctx.fillRect(px - 52, py - 52, 104, 104);
    ctx.restore();
    // flame sprite
    ctx.save();
    ctx.fillStyle = col;
    ctx.globalAlpha = 0.9;
    ctx.beginPath();
    ctx.ellipse(px, py - 2, 3, 5 + ph * 2, 0, 0, TAU);
    ctx.fill();
    ctx.restore();
    if (chance(0.01)) Particles.spawn({ x: x * TS + 16 + rand(-3, 3), y: y * TS + 8, vx: rand(-8, 8), vy: rand(-36, -18), life: 0.7, col, size: 1.8 });
  }
  // portal shimmer
  for (const por of G.mapDef.portals) {
    const px = (por.x + por.w / 2) * TS - camX, py = (por.y + por.h / 2) * TS - camY;
    const ph = 0.5 + 0.5 * Math.sin(t * 3);
    ctx.save();
    ctx.globalAlpha = 0.25 + ph * 0.2;
    ctx.fillStyle = "#ffe27a";
    for (let i = 0; i < 3; i++) {
      const yy = py + 10 - ((t * 26 + i * 14) % 40);
      ctx.fillRect(px - 8 + i * 8, yy, 3, 3);
    }
    ctx.restore();
  }
}

function drawShadow(x, y, r) {
  ctx.save();
  ctx.globalAlpha = 0.28;
  ctx.fillStyle = "#000";
  ctx.beginPath();
  ctx.ellipse(x, y, r, r * 0.42, 0, 0, TAU);
  ctx.fill();
  ctx.restore();
}

/* ---------- player ---------- */
const ARMOR_COL = { cloth_tunic: "#6b8fc9", leather_vest: "#a97c50", chainmail: "#9aa5b8", knight_plate: "#cfd8e8", dawn_mail: "#ffd24c" };
const WEAPON_COL = { trainee_sword: "#b8c4d4", iron_sword: "#dfe8f4", steel_claymore: "#eef4ff", flamebrand: "#ff9a4c", dawn_saber: "#ffe27a" };

function drawPlayer(t) {
  const p = G.player;
  if (p.invulnT > 0 && Math.floor(t * 14) % 2 === 0 && p.dashT <= 0) return;
  const camX = G.cam.x - G.cam.sx, camY = G.cam.y - G.cam.sy;

  // dash ghosts
  for (const gh of p.ghosts) {
    ctx.save();
    ctx.globalAlpha = gh.t * 1.6;
    ctx.fillStyle = "#bfe6ff";
    ctx.beginPath();
    ctx.arc(gh.x - camX, gh.y - camY - 6, 9, 0, TAU);
    ctx.fill();
    ctx.restore();
  }

  const x = p.x - camX, y = p.y - camY;
  drawShadow(x, y + 10, 11);
  const bob = Math.sin(p.walkT * 11) * (p.state === "move" ? 2 : 0.6);
  const armor = ARMOR_COL[p.equip.armor] || "#6b8fc9";
  const blade = WEAPON_COL[p.equip.weapon] || "#dfe8f4";

  ctx.save();
  ctx.translate(x, y + bob);
  if (p.hurtT > 0) { ctx.translate(rand(-1.5, 1.5), rand(-1.5, 1.5)); }

  // cape
  ctx.fillStyle = "#31518a";
  ctx.beginPath();
  ctx.moveTo(-7, -8); ctx.lineTo(7, -8); ctx.lineTo(9 - p.faceX * 3, 8 + Math.sin(p.walkT * 9) * 2); ctx.lineTo(-9 - p.faceX * 3, 8);
  ctx.closePath(); ctx.fill();
  // legs
  ctx.fillStyle = "#3a3f52";
  const lo = Math.sin(p.walkT * 11) * (p.state === "move" ? 3 : 0);
  ctx.fillRect(-5, 6, 4, 6 + lo); ctx.fillRect(1, 6, 4, 6 - lo);
  // body
  ctx.fillStyle = armor;
  roundRect(ctx, -7, -8, 14, 15, 4); ctx.fill();
  ctx.fillStyle = "rgba(0,0,0,0.18)";
  roundRect(ctx, -7, 0, 14, 7, 3); ctx.fill();
  // head
  ctx.fillStyle = "#ffd9b3";
  ctx.beginPath(); ctx.arc(0, -14, 7.5, 0, TAU); ctx.fill();
  ctx.fillStyle = "#7a4a22";
  ctx.beginPath(); ctx.arc(0, -16, 7.5, Math.PI, TAU); ctx.fill();
  ctx.fillRect(-7.5, -16, 15, 3);
  // headband
  ctx.fillStyle = "#ffd24c"; ctx.fillRect(-7.5, -13.4, 15, 2);
  // eyes
  ctx.fillStyle = "#2a2440";
  const ex = clamp(p.faceX, -1, 1) * 2;
  ctx.fillRect(-3 + ex, -13, 2, 3); ctx.fillRect(1.5 + ex, -13, 2, 3);

  // sword
  let swordA;
  if (p.atkT > 0) {
    const pr = 1 - p.atkT / p.atkDur;
    swordA = p.dir + lerp(-1.9, 1.9, easeOut(pr)) * (p.combo === 2 ? 1.6 : 1);
  } else {
    swordA = p.dir + 1.1;
  }
  ctx.save();
  ctx.translate(0, -2);
  ctx.rotate(swordA);
  ctx.fillStyle = "#6b4a2f"; ctx.fillRect(6, -1.5, 6, 3);
  ctx.fillStyle = blade; ctx.fillRect(12, -1.8, 16, 3.6);
  ctx.fillStyle = "rgba(255,255,255,0.7)"; ctx.fillRect(12, -1.8, 16, 1.4);
  if (p.equip.weapon === "flamebrand" || p.equip.weapon === "dawn_saber") {
    ctx.globalCompositeOperation = "lighter";
    ctx.globalAlpha = 0.5;
    ctx.fillStyle = p.equip.weapon === "flamebrand" ? "#ff9a4c" : "#ffe27a";
    ctx.fillRect(12, -3, 17, 6);
  }
  ctx.restore();

  // slash arc
  if (p.atkT > 0) {
    const pr = 1 - p.atkT / p.atkDur;
    ctx.save();
    ctx.translate(0, -2);
    ctx.rotate(p.dir);
    ctx.globalAlpha = (1 - pr) * 0.55;
    ctx.fillStyle = "#ffffff";
    ctx.beginPath();
    ctx.arc(0, 0, 42, -0.95 + (p.combo === 2 ? pr * 2 : 0), 0.95);
    ctx.arc(0, 0, 20, 0.95, -0.95, true);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }
  ctx.restore();
}

/* ---------- enemies ---------- */
function drawEnemy(e, t) {
  const camX = G.cam.x - G.cam.sx, camY = G.cam.y - G.cam.sy;
  const x = e.x - camX, y = e.y - camY;
  if (x < -60 || y < -60 || x > VIEW_W + 60 || y > VIEW_H + 60) return;
  drawShadow(x, y + e.r * 0.8, e.r * 0.9);

  ctx.save();
  ctx.translate(x, y + (e.boss ? 0 : Math.sin(e.t * 6) * 1.5));
  const wob = Math.sin(e.t * 7);
  switch (e.kind) {
    case "slime": {
      ctx.fillStyle = e.def.col;
      ctx.beginPath();
      ctx.ellipse(0, 2, e.r * (1 + wob * 0.08), e.r * (0.85 - wob * 0.08), 0, 0, TAU);
      ctx.fill();
      ctx.fillStyle = "rgba(255,255,255,0.5)";
      ctx.beginPath(); ctx.ellipse(-4, -3, 3, 2, 0, 0, TAU); ctx.fill();
      ctx.fillStyle = "#173a10";
      ctx.fillRect(-5, -1, 3, 4); ctx.fillRect(2, -1, 3, 4);
      break;
    }
    case "bee": {
      ctx.fillStyle = "#f2c94c";
      ctx.beginPath(); ctx.ellipse(0, 0, 8, 6, 0, 0, TAU); ctx.fill();
      ctx.fillStyle = "#2a2440";
      ctx.fillRect(-4, -5, 3, 10); ctx.fillRect(2, -5, 3, 10);
      ctx.fillStyle = "rgba(220,240,255,0.8)";
      const wf = Math.sin(e.t * 40) * 5;
      ctx.beginPath(); ctx.ellipse(-3, -8 - wf * 0.4, 5, 3, -0.4, 0, TAU); ctx.fill();
      ctx.beginPath(); ctx.ellipse(3, -8 + wf * 0.4, 5, 3, 0.4, 0, TAU); ctx.fill();
      break;
    }
    case "bat": {
      const wf = Math.sin(e.t * 16) * 7;
      ctx.fillStyle = "#5a3f8a";
      ctx.beginPath(); ctx.moveTo(-6, 0); ctx.lineTo(-18, -6 - wf); ctx.lineTo(-8, 4); ctx.closePath(); ctx.fill();
      ctx.beginPath(); ctx.moveTo(6, 0); ctx.lineTo(18, -6 + wf); ctx.lineTo(8, 4); ctx.closePath(); ctx.fill();
      ctx.fillStyle = e.def.col;
      ctx.beginPath(); ctx.arc(0, 0, 7, 0, TAU); ctx.fill();
      ctx.fillStyle = "#ff5a5a";
      ctx.fillRect(-4, -2, 2, 3); ctx.fillRect(2, -2, 2, 3);
      break;
    }
    case "wolf": {
      ctx.rotate(Math.cos(e.dir) >= 0 ? 0 : 0);
      const fx = Math.cos(e.dir) >= 0 ? 1 : -1;
      ctx.scale(fx, 1);
      ctx.fillStyle = e.def.col;
      ctx.beginPath(); ctx.ellipse(-2, 2, 13, 7, 0, 0, TAU); ctx.fill();
      ctx.beginPath(); ctx.arc(10, -2, 6, 0, TAU); ctx.fill();
      ctx.beginPath(); ctx.moveTo(7, -6); ctx.lineTo(10, -12); ctx.lineTo(12, -6); ctx.closePath(); ctx.fill();
      ctx.fillStyle = "#2a2440"; ctx.fillRect(11, -4, 2, 2);
      ctx.fillStyle = e.def.col;
      ctx.fillRect(-12, 4, 4, 6); ctx.fillRect(-4, 4, 4, 6); ctx.fillRect(4, 4, 4, 6);
      if (e.state === "windup") { ctx.fillStyle = "#ff5a5a"; ctx.fillRect(11, -4, 2, 2); }
      break;
    }
    case "archer": case "necro": case "imp": {
      drawHumanoid(e, t);
      break;
    }
    case "skeleton": {
      ctx.fillStyle = e.def.col;
      ctx.beginPath(); ctx.arc(0, -8, 6, 0, TAU); ctx.fill();
      ctx.fillRect(-5, -2, 10, 9);
      ctx.fillStyle = "#2a2440";
      ctx.fillRect(-3, -10, 2, 3); ctx.fillRect(1, -10, 2, 3);
      ctx.strokeStyle = e.def.col; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(-5, 0); ctx.lineTo(5, 0); ctx.moveTo(-5, 3); ctx.lineTo(5, 3); ctx.stroke();
      ctx.fillRect(-4, 7, 3, 6); ctx.fillRect(1, 7, 3, 6);
      break;
    }
    case "golem": {
      ctx.fillStyle = "#4a342c";
      roundRect(ctx, -14, -16, 28, 30, 7); ctx.fill();
      ctx.fillStyle = e.def.col;
      roundRect(ctx, -11, -13, 22, 12, 4); ctx.fill();
      ctx.fillStyle = "#ffb02e";
      const gl = 0.6 + 0.4 * Math.sin(e.t * 4);
      ctx.globalAlpha = gl;
      ctx.fillRect(-8, -4, 16, 2); ctx.fillRect(-6, 2, 12, 2);
      ctx.globalAlpha = 1;
      ctx.fillStyle = "#ffd24c";
      ctx.fillRect(-7, -11, 4, 3); ctx.fillRect(3, -11, 4, 3);
      ctx.fillStyle = "#4a342c";
      ctx.fillRect(-18, -8, 5, 16); ctx.fillRect(13, -8, 5, 16);
      break;
    }
    default: drawHumanoid(e, t);
  }
  // hit flash
  if (e.hitT > 0) {
    ctx.globalAlpha = e.hitT * 4;
    ctx.fillStyle = "#fff";
    ctx.beginPath(); ctx.arc(0, e.boss ? -e.r * 0.4 : -2, e.r + 2, 0, TAU); ctx.fill();
  }
  ctx.restore();

  // hp bar
  if (e.hp < e.maxhp && !e.boss) {
    const bw = Math.max(24, e.r * 2.4);
    bar(ctx, x - bw / 2, y - e.r - 16, bw, 4, e.hp / e.maxhp, "#8aff8a", "#2f9e44");
  }
}

function drawHumanoid(e, t) {
  const d = e.def;
  const robe = d.col;
  ctx.fillStyle = "rgba(0,0,0,0.25)";
  ctx.beginPath(); ctx.ellipse(0, 10, 8, 3, 0, 0, TAU); ctx.fill();
  ctx.fillStyle = robe;
  ctx.beginPath();
  ctx.moveTo(-8, 10); ctx.lineTo(-6, -8); ctx.lineTo(6, -8); ctx.lineTo(8, 10);
  ctx.closePath(); ctx.fill();
  ctx.fillStyle = shade(robe, -18);
  ctx.beginPath(); ctx.arc(0, -12, 6.5, 0, TAU); ctx.fill();
  // eyes
  ctx.fillStyle = e.kind === "necro" ? "#c9a6ff" : e.kind === "imp" ? "#ffd24c" : "#2a2440";
  ctx.fillRect(-3, -13, 2, 3); ctx.fillRect(1, -13, 2, 3);
  if (e.kind === "archer") {
    ctx.strokeStyle = "#8a6a3f"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(9, -2, 8, -1.1, 1.1); ctx.stroke();
  }
  if (e.kind === "necro" || e.kind === "imp") {
    ctx.strokeStyle = "#6b4a2f"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(9, 8); ctx.lineTo(9, -14); ctx.stroke();
    ctx.fillStyle = e.kind === "necro" ? "#c9a6ff" : "#ff8a4c";
    ctx.beginPath(); ctx.arc(9, -16, 3, 0, TAU); ctx.fill();
  }
  if (e.kind === "imp") {
    ctx.fillStyle = robe;
    ctx.beginPath(); ctx.moveTo(-5, -17); ctx.lineTo(-2, -22); ctx.lineTo(-1, -16); ctx.closePath(); ctx.fill();
    ctx.beginPath(); ctx.moveTo(5, -17); ctx.lineTo(2, -22); ctx.lineTo(1, -16); ctx.closePath(); ctx.fill();
  }
}

function shade(hex, amt) {
  const n = parseInt(hex.slice(1), 16);
  let r = (n >> 16) + amt, g = ((n >> 8) & 0xff) + amt, b = (n & 0xff) + amt;
  r = clamp(r, 0, 255); g = clamp(g, 0, 255); b = clamp(b, 0, 255);
  return `rgb(${r},${g},${b})`;
}

/* ---------- bosses ---------- */
function drawBoss(e, t) {
  const camX = G.cam.x - G.cam.sx, camY = G.cam.y - G.cam.sy;
  const x = e.x - camX, y = e.y - camY;
  if (x < -120 || y < -120 || x > VIEW_W + 120 || y > VIEW_H + 120) return;
  drawShadow(x, y + e.r * 0.8, e.r);
  ctx.save();
  ctx.translate(x, y);

  if (e.kind === "guardian") {
    const sway = Math.sin(e.t * 1.6) * 3;
    ctx.fillStyle = "#5b3a24";
    roundRect(ctx, -14, -18, 28, 40, 8); ctx.fill();
    ctx.fillStyle = "#2e5f2c";
    ctx.beginPath(); ctx.arc(sway, -34, 26, 0, TAU); ctx.fill();
    ctx.fillStyle = "#3e7d3a";
    ctx.beginPath(); ctx.arc(sway - 12, -26, 14, 0, TAU); ctx.arc(sway + 13, -27, 13, 0, TAU); ctx.fill();
    ctx.fillStyle = "#8aff6b";
    ctx.shadowColor = "#8aff6b"; ctx.shadowBlur = 8;
    ctx.fillRect(-8 + sway * 0.3, -30, 5, 6); ctx.fillRect(3 + sway * 0.3, -30, 5, 6);
    ctx.shadowBlur = 0;
    ctx.strokeStyle = "#5b3a24"; ctx.lineWidth = 6;
    ctx.beginPath(); ctx.moveTo(-14, -6); ctx.lineTo(-30, 8 + Math.sin(e.t * 2) * 4); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(14, -6); ctx.lineTo(30, 8 - Math.sin(e.t * 2) * 4); ctx.stroke();
  } else if (e.kind === "wraith") {
    const fl = Math.sin(e.t * 2.4) * 5;
    ctx.translate(0, fl - 8);
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    ctx.globalAlpha = 0.3;
    const g = ctx.createRadialGradient(0, 0, 2, 0, 0, 44);
    g.addColorStop(0, "#b06bff"); g.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = g; ctx.fillRect(-44, -44, 88, 88);
    ctx.restore();
    ctx.fillStyle = "#241a3a";
    ctx.beginPath();
    ctx.moveTo(0, -26); ctx.quadraticCurveTo(24, -10, 16, 20);
    ctx.lineTo(8, 14); ctx.lineTo(0, 22); ctx.lineTo(-8, 14); ctx.lineTo(-16, 20);
    ctx.quadraticCurveTo(-24, -10, 0, -26);
    ctx.fill();
    ctx.fillStyle = "#b06bff";
    ctx.shadowColor = "#b06bff"; ctx.shadowBlur = 10;
    ctx.fillRect(-7, -14, 4, 6); ctx.fillRect(3, -14, 4, 6);
    ctx.beginPath(); ctx.arc(0, 2, 5, 0, TAU); ctx.fill();
    ctx.shadowBlur = 0;
  } else if (e.kind === "tyrant") {
    ctx.fillStyle = "#241a16";
    roundRect(ctx, -26, -30, 52, 52, 12); ctx.fill();
    ctx.fillStyle = "#3a2b25";
    roundRect(ctx, -20, -24, 40, 20, 6); ctx.fill();
    // horns
    ctx.fillStyle = "#241a16";
    ctx.beginPath(); ctx.moveTo(-18, -28); ctx.lineTo(-30, -46); ctx.lineTo(-8, -30); ctx.closePath(); ctx.fill();
    ctx.beginPath(); ctx.moveTo(18, -28); ctx.lineTo(30, -46); ctx.lineTo(8, -30); ctx.closePath(); ctx.fill();
    // lava cracks
    const gl = 0.6 + 0.4 * Math.sin(e.t * 5);
    ctx.fillStyle = `rgba(255,138,76,${gl})`;
    ctx.fillRect(-14, -8, 28, 3); ctx.fillRect(-10, 0, 20, 3); ctx.fillRect(-16, 8, 32, 3);
    ctx.fillStyle = "#ffd24c";
    ctx.shadowColor = "#ff8a4c"; ctx.shadowBlur = 10;
    ctx.fillRect(-12, -20, 7, 5); ctx.fillRect(5, -20, 7, 5);
    ctx.shadowBlur = 0;
  } else if (e.kind === "serpent") {
    if (!e.extra.form) {
      // dark knight
      ctx.save();
      ctx.scale(1.5, 1.5);
      ctx.fillStyle = "#171226";
      ctx.beginPath(); ctx.moveTo(-10, 14); ctx.lineTo(-7, -12); ctx.lineTo(7, -12); ctx.lineTo(10, 14); ctx.closePath(); ctx.fill();
      ctx.fillStyle = "#241a3a";
      ctx.beginPath(); ctx.arc(0, -16, 8, 0, TAU); ctx.fill();
      ctx.fillStyle = "#7a5cff";
      ctx.shadowColor = "#7a5cff"; ctx.shadowBlur = 12;
      ctx.fillRect(-4, -18, 3, 4); ctx.fillRect(1, -18, 3, 4);
      ctx.shadowBlur = 0;
      ctx.strokeStyle = "#3a3050"; ctx.lineWidth = 3;
      ctx.beginPath(); ctx.moveTo(10, 10); ctx.lineTo(20, -16); ctx.stroke();
      ctx.strokeStyle = "#7a5cff"; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(10, 10); ctx.lineTo(20, -16); ctx.stroke();
      ctx.restore();
    } else {
      // serpent: segments
      for (let i = 7; i >= 1; i--) {
        const a = e.dir + Math.PI + Math.sin(e.t * 4 - i * 0.6) * 0.5;
        const sx = Math.cos(a) * i * 15, sy = Math.sin(a) * i * 15;
        ctx.fillStyle = i % 2 ? "#241a3a" : "#171226";
        ctx.beginPath(); ctx.arc(sx, sy, 14 - i, 0, TAU); ctx.fill();
      }
      ctx.fillStyle = "#241a3a";
      ctx.beginPath(); ctx.arc(0, 0, 20, 0, TAU); ctx.fill();
      ctx.fillStyle = "#7a5cff";
      ctx.shadowColor = "#7a5cff"; ctx.shadowBlur = 16;
      ctx.fillRect(-9, -6, 6, 7); ctx.fillRect(3, -6, 6, 7);
      ctx.beginPath(); ctx.arc(0, 8, 5, 0, TAU); ctx.fill();
      ctx.shadowBlur = 0;
      // wings
      const wf = Math.sin(e.t * 6) * 8;
      ctx.fillStyle = "rgba(122,92,255,0.35)";
      ctx.beginPath(); ctx.moveTo(-12, -8); ctx.lineTo(-44, -26 - wf); ctx.lineTo(-16, 6); ctx.closePath(); ctx.fill();
      ctx.beginPath(); ctx.moveTo(12, -8); ctx.lineTo(44, -26 + wf); ctx.lineTo(16, 6); ctx.closePath(); ctx.fill();
    }
  }

  if (e.hitT > 0) {
    ctx.globalAlpha = e.hitT * 3;
    ctx.fillStyle = "#fff";
    ctx.beginPath(); ctx.arc(0, -e.r * 0.4, e.r + 4, 0, TAU); ctx.fill();
  }
  ctx.restore();
}

function drawNPC(n, t) {
  const camX = G.cam.x - G.cam.sx, camY = G.cam.y - G.cam.sy;
  const x = n.px - camX, y = n.py - camY + Math.sin((t + n.t) * 2.4) * 1.5;
  drawShadow(x, n.py - camY + 10, 9);
  ctx.save();
  ctx.translate(x, y);
  ctx.fillStyle = n.col;
  ctx.beginPath(); ctx.moveTo(-7, 9); ctx.lineTo(-5, -7); ctx.lineTo(5, -7); ctx.lineTo(7, 9); ctx.closePath(); ctx.fill();
  ctx.fillStyle = "#ffd9b3";
  ctx.beginPath(); ctx.arc(0, -11, 6, 0, TAU); ctx.fill();
  ctx.fillStyle = "#2a2440";
  ctx.fillRect(-2.6, -12, 2, 2.6); ctx.fillRect(0.8, -12, 2, 2.6);
  if (n.id === "elder") { ctx.fillStyle = "#dfe8f4"; ctx.fillRect(-4, -8, 8, 2); }
  ctx.restore();
  // name + prompt
  const d = dist(G.player.x, G.player.y, n.px, n.py);
  if (d < 90) {
    ctx.save();
    ctx.globalAlpha = clamp(1.4 - d / 90, 0, 1);
    txt(ctx, n.name, x, y - 26, 12, "#fff", "center");
    ctx.restore();
  }
}

function drawProjectile(pr, t) {
  const camX = G.cam.x - G.cam.sx, camY = G.cam.y - G.cam.sy;
  const x = pr.x - camX, y = pr.y - camY;
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  const g = ctx.createRadialGradient(x, y, 1, x, y, pr.r * 3);
  g.addColorStop(0, pr.col); g.addColorStop(1, "rgba(0,0,0,0)");
  ctx.globalAlpha = 0.5;
  ctx.fillStyle = g;
  ctx.fillRect(x - pr.r * 3, y - pr.r * 3, pr.r * 6, pr.r * 6);
  ctx.globalAlpha = 1;
  ctx.fillStyle = pr.col;
  ctx.beginPath(); ctx.arc(x, y, pr.r, 0, TAU); ctx.fill();
  ctx.fillStyle = "#fff";
  ctx.beginPath(); ctx.arc(x, y, pr.r * 0.4, 0, TAU); ctx.fill();
  ctx.restore();
}

function drawPickup(u, t) {
  const camX = G.cam.x - G.cam.sx, camY = G.cam.y - G.cam.sy;
  const x = u.x - camX, y = u.y - camY + Math.sin((t + u.t) * 4) * 3;
  ctx.save();
  if (u.type === "heart") {
    ctx.fillStyle = "#ff6b81";
    ctx.beginPath();
    ctx.arc(x - 3, y - 2, 4, 0, TAU); ctx.arc(x + 3, y - 2, 4, 0, TAU); ctx.fill();
    ctx.beginPath(); ctx.moveTo(x - 7, y); ctx.lineTo(x, y + 8); ctx.lineTo(x + 7, y); ctx.closePath(); ctx.fill();
  } else if (u.type === "shard") {
    ctx.shadowColor = "#ffe27a"; ctx.shadowBlur = 12;
    ctx.fillStyle = "#ffe27a";
    ctx.beginPath(); ctx.moveTo(x, y - 9); ctx.lineTo(x + 6, y); ctx.lineTo(x, y + 9); ctx.lineTo(x - 6, y); ctx.closePath(); ctx.fill();
    ctx.shadowBlur = 0;
    ctx.fillStyle = "#fff";
    ctx.beginPath(); ctx.moveTo(x, y - 5); ctx.lineTo(x + 3, y); ctx.lineTo(x, y + 5); ctx.lineTo(x - 3, y); ctx.closePath(); ctx.fill();
  } else if (u.type === "item") {
    ctx.fillStyle = ITEMS[u.itemId].col || "#bfe6ff";
    roundRect(ctx, x - 5, y - 7, 10, 14, 3); ctx.fill();
    ctx.fillStyle = "#fff"; ctx.fillRect(x - 3, y - 5, 6, 3);
  }
  ctx.restore();
}

function drawChest(c, t) {
  const camX = G.cam.x - G.cam.sx, camY = G.cam.y - G.cam.sy;
  const x = (c.x + 0.5) * TS - camX, y = (c.y + 0.5) * TS - camY;
  ctx.save();
  ctx.translate(x, y);
  drawShadow(0, 10, 12);
  ctx.fillStyle = "#6b4a2f";
  roundRect(ctx, -12, -8, 24, 18, 3); ctx.fill();
  ctx.fillStyle = "#8a6a3f";
  roundRect(ctx, -12, -8, 24, 7, 3); ctx.fill();
  ctx.fillStyle = "#ffd24c";
  ctx.fillRect(-2, -4, 4, 8);
  if (!c.opened) {
    ctx.globalCompositeOperation = "lighter";
    ctx.globalAlpha = 0.3 + 0.2 * Math.sin(t * 3 + c.x);
    const g = ctx.createRadialGradient(0, 0, 2, 0, 0, 26);
    g.addColorStop(0, "#ffe27a"); g.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = g; ctx.fillRect(-26, -26, 52, 52);
  }
  ctx.restore();
}

/* ---------- world frame ---------- */
function drawWorld(t) {
  const camX = G.cam.x - G.cam.sx, camY = G.cam.y - G.cam.sy;
  ctx.fillStyle = "#05060e";
  ctx.fillRect(0, 0, VIEW_W, VIEW_H);
  ctx.drawImage(G.mapBaked, -camX, -camY);
  drawDynamicTiles(t);

  for (const c of G.chests) drawChest(c, t);
  for (const u of G.pickups) drawPickup(u, t);

  // y-sorted actors
  const actors = [];
  for (const e of G.enemies) actors.push({ y: e.y, f: () => (e.boss ? drawBoss(e, t) : drawEnemy(e, t)) });
  for (const n of G.npcs) actors.push({ y: n.py, f: () => drawNPC(n, t) });
  actors.push({ y: G.player.y, f: () => drawPlayer(t) });
  actors.sort((a, b) => a.y - b.y);
  for (const a of actors) a.f();

  for (const pr of G.projectiles) drawProjectile(pr, t);
  Particles.draw(ctx, camX, camY);
  FloatTexts.draw(ctx, camX, camY);

  // ambient tint
  if (G.mapDef.ambient && G.mapDef.ambient !== "#00000000") {
    ctx.fillStyle = G.mapDef.ambient;
    ctx.fillRect(0, 0, VIEW_W, VIEW_H);
  }

  // cave darkness + light around player/torches
  if (G.mapDef.theme === "cave") {
    lctx.clearRect(0, 0, VIEW_W, VIEW_H);
    lctx.fillStyle = "rgba(4,2,14,0.88)";
    lctx.fillRect(0, 0, VIEW_W, VIEW_H);
    lctx.globalCompositeOperation = "destination-out";
    const punch = (px, py, r) => {
      const g = lctx.createRadialGradient(px, py, r * 0.15, px, py, r);
      g.addColorStop(0, "rgba(0,0,0,1)"); g.addColorStop(1, "rgba(0,0,0,0)");
      lctx.fillStyle = g;
      lctx.beginPath(); lctx.arc(px, py, r, 0, TAU); lctx.fill();
    };
    punch(G.player.x - camX, G.player.y - camY, 170);
    for (const [gx, gy, col] of G.glowTiles) punch(gx * TS + 16 - camX, gy * TS + 12 - camY, 90);
    lctx.globalCompositeOperation = "source-over";
    ctx.drawImage(lightCv, 0, 0);
  }

  // vignette
  const vg = ctx.createRadialGradient(VIEW_W / 2, VIEW_H / 2, VIEW_H * 0.42, VIEW_W / 2, VIEW_H / 2, VIEW_H * 0.85);
  vg.addColorStop(0, "rgba(0,0,0,0)"); vg.addColorStop(1, "rgba(0,0,0,0.42)");
  ctx.fillStyle = vg;
  ctx.fillRect(0, 0, VIEW_W, VIEW_H);
}

/* ============================================================
   HUD
   ============================================================ */
function questHint() {
  const q = G.quests;
  if (q.main === 0) return "◆ Talk to Elder Maro in the village";
  if (q.main === 1) return `◆ Recover the Dawn Shards (${G.flags.shards || 0}/3)`;
  if (q.main === 2) return "◆ Enter the Shrine of Dawn (NE meadow)";
  return "";
}

function drawHUD(t) {
  const p = G.player, st = pstats();

  // HP / MP
  bar(ctx, 20, VIEW_H - 58, 250, 16, p.hp / st.maxhp, "#ff8a8a", "#c0392b");
  txt(ctx, `${Math.ceil(p.hp)} / ${st.maxhp}`, 26, VIEW_H - 46, 11, "#fff", "left", 800);
  bar(ctx, 20, VIEW_H - 38, 210, 11, p.mp / st.maxmp, "#7ec2ff", "#2c6fbb");
  txt(ctx, `${Math.floor(p.mp)} / ${st.maxmp}`, 26, VIEW_H - 29, 10, "#eaf4ff", "left", 700);
  // level badge + xp
  ctx.fillStyle = "#141a2e";
  ctx.beginPath(); ctx.arc(34, VIEW_H - 84, 14, 0, TAU); ctx.fill();
  ctx.strokeStyle = "#ffd24c"; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(34, VIEW_H - 84, 14, 0, TAU); ctx.stroke();
  txt(ctx, "Lv", 34, VIEW_H - 90, 8, "#ffd24c", "center", 800);
  txt(ctx, String(p.lvl), 34, VIEW_H - 77, 12, "#fff", "center", 800);
  bar(ctx, 54, VIEW_H - 88, 120, 6, p.xp / xpNeed(p.lvl), "#ffd24c", "#c99a2c");
  txt(ctx, "XP", 180, VIEW_H - 82, 9, "#c9b46a", "left");

  // skills
  const skills = [
    { key: "1", name: "Radiant Nova", cd: 5, mp: 8 },
    { key: "2", name: "Sun Lance", cd: 2.2, mp: 5 },
    { key: "3", name: "Dawn's Grace", cd: 9, mp: 14 },
  ];
  for (let i = 0; i < 3; i++) {
    const x = VIEW_W / 2 - 78 + i * 52, y = VIEW_H - 46;
    const s = skills[i];
    ctx.fillStyle = "rgba(10,14,30,0.8)";
    roundRect(ctx, x, y, 40, 40, 8); ctx.fill();
    ctx.strokeStyle = p.mp >= s.mp ? "#5a7ab8" : "#3a3050";
    ctx.lineWidth = 1.5;
    roundRect(ctx, x, y, 40, 40, 8); ctx.stroke();
    // icon
    ctx.save();
    ctx.translate(x + 20, y + 19);
    if (i === 0) { ctx.strokeStyle = "#ffe27a"; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.arc(0, 0, 9, 0, TAU); ctx.stroke(); ctx.fillStyle = "#ffe27a"; ctx.beginPath(); ctx.arc(0, 0, 3, 0, TAU); ctx.fill(); }
    if (i === 1) { ctx.strokeStyle = "#ffe27a"; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(-8, 8); ctx.lineTo(8, -8); ctx.stroke(); ctx.beginPath(); ctx.moveTo(8, -8); ctx.lineTo(2, -7); ctx.moveTo(8, -8); ctx.lineTo(7, -2); ctx.stroke(); }
    if (i === 2) { ctx.fillStyle = "#8aff8a"; ctx.fillRect(-2.5, -9, 5, 18); ctx.fillRect(-9, -2.5, 18, 5); }
    ctx.restore();
    // cooldown sweep
    const frac = p.cds[i] / s.cd;
    if (frac > 0) {
      ctx.fillStyle = "rgba(0,0,0,0.62)";
      ctx.beginPath();
      ctx.moveTo(x + 20, y + 20);
      ctx.arc(x + 20, y + 20, 26, -Math.PI / 2, -Math.PI / 2 + frac * TAU);
      ctx.closePath(); ctx.fill();
      txt(ctx, p.cds[i].toFixed(1), x + 20, y + 25, 12, "#fff", "center", 800);
    }
    txt(ctx, s.key, x + 6, y + 12, 10, "#c9d4ea", "left", 800);
  }

  // gold & shards
  ctx.fillStyle = "#ffd24c";
  ctx.beginPath(); ctx.arc(30, 26, 7, 0, TAU); ctx.fill();
  ctx.fillStyle = "#c99a2c";
  ctx.beginPath(); ctx.arc(30, 26, 4, 0, TAU); ctx.fill();
  txt(ctx, String(G.gold), 44, 31, 15, "#ffd24c", "left", 800);
  for (let i = 0; i < 3; i++) {
    const got = (G.flags.shards || 0) > i;
    ctx.save();
    ctx.translate(26 + i * 22, 50);
    ctx.rotate(Math.PI / 4);
    ctx.fillStyle = got ? "#ffe27a" : "rgba(255,226,122,0.15)";
    if (got) { ctx.shadowColor = "#ffe27a"; ctx.shadowBlur = 8; }
    ctx.fillRect(-6, -6, 12, 12);
    ctx.restore();
  }

  // minimap
  if (G.mapMini) {
    const mw = G.mapMini.width, mh = G.mapMini.height;
    const mx = VIEW_W - mw - 16, my = 16;
    ctx.fillStyle = "rgba(5,8,18,0.75)";
    roundRect(ctx, mx - 4, my - 4, mw + 8, mh + 8, 6); ctx.fill();
    ctx.drawImage(G.mapMini, mx, my);
    ctx.fillStyle = "#fff";
    ctx.fillRect(mx + G.player.x / TS * 2 - 1.5, my + G.player.y / TS * 2 - 1.5, 3, 3);
    for (const e of G.enemies) if (e.boss) {
      ctx.fillStyle = "#ff5a5a";
      ctx.fillRect(mx + e.x / TS * 2 - 2, my + e.y / TS * 2 - 2, 4, 4);
    }
  }

  // quest tracker
  const hint = questHint();
  if (hint) {
    ctx.font = "600 13px 'Segoe UI', system-ui, sans-serif";
    const w = ctx.measureText(hint).width;
    ctx.fillStyle = "rgba(5,8,18,0.6)";
    roundRect(ctx, VIEW_W - w - 30, VIEW_H - 40, w + 18, 24, 6); ctx.fill();
    txt(ctx, hint, VIEW_W - 21, VIEW_H - 24, 13, "#e8d9a0", "right");
  }

  // boss bar
  if (G.bossActive && !G.bossActive.dormant) {
    const b = G.bossActive;
    txt(ctx, b.def.name, VIEW_W / 2, 30, 15, "#ffd9d9", "center", 800);
    bar(ctx, VIEW_W / 2 - 210, 38, 420, 12, b.hp / b.maxhp, "#ff8a8a", "#8a1f1f");
  }

  // interact prompt
  const it = nearestInteract();
  if (it) {
    let label = "Talk", px = 0, py = 0;
    if (it.type === "npc") { label = `Talk to ${it.o.name}`; px = it.o.px; py = it.o.py - 38; }
    if (it.type === "chest") { label = "Open chest"; px = (it.o.x + 0.5) * TS; py = it.o.y * TS - 14; }
    if (it.type === "petal") { label = "Pick Moonpetal"; px = (it.o.x + 0.5) * TS; py = it.o.y * TS - 12; }
    const sx = px - (G.cam.x - G.cam.sx), sy = py - (G.cam.y - G.cam.sy);
    ctx.font = "700 12px 'Segoe UI', system-ui, sans-serif";
    const tw = ctx.measureText(label).width + 34;
    ctx.fillStyle = "rgba(5,8,18,0.85)";
    roundRect(ctx, sx - tw / 2, sy - 12, tw, 22, 6); ctx.fill();
    ctx.strokeStyle = "rgba(255,226,122,0.5)"; ctx.lineWidth = 1;
    roundRect(ctx, sx - tw / 2, sy - 12, tw, 22, 6); ctx.stroke();
    txt(ctx, "E", sx - tw / 2 + 12, sy + 4, 12, "#ffd24c", "center", 800);
    txt(ctx, label, sx - tw / 2 + 24, sy + 4, 12, "#fff", "left");
  }

  // toast
  if (G.toast) {
    const a = clamp(Math.min(G.toast.t, G.toast.dur - G.toast.t) * 2.4, 0, 1);
    ctx.save();
    ctx.globalAlpha = a;
    ctx.font = "700 17px 'Segoe UI', system-ui, sans-serif";
    const tw = ctx.measureText(G.toast.txt).width;
    ctx.fillStyle = "rgba(5,8,18,0.8)";
    roundRect(ctx, VIEW_W / 2 - tw / 2 - 18, 66, tw + 36, 34, 8); ctx.fill();
    txt(ctx, G.toast.txt, VIEW_W / 2, 88, 17, "#ffe27a", "center", 700);
    ctx.restore();
  }
  // map banner
  if (G.banner) {
    const b = G.banner;
    const pr = 1 - b.t / b.dur;
    const a = clamp(Math.min(b.t, b.dur - b.t) * 2, 0, 1);
    ctx.save();
    ctx.globalAlpha = a;
    txtGlow(ctx, b.txt, VIEW_W / 2, 180 - easeOut(Math.min(1, pr * 4)) * 0, 40, "#fff", "rgba(120,160,255,0.8)");
    if (b.sub) txt(ctx, b.sub, VIEW_W / 2, 210, 16, "#c9d4ea", "center");
    ctx.restore();
  }

  if (AudioSys.muted) txt(ctx, "MUTED (M)", VIEW_W - 20, VIEW_H - 46, 11, "#8a94ad", "right");
  txt(ctx, "I: menu · Esc: pause", VIEW_W - 20, VIEW_H - 14, 11, "rgba(160,170,200,0.55)", "right");

  // low-hp heartbeat vignette
  const hpf = p.hp / st.maxhp;
  if (hpf < 0.32) {
    const pulse = 0.5 + 0.5 * Math.sin(t * (3 + (0.32 - hpf) * 22));
    const g2 = ctx.createRadialGradient(VIEW_W / 2, VIEW_H / 2, VIEW_H * 0.3, VIEW_W / 2, VIEW_H / 2, VIEW_H * 0.75);
    g2.addColorStop(0, "rgba(120,0,0,0)");
    g2.addColorStop(1, `rgba(140,10,10,${0.22 + pulse * 0.22})`);
    ctx.fillStyle = g2;
    ctx.fillRect(0, 0, VIEW_W, VIEW_H);
  }
}

/* ============================================================
   DIALOG / MENU / SHOP / SCREENS
   ============================================================ */
function drawDialog(t) {
  const d = G.dialog;
  if (!d) return;
  const [who, line] = d.lines[d.i];
  const shown = line.slice(0, Math.floor(d.ch));
  const bx = 60, by = VIEW_H - 150, bw = VIEW_W - 120, bh = 118;
  ctx.fillStyle = "rgba(8,12,28,0.94)";
  roundRect(ctx, bx, by, bw, bh, 10); ctx.fill();
  ctx.strokeStyle = "rgba(120,160,255,0.35)"; ctx.lineWidth = 2;
  roundRect(ctx, bx, by, bw, bh, 10); ctx.stroke();
  if (who) {
    ctx.fillStyle = "#141a2e";
    roundRect(ctx, bx + 16, by - 15, ctx.measureText(who).width + 60, 30, 8);
    ctx.font = "800 15px 'Segoe UI', system-ui, sans-serif";
    roundRect(ctx, bx + 16, by - 15, ctx.measureText(who).width + 34, 30, 8); ctx.fill();
    txt(ctx, who, bx + 33, by + 5, 15, "#ffd24c", "left", 800);
  }
  ctx.font = "600 16px 'Segoe UI', system-ui, sans-serif";
  ctx.fillStyle = "#e8ecf6";
  ctx.textAlign = "left";
  wrapText(shown, bx + 26, by + 40, bw - 52, 22);
  if (Math.floor(d.ch) >= line.length) {
    const bob2 = Math.sin(t * 5) * 3;
    ctx.fillStyle = "#ffd24c";
    ctx.beginPath();
    ctx.moveTo(bx + bw - 30, by + bh - 24 + bob2);
    ctx.lineTo(bx + bw - 20, by + bh - 24 + bob2);
    ctx.lineTo(bx + bw - 25, by + bh - 16 + bob2);
    ctx.closePath(); ctx.fill();
  }
}
function wrapText(s, x, y, maxW, lh) {
  const words = s.split(" ");
  let line = "", yy = y;
  for (const w of words) {
    const test = line ? line + " " + w : w;
    if (ctx.measureText(test).width > maxW) {
      ctx.fillText(line, x, yy);
      line = w; yy += lh;
    } else line = test;
  }
  if (line) ctx.fillText(line, x, yy);
}

function tickDialog(dt) {
  const d = G.dialog;
  if (!d) return;
  const [, line] = d.lines[d.i];
  d.t += dt;
  d.ch += dt * 46;
  if (UI.confirm()) {
    if (Math.floor(d.ch) < line.length) { d.ch = line.length; }
    else {
      AudioSys.sfx("ui");
      if (d.i < d.lines.length - 1) { d.i++; d.ch = 0; }
      else {
        const onEnd = d.onEnd;
        const next = d.queue.shift();
        if (next) { G.dialog = { lines: next.lines, i: 0, ch: 0, t: 0, onEnd: next.onEnd, queue: d.queue }; }
        else G.dialog = null;
        if (onEnd) onEnd();
      }
    }
  }
}

/* ---------- pause ---------- */
function drawPause() {
  ctx.fillStyle = "rgba(4,6,14,0.8)";
  ctx.fillRect(0, 0, VIEW_W, VIEW_H);
  txtGlow(ctx, "PAUSED", VIEW_W / 2, 150, 42, "#fff", "rgba(120,160,255,0.7)");
  const lines = [
    "WASD / Arrows / Left Stick — move",
    "J or Z — attack (3-hit combo)",
    "K, X or Shift — dash (invincible while dashing)",
    "1 — Radiant Nova · 2 — Sun Lance · 3 — Dawn's Grace",
    "E — talk / open / gather · I — menu · M — mute",
    "",
    "Gamepad: stick move · A attack · B dash · X talk · Y/LB/RB skills",
  ];
  lines.forEach((l, i) => txt(ctx, l, VIEW_W / 2, 210 + i * 26, 15, "#c9d4ea", "center"));
  txt(ctx, "Esc to resume", VIEW_W / 2, 440, 14, "#ffd24c", "center", 800);
}

/* ---------- menu ---------- */
function initMenu() { G.menuOpen = true; G.menu = { tab: 0, sel: 0 }; }

function tickMenu(dt) {
  const m = G.menu;
  const ax = UI.axis(dt);
  if (ax.y) { m.sel += ax.y; AudioSys.sfx("ui"); }
  if (ax.x) { m.tab = (m.tab + ax.x + 3) % 3; m.sel = 0; AudioSys.sfx("ui"); }
  const tabs = ["ITEMS", "QUESTS", "STATUS"];
  if (m.tab === 0) m.sel = clamp(m.sel, 0, Math.max(0, G.player.inv.length - 1));
  if (UI.confirm()) {
    if (m.tab === 0 && G.player.inv.length) useMenuItem(G.player.inv[m.sel]);
  }
  if (UI.back()) { G.menuOpen = false; G.menu = null; AudioSys.sfx("ui"); }
}

function useMenuItem(entry) {
  const it = ITEMS[entry.id];
  const p = G.player;
  if (it.type === "use") {
    const st = pstats();
    if (it.heal && p.hp >= st.maxhp && !it.mp) { AudioSys.sfx("deny"); return; }
    if (it.heal) p.hp = Math.min(st.maxhp, p.hp + it.heal);
    if (it.mp) p.mp = Math.min(st.maxmp, p.mp + it.mp);
    removeItem(entry.id, 1);
    AudioSys.sfx("heal");
    Particles.burst(p.x, p.y, "#8aff8a", 12, 120, 0.6);
  } else if (it.type === "weapon" || it.type === "armor" || it.type === "acc") {
    const slot = it.type === "weapon" ? "weapon" : it.type === "armor" ? "armor" : "acc";
    const prev = p.equip[slot];
    p.equip[slot] = entry.id;
    removeItem(entry.id, 1);
    if (prev) addItem(prev);
    AudioSys.sfx("buy");
    toast(`Equipped ${it.name}.`);
  } else { AudioSys.sfx("ui"); }
}

function drawMenu(t) {
  const m = G.menu;
  const p = G.player, st = pstats();
  ctx.fillStyle = "rgba(4,6,14,0.88)";
  ctx.fillRect(0, 0, VIEW_W, VIEW_H);
  ctx.fillStyle = "rgba(12,18,38,0.92)";
  roundRect(ctx, 60, 40, VIEW_W - 120, VIEW_H - 80, 12); ctx.fill();
  ctx.strokeStyle = "rgba(120,160,255,0.3)"; ctx.lineWidth = 2;
  roundRect(ctx, 60, 40, VIEW_W - 120, VIEW_H - 80, 12); ctx.stroke();

  const tabs = ["ITEMS", "QUESTS", "STATUS"];
  for (let i = 0; i < 3; i++) {
    const active = m.tab === i;
    txt(ctx, tabs[i], VIEW_W / 2 + (i - 1) * 160, 74, 18, active ? "#ffd24c" : "#5a6a8a", "center", active ? 800 : 600);
    if (active) { ctx.fillStyle = "#ffd24c"; ctx.fillRect(VIEW_W / 2 + (i - 1) * 160 - 30, 82, 60, 3); }
  }

  if (m.tab === 0) {
    if (!p.inv.length) txt(ctx, "Your satchel is empty.", VIEW_W / 2, 200, 15, "#8a94ad", "center");
    const shown = p.inv.slice(0, 9);
    shown.forEach((entry, i) => {
      const it = ITEMS[entry.id];
      const y = 116 + i * 42;
      if (i === m.sel) {
        ctx.fillStyle = "rgba(255,226,122,0.1)";
        roundRect(ctx, 90, y - 16, 480, 36, 8); ctx.fill();
        txt(ctx, "▶", 100, y + 7, 15, "#ffd24c", "left");
      }
      ctx.fillStyle = it.col || "#c9d4ea";
      roundRect(ctx, 122, y - 12, 24, 24, 5); ctx.fill();
      txt(ctx, it.name + (entry.n > 1 ? ` ×${entry.n}` : ""), 158, y + 5, 15, "#fff", "left", 700);
      txt(ctx, typeLabel(it), 430, y + 5, 12, "#8a94ad", "left");
    });
    // detail pane
    const cur = p.inv[m.sel];
    if (cur) {
      const it = ITEMS[cur.id];
      ctx.fillStyle = "rgba(8,12,28,0.8)";
      roundRect(ctx, 596, 116, 268, 220, 10); ctx.fill();
      txt(ctx, it.name, 614, 146, 17, "#ffd24c", "left", 800);
      let yy = 172;
      for (const [k, v] of Object.entries(it)) {
        if (["atk", "def", "hp", "mp", "luck", "spd", "heal"].includes(k) && v) {
          txt(ctx, `${k.toUpperCase()} +${v}`, 614, yy, 13, "#8aff8a", "left"); yy += 20;
        }
      }
      ctx.fillStyle = "#c9d4ea";
      ctx.font = "500 13px 'Segoe UI', system-ui, sans-serif";
      wrapText(it.desc, 614, yy + 8, 236, 18);
      txt(ctx, "Enter: use / equip", 614, 320, 12, "#8a94ad", "left");
    }
  } else if (m.tab === 1) {
    const q = G.quests;
    const rows = [];
    rows.push(["The Shattered Dawn", mainQuestText(), "#ffd24c"]);
    if (q.pests > 0) rows.push(["Meadow Pests", q.pests === 2 ? "Complete — Petra is grateful." : `Cull slimes in the meadow (${Math.min(6, q.pestsN)}/6) — report to Petra.`, "#c9d4ea"]);
    if (q.petals > 0) rows.push(["Moonpetal Request", q.petals === 2 ? "Complete — Tillo got his flowers." : `Gather Moonpetals (${Math.min(6, q.petalsN)}/6) — return to Tillo.`, "#c9d4ea"]);
    rows.forEach((r, i) => {
      const y = 130 + i * 64;
      txt(ctx, r[0], 100, y, 16, r[2], "left", 800);
      ctx.font = "500 13px 'Segoe UI', system-ui, sans-serif";
      ctx.fillStyle = "#aab4cc"; ctx.textAlign = "left";
      wrapText(r[1], 100, y + 22, 740, 17);
    });
  } else {
    const rows = [
      ["Level", p.lvl], ["XP", `${p.xp} / ${xpNeed(p.lvl)}`],
      ["HP", `${Math.ceil(p.hp)} / ${st.maxhp}`], ["MP", `${Math.floor(p.mp)} / ${st.maxmp}`],
      ["Attack", st.atk], ["Defense", st.def], ["Speed", st.spd], ["Luck", st.luck],
      ["Gold", G.gold], ["Shards", `${G.flags.shards || 0} / 3`],
      ["Foes felled", G.kills], ["Journey", fmtTime(G.playT)],
    ];
    rows.forEach((r, i) => {
      const col = i % 2, row = Math.floor(i / 2);
      const x = 140 + col * 380, y = 140 + row * 52;
      txt(ctx, r[0], x, y, 14, "#8a94ad", "left");
      txt(ctx, String(r[1]), x + 150, y, 16, "#fff", "left", 800);
    });
  }
  txt(ctx, "← → tabs · ↑↓ select · Enter use · I/Esc close", VIEW_W / 2, VIEW_H - 52, 12, "#5a6a8a", "center");
}
function typeLabel(it) {
  return it.type === "use" ? "Consumable" : it.type === "weapon" ? "Weapon" : it.type === "armor" ? "Armor" : it.type === "acc" ? "Accessory" : "Key item";
}
function mainQuestText() {
  const q = G.quests;
  if (q.main === 0) return "Something is wrong at the fountain. Speak with Elder Maro.";
  if (q.main === 1) return `Defeat the three shard-guardians and recover the Dawn Shards (${G.flags.shards || 0}/3). Whisperwood · Hollowdeep · Emberfall Ridge.`;
  if (q.main === 2) return "The seal is broken. Enter the Shrine of Dawn in the north-east meadow and face the thief of the dawn.";
  return "The dawn has returned.";
}
function fmtTime(s) {
  const m = Math.floor(s / 60);
  return `${m}m ${String(Math.floor(s % 60)).padStart(2, "0")}s`;
}

/* ---------- shop ---------- */
function tickShop(dt) {
  const s = G.shop;
  const list = SHOPS[s.id];
  const ax = UI.axis(dt);
  if (ax.y) { s.sel = clamp(s.sel + ax.y, 0, list.length - 1); AudioSys.sfx("ui"); }
  if (UI.confirm()) buyItem(list[s.sel]);
  if (UI.back()) { G.shop = null; AudioSys.sfx("ui"); }
}
function drawShop(t) {
  const s = G.shop;
  const list = SHOPS[s.id];
  ctx.fillStyle = "rgba(4,6,14,0.86)";
  ctx.fillRect(0, 0, VIEW_W, VIEW_H);
  ctx.fillStyle = "rgba(12,18,38,0.94)";
  roundRect(ctx, 160, 50, VIEW_W - 320, VIEW_H - 100, 12); ctx.fill();
  ctx.strokeStyle = "rgba(255,210,76,0.35)"; ctx.lineWidth = 2;
  roundRect(ctx, 160, 50, VIEW_W - 320, VIEW_H - 100, 12); ctx.stroke();
  txt(ctx, "Petra's Provisions", VIEW_W / 2, 88, 22, "#ffd24c", "center", 800);
  txt(ctx, `${G.gold} G`, VIEW_W / 2, 112, 15, "#ffd24c", "center");
  list.forEach((id, i) => {
    const it = ITEMS[id];
    const y = 140 + i * 33;
    const afford = G.gold >= it.price;
    if (i === s.sel) {
      ctx.fillStyle = "rgba(255,226,122,0.1)";
      roundRect(ctx, 190, y - 14, VIEW_W - 380, 30, 7); ctx.fill();
      txt(ctx, "▶", 200, y + 6, 14, "#ffd24c");
    }
    txt(ctx, it.name, 226, y + 6, 14, afford ? "#fff" : "#5a6a8a", "left", 700);
    txt(ctx, itemShort(it), 470, y + 6, 12, "#8a94ad", "left");
    txt(ctx, `${it.price} G`, VIEW_W - 210, y + 6, 14, afford ? "#ffd24c" : "#5a4a4a", "right", 800);
  });
  txt(ctx, "Enter: buy · I/Esc: leave", VIEW_W / 2, VIEW_H - 62, 12, "#5a6a8a", "center");
}
function itemShort(it) {
  const parts = [];
  if (it.atk) parts.push(`ATK+${it.atk}`);
  if (it.def) parts.push(`DEF+${it.def}`);
  if (it.hp) parts.push(`HP+${it.hp}`);
  if (it.spd) parts.push(`SPD+${it.spd}`);
  if (it.luck) parts.push(`LCK+${it.luck}`);
  if (it.heal) parts.push(`Heal ${it.heal > 999 ? "MAX" : it.heal}`);
  if (it.mp) parts.push(`MP+${it.mp}`);
  return parts.join(" ");
}

/* ---------- title ---------- */
const titleStars = Array.from({ length: 90 }, () => ({ x: rand(VIEW_W), y: rand(VIEW_H * 0.7), s: rand(0.5, 1.8), p: rand(TAU) }));
let titleSel = 0;

function drawTitle(t) {
  // sky
  const g = ctx.createLinearGradient(0, 0, 0, VIEW_H);
  g.addColorStop(0, "#0a0e24"); g.addColorStop(0.55, "#241a3a"); g.addColorStop(0.8, "#6b3a4a"); g.addColorStop(1, "#c9703a");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, VIEW_W, VIEW_H);
  // stars
  for (const s of titleStars) {
    ctx.globalAlpha = 0.4 + 0.6 * Math.abs(Math.sin(t * 1.4 + s.p));
    ctx.fillStyle = "#fff";
    ctx.fillRect(s.x, s.y, s.s, s.s);
  }
  ctx.globalAlpha = 1;
  // rising sun
  const sunY = VIEW_H * 0.66 + Math.sin(t * 0.5) * 4;
  const sg = ctx.createRadialGradient(VIEW_W / 2, sunY, 10, VIEW_W / 2, sunY, 240);
  sg.addColorStop(0, "#ffe8b0"); sg.addColorStop(0.3, "rgba(255,180,90,0.8)"); sg.addColorStop(1, "rgba(255,120,60,0)");
  ctx.fillStyle = sg;
  ctx.fillRect(0, 0, VIEW_W, VIEW_H);
  ctx.fillStyle = "#ffd98a";
  ctx.beginPath(); ctx.arc(VIEW_W / 2, sunY, 60, 0, TAU); ctx.fill();
  // mountains
  ctx.fillStyle = "#171226";
  ctx.beginPath();
  ctx.moveTo(0, VIEW_H);
  ctx.lineTo(0, VIEW_H * 0.78); ctx.lineTo(140, VIEW_H * 0.62); ctx.lineTo(300, VIEW_H * 0.8);
  ctx.lineTo(430, VIEW_H * 0.66); ctx.lineTo(600, VIEW_H * 0.82); ctx.lineTo(760, VIEW_H * 0.64);
  ctx.lineTo(VIEW_W, VIEW_H * 0.8); ctx.lineTo(VIEW_W, VIEW_H);
  ctx.closePath(); ctx.fill();
  // shrine silhouette perched on the eastern ridge
  const shx = VIEW_W * 0.76;
  ctx.fillStyle = "#0d0a18";
  ctx.fillRect(shx - 11, VIEW_H * 0.52, 22, VIEW_H * 0.13);
  ctx.beginPath();
  ctx.moveTo(shx - 24, VIEW_H * 0.52);
  ctx.lineTo(shx, VIEW_H * 0.455);
  ctx.lineTo(shx + 24, VIEW_H * 0.52);
  ctx.closePath(); ctx.fill();
  // a warm window in the dark
  ctx.fillStyle = "rgba(255,210,122,0.85)";
  ctx.fillRect(shx - 2, VIEW_H * 0.565, 4, 7);
  // drifting embers
  for (let i = 0; i < 20; i++) {
    const yy = (t * 18 + i * 60) % (VIEW_H + 40);
    const xx = (i * 137 + Math.sin(t + i) * 30) % VIEW_W;
    ctx.globalAlpha = 0.35;
    ctx.fillStyle = "#ffd24c";
    ctx.fillRect(xx, VIEW_H - yy, 2, 2);
  }
  ctx.globalAlpha = 1;

  // logo
  ctx.save();
  ctx.textAlign = "center";
  ctx.font = "900 74px 'Segoe UI', system-ui, sans-serif";
  const lg = ctx.createLinearGradient(0, 120, 0, 190);
  lg.addColorStop(0, "#fff6dd"); lg.addColorStop(0.5, "#ffd24c"); lg.addColorStop(1, "#c9703a");
  ctx.shadowColor = "rgba(255,180,80,0.7)"; ctx.shadowBlur = 26;
  ctx.fillStyle = lg;
  ctx.fillText("AETHERFALL", VIEW_W / 2, 176);
  ctx.shadowBlur = 0;
  ctx.font = "600 19px 'Segoe UI', system-ui, sans-serif";
  ctx.fillStyle = "#e8d9a0";
  ctx.fillText("— Requiem of the Dawn —", VIEW_W / 2, 208);
  ctx.restore();

  // menu
  const opts = ["New Journey"];
  if (Save.has()) opts.push("Continue");
  opts.forEach((o, i) => {
    const y = 330 + i * 44;
    const active = titleSel === i;
    if (active) {
      ctx.fillStyle = "rgba(255,226,122,0.12)";
      roundRect(ctx, VIEW_W / 2 - 130, y - 24, 260, 36, 8); ctx.fill();
    }
    txt(ctx, (active ? "▶ " : "") + o, VIEW_W / 2, y, active ? 20 : 17, active ? "#ffe27a" : "#9aa5c0", "center", active ? 800 : 600);
  });
  txt(ctx, "WASD move · J attack · K dash · E talk · 1/2/3 skills · gamepad ready", VIEW_W / 2, VIEW_H - 40, 13, "rgba(200,210,230,0.55)", "center");
  txt(ctx, "a lightemulator garden production — no assets, all code", VIEW_W / 2, VIEW_H - 18, 11, "rgba(160,170,200,0.4)", "center");
}

function tickTitle(dt) {
  const ax = UI.axis(dt);
  const opts = Save.has() ? 2 : 1;
  if (ax.y) { titleSel = (titleSel + ax.y + opts) % opts; AudioSys.sfx("ui"); }
  if (UI.confirm()) {
    AudioSys.unlock();
    AudioSys.sfx("chest");
    if (titleSel === 1 && Save.has()) {
      const d = Save.read();
      loadSaveData(d);
      G.state = "play";
      toast("Journey resumed.");
    } else {
      Save.clear();
      newGame();
    }
  }
}

/* ---------- death / victory / ending ---------- */
function drawDead(t) {
  drawWorld(t);
  ctx.fillStyle = `rgba(30,4,8,${clamp(G.deadT * 0.9, 0, 0.82)})`;
  ctx.fillRect(0, 0, VIEW_W, VIEW_H);
  if (G.deadT > 0.6) {
    txtGlow(ctx, "YOU FELL", VIEW_W / 2, 240, 54, "#ff8a8a", "rgba(255,60,60,0.6)");
    txt(ctx, "The dawn is not done with you yet.", VIEW_W / 2, 282, 16, "#c9a0a0", "center");
    if (G.deadT > 1.4 && Math.floor(t * 2) % 2 === 0)
      txt(ctx, "Press Enter — return to the village (lose ¼ gold)", VIEW_W / 2, 340, 15, "#ffd24c", "center", 700);
  }
}
function tickDead(dt) {
  if (G.deadT > 1.4 && UI.confirm()) {
    const p = G.player;
    G.gold = Math.floor(G.gold * 0.75);
    const st = pstats();
    p.hp = st.maxhp; p.mp = st.maxmp;
    p.invulnT = 2;
    G.state = "play";
    AudioSys.setTheme("village");
    loadMap("village", 20, 25);
    dialogPush([["", "You wake by the fountain. The village is quiet. (Lost ¼ of your gold.)"]]);
  }
}

const ENDING_LINES = [
  "The shard-shards sing as one, and the Sun Crystal remembers its shape.",
  "Across the land, the guardians close their eyes — freed, at last, from borrowed madness.",
  "In Dawnspire Village, a child braids moonpetals and tells anyone who listens about the knight who chased the dark.",
  "And every morning, just before the sun clears the mountains,",
  "the light arrives a little earlier than it used to.",
];

function drawEnding(t) {
  ctx.fillStyle = "#05060e";
  ctx.fillRect(0, 0, VIEW_W, VIEW_H);
  const e = G.endingT;
  if (e < 1.2) {
    ctx.fillStyle = `rgba(255,240,200,${1 - e / 1.2})`;
    ctx.fillRect(0, 0, VIEW_W, VIEW_H);
    return;
  }
  const lt = e - 1.2;
  ENDING_LINES.forEach((l, i) => {
    const a = clamp((lt - i * 1.5) / 0.8, 0, 1);
    if (a > 0) {
      ctx.globalAlpha = a;
      txt(ctx, l, VIEW_W / 2, 150 + i * 52, 17, "#e8d9a0", "center");
    }
  });
  ctx.globalAlpha = 1;
  if (e > 9.5) {
    const a = clamp((e - 9.5) / 1, 0, 1);
    ctx.globalAlpha = a;
    txt(ctx, "Enter to continue", VIEW_W / 2, VIEW_H - 50, 13, "#8a94ad", "center");
    ctx.globalAlpha = 1;
  }
}
function tickEnding(dt) {
  if ((G.endingT > 9.5 && UI.confirm()) || G.endingT > 16) {
    G.state = "victory";
  }
}

function drawVictory(t) {
  const g = ctx.createLinearGradient(0, 0, 0, VIEW_H);
  g.addColorStop(0, "#1a2140"); g.addColorStop(1, "#4a3050");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, VIEW_W, VIEW_H);
  for (let i = 0; i < 30; i++) {
    const yy = (t * 30 + i * 40) % (VIEW_H + 20);
    ctx.globalAlpha = 0.4;
    ctx.fillStyle = "#ffe27a";
    ctx.fillRect((i * 211) % VIEW_W, VIEW_H - yy, 2.5, 2.5);
  }
  ctx.globalAlpha = 1;
  txtGlow(ctx, "THE DAWN RETURNS", VIEW_W / 2, 150, 52, "#ffe8b0", "rgba(255,200,100,0.8)");
  txt(ctx, "AETHERFALL — Requiem of the Dawn", VIEW_W / 2, 190, 16, "#c9b4a0", "center");
  const p = G.player;
  const stats = [
    ["Knight Level", p.lvl], ["Foes felled", G.kills],
    ["Gold amassed", G.gold], ["Journey time", fmtTime(G.playT)],
    ["Moonpetals picked", G.petalsCollectedTotal],
  ];
  stats.forEach((s, i) => {
    txt(ctx, s[0], VIEW_W / 2 - 40, 260 + i * 34, 16, "#9aa5c0", "right");
    txt(ctx, String(s[1]), VIEW_W / 2 + 40, 260 + i * 34, 18, "#ffd24c", "left", 800);
  });
  if (Math.floor(t * 2) % 2 === 0)
    txt(ctx, "Press Enter — back to the title", VIEW_W / 2, VIEW_H - 46, 15, "#e8d9a0", "center", 700);
}
function tickVictory() {
  if (UI.confirm()) { G.state = "title"; titleSel = 0; AudioSys.setTheme("title"); }
}

/* ============================================================
   MAIN LOOP
   ============================================================ */
function tickUI(dt) {
  const t = G.t;
  if (Input.consume("mute")) {
    AudioSys.toggleMute();
    toast(AudioSys.muted ? "Sound off" : "Sound on", 1.2);
  }

  switch (G.state) {
    case "title": tickTitle(dt); break;
    case "dead": tickDead(dt); break;
    case "ending": tickEnding(dt); break;
    case "victory": tickVictory(); break;
    case "play":
      if (G.dialog) { tickDialog(dt); break; }
      if (G.shop) { tickShop(dt); break; }
      if (G.menuOpen) { tickMenu(dt); break; }
      if (Input.consume("menu")) { initMenu(); AudioSys.sfx("ui"); break; }
      if (Input.consume("pause")) { G.paused = !G.paused; AudioSys.sfx("ui"); break; }
      if (!G.paused) gameTick(dt);
      break;
    case "transition":
      gameTick(dt);
      break;
  }
}

function render() {
  const t = G.t;
  if (G.state === "title") { drawTitle(t); return; }
  if (G.state === "victory") { drawVictory(t); return; }
  if (G.state === "ending") { drawEnding(t); return; }
  if (G.state === "dead") { drawDead(t); return; }

  drawWorld(t);
  drawHUD(t);
  if (G.dialog) drawDialog(t);
  if (G.shop) drawShop(t);
  if (G.menuOpen) drawMenu(t);
  if (G.paused) drawPause();

  // transition fade
  if (G.trans) {
    const a = G.trans.phase === 0 ? easeIn(G.trans.t) : 1 - easeOut(G.trans.t);
    ctx.fillStyle = `rgba(3,4,10,${clamp(a, 0, 1)})`;
    ctx.fillRect(0, 0, VIEW_W, VIEW_H);
  }
}

let lastFrame = performance.now();
function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min(0.033, (now - lastFrame) / 1000 || 0.016);
  lastFrame = now;
  Input.pollPad();
  tickUI(dt);
  render();
  Input.endFrame();
}

/* ---------- boot ---------- */
Input.init();
G.state = "title";
AudioSys.setTheme("title");
requestAnimationFrame(frame);
