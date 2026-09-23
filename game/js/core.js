"use strict";
/* ============================================================
   AETHERFALL — core.js
   utils · input · camera · particles · save · procedural audio
   ============================================================ */

/* ---------- math / misc ---------- */
const clamp = (v, a, b) => v < a ? a : v > b ? b : v;
const lerp  = (a, b, t) => a + (b - a) * t;
const rand  = (a = 1, b) => b === undefined ? Math.random() * a : a + Math.random() * (b - a);
const randi = (a, b) => Math.floor(rand(a, b + 1));
const irand = (a, b) => a + Math.floor(Math.random() * (b - a + 1));
const dist  = (ax, ay, bx, by) => Math.hypot(bx - ax, by - ay);
const angleTo = (ax, ay, bx, by) => Math.atan2(by - ay, bx - ax);
const chance = p => Math.random() < p;
const pick = arr => arr[Math.floor(Math.random() * arr.length)];
const TAU = Math.PI * 2;
const easeOut = t => 1 - Math.pow(1 - clamp(t, 0, 1), 3);
const easeIn  = t => Math.pow(clamp(t, 0, 1), 3);

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ---------- Input (keyboard + gamepad) ---------- */
const ACTIONS = {
  attack:  ["KeyJ", "KeyZ"],
  dodge:   ["KeyK", "KeyX", "ShiftLeft", "ShiftRight"],
  interact:["KeyE", "Enter"],
  menu:    ["KeyI"],
  pause:   ["Escape", "KeyP"],
  sk1:     ["Digit1"],
  sk2:     ["Digit2"],
  sk3:     ["Digit3"],
  mute:    ["KeyM"],
};
const PREVENT = new Set(["Space", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Tab"]);

const Input = {
  held: new Set(),
  pressedQueue: new Set(),
  move: { x: 0, y: 0 },
  padPressed: new Set(),
  padHeld: new Set(),
  lastPad: [],
  padActive: false,

  init() {
    window.addEventListener("keydown", e => {
      if (PREVENT.has(e.code)) e.preventDefault();
      if (!e.repeat) {
        this.held.add(e.code);
        this.pressedQueue.add(e.code);
        AudioSys.unlock();
      }
    });
    window.addEventListener("keyup", e => this.held.delete(e.code));
    window.addEventListener("blur", () => this.held.clear());
    window.addEventListener("pointerdown", () => AudioSys.unlock());
  },

  pollPad() {
    if (!navigator.getGamepads) return;
    const gp = Array.from(navigator.getGamepads()).find(g => g && g.connected);
    if (!gp) { this.padActive = false; return; }
    this.padActive = true;
    // axes -> movement
    let ax = gp.axes[0] || 0, ay = gp.axes[1] || 0;
    if (Math.abs(ax) < 0.22) ax = 0;
    if (Math.abs(ay) < 0.22) ay = 0;
    this.padMove = { x: clamp(ax * 1.25, -1, 1), y: clamp(ay * 1.25, -1, 1) };
    // dpad
    if (gp.buttons[14] && gp.buttons[14].pressed) this.padMove.x = -1;
    if (gp.buttons[15] && gp.buttons[15].pressed) this.padMove.x = 1;
    if (gp.buttons[12] && gp.buttons[12].pressed) this.padMove.y = -1;
    if (gp.buttons[13] && gp.buttons[13].pressed) this.padMove.y = 1;
    // buttons: 0=A attack, 1=B dodge, 2=X interact, 3=Y sk1, 4=LB sk2, 5=RB sk3, 9=start pause
    const map = { 0: "attack", 1: "dodge", 2: "interact", 3: "sk1", 4: "sk2", 5: "sk3", 9: "pause" };
    const heldNow = new Set();
    for (const [idx, act] of Object.entries(map)) {
      const b = gp.buttons[+idx];
      if (b && b.pressed) {
        heldNow.add(act);
        if (!this.padHeld.has(act)) this.padPressed.add(act);
      }
    }
    this.padHeld = heldNow;
  },

  axis() {
    let x = 0, y = 0;
    if (this.held.has("KeyA") || this.held.has("ArrowLeft"))  x -= 1;
    if (this.held.has("KeyD") || this.held.has("ArrowRight")) x += 1;
    if (this.held.has("KeyW") || this.held.has("ArrowUp"))    y -= 1;
    if (this.held.has("KeyS") || this.held.has("ArrowDown"))  y += 1;
    if (this.padMove) { x += this.padMove.x; y += this.padMove.y; }
    const m = Math.hypot(x, y);
    if (m > 1) { x /= m; y /= m; }
    return { x, y };
  },

  consume(action) {
    let fired = false;
    for (const code of ACTIONS[action] || []) {
      if (this.pressedQueue.has(code)) { fired = true; this.pressedQueue.delete(code); }
    }
    if (this.padPressed.has(action)) { fired = true; this.padPressed.delete(action); }
    return fired;
  },

  endFrame() { this.pressedQueue.clear(); this.padPressed.clear(); },
};

/* ---------- Camera ---------- */
class Camera {
  constructor() { this.x = 0; this.y = 0; this.shake = 0; this.sx = 0; this.sy = 0; }
  snap(px, py, mapW, mapH, vw, vh) {
    this.x = clamp(px - vw / 2, 0, Math.max(0, mapW - vw));
    this.y = clamp(py - vh / 2, 0, Math.max(0, mapH - vh));
  }
  update(dt, px, py, mapW, mapH, vw, vh) {
    const tx = clamp(px - vw / 2, 0, Math.max(0, mapW - vw));
    const ty = clamp(py - vh / 2, 0, Math.max(0, mapH - vh));
    const k = 1 - Math.pow(0.0015, dt);
    this.x = lerp(this.x, tx, k);
    this.y = lerp(this.y, ty, k);
    this.shake = Math.max(0, this.shake - dt * 26);
    const s = this.shake;
    this.sx = rand(-s, s);
    this.sy = rand(-s, s);
  }
  addShake(v) { this.shake = Math.min(14, Math.max(this.shake, v)); }
}

/* ---------- Particles & floating text ---------- */
const Particles = {
  list: [],
  cap: 420,
  spawn(p) {
    if (this.list.length >= this.cap) this.list.shift();
    this.list.push(Object.assign({ t: 0, life: 0.6, vx: 0, vy: 0, g: 0, size: 3, col: "#fff", kind: "spark", drag: 0 }, p));
  },
  burst(x, y, col, n = 10, spd = 140, life = 0.5, size = 3) {
    for (let i = 0; i < n; i++) {
      const a = rand(TAU), s = rand(spd * 0.25, spd);
      this.spawn({ x, y, vx: Math.cos(a) * s, vy: Math.sin(a) * s, life: rand(life * 0.5, life), col, size: rand(size * 0.6, size * 1.4), g: 60 });
    }
  },
  ring(x, y, col, n = 22, r0 = 6, spd = 220) {
    for (let i = 0; i < n; i++) {
      const a = (i / n) * TAU;
      this.spawn({ x: x + Math.cos(a) * r0, y: y + Math.sin(a) * r0, vx: Math.cos(a) * spd, vy: Math.sin(a) * spd, life: 0.45, col, size: 3, drag: 3 });
    }
  },
  update(dt) {
    const L = this.list;
    for (let i = L.length - 1; i >= 0; i--) {
      const p = L[i];
      p.t += dt;
      if (p.t >= p.life) { L.splice(i, 1); continue; }
      if (p.drag) { p.vx -= p.vx * p.drag * dt; p.vy -= p.vy * p.drag * dt; }
      p.vy += p.g * dt;
      p.x += p.vx * dt; p.y += p.vy * dt;
    }
  },
  draw(ctx, camX, camY) {
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    for (const p of this.list) {
      const a = 1 - p.t / p.life;
      ctx.globalAlpha = a * 0.95;
      ctx.fillStyle = p.col;
      const s = p.size * (0.5 + a * 0.7);
      ctx.beginPath();
      ctx.arc(p.x - camX, p.y - camY, s, 0, TAU);
      ctx.fill();
    }
    ctx.restore();
  },
};

const FloatTexts = {
  list: [],
  add(x, y, txt, col = "#fff", size = 15, life = 0.9) {
    if (this.list.length > 60) this.list.shift();
    this.list.push({ x, y, txt, col, size, life, t: 0 });
  },
  update(dt) {
    for (let i = this.list.length - 1; i >= 0; i--) {
      const f = this.list[i];
      f.t += dt; f.y -= 34 * dt;
      if (f.t >= f.life) this.list.splice(i, 1);
    }
  },
  draw(ctx, camX, camY) {
    ctx.save();
    ctx.textAlign = "center";
    for (const f of this.list) {
      const a = 1 - Math.pow(f.t / f.life, 2);
      ctx.globalAlpha = a;
      ctx.font = `800 ${f.size}px "Segoe UI", system-ui, sans-serif`;
      ctx.strokeStyle = "rgba(0,0,0,.75)";
      ctx.lineWidth = 3;
      ctx.strokeText(f.txt, f.x - camX, f.y - camY);
      ctx.fillStyle = f.col;
      ctx.fillText(f.txt, f.x - camX, f.y - camY);
    }
    ctx.restore();
  },
};

/* ---------- Save (safe in sandboxed iframes) ---------- */
const Save = {
  key: "aetherfall_save_v1",
  mem: null,
  write(obj) {
    const s = JSON.stringify(obj);
    this.mem = s;
    try { localStorage.setItem(this.key, s); return true; } catch (e) { return false; }
  },
  read() {
    try {
      const s = localStorage.getItem(this.key);
      if (s) return JSON.parse(s);
    } catch (e) { /* fall through */ }
    if (this.mem) { try { return JSON.parse(this.mem); } catch (e) { /* noop */ } }
    return null;
  },
  has() { return this.read() !== null; },
  clear() {
    this.mem = null;
    try { localStorage.removeItem(this.key); } catch (e) { /* noop */ }
  },
};

/* ---------- Procedural audio: WebAudio music + SFX ---------- */
const AudioSys = {
  ctx: null, master: null, musicGain: null, sfxGain: null,
  muted: false, unlockedOnce: false,
  noiseBuf: null,
  music: { theme: null, step: 0, nextT: 0, timer: null, seed: 1 },

  unlock() {
    if (!this.ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return;
      this.ctx = new AC();
      this.master = this.ctx.createGain();
      this.master.gain.value = 0.6;
      this.master.connect(this.ctx.destination);
      this.musicGain = this.ctx.createGain();
      this.musicGain.gain.value = 0.34;
      this.musicGain.connect(this.master);
      this.sfxGain = this.ctx.createGain();
      this.sfxGain.gain.value = 0.8;
      this.sfxGain.connect(this.master);
      const len = this.ctx.sampleRate * 0.5;
      this.noiseBuf = this.ctx.createBuffer(1, len, this.ctx.sampleRate);
      const d = this.noiseBuf.getChannelData(0);
      for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
    }
    if (this.ctx.state === "suspended") this.ctx.resume();
  },

  toggleMute() {
    this.muted = !this.muted;
    if (this.master) this.master.gain.value = this.muted ? 0 : 0.6;
    return this.muted;
  },

  tone(freq, dur, type = "square", vol = 0.2, slide = 0, delay = 0, dest) {
    if (!this.ctx) return;
    const t0 = this.ctx.currentTime + delay;
    const o = this.ctx.createOscillator();
    const g = this.ctx.createGain();
    o.type = type;
    o.frequency.setValueAtTime(freq, t0);
    if (slide) o.frequency.exponentialRampToValueAtTime(Math.max(20, freq + slide), t0 + dur);
    g.gain.setValueAtTime(0, t0);
    g.gain.linearRampToValueAtTime(vol, t0 + 0.008);
    g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
    o.connect(g); g.connect(dest || this.sfxGain);
    o.start(t0); o.stop(t0 + dur + 0.02);
  },

  noise(dur, vol = 0.2, fFrom = 3000, fTo = 300, q = 1, delay = 0, type = "bandpass") {
    if (!this.ctx) return;
    const t0 = this.ctx.currentTime + delay;
    const src = this.ctx.createBufferSource();
    src.buffer = this.noiseBuf; src.loop = true;
    const f = this.ctx.createBiquadFilter();
    f.type = type; f.Q.value = q;
    f.frequency.setValueAtTime(fFrom, t0);
    f.frequency.exponentialRampToValueAtTime(Math.max(30, fTo), t0 + dur);
    const g = this.ctx.createGain();
    g.gain.setValueAtTime(vol, t0);
    g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
    src.connect(f); f.connect(g); g.connect(this.sfxGain);
    src.start(t0); src.stop(t0 + dur + 0.02);
  },

  sfx(name) {
    if (!this.ctx) return;
    const T = this;
    switch (name) {
      case "sword":  T.noise(0.09, 0.25, 3400, 700, 2); break;
      case "hit":    T.tone(190, 0.07, "square", 0.22, -120); T.noise(0.05, 0.16, 1800, 500); break;
      case "crit":   T.tone(320, 0.09, "square", 0.26, -200); T.tone(640, 0.08, "square", 0.14, -300, 0.02); T.noise(0.08, 0.2, 2600, 400); break;
      case "hurt":   T.tone(280, 0.2, "sawtooth", 0.24, -200); T.noise(0.12, 0.14, 900, 200); break;
      case "dash":   T.noise(0.16, 0.14, 500, 2400, 1, 0, "bandpass"); break;
      case "pickup": T.tone(660, 0.07, "sine", 0.2, 330); break;
      case "gold":   T.tone(1318, 0.05, "square", 0.12); T.tone(1760, 0.07, "square", 0.12, 0, 0.05); break;
      case "heart":  T.tone(523, 0.09, "sine", 0.2, 200); break;
      case "level":  [523, 659, 784, 1046].forEach((f, i) => T.tone(f, 0.14, "square", 0.16, 0, i * 0.09)); break;
      case "heal":   T.tone(440, 0.3, "sine", 0.16, 330); T.tone(660, 0.3, "sine", 0.1, 330, 0.1); break;
      case "nova":   T.tone(110, 0.45, "sawtooth", 0.3, -70); T.noise(0.4, 0.22, 700, 90); break;
      case "lance":  T.tone(880, 0.16, "square", 0.18, -620); break;
      case "fire":   T.noise(0.22, 0.18, 1400, 240, 1.4); T.tone(220, 0.2, "sawtooth", 0.1, -140); break;
      case "chest":  [392, 523, 659, 784].forEach((f, i) => T.tone(f, 0.16, "triangle", 0.18, 0, i * 0.08)); break;
      case "buy":    T.tone(988, 0.06, "square", 0.14); T.tone(1319, 0.09, "square", 0.14, 0, 0.06); break;
      case "deny":   T.tone(196, 0.12, "square", 0.16, -60); break;
      case "ui":     T.tone(880, 0.035, "square", 0.07); break;
      case "talk":   T.tone(620, 0.03, "square", 0.06); break;
      case "death":  T.tone(220, 0.8, "sawtooth", 0.24, -185); T.noise(0.6, 0.14, 500, 60); break;
      case "boss":   T.tone(98, 0.7, "sawtooth", 0.26); T.tone(147, 0.7, "sawtooth", 0.2, 0, 0.05); T.noise(0.5, 0.12, 300, 70); break;
      case "shard":  [784, 988, 1175, 1568].forEach((f, i) => T.tone(f, 0.22, "sine", 0.16, 0, i * 0.11)); break;
      case "roar":   T.tone(70, 0.9, "sawtooth", 0.3, -30); T.noise(0.8, 0.2, 350, 60, 0.8); break;
      case "step":   break;
    }
  },

  /* --- generative chiptune music --- */
  setTheme(name) {
    if (this.music.theme === name) return;
    this.music.theme = name;
    this.music.step = 0;
    this.music.seed = 1 + Math.floor(rand(9999));
    if (this.ctx) this.music.nextT = this.ctx.currentTime + 0.06;
  },

  scheduler() {
    if (!this.ctx || !this.music.theme) return;
    const th = MUSIC_THEMES[this.music.theme];
    if (!th) return;
    const spb = 60 / th.bpm / 2; // eighth-note
    while (this.music.nextT < this.ctx.currentTime + 0.18) {
      this.scheduleStep(this.music.step, this.music.nextT, th);
      this.music.nextT += spb;
      this.music.step++;
    }
  },

  scheduleStep(step, t, th) {
    const bar = Math.floor(step / 8) % th.prog.length;
    const root = th.prog[bar];
    const inBar = step % 8;
    const rng = mulberry32(this.music.seed * 7919 + step);
    const midi2f = m => 440 * Math.pow(2, (m - 69) / 12);
    // bass on every beat
    if (inBar % 2 === 0) {
      const f = midi2f(root - 12);
      this.mus(f, 0.19, th.bassType || "triangle", 0.16, t);
    }
    // lead
    if (rng() < th.density) {
      const scale = [0, 3, 5, 7, 10, 12, 15];
      const deg = scale[Math.floor(rng() * scale.length)];
      const f = midi2f(root + 12 + deg + (th.leadOct || 0));
      this.mus(f, 0.16, th.leadType || "square", th.leadVol || 0.07, t);
    }
    // hats for driving themes
    if (th.drive && inBar % 2 === 1) this.musHat(t);
    // pad chord at bar start
    if (inBar === 0 && th.pad) {
      [0, 3, 7].forEach(iv => this.mus(midi2f(root + iv), spbPad(th), "sine", 0.035, t));
    }
    function spbPad(th2) { return (60 / th2.bpm) * 4 * 0.9; }
  },

  mus(freq, dur, type, vol, t0) {
    if (!this.ctx) return;
    const o = this.ctx.createOscillator();
    const g = this.ctx.createGain();
    o.type = type; o.frequency.value = freq;
    g.gain.setValueAtTime(0, t0);
    g.gain.linearRampToValueAtTime(vol, t0 + 0.012);
    g.gain.setValueAtTime(vol, t0 + dur * 0.6);
    g.gain.linearRampToValueAtTime(0.0001, t0 + dur);
    o.connect(g); g.connect(this.musicGain);
    o.start(t0); o.stop(t0 + dur + 0.02);
  },

  musHat(t0) {
    if (!this.ctx) return;
    const src = this.ctx.createBufferSource();
    src.buffer = this.noiseBuf;
    const f = this.ctx.createBiquadFilter();
    f.type = "highpass"; f.frequency.value = 6000;
    const g = this.ctx.createGain();
    g.gain.setValueAtTime(0.045, t0);
    g.gain.exponentialRampToValueAtTime(0.0001, t0 + 0.05);
    src.connect(f); f.connect(g); g.connect(this.musicGain);
    src.start(t0); src.stop(t0 + 0.06);
  },
};

const MUSIC_THEMES = {
  title:  { bpm: 84, prog: [45, 50, 57, 52], density: 0.5, leadType: "triangle", leadVol: 0.08, pad: true, drive: false },
  village:{ bpm: 96, prog: [48, 53, 57, 55], density: 0.55, leadType: "square", leadVol: 0.05, pad: true, drive: false },
  field:  { bpm: 122, prog: [50, 48, 53, 55], density: 0.7, leadType: "square", leadVol: 0.065, drive: true },
  forest: { bpm: 104, prog: [47, 50, 45, 43], density: 0.5, leadType: "triangle", leadVol: 0.08, pad: true, drive: false, leadOct: 12 },
  cave:   { bpm: 88, prog: [40, 43, 38, 36], density: 0.4, leadType: "triangle", leadVol: 0.075, pad: true, drive: false },
  ember:  { bpm: 116, prog: [42, 45, 40, 47], density: 0.65, leadType: "sawtooth", leadVol: 0.045, drive: true },
  shrine: { bpm: 100, prog: [45, 41, 48, 50], density: 0.55, leadType: "triangle", leadVol: 0.085, pad: true, drive: false, leadOct: 12 },
  boss:   { bpm: 148, prog: [40, 40, 43, 38], density: 0.85, leadType: "sawtooth", leadVol: 0.055, bassType: "square", drive: true },
  final:  { bpm: 160, prog: [38, 41, 45, 40], density: 0.9, leadType: "sawtooth", leadVol: 0.06, bassType: "square", drive: true },
  victory:{ bpm: 120, prog: [48, 55, 57, 60], density: 0.8, leadType: "square", leadVol: 0.07, drive: true },
};

// keep scheduler alive
setInterval(() => AudioSys.scheduler(), 60);
