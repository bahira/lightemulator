"use strict";
/* ============================================================
   AETHERFALL — content.js
   tilesets · world maps · items · enemies · bosses · quests
   ============================================================ */

const TS = 32; // tile size (px)

/* Tile legend
   #  wall / tree / rock (solid, themed)     .  plain ground
   ,  grass                                   f  flower
   *  moonpetal node (gather)                 W  water (solid)
   ~  lava (solid, glow)                     R  road
   s  ash / sand                             S  stone floor
   c  crystal (solid, glow)                  h  house wall
   r  roof                                   d  door
   F  fence                                  b  bridge
   T  torch / pillar (solid, glow)          O  fountain (solid)
*/
const SOLID_TILES = new Set(["#", "W", "~", "h", "r", "d", "F", "c", "T", "O"]);

function buildGrid(w, h, fill, feats, seed) {
  const rng = mulberry32(seed || 1337);
  const g = Array.from({ length: h }, () => Array(w).fill(fill));
  const inB = (x, y) => x >= 0 && y >= 0 && x < w && y < h;
  for (const f of feats) {
    if (f.rect) {
      const [x, y, fw, fh, ch] = f.rect;
      for (let j = 0; j < fh; j++) for (let i = 0; i < fw; i++)
        if (inB(x + i, y + j) && (!f.over || f.over.includes(g[y + j][x + i]))) g[y + j][x + i] = ch;
    } else if (f.disc) {
      const [cx, cy, r, ch] = f.disc;
      for (let j = -r; j <= r; j++) for (let i = -r; i <= r; i++) {
        if (i * i + j * j <= r * r && inB(cx + i, cy + j) && (!f.over || f.over.includes(g[cy + j][cx + i])))
          g[cy + j][cx + i] = ch;
      }
    } else if (f.scatter) {
      const [n, ch, over] = f.scatter;
      let placed = 0, tries = n * 12;
      while (placed < n && tries-- > 0) {
        const x = 1 + Math.floor(rng() * (w - 2));
        const y = 1 + Math.floor(rng() * (h - 2));
        if (over.includes(g[y][x])) { g[y][x] = ch; placed++; }
      }
    } else if (f.pts) {
      for (const [x, y, ch] of f.pts) if (inB(x, y)) g[y][x] = ch;
    }
  }
  return g;
}

function house(x, y) {
  return [
    { rect: [x, y, 5, 2, "r"] },
    { rect: [x, y + 2, 5, 1, "h"] },
    { pts: [[x + 2, y + 2, "d"]] },
  ];
}

/* ---------------- MAPS ---------------- */
const MAPS = {
  village: {
    name: "Dawnspire Village", theme: "village", music: "village", w: 40, h: 30, seed: 101,
    fill: ",",
    feats: [
      { scatter: [16, "#", ",."] }, { scatter: [30, "f", ","] },
      { rect: [0, 0, 40, 1, "#"] }, { rect: [0, 29, 40, 1, "#"] },
      { rect: [0, 0, 1, 30, "#"] }, { rect: [39, 0, 1, 30, "#"] },
      { rect: [6, 12, 28, 2, "R"] }, { rect: [19, 6, 2, 23, "R"] },
      { rect: [14, 9, 12, 8, "R"] },
      ...house(6, 4), ...house(29, 4), ...house(6, 19), ...house(29, 19),
      { rect: [19, 12, 2, 2, "O"] },
      { pts: [[19, 29, "R"], [20, 29, "R"]] },
    ],
    portals: [
      { x: 19, y: 28, w: 2, h: 2, to: "meadow", tx: 28, ty: 3, label: "Sunfall Meadow" },
    ],
    npcs: [
      { id: "elder", x: 22, y: 11, name: "Elder Maro", col: "#e8d9a0" },
      { id: "petra", x: 8, y: 8, name: "Petra", col: "#f2a6c0" },
      { id: "tillo", x: 13, y: 23, name: "Tillo", col: "#8ad47a" },
      { id: "traveler", x: 31, y: 23, name: "Wandering Nomad", col: "#a0c4e8" },
    ],
    chests: [],
    spawns: [],
    ambient: "#00000000",
  },

  meadow: {
    name: "Sunfall Meadow", theme: "meadow", music: "field", w: 56, h: 44, seed: 202,
    fill: ".",
    feats: [
      { scatter: [46, "#", "."] }, { scatter: [70, ",", "."] }, { scatter: [44, "f", ".,"] },
      { rect: [0, 0, 56, 1, "#"] }, { rect: [0, 43, 56, 1, "#"] },
      { rect: [0, 0, 1, 44, "#"] }, { rect: [55, 0, 1, 44, "#"] },
      { rect: [36, 0, 3, 44, "W"] },
      { rect: [36, 21, 3, 1, "b"] },
      { disc: [12, 34, 4, "W"] },
      { rect: [27, 1, 2, 21, "R"] },
      { rect: [14, 21, 41, 2, "R"] },
      { rect: [14, 21, 2, 22, "R"] },
      { rect: [46, 2, 2, 20, "R"] },
      { rect: [27, 23, 2, 20, "R"] },
      { pts: [[10, 12, "*"], [22, 8, "*"], [33, 27, "*"], [8, 25, "*"], [24, 33, "*"], [44, 16, "*"]] },
    ],
    portals: [
      { x: 27, y: 1, w: 2, h: 1, to: "village", tx: 19, ty: 27, label: "Dawnspire Village" },
      { x: 54, y: 20, w: 1, h: 3, to: "forest", tx: 2, ty: 21, label: "Whisperwood" },
      { x: 14, y: 42, w: 2, h: 1, to: "cave", tx: 24, ty: 32, label: "Hollowdeep" },
      { x: 52, y: 40, w: 2, h: 2, to: "ember", tx: 28, ty: 40, label: "Emberfall Ridge" },
      { x: 46, y: 2, w: 2, h: 2, to: "shrine", tx: 22, ty: 39, label: "Shrine of Dawn", needs: { shards: 3 }, lockMsg: "A seal of three lights bars the way. (Recover the 3 Dawn Shards.)" },
    ],
    npcs: [
      { id: "brennan", x: 44, y: 10, name: "Guard Brennan", col: "#c4c9d4" },
    ],
    chests: [
      { id: "meadow_chest", x: 48, y: 30, items: [{ id: "potion", n: 2 }], gold: 60 },
    ],
    spawns: [
      { e: "slime", x: 10, y: 15 }, { e: "slime", x: 18, y: 18 }, { e: "slime", x: 24, y: 12 },
      { e: "slime", x: 16, y: 28 }, { e: "slime", x: 30, y: 30 }, { e: "slime", x: 42, y: 30 },
      { e: "slime", x: 48, y: 24 },
      { e: "bee", x: 20, y: 16 }, { e: "bee", x: 32, y: 12 }, { e: "bee", x: 44, y: 10 },
    ],
    ambient: "rgba(255,244,180,0.05)",
  },

  forest: {
    name: "Whisperwood", theme: "forest", music: "forest", w: 56, h: 44, seed: 303,
    fill: ",",
    feats: [
      { scatter: [170, "#", ","] }, { scatter: [40, "f", ","] },
      { rect: [0, 0, 56, 1, "#"] }, { rect: [0, 43, 56, 1, "#"] },
      { rect: [0, 0, 1, 44, "#"] }, { rect: [55, 0, 1, 44, "#"] },
      { rect: [1, 20, 20, 2, "R"] },
      { rect: [19, 10, 2, 12, "R"] },
      { rect: [19, 10, 20, 2, "R"] },
      { rect: [37, 4, 2, 8, "R"] },
      { disc: [38, 6, 6, "."] },
      { rect: [30, 26, 22, 2, "W"] },
      { disc: [10, 34, 3, "W"] },
      { pts: [[10, 26, "*"], [26, 30, "*"], [44, 34, "*"]] },
    ],
    portals: [
      { x: 1, y: 20, w: 1, h: 2, to: "meadow", tx: 53, ty: 21, label: "Sunfall Meadow" },
    ],
    npcs: [],
    chests: [
      { id: "forest_chest", x: 48, y: 38, items: [{ id: "steel_claymore", n: 1 }, { id: "hipotion", n: 1 }], gold: 90 },
    ],
    boss: { id: "guardian", x: 38, y: 5, flag: "boss_forest" },
    spawns: [
      { e: "bat", x: 10, y: 15 }, { e: "bat", x: 14, y: 25 }, { e: "bat", x: 30, y: 28 },
      { e: "bat", x: 44, y: 20 }, { e: "bat", x: 25, y: 18 },
      { e: "wolf", x: 22, y: 14 }, { e: "wolf", x: 34, y: 16 }, { e: "wolf", x: 12, y: 30 }, { e: "wolf", x: 42, y: 24 },
      { e: "archer", x: 20, y: 12 }, { e: "archer", x: 36, y: 12 },
    ],
    ambient: "rgba(20,60,30,0.16)",
  },

  cave: {
    name: "Hollowdeep", theme: "cave", music: "cave", w: 48, h: 36, seed: 404,
    fill: "S",
    feats: [
      { scatter: [64, "#", "S"] }, { scatter: [18, "c", "S"] }, { scatter: [10, "T", "S"] },
      { rect: [0, 0, 48, 1, "#"] }, { rect: [0, 35, 48, 1, "#"] },
      { rect: [0, 0, 1, 36, "#"] }, { rect: [47, 0, 1, 36, "#"] },
      { disc: [10, 10, 3, "~"] }, { disc: [38, 26, 3, "~"] },
      { disc: [38, 8, 6, "S"] },
      { rect: [22, 8, 3, 28, "R"] },
      { rect: [8, 16, 32, 2, "R"] },
    ],
    portals: [
      { x: 23, y: 34, w: 2, h: 2, to: "meadow", tx: 15, ty: 40, label: "Sunfall Meadow" },
    ],
    npcs: [],
    chests: [
      { id: "cave_chest", x: 6, y: 6, items: [{ id: "guard_ring", n: 1 }, { id: "ether", n: 1 }], gold: 120 },
    ],
    boss: { id: "wraith", x: 38, y: 8, flag: "boss_cave" },
    spawns: [
      { e: "skeleton", x: 12, y: 28 }, { e: "skeleton", x: 20, y: 17 }, { e: "skeleton", x: 30, y: 28 },
      { e: "skeleton", x: 36, y: 16 }, { e: "skeleton", x: 16, y: 10 }, { e: "skeleton", x: 28, y: 12 },
      { e: "necro", x: 10, y: 18 }, { e: "necro", x: 34, y: 24 }, { e: "necro", x: 40, y: 14 },
      { e: "bat", x: 22, y: 26 }, { e: "bat", x: 26, y: 6 },
    ],
    ambient: "rgba(8,6,24,0.42)",
  },

  ember: {
    name: "Emberfall Ridge", theme: "ember", music: "ember", w: 56, h: 44, seed: 505,
    fill: "s",
    feats: [
      { scatter: [44, "#", "s"] }, { scatter: [30, ".", "s"] },
      { rect: [0, 0, 56, 1, "#"] }, { rect: [0, 43, 56, 1, "#"] },
      { rect: [0, 0, 1, 44, "#"] }, { rect: [55, 0, 1, 44, "#"] },
      { rect: [0, 20, 56, 3, "~"] },
      { rect: [27, 20, 2, 3, "b"] },
      { disc: [12, 10, 3, "~"] }, { disc: [45, 33, 4, "~"] },
      { rect: [27, 23, 2, 20, "R"] },
      { rect: [27, 6, 2, 14, "R"] },
      { rect: [8, 30, 40, 2, "R"] },
      { disc: [28, 8, 6, "s"] },
    ],
    portals: [
      { x: 27, y: 42, w: 2, h: 2, to: "meadow", tx: 50, ty: 41, label: "Sunfall Meadow" },
    ],
    npcs: [],
    chests: [
      { id: "ember_chest", x: 50, y: 12, items: [{ id: "flamebrand", n: 1 }, { id: "elixir", n: 1 }], gold: 160 },
    ],
    boss: { id: "tyrant", x: 28, y: 7, flag: "boss_ember" },
    spawns: [
      { e: "imp", x: 12, y: 14 }, { e: "imp", x: 40, y: 12 }, { e: "imp", x: 16, y: 30 },
      { e: "imp", x: 36, y: 34 }, { e: "imp", x: 46, y: 26 },
      { e: "golem", x: 20, y: 12 }, { e: "golem", x: 32, y: 36 }, { e: "golem", x: 10, y: 38 },
    ],
    ambient: "rgba(120,30,0,0.16)",
  },

  shrine: {
    name: "Shrine of Dawn", theme: "shrine", music: "shrine", w: 44, h: 44, seed: 606,
    fill: "S",
    feats: [
      { scatter: [26, "#", "S"] },
      { rect: [0, 0, 44, 1, "#"] }, { rect: [0, 43, 44, 1, "#"] },
      { rect: [0, 0, 1, 44, "#"] }, { rect: [43, 0, 1, 44, "#"] },
      { rect: [21, 14, 2, 29, "R"] },
      { disc: [22, 10, 7, "S"] },
      { pts: [[17, 20, "T"], [26, 20, "T"], [17, 26, "T"], [26, 26, "T"], [17, 32, "T"], [26, 32, "T"], [17, 38, "T"], [26, 38, "T"]] },
      { rect: [10, 16, 24, 1, "#"] },
      { rect: [21, 16, 2, 1, "S"] },
    ],
    portals: [
      { x: 21, y: 42, w: 2, h: 2, to: "meadow", tx: 47, ty: 4, label: "Sunfall Meadow" },
    ],
    npcs: [],
    chests: [
      { id: "shrine_chest_w", x: 9, y: 30, items: [{ id: "dawn_saber", n: 1 }], gold: 0 },
      { id: "shrine_chest_e", x: 34, y: 30, items: [{ id: "dawn_mail", n: 1 }, { id: "elixir", n: 1 }], gold: 200 },
    ],
    boss: { id: "serpent", x: 22, y: 9, flag: "boss_final" },
    spawns: [
      { e: "skeleton", x: 18, y: 34 }, { e: "skeleton", x: 26, y: 34 },
      { e: "skeleton", x: 18, y: 24 }, { e: "skeleton", x: 26, y: 24 },
      { e: "necro", x: 14, y: 20 }, { e: "necro", x: 30, y: 20 },
    ],
    ambient: "rgba(40,30,80,0.28)",
  },
};

/* ---------------- ITEMS ---------------- */
const ITEMS = {
  potion:    { name: "Potion", type: "use", heal: 50, price: 25, col: "#ff6b81", desc: "Restores 50 HP." },
  hipotion:  { name: "Hi-Potion", type: "use", heal: 140, price: 70, col: "#ff3860", desc: "Restores 140 HP." },
  ether:     { name: "Ether", type: "use", mp: 30, price: 55, col: "#4ca9ff", desc: "Restores 30 MP." },
  elixir:    { name: "Elixir", type: "use", heal: 9999, mp: 9999, price: 400, col: "#ffd24c", desc: "Fully restores HP and MP." },

  trainee_sword:  { name: "Trainee Sword", type: "weapon", atk: 3, price: 40, desc: "A squire's first blade." },
  iron_sword:     { name: "Iron Sword", type: "weapon", atk: 7, price: 130, desc: "Reliable village-forged steel." },
  steel_claymore: { name: "Steel Claymore", type: "weapon", atk: 12, price: 320, desc: "Heavy blade of a fallen knight." },
  flamebrand:     { name: "Flamebrand", type: "weapon", atk: 18, price: 700, desc: "Burns with the ember of the ridge." },
  dawn_saber:     { name: "Dawn Saber", type: "weapon", atk: 26, luck: 4, price: 0, desc: "Forged from the first light." },

  cloth_tunic:  { name: "Cloth Tunic", type: "armor", def: 2, price: 30, desc: "Better than nothing." },
  leather_vest: { name: "Leather Vest", type: "armor", def: 4, price: 110, desc: "Hardened boar leather." },
  chainmail:    { name: "Chainmail", type: "armor", def: 7, price: 280, desc: "Interlocked steel rings." },
  knight_plate: { name: "Knight Plate", type: "armor", def: 11, price: 620, desc: "Plate of the Lumina Order." },
  dawn_mail:    { name: "Dawn Mail", type: "armor", def: 16, hp: 30, price: 0, desc: "Blessed by the dawn itself." },

  lucky_charm:   { name: "Lucky Charm", type: "acc", luck: 8, price: 0, desc: "A moonpetal braid. Greatly raises crit." },
  swift_boots:   { name: "Swift Boots", type: "acc", spd: 45, price: 240, desc: "Raises move speed." },
  power_band:    { name: "Power Band", type: "acc", atk: 5, price: 300, desc: "Raises attack." },
  guard_ring:    { name: "Guardian Ring", type: "acc", def: 4, hp: 25, price: 0, desc: "Raises DEF and Max HP." },
  sun_medallion: { name: "Sun Medallion", type: "acc", atk: 3, def: 3, luck: 5, price: 0, desc: "Warmth of the Sun Crystal." },

  moonpetal:  { name: "Moonpetal", type: "key", price: 0, col: "#bfe6ff", desc: "A pale flower that blooms in starlight." },
  shard_dawn: { name: "Dawn Shard", type: "key", price: 0, col: "#ffe27a", desc: "A fragment of the Sun Crystal." },
};

const SHOPS = {
  village: ["potion", "hipotion", "ether", "elixir", "iron_sword", "leather_vest", "chainmail", "knight_plate", "swift_boots", "power_band"],
};

/* ---------------- ENEMIES ---------------- */
const ENEMIES = {
  slime:    { name: "Moss Slime", hp: 26, atk: 7, def: 0, spd: 62, xp: 9, gold: [2, 6], r: 11, dmg: 8, ai: "chase", sight: 175, col: "#6fd44c" },
  bee:      { name: "Sting Bee", hp: 18, atk: 8, def: 0, spd: 125, xp: 11, gold: [2, 5], r: 8, dmg: 8, ai: "flutter", sight: 210, col: "#f2c94c" },
  bat:      { name: "Gloom Bat", hp: 22, atk: 9, def: 1, spd: 135, xp: 13, gold: [3, 7], r: 9, dmg: 9, ai: "flutter", sight: 230, col: "#9b6fd4" },
  wolf:     { name: "Dire Wolf", hp: 42, atk: 12, def: 2, spd: 155, xp: 20, gold: [5, 10], r: 13, dmg: 12, ai: "charger", sight: 240, col: "#a8b4c4" },
  archer:   { name: "Hollow Archer", hp: 34, atk: 11, def: 1, spd: 90, xp: 22, gold: [6, 12], r: 11, dmg: 11, ai: "ranged", sight: 270, proj: "arrow", col: "#c4a86f" },
  skeleton: { name: "Restless Skeleton", hp: 54, atk: 14, def: 3, spd: 96, xp: 28, gold: [8, 14], r: 12, dmg: 13, ai: "chase", sight: 205, col: "#e8e6da" },
  necro:    { name: "Necromancer", hp: 46, atk: 15, def: 2, spd: 82, xp: 36, gold: [10, 18], r: 11, dmg: 14, ai: "ranged", sight: 290, proj: "shadow", col: "#7b5ea7" },
  imp:      { name: "Ember Imp", hp: 50, atk: 16, def: 3, spd: 112, xp: 42, gold: [12, 20], r: 10, dmg: 15, ai: "ranged", sight: 270, proj: "fire", col: "#ff8a4c" },
  golem:    { name: "Magma Golem", hp: 115, atk: 20, def: 6, spd: 58, xp: 70, gold: [20, 34], r: 17, dmg: 18, ai: "chase", sight: 195, col: "#d4574c", big: true },
};

/* ---------------- BOSSES ---------------- */
const BOSSES = {
  guardian: { name: "Verdant Guardian", title: "Warden of the First Shard", hp: 640, atk: 16, def: 4, r: 26, xp: 320, gold: [90, 130], shard: 1 },
  wraith:   { name: "Abyssal Wraith", title: "Echo of the Second Shard", hp: 860, atk: 20, def: 5, r: 22, xp: 470, gold: [120, 170], shard: 2 },
  tyrant:   { name: "Magma Tyrant", title: "Furnace of the Third Shard", hp: 1150, atk: 24, def: 7, r: 28, xp: 660, gold: [160, 230], shard: 3 },
  serpent:  { name: "Twilight Serpent", title: "Thief of the Dawn", hp: 1700, atk: 28, def: 8, r: 26, xp: 999, gold: [500, 500], shard: 0 },
};

/* ---------------- QUESTS ---------------- */
const QUEST_INFO = {
  main: { name: "The Shattered Dawn", icon: "◆" },
  pests: { name: "Meadow Pests", icon: "✿" },
  petals: { name: "Moonpetal Request", icon: "❀" },
};

/* ---------------- DIALOGUE ---------------- */
const DLG = {
  elder_intro: [
    ["Elder Maro", "Kael! Thank the light you're unhurt. Last night... the Sun Crystal shattered."],
    ["Elder Maro", "Three shards were scattered across the land — each seized by a guardian twisted by its power."],
    ["Elder Maro", "The Verdant Guardian holds one in the Whisperwood, east of the meadow. The others lie deeper still."],
    ["Elder Maro", "Recover all three Dawn Shards and the seal on the Shrine of Dawn will break. There, the thief awaits."],
    ["Elder Maro", "You were the Order's finest squire. Become its finest knight. Go, with the dawn's blessing."],
  ],
  elder_progress: [
    ["Elder Maro", "The shards sing when they are near each other... how many do you carry, Kael?"],
    ["Elder Maro", "The guardians were once protectors of this land. Free them from the shard's madness."],
  ],
  elder_shards3: [
    ["Elder Maro", "All three shards! Then the seal on the Shrine of Dawn is broken."],
    ["Elder Maro", "The thief waits at the top of the shrine, north-east of the meadow. End this, Kael."],
  ],
  elder_after: [
    ["Elder Maro", "You carry the dawn itself in your pocket, Kael. The village will sing of this."],
  ],
  petra_hello: [
    ["Petra", "Welcome, welcome! Potions for the brave, steel for the braver."],
  ],
  tillo_intro: [
    ["Tillo", "Whoa, a real knight! Hey, hey — will you get me six Moonpetals?"],
    ["Tillo", "They glow all pale-blue in the meadow and the woods. I'll braid you a lucky charm!"],
  ],
  tillo_progress: [
    ["Tillo", "Moonpetals glow pale-blue... you need six! You've got {petals} so far."],
  ],
  tillo_done: [
    ["Tillo", "Six! They're beautiful! Here — I braided it myself. It hums when danger's near... sometimes."],
  ],
  tillo_after: [
    ["Tillo", "When I grow up I'll fight the twilight too! Watch me train!"],
  ],
  brennan_locked: [
    ["Guard Brennan", "Halt. The Shrine of Dawn is sealed by three lights — only the Dawn Shards break it."],
    ["Guard Brennan", "The guardians in the Whisperwood, Hollowdeep and Emberfall Ridge each hold one. Tread carefully."],
  ],
  brennan_open: [
    ["Guard Brennan", "The seal has shattered... the shrine road is open. Whatever stole the dawn waits up there."],
    ["Guard Brennan", "Go, Knight of Dawnspire. We'll hold the village behind you."],
  ],
  traveler: [
    ["Wandering Nomad", "They say the Sun Crystal chose this village because the first light ever fell here."],
    ["Wandering Nomad", "In Hollowdeep, the crystals remember songs. In Emberfall, the stones still burn from the old war."],
    ["Wandering Nomad", "Roll under an enemy's strike, friend. Even darkness cannot hit what is not there."],
  ],
  shrine_sealed: [
    ["", "A seal of three lights bars the way. (Recover the 3 Dawn Shards.)"],
  ],
};

/* parse helper — returns { w, h, grid } from a map def */
function parseMapDef(def) {
  const grid = buildGrid(def.w, def.h, def.fill, def.feats, def.seed);
  return { w: def.w, h: def.h, grid };
}
function isSolidChar(ch) { return SOLID_TILES.has(ch); }
