# ⚔️ AETHERFALL — Requiem of the Dawn

A complete **action JRPG** in pure JavaScript — **zero assets, zero dependencies, zero build step**.
Every sprite is drawn procedurally on canvas, every note of music is synthesized with WebAudio,
and the entire world (6 regions, 4 bosses, quests, shops, leveling, gear, save system) lives in
~5,000 lines of hand-written code.

It belongs here naturally: `lightemulator` is about building whole systems from scratch and
measuring everything — this is the same spirit, pointed at a game.

## Play

```bash
cd game
python3 -m http.server 8080     # or any static server
# open http://localhost:8080
```

Or just open `index.html` directly — it works over `file://` too.

## Controls

| Action | Keyboard | Gamepad |
|---|---|---|
| Move | WASD / Arrows | Left stick / D-pad |
| Attack (3-hit combo) | `J` or `Z` | `A` |
| Dash (invincible i-frames) | `K` / `X` / `Shift` | `B` |
| Talk / Open / Gather | `E` or `Enter` | `X` |
| Radiant Nova (AoE, 8 MP) | `1` | `Y` |
| Sun Lance (projectile, 5 MP) | `2` | `LB` |
| Dawn's Grace (heal, 14 MP) | `3` | `RB` |
| Menu / Pause / Mute | `I` / `Esc` / `M` | `Start` |

## The game

The **Sun Crystal** shattered in the night. Three shards were seized by the land's own guardians,
twisted by the power they were meant to protect. You are **Kael**, squire of the Lumina Order —
recover the three **Dawn Shards**, break the seal on the **Shrine of Dawn**, and face the thief of the dawn.

- **6 handcrafted regions** — Dawnspire Village, Sunfall Meadow, Whisperwood, Hollowdeep,
  Emberfall Ridge, Shrine of Dawn — each with its own palette, lighting and music theme.
- **9 enemy archetypes** (chase / flutter / charger / ranged) + **4 multi-phase bosses**
  with telegraphed bullet patterns, summons, meteors and a two-form final boss.
- **Full JRPG progression** — XP & levels, weapons / armor / accessories, consumables,
  shops, chests, crits, stat growth.
- **Quests** — a 3-shard main quest plus side quests (slime culling, moonpetal gathering).
- **Juice** — hit-stop, screen shake, dash after-images, particles, floating damage numbers,
  aim-assist combos, low-HP heartbeat vignette, cave lighting, lava/ember atmosphere.
- **Procedural everything** — generative chiptune per region (boss & victory themes included),
  synthesized SFX, baked tile art from a seeded feature DSL.
- **Save system** — autosaves to `localStorage` on every map change and boss kill
  (falls back to in-memory in sandboxed iframes). Continue from the title screen.

## Verification

The engine ships with headless test harnesses (they run the real game logic in Node with a
stubbed DOM — no browser needed):

```bash
node tools/validate.cjs   # content integrity (portals, spawns, items, shops) + engine smoke test
node tools/balance.cjs    # combat TTK + a bot that dodges/heals beating every boss, end to end
node tools/uitest.cjs     # renders every screen headlessly: title → play → menus → death → victory
```

All three are expected to print PASS.
