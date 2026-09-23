#!/usr/bin/env python3
"""Référence INDÉPENDANTE (numpy, float64) du modèle SPEAR-T1.

Le même graphe que lm_cpu/ti_model.h (RMSNorm → QKV → attention causale softmax
→ Wo → résidu → RMSNorm → fc1 → ReLU² → fc2 → résidu → norme finale → Wout) est
réécrit ici, sans partager une ligne de code avec l'implémentation C. Le script
lit un vidage produit par ti_dump.c et compare :

  1. la perte (attendu : accord ~1e-6, l'implémentation étant en float32) ;
  2. le gradient complet, tenseur par tenseur (attendu : accord ~1e-3) ;
  3. des différences finies centrées calculées en float64 avec des perturbations
     exactes (aucune limite de précision) — l'arbitre en cas de désaccord.

Usage : python3 tools/ti_reference.py <vidage.txt>
"""
import sys
import numpy as np

NOMS = ["emb", "pos", "g1", "g2", "gf", "wqkv", "wo", "w1", "w2", "wout"]


# ----------------------------------------------------------------- lecture
def lire(chemin):
    """Lecture tolérante : les sections sont remplies jeton par jeton."""
    with open(chemin) as f:
        lignes = [l.split() for l in f]
    d, i = {}, 0
    while i < len(lignes):
        t = lignes[i]
        if not t:
            i += 1
            continue
        if t[0] == "geom":
            d["T"], d["D"], d["L"], d["H"], d["HD"], d["F"], d["V"], d["B"] = map(int, t[1:])
            i += 1
            continue
        if t[0] == "loss":
            d["loss"] = float(t[1])
            i += 1
            continue
        # sections numériques : ids/tgt (B*T valeurs) ou par/grd <nom> <n> (n valeurs)
        if t[0] in ("ids", "tgt"):
            cle, besoin = t[0], d["B"] * d["T"]
        else:
            cle, besoin = (t[0], t[1]), int(t[2])
        vals, j = [], i + 1
        while len(vals) < besoin and j < len(lignes):
            vals.extend(lignes[j])
            j += 1
        if len(vals) < besoin:
            raise SystemExit(f"vidage tronqué : section {cle} ({len(vals)}/{besoin})")
        vals = vals[:besoin]
        d[cle] = (np.array([int(x) for x in vals]) if cle in ("ids", "tgt")
                  else np.array([float(x) for x in vals]))
        i = j
    return d


# ---------------------------------------------------- vues sur les paramètres
def vues(d):
    T, D, L, H, HD, F, V, B = (d[k] for k in ("T", "D", "L", "H", "HD", "F", "V", "B"))
    p = lambda nom: d[("par", nom)]
    return dict(
        emb=p("emb").reshape(V, D), pos=p("pos").reshape(T, D),
        g1=p("g1").reshape(L, D), g2=p("g2").reshape(L, D), gf=p("gf"),
        wqkv=p("wqkv").reshape(L, 3 * D, D), wo=p("wo").reshape(L, D, D),
        w1=p("w1").reshape(L, F, D), w2=p("w2").reshape(L, D, F),
        wout=p("wout").reshape(V, D),
    )


# ------------------------------------------------------------------ forward
def forward(P, ids, tgt, g):
    T, D, L, H, HD, F, V, B = (g[k] for k in ("T", "D", "L", "H", "HD", "F", "V", "B"))
    NT, n = B * T, T
    x = np.zeros((L + 1, NT, D))
    for b in range(B):
        for t in range(n):
            x[0, b * T + t] = P["emb"][ids[b * T + t]] + P["pos"][t]
    cache = {}
    for l in range(L):
        inv1 = 1.0 / np.sqrt((x[l] ** 2).mean(-1, keepdims=True) + 1e-6)
        n1 = x[l] * inv1 * P["g1"][l]
        qkv = n1 @ P["wqkv"][l].T
        pr = np.zeros((B, H, T, T))
        o = np.zeros((NT, D))
        for b in range(B):
            for h in range(H):
                for t in range(n):
                    q = qkv[b * T + t, h * HD:(h + 1) * HD]
                    for u in range(t + 1):
                        k = qkv[b * T + u, D + h * HD:D + (h + 1) * HD]
                        pr[b, h, t, u] = (q @ k) / np.sqrt(HD)
                    s = pr[b, h, t, :t + 1]
                    e = np.exp(s - s.max())
                    pr[b, h, t, :t + 1] = e / e.sum()
                    for u in range(t + 1):
                        o[b * T + t, h * HD:(h + 1) * HD] += pr[b, h, t, u] * \
                            qkv[b * T + u, 2 * D + h * HD:2 * D + (h + 1) * HD]
        x[l + 1] = x[l] + o @ P["wo"][l].T
        xpre = x[l + 1].copy()          # entrée réelle de la norme 2 (avant le FFN)
        inv2 = 1.0 / np.sqrt((xpre ** 2).mean(-1, keepdims=True) + 1e-6)
        n2 = xpre * inv2 * P["g2"][l]
        f1 = n2 @ P["w1"][l].T
        fsq = np.where(f1 > 0, f1 * f1, 0.0)
        x[l + 1] = x[l + 1] + fsq @ P["w2"][l].T
        cache[l] = dict(inv1=inv1, n1=n1, qkv=qkv, pr=pr, o=o, inv2=inv2, n2=n2, f1=f1, fsq=fsq, xpre=xpre)
    invf = 1.0 / np.sqrt((x[L] ** 2).mean(-1, keepdims=True) + 1e-6)
    nf = x[L] * invf * P["gf"]
    logits = nf @ P["wout"].T
    mx = logits.max(-1, keepdims=True)
    lse = mx[:, 0] + np.log(np.exp(logits - mx).sum(-1))
    perte = float(np.mean(lse - logits[np.arange(NT), tgt]))
    dlogits = np.exp(logits - mx) / np.exp(logits - mx).sum(-1, keepdims=True) / NT
    dlogits[np.arange(NT), tgt] -= 1.0 / NT
    cache["invf"], cache["nf"], cache["x"] = invf, nf, x
    return perte, dlogits, cache


def norm_bwd(dy, xv, gain, inv):
    D = xv.shape[-1]
    sxy = (dy * xv * gain).sum(-1, keepdims=True)
    c = inv ** 3 * sxy / D
    dx = dy * gain * inv - c * xv
    dg = (dy * xv * inv).sum(0)
    return dx, dg


def backward(P, ids, tgt, g, dlogits, cache):
    T, D, L, H, HD, F, V, B = (g[k] for k in ("T", "D", "L", "H", "HD", "F", "V", "B"))
    NT = B * T
    G = {k: np.zeros_like(v) for k, v in P.items()}
    G["wout"] += dlogits.T @ cache["nf"]
    dnf = dlogits @ P["wout"]
    dcur, dgf = norm_bwd(dnf, cache["x"][L], P["gf"], cache["invf"])
    G["gf"] += dgf
    for l in range(L - 1, -1, -1):
        c = cache[l]
        # --- FFN d'abord : il partage l'état intermédiaire avec l'attention ---
        G["w2"][l] += dcur.T @ c["fsq"]
        dfsq = dcur @ P["w2"][l]
        df1 = np.where(c["f1"] > 0, 2 * c["f1"] * dfsq, 0.0)
        G["w1"][l] += df1.T @ c["n2"]
        dn2 = df1 @ P["w1"][l]
        dxn, dg2 = norm_bwd(dn2, c["xpre"], P["g2"][l], c["inv2"])
        G["g2"][l] += dg2
        # gradient TOTAL de l'état intermédiaire Xint = X[l] + ATTN
        dx = dcur + dxn
        # --- attention, alimentée par le gradient de Xint ---
        G["wo"][l] += dx.T @ c["o"]
        do = dx @ P["wo"][l]
        NT = B * T
        dqkv = np.zeros((NT, 3 * D))
        for b in range(B):
            for h in range(H):
                for t in range(T):
                    i = b * T + t
                    p = c["pr"][b, h, t, :t + 1]
                    dP = np.array([do[i, h * HD:(h + 1) * HD] @
                                   c["qkv"][b * T + u, 2 * D + h * HD:2 * D + (h + 1) * HD]
                                   for u in range(t + 1)])
                    ds = p * (dP - (p * dP).sum()) / np.sqrt(HD)
                    for u in range(t + 1):
                        dqkv[i, h * HD:(h + 1) * HD] += ds[u] * \
                            c["qkv"][b * T + u, D + h * HD:D + (h + 1) * HD]
                        dqkv[b * T + u, D + h * HD:D + (h + 1) * HD] += ds[u] * \
                            c["qkv"][i, h * HD:(h + 1) * HD]
                        dqkv[b * T + u, 2 * D + h * HD:2 * D + (h + 1) * HD] += p[u] * \
                            do[i, h * HD:(h + 1) * HD]
        G["wqkv"][l] += dqkv.T @ c["n1"]
        dn1 = dqkv @ P["wqkv"][l]
        dx1, dg1 = norm_bwd(dn1, cache["x"][l], P["g1"][l], c["inv1"])
        G["g1"][l] += dg1
        dcur = dx + dx1
    for b in range(B):
        for t in range(T):
            i = b * T + t
            G["pos"][t] += dcur[i]
            G["emb"][ids[i]] += dcur[i]
    return G


def main():
    d = lire(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ti_dump.txt")
    P = vues(d)
    ids, tgt = d["ids"], d["tgt"]
    perte, dlogits, cache = forward(P, ids, tgt, d)
    G = backward(P, ids, tgt, d, dlogits, cache)
    print(f"== référence numpy float64 (T={d['T']} D={d['D']} L={d['L']} F={d['F']} V={d['V']}) ==")
    print(f"  perte  C (float32) : {d['loss']:.8f}")
    print(f"  perte  numpy (f64) : {perte:.8f}   écart relatif {abs(perte - d['loss']) / abs(perte):.2e}")
    gmax = 0.0
    for nom in NOMS:
        a = G[nom].ravel()
        b = d[("grd", nom)]
        rel = np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-30)
        gmax = max(gmax, rel)
        print(f"  grad {nom:<5}: ||Δg||/||g|| = {rel:.3e}   (||g||={np.linalg.norm(a):.4e})")
    print(f"  pire tenseur : {gmax:.3e}")
    # différences finies en float64 (arbitre)
    print("  -- différences finies float64 (perturbations exactes, h=1e-5) --")
    for nom, idx in [("g1", (0, 3)), ("g2", (0, 5)), ("wqkv", (0, 2, 1)), ("wo", (0, 1, 4)),
                     ("w1", (0, 3, 2)), ("w2", (0, 2, 6)), ("wout", (2, 3)), ("pos", (1, 5)),
                     ("emb", (3, 2)), ("gf", (4,))]:
        h = 1e-5
        base = P[nom][idx]
        Pa = {k: v.copy() for k, v in P.items()}
        Pa[nom][idx] = base + h
        la = forward(Pa, ids, tgt, d)[0]
        Pb = {k: v.copy() for k, v in P.items()}
        Pb[nom][idx] = base - h
        lb = forward(Pb, ids, tgt, d)[0]
        fd = (la - lb) / (2 * h)
        an = G[nom][idx]
        an_c = d[("grd", nom)][np.ravel_multi_index(idx, G[nom].shape)]
        rel = abs(fd - an) / (abs(fd) + abs(an) + 1e-30)
        relc = abs(fd - an_c) / (abs(fd) + abs(an_c) + 1e-30)
        print(f"     {nom}{idx}: DF={fd: .6e}  numpy={an: .6e} (rel {rel:.1e})  "
              f"C={an_c: .6e} (rel {relc:.1e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
