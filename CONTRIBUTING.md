# Contributing

Merci de votre intérêt. Ce projet tient par une règle : **chaque chiffre est mesuré, chaque affirmation est vérifiée**.

## La règle d'or

Toute contribution qui ajoute une affirmation numérique doit l'accompagner d'une **référence indépendante** : solution analytique, loi de conservation, énumération exhaustive ou différences finies — voir `src/physics/validate.ts`. Un résultat sans vérification ne sera pas fusionné.

## En pratique — démarrage rapide (première PR en 15 min)

```bash
git clone https://github.com/<ton-fork>/lightemulator && cd lightemulator
npm install
npm run test          # 29 tests grounded — doit être vert
npx tsc --noEmit      # TypeScript strict — doit être vert
npm run dev           # http://localhost:5173
```

Première contribution idéale : une issue `good first issue` (le drag des balises était le prototype — fermé, regarde le diff comme modèle).

## Le workflow complet

1. **Fork + branche** : `git checkout -b feat/mon-truc`
2. **Physique pure** : tout nouveau module dans `src/physics/` doit être TypeScript pur, zéro dépendance — copiez le fichier, il marche.
3. **Test obligatoire** : ajoutez un test dans `src/physics/validate.ts` (groupe + tolérance), il apparaît dans `npm run test` et dans l'onglet Validation.
4. **TypeScript strict** : `npx tsc --noEmit` doit passer.
5. **Boucle grounded verte** : `npm run test` — 29/29 minimum, code de sortie = nombre d'échecs.
6. **PR** : décrivez ce qui est mesuré, pas seulement ce qui est ajouté.

## Les parts honnêtes

Les simplifications délibérées (goulots connus, heuristiques naïves) sont marquées dans le code par des commentaires `ponytail:` nommant le plafond et le chemin d'amélioration. Si votre contribution en ajoute une, marquez-la pareil — c'est de la documentation, pas de la dette cachée.

## Roadmap

Voir les [milestones](https://github.com/bahira/lightemulator/milestones) — v0.2 interactivité, v0.3 performance & portage, v1.0 release sérieuse. Les issues `good first issue` sont un bon point d'entrée.
