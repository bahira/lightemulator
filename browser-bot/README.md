# 🤖 browser-bot — bot d'automatisation navigateur (browser use) avec Pomerado

Ce dossier ajoute au dépôt `lightemulator` un **bot capable d'agir dans un navigateur** grâce à
[Pomerado](https://github.com/Pomerado/pomerado) — l'outil qui transforme n'importe quel site web
en une **intégration MCP déterministe** que votre agent (ou ce bot) peut appeler.

## Ce que fait Pomerado (et ce que ce bot en fait)

Pomerado ne est **pas** un agent "browser use" généraliste qui fait n'importe quoi sur n'importe quel site.
C'est un **bâtisseur d'intégrations par site** :

1. Vous décrivez une tâche en langage naturel (ex. "lis le titre de https://example.com").
2. Pomerado explore le site dans un vrai Chromium, construit l'intégration, et la fait relire par
   **Guardian** (revue des actions et du code généré).
3. L'intégration est sauvegardée sur votre disque comme du code source que vous possédez.
4. Votre agent l'appelle ensuite comme un **serveur MCP** (un outil par intégration, schéma d'entrée/sortie).

Points forts :
- Fonctionne sur des sites **sans API**, même derrière un **login et une 2FA**.
- **Déterministe** : beaucoup plus rapide et fiable que le "computer use" classique.
- L'autorité (`read` / `write`) est choisie délibérément ; une écriture est exécutée une seule fois pendant le mint.

Ce que ce bot (dossier `browser-bot/`) vous apporte :
- Un **runner CLI** (`bot.mjs`) pour créer (mint), suivre et exécuter vos intégrations Pomerado.
- Des **exemples d'intégrations** prêts à l'emploi (`integrations/examples.json`).
- Une **configuration client MCP** exemple (`mcp.example.json`).
- Une commande `doctor` pour vérifier l'environnement.

## 🧱 Prérequis

| Prérequis | Détail |
|-----------|--------|
| **Node.js** | **24.21 ou +**, version 24 (`< 25`) — vérifiez avec `node --version` |
| **Clé OpenAI** | `OPENAI_API_KEY` — Pomerado s'en sert pour construire/revoir les intégrations (votre agent peut tourner sur n'importe quel modèle) |
| **Chromium** | `npx -y -p pomerado playwright install chromium` (ajoutez `--with-deps` sous Linux si des bibliothèques système manquent) |
| **Client MCP** | Claude Code, Cursor, VS Code, Codex, Gemini CLI, etc. |

> ⚠️ **Ce sandbox exécute Node 22.22.3**, et Pomerado exige Node ≥ 24.21. Le code ci-dessous est **prêt à l'emploi mais ne peut pas s'exécuter dans ce sandbox**. Exécutez-le sur une machine avec Node 24.

## 🚀 Démarrage rapide

### 1. Vérifier l'environnement
```bash
cd browser-bot
node bot.mjs doctor
```
Cela affiche la version de Node, l'accès à Pomerado/npx, la reconnaissance de Chromium/Playwright et la présence de `OPENAI_API_KEY`.

### 2. Installer Chromium (une fois)
```bash
npx -y -p pomerado playwright install chromium --with-deps
```

### 3. Créer une intégration (mint)
Exemple — lire le titre principal d'https://example.com :
```bash
export OPENAI_API_KEY="votre_cle"
node bot.mjs mint --name lecteur-exemple --url https://example.com \
  --task "Lis le titre principal (balise <h1>) de la page et renvoie-le." --read
```
Pomerado retourne un `jobId`. Suivez l'avancement :
```bash
node bot.mjs job <jobId>
```
Quand le mint est terminé, Pomerado sauvegarde l'intégration dans `integrations/<name>/` et votre agent peut l'utiliser.

### 4. Utiliser l'intégration (via votre client MCP)
Ajoutez Pomerado à votre client (exemple Cursor / VS Code `.mcp.json`) :
```json
{
  "mcpServers": {
    "pomerado": {
      "command": "npx",
      "args": ["-y", "-p", "pomerado", "pomerado-mcp", "mint", "--root", "/chemin/absolu/vers/lightemulator/browser-bot/integrations"],
      "env": { "OPENAI_API_KEY": "votre_cle_openai" }
    }
  }
}
```
Puis demandez à votre agent :
> Utilise `lecteur-exemple` pour lire le titre principal de la page.

Chaque intégration apparaît comme son propre serveur MCP avec un outil nommé d'après elle, plus `get_job`, `provide_input` et `cancel_job`.

## 📁 Contenu du dossier

```
browser-bot/
├── README.md                # ce fichier
├── bot.mjs                  # runner CLI (mint, job, input, cancel, run, list, doctor)
├── package.json             # métadonnées + scripts
├── mcp.example.json         # exemple de configuration client MCP
└── integrations/
    └── examples.json        # exemples de requêtes de mint (copiez-les dans bot.mjs)
```

Les intégrations créées par le mint sont enregistrées dans `integrations/<nom>/` (configurable avec `--root`) et contiennent :
- `src/` — modules d'opération générés
- `pomerado.json` — point d'entrée, schémas d'entrée/sortie, questions possibles
- `deployment.json` — nom de l'outil, description, URL, tâche, autorité, origines de connexion
- `auth-fill.json` — écrans de connexion (si sign-in), sans les valeurs
- `mcp.mjs` / `mcp.json` — lancement du runtime Pomerado et point d'entrée MCP standard
- `README.md` — commandes d'ajout pour chaque client MCP

## 🛠 Commandes du runner

```
node bot.mjs doctor                                vérifie l'environnement
node bot.mjs mint --name X --url Y --task Z [--read|--write] [--example-input '{}'] [--root DIR]
                                                   crée une intégration (retourne un jobId)
node bot.mjs job <jobId>                           suit l'avancement d'un mint
node bot.mjs input <jobId> <réponse>               répond à une question de Pomerado
node bot.mjs cancel <jobId>                        annule un job en cours
node bot.mjs run <nomIntegration> [inputJson]      lance une intégration construite
node bot.mjs list                                  liste les jobs enregistrés localement
```

## ⚠️ Limites importantes

- **Node 24 requis** : ce sandbox est en Node 22 et ne peut pas télécharger Node 24 (host `nodejs.org` non autorisé). Le bot est écrit, non testé à l'exécution ici.
- **Pomerado n'est pas un "browser use" illimité** : il construit une intégration déterministe **par site et par tâche**. Pour automatiser un nouveau site ou une nouvelle tâche non couverte, il faut un nouveau mint.
- **Clé OpenAI obligatoire** pour le mint (construction de l'intégration). L'exécution de l'intégration par votre agent n'a pas besoin de Pomerado, mais le mint oui.
- L'autorité `write` est exécutée une fois pendant le mint ; Pomerado demande votre accord avant de choisir `write`.

## 🔗 Liens

- Dépôt Pomerado : https://github.com/Pomerado/pomerado
- Site officiel : https://pomerado.ai
- Documentation : https://docs.pomerado.ai
- Discord : https://discord.gg/QCwQFydAym
- Licence Pomerado : MIT

---

Ce bot est ajouté à `bahira/lightemulator` (branche `arena/1a6c4bcf-lightemulator`). Le dépôt principal reste l'émulateur physique + GPT en C ; `browser-bot/` est une extension d'automatisation web indépendante.
