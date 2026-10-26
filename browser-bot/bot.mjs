#!/usr/bin/env node
/**
 * browser-bot/bot.mjs
 * -------------------
 * Runner du "bot d'automatisation navigateur" basé sur Pomerado.
 *
 * Pomerado (https://github.com/Pomerado/pomerado) transforme n'importe quel
 * site web — même sans API, même derrière un login / 2FA — en une intégration
 * MCP déterministe que votre agent (ou ce bot) peut appeler depuis le navigateur.
 *
 * ⚠️  PRÉREQUIS (machine où vous lancez ce bot) :
 *   • Node.js 24.21 ou +, version 24 (< 25)  ← CE SANDBOX EST EN Node 22,
 *                                                le code est prêt mais ne s'exécute PAS ici
 *   • Une clé OPENAI_API_KEY (Pomerado s'en sert pour construire/revoir les intégrations)
 *   • Chromium : npx -y -p pomerado playwright install chromium [--with-deps]
 *   • Un client MCP (Claude Code, Cursor, VS Code, Codex, Gemini CLI …)
 *
 * Ce script enveloppe les commandes Pomerado (mint / get_job / provide_input /
 * cancel_job / run) pour créer et exécuter des intégrations "browser use" depuis la CLI.
 *
 * Les appels exacts sont délégués à `npx -y -p pomerado pomerado-mcp <sous-commande>`.
 * Une fois sur Node 24, faites `npx -y -p pomerado pomerado-mcp --help` pour
 * confirmer les noms de drapeaux, puis ajustez-les dans ce fichier si besoin.
 *
 * Usage :
 *   node bot.mjs doctor
 *   node bot.mjs mint --name lecteur-exemple --url https://example.com \
 *       --task "Lis le titre principal de la page." --read
 *   node bot.mjs job <jobId>
 *   node bot.mjs input <jobId> <réponse>
 *   node bot.mjs cancel <jobId>
 *   node bot.mjs run <nomIntegration> '<{"cle": "valeur"}>'
 *   node bot.mjs list
 */

import { execFileSync, spawn } from 'node:child_process';
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

const ROOT = process.cwd();
const STATE_FILE = join(ROOT, '.pomerado-jobs.json');
const BOT_NAME = 'browser-bot';

let verbose = process.env.BROWSER_BOT_DEBUG === '1';

function log(...args) {
  if (verbose) console.error(`[${BOT_NAME}]`, ...args);
}

/** Charge l'historique des jobs (fichier JSON local, créé au besoin). */
function loadJobs() {
  try {
    if (!existsSync(STATE_FILE)) return {};
    return JSON.parse(readFileSync(STATE_FILE, 'utf8'));
  } catch {
    return {};
  }
}

function saveJobs(jobs) {
  writeFileSync(STATE_FILE, JSON.stringify(jobs, null, 2));
}

/** Exécute une commande Pomerado (via npx) et retourne son stdout. */
function pomerado(args, { stdin } = {}) {
  const cmd = 'npx';
  const npxArgs = ['-y', '-p', 'pomerado', 'pomerado-mcp', ...args];
  log('>', cmd, npxArgs.join(' '));
  try {
    const opts = { encoding: 'utf8', maxBuffer: 1024 * 1024 * 16 };
    if (stdin !== undefined) opts.input = stdin;
    return execFileSync(cmd, npxArgs, opts);
  } catch (err) {
    const out = err.stdout ? err.stdout.toString() : '';
    const errOut = err.stderr ? err.stderr.toString() : '';
    throw new Error(
      `Pomerado a échoué (code ${err.code ?? '?'}).\nSTDOUT:\n${out}\nSTDERR:\n${errOut}`
    );
  }
}

/** Vérifie l'environnement (Node, npx, pomerado, Chromium). */
function cmdDoctor() {
  console.log('=== Docteur du bot browser (Pomerado) ===');
  console.log(`Node       : ${process.version} (requis: >=24.21.0 <25)`);
  const major = Number(process.version.slice(1).split('.')[0]);
  const minor = Number(process.version.slice(1).split('.')[1]);
  const nodeOk = major === 24 && minor >= 21;
  console.log(`Node OK    : ${nodeOk ? 'OUI' : 'NON — installez Node 24.21+'}`);

  try {
    const npv = execFileSync('npm', ['--version'], { encoding: 'utf8' }).trim();
    console.log(`npm        : ${npv}`);
  } catch {
    console.log(`npm        : introuvable`);
  }

  try {
    execFileSync('npx', ['-y', '-p', 'pomerado', 'pomerado-mcp', '--help'], {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe'],
      timeout: 60000,
    });
    console.log(`pomerado   : accessible via npx`);
  } catch (err) {
    console.log(`pomerado   : non accessible ou indisponible (${err.message.slice(0, 120)})`);
  }

  try {
    execFileSync(
      'npx',
      ['-y', '-p', 'pomerado', 'playwright', 'install', '--dry-run', 'chromium'],
      { encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'], timeout: 60000 }
    );
    console.log(`Chromium   : command Pomerado/Playwright reconnue`);
  } catch {
    console.log(`Chromium   : à installer (npx -y -p pomerado playwright install chromium)`);
  }

  console.log(`OPENAI_KEY : ${process.env.OPENAI_API_KEY ? 'définie' : 'NON DÉFINIE (requise pour mint)'}`);
  console.log(`==============================`);
  process.exit(nodeOk ? 0 : 1);
}

/** Crée (mint) une intégration Pomerado et retourne / enregistre le jobId. */
function cmdMint(argv) {
  const opts = parseFlags(argv, {
    name: { type: 'string', required: true },
    url: { type: 'string', required: true },
    task: { type: 'string', required: true },
    read: { type: 'boolean' },
    write: { type: 'boolean' },
    'example-input': { type: 'string' },
    root: { type: 'string', default: join(ROOT, 'integrations') },
  });
  if (!opts) return;

  const authority = opts.write ? 'write' : 'read';
  const args = [
    'mint',
    '--name', opts.name,
    '--url', opts.url,
    '--task', opts.task,
    '--authority', authority,
    '--root', opts.root,
  ];
  if (opts['example-input']) args.push('--example-input', opts['example-input']);

  console.log(`→ Mint de l'intégration "${opts.name}" sur ${opts.url} (${authority})...`);
  const out = pomerado(args);
  console.log(out);

  const jobId = extractJobId(out) || opts.name;
  const jobs = loadJobs();
  jobs[jobId] = {
    id: jobId,
    name: opts.name,
    url: opts.url,
    task: opts.task,
    authority,
    root: opts.root,
    createdAt: new Date().toISOString(),
    status: 'pending',
  };
  saveJobs(jobs);
  console.log(`\nJob enregistré : ${jobId}`);
  console.log(`Suivez avec :  node bot.mjs job ${jobId}`);
}

/** Suivre un job (get_job). */
function cmdJob(argv) {
  const jobId = argv[0];
  if (!jobId) {
    console.error('Usage: node bot.mjs job <jobId>');
    process.exit(2);
  }
  console.log(`→ Suivi du job ${jobId}…`);
  const out = pomerado(['get_job', jobId]);
  console.log(out);

  const jobs = loadJobs();
  if (jobs[jobId]) {
    jobs[jobId].status = 'updated';
    jobs[jobId].updatedAt = new Date().toISOString();
    saveJobs(jobs);
  }
}

/** Fournir une réponse à une question posée par Pomerado. */
function cmdInput(argv) {
  const [jobId, ...answerParts] = argv;
  if (!jobId || answerParts.length === 0) {
    console.error('Usage: node bot.mjs input <jobId> <réponse>');
    process.exit(2);
  }
  const answer = answerParts.join(' ');
  console.log(`→ Envoi de la réponse pour le job ${jobId}…`);
  const out = pomerado(['provide_input', jobId], { stdin: answer });
  console.log(out);
}

/** Annuler un job. */
function cmdCancel(argv) {
  const jobId = argv[0];
  if (!jobId) {
    console.error('Usage: node bot.mjs cancel <jobId>');
    process.exit(2);
  }
  console.log(`→ Annulation du job ${jobId}…`);
  console.log(pomerado(['cancel_job', jobId]));
  const jobs = loadJobs();
  if (jobs[jobId]) {
    jobs[jobId].status = 'cancelled';
    saveJobs(jobs);
  }
}

/** Lancer une intégration déjà construite. */
function cmdRun(argv) {
  const [integrationName, inputJson] = argv;
  if (!integrationName) {
    console.error('Usage: node bot.mjs run <nomIntegration> [\'{"cle":"valeur"}\']');
    process.exit(2);
  }
  const input = inputJson || '{}';
  console.log(`→ Exécution de l'intégration "${integrationName}"…`);
  try {
    // La CLI autonome Pomerado expose `pomerado run` ; fallback sur npx sinon.
    const out = execFileSync(
      'npx',
      ['-y', '-p', 'pomerado', 'pomerado-mcp', 'run', integrationName],
      { encoding: 'utf8', input, maxBuffer: 1024 * 1024 * 16 }
    );
    console.log(out);
  } catch (err) {
    console.error(`Impossible de lancer l'intégration : ${err.message}`);
    console.error(
      'Astuce : ajoutez l\'intégration à votre client MCP (voir mcp.example.json) et appelez-la depuis celle-ci.'
    );
    process.exit(1);
  }
}

/** Liste les jobs enregistrés localement. */
function cmdList() {
  const jobs = loadJobs();
  const ids = Object.keys(jobs);
  if (ids.length === 0) {
    console.log('Aucun job enregistré.');
    return;
  }
  console.log('Jobs enregistrés :');
  for (const id of ids) {
    const j = jobs[id];
    console.log(`  - ${id} | ${j.name} | ${j.url} | ${j.authority} | ${j.status}`);
  }
}

/** Extrait un jobId de la sortie Pomerado (heuristique). */
function extractJobId(text) {
  const m = String(text).match(/job (?:id )?([A-Za-z0-9_.-]+)/i);
  return m ? m[1] : null;
}

/** Analyse les flags simples --cle valeur. */
function parseFlags(argv, spec) {
  const opts = {};
  let i = 0;
  while (i < argv.length) {
    const a = argv[i];
    if (!a.startsWith('--')) {
      console.error(`Argument inattendu : ${a}`);
      return null;
    }
    const key = a.slice(2);
    const s = spec[key];
    if (!s) {
      console.error(`Flag inconnu : --${key}`);
      return null;
    }
    if (s.type === 'boolean') {
      opts[key] = true;
      i++;
    } else {
      const v = argv[++i];
      if (v === undefined) {
        console.error(`Valeur manquante pour --${key}`);
        return null;
      }
      opts[key] = v;
      i++;
    }
  }
  for (const [key, s] of Object.entries(spec)) {
    if (s.required && opts[key] === undefined) {
      console.error(`Flag requis manquant : --${key}`);
      return null;
    }
    if (opts[key] === undefined && s.default !== undefined) opts[key] = s.default;
  }
  return opts;
}

function printHelp() {
  console.log(`
Bot browser use — Pomerado (https://github.com/Pomerado/pomerado)

Usage: node bot.mjs <commande> [arguments]

Commandes:
  doctor                              Vérifie l'environnement (Node 24+, Pomerado, Chromium, clé OpenAI)
  mint --name X --url Y --task Z      Crée une intégration (mint) et retourne un jobId
      [--read|--write]                Autorité : lecture seule (défaut) ou écriture
      [--example-input '{}']          Exemple d'entrée pour guider la construction
      [--root ./integrations]         Dossier où enregistrer l'intégration
  job <jobId>                         Suit l'avancement d'un mint (get_job)
  input <jobId> <réponse>             Répond à une question posée par Pomerado
  cancel <jobId>                      Annule un job en cours
  run <nom> [inputJson]               Lance une intégration déjà construite
  list                                Liste les jobs enregistrés localement

Exemple:
  node bot.mjs mint --name lecteur-exemple --url https://example.com \
      --task "Lis le titre principal (h1) de la page." --read

Variables d'environnement:
  OPENAI_API_KEY  clé requise par Pomerado pour le mint
  BROWSER_BOT_DEBUG=1  affiche les commandes Pomerado exécutées

⚠️  Nécessite Node.js 24.21+. Ce sandbox est en Node 22 : le code est prêt,
    exécutez-le sur une machine disposant de Node 24.
`);
}

const command = process.argv[2];
const argv = process.argv.slice(3);

switch (command) {
  case 'doctor':
    cmdDoctor();
    break;
  case 'mint':
    cmdMint(argv);
    break;
  case 'job':
  case 'status':
    cmdJob(argv);
    break;
  case 'input':
    cmdInput(argv);
    break;
  case 'cancel':
    cmdCancel(argv);
    break;
  case 'run':
    cmdRun(argv);
    break;
  case 'list':
    cmdList();
    break;
  case '--help':
  case '-h':
  case undefined:
    printHelp();
    break;
  default:
    console.error(`Commande inconnue : ${command}`);
    printHelp();
    process.exit(2);
}
