/**
 * Helper condiviso dai test: trova il pacchetto di Pi e prepara un sandbox.
 *
 * Perche' esiste. Fino al 30/09/2026 i test importavano `typescript` e `typebox`
 * con path ASSOLUTI:
 *
 *     import ts from '/home/riccardo/.hermes/lsp/node_modules/typescript/lib/typescript.js';
 *
 * Funzionava su una sola macchina con un solo nome utente. Su cubotto l'utente
 * e' `serverino`, quindi `/home/riccardo/...` non esiste e NESSUN test girava:
 * la suite era verde solo dove era stata scritta. Peggio, importava `typescript`
 * per fare a mano il transpile, quando Node lo fa da solo dalla 22.18 (il type
 * stripping e' attivo senza flag), e `index.ts` usa solo sintassi strippabile.
 *
 * Qui: zero path assoluti, zero dipendenza da `typescript`. Il pacchetto di Pi
 * viene scoperto a runtime dal binario `pi`, col pacchetto dichiarato come
 * devDependency non ci sarebbe nemmeno bisogno di indovinare dove sta.
 */
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { execFileSync } from 'node:child_process';
import { pathToFileURL } from 'node:url';

export const REPO_ROOT = path.resolve(import.meta.dirname, '..');

/**
 * Radice del pacchetto @earendil-works/pi-coding-agent.
 *
 * Ordine: variabile d'ambiente, poi il binario `pi` (la fonte di verita': e'
 * l'installazione che esegue davvero l'estensione), poi `npm root -g`.
 */
export function piPackageDir() {
  const override = process.env.PI_PACKAGE_DIR;
  if (override && fs.existsSync(path.join(override, 'package.json'))) return override;

  const candidates = [];
  try {
    const bin = execFileSync('bash', ['-lc', 'command -v pi'], { encoding: 'utf8' }).trim();
    if (bin) {
      // bin e' .../node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js
      const real = fs.realpathSync(bin);
      candidates.push(path.resolve(path.dirname(real), '..', '..', '..'));
    }
  } catch { /* pi non nel PATH: si prova altro */ }

  try {
    const root = execFileSync('npm', ['root', '-g'], { encoding: 'utf8' }).trim();
    if (root) candidates.push(path.join(root, '@earendil-works/pi-coding-agent'));
  } catch { /* npm assente */ }

  // Ultima spiaggia: layout noti, senza nomi utente cablati.
  candidates.push(
    path.join(os.homedir(), '.hermes', 'node', 'lib', 'node_modules', '@earendil-works', 'pi-coding-agent'),
    path.join(os.homedir(), '.npm-global', 'lib', 'node_modules', '@earendil-works', 'pi-coding-agent'),
  );

  for (const c of candidates) {
    if (c && fs.existsSync(path.join(c, 'package.json'))) return c;
  }
  throw new Error(
    'pacchetto @earendil-works/pi-coding-agent non trovato. ' +
    'Imposta PI_PACKAGE_DIR=/path/al/pacchetto oppure assicurati che `pi` sia nel PATH.',
  );
}

/**
 * Collega le dipendenze di Pi dentro node_modules/ del repo.
 *
 * `node_modules/` e' gitignorato: sono link derivati, non sorgenti. Serve perche'
 * `typebox` vive annidato nel pacchetto di Pi, e la risoluzione dei moduli di
 * Node non lo troverebbe partendo dalla radice del repo.
 */
export function ensureRepoNodeModules() {
  const pi = piPackageDir();
  const nm = path.join(REPO_ROOT, 'node_modules');
  fs.mkdirSync(path.join(nm, '@earendil-works'), { recursive: true });

  const links = [
    ['typebox', path.join(pi, 'node_modules', 'typebox')],
    ['@earendil-works/pi-agent-core', path.join(pi, 'node_modules', '@earendil-works', 'pi-agent-core')],
    ['@earendil-works/pi-coding-agent', pi],
  ];
  const created = [];
  for (const [rel, target] of links) {
    const link = path.join(nm, rel);
    if (!fs.existsSync(target)) continue;
    try {
      if (fs.lstatSync(link)) continue; // gia' presente
    } catch { /* non esiste: lo creo */ }
    fs.mkdirSync(path.dirname(link), { recursive: true });
    fs.symlinkSync(target, link, 'dir');
    created.push(rel);
  }
  return { pi, created };
}

/** Verifica che questo Node sappia importare un modulo .ts senza transpiler. */
export function assertTypeStrippingWorks() {
  const [major, minor] = process.versions.node.split('.').map(Number);
  const ok = major > 22 || (major === 22 && minor >= 18);
  if (!ok) {
    throw new Error(
      `Node ${process.versions.node} non importa i .ts senza transpiler: ` +
      'serve Node >= 22.18 (type stripping attivo di default).',
    );
  }
}

/**
 * Prepara una HOME temporanea con una copia importabile dell'estensione.
 *
 * La copia serve dove occorre una SECONDA istanza indipendente del modulo (il
 * test di persistenza simula un riavvio): Node cachea i moduli per URL, quindi
 * importare due volte lo stesso file darebbe lo stesso oggetto.
 *
 * `node_modules/typebox` viene collegato anche dentro il sandbox: partendo da
 * /tmp la risoluzione non troverebbe il pacchetto annidato di Pi.
 */
export function makeSandbox({ name = 'cwl', config = null } = {}) {
  assertTypeStrippingWorks();
  const repoLinks = ensureRepoNodeModules();
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), `cwl-${name}-`));

  for (const f of ['index.ts', 'recall.mjs', 'recall.d.mts', 'config.json']) {
    const src = path.join(REPO_ROOT, f);
    if (fs.existsSync(src)) fs.copyFileSync(src, path.join(dir, f));
  }

  // typebox raggiungibile dal sandbox.
  const sbNm = path.join(dir, 'node_modules');
  fs.mkdirSync(path.join(sbNm, '@earendil-works'), { recursive: true });
  const tb = path.join(repoLinks.pi, 'node_modules', 'typebox');
  if (fs.existsSync(tb)) fs.symlinkSync(tb, path.join(sbNm, 'typebox'), 'dir');
  const core = path.join(repoLinks.pi, 'node_modules', '@earendil-works', 'pi-agent-core');
  if (fs.existsSync(core)) {
    fs.symlinkSync(core, path.join(sbNm, '@earendil-works', 'pi-agent-core'), 'dir');
  }

  // La config passata va nella HOME del sandbox: e' il percorso che loadConfig legge.
  if (config !== null) {
    const cdir = path.join(dir, '.pi', 'cwl');
    fs.mkdirSync(cdir, { recursive: true });
    fs.writeFileSync(
      path.join(cdir, 'config.json'),
      typeof config === 'string' ? config : JSON.stringify(config),
    );
  }

  return {
    dir,
    indexUrl: pathToFileURL(path.join(dir, 'index.ts')).href,
    cleanup: () => fs.rmSync(dir, { recursive: true, force: true }),
  };
}

/**
 * Istanzia l'estensione in un sandbox e restituisce tool e hook registrati,
 * come li vede il runtime di Pi.
 */
export async function bootExtension(sandbox, { name = 'ext' } = {}) {
  // Un nome diverso per file evita la cache dei moduli quando serve una
  // seconda istanza indipendente (simulazione di riavvio).
  const copy = path.join(sandbox.dir, `${name}.index.ts`);
  if (name !== 'ext') {
    fs.copyFileSync(path.join(sandbox.dir, 'index.ts'), copy);
  }
  const url = name === 'ext' ? sandbox.indexUrl : pathToFileURL(copy).href;

  const { default: extension } = await import(url);
  const tools = new Map();
  const hooks = new Map();
  const notes = [];
  const pi = {
    on(event, handler) { hooks.set(event, handler); },
    registerTool(tool) { tools.set(tool.name, tool); },
    registerCommand() {},
    sendMessage(msg) { notes.push(msg); },
  };
  extension(pi);
  return { tools, hooks, notes };
}

/**
 * Contesto di sessione: identita' stabile fornita dal transcript.
 *
 * `buildContextEntries` e' cio' da cui Pi costruisce la lista del contesto, ed e' l'unica via
 * che un tool ha per rivedere i messaggi: `cwl_compress` la usa per controllare che un
 * intervallo scelto A MANO non stia dentro una foglia gia' esistente. Il test la riempie con
 * `ctx.__mostra(lista)`; la lista e' quella che l'hook riceve, perche' le iniezioni delle
 * estensioni non sono voci di sessione.
 */
export const sessionCtx = (sessionFile, cwd = '/tmp/progetto') => {
  const entries = [];
  const ctx = {
    cwd,
    hasUI: false,
    sessionManager: {
      getSessionFile: () => sessionFile,
      getSessionId: () => 'non-usato-quando-c-e-il-file',
      buildContextEntries: () => entries,
    },
  };
  ctx.__mostra = (messaggi) => {
    entries.length = 0;
    for (const m of messaggi) entries.push({ type: 'message', message: m });
  };
  return ctx;
};

/** Contesto senza alcun identificativo: e' il caso che collassava su "default". */
export const anonymousCtx = (cwd = '/tmp/progetto') => ({ cwd, hasUI: false });

/** Esegue cwl_status e restituisce testo e details (i numeri sono locale-indipendenti). */
export async function status(tools, ctx) {
  const res = await tools.get('cwl_status').execute('id', {}, undefined, undefined, ctx);
  return { text: res.content.map((c) => c.text).join('\n'), details: res.details };
}

/**
 * Rileva una HOME temporanea per il tempo del test.
 * Uso: `const home = withHome(sandbox.dir); ... home.restore();`
 */
export function withHome(dir) {
  const previous = process.env.HOME;
  process.env.HOME = dir;
  return {
    restore: () => { process.env.HOME = previous; },
  };
}

// ---------------------------------------------------------------------------
// Entry point CLI: usato da check.sh per collegare le dipendenze prima dei test.
// Senza questo, `node tests/_helpers.mjs --link-deps` sarebbe un no-op silenzioso
// (l'import non esegue nulla) e il gate sembrerebbe verde senza aver fatto nulla.
// ---------------------------------------------------------------------------
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const args = process.argv.slice(2);
  if (args.includes('--link-deps')) {
    try {
      assertTypeStrippingWorks();
      const { pi, created } = ensureRepoNodeModules();
      console.log(`[deps] pacchetto Pi: ${pi}`);
      console.log(`[deps] link: ${created.length ? created.join(', ') : "gia' a posto"}`);
      process.exit(0);
    } catch (err) {
      console.error(`[deps] ${err instanceof Error ? err.message : String(err)}`);
      process.exit(1);
    }
  }
  console.log('uso: node tests/_helpers.mjs --link-deps');
  process.exit(0);
}
