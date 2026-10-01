// Pin the product language so the suite is deterministic. index.ts detects the
// language from LC_ALL/LC_MESSAGES/LANG/LANGUAGE at load time and then falls back
// on Intl; on a machine whose locale is Italian these English assertions would
// fail. The extension is imported dynamically further down, so setting the
// variables at module load is early enough.
for (const name of ["LC_ALL", "LC_MESSAGES", "LANGUAGE"]) delete process.env[name];
process.env.LANG = "en_US.UTF-8";

/**
 * Helper shared by the tests: finds the Pi package and prepares a sandbox.
 *
 * Why it exists. Until 2026-09-30 the tests imported `typescript` and `typebox`
 * with ABSOLUTE paths:
 *
 *     import ts from '/home/riccardo/.hermes/lsp/node_modules/typescript/lib/typescript.js';
 *
 * It worked on a single machine with a single user name. On cubotto the user
 * is `serverino`, so `/home/riccardo/...` does not exist and NO test ran:
 * the suite was green only where it had been written. Worse, it imported `typescript`
 * to transpile by hand, while Node does it by itself since 22.18 (type
 * stripping is on without a flag), and `index.ts` only uses strippable syntax.
 *
 * Here: zero absolute paths, zero dependency on `typescript`. The Pi package
 * is discovered at runtime from the `pi` binary; with the package declared as a
 * devDependency there would be no need to guess where it is.
 */
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { execFileSync } from 'node:child_process';
import { pathToFileURL } from 'node:url';

export const REPO_ROOT = path.resolve(import.meta.dirname, '..');

/**
 * Root of the @earendil-works/pi-coding-agent package.
 *
 * Order: environment variable, then the `pi` binary (the source of truth: it is
 * the installation that actually runs the extension), then `npm root -g`.
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
  } catch { /* pi not in PATH: try something else */ }

  try {
    const root = execFileSync('npm', ['root', '-g'], { encoding: 'utf8' }).trim();
    if (root) candidates.push(path.join(root, '@earendil-works/pi-coding-agent'));
  } catch { /* npm missing */ }

  // Last resort: known layouts, with no hardcoded user names.
  candidates.push(
    path.join(os.homedir(), '.hermes', 'node', 'lib', 'node_modules', '@earendil-works', 'pi-coding-agent'),
    path.join(os.homedir(), '.npm-global', 'lib', 'node_modules', '@earendil-works', 'pi-coding-agent'),
  );

  for (const c of candidates) {
    if (c && fs.existsSync(path.join(c, 'package.json'))) return c;
  }
  throw new Error(
    'package @earendil-works/pi-coding-agent not found. ' +
    'Set PI_PACKAGE_DIR=/path/to/the/package or make sure `pi` is in the PATH.',
  );
}

/**
 * Links the Pi dependencies inside the repo's node_modules/.
 *
 * `node_modules/` is gitignored: they are derived links, not sources. It is needed
 * because `typebox` lives nested inside the Pi package, and Node's module
 * resolution would not find it starting from the repo root.
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
      if (fs.lstatSync(link)) continue; // already present
    } catch { /* does not exist: create it */ }
    fs.mkdirSync(path.dirname(link), { recursive: true });
    fs.symlinkSync(target, link, 'dir');
    created.push(rel);
  }
  return { pi, created };
}

/** Verifies that this Node can import a .ts module without a transpiler. */
export function assertTypeStrippingWorks() {
  const [major, minor] = process.versions.node.split('.').map(Number);
  const ok = major > 22 || (major === 22 && minor >= 18);
  if (!ok) {
    throw new Error(
      `Node ${process.versions.node} does not import .ts files without a transpiler: ` +
      'Node >= 22.18 is required (type stripping on by default).',
    );
  }
}

/**
 * Prepares a temporary HOME with an importable copy of the extension.
 *
 * The copy is needed where a SECOND independent instance of the module is
 * required (the persistence test simulates a restart): Node caches modules per
 * URL, so importing the same file twice would give the same object.
 *
 * `node_modules/typebox` is linked inside the sandbox too: starting from
 * /tmp the resolution would not find Pi's nested package.
 */
export function makeSandbox({ name = 'cwl', config = null } = {}) {
  assertTypeStrippingWorks();
  const repoLinks = ensureRepoNodeModules();
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), `cwl-${name}-`));

  for (const f of ['index.ts', 'recall.mjs', 'recall.d.mts', 'config.json']) {
    const src = path.join(REPO_ROOT, f);
    if (fs.existsSync(src)) fs.copyFileSync(src, path.join(dir, f));
  }

  // typebox reachable from the sandbox.
  const sbNm = path.join(dir, 'node_modules');
  fs.mkdirSync(path.join(sbNm, '@earendil-works'), { recursive: true });
  const tb = path.join(repoLinks.pi, 'node_modules', 'typebox');
  if (fs.existsSync(tb)) fs.symlinkSync(tb, path.join(sbNm, 'typebox'), 'dir');
  const core = path.join(repoLinks.pi, 'node_modules', '@earendil-works', 'pi-agent-core');
  if (fs.existsSync(core)) {
    fs.symlinkSync(core, path.join(sbNm, '@earendil-works', 'pi-agent-core'), 'dir');
  }

  // The config passed in goes into the sandbox HOME: it is the path loadConfig reads.
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
 * Instantiates the extension in a sandbox and returns the registered tools and hooks,
 * as Pi's runtime sees them.
 */
export async function bootExtension(sandbox, { name = 'ext' } = {}) {
  // A different name per file avoids the module cache when a second independent
  // instance is needed (restart simulation).
  const copy = path.join(sandbox.dir, `${name}.index.ts`);
  if (name !== 'ext') {
    fs.copyFileSync(path.join(sandbox.dir, 'index.ts'), copy);
  }
  const url = name === 'ext' ? sandbox.indexUrl : pathToFileURL(copy).href;

  const { default: extension } = await import(url);
  const tools = new Map();
  const hooks = new Map();
  const commands = new Map();
  const notes = [];
  const pi = {
    on(event, handler) { hooks.set(event, handler); },
    registerTool(tool) { tools.set(tool.name, tool); },
    registerCommand(name, definition) { commands.set(name, definition); },
    sendMessage(msg) { notes.push(msg); },
  };
  extension(pi);
  return { tools, hooks, notes, commands };
}

/**
 * Session context: stable identity provided by the transcript.
 *
 * `buildContextEntries` is what Pi builds the context list from, and it is the only way
 * a tool has to review the messages: `cwl_compress` uses it to check that an
 * interval chosen BY HAND does not sit inside an already existing leaf. The test fills it with
 * `ctx.__show(list)`; the list is the one the hook receives, because extension
 * injections are not session entries.
 */
export const sessionCtx = (sessionFile, cwd = '/tmp/progetto') => {
  const entries = [];
  const ctx = {
    cwd,
    hasUI: false,
    sessionManager: {
      getSessionFile: () => sessionFile,
      getSessionId: () => 'unused-when-the-file-exists',
      buildContextEntries: () => entries,
    },
  };
  ctx.__show = (messages) => {
    entries.length = 0;
    for (const m of messages) entries.push({ type: 'message', message: m });
  };
  return ctx;
};

/** Context without any identifier: it is the case that collapsed onto "default". */
export const anonymousCtx = (cwd = '/tmp/progetto') => ({ cwd, hasUI: false });

/** Runs cwl_status and returns text and details (the numbers are locale-independent). */
export async function status(tools, ctx) {
  const res = await tools.get('cwl_status').execute('id', {}, undefined, undefined, ctx);
  return { text: res.content.map((c) => c.text).join('\n'), details: res.details };
}

/**
 * Redirects HOME temporarily for the duration of the test.
 * Usage: `const home = withHome(sandbox.dir); ... home.restore();`
 */
export function withHome(dir) {
  const previous = process.env.HOME;
  process.env.HOME = dir;
  return {
    restore: () => { process.env.HOME = previous; },
  };
}

// ---------------------------------------------------------------------------
// CLI entry point: used by check.sh to link the dependencies before the tests.
// Without this, `node tests/_helpers.mjs --link-deps` would be a silent no-op
// (the import runs nothing) and the gate would look green without doing anything.
// ---------------------------------------------------------------------------
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const args = process.argv.slice(2);
  if (args.includes('--link-deps')) {
    try {
      assertTypeStrippingWorks();
      const { pi, created } = ensureRepoNodeModules();
      console.log(`[deps] Pi package: ${pi}`);
      console.log(`[deps] link: ${created.length ? created.join(', ') : 'already in place'}`);
      process.exit(0);
    } catch (err) {
      console.error(`[deps] ${err instanceof Error ? err.message : String(err)}`);
      process.exit(1);
    }
  }
  console.log('usage: node tests/_helpers.mjs --link-deps');
  process.exit(0);
}
