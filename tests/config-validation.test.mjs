/**
 * Regressione: una config.json con tipi sbagliati non deve rompere la policy.
 *
 * Perche' esiste: fino al 30/09/2026 `loadConfig` faceva
 * `{ ...DEFAULT_CONFIG, ...user }` senza validare nulla. Con
 * `thresholdRatio: "high"` il trigger diventava `tokenBudget * "high"` = NaN,
 * quindi `currentTokens <= NaN` era sempre falsa e l'estensione evictava a OGNI
 * turno. Con `thresholdRatio: 5` disattivava l'eviction per sempre. Un errore di
 * battitura nella config, nessun crash, nessun avviso: solo una policy sbagliata
 * in silenzio.
 *
 * Il test passa dalla `cwl_status` reale, che stampa budget e soglia: cosi'
 * verifica il comportamento osservabile dell'estensione, non una funzione
 * interna che si potrebbe rinominare.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { pathToFileURL } from 'node:url';
import ts from '/home/riccardo/.hermes/lsp/node_modules/typescript/lib/typescript.js';

const root = path.resolve(import.meta.dirname, '..');
const TYPEBOX = '/home/riccardo/.hermes/node/lib/node_modules/@earendil-works/pi-coding-agent/node_modules/typebox/build/index.mjs';

/** Carica l'estensione con una `~/.pi/cwl/config.json` data e restituisce i tool. */
async function loadWithConfig(config) {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'cwl-config-'));
  const home = process.env.HOME;
  process.env.HOME = temp;
  try {
    const source = fs.readFileSync(path.join(root, 'index.ts'), 'utf8');
    const js = ts
      .transpileModule(source, {
        compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
      })
      .outputText.replace("from 'typebox'", `from '${pathToFileURL(TYPEBOX).href}'`);
    fs.writeFileSync(path.join(temp, 'index.mjs'), js);

    if (config !== null) {
      const dir = path.join(temp, '.pi', 'cwl');
      fs.mkdirSync(dir, { recursive: true });
      fs.writeFileSync(
        path.join(dir, 'config.json'),
        typeof config === 'string' ? config : JSON.stringify(config),
      );
    }

    const { default: extension } = await import(pathToFileURL(path.join(temp, 'index.mjs')).href);
    const tools = new Map();
    const pi = {
      on() {},
      registerTool(tool) { tools.set(tool.name, tool); },
      registerCommand() {},
      sendMessage() {},
    };
    extension(pi);
    return tools;
  } finally {
    process.env.HOME = home;
    // Il temp resta per la lettura successiva; viene rimosso dal chiamante.
    loadWithConfig._last = temp;
  }
}

/** Esegue cwl_status e restituisce il payload tipizzato (testo + details). */
async function status(tools) {
  const ctx = { cwd: '/tmp', sessionManager: { getSessionId: () => 'test-session' } };
  const res = await tools.get('cwl_status').execute('id', {}, undefined, undefined, ctx);
  return {
    text: res.content.map((c) => c.text).join('\n'),
    details: res.details,
  };
}

test('una thresholdRatio non numerica non produce NaN e ricade sul default', async () => {
  const tools = await loadWithConfig({ thresholdRatio: 'high' });
  const { text, details } = await status(tools);
  assert.doesNotMatch(text, /NaN/, 'il budget non deve mai renderizzare NaN');
  // 0.85 e' il default: la soglia renderizzata e' locale-indipendente.
  assert.match(text, /threshold: 85%/, 'una soglia non valida deve ricadere sul default 0.85');
  assert.equal(details.budget, 80000, 'il budget deve restare il default 80k');
  fs.rmSync(loadWithConfig._last, { recursive: true, force: true });
});

test('una thresholdRatio fuori range non disattiva in silenzio l\'eviction', async () => {
  const tools = await loadWithConfig({ thresholdRatio: 5 });
  const { text } = await status(tools);
  assert.match(text, /threshold: 85%/, 'una soglia > 1 deve ricadere sul default');
  fs.rmSync(loadWithConfig._last, { recursive: true, force: true });
});

test('un tokenBudget non numerico ricade sul default', async () => {
  const tools = await loadWithConfig({ tokenBudget: 'molti' });
  const { text, details } = await status(tools);
  assert.doesNotMatch(text, /NaN/);
  assert.equal(details.budget, 80000, "un budget non valido deve ricadere sull'80k di default");
  fs.rmSync(loadWithConfig._last, { recursive: true, force: true });
});

test('una config.json malformata non uccide l\'estensione', async () => {
  const tools = await loadWithConfig('{ questo non e json');
  const { details } = await status(tools);
  assert.equal(details.budget, 80000, 'un JSON rotto deve ricadere sul config bundled o sul default');
  fs.rmSync(loadWithConfig._last, { recursive: true, force: true });
});

test('una config valida continua a essere rispettata', async () => {
  const tools = await loadWithConfig({ tokenBudget: 123456, thresholdRatio: 0.5 });
  const { text, details } = await status(tools);
  assert.equal(details.budget, 123456, 'un budget valido non deve essere scartato');
  assert.match(text, /threshold: 50%/, 'una soglia valida non deve essere scartata');
  fs.rmSync(loadWithConfig._last, { recursive: true, force: true });
});
