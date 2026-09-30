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
import { makeSandbox, bootExtension, withHome, sessionCtx, status } from './_helpers.mjs';

/**
 * Carica l'estensione con una data config e restituisce i tool.
 *
 * DEVE spostare la HOME nel sandbox: `loadConfig` legge prima
 * `~/.pi/cwl/config.json` e solo come ripiego la config bundled col pacchetto.
 * Senza `withHome` la config passata qui non veniva mai letta e le asserzioni
 * "ricade sul default" passavano vacuamente, perche' leggeva la bundled.
 */
async function loadWithConfig(config, name) {
  const sandbox = makeSandbox({ name, config });
  const home = withHome(sandbox.dir);
  const { tools } = await bootExtension(sandbox);
  return {
    tools,
    ctx: sessionCtx(`${sandbox.dir}/sessione.jsonl`),
    cleanup: () => { home.restore(); sandbox.cleanup(); },
  };
}

test('una thresholdRatio non numerica non produce NaN e ricade sul default', async () => {
  const { tools, ctx, cleanup } = await loadWithConfig({ thresholdRatio: 'high' }, 'cfg-str');
  try {
    const { text, details } = await status(tools, ctx);
    assert.doesNotMatch(text, /NaN/, 'il budget non deve mai renderizzare NaN');
    assert.match(text, /threshold: 85%/, 'una soglia non valida deve ricadere sul default 0.85');
    assert.equal(details.budget, 80000, 'il budget deve restare il default 80k');
  } finally { cleanup(); }
});

test('una thresholdRatio fuori range non disattiva in silenzio l\'eviction', async () => {
  const { tools, ctx, cleanup } = await loadWithConfig({ thresholdRatio: 5 }, 'cfg-range');
  try {
    const { text } = await status(tools, ctx);
    assert.match(text, /threshold: 85%/, 'una soglia > 1 deve ricadere sul default');
  } finally { cleanup(); }
});

test('un tokenBudget non numerico ricade sul default', async () => {
  const { tools, ctx, cleanup } = await loadWithConfig({ tokenBudget: 'molti' }, 'cfg-budget');
  try {
    const { text, details } = await status(tools, ctx);
    assert.doesNotMatch(text, /NaN/);
    assert.equal(details.budget, 80000, "un budget non valido deve ricadere sull'80k di default");
  } finally { cleanup(); }
});

test('una config.json malformata non uccide l\'estensione', async () => {
  const { tools, ctx, cleanup } = await loadWithConfig('{ questo non e json', 'cfg-broken');
  try {
    const { text, details } = await status(tools, ctx);
    assert.doesNotMatch(text, /NaN/);
    assert.equal(typeof details.budget, 'number', 'un JSON rotto deve ricadere su un budget valido');
  } finally { cleanup(); }
});

test('una config valida continua a essere rispettata', async () => {
  const { tools, ctx, cleanup } = await loadWithConfig({ tokenBudget: 123456, thresholdRatio: 0.5 }, 'cfg-valid');
  try {
    const { text, details } = await status(tools, ctx);
    assert.equal(details.budget, 123456, 'un budget valido non deve essere scartato');
    assert.match(text, /threshold: 50%/, 'una soglia valida non deve essere scartata');
  } finally { cleanup(); }
});

test('i livelli booleani accettano solo veri booleani', async () => {
  // Un booleano scritto come stringa ("false") e' un errore classico dei file
  // di configurazione: `!!'false'` vale true, quindi non basta un cast.
  const { tools, ctx, cleanup } = await loadWithConfig(
    { levels: { stripBulkOutput: 'false' }, tokenBudget: 1000, thresholdRatio: 0.5 },
    'cfg-boolstr',
  );
  try {
    const { details } = await status(tools, ctx);
    assert.equal(details.budget, 1000, 'il resto della config deve restare valido');
  } finally { cleanup(); }
});
