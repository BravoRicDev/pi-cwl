/**
 * Livello B: il gate che chiede all'AGENTE di compattare.
 *
 * Perche' esiste. L'eviction deterministica e' cieca: taglia per eta', non per
 * significato. Il gate e' l'unico canale che puo' ottenere dal modello una
 * decisione semantica.
 *
 * Differenza chiave da anti-amnesia: la' il gate si chiude quando il modello
 * ECCHEGGIA un token `[CARD OK]`; qui si chiude per EFFETTO, perche' la cosa che
 * vogliamo (meno token) e' direttamente misurabile. Una conferma inventata non
 * vale niente: conta solo il contesto misurato.
 *
 * Il gate e' una PREFERENZA, mai una garanzia: se il modello lo ignora, il
 * paracadute del Livello A e' quello che continua a tenere il contesto.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as path from 'node:path';
import { makeSandbox, bootExtension, withHome, sessionCtx } from './_helpers.mjs';

// Nessun livello attivo: cosi' NIENTE viene compattato in modo deterministico e
// il contesto resta sopra budget, che e' la condizione in cui il gate deve agire.
const overBudgetConfig = (extra = {}) => ({
  tokenBudget: 100,
  thresholdRatio: 0.5,
  protectedTurns: 0,
  gate: true,
  levels: { stripReasoning: false, stripBulkOutput: false, stripIntermediate: false, removeEpisode: false },
  showWidget: false,
  debug: false,
  ...extra,
});

let seq = 0;
async function boot(config) {
  const sandbox = makeSandbox({ name: `gate-${seq++}`, config });
  const home = withHome(sandbox.dir);
  const { hooks } = await bootExtension(sandbox);
  const ctx = sessionCtx(path.join(sandbox.dir, 'sessione.jsonl'));
  await hooks.get('session_start')({}, ctx);
  return { sandbox, home, hooks, ctx };
}

/** Messaggi senza thinking: niente da compattare, quindi si resta sopra budget. */
const heavy = (n = 6) =>
  Array.from({ length: n }, (_, i) => ({ role: 'user', content: `messaggio ${i} ` + 'P'.repeat(400) }));

const gateOf = (res) => (res?.messages ?? []).filter((m) => m.customType === 'cwl-budget-gate');
/** Un giro completo: hook di contesto + fine turno. */
async function round(hooks, ctx, messages) {
  const res = await hooks.get('context')({ messages }, ctx);
  await hooks.get('turn_end')({}, ctx);
  return res;
}

test('il gate inietta la richiesta di compattare quando il budget non scende', async () => {
  const { sandbox, home, hooks, ctx } = await boot(overBudgetConfig());
  try {
    const messages = heavy();
    // Turni 1 e 2: sopra budget ma il gate non e' ancora armato.
    const r1 = await round(hooks, ctx, messages);
    assert.equal(gateOf(r1).length, 0, 'il gate non deve chiedere nulla al primo turno');
    const r2 = await round(hooks, ctx, messages);
    assert.equal(gateOf(r2).length, 0, 'il gate non deve chiedere nulla al secondo turno');

    // Turno 3: turn_end ha armato il gate, quindi l'hook successivo lo rende.
    const r3 = await round(hooks, ctx, messages);
    const demands = gateOf(r3);
    assert.equal(demands.length, 1, 'il gate doveva iniettare la richiesta');
    const text = String(demands[0].content);
    assert.match(text, /cwl_compress/, 'la richiesta non dice cosa fare');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('la richiesta non si accumula: viene sostituita, non impilata', async () => {
  const { sandbox, home, hooks, ctx } = await boot(overBudgetConfig());
  try {
    // Si incatena la RISPOSTA dell'hook dentro quello SUCCESSIVO, come fa Pi:
    // e' l'unico modo per accorgersi di un accumulo. Passando sempre lo stesso
    // array di partenza la richiesta precedente non entrerebbe mai nel giro
    // successivo, e un filtro mancante resterebbe invisibile.
    let cur = heavy();
    for (let i = 0; i < 5; i++) {
      const res = await hooks.get('context')({ messages: cur }, ctx);
      cur = res?.messages ?? cur;
      assert.ok(gateOf(res).length <= 1,
        `giro ${i}: nel contesto ci sono ${gateOf(res).length} richieste insieme`);
      await hooks.get('turn_end')({}, ctx);
    }
    assert.equal(gateOf({ messages: cur }).length, 1,
      'dopo l\'armamento la richiesta deve esserci, ed essere una sola');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('il gate si chiude per EFFETTO: contesto sotto budget = nessuna richiesta', async () => {
  const { sandbox, home, hooks, ctx } = await boot(overBudgetConfig());
  try {
    const messages = heavy();
    for (let i = 0; i < 3; i++) await round(hooks, ctx, messages);
    const attivo = await round(hooks, ctx, messages);
    assert.equal(gateOf(attivo).length, 1, 'il gate doveva essere attivo');

    // Il contesto scende sotto la soglia (trigger = 100 * 0.5 = 50 token).
    const leggero = [{ role: 'user', content: 'corto' }];
    const sotto = await hooks.get('context')({ messages: leggero }, ctx);
    assert.equal(gateOf(sotto).length, 0, 'sotto budget la richiesta non deve comparire');

    // E il gate e' DISINNESCATO: il hook successivo, di nuovo sopra budget, non
    // deve chiedere nulla. Senza il disarmo resterebbe armato e tornerebbe a
    // chiedere subito, il che e' esattamente cio' che questo test deve cogliere.
    const dopo = await hooks.get('context')({ messages: messages }, ctx);
    assert.equal(gateOf(dopo).length, 0,
      'il gate e\' rimasto armato dopo essere stato soddisfatto');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('dopo N tentativi senza risposta il gate si arrende (con cooldown)', async () => {
  const { sandbox, home, hooks, ctx } = await boot(overBudgetConfig());
  try {
    const messages = heavy();
    // Si arriva al gate attivo, poi si lasciano passare i tentativi.
    for (let i = 0; i < 3; i++) await round(hooks, ctx, messages);
    assert.equal(gateOf(await round(hooks, ctx, messages)).length, 1, 'gate non attivo');

    // GATE_MAX_ATTEMPTS = 3: si supera il limite.
    for (let i = 0; i < 5; i++) await round(hooks, ctx, messages);
    const r = await round(hooks, ctx, messages);
    assert.equal(gateOf(r).length, 0,
      'dopo il numero massimo di tentativi il gate deve tacere invece di insistere per sempre');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('con gate=false non viene mai iniettata nessuna richiesta', async () => {
  const { sandbox, home, hooks, ctx } = await boot(overBudgetConfig({ gate: false }));
  try {
    const messages = heavy();
    for (let i = 0; i < 6; i++) {
      const r = await round(hooks, ctx, messages);
      assert.equal(gateOf(r).length, 0, 'gate=false deve disattivare del tutto il canale');
    }
  } finally { home.restore(); sandbox.cleanup(); }
});
