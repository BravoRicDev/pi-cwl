import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { makeSandbox, bootExtension, withHome, sessionCtx } from './_helpers.mjs';

// IL TRIGGER NON SI RAGGIUNGE MAI: e' il punto di tutto il file. La compattazione continua non
// deve dipendere dal budget — se aspettasse la soglia, una sessione che non la raggiunge non
// vedrebbe mai nascere una foglia (misurato sul vivo: 220 turni, UNA foglia).
const config = {
  tokenBudget: 10_000_000,
  thresholdRatio: 1,
  protectedTurns: 2,
  autoDelimiterTurns: 0, // isola la compattazione dalla registrazione degli episodi
  autoCompactMinTokens: 1_000,
  levels: { stripReasoning: false, stripBulkOutput: false, stripIntermediate: false, removeEpisode: false },
  showWidget: false,
  debug: true,
};

/** Ogni turno = un prompt utente e una risposta, abbastanza lunghi da superare il minimo. */
const conversation = (turns = 6) => {
  const messages = [];
  for (let turn = 1; turn <= turns; turn++) {
    messages.push({ role: 'user', content: `question ${turn} ` + 'U'.repeat(1500), timestamp: turn * 2 });
    messages.push({ role: 'assistant', content: `answer ${turn} ` + 'A'.repeat(1500), timestamp: turn * 2 + 1 });
  }
  return messages;
};

const tick = () => new Promise((resolve) => setImmediate(resolve));
/** Lascia atterrare il job in background: qualche giro di event loop, non un'attesa a tempo. */
const settle = async () => { for (let i = 0; i < 25; i++) await tick(); };

const result = (micro, summary) => ({
  stopReason: 'stop',
  content: [{ type: 'text', text: JSON.stringify({ micro, summary }) }],
});

async function setup(name, complete, overrides = {}) {
  const sandbox = makeSandbox({ name, config: { ...config, ...overrides } });
  const home = withHome(sandbox.dir);
  const { tools, hooks } = await bootExtension(sandbox);
  const ctx = sessionCtx(path.join(sandbox.dir, 'session.jsonl'));
  const model = { provider: 'test-provider', id: 'test-summarizer' };
  ctx.model = model;
  ctx.modelRegistry = {
    find: (provider, id) => (provider === model.provider && id === model.id ? model : null),
    hasConfiguredAuth: () => true,
    complete,
  };
  await hooks.get('session_start')({}, ctx);
  return { sandbox, home, tools, hooks, ctx };
}

/** Guida N turni consegnando all'hook la fetta di conversazione che esiste a quel turno. */
async function drive(hooks, ctx, messages, turns, { after = 2 } = {}) {
  let seen = 0;
  for (let round = 0; round < turns; round++) {
    seen = Math.min(messages.length, seen + after);
    await hooks.get('context')({ messages: messages.slice(0, seen) }, ctx);
    await hooks.get('turn_end')({}, ctx);
    await settle();
  }
}

const logOf = (sandbox) => fs.readFileSync(path.join(sandbox.dir, '.pi', 'cwl', 'cwl.log'), 'utf8');

function persistedState(sandbox) {
  const stateDir = path.join(sandbox.dir, '.pi', 'cwl', 'state');
  const file = fs.readdirSync(stateDir).find((name) => name.endsWith('.json'));
  assert.ok(file, 'il lavoro in background deve aver persistito lo stato di sessione');
  return JSON.parse(fs.readFileSync(path.join(stateDir, file), 'utf8'));
}

test('sotto il trigger il pavimento che avanza consegna il materiale sprotetto al background', async () => {
  let calls = 0;
  const { sandbox, hooks, ctx } = await setup('auto-compact-below-trigger', () => {
    calls++;
    return Promise.resolve(result('micro del turno', 'sintesi del materiale sprotetto'));
  });

  await drive(hooks, ctx, conversation(4), 4);

  assert.ok(calls > 0, 'il canale in background deve partire sotto soglia: e\` la molla del pavimento');
  const log = logOf(sandbox);
  assert.match(log, /AUTO-COMPACT turn \d+: the protected window moved to \d+/, 'la molla deve dire cosa ha fatto');
  assert.match(
    log,
    /under threshold/,
    'CONTROLLO DEL TEST: se il contesto fosse sopra soglia il test non proverebbe niente',
  );
  assert.doesNotMatch(log, /GATE armed/, 'il gate del budget non deve essere entrato in scena');
  const state = persistedState(sandbox);
  assert.ok(state.backgroundSummary, 'la richiesta di sintesi deve esistere nello stato');
});

test('la sintesi pronta diventa una FOGLIA anche sotto il trigger', async () => {
  let calls = 0;
  const { sandbox, hooks, ctx } = await setup('auto-compact-leaf', () => {
    calls++;
    return Promise.resolve(result('micro del turno', 'sintesi del materiale sprotetto'));
  });

  const messages = conversation(4);
  await drive(hooks, ctx, messages, 4);
  assert.ok(calls > 0, 'prerequisito: il background deve essere partito');

  // Un altro giro dell'hook: la sintesi e' pronta, quindi deve essere APPLICATA (foglia).
  await hooks.get('context')({ messages }, ctx);
  await hooks.get('turn_end')({}, ctx);
  await settle();

  const state = persistedState(sandbox);
  assert.ok(Array.isArray(state.spans), 'lo stato deve avere gli span');
  assert.ok(state.spans.length > 0, 'la foglia deve esistere: senza applicazione la sintesi sarebbe lavoro buttato');
  assert.match(
    logOf(sandbox),
    /under threshold/,
    'CONTROLLO DEL TEST: la foglia deve nascere anche senza superare il budget',
  );
});

test('un turno che non sprote niente non fa ripartire il lavoro (niente churn)', async () => {
  let calls = 0;
  const { sandbox, hooks, ctx } = await setup('auto-compact-no-churn', () => {
    calls++;
    return Promise.resolve(result('micro del turno', 'sintesi del materiale sprotetto'));
  });

  const messages = conversation(4);
  await drive(hooks, ctx, messages, 4);
  const afterFirst = calls;
  assert.ok(afterFirst > 0, 'prerequisito: il background deve essere partito');

  // LO STESSO elenco, ripetuto: il pavimento non si muove, quindi non c'e' nuovo materiale.
  for (let i = 0; i < 4; i++) {
    await hooks.get('context')({ messages }, ctx);
    await hooks.get('turn_end')({}, ctx);
    await settle();
  }

  assert.equal(calls, afterFirst, 'il pavimento fermo non deve costare chiamate LLM');
  const rows = (logOf(sandbox).match(/AUTO-COMPACT turn/g) ?? []).length;
  assert.equal(rows, 1, 'una sola riga di molla: un tentativo per AVANZAMENTO, non per hook');
});

test('`autoCompactMinTokens: 0` spegne la compattazione continua', async () => {
  let calls = 0;
  const { sandbox, hooks, ctx } = await setup(
    'auto-compact-off',
    () => { calls++; return Promise.resolve(result('micro', 'sintesi')); },
    { autoCompactMinTokens: 0 },
  );

  await drive(hooks, ctx, conversation(5), 5);

  assert.equal(calls, 0, '0 deve spegnere tutto');
  assert.doesNotMatch(logOf(sandbox), /AUTO-COMPACT/);
});

test('sotto il minimo di token non si spende una chiamata', async () => {
  let calls = 0;
  const { sandbox, hooks, ctx } = await setup(
    'auto-compact-below-min',
    () => { calls++; return Promise.resolve(result('micro', 'sintesi')); },
    { autoCompactMinTokens: 100_000_000 },
  );

  await drive(hooks, ctx, conversation(5), 5);

  assert.equal(calls, 0, 'materiale troppo poco: la foglia costerebbe una chiamata e non libererebbe niente');
  assert.doesNotMatch(logOf(sandbox), /AUTO-COMPACT/);
});
