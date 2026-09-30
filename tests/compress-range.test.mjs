/**
 * `cwl_compress_range`: l'estensione calcola l'indirizzo, il modello scrive il testo.
 *
 * Perche' esiste. Il budget gate chiedeva di compattare con
 * `cwl_compress(startHash=..., endHash=...)`. In una sessione VERA quella richiesta
 * non era eseguibile: gli hash vivono nello stato, sono opachi e l'agente non ha
 * modo di sapere quale hash corrisponda a quale messaggio. Risultato misurato:
 * gate armato al turno 2, 3 tentativi, zero compattazioni, contesto fermo a 370k
 * contro una soglia di 68k.
 *
 * La divisione che funziona: l'estensione sceglie gli indirizzi (li ha), il
 * modello scrive il riassunto (l'unica parte che solo il modello puo' fare).
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as path from 'node:path';
import { makeSandbox, bootExtension, withHome, sessionCtx } from './_helpers.mjs';

const config = (extra = {}) => ({
  tokenBudget: 100,
  thresholdRatio: 0.5,
  protectedTurns: 2,
  gate: false,
  levels: { stripReasoning: false, stripBulkOutput: false, stripIntermediate: false, removeEpisode: false },
  showWidget: false,
  debug: false,
  ...extra,
});

let seq = 0;
async function boot(cfg) {
  const sandbox = makeSandbox({ name: `range-${seq++}`, config: cfg });
  const home = withHome(sandbox.dir);
  const { tools, hooks } = await bootExtension(sandbox);
  const ctx = sessionCtx(path.join(sandbox.dir, 'sessione.jsonl'));
  await hooks.get('session_start')({}, ctx);
  return { sandbox, home, tools, hooks, ctx };
}

/** Sei turni user+assistant: gli ultimi 2 restano nella finestra di sicurezza. */
const conversation = () => {
  const out = [];
  for (let i = 1; i <= 6; i++) {
    out.push({ role: 'user', content: `turno ${i} contenuto ` + 'U'.repeat(200) });
    out.push({ role: 'assistant', content: `risposta ${i} contenuto ` + 'A'.repeat(200) });
  }
  return out;
};

const call = (tools, ctx, summary) =>
  tools.get('cwl_compress_range').execute('t', { summary }, undefined, undefined, ctx);

test('cwl_compress_range comprime l\'intervallo piu\' vecchio, senza hash', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot(config());
  try {
    await hooks.get('context')({ messages: conversation() }, ctx);
    const out = await call(tools, ctx, 'sintesi dei primi turni: obiettivo, decisioni, path');
    assert.equal(out.details.ok, true, `rifiutato: ${JSON.stringify(out.details)}`);
    assert.ok(out.details.tokens > 0, 'l\'intervallo compresso deve avere token');
    // L'indirizzo e' stato consumato: un secondo colpo non puo' ricomprimere lo stesso.
    const again = await call(tools, ctx, 'secondo tentativo');
    assert.equal(again.details.ok, false);
    assert.equal(again.details.error, 'nothing-to-compress');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('la finestra di sicurezza non viene toccata: l\'intervallo si ferma prima', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot(config({ protectedTurns: 2 }));
  try {
    const messages = conversation();
    await hooks.get('context')({ messages }, ctx);
    const out = await call(tools, ctx, 'sintesi');
    assert.equal(out.details.ok, true);

    // Verifica diretta: gli hash dell'intervallo NON devono includere i messaggi
    // degli ultimi 2 turni (gli ultimi 4 messaggi della lista).
    const { createHash } = await import('node:crypto');
    const hashOf = (s) => createHash('sha256').update(s).digest('hex').slice(0, 12);
    const protetti = messages.slice(-4).map((m) => hashOf(m.content));
    assert.ok(!protetti.includes(out.details.startHash ?? ''), 'start dentro la finestra');
    assert.ok(!protetti.includes(out.details.endHash ?? ''), 'end dentro la finestra');
    // E deve invece includere il primo messaggio (fuori dalla finestra).
    assert.equal(out.content[0].text.includes('Compresso'), true);
  } finally { home.restore(); sandbox.cleanup(); }
});

test('senza niente da comprimere rifiuta invece di inventare un intervallo', async () => {
  // Finestra di sicurezza cosi' larga da coprire tutto: non resta nulla.
  const { sandbox, home, tools, hooks, ctx } = await boot(config({ protectedTurns: 99 }));
  try {
    await hooks.get('context')({ messages: conversation() }, ctx);
    const out = await call(tools, ctx, 'sintesi');
    assert.equal(out.details.ok, false);
    assert.equal(out.details.error, 'nothing-to-compress');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('un riassunto vuoto viene rifiutato: e\' l\'unica parte che deve scrivere il modello', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot(config());
  try {
    await hooks.get('context')({ messages: conversation() }, ctx);
    const out = await call(tools, ctx, '   ');
    assert.equal(out.details.ok, false);
    assert.equal(out.details.error, 'missing-summary');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('un intervallo gia\' compresso non viene riproposto', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot(config());
  try {
    const messages = conversation();
    await hooks.get('context')({ messages }, ctx);
    const first = await call(tools, ctx, 'prima sintesi');
    assert.equal(first.details.ok, true);
    // Il hook successivo ricalcola l'intervallo SU QUELLO CHE RESTA.
    await hooks.get('context')({ messages }, ctx);
    const status = await tools.get('cwl_status').execute('id', {}, undefined, undefined, ctx);
    const testo = status.content.map((c) => c.text).join('\n');
    // O non resta niente, o resta un intervallo diverso dal primo.
    const hasRange = /Intervallo comprimibile/.test(testo);
    if (hasRange) {
      assert.ok(!testo.includes(`${first.details.startHash}..${first.details.endHash}`),
        'l\'intervallo gia\' compresso viene riproposto');
    }
  } finally { home.restore(); sandbox.cleanup(); }
});
