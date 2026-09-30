/**
 * Livello A: la rete di sicurezza che chiude davvero il contesto.
 *
 * Perche' esiste. Il taglio dei blocchi di reasoning girava SOLO quando il grafo
 * degli episodi era VUOTO (`if (g.isEmpty)`). Misurato: in un contesto reale i
 * blocchi di thinking dell'assistant sono il 43% del contenuto (~280k token su
 * 650k). Quindi, appena l'agente apriva UN episodio, per il resto della sessione
 * la parte piu' grande e piu' sicura da togliere diventava intoccabile.
 *
 * Le due regressioni qui sotto sono complementari:
 *
 *  1. con un episodio presente, i thinking FUORI episodio devono sparire;
 *  2. gli ultimi N turni utente non devono MAI essere toccati.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as path from 'node:path';
import { makeSandbox, bootExtension, withHome, sessionCtx } from './_helpers.mjs';

const baseConfig = (extra = {}) => ({
  tokenBudget: 1000,
  thresholdRatio: 0.5,
  protectedTurns: 0,
  levels: { stripReasoning: true, stripBulkOutput: false, stripIntermediate: false, removeEpisode: false },
  showWidget: false,
  debug: false,
  ...extra,
});

let seq = 0;
async function boot(config) {
  const sandbox = makeSandbox({ name: `safety-${seq++}`, config });
  const home = withHome(sandbox.dir);
  const { tools, hooks } = await bootExtension(sandbox);
  const ctx = sessionCtx(path.join(sandbox.dir, 'sessione.jsonl'));
  await hooks.get('session_start')({}, ctx);
  return { sandbox, home, tools, hooks, ctx };
}

const thinking = (tag) => ({
  id: `a-${tag}`,
  role: 'assistant',
  content: [{ type: 'thinking', thinking: `${tag}:` + 'T'.repeat(4000) }],
});
const hasThinking = (m) => Array.isArray(m?.content) && m.content.some((b) => b?.type === 'thinking');

test('con un episodio presente, i reasoning block FUORI episodio vengono rimossi', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot(baseConfig());
  try {
    // Un episodio esiste: e' proprio la condizione che prima disattivava la rete.
    await tools.get('delimiter').execute('call-s', { action: 'start', name: 'ep', type: 'expl' }, undefined, undefined, ctx);
    await tools.get('delimiter').execute('call-e', { action: 'end', name: 'ep', description: 'n' }, undefined, undefined, ctx);

    const messages = [
      { role: 'user', content: 'apriamo' },
      { role: 'toolResult', toolCallId: 'call-s', toolName: 'delimiter', content: [{ type: 'text', text: 'aperto' }] },
      { role: 'toolResult', toolCallId: 'call-e', toolName: 'delimiter', content: [{ type: 'text', text: 'chiuso' }] },
      // Questi stanno FUORI da qualunque episodio.
      thinking('fuori-1'), thinking('fuori-2'), thinking('fuori-3'),
      { role: 'user', content: 'domanda' },
    ];
    const prima = messages.filter(hasThinking).length;
    assert.equal(prima, 3);

    const res = await hooks.get('context')({ messages }, ctx);
    const out = (res && res.messages) || messages;
    assert.equal(out.filter(hasThinking).length, 0,
      'i thinking fuori episodio sono sopravvissuti: la rete di sicurezza non gira con un episodio presente');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('la finestra di sicurezza: gli ultimi N turni utente non vengono toccati', async () => {
  const N = 10;
  const { sandbox, home, hooks, ctx } = await boot(baseConfig({ protectedTurns: N }));
  try {
    // 12 turni: user + assistant(thinking) ciascuno.
    const messages = [];
    for (let i = 1; i <= 12; i++) {
      messages.push({ role: 'user', content: `turno ${i}` });
      messages.push(thinking(`t${i}`));
    }
    const res = await hooks.get('context')({ messages }, ctx);
    const out = (res && res.messages) || messages;

    const sopravvissuti = out.filter(hasThinking).map((m) => m.content[0].thinking.slice(0, m.content[0].thinking.indexOf(':')));
    // I primi turni (fuori dalla finestra) vengono ripuliti; gli ultimi N no.
    assert.ok(!sopravvissuti.includes('t1'), 'il turno piu\' vecchio doveva essere ripulito');
    for (let i = 12 - N + 1; i <= 12; i++) {
      assert.ok(sopravvissuti.includes(`t${i}`), `il turno ${i} e' dentro la finestra di sicurezza e non doveva essere toccato`);
    }
    assert.equal(out.filter((m) => m.role === 'user').length, 12, 'un turno utente e\' stato rimosso');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('senza finestra (protectedTurns=0) non resta nessun thinking', async () => {
  const { sandbox, home, hooks, ctx } = await boot(baseConfig({ protectedTurns: 0 }));
  try {
    const messages = [];
    for (let i = 1; i <= 12; i++) {
      messages.push({ role: 'user', content: `turno ${i}` });
      messages.push(thinking(`t${i}`));
    }
    const res = await hooks.get('context')({ messages }, ctx);
    const out = (res && res.messages) || messages;
    assert.equal(out.filter(hasThinking).length, 0);
  } finally { home.restore(); sandbox.cleanup(); }
});

test('una conversazione piu\' corta della finestra non viene compattata affatto', async () => {
  const { sandbox, home, hooks, ctx } = await boot(baseConfig({ protectedTurns: 10 }));
  try {
    // 3 turni soltanto: tutto dentro la finestra.
    const messages = [
      { role: 'user', content: 'uno' }, thinking('t1'),
      { role: 'user', content: 'due' }, thinking('t2'),
      { role: 'user', content: 'tre' }, thinking('t3'),
    ];
    const res = await hooks.get('context')({ messages }, ctx);
    const out = (res && res.messages) || messages;
    assert.equal(out.filter(hasThinking).length, 3,
      'una conversazione dentro la finestra di sicurezza non deve essere toccata');
  } finally { home.restore(); sandbox.cleanup(); }
});
