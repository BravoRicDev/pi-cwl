/**
 * Un episodio evictato e' RECUPERABILE dal transcript.
 *
 * Perche' esiste. La compattazione episodica era una porta a senso unico:
 * al livello `removed` buttava via ogni messaggio dell'episodio e teneva solo
 * la descrizione in prosa scritta dall'agente alla chiusura. Due buchi:
 *
 *   1. un episodio `act` (descrizione sempre vuota) non lasciava NULLA, e un
 *      `expl` senza descrizione perdeva il contenuto per sempre;
 *   2. `findTranscript` costruiva `_${sessionKey}.jsonl`. La chiave prodotta da
 *      `sessionKey()` e' un PATH completo, quindi il suffisso diventava
 *      `_/home/.../sessione.jsonl.jsonl`: non matchava mai. Perfino `cwl_recall`
 *      rispondeva "transcript non trovato" mentre il file era li'.
 *
 * Il dato non mancava: il transcript di Pi e' append-only e conserva gli
 * originali, e l'episodio ha GIA' le due ancore (startToolCallId/endToolCallId)
 * persistite nello stato. Mancava il PUNTATORE.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { makeSandbox, bootExtension, withHome, sessionCtx } from './_helpers.mjs';

/** Costruisce un sandbox con HOME spostata e transcript scrivibile. */
function setup({ name, config = null }) {
  const sandbox = makeSandbox({ name, config });
  const home = withHome(sandbox.dir);
  const realFile = path.join(
    sandbox.dir, '.pi', 'agent', 'sessions', '--fake--',
    '2026-01-01T00-00-00-000Z_fake-session-id.jsonl',
  );
  fs.mkdirSync(path.dirname(realFile), { recursive: true });
  return { sandbox, home, realFile };
}

/** Una riga di transcript nella forma che scrive Pi. */
const record = (id, role, content, extra = {}) =>
  JSON.stringify({ type: 'message', id, parentId: null, timestamp: '2026-01-01T00:00:00.000Z', message: { role, content, ...extra } });

const text = (s) => [{ type: 'text', text: s }];

/** Apre e chiude un episodio con ancore scelte da noi. */
async function withEpisode(tools, hooks, ctx, { name = 'ep1', type = 'expl', start = 'call-start-1', end = 'call-end-1', description = '' } = {}) {
  await hooks.get('session_start')({}, ctx);
  const d = tools.get('delimiter');
  const opened = await d.execute(start, { action: 'start', name, type }, undefined, undefined, ctx);
  assert.equal(opened.details.ok, true, `apertura fallita: ${JSON.stringify(opened.details)}`);
  const closed = await d.execute(end, { action: 'end', name, description }, undefined, undefined, ctx);
  assert.equal(closed.details.ok, true, `chiusura fallita: ${JSON.stringify(closed.details)}`);
}

test('cwl_recall_episode restituisce il testo originale di un episodio evictato', async () => {
  const { sandbox, home, realFile } = setup({ name: 'ep-recall' });
  try {
    fs.writeFileSync(realFile, [
      record('r0', 'user', text('apriamo')),
      record('r1', 'toolResult', text('episodio aperto'), { toolCallId: 'call-start-1', toolName: 'delimiter' }),
      record('r2', 'assistant', text('il file segreto e\'-/tmp/alfa.txt e la funzione parseZeta()')),
      record('r3', 'toolResult', text('contenuto grep: parseZeta in alfa.txt'), { toolCallId: 'call-tool-2', toolName: 'bash' }),
      record('r4', 'toolResult', text('episodio chiuso'), { toolCallId: 'call-end-1', toolName: 'delimiter' }),
      record('r5', 'user', text('dopo')),
    ].join('\n') + '\n');

    const ctx = sessionCtx(realFile);
    const { tools, hooks } = await bootExtension(sandbox);
    await withEpisode(tools, hooks, ctx, { description: 'imparato il nome parseZeta' });

    const out = await tools.get('cwl_recall_episode').execute('t', { name: 'ep1', full: true }, undefined, undefined, ctx);

    assert.equal(out.details.ok, true, `recupero fallito: ${JSON.stringify(out.details)}`);
    const body = out.content.map((c) => c.text).join('\n');
    // Il contenuto dell'episodio, non solo la descrizione.
    assert.match(body, /parseZeta/, 'manca il testo del messaggio assistant');
    assert.match(body, /contenuto grep: parseZeta in alfa\.txt/, 'manca il toolResult, che e\' il 45%% del transcript');
    // I messaggi FUORI dall'episodio non devono entrare nel recupero.
    assert.doesNotMatch(body, /apriamo/, 'incluso un messaggio prima dell\'ancora di apertura');
    assert.doesNotMatch(body, /dopo/, 'incluso un messaggio dopo l\'ancora di chiusura');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('cwl_recall_episode tronca di default e non gonfia il contesto appena liberato', async () => {
  const { sandbox, home, realFile } = setup({ name: 'ep-trunc' });
  try {
    const lungo = 'X'.repeat(20000);
    fs.writeFileSync(realFile, [
      record('r1', 'toolResult', text('aperto'), { toolCallId: 'call-start-1', toolName: 'delimiter' }),
      record('r2', 'assistant', text(lungo)),
      record('r4', 'toolResult', text('chiuso'), { toolCallId: 'call-end-1', toolName: 'delimiter' }),
    ].join('\n') + '\n');
    const ctx = sessionCtx(realFile);
    const { tools, hooks } = await bootExtension(sandbox);
    await withEpisode(tools, hooks, ctx);

    const short = await tools.get('cwl_recall_episode').execute('t', { name: 'ep1' }, undefined, undefined, ctx);
    assert.equal(short.details.full, false);
    assert.ok(short.details.chars > 4000, 'il contenuto recuperato dovrebbe essere grande');
    const shortBody = short.content.map((c) => c.text).join('\n');
    assert.match(shortBody, /troncato/, 'manca l\'avviso di troncamento');
    assert.ok(shortBody.length < 12000, `risposta troppo lunga: ${shortBody.length}`);

    const full = await tools.get('cwl_recall_episode').execute('t', { name: 'ep1', full: true }, undefined, undefined, ctx);
    assert.equal(full.details.full, true);
    assert.ok(full.content.map((c) => c.text).join('\n').length > 20000, 'full=true deve restituire tutto');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('cwl_recall_episode rifiuta un episodio inesistente o ancora aperto', async () => {
  const { sandbox, home, realFile } = setup({ name: 'ep-guard' });
  try {
    fs.writeFileSync(realFile, record('r1', 'user', text('niente')) + '\n');
    const ctx = sessionCtx(realFile);
    const { tools, hooks } = await bootExtension(sandbox);
    await hooks.get('session_start')({}, ctx);
    await tools.get('delimiter').execute('call-start-1', { action: 'start', name: 'aperto', type: 'expl' }, undefined, undefined, ctx);

    const unknown = await tools.get('cwl_recall_episode').execute('t', { name: 'non-esiste' }, undefined, undefined, ctx);
    assert.equal(unknown.details.ok, false);
    assert.equal(unknown.details.error, 'episode-not-found');

    const open = await tools.get('cwl_recall_episode').execute('t', { name: 'aperto' }, undefined, undefined, ctx);
    assert.equal(open.details.ok, false);
    assert.equal(open.details.error, 'episode-still-open');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('cwl_recall risolve la chiave che sessionKey produce davvero (path completo)', async () => {
  const { sandbox, home, realFile } = setup({ name: 'ep-find' });
  try {
    fs.writeFileSync(realFile, record('a1', 'user', text('parola chiave unica zebra42')) + '\n');
    const ctx = sessionCtx(realFile);
    const { tools, hooks } = await bootExtension(sandbox);
    await hooks.get('session_start')({}, ctx);

    const out = await tools.get('cwl_recall').execute('t', { query: 'zebra42', limit: 5 }, undefined, undefined, ctx);
    assert.equal(out.details.ok, true, `recall fallito: ${JSON.stringify(out.details)}`);
    assert.equal(out.details.hits, 1);
  } finally { home.restore(); sandbox.cleanup(); }
});

test('l\'eviction emette il puntatore, anche per un episodio act', async () => {
  // Solo `removeEpisode` attivo: cosi' l'eviction va DIRETTA al livello removed,
  // senza dover escalare reasoning -> bulk -> intermediate in piu' turni.
  // protectedTurns=0: qui la finestra di sicurezza va SPENTA, altrimenti i due
  // soli turni utente della scena sono tutti protetti e nulla viene evicto.
  const config = {
    tokenBudget: 10,
    thresholdRatio: 0.5,
    protectedTurns: 0,
    levels: { stripReasoning: false, stripBulkOutput: false, stripIntermediate: false, removeEpisode: true },
    showWidget: false,
    debug: false,
  };
  for (const type of ['expl', 'act']) {
    const { sandbox, home, realFile } = setup({ name: `ep-marker-${type}`, config });
    try {
      fs.writeFileSync(realFile, record('r1', 'user', text('x')) + '\n');
      const ctx = sessionCtx(realFile);
      const { tools, hooks } = await bootExtension(sandbox);
      await withEpisode(tools, hooks, ctx, { type, name: `ep-${type}`, description: type === 'expl' ? 'nota utile' : '' });

      const messages = [
        { role: 'user', content: 'apriamo' },
        { role: 'toolResult', toolCallId: 'call-start-1', toolName: 'delimiter', content: text('aperto') },
        { role: 'assistant', content: 'molto contenuto da evictare '.repeat(40) },
        { role: 'toolResult', toolCallId: 'call-end-1', toolName: 'delimiter', content: text('chiuso') },
        { role: 'user', content: 'dopo' },
      ];
      const res = await hooks.get('context')({ messages }, ctx);
      assert.ok(res && Array.isArray(res.messages), 'il context hook non ha evictato nulla');

      const marker = res.messages.find((m) => m.customType === 'cwl-evicted');
      assert.ok(marker, `nessun marker emesso per un episodio ${type}`);
      assert.match(marker.content, /cwl_recall_episode\("ep-(expl|act)"\)/, 'il marker non contiene il puntatore');
      // I messaggi fuori dall'episodio restano.
      assert.ok(res.messages.some((m) => m.role === 'user' && m.content === 'dopo'), 'il turno user e\' stato toccato');
    } finally { home.restore(); sandbox.cleanup(); }
  }
});
