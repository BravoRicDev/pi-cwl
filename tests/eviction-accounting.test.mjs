/**
 * Contabilita' dell'evacuazione: la riga di log non deve dichiarare due
 * risparmi diversi.
 *
 * IL DIFETTO MISURATO. La riga
 *   `EVICTION applied: 32 msg removed, 0 reduced, 79683t -> 58987t (saved 21053t)`
 * dichiara 21053t ma l'evacuazione reale e' `79683 - 58987 = 20696t`. Il numero
 * fra parentesi era la SOMMA DELLE STIME DEL PIANO (`removedTokens +
 * truncatedTokens`), non il risparmio: due quantita' diverse stampate come se
 * fossero la stessa. Altre righe dello stesso log sbagliavano nell'altro verso
 * (77164 -> 68800 = 8364, dichiarava 8056), ed e' proprio l'alternarsi dei segni
 * a dimostrare che non e' un arrotondamento.
 *
 * LE TRE CAUSE, tutte nel corpo dell'applier:
 *  1. il MARCATORE di rientro dell'episodio (`role:'custom'`, `cwl-evicted`)
 *     viene AGGIUNTO alla lista: non era nell'input, quindi alza `afterTokens`
 *     e il risparmio reale e' MINORE del dichiarato (+357t su quella riga);
 *  2. la potatura dei `toolCall` orfani RIMPICCIOLISCE messaggi sopravvissuti,
 *     e nessuno la conta: il risparmio reale e' MAGGIORE (-308t, -121t);
 *  3. per gli strip `truncatedTokens` misura `textLengthOf` (solo il contenuto)
 *     mentre `currentTokens`/`afterTokens` misurano il messaggio SERIALIZZATO
 *     (`JSON.stringify`): due basi diverse nella stessa riga.
 *
 * E il messaggio che l'utente legge (`evictionNotice`, che riceve
 * `currentTokens` e `afterTokens`) usava GIA' il numero misurato: log e
 * interfaccia raccontavano due cose diverse sullo stesso evento, e il contatore
 * di stato (`totalEvictedTokens`) seguiva la stima.
 *
 * Il test non chiede che il piano sia preciso: chiede che la riga sia
 * ARITMETICAMENTE UNA. Un operatore che legge due numeri e li trova diversi non
 * sa piu' quale credere, ed e' esattamente cosi' che questo difetto e' stato
 * trovato.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { makeSandbox, bootExtension, withHome, sessionCtx } from './_helpers.mjs';

let seq = 0;

/** Solo l'evacuazione piena: il livello `removed` e' quello che aggiunge il marcatore. */
const soloRimozione = (extra = {}) => ({
  tokenBudget: 1000,
  thresholdRatio: 0.5,
  protectedTurns: 0,
  levels: { stripReasoning: false, stripBulkOutput: false, stripIntermediate: false, removeEpisode: true },
  showWidget: false,
  debug: true,
  ...extra,
});

async function boot(config) {
  const sandbox = makeSandbox({ name: `accounting-${seq++}`, config });
  const home = withHome(sandbox.dir);
  const { tools, hooks } = await bootExtension(sandbox);
  const ctx = sessionCtx(path.join(sandbox.dir, 'sessione.jsonl'));
  await hooks.get('session_start')({}, ctx);
  return { sandbox, home, tools, hooks, ctx };
}

const logDi = (sandbox) => {
  const p = path.join(sandbox.dir, '.pi', 'cwl', 'cwl.log');
  return fs.existsSync(p) ? fs.readFileSync(p, 'utf8') : '';
};

const hook = async (hooks, ctx, messages) => {
  const res = await hooks.get('context')({ messages }, ctx);
  return (res && res.messages) || messages;
};

const assistant = (marker) => ({
  role: 'assistant',
  content: [{ type: 'text', text: `${marker} ` + 'X'.repeat(3000) }],
});

/** La riga che sto misurando, letta dal log vero. */
const RIGA = /EVICTION applied: (\d+) msg removed, (\d+) reduced, (\d+)t -> (\d+)t \(saved (\d+)t\)/;

test('la riga EVICTION dichiara il risparmio MISURATO, non la stima del piano', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot(soloRimozione());
  try {
    // Un episodio chiuso e interamente dentro il range: la sua evacuazione piena
    // e' cio' che aggiunge il marcatore di rientro (la causa n.1 del difetto).
    await tools.get('delimiter').execute('call-s', { action: 'start', name: 'contabilita', type: 'expl' }, undefined, undefined, ctx);
    await tools.get('delimiter').execute('call-e', { action: 'end', name: 'contabilita', description: 'imparato' }, undefined, undefined, ctx);
    const messages = [
      { role: 'user', content: 'apertura' },
      { role: 'toolResult', toolCallId: 'call-s', toolName: 'delimiter', content: [{ type: 'text', text: 'aperto' }] },
      assistant('DENTRO-1'),
      assistant('DENTRO-2'),
      { role: 'toolResult', toolCallId: 'call-e', toolName: 'delimiter', content: [{ type: 'text', text: 'chiuso' }] },
      { role: 'user', content: 'domanda recente' },
    ];
    const out = await hook(hooks, ctx, messages);
    assert.ok(!JSON.stringify(out).includes('DENTRO-1'),
      'nessuna evacuazione e\' avvenuta: il test non prova niente');

    const m = RIGA.exec(logDi(sandbox));
    assert.ok(m, 'la riga EVICTION non e\' stata scritta nel log');
    const [, dropped, , before, after, saved] = m.map(Number);
    assert.ok(dropped >= 1, 'la riga dice zero messaggi rimossi: non e\' un\'evacuazione');
    assert.equal(
      saved,
      before - after,
      `la riga dichiara ${saved}t risparmiati ma ${before}t -> ${after}t ne fa ${before - after}: ` +
      'due numeri diversi nella stessa riga, e in mezzo ci sono il marcatore aggiunto, i toolCall orfani potati ' +
      'e due stimatori (contenuto contro JSON serializzato)',
    );
  } finally { home.restore(); sandbox.cleanup(); }
});

test('la riga EVICTION e\' coerente anche quando l\'evacuazione RIDUCE invece di rimuovere', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot({
    tokenBudget: 1000,
    thresholdRatio: 0.5,
    protectedTurns: 0,
    levels: { stripReasoning: false, stripBulkOutput: true, stripIntermediate: false, removeEpisode: false },
    showWidget: false,
    debug: true,
  });
  try {
    await tools.get('delimiter').execute('call-s', { action: 'start', name: 'riduzione', type: 'expl' }, undefined, undefined, ctx);
    await tools.get('delimiter').execute('call-e', { action: 'end', name: 'riduzione', description: 'imparato' }, undefined, undefined, ctx);
    // Un toolResult enorme dentro l'episodio: il livello `bulk` lo taglia senza
    // rimuoverlo, ed e' il ramo dove `truncatedTokens` usa l'altro stimatore.
    const grasso = { role: 'toolResult', toolCallId: 'call-big', toolName: 'bash', content: [{ type: 'text', text: 'B'.repeat(60000) }] };
    const messages = [
      { role: 'user', content: 'apertura' },
      { role: 'toolResult', toolCallId: 'call-s', toolName: 'delimiter', content: [{ type: 'text', text: 'aperto' }] },
      grasso,
      assistant('DENTRO-1'),
      { role: 'toolResult', toolCallId: 'call-e', toolName: 'delimiter', content: [{ type: 'text', text: 'chiuso' }] },
      { role: 'user', content: 'domanda recente' },
    ];
    const out = await hook(hooks, ctx, messages);
    const ridotto = JSON.stringify(out).length < JSON.stringify(messages).length;
    assert.ok(ridotto, 'nessuna riduzione e\' avvenuta: il test non prova niente');

    const m = RIGA.exec(logDi(sandbox));
    assert.ok(m, 'la riga EVICTION non e\' stata scritta nel log');
    const [, , , before, after, saved] = m.map(Number);
    assert.equal(
      saved,
      before - after,
      `la riga dichiara ${saved}t risparmiati ma ${before}t -> ${after}t ne fa ${before - after}: ` +
      'per gli strip il conto usa la lunghezza del CONTENUTO mentre la freccia usa il messaggio SERIALIZZATO',
    );
  } finally { home.restore(); sandbox.cleanup(); }
});
