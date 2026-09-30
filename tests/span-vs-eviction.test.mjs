/**
 * Uno span vivo NON deve spegnere l'evacuazione deterministica.
 *
 * IL DIFETTO MISURATO, DAL VIVO. In `index.ts` (~2419-2459), dentro
 * `if (st.spans.length > 0)`, quando almeno uno span si applica
 * (`applied.applied > 0`) l'hook rinnova l'indirizzo del range e fa
 * `return finish(applied.kept)`. Tutto cio' che sta sotto — il pavimento di
 * sicurezza, `reasoningFallback` (~2493) e `runEvictionPass` (~2524) — diventa
 * IRRAGGIUNGIBILE. E `applied.applied > 0` e' vero a OGNI turno finche' uno span
 * risolve, perche' lo span va riapplicato ogni volta: quindi UNO SPAN NELLO STATO
 * SPEGNE L'EVACUAZIONE E IL FALLBACK PER IL RESTO DELLA SESSIONE.
 *
 * Le prove erano nel log di una sessione vera (dopo la ricarica del 30/09):
 *   `SPANS re-applied: 3, nothing new to count`
 *   `RANGE none | 178 msgs, 133852t vs trigger 68000t, 3 span(s)`
 * a ogni turno, con ZERO righe `EVICTION`, ZERO `CONTEXT ...`, ZERO `no safe
 * candidate`, ZERO `FALLBACK reasoning-strip`, mentre `cwl_status` dichiarava
 * `Episodi: 2 | attivi: 0 | con contenuto evictabile: 1`. Cioe': 133k token
 * contro un trigger di 68k, un episodio evacuabile, e l'estensione non faceva
 * NULLA — ne' evacua ne' riduce, e non ha nemmeno piu' un intervallo da chiedere
 * all'agente (`RANGE none`).
 *
 * Il commento sopra quel `return` dimostra che il ritorno era NOTO: era stato
 * corretto per un altro motivo (il rinnovo dell'indirizzo, commit a7a8da0),
 * perche' lasciandolo prima del rinnovo l'agente non poteva comprimere una
 * seconda volta. La conseguenza — evacuazione e fallback irraggiungibili — non
 * era stata vista. Il test la fissa: uno span vivo puo' far risparmiare, ma non
 * puo' essere l'ULTIMA cosa che l'estensione sa fare.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { makeSandbox, bootExtension, withHome, sessionCtx } from './_helpers.mjs';

let seq = 0;

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
  const sandbox = makeSandbox({ name: `span-vs-eviction-${seq++}`, config });
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

const testo = (m) => JSON.stringify(m ?? {});
const contiene = (out, marker) => out.some((m) => testo(m).includes(marker));

/** Conversazione con estremi eleggibili: e' cio' che rende comprimibile un intervallo. */
const conversazione = () => {
  const out = [];
  for (let i = 1; i <= 6; i++) {
    out.push({ role: 'user', content: `turno ${i} contenuto ` + 'U'.repeat(200) });
    out.push({ role: 'assistant', content: `risposta ${i} contenuto ` + 'A'.repeat(200) });
  }
  return out;
};

const assistant = (marker) => ({ role: 'assistant', content: [{ type: 'text', text: `${marker} ` + 'X'.repeat(3000) }] });
const assistantPiccolo = (marker) => ({ role: 'assistant', content: [{ type: 'text', text: marker }] });

/**
 * Il messaggio che PORTA la chiamata al tool.
 *
 * Perche' serve. Le ancore di un episodio sono i DUE `toolResult` del `delimiter`,
 * e la riparazione delle coppie dello span (`PAIR REPAIR`) CANCELLA i toolResult
 * rimasti senza la loro chiamata. Un test che invoca `delimiter` direttamente e
 * mette nel proprio array solo i `toolResult` costruisce ancore ORFANE per colpa
 * del fixture, non del codice (in una sessione vera la coppia esiste), e la
 * diagnosi finisce per accusare lo span di una cosa che ha fatto il test. La
 * forma del blocco e' quella usata da `tests/compress-range.test.mjs`.
 */
const chiamata = (id) => ({
  role: 'assistant',
  content: [{ type: 'toolCall', id, name: 'delimiter', arguments: { action: 'end', name: 'dopo-lo-span' } }],
});

/** Crea uno span vero, con lo stesso strumento che usa l'agente in produzione. */
async function creaSpan(tools, hooks, ctx, messages) {
  await hook(hooks, ctx, messages);
  const out = await tools.get('cwl_compress_range').execute('t', { summary: 'SINTESI-1 dei primi turni' }, undefined, undefined, ctx);
  assert.equal(out.details.ok, true, `lo span non e' stato creato (senza span il test non prova niente): ${JSON.stringify(out.details)}`);
  return out;
}

test('con uno span vivo che si riapplica, l\'episodio evacuabile viene comunque evacuato', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot(soloRimozione());
  try {
    // 1. Una conversazione comprimibile: serve a creare lo span.
    const base = conversazione();
    await creaSpan(tools, hooks, ctx, base);

    // 2. Un episodio chiuso, con contenuto grande, DOPO lo span: e' evacuabile.
    await tools.get('delimiter').execute('call-s', { action: 'start', name: 'dopo-lo-span', type: 'expl' }, undefined, undefined, ctx);
    await tools.get('delimiter').execute('call-e', { action: 'end', name: 'dopo-lo-span', description: 'imparato' }, undefined, undefined, ctx);
    const messages = [
      ...base,
      { role: 'user', content: 'apri' },
      chiamata('call-s'),
      { role: 'toolResult', toolCallId: 'call-s', toolName: 'delimiter', content: [{ type: 'text', text: 'aperto' }] },
      assistant('DENTRO-1'),
      assistant('DENTRO-2'),
      chiamata('call-e'),
      { role: 'toolResult', toolCallId: 'call-e', toolName: 'delimiter', content: [{ type: 'text', text: 'chiuso' }] },
      { role: 'user', content: 'domanda recente' },
    ];

    const out = await hook(hooks, ctx, messages);
    const log = logDi(sandbox);
    // Non-vacuita': lo span DEVE essersi riapplicato in questo turno, altrimenti
    // il test passerebbe anche senza il difetto che vuole dimostrare.
    assert.match(log, /SPANS (re-applied|applied): \d+/, 'lo span non si e\' riapplicato: il test non prova niente');

    assert.ok(
      !contiene(out, 'DENTRO-1') || !contiene(out, 'DENTRO-2'),
      'il contesto e\' sopra il trigger con un episodio evacuabile, lo span si e\' riapplicato, e il contenuto ' +
      'dell\'episodio e\' ancora li\': il ramo dello span e\' tornato PRIMA di `runEvictionPass` e `reasoningFallback`, ' +
      'quindi un solo span nello stato spegne l\'evacuazione per il resto della sessione',
    );
  } finally { home.restore(); sandbox.cleanup(); }
});

test('se lo span basta a rientrare nel budget, l\'episodio NON viene toccato', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot(soloRimozione({ tokenBudget: 1500 }));
  try {
    const base = conversazione();
    await creaSpan(tools, hooks, ctx, base);

    // Episodio piccolo: dopo lo span il contesto rientra nel trigger, quindi non
    // c'e' niente da evacuare e l'evacuazione non deve toccare niente. E' la
    // assicurazione contro un fix troppo aggressivo ("evacua sempre").
    await tools.get('delimiter').execute('call-s', { action: 'start', name: 'piccolo', type: 'expl' }, undefined, undefined, ctx);
    await tools.get('delimiter').execute('call-e', { action: 'end', name: 'piccolo', description: 'imparato' }, undefined, undefined, ctx);
    const messages = [
      ...base,
      { role: 'user', content: 'apri' },
      chiamata('call-s'),
      { role: 'toolResult', toolCallId: 'call-s', toolName: 'delimiter', content: [{ type: 'text', text: 'aperto' }] },
      assistantPiccolo('PICCOLO'),
      chiamata('call-e'),
      { role: 'toolResult', toolCallId: 'call-e', toolName: 'delimiter', content: [{ type: 'text', text: 'chiuso' }] },
      { role: 'user', content: 'recente' },
    ];

    const out = await hook(hooks, ctx, messages);
    assert.ok(contiene(out, 'PICCOLO'), 'l\'episodio e\' stato evacuato mentre il contesto era gia\' rientrato: nessun motivo di toccarlo');
    // E la lista compressa DEVE essere quella restituita: senza il ritorno
    // anticipato nel ramo span, il flusso cade nella coda, che quando e' sotto
    // il trigger risponde `undefined` (nessuna modifica) e il provider riceve la
    // storia NON compressa. Il risparmio dello span andrebbe perso per quel turno.
    assert.ok(contiene(out, 'SINTESI-1'), 'lo span applicato non e\' stato restituito: la compressione e\' stata persa');
  } finally { home.restore(); sandbox.cleanup(); }
});
