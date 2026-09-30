/**
 * #7: l'originale torna dal transcript quando lo stato ha potato la foglia.
 *
 * IL PROBLEMA, detto preciso. Un riassunto vive SOLO nello stato. Quando la
 * compattazione nativa porta via le ancore di uno span, `locateSpans` lo dichiara
 * morto e l'hook lo pota: da quel momento il suo riassunto non esiste piu'. I
 * MESSAGGI che aveva sostituito, invece, sono ancora nel transcript append-only.
 *
 * PERCHE' L'ID DA SOLO NON BASTAVA. L'id di una foglia e' `sp-` + un hash dei due
 * ancoraggi: e' una funzione a senso unico, quindi da un id non si risale agli
 * ancoraggi. Per questo `CompressedSpan` porta anche `startSid`/`endSid` — gli
 * stable id dei messaggi, `stableIdOf` — scritti quando gli ancoraggi erano ancora
 * visibili, e per questo la potatura li conserva in una tomba.
 *
 * E PERCHE' GLI HASH NON BASTAVANO: `addressOf` mescola l'id col TESTO, e questa
 * estensione fa lo stripping del ragionamento sui messaggi che comprime —
 * `contentToText` include `part.thinking`. Un hash calcolato dal transcript puo'
 * quindi non combaciare con quello calcolato dal contesto, e la ricerca fallirebbe
 * senza dire perche'. Il timestamp non deriva.
 *
 * E QUANDO NON SI TROVA NIENTE, LO DICE: c'e' `openOriginalLost`, che nomina
 * `cwl_recall`. Un limite silenzioso e' la promessa rotta in silenzio.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { makeSandbox, bootExtension, withHome, sessionCtx } from './_helpers.mjs';

let seq = 0;

const config = () => ({
  tokenBudget: 600,
  thresholdRatio: 0.5,
  protectedTurns: 0,
  levels: { stripReasoning: false, stripBulkOutput: false, stripIntermediate: false, removeEpisode: false },
  showWidget: false,
  debug: true,
});

async function boot() {
  const sandbox = makeSandbox({ name: `originale-${seq++}`, config: config() });
  const home = withHome(sandbox.dir);
  const { tools, hooks } = await bootExtension(sandbox);
  const ctx = sessionCtx(path.join(sandbox.dir, 'sessione.jsonl'));
  await hooks.get('session_start')({}, ctx);
  return { sandbox, home, tools, hooks, ctx };
}

const hook = async (hooks, ctx, messages) => {
  const res = await hooks.get('context')({ messages }, ctx);
  return (res && res.messages) || messages;
};

const statoDi = (sandbox) => {
  const dir = path.join(sandbox.dir, '.pi', 'cwl', 'state');
  const file = fs.readdirSync(dir).find((f) => f.endsWith('.json'));
  assert.ok(file, 'lo stato della sessione non e\' stato creato');
  return JSON.parse(fs.readFileSync(path.join(dir, file), 'utf8'));
};

const testo = (res) => res.content.map((c) => c.text).join('\n');

/** Messaggi con un timestamp: e' quello che `stableIdOf` usa come identita'. */
const conversazione = () => {
  const out = [];
  for (let i = 1; i <= 6; i++) {
    out.push({ role: 'user', content: `turno ${i} contenuto ` + 'U'.repeat(200), timestamp: 1000 + i * 2 });
    out.push({ role: 'assistant', content: `risposta ${i} contenuto ` + 'A'.repeat(200), timestamp: 1001 + i * 2 });
  }
  return out;
};

test('una foglia potato si riapre dal transcript, e lo dichiara', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    const conv = conversazione();
    await hook(hooks, ctx, conv);
    const comp = await tools.get('cwl_compress_range').execute('t', { summary: 'SINTESI-A dei primi turni' }, undefined, undefined, ctx);
    assert.equal(comp.details.ok, true, `la foglia non e' nata: ${JSON.stringify(comp.details)}`);

    // Un turno per far RISOLVERE lo span: e' li' che gli stable id vengono scritti.
    await hook(hooks, ctx, conv);
    const sp = statoDi(sandbox).spans[0];
    assert.ok(
      sp && sp.startSid && sp.endSid,
      `la foglia non ha gli stable id: senza di loro, dopo la potatura, l'id non potra' piu' ritrovare niente. Trovato: ${JSON.stringify(sp)}`,
    );

    // Il transcript, come lo scrive Pi: un record per messaggio, col timestamp.
    const righe = conv.map((m) =>
      JSON.stringify({ message: { role: m.role, timestamp: m.timestamp, content: [{ type: 'text', text: String(m.content) }] } }),
    );
    fs.writeFileSync(path.join(sandbox.dir, 'sessione.jsonl'), righe.join('\n') + '\n');

    // Una lista che NON contiene piu' gli ancoraggi: la compattazione nativa ha
    // sostituito quel prefisso, quindi lo span e' morto e viene potato.
    const altrove = [{ role: 'user', content: 'la storia e\' andata avanti, questo prefisso non c\'e\' piu\'', timestamp: 999999 }];
    await hook(hooks, ctx, altrove);
    const st = statoDi(sandbox);
    assert.equal(st.spans.length, 0, `la foglia doveva essere potato, ne restano ${st.spans.length}`);
    assert.ok(
      st.graves && st.graves.length === 1,
      `la potatura non ha lasciato una tomba con gli ancoraggi: ${JSON.stringify(st.graves)} — da qui in poi l'id non trova piu' niente`,
    );
    assert.equal(st.graves[0].id, sp.id, 'la tomba non corrisponde alla foglia potato');

    // E ora: l'id, che era stato potato, restituisce l'ORIGINALE.
    const res = await tools.get('cwl_open').execute('t', { id: sp.id }, undefined, undefined, ctx);
    assert.equal(res.details.ok, true, `la foglia potato non si riapre: ${JSON.stringify(res.details)}`);
    assert.equal(res.details.kind, 'original', `il tool non dichiara che sta consegnando l'originale: ${JSON.stringify(res.details)}`);
    assert.ok(
      testo(res).includes('turno 1 contenuto'),
      `il testo originale non e' tornato: ${testo(res).slice(0, 200)}`,
    );
    assert.ok(
      !testo(res).includes('SINTESI-A'),
      'il tool ha consegnato il RIASSUNTO: ma il riassunto e\' andato perso con lo stato, quindi o lo sta inventando o non ha capito cosa gli e\' stato chiesto',
    );
  } finally {
    home.restore();
  }
});
