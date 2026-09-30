/**
 * IL GRILLETTO: la richiesta di scrivere il riassuntone deve arrivare all'AGENTE.
 *
 * Questo test esiste per un difetto che non era un difetto del codice ma del canale:
 * il merge del nodo vecchio funzionava ed era coperto da test, e in una sessione vera
 * NON poteva scattare. `OLD NODE due: ...` era una riga di LOG: l'estensione lo sapeva,
 * l'operatore poteva leggerlo, e l'unico che puo' scrivere la sintesi — l'agente — non
 * vedeva niente. Un meccanismo che nessuno puo' azionare non e' un meccanismo.
 *
 * La richiesta va dove va gia' la richiesta di comprimere: nel CONTESTO. E come per
 * quella, l'estensione non inventa la sintesi: chiede, e aspetta.
 *
 * LE TRE COSE CHE IL TEST PRETENDE:
 *  1. quando il merge NON e' dovuto la richiesta NON c'e' — un avviso perenne e' rumore
 *     che insegna all'agente a ignorare gli avvisi;
 *  2. quando e' dovuto, la richiesta c'e' ed e' nel contesto che il provider riceve;
 *  3. dopo l'accorpamento sparisce, perche' non c'e' piu' niente da chiedere.
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
  looseLeaves: 1,
  nodeCapacity: 2,
  mergeNodesAt: 2,
});

async function boot() {
  const sandbox = makeSandbox({ name: `grilletto-${seq++}`, config: config() });
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

const conversazione = (da, a) => {
  const out = [];
  for (let i = da; i <= a; i++) {
    out.push({ role: 'user', content: `turno ${i} contenuto ` + 'U'.repeat(200) });
    out.push({ role: 'assistant', content: `risposta ${i} contenuto ` + 'A'.repeat(200) });
  }
  return out;
};

/** La richiesta, se c'e', come la vedrebbe il provider. */
const richiesta = (msgs) =>
  msgs.filter((m) => m && m.customType === 'cwl-demand').map((m) => String(m.content)).join('\n');

test('la richiesta di accorpamento arriva nel contesto, e solo quando e\' dovuta', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    // 1. Una foglia sola: non c'e' nessun nodo, quindi niente da chiedere.
    await hook(hooks, ctx, conversazione(1, 4));
    const uno = await tools.get('cwl_compress_range').execute('t', { summary: 'CORPO-1' }, undefined, undefined, ctx);
    assert.equal(uno.details.ok, true, `la prima foglia non e' nata: ${JSON.stringify(uno.details)}`);
    const conUna = await hook(hooks, ctx, conversazione(1, 6));
    assert.equal(
      richiesta(conUna).length,
      0,
      `la richiesta c'e' gia' con UNA foglia e nessun nodo: un avviso perenne e' rumore, e il rumore insegna a ignorare gli avvisi. Trovato: ${richiesta(conUna).slice(0, 160)}`,
    );

    // 2. Cinque foglie con micro: si formano due nodi giovani, e il merge diventa dovuto.
    for (let i = 2; i <= 5; i++) {
      await hook(hooks, ctx, conversazione(1, i + 3));
      const res = await tools.get('cwl_compress_range').execute(
        't', { summary: `CORPO-${i} ` + 'x'.repeat(300) }, undefined, undefined, ctx,
      );
      assert.equal(res.details.ok, true, `giro ${i}: la foglia non e' nata: ${JSON.stringify(res.details)}`);
    }
    const foglie = statoDi(sandbox).spans;
    for (let i = 0; i < foglie.length; i++) {
      await tools.get('cwl_micro').execute('t', { id: foglie[i].id, text: `MICRO-${i + 1}` }, undefined, undefined, ctx);
    }

    const dovuto = await hook(hooks, ctx, conversazione(1, 12));
    const testo = richiesta(dovuto);
    assert.ok(
      testo.length > 0,
      'il merge e\' dovuto e la richiesta NON e\' nel contesto: l\'estensione lo sa, l\'operatore lo legge nel log, e l\'agente — l\'unico che puo\' scrivere la sintesi — non lo sapra\' mai',
    );
    assert.match(testo, /cwl_old/, `la richiesta non dice cosa fare (cwl_old): ${testo.slice(0, 200)}`);
    assert.match(testo, /2/, `la richiesta non dice QUANTI nodi sono da accorpare: ${testo.slice(0, 200)}`);

    // 3. Accorpato: il merge non e' piu' dovuto, e la richiesta sparisce.
    const acc = await tools.get('cwl_old').execute('t', { text: 'RIASSUNTONE-1' }, undefined, undefined, ctx);
    assert.equal(acc.details.ok, true, `l'accorpamento e' fallito: ${JSON.stringify(acc.details)}`);
    const dopo = await hook(hooks, ctx, conversazione(1, 12));
    assert.equal(
      richiesta(dopo).length,
      0,
      `la richiesta e' rimasta dopo l'accorpamento: ${richiesta(dopo).slice(0, 200)} — l'agente rifarebbe un lavoro gia' fatto`,
    );
  } finally {
    home.restore();
  }
});
