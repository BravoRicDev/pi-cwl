/**
 * I CONTATORI: la pagina del nodo vecchio mostra le foglie CONSULTATE, non le piu'
 * recenti.
 *
 * LA REGOLA E' DELL'OPERATORE, e vale la pena riscriverla perche' e' controintuitiva:
 * *"le 30 assorbite piu' recentemente non mi interessano: 'consultate spesso', che
 * devono comunque rimanere visibili dentro il nodo vecchio. Le informazioni delle
 * foglie piu' recenti, se sono state usate, hanno comunque uno strascico di
 * ragionamento nelle 5 piu' giovani e nella parte intoccabile: non hanno bisogno di
 * stare anche li'. Sono le informazioni che SEMBRANO STANTIE che devono essere
 * visibili dentro il nodo vecchio senza andare a cercarle nel nodo giovane assorbito."*
 *
 * PERCHE' IL CONTATORE E' IL SEGNALE GIUSTO, anche se sembra bucato. Registra solo le
 * aperture ESPLICITE: una foglia usata quando era fresca non lascia traccia, perche'
 * il suo corpo era gia' nel contesto e nessuno doveva aprirla. Sembra un buco ed e'
 * il comportamento giusto — una foglia utile da fresca non ha bisogno di aiuto. Il
 * contatore misura esattamente l'altro caso: le foglie RI-CONSULTATE DOPO essere
 * invecchiate. Il buco e il caso che non richiede aiuto coincidono.
 *
 * E I CONTATORI NON SONO UNA POLITICA: ordinano una pagina e basta. Non cancellano
 * niente, perche' "mai aperta" non vuol dire "inutile" (una foglia puo' essere stata
 * usata benissimo da fresca). Nel fixture la foglia piu' VECCHIA viene aperta due
 * volte e la piu' recente zero: senza i contatori l'ordine per data metterebbe prima
 * la piu' recente, ed e' esattamente la direzione in cui il test deve morire.
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
  const sandbox = makeSandbox({ name: `contatori-${seq++}`, config: config() });
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

const logDi = (sandbox) => fs.readFileSync(path.join(sandbox.dir, '.pi', 'cwl', 'cwl.log'), 'utf8');

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

const testo = (res) => res.content.map((c) => c.text).join('\n');
const apri = (tools, ctx, id) => tools.get('cwl_open').execute('t', { id }, undefined, undefined, ctx);

test('la pagina del nodo vecchio ordina le foglie per consultazioni, non per data', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    for (let i = 1; i <= 5; i++) {
      await hook(hooks, ctx, conversazione(1, i + 3));
      const res = await tools.get('cwl_compress_range').execute(
        't', { summary: `CORPO-${i} ` + 'x'.repeat(300) }, undefined, undefined, ctx,
      );
      assert.equal(res.details.ok, true, `giro ${i}: la foglia non e' nata: ${JSON.stringify(res.details)}`);
    }
    const foglie = statoDi(sandbox).spans;
    for (let i = 0; i < 5; i++) {
      const r = await tools.get('cwl_micro').execute('t', { id: foglie[i].id, text: `MICRO-${i + 1}` }, undefined, undefined, ctx);
      assert.equal(r.details.ok, true, `micro sulla foglia ${i + 1} fallito: ${JSON.stringify(r.details)}`);
    }
    await hook(hooks, ctx, conversazione(1, 12));
    const acc = await tools.get('cwl_old').execute('t', { text: 'RIASSUNTONE-1' }, undefined, undefined, ctx);
    assert.equal(acc.details.ok, true, `il pozzo non si e' formato: ${JSON.stringify(acc.details)}`);
    const pozzo = acc.details.id;

    // 1. Storia VUOTA: la pagina elenca comunque le foglie del pozzo (ordine per
    //    data, il ripiego sensato) e dichiara che nessuna e' mai stata aperta.
    const fredda = await apri(tools, ctx, pozzo);
    assert.equal(fredda.details.ok, true, `il pozzo non si apre: ${JSON.stringify(fredda.details)}`);
    assert.equal(fredda.details.opened, 0, `a storia vuota nessuna foglia e' stata aperta, ne dichiara ${fredda.details.opened}`);
    assert.equal(fredda.details.leaves, 2, `il pozzo tiene 2 foglie, ne dichiara ${fredda.details.leaves}`);
    assert.match(testo(fredda), /opened 0x/, 'la pagina non dichiara quante volte ogni foglia e\' stata aperta');

    // 2. La foglia piu' VECCHIA viene aperta due volte.
    for (let k = 0; k < 2; k++) await apri(tools, ctx, foglie[0].id);

    // Il contatore vive con la foglia nello stato: e' questo che lo fa sopravvivere
    // al /reload, ed e' l'unica cosa che il test deve vedere per crederci.
    const dopoTurno = statoDi(sandbox).spans.find((s) => s.id === foglie[0].id);
    assert.equal(dopoTurno.opens, 2, `la foglia aperta due volte ne conta ${dopoTurno.opens}: il contatore non e' nello stato, quindi non sopravvive a un riavvio`);
    assert.ok(dopoTurno.lastOpen, 'manca l\'istante dell\'ultima apertura');

    // 3. Ora la pagina deve metterla PRIMA: e' la regola dell'operatore.
    const primaDellApertura = logDi(sandbox).length;
    const calda = await apri(tools, ctx, pozzo);
    const log = logDi(sandbox).slice(primaDellApertura);
    const righe = testo(calda).split('\n').filter((l) => l.startsWith('- sp-'));
    assert.ok(righe.length >= 2, `la pagina non elenca le foglie: ${testo(calda).slice(0, 200)}`);
    assert.ok(
      righe[0].includes(foglie[0].id),
      `la prima foglia elencata non e' quella piu' consultata: la pagina ordina per data, non per consultazioni. Prima riga: ${righe[0]}`,
    );
    assert.ok(righe[0].includes('opened 2x'), `la riga della foglia piu' consultata non dichiara le 2 aperture: ${righe[0]}`);
    assert.equal(calda.details.opened, 1, `le foglie mai aperte sono 1 (su 2), ne dichiara ${calda.details.opened}`);
    assert.match(
      log,
      /most opened 2x/,
      `il log non misura l'uso del pozzo (quante volte la foglia piu' aperta e' stata aperta): ${log.trim().split('\n').slice(-2).join(' | ')}`,
    );
  } finally {
    home.restore();
  }
});
