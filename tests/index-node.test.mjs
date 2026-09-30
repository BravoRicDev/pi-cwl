/**
 * Il NODO di livello 1: un contenitore di foglie, e niente di piu' — per ora.
 *
 * A COSA SERVE, E PERCHE' NON SI VEDE ANCORA. Il nodo non cambia cio' che sta nel
 * contesto: una foglia assorbita mostra il suo micro dovunque sia, e il nodo si
 * limita a RAGGRUPPARLA con le altre (5 restano sciolte, le piu' vecchie entrano).
 * Quello che il nodo abilita e' il passo successivo: quando i nodi sono tre, i due
 * piu' vecchi si accorgono in un NODO VECCHIO e 60 micro (18k token) diventano UN
 * riassuntone (300-500t). Li' sta il risparmio; qui stanno le fondamenta.
 *
 * GLI INVARIANTI CHE QUESTO TEST FISSA (sono quelli che il nodo vecchio ereditera'):
 *  1. ogni foglia vive in AL MOLO UN nodo: se una foglia finisse in due posti, il
 *     contenuto sarebbe contato due volte e la promessa "non si perde niente"
 *     diventerebbe "non si perde niente, ma qualcosa si paga due volte";
 *  2. le foglie di un nodo sono le PIU' VECCHIE oltre le 5 sciolte, e in ordine
 *     cronologico: l'ordine della cima e' l'ordine della storia;
 *  3. una foglia che esce dallo stato esce anche dal nodo, o il nodo racconterebbe
 *     materiale che non esiste piu';
 *  4. una foglia vecchia SENZA micro non viene assorbita e viene DICHIARATA: il suo
 *     corpo e' ancora nel contesto, ed e' un costo che si dice, non si nasconde.
 *
 * Il test guida l'hook DUE volte dopo aver scritto i micro: la seconda serve perche'
 * un difetto di idempotenza (una foglia rimessa nel nodo a ogni turno) si vede solo
 * al secondo giro. E' il verso in cui il test morde davvero.
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
  const sandbox = makeSandbox({ name: `nodo-${seq++}`, config: config() });
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

test('le foglie piu\' vecchie entrano in UN nodo, una volta sola e in ordine', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    // Otto foglie. Ogni giro: la conversazione cresce (cosi' esiste una regione
    // comprimibile nuova) e si comprime quella regione.
    for (let i = 1; i <= 8; i++) {
      await hook(hooks, ctx, conversazione(1, i + 3));
      const res = await tools.get('cwl_compress_range').execute(
        't', { summary: `CORPO-${i} ` + 'x'.repeat(300) }, undefined, undefined, ctx,
      );
      assert.equal(res.details.ok, true, `giro ${i}: la foglia non e' nata: ${JSON.stringify(res.details)}`);
      assert.equal(
        statoDi(sandbox).spans.length,
        i,
        `giro ${i}: le foglie nello stato dovrebbero essere ${i}, non ${statoDi(sandbox).spans.length}`,
      );
    }

    const foglie = statoDi(sandbox).spans;
    assert.equal(foglie.length, 8, 'servono otto foglie: con meno, le 5 sciolte coprono tutto e il nodo non ha oggetto');

    // Micro alle due PIU' VECCHIE e non alla terza: cosi' il nodo ha 2 foglie e la
    // terza e' quella che "aspetta un micro".
    for (const [i, t] of [[0, 'MICRO-1: la prima storia'], [1, 'MICRO-2: la seconda storia']]) {
      const r = await tools.get('cwl_micro').execute('t', { id: foglie[i].id, text: t }, undefined, undefined, ctx);
      assert.equal(r.details.ok, true, `micro sulla foglia ${i + 1} fallito: ${JSON.stringify(r.details)}`);
    }

    // Due turni: il secondo e' quello che smaschera un difetto di idempotenza.
    await hook(hooks, ctx, conversazione(1, 12));
    const prima = logDi(sandbox).length;
    await hook(hooks, ctx, conversazione(1, 12));
    const log = logDi(sandbox).slice(prima);

    // Il nodo NON si legge dal file: e' DERIVATO (refreshNodes lo ricostruisce da
    // foglie e micro a ogni turno), quindi l'unico modo onesto di osservarlo e'
    // cio' che l'estensione DICE — il log — e cio' che il tool CONSEGNA.
    const rigaNodi = /NODES: (\d+) node\(s\) \[([^\]]*)\]/.exec(log);
    assert.ok(
      rigaNodi,
      `il turno non dichiara lo stato dei nodi: senza quella riga non si sa quali foglie siano state raggruppate. Log del turno: ${log.trim().split('\n').slice(-3).join(' | ')}`,
    );
    assert.equal(Number(rigaNodi[1]), 1, `atteso UN nodo, dichiarati ${rigaNodi[1]}: ${rigaNodi[2]}`);

    const ndId = /^(nd-[0-9a-f]{8}):(\d+)$/.exec(rigaNodi[2].trim());
    assert.ok(ndId, `la riga dei nodi non nomina il nodo e la sua dimensione: "${rigaNodi[2]}"`);
    assert.equal(
      Number(ndId[2]),
      2,
      `il nodo deve tenere le due foglie piu' vecchie (le uniche vecchie con un micro), ne dichiara ${ndId[2]}`,
    );

    // La pagina del nodo: i micro delle sue foglie, con gli id che le aprono.
    const aperto = await tools.get('cwl_open').execute('t', { id: ndId[1] }, undefined, undefined, ctx);
    assert.equal(aperto.details.ok, true, `il nodo non si apre: ${JSON.stringify(aperto.details)}`);
    assert.equal(aperto.details.leaves, 2, `la pagina del nodo dichiara ${aperto.details.leaves} foglie invece di 2`);

    // 1. Le due foglie del nodo sono ESATTAMENTE le due piu' vecchie, in ordine
    //    cronologico, e nessuna compare due volte: se una foglia stesse in due
    //    posti il contenuto sarebbe contato due volte.
    const idsPagina = [...testo(aperto).matchAll(/(sp-[0-9a-f]{8})/g)].map((m) => m[1]);
    assert.deepEqual(
      idsPagina,
      [foglie[0].id, foglie[1].id],
      `la pagina del nodo deve nominare le due foglie piu' vecchie in ordine: ${JSON.stringify(idsPagina)}`,
    );
    assert.ok(
      testo(aperto).includes('MICRO-1') && testo(aperto).includes('MICRO-2'),
      `i micro non sono nella pagina: ${testo(aperto).slice(0, 160)}`,
    );

    // 4. La foglia vecchia senza micro e' DICHIARATA, non nascosta.
    assert.match(
      log,
      /NODES: .*waiting for a micro/,
      'la foglia vecchia senza micro non e\' dichiarata: il suo corpo e\' ancora nel contesto e nessuno lo dice',
    );

    // 5. La via di ritorno passa ANCHE dal nodo: togliendo il micro, la foglia esce.
    const via = await tools.get('cwl_micro').execute('t', { id: foglie[0].id, text: '' }, undefined, undefined, ctx);
    assert.equal(via.details.ok, true, `la rimozione del micro e' fallita: ${JSON.stringify(via.details)}`);
    // L'offset si prende QUI, non prima: da `prima` in poi ci sono DUE turni, e
    // `exec` restituirebbe la riga del primo — che dice ancora :2. E' la terza volta
    // oggi che questa trappola morde: si legge il tratto di log del turno che si sta
    // misurando, mai "dal punto X in poi".
    const primaDelRitorno = logDi(sandbox).length;
    await hook(hooks, ctx, conversazione(1, 12));
    const dopo = /NODES: (\d+) node\(s\) \[([^\]]*)\]/.exec(logDi(sandbox).slice(primaDelRitorno));
    assert.ok(dopo, 'dopo la rimozione del micro il turno non dichiara piu\' lo stato dei nodi');
    assert.equal(
      dopo[2].trim(),
      `${ndId[1]}:1`,
      `togliendo il micro la foglia e' rimasta nel nodo: ${dopo[2]} — il corpo e' tornato nel contesto, quindi il nodo la conterebbe due volte`,
    );
  } finally {
    home.restore();
  }
});
