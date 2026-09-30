/**
 * IL PRIMO PASSO DELL'INDICE DEVE CHIEDERE, NON SOLO REGISTRARE.
 *
 * MISURATO DAL VIVO, subito dopo la ricarica che ha fatto girare l'indice:
 *
 *     NODES: 0 node(s) [none], 40 leaf/leaves waiting for a micro — their body is still in
 *     the context
 *
 * a OGNI turno, con 45 span e 90.415t di riassunti ancora nel contesto. Cioe': il codice
 * girava, ma il meccanismo NON poteva partire. La catena e': le foglie senza micro non
 * entrano in un nodo -> zero nodi -> `piano.due` e' zero -> la richiesta del merge
 * (`indexDue`) non compare MAI. E del micro nessuno parlava: il conteggio finiva nel LOG, non
 * nel contesto. L'estensione lo sapeva, l'operatore poteva leggerlo, e l'unico che puo'
 * scrivere un micro — l'agente — non riceveva nessuna domanda.
 *
 * E' esattamente il difetto che `bd0734c` ha chiuso per `cwl_old` (il riassuntone), rimasto
 * aperto sul passo PRIMA. La richiesta deve dire due cose per essere azionabile: che si deve
 * chiamare `cwl_micro`, e SU QUALI foglie — senza gli id l'agente deve indovinare quale, o
 * andare a leggere il file di stato.
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
  const sandbox = makeSandbox({ name: `micro-${seq++}`, config: config() });
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
    out.push({ role: 'user', content: `turno ${i} contenuto ` + 'U'.repeat(200), timestamp: 1000 + i * 2 });
    out.push({ role: 'assistant', content: `risposta ${i} contenuto ` + 'A'.repeat(200), timestamp: 1001 + i * 2 });
  }
  return out;
};

test('la richiesta del micro arriva all\'agente, e nomina la foglia', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    // SETTE foglie, non una. Con `looseLeaves` = 5 le ultime cinque restano SCIOLTE e non
    // devono ancora un micro: la richiesta nasce solo quando una foglia esce da quella
    // finestra, ed e' esattamente il caso che conta (il suo corpo deve lasciare il
    // contesto). Il mio primo fixture ne faceva UNA e il test era rosso per questo, non per
    // il codice: la lezione di sempre, il difetto era nel test.
    for (let i = 1; i <= 7; i++) {
      await hook(hooks, ctx, conversazione(1, i + 3));
      const res = await tools.get('cwl_compress_range').execute(
        't', { summary: `CORPO-${i} ` + 'x'.repeat(300) }, undefined, undefined, ctx,
      );
      assert.equal(res.details.ok, true, `giro ${i}: la foglia non e' nata: ${JSON.stringify(res.details)}`);
    }
    assert.equal(
      statoDi(sandbox).spans.length,
      7,
      'servono sette foglie: con cinque o meno sono tutte sciolte e nessuna aspetta un micro',
    );

    // Il turno dopo: le foglie piu' vecchie non sono piu' sciolte, non hanno un micro, e
    // l'agente deve SAPERLO.
    const out = await hook(hooks, ctx, conversazione(1, 10));
    const richieste = out.filter((m) => m.customType === 'cwl-demand');
    assert.ok(
      richieste.length > 0,
      'nessuna richiesta nel contesto: le foglie senza micro finiscono solo nel LOG, quindi l\'agente non sa che deve ' +
        `scriverli e il meccanismo non parte (0 nodi -> piano.due = 0 -> la richiesta del merge non compare mai). Messaggi: ${out
          .map((m) => m.customType || m.role)
          .join(', ')}`,
    );

    const testo = richieste.map((m) => String(m.content)).join('\n');
    assert.ok(
      testo.includes('cwl_micro'),
      `la richiesta non nomina lo strumento da usare, quindi non e' azionabile: ${testo.slice(0, 200)}`,
    );

    const id = statoDi(sandbox).spans[0].id;
    assert.ok(
      testo.includes(id),
      `la richiesta non nomina la foglia ${id}: senza l'id l'agente deve indovinare quale, o andare a leggere il file di ` +
        `stato. Testo: ${testo.slice(0, 300)}`,
    );
  } finally {
    home.restore();
  }
});
