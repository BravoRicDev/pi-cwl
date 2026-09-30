/**
 * L'ETICHETTA NASCE CON LA FOGLIA, NON IN UN SECONDO PASSAGGIO.
 *
 * Richiesta dell'operatore: *"Possiamo fare in modo che quando viene creata la 'foglia' si
 * crei gia' la sua 'descrizione/etichetta'? cosi' quando va sparata nel nodo e' gia' pronta e
 * non dobbiamo fare batch da 5 foglie al colpo."*
 *
 * Chi scrive il riassunto di compressione ha appena letto i messaggi: il micro (~200 parole,
 * l'etichetta che l'indice mostra al posto del corpo) e' quasi gratis in quel momento.
 * Chiederlo dopo significa un secondo passaggio su un corpo che nel frattempo vive solo nello
 * stato, e un lotto di cinque foglie da etichettare tutte insieme.
 *
 * Due cose il test pretende, e sono due invarianti diversi:
 *  1. con l'etichetta alla nascita le foglie NON aspettano piu' niente (`0 leaf/leaves
 *     waiting`): il nodo si forma da solo e nessuno deve chiedere il micro;
 *  2. il micro NON entra nel corpo. `cwl_open` deve continuare a restituire il riassunto
 *     intero e pulito: un'etichetta scritta DENTRO il corpo farebbe diventare falsa, con una
 *     riga, la promessa "non si perde niente".
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
  const sandbox = makeSandbox({ name: `etichetta-${seq++}`, config: config() });
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
    out.push({ role: 'user', content: `turno ${i} contenuto ` + 'U'.repeat(200), timestamp: 1000 + i * 2 });
    out.push({ role: 'assistant', content: `risposta ${i} contenuto ` + 'A'.repeat(200), timestamp: 1001 + i * 2 });
  }
  return out;
};

/** La foglia nasce COL suo corpo e la sua etichetta, nella stessa chiamata. */
const ETICHETTA = (i) => `ETICHETTA-${i}: la foglia ${i} racconta il turno ${i} e cio' che si e' deciso li'`;

test('la foglia nasce con la sua etichetta, e il corpo resta intero', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    for (let i = 1; i <= 7; i++) {
      await hook(hooks, ctx, conversazione(1, i + 3));
      const res = await tools.get('cwl_compress_range').execute(
        't',
        { summary: `CORPO-${i} ` + 'x'.repeat(300), micro: ETICHETTA(i) },
        undefined,
        undefined,
        ctx,
      );
      assert.equal(res.details.ok, true, `giro ${i}: la foglia non e' nata: ${JSON.stringify(res.details)}`);
    }

    const foglie = statoDi(sandbox).spans;
    assert.equal(foglie.length, 7, 'servono sette foglie: con `looseLeaves` = 5 le ultime cinque restano sciolte');
    assert.ok(
      foglie.slice(0, 2).every((f) => f.micro && f.micro.includes('ETICHETTA')),
      'le etichette non sono state salvate sulla foglia: il parametro e\' arrivato e non e\' stato scritto',
    );

    // Il turno dopo le foglie piu' vecchie escono dalla finestra `looseLeaves`.
    const prima = logDi(sandbox).length;
    const out = await hook(hooks, ctx, conversazione(1, 10));
    const log = logDi(sandbox).slice(prima);

    // 1. Nessuno aspetta piu' niente: le etichette c'erano gia'.
    const riga = /NODES: (\d+) node\(s\) \[([^\]]*)\], (\d+) leaf\/leaves waiting/.exec(log);
    assert.ok(riga, `il turno non dichiara i nodi: ${log.trim().split('\n').slice(-3).join(' | ')}`);
    assert.equal(
      Number(riga[3]),
      0,
      `le foglie aspettano ancora un micro (${riga[3]} in attesa): le etichette non sono state usate alla nascita, ` +
        'quindi il nodo non si forma e l\'estensione deve chiederle dopo — il lotto da cinque che si voleva evitare.',
    );
    assert.ok(Number(riga[1]) >= 1, `nessun nodo formato (${riga[1]}): le foglie con etichetta non sono entrate in un nodo`);

    // 2. Il corpo resta intero: l'etichetta non e' finita dentro di lui.
    const aperta = await tools.get('cwl_open').execute('t', { id: foglie[0].id }, undefined, undefined, ctx);
    const testo = aperta.content.map((c) => c.text).join('\n');
    assert.ok(
      testo.includes('CORPO-1 '),
      `cwl_open non restituisce piu' il corpo della foglia: ${testo.slice(0, 200)}`,
    );
    assert.ok(
      !testo.includes('ETICHETTA-1'),
      'il micro e\' stato scritto DENTRO il corpo: `cwl_open` non restituisce piu\' l\'originale, e la promessa ' +
        '"non si perde niente" diventa falsa. Il micro e\' un campo separato, mai una modifica del corpo.',
    );

    // 3. E nessuno chiede il micro, perche' non c'e' niente da chiedere.
    const richieste = out.filter((m) => m.customType === 'cwl-demand').map((m) => String(m.content)).join('\n');
    assert.ok(
      !richieste.includes('cwl_micro'),
      `l'estensione chiede ancora i micro che sono gia' arrivati: ${richieste.slice(0, 200)}`,
    );
  } finally {
    home.restore();
  }
});
