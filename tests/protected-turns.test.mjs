/**
 * IN UNA SESSIONE AUTONOMA I TURNI NON SONO SOLO I MIEI.
 *
 * La richiesta dell'operatore, parola per parola: *"quando setto bene gli agenti autonomi io
 * non scrivo per giorni interi, la memory card e cronjob devono contare come 'turni'
 * esattamente come fossero miei. Altrimenti questo non funzionera' mai."*
 *
 * PERCHE' AVEVA RAGIONE. `protectedFromIndex` risaliva la lista contando SOLO i messaggi
 * `user`: la finestra partiva dall'ultimo prompt digitato dall'operatore. In una sessione
 * autonoma quel prompt e' di GIORNI prima, quindi tutto cio' che veniva dopo — giorni di
 * lavoro, risvegli, carte — finiva dentro la finestra protetta. E quando la lista ha meno
 * turni `user` della finestra, scatta il ramo degenerato: il pavimento va all'80% della
 * lista e NESSUNA delle tre vie di compressione puo' piu' toccare niente.
 * MISURATO dal vivo, nella sessione dell'operatore: `CONTEXT 436837t ... 348650t in the
 * protected window` — 348.650 su 436.837 e' l'80%, cioe' esattamente quel ramo.
 *
 * IL FIX. Un confine di turno e' un messaggio `user` OPPURE un'iniezione di UN'ALTRA
 * estensione: il risveglio di un cronjob (`background-task-notification`) e la carta
 * (`anti-amnesia`) aprono un turno esattamente come un prompt. Le iniezioni NOSTRE no: un
 * riepilogo sta dove stavano i messaggi compressi, e la richiesta dell'indice sta in FONDO
 * alla lista — contarle sposterebbe la finestra sulla cosa sbagliata.
 *
 * LE DUE COSE CHE IL TEST PRETENDE:
 *  1. la finestra protetta NON copre quasi tutto (con i soli `user` coprirebbe l'80%);
 *  2. la compressione e' POSSIBILE: `cwl_compress_range` deve riuscire. E' la prova che
 *     conta per l'operatore — "non resta niente da comprimere" e' il sintomo che lo ha
 *     portato a chiedere questo fix.
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
  protectedTurns: 4,
  levels: { stripReasoning: false, stripBulkOutput: false, stripIntermediate: false, removeEpisode: false },
  showWidget: false,
  debug: true,
});

async function boot() {
  const sandbox = makeSandbox({ name: `turni-${seq++}`, config: config() });
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

/** Sedici turni autonomi dopo l'ultimo prompt dell'operatore, che e' di giorni prima. */
const listaAutonoma = () => {
  const out = [
    { role: 'user', content: 'operatore, giorni fa ' + 'U'.repeat(300), timestamp: 1 },
    { role: 'assistant', content: 'risposta di allora ' + 'A'.repeat(300), timestamp: 2 },
  ];
  for (let i = 1; i <= 16; i++) {
    // Un risveglio di cronjob: per l'estensione e' un'iniezione `custom`, ma per la
    // conversazione e' l'inizio di un turno come un prompt.
    out.push({ role: 'custom', customType: 'background-task-notification', content: `risveglio ${i}`, timestamp: 100 + i * 3 });
    out.push({ role: 'assistant', content: `lavoro autonomo ${i} ` + 'A'.repeat(300), timestamp: 101 + i * 3 });
    out.push({ role: 'toolResult', content: `output ${i} ` + 'T'.repeat(300), timestamp: 102 + i * 3 });
  }
  // E in fondo una NOSTRA iniezione: la richiesta dell'indice. Non e' un turno: se
  // contasse, il pavimento scivolerebbe indietro di un turno ogni volta che la si chiede.
  out.push({ role: 'custom', customType: 'cwl-compressed', content: 'riepilogo nostro', timestamp: 9000 });
  return out;
};

test('un turno autonomo (risveglio, carta) conta come un turno dell\'operatore', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    const lista = listaAutonoma();
    const prima = logDi(sandbox).length;
    await hook(hooks, ctx, lista);
    const log = logDi(sandbox).slice(prima);

    const riga = /CONTEXT (\d+)t still above trigger \d+t: (\d+)t in the protected window/.exec(log);
    assert.ok(
      riga,
      `il turno non dichiara il pavimento (servono le due cifre per misurare): ${log.trim().split('\n').slice(-3).join(' | ')}`,
    );
    const totale = Number(riga[1]);
    const protetto = Number(riga[2]);
    const quota = protetto / totale;
    assert.ok(
      quota < 0.5,
      `la finestra protetta copre ${protetto}t di ${totale}t (${Math.round(quota * 100)}%): con i soli messaggi \`user\` il ` +
        'pavimento scivola sull\'ultimo prompt dell\'operatore — che in una sessione autonoma e\' di giorni prima — e resta ' +
        'protetto quasi tutto il contesto. I risvegli e la carta devono contare come turni.',
    );

    // 2. E la prova che per l'operatore conta: si deve poter comprimere.
    const range = await tools.get('cwl_compress_range').execute('t', { summary: 'SINTESI-A' }, undefined, undefined, ctx);
    assert.equal(
      range.details.ok,
      true,
      `"non resta niente da comprimere": e' il sintomo che ha portato a questo fix — ${JSON.stringify(range.details)}`,
    );
  } finally {
    home.restore();
  }
});

/**
 * Quattro turni autonomi e, in fondo, QUATTRO iniezioni nostre.
 *
 * Serve un caso dove la differenza si VEDE: con pochi turni, se le nostre iniezioni
 * contassero come confini il pavimento scivolerebbe di quattro turni in un colpo solo.
 */
const listaCorta = () => {
  const out = [
    { role: 'user', content: 'operatore, giorni fa ' + 'U'.repeat(300), timestamp: 1 },
    { role: 'assistant', content: 'risposta di allora ' + 'A'.repeat(300), timestamp: 2 },
  ];
  for (let i = 1; i <= 4; i++) {
    out.push({ role: 'custom', customType: 'background-task-notification', content: `risveglio ${i}`, timestamp: 100 + i * 3 });
    out.push({ role: 'assistant', content: `lavoro autonomo ${i} ` + 'A'.repeat(400), timestamp: 101 + i * 3 });
    out.push({ role: 'toolResult', content: `output ${i} ` + 'T'.repeat(400), timestamp: 102 + i * 3 });
  }
  for (const ct of ['cwl-compressed', 'cwl-demand', 'cwl-compressed', 'cwl-demand']) {
    out.push({ role: 'custom', customType: ct, content: 'iniezione nostra', timestamp: 9000 });
  }
  return out;
};

test('le iniezioni NOSTRE non contano come turni', async () => {
  const { sandbox, home, hooks, ctx } = await boot();
  try {
    const prima = logDi(sandbox).length;
    await hook(hooks, ctx, listaCorta());
    const log = logDi(sandbox).slice(prima);
    const riga = /CONTEXT (\d+)t still above trigger \d+t: (\d+)t in the protected window/.exec(log);
    assert.ok(riga, `il turno non dichiara il pavimento: ${log.trim().split('\n').slice(-3).join(' | ')}`);
    const quota = Number(riga[2]) / Number(riga[1]);
    assert.ok(
      quota > 0.5,
      `la finestra protetta copre solo il ${Math.round(quota * 100)}% del contesto: le quattro iniezioni NOSTRE in fondo ` +
        'alla lista stanno contando come turni, quindi il pavimento scivola di quattro turni e i turni autonomi veri restano ' +
        'scoperti. Una richiesta dell\'indice o un riepilogo non aprono un turno: stanno dove sta la compressione, o in fondo ' +
        'alla lista perche\' servono adesso.',
    );
  } finally {
    home.restore();
  }
});
