import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { makeSandbox, bootExtension, withHome, sessionCtx } from './_helpers.mjs';

// IL BANCO DI PROVA E' UN CONFRONTO: lo stesso scenario con e senza chiusure automatiche.
// E' anche la prova di non-vacuita': l'arm "senza" E' la mutazione.
//
// Il budget e' volutamente PICCOLO: sotto il trigger l'hook ritorna prima di registrare l'intervallo
// comprimibile, quindi il pavimento non sarebbe osservabile. Sopra il trigger `storeRange` scrive
// `rangeTokens` e la riga RANGE nel log, ed e' da li' che si legge se il pavimento si e' mosso.
const base = {
  tokenBudget: 1_000,
  thresholdRatio: 1,
  protectedTurns: 3,
  autoCompactMinTokens: 0, // isola P2: qui non c'entra la compattazione continua
  gate: false,
  levels: { stripReasoning: false, stripBulkOutput: false, stripIntermediate: false, removeEpisode: false },
  showWidget: false,
  debug: true,
};

/**
 * QUATTRO turni utente e poi una CODA AUTONOMA di otto turni dell'agente, senza altri prompt.
 * E' esattamente la forma che rompeva tutto: la finestra "ultimi 3 turni utente" copre la coda
 * intera, quindi senza confini dentro la coda non c'e' niente da comprimere.
 */
const scenario = () => {
  const messages = [];
  let t = 0;
  const body = (tag, n = 900) => `${tag} ${'x'.repeat(n)}`;
  for (let turn = 1; turn <= 4; turn++) {
    messages.push({ role: 'user', content: body(`prompt ${turn}`), timestamp: ++t });
    messages.push({ role: 'assistant', content: body(`risposta ${turn}`), timestamp: ++t });
    messages.push({
      role: 'toolResult', toolCallId: `call-${turn}`, toolName: 'bash',
      content: [{ type: 'text', text: body(`output ${turn}`) }], timestamp: ++t,
    });
  }
  for (let step = 1; step <= 8; step++) {
    messages.push({ role: 'assistant', content: body(`lavoro ${step}`), timestamp: ++t });
    messages.push({
      role: 'toolResult', toolCallId: `auto-call-${step}`, toolName: 'bash',
      content: [{ type: 'text', text: body(`esito ${step}`) }], timestamp: ++t,
    });
  }
  return messages;
};

const tick = () => new Promise((resolve) => setImmediate(resolve));

async function run(name, autoDelimiterTurns) {
  const sandbox = makeSandbox({ name, config: { ...base, autoDelimiterTurns } });
  const home = withHome(sandbox.dir);
  const { hooks } = await bootExtension(sandbox);
  const ctx = sessionCtx(path.join(sandbox.dir, 'session.jsonl'));
  await hooks.get('session_start')({}, ctx);

  const all = scenario();
  let seen = 0;
  for (let round = 0; round < 10; round++) {
    seen = Math.min(all.length, seen + 3);
    await hooks.get('context')({ messages: all.slice(0, seen) }, ctx);
    await hooks.get('turn_end')({}, ctx);
    await tick();
  }
  const stateDir = path.join(sandbox.dir, '.pi', 'cwl', 'state');
  const file = fs.existsSync(stateDir)
    ? fs.readdirSync(stateDir).find((n) => n.endsWith('.json'))
    : undefined;
  const state = file ? JSON.parse(fs.readFileSync(path.join(stateDir, file), 'utf8')) : {};
  const log = fs.existsSync(path.join(sandbox.dir, '.pi', 'cwl', 'cwl.log'))
    ? fs.readFileSync(path.join(sandbox.dir, '.pi', 'cwl', 'cwl.log'), 'utf8')
    : '';
  // L'osservabile VERO e' la riga RANGE, non lo stato: l'arm di controllo non scrive stato
  // (nessun episodio automatico, nessun job in background) e leggere un file che non esiste
  // sarebbe un test che prova la cosa sbagliata.
  const ranges = [...log.matchAll(/RANGE source=hook-input \S+\.\.\S+ \(~(\d+)t\)/g)].map((m) => Number(m[1]));
  // Il pavimento si legge DIRETTAMENTE: "Nt in the protected window" e' la finestra inviolabile,
  // quindi se P2 funziona quel numero SCENDE a parita' di scenario.
  const prot = [...log.matchAll(/(\d+)t in the protected window/g)].map((m) => Number(m[1]));
  const logPath = path.join(sandbox.dir, '.pi', 'cwl', 'cwl.log');
  return {
    state, log, home,
    ranges,
    maxRange: ranges.length ? Math.max(...ranges) : 0,
    lastProtected: prot.length ? prot[prot.length - 1] : 0,
    logPath,
  };
}

test('le chiusure automatiche fanno scorrere il pavimento dentro una coda autonoma', async () => {
  const without = await run('p2-senza-automatici', 0);
  const withAuto = await run('p2-con-automatici', 3);

  // Le ancore devono essere REALI: gli episodi automatici esistono e sono ancorati a messaggi
  // che stanno in lista.
  const episodes = withAuto.state.graph?.episodes ?? [];
  const autoAnchors = episodes.filter((ep) => ep.name.startsWith('auto-')).map((ep) => ep.endToolCallId);
  assert.ok(autoAnchors.length > 0, `attesi episodi automatici, trovati ${episodes.map((e) => e.name).join(', ')}`);
  assert.ok(autoAnchors.every((id) => typeof id === 'string' && id.length > 0), 'ogni ancora deve esistere');

  // 1) IL PAVIMENTO SCENDE. A parita' di conversazione, con le chiusure automatiche la finestra
  //    inviolabile e' piu' piccola: e' letteralmente "un turno e' stato sprotetto".
  assert.ok(
    withAuto.lastProtected < without.lastProtected,
    `il pavimento deve scendere: ${withAuto.lastProtected}t protetti con le chiusure automatiche contro ${without.lastProtected}t senza`,
  );

  // 2) E di conseguenza si offre PIU' materiale da comprimere.
  assert.ok(
    withAuto.maxRange > without.maxRange,
    `P2 deve far offrire piu' materiale: ${withAuto.maxRange}t contro ${without.maxRange}t`,
  );
});

test('la riga RANGE dice quanti token sono stati sprotetti', async () => {
  const withAuto = await run('p2-riga-range', 3);
  assert.ok(withAuto.ranges.length > 0, 'la riga RANGE deve esistere: e lo strumento con cui si diagnostica perche non offre niente');
  assert.ok(withAuto.maxRange > 0, 'la riga deve dire quanti token sono stati sprotetti, non solo che esiste');
});
