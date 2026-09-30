/**
 * The context floor must be DECLARED, with numbers that add up.
 *
 * THE PROBLEM. `finish` already knows how to reject the impossible request (it
 * checks `canClose`/`canCompress` before asking to compress), but it never said
 * WHY there was nothing to hand over. The operator read "the context is above
 * budget" and could not tell a broken extension from an inviolable context.
 * MEASURED live: 184,490 tokens against a trigger of 68,000 with
 * only 6,307 tokens of compressible range — the rest was the protected window
 * (the last 10 `user` turns, the default the operator asked for) and the
 * content the spans hold in place.
 *
 * THE CONTRACT. The line `CONTEXT ...: Nt in the protected window (last K user
 * turns), Mt inside the spans, Ft freely compressible, Et elsewhere` is a
 * PARTITION of the active context: the four numbers add up to the total. It is
 * arithmetic, not a story, and this test checks exactly the identity — so that
 * a slip in the count (counting the spans twice, forgetting the window, not
 * subtracting `free` from `elsewhere`) becomes a red test instead of a number
 * that looks plausible.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { makeSandbox, bootExtension, withHome, sessionCtx } from './_helpers.mjs';

let seq = 0;

/**
 * Low trigger (300) and a ONE-turn window: so after the span it stays above
 * the trigger (the line appears) and the protected window really holds something
 * (otherwise the test would pass with `protectedTokens = 0` by vacuity).
 * All levels false: no fallback, the context does not drop by other routes.
 */
const config = () => ({
  tokenBudget: 600,
  thresholdRatio: 0.5,
  protectedTurns: 1,
  levels: { stripReasoning: false, stripBulkOutput: false, stripIntermediate: false, removeEpisode: false },
  showWidget: false,
  debug: true,
});

async function boot() {
  const sandbox = makeSandbox({ name: `floor-declaration-${seq++}`, config: config() });
  const home = withHome(sandbox.dir);
  const { tools, hooks } = await bootExtension(sandbox);
  const ctx = sessionCtx(path.join(sandbox.dir, 'sessione.jsonl'));
  await hooks.get('session_start')({}, ctx);
  return { sandbox, home, tools, hooks, ctx };
}

const logDi = (sandbox) => fs.readFileSync(path.join(sandbox.dir, '.pi', 'cwl', 'cwl.log'), 'utf8');

const hook = async (hooks, ctx, messages) => {
  const res = await hooks.get('context')({ messages }, ctx);
  return (res && res.messages) || messages;
};

const conversazione = () => {
  const out = [];
  for (let i = 1; i <= 6; i++) {
    out.push({ role: 'user', content: `turn ${i} content ` + 'U'.repeat(200) });
    out.push({ role: 'assistant', content: `reply ${i} content ` + 'A'.repeat(200) });
  }
  return out;
};

test('the floor line partitions the context: the four numbers add up to the total', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    // The CLOSING anchor of a span is the last ELIGIBLE message below the
    // floor, and an `assistant` is REMOVED from the span itself (inside a span
    // only user/system/developer/custom survive). With a non-empty user turn
    // just below the floor the closing would be a `user`, it would survive, and
    // the production case would NOT be seen. A user turn with an EMPTY body is
    // not an address (sha256("") would map 1407 messages onto a single entry)
    // and is skipped by the range computation: the closing falls then on the
    // previous assistant, which is exactly the shape measured in
    // production (`0 of 27 spans located here` with 27 spans applied).
    const conv = conversazione();
    const base = [...conv.slice(0, 10), { role: 'user', content: '' }, ...conv.slice(10)];
    await hook(hooks, ctx, base);
    const comp = await tools.get('cwl_compress_range').execute('t', { summary: 'SINTESI-1 of the first turns' }, undefined, undefined, ctx);
    assert.equal(comp.details.ok, true, `the span was not created: ${JSON.stringify(comp.details)}`);
    // The span is applied; it stays above the trigger; with no episodes and no
    // active levels there is nothing else to do, so it goes through `finish` above budget.
    await hook(hooks, ctx, base);

    const log = logDi(sandbox);
    // The LAST line, not the first: the declaration appears once for every
    // turn passed through `finish` above budget, and in the FIRST turn (the one
    // that creates the address) the spans do not exist yet, so `dentro` would be
    // zero by construction and the test would prove nothing.
    const righe = [...log.matchAll(/CONTEXT (\d+)t still above trigger \d+t: (\d+)t in the protected window \(last \d+ user turns\), (\d+)t inside the spans \(counted from (\d+) of (\d+) spans\), (\d+)t freely compressible, (\d+)t elsewhere/g)];
    assert.ok(righe.length > 0, 'the floor line was not written: the context is above the trigger and `finish` did not declare why');
    const [, totale, protetto, dentro, contati, tenuti, libero, altrove] = righe[righe.length - 1].map(Number);
    assert.equal(
      protetto + dentro + libero + altrove,
      totale,
      `the four numbers must partition the context: ${protetto} + ${dentro} + ${libero} + ${altrove} != ${totale}`,
    );
    // The line DECLARES how many spans it counted. If the count comes from a
    // RE-RESOLUTION of the compressed list, there it finds nothing anymore
    // (the closing anchor is an `assistant`, and the span removed it) and the number
    // would be 0: this is the defect this test must kill.
    assert.ok(tenuti > 0, 'the test did not create any span: it proves nothing');
    assert.equal(
      contati,
      tenuti,
      `the count must come from ALL ${tenuti} kept spans, not from ${contati}: if it comes from a re-resolution of the compressed list, the closing anchors are gone`,
    );
    // NON-VACUITY: with a 1-turn window and one applied span, two of the four
    // terms must be non-zero, otherwise the identity is true by accident.
    assert.ok(protetto > 0, 'the protected window is empty with protectedTurns=1: the test proves nothing');
    assert.ok(dentro > 0, 'the content inside the spans is zero: the test proves nothing');
    // The breakdown by role. The three parts must add up to the total of the
    // line ABOVE (two lines, one number: you cannot make it up), and in the
    // fixture only the `user` turns and the injected summary survive inside the
    // span: NO system/developer/custom. If the classification is wrong, either
    // `utente` goes to zero, or `altro` stops being zero.
    const contenuto = [...log.matchAll(/SPANS content: (\d+)t inside the spans = (\d+)t of summaries \+ (\d+)t of user turns \+ (\d+)t of other roles/g)];
    assert.ok(contenuto.length > 0, 'the breakdown line was not written with an applied span: you cannot know what the spans hold');
    const [, dentro2, riassunti, utente, altro] = contenuto[contenuto.length - 1].map(Number);
    assert.equal(dentro2, dentro, `the breakdown must concern the same total as the CONTEXT line: ${dentro2} != ${dentro}`);
    assert.equal(riassunti + utente + altro, dentro, `the three parts must add up to the total: ${riassunti} + ${utente} + ${altro} != ${dentro}`);
    assert.ok(utente > 0, 'the user turns must survive inside the span: if the count does not see them, the classification is broken');
    assert.ok(riassunti > 0, 'the injected summary is inside the span: if the count does not see it, the classification is broken');
    assert.equal(altro, 0, `there is no system/developer/custom message inside the span in the fixture, but ${altro}t show up: the classification is counting the wrong thing`);
  } finally { home.restore(); sandbox.cleanup(); }
});
