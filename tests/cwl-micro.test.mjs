/**
 * `cwl_micro`: the body leaves the context, the micro takes its place, and the
 * body remains WHOLE one call away.
 *
 * It is the lever the whole project was born from. MEASURED in a real session
 * (commit de9672a):
 *
 *   SPANS content: 52535t inside the spans = 52027t of summaries + 508t of user
 *   turns + 0t of other roles
 *
 * 52,027 tokens out of 114,327 were the summaries the extension had written ITSELF
 * (30 leaves, 1,707t on average), and no path could touch them: `keptInsideSpan` keeps
 * `custom` and the eviction applier protects `custom`. The extension knew how to
 * WRITE a summary and did not know how to ABSORB an old one: every compression
 * added a story, no lever ever withdrew one.
 *
 * THE TWO THINGS THIS TEST DEFENDS:
 *  1. absorbing makes the context SMALLER: the body leaves, the micro enters. If the
 *     body stayed, the absorption would do nothing;
 *  2. the micro does NOT touch the body. If it wrote itself inside it, `cwl_open` would come
 *     back short and the promise "nothing is lost" would be false in one line: it is
 *     the only way of breaking this design that cannot be seen from the context.
 *
 * There is also the way back: an EMPTY text removes the micro and puts the
 * whole body back in the context. An absorption nobody can undo is a
 * one-way door, and this one has a way back.
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

const stateOf = (sandbox) => {
  const dir = path.join(sandbox.dir, '.pi', 'cwl', 'state');
  const file = fs.readdirSync(dir).find((f) => f.endsWith('.json'));
  assert.ok(file, 'the session state was not created');
  return JSON.parse(fs.readFileSync(path.join(dir, file), 'utf8'));
};

const conversation = () => {
  const out = [];
  for (let i = 1; i <= 6; i++) {
    out.push({ role: 'user', content: `prompt ${i} content. ` + 'U'.repeat(200) });
    out.push({ role: 'assistant', content: `answers ${i} content is ` + 'A'.repeat(200) });
  }
  return out;
};

/** What the provider would receive: the blocks injected in place of the compressed ones. */
const inContext = (msgs) =>
  msgs.filter((m) => m && m.customType === 'cwl-compressed').map((m) => String(m.content)).join('\n');

const text = (res) => res.content.map((c) => c.text).join('\n');

test('the micro replaces the body in the context without destroying it', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    await hook(hooks, ctx, conversation());

    const body = 'COMPLETED-SUMMARIES ' + 'x'.repeat(8000) + ' BODY-ENDS-HERE';
    const micro = 'MICRO-SHORT: the points that matter most.';
    const comp = await tools.get('cwl_compress_range').execute('t', { summary: body }, undefined, undefined, ctx);
    assert.equal(comp.details.ok, true, `the span was not created: ${JSON.stringify(comp.details)}`);
    const sp = stateOf(sandbox).spans[0];
    assert.ok(sp && sp.id, `the span has no id: ${JSON.stringify(sp)}`);

    // Before the absorption the body is in the context, in full.
    const before = await hook(hooks, ctx, conversation());
    assert.ok(
      inContext(before).includes('BODY-ENDS-HERE'),
      'the premise of the test does not hold: the body was not in the context even before the absorption',
    );

    const ass = await tools.get('cwl_micro').execute('t', { id: sp.id, text: micro }, undefined, undefined, ctx);
    assert.equal(ass.details.ok, true, `the absorption failed: ${JSON.stringify(ass.details)}`);

    const after = await hook(hooks, ctx, conversation());
    const context = inContext(after);

    // 1. The body has GONE out of the context and the micro has entered: that is the whole point.
    assert.ok(context.includes(micro), `the micro is not in the context: ${context.slice(0, 160)}`);
    assert.ok(
      !context.includes('BODY-ENDS-HERE'),
      'the body is still in the context: the absorption freed nothing, ' +
        'so the extension keeps paying for the same story',
    );

    // 2. The body was NOT destroyed: `cwl_open` gives it back whole.
    const reopened = await tools.get('cwl_open').execute('t', { id: sp.id }, undefined, undefined, ctx);
    assert.equal(reopened.details.ok, true, `the leaf no longer reopens: ${JSON.stringify(reopened.details)}`);
    assert.ok(
      text(reopened).includes(body),
      `the micro ate the body: cwl_open answers with ${text(reopened).length} characters, the body has ${body.length}. ` +
        'The promise "nothing is lost" would be false, and from the context one would not see it.',
    );

    // 3. A micro LONGER than the body is declared for what it is.
    const longer = await tools.get('cwl_micro').execute(
      't', { id: sp.id, text: 'y'.repeat(body.length + 100) }, undefined, undefined, ctx,
    );
    assert.equal(
      longer.details.shorter,
      false,
      'a micro longer than the body it replaces was accepted as a saving: the context does not shrink, and the agent would not know',
    );

    // 4. The way back: an empty text puts the whole body back in the context.
    const wayBack = await tools.get('cwl_micro').execute('t', { id: sp.id, text: '' }, undefined, undefined, ctx);
    assert.equal(wayBack.details.ok, true, `removing the micro failed: ${JSON.stringify(wayBack.details)}`);
    const returned = inContext(await hook(hooks, ctx, conversation()));
    assert.ok(
      returned.includes('BODY-ENDS-HERE'),
      'removing the micro did not put the body back in the context: the absorption was a one-way door',
    );
  } finally {
    home.restore();
  }
});
