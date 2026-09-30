/**
 * THE TRIGGER: the request to write the big summary must reach the AGENT.
 *
 * This test exists for a defect that was not a defect of the code but of the channel:
 * the merge of the old node worked and was covered by tests, and in a real session
 * it could NOT fire. `OLD NODE due: ...` was a LOG line: the extension knew it,
 * the operator could read it, and the only one who can write the summary — the agent — saw
 * nothing. A mechanism nobody can trigger is not a mechanism.
 *
 * The request goes where the compress request already goes: into the CONTEXT. And like
 * that one, the extension does not invent the summary: it asks, and waits.
 *
 * THE THREE THINGS THE TEST DEMANDS:
 *  1. when the merge is NOT due the request is NOT there — a perpetual warning is noise
 *     that teaches the agent to ignore warnings;
 *  2. when it is due, the request is there and it is in the context the provider receives;
 *  3. after the merge it disappears, because there is nothing left to ask.
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
  assert.ok(file, 'the session state was not created');
  return JSON.parse(fs.readFileSync(path.join(dir, file), 'utf8'));
};

const conversazione = (da, a) => {
  const out = [];
  for (let i = da; i <= a; i++) {
    out.push({ role: 'user', content: `turn ${i} content ` + 'U'.repeat(200) });
    out.push({ role: 'assistant', content: `answer ${i} content ` + 'A'.repeat(200) });
  }
  return out;
};

/** The request, if any, as the provider would see it. */
const richiesta = (msgs) =>
  msgs.filter((m) => m && m.customType === 'cwl-demand').map((m) => String(m.content)).join('\n');

test('the merge request reaches the context, and only when it is due', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    // 1. A single leaf: there is no node, so nothing to ask for.
    await hook(hooks, ctx, conversazione(1, 4));
    const uno = await tools.get('cwl_compress_range').execute('t', { summary: 'BODY-1' }, undefined, undefined, ctx);
    assert.equal(uno.details.ok, true, `the first leaf was not born: ${JSON.stringify(uno.details)}`);
    const conUna = await hook(hooks, ctx, conversazione(1, 6));
    assert.equal(
      richiesta(conUna).length,
      0,
      `the request is already there with ONE leaf and no node: a perpetual warning is noise, and noise teaches to ignore warnings. Found: ${richiesta(conUna).slice(0, 160)}`,
    );

    // 2. Five leaves with a micro: two young nodes form, and the merge becomes due.
    for (let i = 2; i <= 5; i++) {
      await hook(hooks, ctx, conversazione(1, i + 3));
      const res = await tools.get('cwl_compress_range').execute(
        't', { summary: `BODY-${i} ` + 'x'.repeat(300) }, undefined, undefined, ctx,
      );
      assert.equal(res.details.ok, true, `round ${i}: the leaf was not born: ${JSON.stringify(res.details)}`);
    }
    const foglie = statoDi(sandbox).spans;
    for (let i = 0; i < foglie.length; i++) {
      await tools.get('cwl_micro').execute('t', { id: foglie[i].id, text: `MICRO-${i + 1}` }, undefined, undefined, ctx);
    }

    const dovuto = await hook(hooks, ctx, conversazione(1, 12));
    const testo = richiesta(dovuto);
    assert.ok(
      testo.length > 0,
      'the merge is due and the request is NOT in the context: the extension knows it, the operator reads it in the log, and the agent — the only one who can write the summary — will never know',
    );
    assert.match(testo, /cwl_old/, `the request does not say what to do (cwl_old): ${testo.slice(0, 200)}`);
    assert.match(testo, /2/, `the request does not say HOW MANY nodes must be merged: ${testo.slice(0, 200)}`);

    // 3. Merged: the merge is no longer due, and the request disappears.
    const acc = await tools.get('cwl_old').execute('t', { text: 'BIG-SUMMARY-1' }, undefined, undefined, ctx);
    assert.equal(acc.details.ok, true, `the merge failed: ${JSON.stringify(acc.details)}`);
    const dopo = await hook(hooks, ctx, conversazione(1, 12));
    assert.equal(
      richiesta(dopo).length,
      0,
      `the request stayed after the merge: ${richiesta(dopo).slice(0, 200)} — the agent would redo work already done`,
    );
  } finally {
    home.restore();
  }
});
