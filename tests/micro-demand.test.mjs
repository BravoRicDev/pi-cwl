/**
 * THE INDEX'S FIRST STEP MUST ASK, NOT JUST RECORD.
 *
 * MEASURED LIVE, right after the reload that let the index run:
 *
 *     NODES: 0 node(s) [none], 40 leaf/leaves waiting for a micro — their body is still in
 *     the context
 *
 * on EVERY turn, with 45 spans and 90.415t of summaries still in the context. That is: the code
 * ran, but the mechanism could NOT start. The chain is: leaves without a micro do not
 * enter a node -> zero nodes -> `plan.due` is zero -> the merge request
 * (`indexDue`) NEVER appears. And nobody talked about the micro: the count ended up in the LOG, not
 * in the context. The extension knew it, the operator could read it, and the only one who can
 * write a micro — the agent — received no question.
 *
 * It is exactly the defect that `bd0734c` closed for `cwl_old` (the big summary), left
 * open on the step BEFORE. The request must say two things to be actionable: that
 * `cwl_micro` must be called, and ON WHICH leaves — without the ids the agent has to guess which one, or
 * go read the state file.
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

const conversation = (from, to) => {
  const out = [];
  for (let i = from; i <= to; i++) {
    out.push({ role: 'user', content: `turn ${i} content ` + 'U'.repeat(200), timestamp: 1000 + i * 2 });
    out.push({ role: 'assistant', content: `answer ${i} content ` + 'A'.repeat(200), timestamp: 1001 + i * 2 });
  }
  return out;
};

test('the micro request reaches the agent, and names the leaf', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    // SEVEN leaves, not one. With `looseLeaves` = 5 the last five stay LOOSE and do not
    // owe a micro yet: the request is born only when a leaf leaves that
    // window, and that is exactly the case that matters (its body must leave the
    // context). My first fixture made ONE and the test was red for that reason, not for
    // the code: the usual lesson, the defect was in the test.
    for (let i = 1; i <= 7; i++) {
      await hook(hooks, ctx, conversation(1, i + 3));
      const res = await tools.get('cwl_compress_range').execute(
        't', { summary: `BODY-${i} ` + 'x'.repeat(300) }, undefined, undefined, ctx,
      );
      assert.equal(res.details.ok, true, `round ${i}: the leaf was not born: ${JSON.stringify(res.details)}`);
    }
    assert.equal(
      stateOf(sandbox).spans.length,
      7,
      'seven leaves are needed: with five or fewer they are all loose and none is waiting for a micro',
    );

    // The turn after: the older leaves are no longer loose, they have no micro, and
    // the agent must KNOW it.
    const out = await hook(hooks, ctx, conversation(1, 10));
    const requests = out.filter((m) => m.customType === 'cwl-demand');
    assert.ok(
      requests.length > 0,
      'no request in the context: the leaves without a micro end up only in the LOG, so the agent does not know it must ' +
        `write them and the mechanism never starts (0 nodes -> plan.due = 0 -> the merge request never appears). Messages: ${out
          .map((m) => m.customType || m.role)
          .join(', ')}`,
    );

    const text = requests.map((m) => String(m.content)).join('\n');
    assert.ok(
      text.includes('cwl_micro'),
      `the request does not name the tool to use, so it is not actionable: ${text.slice(0, 200)}`,
    );

    const id = stateOf(sandbox).spans[0].id;
    assert.ok(
      text.includes(id),
      `the request does not name the leaf ${id}: without the id the agent has to guess which one, or go read the ` +
        `state file. Text: ${text.slice(0, 300)}`,
    );
  } finally {
    home.restore();
  }
});
