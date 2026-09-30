/**
 * THE LABEL IS BORN WITH THE LEAF, NOT IN A SECOND PASS.
 *
 * Operator request: *"Can we make it so that when the 'leaf' is created its own
 * 'description/label' is created too? so that when it is fired into the node it is already ready and
 * we don't have to do batches of 5 leaves at a time."*
 *
 * Whoever writes the compression summary has just read the messages: the micro (~200 words,
 * the label that the index shows in place of the body) is almost free at that moment.
 * Asking for it later means a second pass over a body that meanwhile lives only in the
 * state, and a batch of five leaves to label all at once.
 *
 * The test demands two things, and they are two different invariants:
 *  1. with the label at birth the leaves no longer wait for anything (`0 leaf/leaves
 *     waiting`): the node forms by itself and nobody has to ask for the micro;
 *  2. the micro does NOT enter the body. `cwl_open` must keep returning the whole
 *     and clean summary: a label written INSIDE the body would make the promise
 *     "nothing is lost" false, with one line.
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
  const sandbox = makeSandbox({ name: `label-${seq++}`, config: config() });
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

const logOf = (sandbox) => fs.readFileSync(path.join(sandbox.dir, '.pi', 'cwl', 'cwl.log'), 'utf8');

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

/** The leaf is born WITH its body and its label, in the same call. */
const LABEL = (i) => `LABEL-${i}: leaf ${i} tells about turn ${i} and what was decided there`;

test('the leaf is born with its label, and the body stays whole', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    for (let i = 1; i <= 7; i++) {
      await hook(hooks, ctx, conversation(1, i + 3));
      const res = await tools.get('cwl_compress_range').execute(
        't',
        { summary: `BODY-${i} ` + 'x'.repeat(300), micro: LABEL(i) },
        undefined,
        undefined,
        ctx,
      );
      assert.equal(res.details.ok, true, `round ${i}: the leaf was not born: ${JSON.stringify(res.details)}`);
    }

    const leaves = stateOf(sandbox).spans;
    assert.equal(leaves.length, 7, 'seven leaves are needed: with `looseLeaves` = 5 the last five stay loose');
    assert.ok(
      leaves.slice(0, 2).every((f) => f.micro && f.micro.includes('LABEL')),
      'the labels were not saved on the leaf: the parameter arrived and was not written',
    );

    // The turn after, the older leaves leave the `looseLeaves` window.
    const before = logOf(sandbox).length;
    const out = await hook(hooks, ctx, conversation(1, 10));
    const log = logOf(sandbox).slice(before);

    // 1. Nobody waits for anything any more: the labels were already there.
    const row = /NODES: (\d+) node\(s\) \[([^\]]*)\], (\d+) leaf\/leaves waiting/.exec(log);
    assert.ok(row, `the turn does not declare the nodes: ${log.trim().split('\n').slice(-3).join(' | ')}`);
    assert.equal(
      Number(row[3]),
      0,
      `the leaves still wait for a micro (${row[3]} waiting): the labels were not used at birth, ` +
        'so the node does not form and the extension has to ask for them later — the batch of five that was meant to be avoided.',
    );
    assert.ok(Number(row[1]) >= 1, `no node formed (${row[1]}): the leaves with a label did not enter a node`);

    // 2. The body stays whole: the label did not end up inside it.
    const opened = await tools.get('cwl_open').execute('t', { id: leaves[0].id }, undefined, undefined, ctx);
    const text = opened.content.map((c) => c.text).join('\n');
    assert.ok(
      text.includes('BODY-1 '),
      `cwl_open no longer returns the leaf body: ${text.slice(0, 200)}`,
    );
    assert.ok(
      !text.includes('LABEL-1'),
      'the micro was written INSIDE the body: `cwl_open` no longer returns the original, and the promise ' +
        '"nothing is lost" becomes false. The micro is a separate field, never a body edit.',
    );

    // 3. And nobody asks for the micro, because there is nothing to ask for.
    const requests = out.filter((m) => m.customType === 'cwl-demand').map((m) => String(m.content)).join('\n');
    assert.ok(
      !requests.includes('cwl_micro'),
      `the extension still asks for the micros that have already arrived: ${requests.slice(0, 200)}`,
    );
  } finally {
    home.restore();
  }
});
