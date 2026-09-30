/**
 * #7: the original comes back from the transcript when the state has pruned the leaf.
 *
 * THE PROBLEM, stated precisely. A summary lives ONLY in the state. When the
 * native compaction takes away the anchors of a span, `locateSpans` declares it
 * dead and the hook prunes it: from that moment its summary no longer exists. The
 * MESSAGES it had replaced, however, are still in the append-only transcript.
 *
 * WHY THE ID ALONE WAS NOT ENOUGH. A leaf's id is `sp-` + a hash of the two
 * anchors: it is a one-way function, so from an id you cannot go back to the anchors.
 * That is why `CompressedSpan` also carries `startSid`/`endSid` — the
 * stable ids of the messages, `stableIdOf` — written when the anchors were still
 * visible, and why pruning keeps them in a grave.
 *
 * AND WHY THE HASHES WERE NOT ENOUGH: `addressOf` mixes the id with the TEXT, and this
 * extension strips the reasoning from the messages it compresses —
 * `contentToText` includes `part.thinking`. A hash computed from the transcript may
 * therefore not match the one computed from the context, and the lookup would fail
 * without saying why. The timestamp cannot be derived either.
 *
 * AND WHEN IT FINDS NOTHING, IT SAYS SO: there is `openOriginalLost`, which names
 * `cwl_recall`. A silent limit is a promise broken in silence.
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
  const sandbox = makeSandbox({ name: `originale-${seq++}`, config: config() });
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

const testo = (res) => res.content.map((c) => c.text).join('\n');

/** Messages with a timestamp: that is what `stableIdOf` uses as identity. */
const conversazione = () => {
  const out = [];
  for (let i = 1; i <= 6; i++) {
    out.push({ role: 'user', content: `turn ${i} content ` + 'U'.repeat(200), timestamp: 1000 + i * 2 });
    out.push({ role: 'assistant', content: `answer ${i} content ` + 'A'.repeat(200), timestamp: 1001 + i * 2 });
  }
  return out;
};

test('a pruned leaf is reopened from the transcript, and it says so', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    const conv = conversazione();
    await hook(hooks, ctx, conv);
    const comp = await tools.get('cwl_compress_range').execute('t', { summary: 'SUMMARY-A of the first turns' }, undefined, undefined, ctx);
    assert.equal(comp.details.ok, true, `the leaf was not born: ${JSON.stringify(comp.details)}`);

    // One turn to let the span RESOLVE: that is where the stable ids are written.
    await hook(hooks, ctx, conv);
    const sp = statoDi(sandbox).spans[0];
    assert.ok(
      sp && sp.startSid && sp.endSid,
      `the leaf has no stable ids: without them, after pruning, the id can no longer find anything. Found: ${JSON.stringify(sp)}`,
    );

    // The transcript, as Pi writes it: one record per message, with the timestamp.
    const righe = conv.map((m) =>
      JSON.stringify({ message: { role: m.role, timestamp: m.timestamp, content: [{ type: 'text', text: String(m.content) }] } }),
    );
    fs.writeFileSync(path.join(sandbox.dir, 'sessione.jsonl'), righe.join('\n') + '\n');

    // A list that NO LONGER contains the anchors: the native compaction has
    // replaced that prefix, so the span is dead and gets pruned.
    const altrove = [{ role: 'user', content: 'the story moved on, this prefix is no longer here', timestamp: 999999 }];
    await hook(hooks, ctx, altrove);
    const st = statoDi(sandbox);
    assert.equal(st.spans.length, 0, `the leaf had to be pruned, ${st.spans.length} are left`);
    assert.ok(
      st.graves && st.graves.length === 1,
      `the pruning did not leave a grave with the anchors: ${JSON.stringify(st.graves)} — from here on the id finds nothing any more`,
    );
    assert.equal(st.graves[0].id, sp.id, 'the grave does not match the pruned leaf');

    // And now: the id, which had been pruned, returns the ORIGINAL.
    const res = await tools.get('cwl_open').execute('t', { id: sp.id }, undefined, undefined, ctx);
    assert.equal(res.details.ok, true, `the pruned leaf does not reopen: ${JSON.stringify(res.details)}`);
    assert.equal(res.details.kind, 'original', `the tool does not declare it is delivering the original: ${JSON.stringify(res.details)}`);
    assert.ok(
      testo(res).includes('turn 1 content'),
      `the original text did not come back: ${testo(res).slice(0, 200)}`,
    );
    assert.ok(
      !testo(res).includes('SUMMARY-A'),
      'the tool delivered the SUMMARY: but the summary was lost with the state, so either it is inventing it or it did not understand what was asked of it',
    );
  } finally {
    home.restore();
  }
});
