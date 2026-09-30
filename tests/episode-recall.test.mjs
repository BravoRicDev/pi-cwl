/**
 * An evicted episode is RECOVERABLE from the transcript.
 *
 * Why it exists. Episodic compaction was a one-way door:
 * at the `removed` level it threw away every message of the episode and kept only
 * the prose description the agent wrote when closing it. Two holes:
 *
 *   1. an `act` episode (description always empty) left NOTHING, and an
 *      `expl` without a description lost its content forever;
 *   2. `findTranscript` built `_${sessionKey}.jsonl`. The key produced by
 *      `sessionKey()` is a complete PATH, so the suffix became
 *      `_/home/.../sessione.jsonl.jsonl`: it never matched. Even `cwl_recall`
 *      answered "transcript non trovato" while the file was right there.
 *
 * The data was not missing: Pi's transcript is append-only and keeps the
 * originals, and the episode ALREADY has the two anchors (startToolCallId/endToolCallId)
 * persisted in the state. What was missing was the POINTER.
 *
 * NOTE on the two strings that stay Italian below (`/troncato/`, "transcript non
 * trovato" in the comment above): they are text the PRODUCT emits, and the product
 * speaks Italian when the locale is Italian (LANG=it_IT here). Translating them
 * would make the tests assert a language, not a behaviour.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { makeSandbox, bootExtension, withHome, sessionCtx } from './_helpers.mjs';

/** Builds a sandbox with a moved HOME and a writable transcript. */
function setup({ name, config = null }) {
  const sandbox = makeSandbox({ name, config });
  const home = withHome(sandbox.dir);
  const realFile = path.join(
    sandbox.dir, '.pi', 'agent', 'sessions', '--fake--',
    '2026-01-01T00-00-00-000Z_fake-session-id.jsonl',
  );
  fs.mkdirSync(path.dirname(realFile), { recursive: true });
  return { sandbox, home, realFile };
}

/** One transcript line in the form Pi writes. */
const record = (id, role, content, extra = {}) =>
  JSON.stringify({ type: 'message', id, parentId: null, timestamp: '2026-01-01T00:00:00.000Z', message: { role, content, ...extra } });

const text = (s) => [{ type: 'text', text: s }];

/** Opens and closes an episode with anchors chosen by us. */
async function withEpisode(tools, hooks, ctx, { name = 'ep1', type = 'expl', start = 'call-start-1', end = 'call-end-1', description = '' } = {}) {
  await hooks.get('session_start')({}, ctx);
  const d = tools.get('delimiter');
  const opened = await d.execute(start, { action: 'start', name, type }, undefined, undefined, ctx);
  assert.equal(opened.details.ok, true, `opening failed: ${JSON.stringify(opened.details)}`);
  const closed = await d.execute(end, { action: 'end', name, description }, undefined, undefined, ctx);
  assert.equal(closed.details.ok, true, `closing failed: ${JSON.stringify(closed.details)}`);
}

test('cwl_recall_episode returns the original text of an evicted episode', async () => {
  const { sandbox, home, realFile } = setup({ name: 'ep-recall' });
  try {
    fs.writeFileSync(realFile, [
      record('r0', 'user', text('opening')),
      record('r1', 'toolResult', text('episode opened'), { toolCallId: 'call-start-1', toolName: 'delimiter' }),
      record('r2', 'assistant', text('the secret file is /tmp/alfa.txt and the function parseZeta()')),
      record('r3', 'toolResult', text('grep content: parseZeta in alfa.txt'), { toolCallId: 'call-tool-2', toolName: 'bash' }),
      record('r4', 'toolResult', text('episode closed'), { toolCallId: 'call-end-1', toolName: 'delimiter' }),
      record('r5', 'user', text('post')),
    ].join('\n') + '\n');

    const ctx = sessionCtx(realFile);
    const { tools, hooks } = await bootExtension(sandbox);
    await withEpisode(tools, hooks, ctx, { description: 'learned the name parseZeta' });

    const out = await tools.get('cwl_recall_episode').execute('t', { name: 'ep1', full: true }, undefined, undefined, ctx);

    assert.equal(out.details.ok, true, `recovery failed: ${JSON.stringify(out.details)}`);
    const body = out.content.map((c) => c.text).join('\n');
    // The content of the episode, not only the description.
    assert.match(body, /parseZeta/, 'the text of the assistant message is missing');
    assert.match(body, /grep content: parseZeta in alfa\.txt/, 'the toolResult is missing, and it is 45%% of the transcript');
    // The messages OUTSIDE the episode must not enter the recovery.
    assert.doesNotMatch(body, /opening/, 'a message before the opening anchor was included');
    assert.doesNotMatch(body, /post/, 'a message after the closing anchor was included');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('cwl_recall_episode truncates by default and does not bloat the context it just freed', async () => {
  const { sandbox, home, realFile } = setup({ name: 'ep-trunc' });
  try {
    const long = 'X'.repeat(20000);
    fs.writeFileSync(realFile, [
      record('r1', 'toolResult', text('opened'), { toolCallId: 'call-start-1', toolName: 'delimiter' }),
      record('r2', 'assistant', text(long)),
      record('r4', 'toolResult', text('closed'), { toolCallId: 'call-end-1', toolName: 'delimiter' }),
    ].join('\n') + '\n');
    const ctx = sessionCtx(realFile);
    const { tools, hooks } = await bootExtension(sandbox);
    await withEpisode(tools, hooks, ctx);

    const short = await tools.get('cwl_recall_episode').execute('t', { name: 'ep1' }, undefined, undefined, ctx);
    assert.equal(short.details.full, false);
    assert.ok(short.details.chars > 4000, 'the recovered content should be large');
    const shortBody = short.content.map((c) => c.text).join('\n');
    // Product text, Italian on an Italian locale: the truncation notice is the one
    // `episodeRecallTruncatedHint` writes (index.ts, I18N.it).
    assert.match(shortBody, /troncato/, 'the truncation notice is missing');
    assert.ok(shortBody.length < 12000, `answer too long: ${shortBody.length}`);

    const full = await tools.get('cwl_recall_episode').execute('t', { name: 'ep1', full: true }, undefined, undefined, ctx);
    assert.equal(full.details.full, true);
    assert.ok(full.content.map((c) => c.text).join('\n').length > 20000, 'full=true must return everything');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('cwl_recall_episode refuses a non-existent or still-open episode', async () => {
  const { sandbox, home, realFile } = setup({ name: 'ep-guard' });
  try {
    fs.writeFileSync(realFile, record('r1', 'user', text('nothing')) + '\n');
    const ctx = sessionCtx(realFile);
    const { tools, hooks } = await bootExtension(sandbox);
    await hooks.get('session_start')({}, ctx);
    await tools.get('delimiter').execute('call-start-1', { action: 'start', name: 'opened', type: 'expl' }, undefined, undefined, ctx);

    const unknown = await tools.get('cwl_recall_episode').execute('t', { name: 'does-not-exist' }, undefined, undefined, ctx);
    assert.equal(unknown.details.ok, false);
    assert.equal(unknown.details.error, 'episode-not-found');

    const open = await tools.get('cwl_recall_episode').execute('t', { name: 'opened' }, undefined, undefined, ctx);
    assert.equal(open.details.ok, false);
    assert.equal(open.details.error, 'episode-still-open');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('cwl_recall resolves the key sessionKey really produces (full path)', async () => {
  const { sandbox, home, realFile } = setup({ name: 'ep-find' });
  try {
    fs.writeFileSync(realFile, record('a1', 'user', text('unique keyword zebra42')) + '\n');
    const ctx = sessionCtx(realFile);
    const { tools, hooks } = await bootExtension(sandbox);
    await hooks.get('session_start')({}, ctx);

    const out = await tools.get('cwl_recall').execute('t', { query: 'zebra42', limit: 5 }, undefined, undefined, ctx);
    assert.equal(out.details.ok, true, `recall failed: ${JSON.stringify(out.details)}`);
    assert.equal(out.details.hits, 1);
  } finally { home.restore(); sandbox.cleanup(); }
});

test('the eviction emits the pointer, also for an act episode', async () => {
  // Only `removeEpisode` active: this way the eviction goes STRAIGHT to the removed level,
  // without having to escalate reasoning -> bulk -> intermediate over several turns.
  // protectedTurns=0: here the safety window must be OFF, otherwise the two
  // sole user turns of the scene are both protected and nothing is evicted.
  const config = {
    tokenBudget: 10,
    thresholdRatio: 0.5,
    protectedTurns: 0,
    levels: { stripReasoning: false, stripBulkOutput: false, stripIntermediate: false, removeEpisode: true },
    showWidget: false,
    debug: false,
  };
  for (const type of ['expl', 'act']) {
    const { sandbox, home, realFile } = setup({ name: `ep-marker-${type}`, config });
    try {
      fs.writeFileSync(realFile, record('r1', 'user', text('x')) + '\n');
      const ctx = sessionCtx(realFile);
      const { tools, hooks } = await bootExtension(sandbox);
      await withEpisode(tools, hooks, ctx, { type, name: `ep-${type}`, description: type === 'expl' ? 'useful note' : '' });

      const messages = [
        { role: 'user', content: 'opening' },
        { role: 'toolResult', toolCallId: 'call-start-1', toolName: 'delimiter', content: text('opened') },
        { role: 'assistant', content: 'plenty of contents to evict '.repeat(40) },
        { role: 'toolResult', toolCallId: 'call-end-1', toolName: 'delimiter', content: text('closed') },
        { role: 'user', content: 'post' },
      ];
      const res = await hooks.get('context')({ messages }, ctx);
      assert.ok(res && Array.isArray(res.messages), 'the context hook evicted nothing');

      const marker = res.messages.find((m) => m.customType === 'cwl-evicted');
      assert.ok(marker, `no marker emitted for an ${type} episode`);
      assert.match(marker.content, /cwl_recall_episode\("ep-(expl|act)"\)/, 'the marker does not contain the pointer');
      // The messages outside the episode stay.
      assert.ok(res.messages.some((m) => m.role === 'user' && m.content === 'post'), 'the user turn was touched');
    } finally { home.restore(); sandbox.cleanup(); }
  }
});
