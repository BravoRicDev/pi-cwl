/**
 * THE TRACE OF AN EVENT IS STATIC.
 *
 * Why it exists. An injected block lands in the MIDDLE of the list, where the
 * compressed content used to be. The provider cache is a PREFIX cache: from the
 * first byte that differs, every following token is charged again. So a block that
 * carries a value regenerated per turn — `timestamp: Date.now()` — is a block that
 * differs from itself on every render, and it silently invalidates everything after
 * it.
 *
 * MEASURED, and it is the reason this is a determinism fix and not a cache fix: the
 * provider payload is built from `role` + `content` alone, in BOTH builders —
 * `convertMessages` (Anthropic) and `toChatMessages` (openai-completions, which is
 * what `scrocco-llm` uses). The timestamp never reaches the model. The operator's
 * rule is the reason the change is right anyway: an event may leave a trace, but the
 * trace must say WHEN THE EVENT HAPPENED — when the leaf was born, when the node was
 * born, when the episode opened — not when the turn was rendered.
 *
 * Two properties, and neither is visible from a green run of the extension:
 *  1. the block is BYTE-IDENTICAL across two turns in which nothing happens;
 *  2. the trace is the leaf's own birth, so it survives a restart and a re-render.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as fs from 'node:fs';
import * as path from 'node:path';
import { makeSandbox, bootExtension, withHome, sessionCtx } from './_helpers.mjs';

let seq = 0;

const config = (extra = {}) => ({
  tokenBudget: 300,
  thresholdRatio: 0.5,
  protectedTurns: 0,
  levels: { stripReasoning: false, stripBulkOutput: false, stripIntermediate: false, removeEpisode: false },
  showWidget: false,
  debug: true,
  ...extra,
});

async function boot() {
  const sandbox = makeSandbox({ name: `trace-${seq++}`, config: config() });
  const home = withHome(sandbox.dir);
  const { tools, hooks } = await bootExtension(sandbox);
  const ctx = sessionCtx(path.join(sandbox.dir, 'sessione.jsonl'));
  await hooks.get('session_start')({}, ctx);
  return { sandbox, home, tools, hooks, ctx };
}

const user = (i) => ({ role: 'user', content: `prompt ${i} ${'P'.repeat(200)}` });
const assistant = (i) => ({ role: 'assistant', content: `answer ${i} ${'A'.repeat(200)}` });

/** The persisted state: the leaf's own birth lives here, not in the render. */
const leafInState = (sandbox) => {
  const dir = path.join(sandbox.dir, '.pi', 'cwl', 'state');
  const file = fs.readdirSync(dir).find((f) => f.endsWith('.json'));
  const st = JSON.parse(fs.readFileSync(path.join(dir, file), 'utf8'));
  return st.spans[st.spans.length - 1];
};

/** Renders one turn and hands back the injected block, whole. */
async function renderBlock(hooks, ctx, messages) {
  const res = await hooks.get('context')({ messages }, ctx);
  const out = res?.messages || messages;
  return out.find((m) => m.customType === 'cwl-compressed');
}

async function bornLeaf(tools, hooks, ctx, messages) {
  await hooks.get('context')({ messages }, ctx);
  const res = await tools.get('cwl_compress_range').execute(
    't', { summary: `BODY ${'x'.repeat(300)}`, micro: 'MICRO: the points that matter most.' }, undefined, undefined, ctx,
  );
  assert.equal(res.details.ok, true, `the leaf was not born: ${JSON.stringify(res.details)}`);
}

test('the injected block is byte-identical across two turns in which nothing happens', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    const messages = [];
    for (let i = 1; i <= 6; i++) { messages.push(user(i)); messages.push(assistant(i)); }
    await bornLeaf(tools, hooks, ctx, messages);

    const first = await renderBlock(hooks, ctx, messages);
    assert.ok(first, 'the leaf block was never injected');
    // Enough for `Date.now()` to have moved: 10ms of real time between the renders.
    await new Promise((r) => setTimeout(r, 10));
    const second = await renderBlock(hooks, ctx, messages);
    assert.ok(second, 'the leaf block disappeared between two renders');

    // The WHOLE block, not just the timestamp: the property the cache needs is that
    // the injected text does not move when no event happened.
    assert.deepEqual(second, first,
      'the injected block changed between two turns with no event: it invalidates every token after it');
  } finally { home.restore(); sandbox.cleanup(); }
});

test('the trace is the leaf birth, not the clock that rendered it', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    const messages = [];
    for (let i = 1; i <= 6; i++) { messages.push(user(i)); messages.push(assistant(i)); }
    await bornLeaf(tools, hooks, ctx, messages);

    const block = await renderBlock(hooks, ctx, messages);
    assert.ok(block, 'the leaf block was never injected');
    const leaf = leafInState(sandbox);
    assert.equal(typeof leaf.at, 'number', 'the leaf carries no birth time');
    assert.equal(block.timestamp, leaf.at,
      `the trace is ${block.timestamp}, but the leaf was born at ${leaf.at}: the block carries the render clock`);
  } finally { home.restore(); sandbox.cleanup(); }
});
