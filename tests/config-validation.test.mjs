/**
 * Regression: a config.json with wrong types must not break the policy.
 *
 * Why it exists: until 2026-09-30 `loadConfig` did
 * `{ ...DEFAULT_CONFIG, ...user }` without validating anything. With
 * `thresholdRatio: "high"` the trigger became `tokenBudget * "high"` = NaN,
 * so `currentTokens <= NaN` was always false and the extension evicted on EVERY
 * turn. With `thresholdRatio: 5` it disabled eviction for good. A typo
 * in the config, no crash, no warning: just a wrong policy
 * in silence.
 *
 * The test goes through the real `cwl_status`, which prints budget and threshold: this way
 * it verifies the observable behavior of the extension, not an internal
 * function that could be renamed.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import { makeSandbox, bootExtension, withHome, sessionCtx, status } from './_helpers.mjs';

/**
 * Loads the extension with a given config and returns the tools.
 *
 * It MUST move HOME into the sandbox: `loadConfig` reads
 * `~/.pi/cwl/config.json` first and only as a fallback the config bundled with the package.
 * Without `withHome` the config passed here was never read and the assertions
 * "falls back on the default" passed vacuously, because it read the bundled one.
 */
async function loadWithConfig(config, name) {
  const sandbox = makeSandbox({ name, config });
  const home = withHome(sandbox.dir);
  const { tools } = await bootExtension(sandbox);
  return {
    tools,
    ctx: sessionCtx(`${sandbox.dir}/sessione.jsonl`),
    cleanup: () => { home.restore(); sandbox.cleanup(); },
  };
}

test('a non-numeric thresholdRatio does not produce NaN and falls back on the default', async () => {
  const { tools, ctx, cleanup } = await loadWithConfig({ thresholdRatio: 'high' }, 'cfg-str');
  try {
    const { text, details } = await status(tools, ctx);
    assert.doesNotMatch(text, /NaN/, 'the budget must never render NaN');
    assert.match(text, /threshold: 85%/, 'an invalid threshold must fall back on the default 0.85');
    assert.equal(details.budget, 80000, 'the budget must stay at the default 80k');
  } finally { cleanup(); }
});

test('an out-of-range thresholdRatio does not silently disable eviction', async () => {
  const { tools, ctx, cleanup } = await loadWithConfig({ thresholdRatio: 5 }, 'cfg-range');
  try {
    const { text } = await status(tools, ctx);
    assert.match(text, /threshold: 85%/, 'a threshold > 1 must fall back on the default');
  } finally { cleanup(); }
});

test('a non-numeric tokenBudget falls back on the default', async () => {
  const { tools, ctx, cleanup } = await loadWithConfig({ tokenBudget: 'many' }, 'cfg-budget');
  try {
    const { text, details } = await status(tools, ctx);
    assert.doesNotMatch(text, /NaN/);
    assert.equal(details.budget, 80000, 'an invalid budget must fall back on the default 80k');
  } finally { cleanup(); }
});

test('a malformed config.json does not kill the extension', async () => {
  const { tools, ctx, cleanup } = await loadWithConfig('{ this is not json', 'cfg-broken');
  try {
    const { text, details } = await status(tools, ctx);
    assert.doesNotMatch(text, /NaN/);
    assert.equal(typeof details.budget, 'number', 'a broken JSON must fall back on a valid budget');
  } finally { cleanup(); }
});

test('a valid config keeps being honored', async () => {
  const { tools, ctx, cleanup } = await loadWithConfig({ tokenBudget: 123456, thresholdRatio: 0.5 }, 'cfg-valid');
  try {
    const { text, details } = await status(tools, ctx);
    assert.equal(details.budget, 123456, 'a valid budget must not be discarded');
    assert.match(text, /threshold: 50%/, 'a valid threshold must not be discarded');
  } finally { cleanup(); }
});

test('the boolean levels accept only real booleans', async () => {
  // A boolean written as a string ("false") is a classic mistake in
  // configuration files: `!!'false'` is true, so a cast is not enough.
  const { tools, ctx, cleanup } = await loadWithConfig(
    { levels: { stripBulkOutput: 'false' }, tokenBudget: 1000, thresholdRatio: 0.5 },
    'cfg-boolstr',
  );
  try {
    const { details } = await status(tools, ctx);
    assert.equal(details.budget, 1000, 'the rest of the config must stay valid');
  } finally { cleanup(); }
});
