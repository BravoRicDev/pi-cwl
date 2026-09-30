/**
 * A REGION ALREADY INSIDE A LEAF IS NOT COMPRESSED AGAIN.
 *
 * `compressibleRange` skips every index that lies inside a resolved span, so the OFFERED
 * road (`cwl_compress_range`) cannot describe the same messages twice. The EXPLICIT tool
 * (`cwl_compress`) does not: it validated the two hashes against `knownHashes` and pushed the leaf,
 * knowing nothing about the coverage. An agent choosing the hashes by hand could therefore
 * create a leaf overlapping an existing one — the region described twice, and the
 * two descriptions free to diverge over time.
 *
 * The tool has no access to the message list (its own comment says so), but the context's
 * `sessionManager` exposes `buildContextEntries()`, which is what Pi builds
 * the context list from: from there one rebuilds the SAME ORDER of addresses, and for a
 * comparison between intervals the order is all that is needed.
 *
 * Two cases, and both are needed:
 *  (1) an interval inside a leaf is REFUSED, with the reason in the log;
 *  (2) the SAME call, when the endpoints are not placeable on the list read, GOES THROUGH and
 *      declares that it did not check. A check that always refuses would be indistinguishable
 *      from an honest check without this second case.
 *
 * DECLARED — what this test does NOT kill: the mutation that makes `covers` always answer
 * "covered" stays ALIVE, because the third case that would be needed — a candidate that IS
 * placeable and NOT covered — does not exist in this fixture: with `protectedTurns: 0` the leaf
 * covers the whole list, so every hash the tool accepts falls inside it. Case (2) does not kill it
 * because it exits first, on the `unknown` branch. A fixture with an uncovered tail is needed (for
 * example a higher `protectedTurns`, and a hash taken from that tail). Better to say it than to let
 * people believe it is covered.
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
  const sandbox = makeSandbox({ name: `coverage-${seq++}`, config: config() });
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

const logDi = (sandbox) => fs.readFileSync(path.join(sandbox.dir, '.pi', 'cwl', 'cwl.log'), 'utf8');

const statoDi = (sandbox) => {
  const dir = path.join(sandbox.dir, '.pi', 'cwl', 'state');
  const file = fs.readdirSync(dir).find((f) => f.endsWith('.json'));
  assert.ok(file, 'the session state was not created');
  return JSON.parse(fs.readFileSync(path.join(dir, file), 'utf8'));
};

const conversazione = (da, a) => {
  const out = [];
  for (let i = da; i <= a; i++) {
    out.push({ role: 'user', content: `prompt ${i} content. ` + 'U'.repeat(200), timestamp: 1000 + i * 2 });
    out.push({ role: 'assistant', content: `answers ${i} content is ` + 'A'.repeat(200), timestamp: 1001 + i * 2 });
  }
  return out;
};

test('an interval inside a leaf is refused, and the same thing without coverage goes through', async () => {
  const { sandbox, home, tools, hooks, ctx } = await boot();
  try {
    const base = conversazione(1, 7);
    await hook(hooks, ctx, base); // the hook computes the offered address
    ctx.__mostra(base); // this is the "session" the tools will see

    const primo = await tools
      .get('cwl_compress_range')
      .execute('t', { summary: 'BLOCK-1 ' + 'x'.repeat(300) }, undefined, undefined, ctx);
    assert.equal(primo.details.ok, true, `the leaf was not born: ${JSON.stringify(primo.details)}`);

    const foglia = statoDi(sandbox).spans[0];
    assert.ok(foglia && foglia.startHash, 'no leaf with an address in the state: the test proves nothing');
    // `cwl_compress` validates the two hashes against `knownHashes`, which holds the hash of the TEXT of
    // every message (not the `id|testo` address the leaves carry: they are two different
    // namespaces, and `locateSpans` resolves both through `addressMaps`). The test therefore takes a
    // hash from the set the tool accepts — the oldest one — and uses it as an endpoint.
    const stato = statoDi(sandbox);
    const h = stato.knownHashes[0];
    assert.ok(typeof h === 'string' && h.length > 0, 'no persisted hash: the tool would refuse with unknown-hash');

    // (1) An interval that falls INSIDE the leaf: refused, with the reason in the log.
    const primaDentro = logDi(sandbox).length;
    const dentro = await tools
      .get('cwl_compress')
      .execute('t', { startHash: h, endHash: h, summary: 'BLOCK-2' }, undefined, undefined, ctx);
    assert.equal(
      dentro.details.error,
      'covered-range',
      `expected the coverage refusal, got ${JSON.stringify(dentro.details)}`,
    );
    assert.match(
      logDi(sandbox).slice(primaDentro),
      /COMPRESS refused [0-9a-f]+\.\.[0-9a-f]+: inside a live leaf \(\d+ span\(s\) resolved, \d+ message\(s\) read\)/,
      'the refusal does not say why, and does not declare on how many messages it looked',
    );
    assert.equal(statoDi(sandbox).spans.length, 1, 'the refused leaf ended up in the state anyway');

    // (2) NON-VACUITY: the same call with endpoints NOT placeable on the list read
    //     goes through, and the log must DECLARE that it did not check (never accept in silence).
    //     The fake session shows only the LAST message: the oldest hash of the set is no
    //     longer placeable on that list.
    ctx.__mostra(base.slice(-1));
    const primaFuori = logDi(sandbox).length;
    const fuori = await tools
      .get('cwl_compress')
      .execute('t', { startHash: h, endHash: h, summary: 'BLOCK-3' }, undefined, undefined, ctx);
    assert.equal(
      fuori.details.ok,
      true,
      `expected the declared pass-through, got ${JSON.stringify(fuori.details)}: a check that cannot place ` +
        'the endpoints must not reject a legitimate request',
    );
    assert.match(
      logDi(sandbox).slice(primaFuori),
      /COMPRESS coverage: endpoints of .* are not placeable on the \d+ message\(s\) read — let through, NOT checked/,
      'the pass-through was not declared in the log: a check that does not run silently is the same thing as a missing check',
    );
  } finally {
    home.restore();
  }
});
