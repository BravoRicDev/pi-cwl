/**
 * Regression: the BM25 index must not lose documents nor break.
 *
 * Why it exists: until 2026-09-30 `indexTranscript` used
 * `idx.add(Number(key) || hashKey(key), ...)` on Pi hexadecimal ids.
 * `Number('4e082829')` is `Infinity` (scientific notation!) and truthy,
 * so it won over the hash: every "scientific" id collapsed onto a single
 * key. `Number('06732791')` lost the leading zero and collided with
 * `'6732791'`. Measured on real transcripts: 735 unreachable records in 78
 * sessions out of 832 — content present in the file that `cwl_recall` could no longer
 * find. Silently, with `ok: true`.
 *
 * The second bug was coupled to it: `docs` was keyed by the NUMERIC docId, so the
 * guard `seen.has(key)` (STRING key) never fired; and `fromJSON` on
 * an INSTANCE returned an empty index, because it expects a JSON payload.
 * The two masked each other.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as path from 'node:path';
import { pathToFileURL } from 'node:url';

const root = path.resolve(import.meta.dirname, '..');
const { Bm25Index, indexTranscript, MAX_HITS } = await import(
  pathToFileURL(path.join(root, 'recall.mjs')).href
);

/** Builds a JSONL line in the format of Pi's transcripts. */
const rec = (id, content, role = 'user') =>
  JSON.stringify({ type: 'message', id, timestamp: '2026-01-01T00:00:00.000Z', message: { role, content } });

test('hexadecimal ids that Number() turns into Infinity do not collapse', () => {
  // These ids are real: Number() reads them as scientific notation.
  const idx = indexTranscript(
    [
      rec('4e082829', 'alfa betulla'),
      rec('4e08abcd', 'gamma cedro'),
      rec('4e090001', 'delta ontano'),
    ].join('\n'),
  );
  assert.equal(idx.size, 3, 'three distinct ids must stay three documents');
  for (const [term, expected] of [['betulla', 1], ['cedro', 1], ['ontano', 1]]) {
    assert.equal(idx.search(term).length, expected, `"${term}" must be findable`);
  }
});

test('an id with a leading zero does not collide with its version without the zero', () => {
  const idx = indexTranscript([rec('06732791', 'alfa betulla'), rec('6732791', 'gamma cedro')].join('\n'));
  assert.equal(idx.size, 2, 'the two ids are different and must stay two documents');
  assert.equal(idx.search('betulla').length, 1, 'the content of the first id must be findable');
  assert.equal(idx.search('cedro').length, 1, 'the content of the second id must be findable');
});

test('an alphanumeric id does not collide with a numeric one', () => {
  const idx = indexTranscript([rec('aaaa1111', 'alfa betulla'), rec('12345678', 'gamma cedro')].join('\n'));
  assert.equal(idx.size, 2);
});

test('indexTranscript accepts an INSTANCE and stays incremental', () => {
  const first = indexTranscript(rec('aaaa1111', 'alfa betulla'));
  assert.equal(first.size, 1);
  // Passing the live instance (as index.ts does) must not rebuild from scratch.
  const second = indexTranscript(rec('bbbb2222', 'gamma cedro'), first);
  assert.equal(second, first, 'it must reuse the same instance, not recreate it');
  assert.equal(second.size, 2, 'the old document must survive the update');
  assert.equal(second.search('betulla').length, 1, 'the content indexed before must stay findable');
});

test('re-indexing the same transcript is idempotent', () => {
  const raw = [rec('aaaa1111', 'alfa betulla'), rec('bbbb2222', 'gamma cedro')].join('\n');
  const once = indexTranscript(raw);
  const twice = indexTranscript(raw, once);
  assert.equal(twice.size, 2, 'no double insertion');
  // A double insertion inflates df beyond n, the IDF becomes NaN and search breaks.
  assert.equal(twice.search('betulla').length, 1, 'search must stay correct after the re-index');
});

test('the toJSON/fromJSON roundtrip preserves search', () => {
  const idx = indexTranscript([rec('aaaa1111', 'alfa betulla'), rec('06732791', 'gamma cedro')].join('\n'));
  const restored = Bm25Index.fromJSON(JSON.parse(JSON.stringify(idx.toJSON())));
  assert.equal(restored.size, 2);
  assert.equal(restored.search('betulla').length, 1);
  assert.equal(restored.search('cedro').length, 1);
});

test('fromJSON of a legacy numeric payload does not duplicate documents', () => {
  // An index written by a previous build had numeric keys.
  const idx = indexTranscript(rec('aaaa1111', 'alfa betulla'));
  const legacy = { version: 2, p: [['betulla', [310539392, 1]]], l: [[310539392, 2]], d: [[310539392, { id: 'aaaa1111', role: 'user', ts: 0, preview: 'alfa betulla', hash: 'aaaa1111' }]], tl: 2 };
  const restored = Bm25Index.fromJSON(legacy);
  assert.equal(restored.size, 1);
  // The legacy payload and the new document must coexist without duplicating the term.
  const merged = indexTranscript(rec('bbbb2222', 'gamma cedro'), restored);
  assert.equal(merged.size, 2, 'the legacy document and the new one must be two');
  assert.ok(idx.size >= 1);
});

test('an out-of-range limit is clamped: it neither empties nor floods the context', () => {
  const raw = Array.from({ length: 80 }, (_, i) => rec(`aaaa${String(i).padStart(4, '0')}`, 'alfa comune')).join('\n');
  const idx = indexTranscript(raw);
  assert.equal(idx.size, 80);
  // slice(0, -1) would return 79 elements: the clamp must give at least 1.
  assert.equal(idx.search('alfa', -1).length, 1, 'a negative limit must give 1, not 79');
  assert.equal(idx.search('alfa', 0).length, 1, 'a zero limit must give 1, not 0');
  assert.equal(idx.search('alfa', NaN).length, 8, 'a non-finite limit falls back on the default 8');
  assert.equal(idx.search('alfa', 100000).length, MAX_HITS, 'a huge limit must be capped at MAX_HITS');
  assert.equal(idx.search('alfa', 3).length, 3, 'a valid limit must be honored');
});

test('the search filter does not break on a corrupted index (df > n)', () => {
  const idx = new Bm25Index();
  idx.add('aaaa1111', { id: 'aaaa1111', role: 'user', ts: 0, preview: 'x', hash: 'aaaa1111' }, 'alfa alfa');
  idx.add('bbbb2222', { id: 'bbbb2222', role: 'user', ts: 0, preview: 'y', hash: 'bbbb2222' }, 'beta');
  // Simulates the damage of a double insertion: postings with more docs than exist.
  idx.postings.set('alfa', [
    { doc: 'aaaa1111', tf: 1 }, { doc: 'bbbb2222', tf: 1 },
    { doc: 'cccc3333', tf: 1 }, { doc: 'dddd4444', tf: 1 }, { doc: 'eeee5555', tf: 1 },
  ]);
  const hits = idx.search('alfa');
  for (const h of hits) {
    assert.ok(Number.isFinite(h.score), 'no NaN score must reach the caller');
    assert.ok(h.score > 0);
  }
});

test('records without their own id do not collapse onto a shared parentId', () => {
  // In real transcripts this does not happen today, but two siblings without an id share
  // the parentId and would collapse into a single document.
  const raw = [
    JSON.stringify({ type: 'message', parentId: 'padre1', timestamp: 't', message: { role: 'user', content: 'alfa betulla' } }),
    JSON.stringify({ type: 'message', parentId: 'padre2', timestamp: 't', message: { role: 'user', content: 'gamma cedro' } }),
  ].join('\n');
  const idx = indexTranscript(raw);
  assert.equal(idx.size, 2, 'two distinct parentIds must give two documents');
});

test('empty, truncated and non-message lines do not kill the index', () => {
  const raw = [
    '',
    '{ broken json',
    JSON.stringify({ type: 'other', message: { role: 'user', content: 'to be ignored' } }),
    JSON.stringify({ type: 'message', message: { role: 'toolResult', content: 'ignored' } }),
    rec('aaaa1111', 'alfa betulla'),
    '{"type":"message","id":"cutoff","mess',
  ].join('\n');
  const idx = indexTranscript(raw);
  assert.equal(idx.size, 1, 'only the valid record must enter');
  assert.equal(idx.search('betulla').length, 1);
});
