/**
 * Regressione: l'indice BM25 non deve perdere documenti ne' rompersi.
 *
 * Perche' esiste: fino al 30/09/2026 `indexTranscript` usava
 * `idx.add(Number(key) || hashKey(key), ...)` su id esadecimali di Pi.
 * `Number('4e082829')` vale `Infinity` (notazione scientifica!) ed e' truthy,
 * quindi vinceva sull'hash: tutti gli id "scientifici" collassavano su un'unica
 * chiave. `Number('06732791')` perdeva lo zero iniziale e collideva con
 * `'6732791'`. Misurato sui transcript reali: 735 record irraggiungibili in 78
 * sessioni su 832 — contenuto presente nel file che `cwl_recall` non trovava
 * piu'. In silenzio, con `ok: true`.
 *
 * Il secondo bug era accoppiato: `docs` era keyed dal docId NUMERICO, quindi il
 * guard `seen.has(key)` (chiave STRINGA) non scattava mai; e `fromJSON` su
 * un'ISTANZA restituiva un indice vuoto, perche' si aspetta un payload JSON.
 * I due si mascheravano a vicenda.
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as path from 'node:path';
import { pathToFileURL } from 'node:url';

const root = path.resolve(import.meta.dirname, '..');
const { Bm25Index, indexTranscript, MAX_HITS } = await import(
  pathToFileURL(path.join(root, 'recall.mjs')).href
);

/** Costruisce una riga JSONL nel formato dei transcript di Pi. */
const rec = (id, content, role = 'user') =>
  JSON.stringify({ type: 'message', id, timestamp: '2026-01-01T00:00:00.000Z', message: { role, content } });

test('id esadecimali che Number() rende Infinity non collassano', () => {
  // Questi id sono reali: Number() li interpreta come notazione scientifica.
  const idx = indexTranscript(
    [
      rec('4e082829', 'alfa betulla'),
      rec('4e08abcd', 'gamma cedro'),
      rec('4e090001', 'delta ontano'),
    ].join('\n'),
  );
  assert.equal(idx.size, 3, 'tre id distinti devono restare tre documenti');
  for (const [term, expected] of [['betulla', 1], ['cedro', 1], ['ontano', 1]]) {
    assert.equal(idx.search(term).length, expected, `"${term}" deve essere trovabile`);
  }
});

test('un id con zero iniziale non collide con la sua versione senza zero', () => {
  const idx = indexTranscript([rec('06732791', 'alfa betulla'), rec('6732791', 'gamma cedro')].join('\n'));
  assert.equal(idx.size, 2, 'i due id sono diversi e devono restare due documenti');
  assert.equal(idx.search('betulla').length, 1, 'il contenuto del primo id deve essere trovabile');
  assert.equal(idx.search('cedro').length, 1, 'il contenuto del secondo id deve essere trovabile');
});

test('un id alfanumerico non collide con uno numerico', () => {
  const idx = indexTranscript([rec('aaaa1111', 'alfa betulla'), rec('12345678', 'gamma cedro')].join('\n'));
  assert.equal(idx.size, 2);
});

test('indexTranscript accetta un\'ISTANZA e resta incrementale', () => {
  const first = indexTranscript(rec('aaaa1111', 'alfa betulla'));
  assert.equal(first.size, 1);
  // Passare l'istanza viva (come fa index.ts) non deve ricostruire da zero.
  const second = indexTranscript(rec('bbbb2222', 'gamma cedro'), first);
  assert.equal(second, first, 'deve riusare la stessa istanza, non ricrearla');
  assert.equal(second.size, 2, 'il vecchio documento deve sopravvivere all\'update');
  assert.equal(second.search('betulla').length, 1, 'il contenuto indicizzato prima deve restare trovabile');
});

test('ri-indicizzare lo stesso transcript e\' idempotente', () => {
  const raw = [rec('aaaa1111', 'alfa betulla'), rec('bbbb2222', 'gamma cedro')].join('\n');
  const once = indexTranscript(raw);
  const twice = indexTranscript(raw, once);
  assert.equal(twice.size, 2, 'nessun doppio inserimento');
  // Un doppio inserimento gonfia df oltre n, l'IDF diventa NaN e la ricerca si rompe.
  assert.equal(twice.search('betulla').length, 1, 'la ricerca deve restare corretta dopo il re-index');
});

test('il roundtrip toJSON/fromJSON conserva la ricerca', () => {
  const idx = indexTranscript([rec('aaaa1111', 'alfa betulla'), rec('06732791', 'gamma cedro')].join('\n'));
  const restored = Bm25Index.fromJSON(JSON.parse(JSON.stringify(idx.toJSON())));
  assert.equal(restored.size, 2);
  assert.equal(restored.search('betulla').length, 1);
  assert.equal(restored.search('cedro').length, 1);
});

test('fromJSON di un payload numerico legacy non duplica i documenti', () => {
  // Un indice scritto da un build precedente aveva chiavi numeriche.
  const idx = indexTranscript(rec('aaaa1111', 'alfa betulla'));
  const legacy = { version: 2, p: [['betulla', [310539392, 1]]], l: [[310539392, 2]], d: [[310539392, { id: 'aaaa1111', role: 'user', ts: 0, preview: 'alfa betulla', hash: 'aaaa1111' }]], tl: 2 };
  const restored = Bm25Index.fromJSON(legacy);
  assert.equal(restored.size, 1);
  // Il payload legacy e il documento nuovo devono coesistere senza duplicare il termine.
  const merged = indexTranscript(rec('bbbb2222', 'gamma cedro'), restored);
  assert.equal(merged.size, 2, 'il documento legacy e quello nuovo devono essere due');
  assert.ok(idx.size >= 1);
});

test('limit fuori range e\' clampato: non svuota e non inonda il contesto', () => {
  const raw = Array.from({ length: 80 }, (_, i) => rec(`aaaa${String(i).padStart(4, '0')}`, 'alfa comune')).join('\n');
  const idx = indexTranscript(raw);
  assert.equal(idx.size, 80);
  // slice(0, -1) restituirebbe 79 elementi: il clamp deve dare almeno 1.
  assert.equal(idx.search('alfa', -1).length, 1, 'un limit negativo deve dare 1, non 79');
  assert.equal(idx.search('alfa', 0).length, 1, 'un limit zero deve dare 1, non 0');
  assert.equal(idx.search('alfa', NaN).length, 8, 'un limit non finito ricade sul default 8');
  assert.equal(idx.search('alfa', 100000).length, MAX_HITS, 'un limit enorme deve essere tappato a MAX_HITS');
  assert.equal(idx.search('alfa', 3).length, 3, 'un limit valido va rispettato');
});

test('il filtro sulla ricerca non si rompe su un indice corrotto (df > n)', () => {
  const idx = new Bm25Index();
  idx.add('aaaa1111', { id: 'aaaa1111', role: 'user', ts: 0, preview: 'x', hash: 'aaaa1111' }, 'alfa alfa');
  idx.add('bbbb2222', { id: 'bbbb2222', role: 'user', ts: 0, preview: 'y', hash: 'bbbb2222' }, 'beta');
  // Simula il danno di un doppio inserimento: postings con piu' doc di quanti ne esistano.
  idx.postings.set('alfa', [
    { doc: 'aaaa1111', tf: 1 }, { doc: 'bbbb2222', tf: 1 },
    { doc: 'cccc3333', tf: 1 }, { doc: 'dddd4444', tf: 1 }, { doc: 'eeee5555', tf: 1 },
  ]);
  const hits = idx.search('alfa');
  for (const h of hits) {
    assert.ok(Number.isFinite(h.score), 'nessun punteggio NaN deve raggiungere il chiamante');
    assert.ok(h.score > 0);
  }
});

test('record senza id proprio non collassano su un parentId condiviso', () => {
  // Nei transcript reali oggi non accade, ma due fratelli senza id condividono
  // il parentId e collasserebbero in un solo documento.
  const raw = [
    JSON.stringify({ type: 'message', parentId: 'padre1', timestamp: 't', message: { role: 'user', content: 'alfa betulla' } }),
    JSON.stringify({ type: 'message', parentId: 'padre2', timestamp: 't', message: { role: 'user', content: 'gamma cedro' } }),
  ].join('\n');
  const idx = indexTranscript(raw);
  assert.equal(idx.size, 2, 'due parentId distinti devono dare due documenti');
});

test('righe vuote, troncate e non-message non uccidono l\'indice', () => {
  const raw = [
    '',
    '{ json rotto',
    JSON.stringify({ type: 'other', message: { role: 'user', content: 'da ignorare' } }),
    JSON.stringify({ type: 'message', message: { role: 'toolResult', content: 'ignorato' } }),
    rec('aaaa1111', 'alfa betulla'),
    '{"type":"message","id":"troncato","mess',
  ].join('\n');
  const idx = indexTranscript(raw);
  assert.equal(idx.size, 1, 'solo il record valido deve entrare');
  assert.equal(idx.search('betulla').length, 1);
});
