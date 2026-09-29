/**
 * recall — BM25 lexical search over Pi's JSONL transcript.
 *
 * Why BM25 and not embeddings: the search is performed by the LLM, which knows
 * which words to look for. Embeddings are for when the searcher does NOT know the
 * vocabulary ("the cursor changes" for "messageCursor"). Here the LLM knows it, and
 * BM25 is exact, local, deterministic and free of network cost. No
 * vector, no service, no external dependency.
 *
 * Indexing: we do NOT index everything. We index only TEXT MESSAGES
 * (user + assistant). Tool results stay with ARC, which already handles them well
 * and has already replaced them with citations: indexing them here would duplicate the
 * work and confuse the two mechanisms.
 *
 * Persistence: NONE. The index is derived data, it is rebuilt from the
 * transcript in ~100 ms and updated by appending only the new records.
 * Serialising it would cost ~1 MB per session and create a class of bugs
 * (stale index on disk) for a gain that is a novelty.
 */

// ---------------------------------------------------------------------------
// Tokenisation
// ---------------------------------------------------------------------------

// Paths, identifiers and numbers are the strongest signal in a technical
// context: we keep them whole, and we also index their fragments
// separated by . / - _ so that "cwl/index.ts" also matches "index.ts".
const STOP = new Set([
  'il', 'lo', 'la', 'i', 'gli', 'le', 'un', 'uno', 'una', 'di', 'a', 'da', 'in',
  'con', 'su', 'per', 'tra', 'fra', 'e', 'o', 'ma', 'che', 'come', 'del', 'della',
  'dei', 'delle', 'nel', 'nella', 'non', 'piu', 'sono', 'essere', 'ho', 'ha', 'gli',
  'the', 'and', 'for', 'with', 'from', 'this', 'that', 'not', 'but', 'you', 'are',
  'was', 'were', 'have', 'has', 'can', 'will', 'would', 'should', 'into', 'than',
  'then', 'them', 'they', 'its', 'it', 'of', 'to', 'in', 'is', 'be', 'on', 'or',
]);

function normalize(word) {
  return word.toLowerCase().normalize('NFKD').replace(/[̀-ͯ]/g, '');
}

/** Splits paths/identifiers too: index.ts -> index, ts. */
function tokenize(text) {
  if (typeof text !== 'string' || !text) return [];
  const raw = text.match(/[\p{L}\p{N}][\p{L}\p{N}_.+#-]*/gu) ?? [];
  const out = [];
  for (const token of raw) {
    const norm = normalize(token);
    if (norm.length < 2) continue;
    if (!STOP.has(norm)) out.push(norm);
    // Long paths also yield their segments as distinct tokens.
    if (/[._/#-]/.test(norm) && norm.length > 6) {
      for (const part of norm.split(/[._/#-]+/)) {
        if (part.length >= 2 && !STOP.has(part)) out.push(part);
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Indice BM25
// ---------------------------------------------------------------------------

const K1 = 1.5;   // tf saturation
const B = 0.75;   // length normalisation

class Bm25Index {
  constructor() {
    /** @type {Map<string, {doc: number, tf: number}[]>} */
    this.postings = new Map();
    /** @type {Map<number, number>} document length in tokens */
    this.docLen = new Map();
    /** @type {Map<number, {id: string, role: string, ts: number, preview: string, hash: string}>} */
    this.docs = new Map();
    this.totalLen = 0;
  }

  add(docId, meta, text) {
    const tokens = tokenize(text);
    if (tokens.length === 0) return;
    this.docs.set(docId, meta);
    this.docLen.set(docId, tokens.length);
    this.totalLen += tokens.length;
    const tf = new Map();
    for (const t of tokens) tf.set(t, (tf.get(t) ?? 0) + 1);
    for (const [term, freq] of tf) {
      let list = this.postings.get(term);
      if (!list) { list = []; this.postings.set(term, list); }
      list.push({ doc: docId, tf: freq });
    }
  }

  get size() { return this.docs.size; }

  /**
   * BM25 search. Returns the most relevant doc ids, already filtered to non-zero.
   */
  search(query, limit = 8) {
    const qTerms = tokenize(query);
    if (qTerms.length === 0 || this.docs.size === 0) return [];
    const avgLen = this.totalLen / this.docs.size || 1;
    const scores = new Map();

    for (const term of qTerms) {
      const list = this.postings.get(term);
      if (!list) continue;
      // positive idf: a term present in every document distinguishes nothing.
      const n = this.docs.size;
      const df = list.length;
      const idf = Math.log(1 + (n - df + 0.5) / (df + 0.5));
      if (idf <= 0) continue;
      for (const { doc, tf } of list) {
        const dl = this.docLen.get(doc) ?? 1;
        const denom = tf + K1 * (1 - B + (B * dl) / avgLen);
        scores.set(doc, (scores.get(doc) ?? 0) + (idf * (tf * (K1 + 1))) / denom);
      }
    }

    return [...scores.entries()]
      .filter(([, score]) => score > 0)
      .sort((a, b) => b[1] - a[1])
      .slice(0, limit)
      .map(([doc, score]) => ({ ...this.docs.get(doc), score }));
  }

  /**
   * Compact serialisation: postings are [doc, tf] juxtaposed in a flat array
   * instead of [{doc, tf}]. With ~55k postings the difference is ~10x in
   * size, and the index lives in RAM, not on disk.
   */
  toJSON() {
    return {
      version: 2,
      p: [...this.postings.entries()].map(([t, l]) => [t, l.flatMap((x) => [x.doc, x.tf])]),
      l: [...this.docLen],
      d: [...this.docs],
      tl: this.totalLen,
    };
  }

  static fromJSON(data) {
    const idx = new Bm25Index();
    if (!data) return idx;
    if (data.version === 1) {
      // Legacy format: a one-off migration, then v2 is written.
      idx.postings = new Map((data.postings ?? []).map(([t, l]) => [t, l.map(([doc, tf]) => ({ doc, tf }))]));
      idx.docLen = new Map(data.docLen ?? []);
      idx.docs = new Map(data.docs ?? []);
      idx.totalLen = data.totalLen ?? 0;
      return idx;
    }
    if (data.version !== 2) return idx;
    idx.postings = new Map(
      (data.p ?? []).map(([t, flat]) => {
        const list = [];
        for (let i = 0; i < flat.length; i += 2) list.push({ doc: flat[i], tf: flat[i + 1] });
        return [t, list];
      }),
    );
    idx.docLen = new Map(data.l ?? []);
    idx.docs = new Map(data.d ?? []);
    idx.totalLen = data.tl ?? 0;
    return idx;
  }
}

// ---------------------------------------------------------------------------
// Text extraction from the transcript
// ---------------------------------------------------------------------------

/** Concatenable text from a content that can be a string or a list of blocks. */
export function contentToText(content) {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const out = [];
  for (const block of content) {
    if (typeof block === 'string') { out.push(block); continue; }
    if (!block || typeof block !== 'object') continue;
    if (typeof block.text === 'string') out.push(block.text);
    else if (typeof block.thinking === 'string') out.push(block.thinking);
    else if (typeof block.name === 'string') out.push(`\$${block.name}(`); // tool name only
  }
  return out.join('\n');
}

const INDEXABLE_ROLES = new Set(['user', 'assistant']);

/**
 * Indexes a Pi JSONL transcript. Idempotent and incremental: it can be called
 * every turn: it only indexes the new records.
 */
export function indexTranscript(jsonlText, existing = null) {
  const idx = existing ? Bm25Index.fromJSON(existing) : new Bm25Index();
  const seen = new Set(idx.docs.keys());
  const lines = jsonlText.split(/\r?\n/);
  // The doc id is the record id: hash-independent, stable, already in the file.
  const docKey = (record) => String(record.id ?? record.parentId ?? '');

  for (const line of lines) {
    if (!line) continue;
    let record;
    try { record = JSON.parse(line); } catch { continue; }
    if (record.type !== 'message') continue;
    const message = record.message;
    if (!message) continue;
    // ARC handles toolResults: we do not duplicate them here.
    if (!INDEXABLE_ROLES.has(message.role)) continue;

    const key = docKey(record);
    if (!key || seen.has(key)) continue;
    const text = contentToText(message.content);
    if (!text.trim()) continue;
    seen.add(key);
    idx.add(Number(key) || hashKey(key), {
      id: key,
      role: message.role,
      ts: record.timestamp ?? 0,
      preview: text.slice(0, 200).replace(/\s+/g, ' '),
      hash: key,
    }, text);
  }
  return idx;
}

/** For non-numeric ids: a stable numeric hash. */
function hashKey(key) {
  let h = 0;
  for (let i = 0; i < key.length; i++) h = (h * 31 + key.charCodeAt(i)) | 0;
  return h >>> 0;
}

export { Bm25Index, tokenize };
