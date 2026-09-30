/**
 * Compact serialised form written by `toJSON`: v2 pairs each posting list as a
 * flat [doc, tf, doc, tf, ...] array, which is ~10x smaller than objects.
 */
export interface SerializedIndex {
  version: 2;
  /** term -> flat [doc, tf, ...] pairs */
  p: [string, (string | number)[]][];
  /** doc id -> token count */
  l: [string, number][];
  /** doc id -> metadata */
  d: [string, DocMeta][];
  /** total token count */
  tl: number;
}

export interface DocMeta {
  id: string;
  role: string;
  /** Pi writes an ISO-8601 string; an older payload may hold a number. */
  ts: string | number;
  preview: string;
  hash: string;
}

export interface Hit extends DocMeta {
  score: number;
}

export const MAX_HITS: number;

export class Bm25Index {
  readonly size: number;
  /** Doc ids are strings: the internal maps are keyed by string. */
  add(docId: string, meta: DocMeta, text: string): void;
  /** `limit` is clamped to 1..MAX_HITS; a non-finite value falls back to 8. */
  search(query: string, limit?: number): Hit[];
  toJSON(): SerializedIndex;
  static fromJSON(data: unknown): Bm25Index;
}

export function tokenize(text: string): string[];
export function contentToText(content: unknown): string;
export function indexTranscript(jsonlText: string, existing?: Bm25Index | null): Bm25Index;
