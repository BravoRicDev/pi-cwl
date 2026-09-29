export interface DocMeta {
  id: string;
  role: string;
  ts: number;
  preview: string;
  hash: string;
}

export interface Hit extends DocMeta {
  score: number;
}

export class Bm25Index {
  readonly size: number;
  add(docId: number, meta: DocMeta, text: string): void;
  search(query: string, limit?: number): Hit[];
  toJSON(): unknown;
  static fromJSON(data: unknown): Bm25Index;
}

export function tokenize(text: string): string[];
export function contentToText(content: unknown): string;
export function indexTranscript(jsonlText: string, existing?: Bm25Index | null): Bm25Index;
