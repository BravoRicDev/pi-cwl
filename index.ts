/**
 * CWL — Context Window Lifecycle
 * Structured context eviction for long-horizon Pi agents.
 *
 * PROBLEMA
 *   La compattazione a soglia (summarization) blocca il turno, perde informazione
 *   in modo imprevedibile, distrugge la struttura causale e introduce allucinazioni
 *   proprio quando il budget di contesto e' sotto pressione.
 *
 * SOLUZIONE (https://arxiv.org/html/2606.11213)
 *   The agent annotates its own trajectory as typed episodes (expl/act)
 *   tramite un tool `delimiter`. Una politica deterministica, LLM-free, evicta
 *   the content in order of recoverability once the budget is exceeded.
 *
 *   - expl (esplorazione): output di ricerca, listing, letture di orientamento.
 *     On closing, the agent supplies a description: the only content kept.
 *   - act (action): writes, edits, tool calls. The effects are persistent in
 *     the environment, so they are the first candidates for eviction.
 *
 *   Le dipendenze dichiarate impediscono di perdere il contesto esplorativo che
 *   produced a decision that is still active.
 *
 *   User content e' inviolabile (Principio 3).
 *   Compression NEVER invokes the model (Principle 5): zero cost, zero
 *   allucinazione introdotta, zero blocco.
 *
 * FORMA REALE DEI MESSAGGI (da @earendil-works/pi-ai)
 *   UserMessage      : { role: "user",      content: string | Content[] }
 *   AssistantMessage : { role: "assistant", content: (Text|Thinking|ToolCall)[] }
 *   ToolResultMessage: { role: "toolResult",content: (Text|Image)[] }
 *   Nota: il role dei tool result e' "toolResult", NON "tool".
 */

import type { ExtensionAPI, ExtensionContext } from '@earendil-works/pi-coding-agent';
import type { AgentMessage } from '@earendil-works/pi-agent-core';
import { Type } from 'typebox';
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';
import { fileURLToPath } from 'node:url';

// ---------------------------------------------------------------------------
// Minimal types for the real message shape (subset of @earendil-works/pi-ai)
// ---------------------------------------------------------------------------

interface RealContentBlock {
  type?: string;
  text?: string;
  thinking?: string;
  data?: string;
}

interface RealMessage {
  role?: string;
  content?: unknown;
  toolCallId?: string;
  toolName?: string;
  timestamp?: number;
}

// ---------------------------------------------------------------------------
// i18n
// ---------------------------------------------------------------------------

type Lang = 'en' | 'it';

const FALLBACK: Lang = 'en';

function primaryOf(tag: string): string | null {
  if (typeof tag !== 'string') return null;
  // Same regex as pi-anti-amnesia/i18n.mjs and pi-cron-bg: a naive split on
  // "." yields "it_it" for "it_IT.UTF-8", which is not a supported language.
  const m = /^\s*([A-Za-z]{2,3})(?:-|_)/.exec(tag) ?? /^\s*([A-Za-z]{2,3})\s*$/.exec(tag);
  return m ? m[1].toLowerCase() : null;
}

function isSupported(primary: string | null): primary is Lang {
  return primary === 'en' || primary === 'it';
}

/**
 * Resolves the system language once, at module load.
 *
 * Must stay in lockstep with pi-anti-amnesia and pi-cron-bg: several
 * extensions inject instructions into the same model context, and a model fed
 * mixed-language directives degrades. Same order (LC_ALL > LC_MESSAGES > LANG
 * > LANGUAGE, then Intl, then English), same unsupported-locale handling.
 */
function detectLang(): Lang {
  const env = process.env;
  for (const name of ['LC_ALL', 'LC_MESSAGES', 'LANG', 'LANGUAGE']) {
    const tag = env?.[name];
    if (!tag || tag === 'C' || tag === 'POSIX') continue;
    const primary = primaryOf(tag);
    if (isSupported(primary)) return primary;
  }
  try {
    const icu = primaryOf(Intl.DateTimeFormat().resolvedOptions().locale ?? '');
    if (isSupported(icu)) return icu;
  } catch { /* no ICU data */ }
  return FALLBACK;
}

const LANG: Lang = detectLang();

type CwlMessages = {
  guidelines: string[];
  strippedReasoning: string;
  truncatedString: (len: number, level: StripLevel) => string;
  truncatedArray: (len: number, level: StripLevel) => string;
  evictedEpisode: (name: string, desc: string) => string;
  startNeedsNameType: string;
  endNeedsName: string;
  duplicateName: (name: string) => string;
  notFoundOrClosed: (name: string) => string;
  invalidDeps: (names: string) => string;
  episodeOpened: (type: string, name: string) => string;
  unknownAction: string;
  emptyDescriptionWarning: string;
  statusHeader: (budget: string, threshold: string) => string;
  statusEpisodes: (total: number, active: number, closed: number, stripped: number) => string;
  statusEvictions: (count: number, tokens: string) => string;
};

const I18N: Record<Lang, CwlMessages> = {
  en: {
    guidelines: [
      'Open an expl episode when you start exploring (reads, searches, listings). Close it as soon as you have the answer you need.',
      'Open an act episode when you make a change (write files, run commands with effects). Declare the exploration episodes it depends on.',
      'When closing an expl, give a concise description of what you learned: it is the only content that survives eviction.',
      'Do not open overly fine-grained episodes: 5-15 per session is a good target.',
    ],
    strippedReasoning: '[CWL: reasoning evicted]',
    truncatedString: (len, level) => `[CWL: content reduced from ${len} chars (level ${level})]`,
    truncatedArray: (len, level) => `[CWL: text reduced from ${len} chars to a marker (level ${level}) — use arc_recall or reopen the episode if needed]`,
    evictedEpisode: (name, desc) => `Content of "${name}" evicted. Notes kept: ${desc}`,
    startNeedsNameType: 'action=start requires name and type.',
    endNeedsName: 'action=end requires name.',
    duplicateName: (name) => `Episode "${name}" already exists. Use a unique name.`,
    notFoundOrClosed: (name) => `Episode "${name}" not found or already closed.`,
    invalidDeps: (names) => `Invalid dependencies (must be closed expl episodes): ${names}`,
    episodeOpened: (type, name) => `Episode ${type} "${name}" opened.`,
    unknownAction: 'Unrecognised action.',
    emptyDescriptionWarning: ' WARNING: empty description — this episode has no fallback content.',
    statusHeader: (budget, threshold) => `CWL — token budget: ${budget} (threshold: ${threshold}%)`,
    statusEpisodes: (total, active, closed, stripped) => `Episodes total: ${total} | active: ${active} | closed: ${closed} | stripped: ${stripped}`,
    statusEvictions: (count, tokens) => `Evictions total: ${count} | tokens saved: ${tokens}`,
  },
  it: {
    guidelines: [
      'Apri un episodio expl quando inizi a esplorare (letture, ricerche, listing). Chiudilo appena hai la risposta che ti serve.',
      'Apri un episodio act quando fai una modifica (scrivi file, esegui comandi con effetti). Dichiara le esplorazioni da cui dipende.',
      'Alla chiusura di un expl, fornisci una descrizione concisa di cosa hai imparato: e\' il contenuto che sopravvive all\'eviction.',
      'Non aprire episodi troppo fini: 5-15 per sessione e\' un buon target.',
    ],
    strippedReasoning: '[CWL: reasoning evictato]',
    truncatedString: (len, level) => `[CWL: contenuto ridotto da ${len} chars (livello ${level})]`,
    truncatedArray: (len, level) => `[CWL: testo ridotto da ${len} chars a marker (livello ${level}) — usa arc_recall o riapri l'episodio se serve]`,
    evictedEpisode: (name, desc) => `Contenuto di "${name}" evictato. Appunti conservati: ${desc}`,
    startNeedsNameType: 'action=start richiede name e type.',
    endNeedsName: 'action=end richiede name.',
    duplicateName: (name) => `Episodio "${name}" gia' esiste. Usa un nome univoco.`,
    notFoundOrClosed: (name) => `Episodio "${name}" non trovato o gia' chiuso.`,
    invalidDeps: (names) => `Dipendenze non valide (devono essere episodi expl chiusi): ${names}`,
    episodeOpened: (type, name) => `Episodio ${type} "${name}" aperto.`,
    unknownAction: 'Azione non riconosciuta.',
    emptyDescriptionWarning: ' ATTENZIONE: descrizione vuota — questo episodio non ha contenuto di fallback.',
    statusHeader: (budget, threshold) => `CWL — token budget: ${budget} (threshold: ${threshold}%)`,
    statusEpisodes: (total, active, closed, stripped) => `Episodi totali: ${total} | attivi: ${active} | chiusi: ${closed} | gia' stripped: ${stripped}`,
    statusEvictions: (count, tokens) => `Eviction totali: ${count} | token risparmiati: ${tokens}`,
  },
} satisfies Record<Lang, CwlMessages>;

/** Localised string for the active language. */
function t<K extends keyof CwlMessages>(key: K): CwlMessages[K] {
  return I18N[LANG][key];
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

interface CwlConfig {
  /** Budget di token attivi oltre il quale l'eviction parte. */
  tokenBudget: number;
  /** Soglia di attivazione come frazione del budget (0..1). */
  thresholdRatio: number;
  /** Livelli di aggressivita' abilitati. */
  levels: {
    stripReasoning: boolean;
    stripBulkOutput: boolean;
    stripIntermediate: boolean;
    removeEpisode: boolean;
  };
  /** Renderizza widget UI con lo stato. */
  showWidget: boolean;
  /** Log di debug su file. */
  debug: boolean;
}

const DEFAULT_CONFIG: CwlConfig = {
  // 80k = ~30% di un contesto da 256k; CWL paper: regime sotto il quale
  // attention does not degrade. If your context is 1M, raise this value.
  tokenBudget: 80_000,
  thresholdRatio: 0.85,
  levels: {
    stripReasoning: true,
    stripBulkOutput: true,
    stripIntermediate: true,
    removeEpisode: true,
  },
  showWidget: true,
  debug: false,
};

// __dirname equivalente in ESM.
const _EXT_DIR = path.dirname(fileURLToPath(import.meta.url));
// Config utente; se assente si cade sul config incluso nell'estensione, cosi' il
// file distribuito col repo serve a qualcosa invece di restare un documento morto.
const CONFIG_PATH = path.join(os.homedir(), '.pi', 'cwl', 'config.json');
const BUNDLED_CONFIG_PATH = path.join(_EXT_DIR, 'config.json');
const LOG_PATH = path.join(os.homedir(), '.pi', 'cwl', 'cwl.log');

/**
 * Deep-merge: i livelli devono essere uniti singolarmente, altrimenti un
 * config.json parziale (es. { levels: { stripBulkOutput: false } }) sostituirebbe
 * l'intero blocco levels perdendo le altre tre flag.
 */
function loadConfig(): CwlConfig {
  for (const candidate of [CONFIG_PATH, BUNDLED_CONFIG_PATH]) {
    try {
      const raw = fs.readFileSync(candidate, 'utf8');
      const user = JSON.parse(raw) as Partial<CwlConfig> & { levels?: Partial<CwlConfig['levels']> };
      return {
        ...DEFAULT_CONFIG,
        ...user,
        levels: { ...DEFAULT_CONFIG.levels, ...(user.levels ?? {}) },
      };
    } catch {
      // prova il candidato successivo
    }
  }
  return { ...DEFAULT_CONFIG, levels: { ...DEFAULT_CONFIG.levels } };
}

function debugLog(cfg: CwlConfig, msg: string) {
  if (!cfg.debug) return;
  try {
    fs.mkdirSync(path.dirname(LOG_PATH), { recursive: true });
    fs.appendFileSync(LOG_PATH, `[${new Date().toISOString()}] ${msg}\n`);
  } catch { /* non critico */ }
}

// ---------------------------------------------------------------------------
// Token estimation (approximate, no tokenizer call)
// ---------------------------------------------------------------------------

/** Stima token: ~4 caratteri per token per testo inglese; e' un overestimate per code. */
function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

function estimateMessageTokens(msg: unknown): number {
  try {
    const s = JSON.stringify(msg);
    return estimateTokens(s);
  } catch {
    return 0;
  }
}

// ---------------------------------------------------------------------------
// Content helpers (forma reale: array di blocchi)
// ---------------------------------------------------------------------------

/** Estrae il testo concatenato da stringa o array di blocchi content. */
function contentToText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  let out = '';
  for (const part of content as RealContentBlock[]) {
    if (typeof part?.text === 'string') out += part.text;
    else if (typeof part?.thinking === 'string') out += part.thinking;
  }
  return out;
}

/** Numero di caratteri di testo effettivamente presente nel messaggio. */
function textLengthOf(msg: unknown): number {
  const m = msg as RealMessage;
  return contentToText(m?.content).length;
}

// ---------------------------------------------------------------------------
// Episode Graph
// ---------------------------------------------------------------------------

type EpisodeType = 'expl' | 'act';
type StripLevel = 'none' | 'reasoning' | 'bulk' | 'intermediate' | 'removed';

interface Episode {
  name: string;
  type: EpisodeType;
  /** Nomi degli episodi expl da cui questo atto dipende. */
  dependencies: string[];
  /**
   * toolCallId del risultato `delimiter` che ha aperto l'episodio: e' l'ancoraggio
   * AFFIDABILE. L'indice non lo e', perche' il cursore si aggiorna solo
   * nell'hook context (prima della chiamata LLM) mentre il tool gira nel turno.
   */
  startToolCallId: string;
  /** Come startToolCallId, per la chiusura. */
  endToolCallId: string | null;
  /** Indice diagnostico, non usato per la mappatura. */
  startIdx: number;
  /** Indice diagnostico, non usato per la mappatura. */
  endIdx: number | null;
  /** Descrizione fornita dall'agente alla chiusura (solo expl). */
  description: string;
  /** Livello di stripping attualmente applicato. */
  level: StripLevel;
  /** Timestamp di apertura. */
  openedAt: number;
}

class EpisodeGraph {
  private episodes: Episode[] = [];
  private activeNames: Set<string> = new Set();
  private nameSet: Set<string> = new Set();

  open(name: string, type: EpisodeType, dependencies: string[], startIdx: number, startToolCallId: string): Episode {
    const ep: Episode = {
      name, type, dependencies,
      startToolCallId, endToolCallId: null,
      startIdx, endIdx: null,
      description: '',
      level: 'none',
      openedAt: Date.now(),
    };
    this.episodes.push(ep);
    this.activeNames.add(name);
    this.nameSet.add(name);
    return ep;
  }

  close(name: string, description: string, endIdx: number, endToolCallId: string): Episode | null {
    const ep = this.episodes.find(e => e.name === name && e.endToolCallId === null);
    if (!ep) return null;
    ep.endIdx = endIdx;
    ep.endToolCallId = endToolCallId;
    ep.description = description || ep.description;
    this.activeNames.delete(name);
    return ep;
  }

  get isEmpty() { return this.episodes.length === 0; }
  get count() { return this.episodes.length; }
  get all() { return this.episodes; }

  active(): Episode[] { return this.episodes.filter(e => e.endIdx === null); }

  /** Episodi chiusi il cui contenuto esiste ancora in contesto. */
  recoverable(): Episode[] {
    return this.episodes.filter(e => e.endIdx !== null && e.level !== 'removed');
  }

  /** Episodi chiusi di QUALSIASI livello: serve per mappare gli indici. */
  closed(): Episode[] {
    return this.episodes.filter(e => e.endIdx !== null);
  }

  /** true se un episodio con questo nome e' gia' stato creato (evita cicli). */
  has(name: string) { return this.nameSet.has(name); }

  /** Dipendenti non ancora completamente evictati. */
  hasLiveDependents(name: string): boolean {
    return this.episodes.some(ep =>
      ep.name !== name &&
      ep.dependencies.includes(name) &&
      ep.level !== 'removed'
    );
  }

  /** Reset completo (nuova sessione). */
  reset() {
    this.episodes = [];
    this.activeNames.clear();
    this.nameSet.clear();
  }

  /** Serializza per persistenza. */
  toJSON() { return { episodes: this.episodes }; }

  static fromJSON(data: unknown): EpisodeGraph {
    const g = new EpisodeGraph();
    const eps = (data as { episodes?: Episode[] })?.episodes;
    if (Array.isArray(eps)) {
      for (const ep of eps) {
        g.episodes.push(ep);
        g.nameSet.add(ep.name);
        if (ep.endIdx === null) g.activeNames.add(ep.name);
      }
    }
    return g;
  }
}

// ---------------------------------------------------------------------------
// Session state (per session: avoids collisions with subagents)
// ---------------------------------------------------------------------------

interface CwlState {
  graph: EpisodeGraph;
  lastEvictionTurn: number;
  totalEvictions: number;
  totalEvictedTokens: number;
  /** Indice del prossimo messaggio: derivato dal conteggio REALE dei messaggi. */
  messageCursor: number;
  /** Numero di messaggi seen nell'ultimo context hook (per calcolare il delta). */
  lastSeenMessages: number;
  /** Token stimati al prossimo check. */
  lastMeasuredTokens: number;
}

function newState(): CwlState {
  return {
    graph: new EpisodeGraph(),
    lastEvictionTurn: -1,
    totalEvictions: 0,
    totalEvictedTokens: 0,
    messageCursor: 0,
    lastSeenMessages: 0,
    lastMeasuredTokens: 0,
  };
}

/** Chiave di sessione: cwd + path sessione se disponibile, altrimenti "default". */
function sessionKey(ctx: ExtensionContext | null | undefined): string {
  try {
    const cwd = typeof ctx?.cwd === 'string' ? ctx.cwd : '';
    const sm = (ctx as unknown as { sessionManager?: { getSessionId?: () => string } })?.sessionManager;
    const sid = typeof sm?.getSessionId === 'function' ? sm.getSessionId() : '';
    return `${cwd}::${sid}`;
  } catch {
    return 'default';
  }
}

const states = new Map<string, CwlState>();
const configs = new Map<string, CwlConfig>();

function getState(key: string): CwlState {
  let st = states.get(key);
  if (!st) { st = newState(); states.set(key, st); }
  return st;
}

function getConfig(key: string): CwlConfig {
  let cf = configs.get(key);
  if (!cf) { cf = loadConfig(); configs.set(key, cf); }
  return cf;
}

function dropState(key: string) {
  states.delete(key);
  configs.delete(key);
}

// ---------------------------------------------------------------------------
// Stripping (forma reale: array di blocchi content)
// ---------------------------------------------------------------------------

/** Soglia oltre la quale un singolo blocco testuale viene ridotto. */
const STRIP_BLOCK_CHARS = 2000;

/**
 * Riduce un messaggio in base al livello di stripping richiesto.
 *
 * Gestisce la forma reale di pi-ai:
 *   - ToolResultMessage: content: (Text|Image)[]
 *   - AssistantMessage : content: (Text|Thinking|ToolCall)[]
 *   - UserMessage      : content: string | (Text|Image)[]
 *
 * Strategia:
 *   - reasoning  -> rimuove i blocchi { type: "thinking" }
 *   - bulk/intermediate -> tronca i blocchi testuali lunghi, preserva ToolCall
 *   - immagini e toolCall NON vengono mai toccati
 */
function stripToolResult(msg: AgentMessage, level: StripLevel): AgentMessage {
  const m = msg as unknown as RealMessage & Record<string, unknown>;

  // Caso content stringa (UserMessage)
  if (typeof m.content === 'string') {
    if (level === 'reasoning') return msg; // niente reasoning in una stringa
    if (m.content.length > STRIP_BLOCK_CHARS) {
      return {
        ...m,
        content: t('truncatedString')(m.content.length, level),
      } as unknown as AgentMessage;
    }
    return msg;
  }

  if (!Array.isArray(m.content)) return msg;

  const blocks = m.content as RealContentBlock[];
  let changed = false;

  const reduced = blocks.map((part) => {
    if (!part || typeof part !== 'object') return part;

    // reasoning: elimina i blocchi thinking
    if (level === 'reasoning' && part.type === 'thinking') {
      changed = true;
      return { type: 'text', text: t('strippedReasoning') } satisfies RealContentBlock;
    }

    // toolCall: intoccabile, semantica d'azione
    if (part.type === 'toolCall') return part;

    // immagini: intoccabili
    if (part.type === 'image') return part;

    // testo lungo: tronca
    if (part.type === 'text' && typeof part.text === 'string' && part.text.length > STRIP_BLOCK_CHARS) {
      changed = true;
      const origLen = part.text.length;
      return {
        ...part,
        text: t('truncatedArray')(origLen, level),
      } satisfies RealContentBlock;
    }

    return part;
  });

  if (!changed) return msg;
  return { ...m, content: reduced } as unknown as AgentMessage;
}

// ---------------------------------------------------------------------------
// Eviction policy (deterministic, LLM-free)
// ---------------------------------------------------------------------------

const LEVEL_ORDER: StripLevel[] = ['none', 'reasoning', 'bulk', 'intermediate', 'removed'];

/**
 * Mappa il livello di stripping alla chiave corrispondente in cfg.levels.
 * Without this map, cfg.levels['bulk'] was undefined and the
 * bulk/intermediate venivano SEMPRE scartati.
 */
const LEVEL_CONFIG_KEY: Record<StripLevel, keyof CwlConfig['levels'] | null> = {
  none: null,
  reasoning: 'stripReasoning',
  bulk: 'stripBulkOutput',
  intermediate: 'stripIntermediate',
  removed: 'removeEpisode',
};

/** Risparmio stimato come frazione dei token dell'episodio, per livello. */
const LEVEL_SAVINGS: Record<StripLevel, number> = {
  none: 0,
  reasoning: 0.2,
  bulk: 0.5,
  intermediate: 0.35,
  removed: 0.7,
};

function levelEnabled(cfg: CwlConfig, level: StripLevel): boolean {
  const key = LEVEL_CONFIG_KEY[level];
  if (!key) return false;
  return cfg.levels[key] === true;
}

/**
 * Esegue un passaggio di eviction: cammina il grafo, seleziona il target
 * most recoverable target, and computes the required stripping level.
 *
 * Returns an array of actions (empty if the budget is already satisfied).
 */
function runEvictionPass(
  cfg: CwlConfig,
  graph: EpisodeGraph,
  currentTokens: number,
  targetTokens: number,
): {
  episode: string;
  level: StripLevel;
  instruction: string;
  estimatedTokens: number;
}[] {
  const actions: {
    episode: string;
    level: StripLevel;
    instruction: string;
    estimatedTokens: number;
  }[] = [];

  // Candidates: closed episodes, not active, without live dependents
  const candidates = graph.recoverable().filter(ep => {
    if (graph.hasLiveDependents(ep.name)) return false;
    return true;
  });

  if (candidates.length === 0) return [];

  // Priority: act before expl (persistent effects), then by age
  const actCandidates = candidates.filter(e => e.type === 'act');
  const explCandidates = candidates.filter(e => e.type === 'expl');

  const ordered = [
    ...actCandidates.sort((a, b) => a.openedAt - b.openedAt),
    ...explCandidates.sort((a, b) => a.openedAt - b.openedAt),
  ];

  let projected = currentTokens;

  for (const target of ordered) {
    if (projected <= targetTokens) break;

    // Trova il livello minimo che ci riporta sotto budget
    for (const level of ['reasoning', 'bulk', 'intermediate', 'removed'] as StripLevel[]) {
      if (level === 'reasoning' && target.type !== 'expl') continue;

      const action = computeStripActionWith(cfg, target, level);
      if (action) {
        target.level = level;
        // Estimate the episode tokens to compute the real saving
        const epTokens = estimateEpisodeTokens(target, projected);
        const saved = Math.floor(epTokens * LEVEL_SAVINGS[level]);
        projected = Math.max(0, projected - saved);
        actions.push({ ...action, estimatedTokens: epTokens });
        break;
      }
    }
  }

  return actions;
}

function computeStripActionWith(cfg: CwlConfig, ep: Episode, level: StripLevel) {
  if (LEVEL_ORDER.indexOf(level) <= LEVEL_ORDER.indexOf(ep.level)) return null;
  if (!levelEnabled(cfg, level)) return null;
  switch (level) {
    case 'reasoning':
      if (ep.type !== 'expl') return null;
      return { episode: ep.name, level, instruction: `Rimuovi i blocchi thinking dall'episodio "${ep.name}". Mantieni tool call e risultati.` };
    case 'bulk':
      return { episode: ep.name, level, instruction: `Rimuovi gli output di grandi tool enumerabili dall'episodio "${ep.name}". Mantieni le azioni.` };
    case 'intermediate':
      return { episode: ep.name, level, instruction: `Rimuovi i risultati intermedi dall'episodio "${ep.name}". Mantieni solo le conclusioni.` };
    case 'removed':
      return { episode: ep.name, level, instruction: ep.type === 'expl'
        ? `Rimuovi completamente l'episodio "${ep.name}". Sostituisci con: "${ep.description || '(nessuna descrizione)'}"`
        : `Rimuovi completamente l'episodio azione "${ep.name}".` };
    default:
      return null;
  }
}

/** Stima i token occupati da un episodio, pro-rata sul totale corrente. */
function estimateEpisodeTokens(ep: Episode, currentTokens: number): number {
  const span = Math.max(1, (ep.endIdx ?? currentTokens) - ep.startIdx);
  return Math.max(1, Math.floor(currentTokens * (span / Math.max(1, currentTokens))));
}

// ---------------------------------------------------------------------------
// Extension entry
// ---------------------------------------------------------------------------

/**
 * Fallback senza episodi: riduce i blocchi di reasoning dei messaggi assistant.
 * E' il recupero piu' sicuro: i blocchi thinking sono traccia interna del
 * modello — non risultati, non azioni, non input dell'utente — e ARC non li
 * tocca. Ritorna null se non c'e' nulla da togliere.
 */
function globalReasoningStrip(
  messages: AgentMessage[],
): { kept: AgentMessage[]; changed: number } | null {
  const kept: AgentMessage[] = [];
  let changed = 0;
  for (const msg of messages) {
    const m = msg as unknown as RealMessage;
    if (m.role === 'assistant' && Array.isArray(m.content)) {
      const hasThinking = (m.content as RealContentBlock[]).some(
        (b) => b && b.type === 'thinking',
      );
      if (hasThinking) {
        changed++;
        kept.push(stripToolResult(msg, 'reasoning'));
        continue;
      }
    }
    kept.push(msg);
  }
  return changed > 0 ? { kept, changed } : null;
}

export default function (pi: ExtensionAPI) {
  // ---- Tools -------------------------------------------------------------

  pi.registerTool({
    name: 'delimiter',
    label: 'CWL Delimiter',
    description:
      'Segna i confini di un episodio CWL. Tipi: "expl" (esplorazione: ricerca, ' +
      'letture, orientamento — il contenuto non serve dopo l\'inferenza) e "act" ' +
      '(azione: scritture, edit, esecuzioni — effetti persistenti, primo candidato ' +
      'all\'eviction). Quando apri un "act", dichiara le esplorazioni da cui dipende. ' +
      'Quando chiudi un "expl", fornisci la descrizione di cosa hai imparato: ' +
      'e\' l\'unico contenuto che sopravvive all\'eviction.',
    promptSnippet: 'delimiter: segna i confini di un episodio CWL (expl/act)',
    promptGuidelines: I18N[LANG].guidelines,
    parameters: Type.Object({
      action: Type.Union([Type.Literal('start'), Type.Literal('end')], {
        description: '"start" per aprire un episodio, "end" per chiuderlo.',
      }),
      name: Type.Optional(Type.String({
        description: 'Nome univoco dell\'episodio (obbligatorio per action=start).',
      })),
      type: Type.Optional(Type.Union([Type.Literal('expl'), Type.Literal('act')], {
        description: 'Tipo episodio: "expl" (esplorazione) o "act" (azione). Obbligatorio per action=start.',
      })),
      dependencies: Type.Optional(Type.Array(Type.String(), {
        description: 'Nomi degli episodi expl da cui questo atto dipende. Obbligatorio per action=start con type="act".',
      })),
      description: Type.Optional(Type.String({
        description: 'Descrizione di cosa hai imparato. Obbligatorio solo per action=end con type="expl".',
      })),
    }),
    async execute(toolCallId, params, _signal, _onUpdate, ctx) {
      const key = sessionKey(ctx);
      const st = getState(key);
      const cf = getConfig(key);

      if (params.action === 'start') {
        if (!params.name || !params.type) {
          return {
            content: [{ type: 'text', text: t('startNeedsNameType') }],
            details: { ok: false, error: 'parametri-mancanti' },
          };
        }
        if (st.graph.has(params.name)) {
          return {
            content: [{ type: 'text', text: t('duplicateName')(params.name) }],
            details: { ok: false, error: 'nome-duplicato' },
          };
        }
        const deps = params.dependencies ?? [];
        const invalidDeps = deps.filter(d => {
          const ep = st.graph.all.find(e => e.name === d);
          return !ep || ep.type !== 'expl' || ep.endIdx === null;
        });
        if (invalidDeps.length > 0) {
          return {
            content: [{ type: 'text', text: t('invalidDeps')(invalidDeps.join(', ')) }],
            details: { ok: false, error: 'dipendenze-non-valide' },
          };
        }

        const ep = st.graph.open(params.name, params.type, deps, st.messageCursor, toolCallId);
        debugLog(cf, `OPEN ${ep.type} "${ep.name}" deps=[${deps.join(',')}] startIdx=${ep.startIdx}`);

        return {
          content: [{ type: 'text', text: t('episodeOpened')(ep.type, ep.name) }],
          details: { ok: true, episode: ep.name, type: ep.type },
        };
      }

      if (params.action === 'end') {
        if (!params.name) {
          return {
            content: [{ type: 'text', text: t('endNeedsName') }],
            details: { ok: false, error: 'parametri-mancanti' },
          };
        }
        const ep = st.graph.close(params.name, params.description ?? '', st.messageCursor, toolCallId);
        if (!ep) {
          return {
            content: [{ type: 'text', text: t('notFoundOrClosed')(params.name) }],
            details: { ok: false, error: 'episodio-non-trovato' },
          };
        }
        debugLog(cf, `CLOSE ${ep.type} "${ep.name}" level=${ep.level} endIdx=${ep.endIdx}`);

        let msg = `Episodio ${ep.type} "${ep.name}" chiuso.`;
        if (ep.type === 'expl' && !ep.description) {
          msg += t('emptyDescriptionWarning');
        }
        return {
          content: [{ type: 'text', text: msg }],
          details: { ok: true, episode: ep.name, type: ep.type },
        };
      }

      return {
        content: [{ type: 'text', text: t('unknownAction') }],
        details: { ok: false },
      };
    },
  });

  pi.registerTool({
    name: 'cwl_status',
    label: 'CWL Status',
    description: 'Mostra lo stato del context lifecycle: budget, episodi attivi/chiusi, eviction eseguite.',
    promptSnippet: 'cwl_status: stato del context lifecycle CWL',
    parameters: Type.Object({}),
    async execute(_toolCallId, _params, _signal, _onUpdate, ctx) {
      const key = sessionKey(ctx);
      const st = getState(key);
      const cf = getConfig(key);
      const g = st.graph;
      const active = g.active();
      const closed = g.recoverable();
      const stripped = closed.filter(e => e.level !== 'none');

      const lines = [
        t('statusHeader')(cf.tokenBudget.toLocaleString(), (cf.thresholdRatio * 100).toFixed(0)),
        `Token contesto misurati: ~${st.lastMeasuredTokens.toLocaleString()}`,
        t('statusEpisodes')(g.count, active.length, closed.length, stripped.length),
        t('statusEvictions')(st.totalEvictions, st.totalEvictedTokens.toLocaleString()),
      ];
      if (active.length > 0) {
        lines.push(`Attivi: ${active.map(e => `${e.name}(${e.type})`).join(', ')}`);
      }
      if (stripped.length > 0) {
        lines.push(`Stripped: ${stripped.map(e => `${e.name}[${e.level}]`).join(', ')}`);
      }

      return {
        content: [{ type: 'text', text: lines.join('\n') }],
        details: {
          ok: true,
          budget: cf.tokenBudget,
          total: g.count,
          active: active.length,
          closed: closed.length,
          stripped: stripped.length,
          evictions: st.totalEvictions,
        },
      };
    },
  });

  // ---- Hooks -------------------------------------------------------------

  pi.on('session_start', async (_event, ctx) => {
    const key = sessionKey(ctx);
    states.set(key, newState());
    configs.set(key, loadConfig());
    debugLog(getConfig(key), 'SESSION START — graph reset');
  });

  /**
   * L'eviction vive nell'hook 'context': e' l'unico punto in cui Pi permette di
   * RISCRIVERE la lista messaggi che verra' inviata al provider. turn_end puo'
   * only observe, so computing there would never have reduced anything.
   */
  pi.on('context', async (event, ctx) => {
    const key = sessionKey(ctx);
    const st = getState(key);
    const cf = getConfig(key);

    const messages: AgentMessage[] = event.messages;
    if (!messages || messages.length === 0) return;

    // Update the cursor with the REAL message count (not turns).
    // I tool risultanti vengono aggiunti durante il turno, quindi il conteggio
    // the message count is the only reliable basis for episode indices.
    st.messageCursor = messages.length; // ricalcolato, non accumulato

    const g = st.graph;

    // 1. Misura il contesto reale
    let currentTokens = 0;
    for (const m of messages) {
      currentTokens += estimateMessageTokens(m);
    }
    st.lastMeasuredTokens = currentTokens;

    const trigger = cf.tokenBudget * cf.thresholdRatio;
    if (currentTokens <= trigger) {
      debugLog(cf, `CONTEXT ${currentTokens}t sotto soglia ${Math.round(trigger)}t: nessuna eviction`);
      return;
    }

    // 1b. DEGRADAZIONE UTILE: senza episodi l'eviction graduata non ha bersagli.
    // Prima di lasciare il contesto intatto si recupera comunque il budget piu'
    // sicuro: i blocchi di reasoning dei messaggi assistant. Sono traccia interna
    // del modello, non risultati ne azioni, e ARC non li tocca. Senza questo un
    // agente che non usa `delimiter` non ottiene compressione per quanto alto sia
    // il contesto — ed e' proprio il caso in cui serve di piu'.
    if (g.isEmpty) {
      const out = globalReasoningStrip(messages);
      if (out) {
        const after = out.kept.reduce((sum: number, m: AgentMessage) => sum + estimateMessageTokens(m), 0);
        st.totalEvictions++;
        st.totalEvictedTokens += Math.max(0, currentTokens - after);
        debugLog(cf, `FALLBACK reasoning-strip: ${out.changed} messaggi, ${currentTokens}t -> ${after}t`);
        if (ctx?.hasUI) {
          ctx.ui.notify(
            `CWL: nessun episodio annotato, ridotti i blocchi di reasoning in ${out.changed} messaggi ` +
            `(${currentTokens} -> ${after} token). Usa \`delimiter\` per un'eviction graduata.`,
            'info',
          );
        }
        return { messages: out.kept };
      }
      debugLog(cf, 'CONTEXT sopra soglia ma nessun episodio E nessun reasoning strip possibile');
      return;
    }

    // 2. Policy deterministica: calcola cosa evictare e a che livello
    const actions = runEvictionPass(cf, g, currentTokens, trigger);
    if (actions.length === 0) {
      debugLog(cf, `CONTEXT ${currentTokens}t sopra soglia ma nessun candidato sicuro: contesto intatto`);
      return;
    }

    // 3. APPLICA davvero: ricostruisci la lista messaggi evictando i segmenti
    const byLevel = new Map<StripLevel, Set<string>>();
    for (const a of actions) {
      let s = byLevel.get(a.level);
      if (!s) { s = new Set(); byLevel.set(a.level, s); }
      s.add(a.episode);
    }
    const evictFull = byLevel.get('removed') ?? new Set<string>();
    const stripBulk = new Set<string>([
      ...(byLevel.get('bulk') ?? []),
      ...(byLevel.get('intermediate') ?? []),
    ]);
    const stripReasoning = byLevel.get('reasoning') ?? new Set<string>();

    // Map message index -> episode. Uses g.closed() (all closed
    // episodes, including level='removed'): with recoverable() the already
    // episodes, including removed ones; with recoverable() they would NEVER
    // be mapped and the full eviction would never fire.
    // Ancoraggio per toolCallId, non per indice: i risultati `delimiter` sono
    // messaggi reali nel transcript, quindi la loro posizione e' sempre esatta.
    // L'indice del cursore non lo e' (cfr. startIdx) e rendeva l'eviction
    // impossibile quando il contesto si accorciava.
    const posByToolCallId = new Map<string, number>();
    messages.forEach((m, i) => {
      const id = (m as unknown as RealMessage).toolCallId;
      if (typeof id === 'string') posByToolCallId.set(id, i);
    });

    const episodeAt = new Map<number, Episode>();
    for (const ep of g.closed()) {
      const from = posByToolCallId.get(ep.startToolCallId);
      const to = ep.endToolCallId !== null ? posByToolCallId.get(ep.endToolCallId) : undefined;
      if (from === undefined) continue; // ancoraggio perso: meglio non evictare
      const end = to !== undefined ? to : messages.length;
      if (end <= from) continue;
      for (let i = from; i <= Math.min(end, messages.length - 1); i++) episodeAt.set(i, ep);
    }

    const kept: AgentMessage[] = [];
    let dropped = 0;
    let truncated = 0;
    let removedTokens = 0;
    let truncatedTokens = 0;

    messages.forEach((msg, idx) => {
      const ep = episodeAt.get(idx);
      const role = (msg as unknown as RealMessage).role;

      // Principio 3: user turns are inviolable. Ma non basta `user`: `system`
      // e `developer` contengono le istruzioni del ruolo e i tool disponibili.
      // Evictarli lascia l'agente senza schema di azione.
      if (role === 'user' || role === 'system' || role === 'developer') {
        kept.push(msg);
        return;
      }

      if (!ep) {
        kept.push(msg);
        return;
      }

      if (evictFull.has(ep.name)) {
        removedTokens += estimateMessageTokens(msg);
        // Per expl con descrizione, conserva il marker di rientro.
        if (ep.type === 'expl' && ep.description) {
          kept.push({
            role: 'custom',
            customType: 'cwl-evicted',
            content: t('evictedEpisode')(ep.name, ep.description),
            display: false,
            timestamp: Date.now(),
          } as unknown as AgentMessage);
        }
        dropped++;
        return;
      }

      if (stripReasoning.has(ep.name)) {
        const before = textLengthOf(msg);
        const out = stripToolResult(msg, 'reasoning');
        // Conta solo se lo strip ha DAVVERO ridotto: altrimenti la notifica
        // dichiarerebbe un risparmio avvenuto, e l'operatore crederebbe che
        // l'eviction lavori mentre non muove nulla.
        if (out !== msg) {
          truncatedTokens += Math.max(0, Math.floor((before - textLengthOf(out)) / 4));
          truncated++;
        }
        kept.push(out);
        return;
      }

      if (stripBulk.has(ep.name)) {
        const before = textLengthOf(msg);
        const out = stripToolResult(msg, 'bulk');
        // Conta solo se lo strip ha DAVVERO ridotto: altrimenti la notifica
        // dichiarerebbe un risparmio avvenuto, e l'operatore crederebbe che
        // l'eviction lavori mentre non muove nulla.
        if (out !== msg) {
          truncatedTokens += Math.max(0, Math.floor((before - textLengthOf(out)) / 4));
          truncated++;
        }
        kept.push(out);
        return;
      }

      kept.push(msg);
    });

    if (dropped === 0 && truncated === 0) {
      debugLog(cf, 'EVICTION: nessun messaggio effettivamente riducibile, contesto lasciato intatto');
      return;
    }

    st.totalEvictions++;
    st.lastEvictionTurn = st.messageCursor;
    // Sum ONLY the tokens actually saved, not the whole context.
    st.totalEvictedTokens += removedTokens + truncatedTokens;

    const afterTokens = kept.reduce((s, m) => s + estimateMessageTokens(m), 0);
    debugLog(cf, `EVICTION applicata: ${dropped} msg rimossi, ${truncated} ridotti, ${currentTokens}t -> ${afterTokens}t (risparmio ${removedTokens + truncatedTokens}t)`);

    if (ctx?.hasUI) {
      ctx.ui.notify(
        `CWL: ${dropped} evictati, ${truncated} ridotti (${currentTokens.toLocaleString()} -> ${afterTokens.toLocaleString()} token).`,
        'info',
      );
    }

    return { messages: kept };
  });

  pi.on('turn_end', async (_event, ctx) => {
    // Osservatore: il cursore dei messaggi viene aggiornato nel context hook,
    // che e' l'unico punto con accesso all'array reale dei messaggi.
    sessionKey(ctx);
  });

  pi.on('session_shutdown', async (_event, ctx) => {
    const key = sessionKey(ctx);
    const st = getState(key);
    debugLog(getConfig(key), `SESSION SHUTDOWN — total evictions: ${st.totalEvictions}`);
    dropState(key);
  });
}
