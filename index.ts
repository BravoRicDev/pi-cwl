/**
 * CWL — Context Window Lifecycle
 * Structured context eviction for long-horizon Pi agents.
 *
 * PROBLEM
 *   Threshold compaction (summarization) blocks the turn, loses information
 *   in unpredictable ways, destroys causal structure and introduces hallucinations
 *   exactly when the context budget is under pressure.
 *
 * SOLUTION (https://arxiv.org/html/2606.11213)
 *   The agent annotates its own trajectory as typed episodes (expl/act)
 *   through a `delimiter` tool. A deterministic, LLM-free policy evicts
 *   the content in order of recoverability once the budget is exceeded.
 *
 *   - expl (exploration): search output, listings, orientation reads.
 *     On closing, the agent supplies a description: the only content kept.
 *   - act (action): writes, edits, tool calls. The effects are persistent in
 *     the environment, so they are the first candidates for eviction.
 *
 *   Declared dependencies prevent losing the exploratory context that
 *   produced a decision that is still active.
 *
 *   User content is inviolable (Principle 3).
 *   Compression NEVER invokes the model (Principle 5): zero cost, zero
 *   introduced hallucination, zero blocking.
 *
 * REAL MESSAGE SHAPE (from @earendil-works/pi-ai)
 *   UserMessage      : { role: "user",      content: string | Content[] }
 *   AssistantMessage : { role: "assistant", content: (Text|Thinking|ToolCall)[] }
 *   ToolResultMessage: { role: "toolResult",content: (Text|Image)[] }
 *   Note: the tool result role is "toolResult", NOT "tool".
 */

import type { ExtensionAPI, ExtensionContext } from '@earendil-works/pi-coding-agent';
import type { AgentMessage } from '@earendil-works/pi-agent-core';
import { Type } from 'typebox';
import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';
import { fileURLToPath } from 'node:url';
import { createHash, randomUUID } from 'node:crypto';
import type * as Recall from './recall.mjs';

// ---------------------------------------------------------------------------
// Minimal types for the real message shape (subset of @earendil-works/pi-ai)
// ---------------------------------------------------------------------------

interface RealContentBlock {
  type?: string;
  text?: string;
  thinking?: string;
  data?: string;
  name?: string;
  arguments?: unknown;
}

interface RealMessage {
  role?: string;
  content?: unknown;
  toolCallId?: string;
  toolName?: string;
  timestamp?: number;
  /** Set by extensions that inject their own messages (role: 'custom'). */
  customType?: string;
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
  /** Message injected instead of a compressed span (goes into the LLM context). */
  compressedNotice: (from: string, to: string, saved: number, id: string) => string;
  episodeClosed: (type: string, name: string) => string;
  statusMeasured: (tokens: string) => string;
  statusActive: (list: string) => string;
  statusStripped: (list: string) => string;
  /** UI notice when falling back to the global reasoning strip. */
  fallbackNotice: (changed: number, from: number, to: number) => string;
  /** UI notice after an eviction pass. */
  evictionNotice: (dropped: number, truncated: number, from: string, to: string) => string;
  /** Strings of the cwl_compress tool. */
  compressMissingParams: string;
  compressRevoked: (start: string, end: string) => string;
  compressUnknownHash: (from: boolean, to: boolean) => string;
  compressApplied: (start: string, end: string) => string;
  /** cwl_compress_range: the extension picks the range, the model writes the text. */
  compressRangeApplied: (start: string, end: string, tokens: number) => string;
  compressRangeNothing: string;
  compressRangeNoSummary: string;
  /** Status line for the currently compressible range. */
  statusRange: (tokens: number, start: string, end: string) => string;
  statusAddresses: (eligible: number, withId: number) => string;
  statusSpans: (n: number) => string;
  statusUnlocatable: (n: number) => string;
  /** Strings of the cwl_recall tool. */
  recallNotLoaded: string;
  recallNoTranscript: string;
  recallUnreadable: string;
  recallNoMatch: (query: string) => string;
  recallFound: (hits: number, indexed: number, query: string, body: string) => string;
  /** Strings of the cwl_recall_episode tool. */
  episodeRecallNotKnown: (name: string) => string;
  episodeRecallStillOpen: (name: string) => string;
  episodeRecallAnchorLost: (name: string) => string;
  episodeRecallEmpty: (name: string) => string;
  episodeRecallTruncatedHint: string;
  episodeRecallFound: (name: string, tokens: number, body: string) => string;
  /** cwl_open: the WHOLE body of a compressed span, by id. */
  openFound: (id: string, tokens: number, when: string) => string;
  openMissing: (id: string) => string;
  /** cwl_open on a node of level 1: the micros of its leaves, each with its own id. */
  nodePage: (id: string, count: number, tokens: number, body: string) => string;
  /** Il nodo vecchio: la sintesi che sostituisce i micro, e cio' che ha accorpato. */
  oldSet: (id: string, nodes: number, leaves: number, microChars: number, tokens: number) => string;
  oldNotDue: (young: number, need: number) => string;
  oldHead: (id: string, nodes: number, tokens: number) => string;
  /** La pagina del nodo vecchio: il riassuntone e la forma di cio' che tiene dentro. */
  oldPage: (id: string, nodes: number, tokens: number, body: string) => string;
  /** L'intestazione della sezione "foglie consultate" dentro la pagina del pozzo. */
  oldHot: (listed: number, total: number) => string;
  /** La richiesta di scrivere il riassuntone: e' l'unico grilletto che l'agente vede. */
  indexDue: (young: number, need: number) => string;
  /** Le foglie che aspettano un micro: la richiesta che fa partire l'indice. */
  leavesDue: (missing: number, ids: string) => string;
  /** Una foglia potato: il riassunto e' perso, l'originale torna dal transcript. */
  openOriginal: (id: string, tokens: number, body: string) => string;
  openOriginalLost: (id: string) => string;
  /** cwl_micro: the body leaves the context, the micro takes its place. */
  microSet: (id: string, microChars: number, bodyChars: number, shorter: boolean) => string;
  /** Budget gate: the agent-driving channel. */
  gateDemand: (current: string, budget: string, turns: number, canClose: boolean, canCompress: boolean) => string;
  gateGiveUp: (attempts: number) => string;
  /** Texts that end up in the LLM context. */
  snippets: { delimiter: string; status: string; compress: string; recall: string; recallEpisode: string; compressRange: string };
  /** Texts of the autonomous tools. */
  tools: {
    compressDesc: string; compressStart: string; compressEnd: string; compressSummary: string;
    recallDesc: string; recallQuery: string; recallLimit: string;
    recallEpisodeDesc: string; recallEpisodeName: string; recallEpisodeFull: string;
    openDesc: string;
    openId: string;
    microDesc: string;
    microId: string;
    microText: string;
    oldDesc: string;
    oldText: string;
    compressRangeDesc: string; compressRangeSummary: string; compressMicro: string;
  };
  /** Parameter descriptions: read by the LLM on every invocation. */
  params: {
    delimiterDesc: string; action: string; name: string; type: string;
    dependencies: string; description: string; statusDesc: string;
  };
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
    evictedEpisode: (name, desc) => `[CWL episode "${name}" evicted — recover the original with cwl_recall_episode("${name}")]` +
      (desc ? ` Notes kept: ${desc}` : ' No notes were kept.'),
    startNeedsNameType: 'action=start requires name and type.',
    endNeedsName: 'action=end requires name.',
    duplicateName: (name) => `Episode "${name}" already exists. Use a unique name.`,
    notFoundOrClosed: (name) => `Episode "${name}" not found or already closed.`,
    invalidDeps: (names) => `Invalid dependencies (must be closed expl episodes): ${names}`,
    episodeOpened: (type, name) => `Episode ${type} "${name}" opened.`,
    unknownAction: 'Unrecognised action.',
    emptyDescriptionWarning: ' WARNING: empty description — this episode has no fallback content.',
    statusHeader: (budget, threshold) => `CWL — token budget: ${budget} (threshold: ${threshold}%)`,
    statusEpisodes: (total, active, closed, stripped) => `Episodes total: ${total} | active: ${active} | with evictable content: ${closed} | already stripped: ${stripped}`,
    statusEvictions: (count, tokens) => `Evictions total: ${count} | tokens saved: ${tokens}`,
    compressedNotice: (from, to, saved, id) => `[CWL · RECALL] The messages from ${from} to ${to} were compressed into ` +
      `this summary (~${saved} tokens saved). Leaf id: ${id}.\n` +
      `To read the WHOLE summary again, call cwl_open with that id. ` +
      `If you need the original text, call cwl_recall with keywords from that content.\n\n`,
    episodeClosed: (type, name) => `Episode ${type} "${name}" closed.`,
    statusMeasured: (tokens) => `Measured context tokens: ~${tokens}`,
    statusActive: (list) => `Active: ${list}`,
    statusStripped: (list) => `Stripped: ${list}`,
    fallbackNotice: (changed, from, to) => `CWL: no episode annotated, reasoning blocks reduced in ${changed} messages ` +
      `(${from} -> ${to} tokens). Use \`delimiter\` for graded eviction.`,
    evictionNotice: (dropped, truncated, from, to) => `CWL: ${dropped} evicted, ${truncated} reduced (${from} -> ${to} tokens).`,
    compressMissingParams: 'startHash, endHash and summary are all required.',
    compressRevoked: (start, end) => `Span ${start}..${end} restored to full text.`,
    compressUnknownHash: (from, to) => `Unknown hash. startHash found: ${from}, endHash found: ${to}. Copy them verbatim from the context.`,
    compressApplied: (start, end) => `Compressed ${start}..${end} into your summary. The original stays on disk: recover it with cwl_recall.`,
    compressRangeApplied: (start, end, tokens) => `Compressed ${start}..${end} (~${tokens} tokens) into your summary. The original stays in the transcript: recover it with cwl_recall_episode or cwl_recall.`,
    compressRangeNothing: 'Nothing left to compress: everything remaining is either inside the protected window or already compressed.',
    compressRangeNoSummary: 'The summary is required: it is the ONLY part of this call you have to write yourself.',
    statusRange: (tokens, start, end) => `Compressible range: ~${tokens} tokens (${start}..${end})`,
    statusAddresses: (eligible, withId) => `Addresses: ${withId}/${eligible} endpoint messages carry a stable id`,
    statusSpans: (n) => `Compressed spans held: ${n}`,
    statusUnlocatable: (n) => `Episodes whose anchors left the context: ${n} (their content is not verifiable)`,
    recallNotLoaded: 'The recall index is not loaded; /reload the extension.',
    recallNoTranscript: 'Transcript not found for this session.',
    recallUnreadable: 'Transcript unreadable.',
    recallNoMatch: (query) => `No match for "${query}".`,
    recallFound: (hits, indexed, query, body) => `Found ${hits} of ${indexed} indexed messages for "${query}":\n\n${body}`,
    episodeRecallNotKnown: (name) => `No episode named "${name}" in this session.`,
    episodeRecallStillOpen: (name) => `Episode "${name}" is still open: its content is in the active context.`,
    episodeRecallAnchorLost: (name) => `Episode "${name}" has no anchor in the transcript: the original cannot be recovered.`,
    episodeRecallEmpty: (name) => `Episode "${name}" resolved to no readable text.`,
    episodeRecallTruncatedHint: '\n\n… (truncated; pass full=true for the whole episode)',
    episodeRecallFound: (name, tokens, body) => `Original content of episode "${name}" (~${tokens} tokens):\n\n${body}`,
    openFound: (id, tokens, when) => `[CWL leaf ${id} — compressed ${when}, ~${tokens} tokens. The WHOLE body follows; nothing is truncated.]\n\n`,
    openMissing: (id) => `No leaf "${id}" in this session's state: either it never existed, or the state dropped it. A pruned span's SUMMARY is not recoverable — it lives only in the state — while the ORIGINAL messages are still in the append-only transcript: recover those with cwl_recall.`,
    nodePage: (id, count, tokens, body) => `[CWL node ${id} — ${count} leaf/leaves, ~${tokens} tokens. Each micro below points to a leaf: cwl_open("<leaf id>") returns its WHOLE body.]\n\n${body}`,
    oldSet: (id, nodes, leaves, microChars, tokens) => `Old node ${id}: merged ${nodes} node(s), ${leaves} leaf/leaves (${microChars} chars of micros) into a synthesis of ~${tokens} tokens. They stay readable: cwl_open("${id}") lists the nodes inside.`,
    oldNotDue: (young, need) => `No merge: ${young} young node(s), the merge starts at ${need}. Nothing was recorded.`,
    oldHead: (id, nodes, tokens) => `[CWL OLD NODE ${id} — ${nodes} older node(s) merged behind this synthesis (~${tokens} tokens). Their micros left the context; cwl_open("${id}") pages through them, leaf by leaf.]\n\n`,
    oldPage: (id, nodes, tokens, body) => `[CWL old node ${id} — ${nodes} node(s) inside, ~${tokens} tokens. The synthesis first, then one line per node with its SHAPE; cwl_open("<node id>") opens one, and its leaves open in full.]\n\n${body}`,
    oldHot: (listed, total) => `--- Most consulted leaves (${listed} of ${total} in the old node; nothing was deleted, this is only the reading order) ---`,
    indexDue: (young, need) => `[CWL INDEX] ${young} node(s) of the index are due to merge (a merge starts at ${need}). Call cwl_old with the riassuntone: your synthesis replaces the micros of the oldest nodes, and their leaves stay readable with cwl_open.`,
    leavesDue: (missing, ids) => `[CWL INDEX] ${missing} leaf/leaves still have their BODY in the context and wait for a micro: ${ids} — call cwl_micro with the id and a micro of ~200 words. The body leaves the context, the micro stands for it, and only then can the leaf enter a node.`,
    openOriginal: (id, tokens, body) => `[CWL leaf ${id} — its SUMMARY was dropped when the state pruned it, so here is the ORIGINAL from the append-only transcript (~${tokens} tokens, in full).]\n\n${body}`,
    openOriginalLost: (id) => `Leaf "${id}" was dropped from the state, and its anchors found NOTHING in the transcript. The original cannot be recovered by id from here: use cwl_recall with keywords from that content.`,
    microSet: (id, microChars, bodyChars, shorter) => microChars === 0
      ? `Leaf ${id}: micro removed — the WHOLE body is back in the context.`
      : `Leaf ${id}: a micro of ${microChars} chars now stands in the context in place of ${bodyChars} chars. ` +
        (shorter
          ? `The body is untouched — cwl_open("${id}") returns all of it.`
          : `WARNING: this micro is NOT shorter than the body it replaces, so the context does not shrink. Shorten it, or the absorption costs more than it saves.`),
    gateDemand: (current, budget, turns, canClose, canCompress) => {
      // Only the options that are ACTUALLY available. Measured in a real session:
      // the demand listed both while all four episodes were closed and no
      // compressible range existed, so it asked for the impossible every turn,
      // burning the very context it was trying to save.
      const opts: string[] = [];
      if (canClose) {
        opts.push('  1. close the episodes you no longer need: ' +
          'delimiter(action="end", name="<episode>", description="<what you learned>")');
      }
      if (canCompress) {
        opts.push(`  ${opts.length + 1}. compress a range you have already worked through: ` +
          'cwl_compress_range(summary="<whole pieces, not a digest>", micro="<~200-word label>") ' +
          '\u2014 you do NOT need any hash, the extension already picked the range; the micro is the label the index ' +
          'will show for this leaf, so write it now and the leaf is ready for a node');
      }
      return `[CWL \u00b7 CONTEXT OVER BUDGET] The active context is ~${current} tokens against a budget of ${budget}, ` +
        `and the deterministic eviction has nothing left to take. Do ONE of these NOW, in this turn:\n` +
        opts.join('\n') +
        `\nThe last ${turns} turns are protected and will NOT be touched: compact something older.`;
    },
    gateGiveUp: (attempts) => `CWL: the compaction demand went unanswered for ${attempts} turns; dropping it for a cooldown.`,
    snippets: {
      delimiter: 'delimiter: marks the boundaries of a CWL episode (expl/act)',
      status: 'cwl_status: CWL context lifecycle status',
      compress: 'cwl_compress: compress a range of messages in the active context',
      recall: 'cwl_recall: retrieve a conversation excerpt by BM25 query',
      recallEpisode: 'cwl_recall_episode: recover the original text of an evicted episode by name',
      compressRange: 'cwl_compress_range: compress the oldest usable range into your summary (no hashes needed)',
    },
    tools: {
      compressDesc: 'Replaces, in the active context, the messages between two hashes with your summary, leaving the original intact in the transcript. Use it only when the extension asks you to.',
      compressStart: 'Hash of the FIRST message to compress.',
      compressEnd: 'Hash of the LAST message to compress.',
      compressSummary: 'The summary that REPLACES the compressed messages. Write WHOLE PIECES, not digests: it must be enough to keep working without re-reading them. Include paths, file names, function names, numeric values, and what you decided and why.',
      recallDesc: 'Searches the session messages (user and assistant) and returns the pieces most relevant to a BM25 query. Use it to bring compressed text back into view.',
      recallQuery: 'Keywords to search for: paths, function names, technical terms.',
      recallLimit: 'How many results to return (default 5, minimum 1, maximum 50).',
      recallEpisodeDesc: 'Recovers the ORIGINAL text of an episode that was evicted from the context, by its name. The episode content still lives in the append-only transcript: eviction moves it out of the CONTEXT, it does not delete it. Use it when a marker says an episode was evicted and you need its detail.',
      recallEpisodeName: 'Name of the evicted episode, as it appears in the eviction marker.',
      openDesc: 'Reads back a compressed span (a "leaf") by id, IN FULL. Use cwl_status to see the ids. Nothing is truncated: the body comes back whole, and its size is declared first.',
      openId: 'Id of the leaf to open, as it appears in the compression notice.',
      microDesc: 'Absorbs a leaf: a short micro-summary stands in the context in place of the whole body, which stays readable IN FULL with cwl_open. This is how the extension stops paying twice for the same story.',
      microId: 'Id of the leaf to absorb, as it appears in the compression notice.',
      microText: 'The micro-summary that REPLACES the body in the context. The body is NOT touched: cwl_open still returns all of it. Write the pieces that matter; ~200 words is the size this design is built for.',
      oldDesc: 'Writes the RIASSUNTONE and merges the oldest young nodes into the old node: their micros leave the context and the synthesis stands for all of them. The nodes and their leaves stay readable — cwl_open pages through them.',
      oldText: 'The synthesis (riassuntone) that replaces the micros of the merged nodes in the context. Write WHOLE PIECES: it is what the agent will see instead of them.',
      recallEpisodeFull: 'false (default) returns a truncated preview; true returns the whole episode.',
      compressRangeDesc: 'Compresses the OLDEST usable range of the conversation into your summary. YOU DO NOT pick the range and you do not need any hash: the extension already computed the address and holds it. Call it when an eviction marker or the budget demand tells you to compact, and write a summary good enough to keep working without re-reading the originals. Nothing inside the protected window is touched.',
      compressRangeSummary: 'The summary that REPLACES the compressed range. Write WHOLE PIECES, not a digest: it must be enough to keep working without re-reading them. Include paths, file names, function names, numeric values, and what you decided and why.',
      compressMicro: 'Your ~200-word LABEL for this leaf: what it contains, detailed enough that the index can show it instead of the body. Write it HERE, while you have the messages in front of you — the leaf is then ready for a node and nothing will have to ask you for it later. A compression without a label stays valid: the extension will ask for it when the leaf is due to join a node.',
    },
    params: {
      delimiterDesc: 'Marks the boundaries of a CWL episode. Types: "expl" (exploration: searches, reads, orientation — the content is not needed after the inference) and "act" (action: writes, edits, executions — persistent effects, first candidate for eviction). When you open an "act", declare the explorations it depends on. When you close an "expl", give the description of what you learned: it is the only content that survives eviction.',
      action: '"start" to open an episode, "end" to close it.',
      name: 'Unique episode name (required for action=start).',
      type: 'Episode type: "expl" (exploration) or "act" (action). Required for action=start.',
      dependencies: 'Names of the expl episodes this action depends on. Required for action=start with type="act".',
      description: 'Marks the boundaries of a CWL episode. Types: "expl" (exploration: searches, reads, orientation — the content is not needed after the inference) and "act" (action: writes, edits, executions — persistent effects, first candidate for eviction). When you open an "act", declare the explorations it depends on. When you close an "expl", give the description of what you learned: it is the only content that survives eviction.',
      statusDesc: 'Shows the context lifecycle status: budget, active/closed episodes, evictions performed.',
    },
  },
  it: {
    guidelines: [
      'Apri un episodio expl quando inizi a esplorare (letture, ricerche, listing). Chiudilo appena hai la risposta che ti serve.',
      'Apri un episodio act quando fai una modifica (scrivi file, esegui comandi con effetti). Dichiara le esplorazioni da cui dipende.',
      "Alla chiusura di un expl, fornisci una descrizione concisa di cosa hai imparato: e' il contenuto che sopravvive all'eviction.",
      "Non aprire episodi troppo fini: 5-15 per sessione e' un buon target.",
    ],
    strippedReasoning: '[CWL: reasoning evictato]',
    truncatedString: (len, level) => `[CWL: contenuto ridotto da ${len} chars (livello ${level})]`,
    truncatedArray: (len, level) => `[CWL: testo ridotto da ${len} chars a marker (livello ${level}) — usa arc_recall o riapri l'episodio se serve]`,
    evictedEpisode: (name, desc) => `[CWL episodio "${name}" evictato — recupera l'originale con cwl_recall_episode("${name}")]` +
      (desc ? ` Appunti conservati: ${desc}` : ' Nessun appunto conservato.'),
    startNeedsNameType: 'action=start richiede name e type.',
    endNeedsName: 'action=end richiede name.',
    duplicateName: (name) => `Episodio "${name}" gia' esiste. Usa un nome univoco.`,
    notFoundOrClosed: (name) => `Episodio "${name}" non trovato o gia' chiuso.`,
    invalidDeps: (names) => `Dipendenze non valide (devono essere episodi expl chiusi): ${names}`,
    episodeOpened: (type, name) => `Episodio ${type} "${name}" aperto.`,
    unknownAction: 'Azione non riconosciuta.',
    emptyDescriptionWarning: ' ATTENZIONE: descrizione vuota — questo episodio non ha contenuto di fallback.',
    statusHeader: (budget, threshold) => `CWL — token budget: ${budget} (threshold: ${threshold}%)`,
    statusEpisodes: (total, active, closed, stripped) => `Episodi totali: ${total} | attivi: ${active} | con contenuto evictabile: ${closed} | gia' stripped: ${stripped}`,
    statusEvictions: (count, tokens) => `Eviction totali: ${count} | token risparmiati: ${tokens}`,
    compressedNotice: (from, to, saved, id) => `[CWL · RICHIAMO] I messaggi da ${from} a ${to} sono stati compressi in ` +
      `questo riepilogo (~${saved} token risparmiati). Id della foglia: ${id}.\n` +
      `Per rileggere il riepilogo INTERO, chiama cwl_open con quell'id. ` +
      `Se ti serve il testo originale, chiama cwl_recall con parole chiave di quel contenuto.\n\n`,
    episodeClosed: (type, name) => `Episodio ${type} "${name}" chiuso.`,
    statusMeasured: (tokens) => `Token contesto misurati: ~${tokens}`,
    statusActive: (list) => `Attivi: ${list}`,
    statusStripped: (list) => `Stripped: ${list}`,
    fallbackNotice: (changed, from, to) => `CWL: nessun episodio annotato, ridotti i blocchi di reasoning in ${changed} messaggi ` +
      `(${from} -> ${to} token). Usa \`delimiter\` per un'eviction graduata.`,
    evictionNotice: (dropped, truncated, from, to) => `CWL: ${dropped} evictati, ${truncated} ridotti (${from} -> ${to} token).`,
    compressMissingParams: 'startHash, endHash e summary sono tutti obbligatori.',
    compressRevoked: (start, end) => `Span ${start}..${end} ripristinato al testo integrale.`,
    compressUnknownHash: (from, to) => `Hash sconosciuto. startHash trovato: ${from}, endHash trovato: ${to}. Copiali verbatim dal contesto.`,
    compressApplied: (start, end) => `Compresso ${start}..${end} nel tuo riepilogo. L'originale resta su disco: recuperalo con cwl_recall.`,
    compressRangeApplied: (start, end, tokens) => `Compresso ${start}..${end} (~${tokens} token) nel tuo riepilogo. L'originale resta nel transcript: recuperalo con cwl_recall_episode o cwl_recall.`,
    compressRangeNothing: "Non resta niente da comprimere: cio' che rimane e' dentro la finestra protetta oppure gia' compresso.",
    compressRangeNoSummary: "Il riassunto e' obbligatorio: e' l'UNICA parte di questa chiamata che devi scrivere tu.",
    statusRange: (tokens, start, end) => `Intervallo comprimibile: ~${tokens} token (${start}..${end})`,
    statusAddresses: (eligible, withId) => `Indirizzi: ${withId}/${eligible} messaggi-endpoint con un id stabile`,
    statusSpans: (n) => `Span di compressione tenuti: ${n}`,
    statusUnlocatable: (n) => `Episodi le cui ancore sono uscite dal contesto: ${n} (contenuto non verificabile)`,
    recallNotLoaded: "L'indice di recall non e' caricato; fai /reload dell'estensione.",
    recallNoTranscript: 'Transcript non trovato per questa sessione.',
    recallUnreadable: 'Transcript illeggibile.',
    recallNoMatch: (query) => `Nessuna corrispondenza per "${query}".`,
    recallFound: (hits, indexed, query, body) => `Trovati ${hits} di ${indexed} messaggi indicizzati per "${query}":\n\n${body}`,
    episodeRecallNotKnown: (name) => `Nessun episodio chiamato "${name}" in questa sessione.`,
    episodeRecallStillOpen: (name) => `L'episodio "${name}" e' ancora aperto: il suo contenuto e' nel contesto attivo.`,
    episodeRecallAnchorLost: (name) => `L'episodio "${name}" non ha ancora un'ancora nel transcript: l'originale non e' recuperabile.`,
    episodeRecallEmpty: (name) => `L'episodio "${name}" non ha prodotto testo leggibile.`,
    episodeRecallTruncatedHint: '\n\n… (troncato; passa full=true per l\'episodio intero)',
    episodeRecallFound: (name, tokens, body) => `Contenuto originale dell'episodio "${name}" (~${tokens} token):\n\n${body}`,
    openFound: (id, tokens, when) => `[CWL foglia ${id} — compressa ${when}, ~${tokens} token. Segue il corpo INTERO; niente e' troncato.]\n\n`,
    openMissing: (id) => `Nessuna foglia "${id}" nello stato di questa sessione: o non e' mai esistita, oppure lo stato l'ha potato. Il RIASSUNTO di uno span potato non e' recuperabile — vive solo nello stato — mentre i messaggi ORIGINALI sono ancora nel transcript append-only: recuperali con cwl_recall.`,
    nodePage: (id, count, tokens, body) => `[CWL nodo ${id} — ${count} foglia/e, ~${tokens} token. Ogni micro qui sotto punta a una foglia: cwl_open("<id foglia>") ne restituisce il corpo INTERO.]\n\n${body}`,
    oldSet: (id, nodes, leaves, microChars, tokens) => `Nodo vecchio ${id}: accorpati ${nodes} nodo/i, ${leaves} foglia/e (${microChars} caratteri di micro) in una sintesi di ~${tokens} token. Restano leggibili: cwl_open("${id}") elenca i nodi dentro.`,
    oldNotDue: (young, need) => `Nessun accorpamento: ${young} nodo/i giovane/i, si accorpa da ${need} in su. Non e' stato registrato niente.`,
    oldHead: (id, nodes, tokens) => `[CWL NODO VECCHIO ${id} — ${nodes} nodo/i piu' vecchi accorpati dietro questa sintesi (~${tokens} token). I loro micro sono usciti dal contesto; cwl_open("${id}") li pagina, foglia per foglia.]\n\n`,
    oldPage: (id, nodes, tokens, body) => `[CWL nodo vecchio ${id} — ${nodes} nodo/i dentro, ~${tokens} token. Prima la sintesi, poi una riga per nodo con la sua FORMA; cwl_open("<id nodo>") ne apre uno, e le sue foglie si aprono intere.]\n\n${body}`,
    oldHot: (listed, total) => `--- Foglie piu' consultate (${listed} di ${total} nel nodo vecchio; niente e' stato cancellato, questo e' solo l'ordine di lettura) ---`,
    indexDue: (young, need) => `[CWL INDICE] ${young} nodo/i dell'indice sono da accorpare (si accorpa da ${need} in su). Chiama cwl_old col riassuntone: la tua sintesi sostituisce i micro dei nodi piu' vecchi, e le loro foglie restano leggibili con cwl_open.`,
    leavesDue: (missing, ids) => `[CWL INDICE] ${missing} foglia/e hanno ancora il CORPO nel contesto e aspettano un micro: ${ids} — chiama cwl_micro con l'id e un micro di ~200 parole. Il corpo esce dal contesto, il micro lo rappresenta, e solo allora la foglia puo' entrare in un nodo.`,
    openOriginal: (id, tokens, body) => `[CWL foglia ${id} — il RIASSUNTO e' andato perso quando lo stato l'ha potato, quindi ecco l'ORIGINALE dal transcript append-only (~${tokens} token, per intero).]\n\n${body}`,
    openOriginalLost: (id) => `La foglia "${id}" era stata potato dallo stato, e le sue ancore nel transcript non hanno trovato niente. Da qui l'originale non e' piu' recuperabile per id: usa cwl_recall con parole chiave di quel contenuto.`,
    microSet: (id, microChars, bodyChars, shorter) => microChars === 0
      ? `Foglia ${id}: micro rimosso — nel contesto e' tornato il corpo INTERO.`
      : `Foglia ${id}: un micro di ${microChars} caratteri sta ora nel contesto al posto di ${bodyChars}. ` +
        (shorter
          ? `Il corpo non e' stato toccato — cwl_open("${id}") lo restituisce tutto.`
          : 'ATTENZIONE: questo micro NON e\' piu\' corto del corpo che sostituisce, quindi il contesto non si riduce. Accorcialo, o l\'assorbimento costa piu\' di quanto risparmia.'),
    gateDemand: (current, budget, turns, canClose, canCompress) => {
      const opts: string[] = [];
      if (canClose) {
        opts.push('  1. chiudi gli episodi che non ti servono piu\': ' +
          'delimiter(action="end", name="<episodio>", description="<cosa hai imparato>")');
      }
      if (canCompress) {
        opts.push(`  ${opts.length + 1}. comprimi un intervallo che hai gia' consumato: ` +
          'cwl_compress_range(summary="<pezzi interi, non un sommario>", micro="<etichetta di ~200 parole>") ' +
          '\u2014 non ti serve nessun hash, l\'intervallo l\'ha gia\' scelto l\'estensione; il micro e\' l\'etichetta che ' +
          'l\'indice mostrera\' per questa foglia, scrivilo adesso e la foglia e\' pronta per un nodo');
      }
      return `[CWL \u00b7 CONTESTO OLTRE IL BUDGET] Il contesto attivo e' ~${current} token contro un budget di ${budget}, ` +
        `e l'eviction deterministica non ha piu' niente da prendere. Fai UNA di queste cose ORA, in questo turno:\n` +
        opts.join('\n') +
        `\nGli ultimi ${turns} turni sono protetti e NON verranno toccati: compatta qualcosa di piu' vecchio.`;
    },
    gateGiveUp: (attempts) => `CWL: la richiesta di compattazione e' rimasta senza risposta per ${attempts} turni; la tolgo per un cooldown.`,
    snippets: {
      delimiter: 'delimiter: segna i confini di un episodio CWL (expl/act)',
      status: 'cwl_status: stato del context lifecycle CWL',
      compress: 'cwl_compress: comprimi un intervallo di messaggi nel contesto attivo',
      recall: 'cwl_recall: recupera un pezzo di conversazione per query BM25',
      recallEpisode: 'cwl_recall_episode: recupera il testo originale di un episodio evictato, per nome',
      compressRange: 'cwl_compress_range: comprimi nel tuo riassunto l\'intervallo piu\' vecchio utilizzabile (senza hash)',
    },
    tools: {
      compressDesc: "Sostituisce nel contesto attivo i messaggi fra due hash con il tuo riassunto, lasciando l'originale intatto nel transcript. Usalo solo se l'estensione te lo chiede.",
      compressStart: 'Hash del PRIMO messaggio da comprimere.',
      compressEnd: "Hash dell'ULTIMO messaggio da comprimere.",
      compressSummary: "Il riassunto che SOSTITUISCE i messaggi compressi. Scrivi PEZZI INTERI, non sommari: deve bastare a lavorare senza rileggere. Includi path, nomi di file, nomi di funzione, valori numerici e cosa hai scelto e perche'.",
      recallDesc: "Cerca nei messaggi della sessione (user e assistant) e restituisce i pezzi piu' pertinenti per una query BM25. Usalo per riportare alla luce testo compresso.",
      recallQuery: 'Le parole chiave da cercare: path, nomi di funzione, termini tecnici.',
      recallLimit: 'Quanti risultati restituire (default 5, minimo 1, massimo 50).',
      recallEpisodeDesc: "Recupera il testo ORIGINALE di un episodio evictato dal contesto, per nome. Il contenuto dell'episodio vive ancora nel transcript append-only: l'eviction lo sposta fuori dal CONTESTO, non lo cancella. Usalo quando un marker ti dice che un episodio e' stato evictato e te ne serve il dettaglio.",
      recallEpisodeName: "Nome dell'episodio evictato, come appare nel marker di eviction.",
      openDesc: 'Rilegge per intero uno span compresso (una "foglia") dato il suo id. Gli id si vedono in cwl_status. Niente viene troncato: il corpo torna intero, e la sua dimensione viene dichiarata prima.',
      openId: "Id della foglia da aprire, come appare nell'avviso di compressione.",
      microDesc: "Assorbe una foglia: un micro-riassunto sta nel contesto al posto del corpo intero, che resta leggibile PER INTERO con cwl_open. E' cosi' che l'estensione smette di pagare due volte la stessa storia.",
      microId: "Id della foglia da assorbire, come appare nell'avviso di compressione.",
      microText: 'Il micro-riassunto che SOSTITUISCE il corpo nel contesto. Il corpo NON viene toccato: cwl_open lo restituisce ancora tutto. Scrivi i pezzi che contano; ~200 parole e\' la misura per cui questo design e\' costruito.',
      oldDesc: 'Scrive il RIASSUNTONE e accorpa i nodi giovani piu\' vecchi nel nodo vecchio: i loro micro escono dal contesto e la sintesi sta per tutti. I nodi e le loro foglie restano leggibili — cwl_open li pagina.',
      oldText: 'La sintesi (riassuntone) che sostituisce i micro dei nodi accorpati nel contesto. Scrivi PEZZI INTERI: e\' quello che l\'agente vedra\' al posto loro.',
      recallEpisodeFull: 'false (default) restituisce un estratto troncato; true restituisce l\'episodio intero.',
      compressRangeDesc: "Comprime nel tuo riassunto l'intervallo PIU' VECCHIO utilizzabile della conversazione. NON scegli tu l'intervallo e non ti serve nessun hash: l'estensione ha gia' calcolato e tiene l'indirizzo. Chiamalo quando un marker di eviction o la richiesta di budget ti dicono di compattare, e scrivi un riassunto che basti a lavorare senza rileggere gli originali. Nulla dentro la finestra protetta viene toccato.",
      compressRangeSummary: "Il riassunto che SOSTITUISCE l'intervallo compresso. Scrivi PEZZI INTERI, non un sommario: deve bastare a lavorare senza rileggere. Includi path, nomi di file, nomi di funzione, valori numerici e cosa hai scelto e perche'.",
      compressMicro: "La tua ETICHETTA di ~200 parole per questa foglia: cosa contiene, con dettaglio sufficiente perche' l'indice la possa mostrare al posto del corpo. Scrivila QUI, mentre hai i messaggi davanti — la foglia e' cosi' pronta per un nodo e nessuno dovra' chiedertela dopo. Una compressione senza etichetta resta valida: l'estensione te la chiedera' quando la foglia dovra' entrare in un nodo.",
    },
    params: {
      delimiterDesc: 'Segna i confini di un episodio CWL. Tipi: "expl" (esplorazione: ricerca, letture, orientamento — il contenuto non serve dopo l\'inferenza) e "act" (azione: scritture, edit, esecuzioni — effetti persistenti, primo candidato all\'eviction). Quando apri un "act", dichiara le esplorazioni da cui dipende. Quando chiudi un "expl", fornisci la descrizione di cosa hai imparato: e\' l\'unico contenuto che sopravvive all\'eviction.',
      action: '"start" per aprire un episodio, "end" per chiuderlo.',
      name: 'Nome univoco dell\'episodio (obbligatorio per action=start).',
      type: 'Tipo episodio: "expl" (esplorazione) o "act" (azione). Obbligatorio per action=start.',
      dependencies: 'Nomi degli episodi expl da cui questo atto dipende. Obbligatorio per action=start con type="act".',
      description: 'Descrizione di cosa hai imparato. Obbligatorio solo per action=end con type="expl".',
      statusDesc: 'Mostra lo stato del context lifecycle: budget, episodi attivi/chiusi, eviction eseguite.',
    },
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
  /** Active token budget above which eviction starts. */
  tokenBudget: number;
  /** Activation threshold as a fraction of the budget (0..1). */
  thresholdRatio: number;
  /**
   * User turns at the tail that are NEVER evicted or stripped.
   *
   * A safety window: compaction must not destroy the context the agent is
   * working on. A turn begins at a user message — the context hook receives
   * messages, not turn numbers, so counting user messages backwards is the only
   * definition available here.
   */
  protectedTurns: number;
  /**
   * The SHAPE OF THE INDEX — the manopole of the plan.
   *
   * `looseLeaves` leaves keep their whole body in the context: that is the working
   * set, and the reason the most recent work is never summarised away. A node of
   * level 1 holds `nodeCapacity` leaves, and at `mergeNodesAt` young nodes the two
   * oldest are due to merge into the old node. Preferences, like `protectedTurns`
   * — and the reason a test can drive the whole tree with three leaves.
   */
  looseLeaves: number;
  nodeCapacity: number;
  mergeNodesAt: number;
  /**
   * Ask the AGENT to compact when the budget is not coming down on its own.
   *
   * The deterministic eviction is blind: it cuts by age, not by meaning. This is
   * the only channel that can get a semantic decision out of the model — but it
   * is an instruction, so it is a PREFERENCE, never a guarantee: `protectedTurns`
   * and the safety net are what actually bound the context.
   */
  gate: boolean;
  /** Enabled aggressiveness levels. */
  levels: {
    stripReasoning: boolean;
    stripBulkOutput: boolean;
    stripIntermediate: boolean;
    removeEpisode: boolean;
  };
  /** Renders the UI widget with the status. */
  showWidget: boolean;
  /** Debug logging to file. */
  debug: boolean;
}

const DEFAULT_CONFIG: CwlConfig = {
  // 80k = ~30% of a 256k context; the CWL paper puts this at the regime under which
  // attention does not degrade. If your context is 1M, raise this value.
  tokenBudget: 80_000,
  thresholdRatio: 0.85,
  // Four turns, not ten. The window is a CONTINUOUS RUN of the list: the last N
  // user turns PLUS everything between and after them — every assistant message,
  // reasoning block, tool call and tool output. An operator who writes little
  // ends up protecting mostly the agent's own text, and MEASURED on a real
  // session ten turns left only ~6.307t compressible out of ~184.000t: the safety
  // window WAS the budget, and the extension could not say so. Four keeps the
  // thread the agent is on and lets compaction bite.
  protectedTurns: 4,
  /**
   * The SHAPE OF THE INDEX — the manopole of the plan.
   *
   * `looseLeaves`: how many leaves stay loose, i.e. keep their whole body in the
   * context. That is the working set, and it is also the safety property: the most
   * recent work is never summarised away. `nodeCapacity`: how many leaves a node of
   * level 1 holds before a new one starts. `mergeNodesAt`: how many young nodes
   * make the two oldest due to merge into the old node.
   *
   * They are config, not constants, for two reasons: the shape is a PREFERENCE of
   * the operator (like `protectedTurns`), and a test can then drive the whole tree
   * with three leaves instead of ninety.
   */
  looseLeaves: 5,
  nodeCapacity: 30,
  mergeNodesAt: 3,
  gate: true,
  levels: {
    stripReasoning: true,
    stripBulkOutput: true,
    stripIntermediate: true,
    removeEpisode: true,
  },
  showWidget: true,
  debug: false,
};

// __dirname equivalent in ESM.
const _EXT_DIR = path.dirname(fileURLToPath(import.meta.url));
// User config; if absent we fall back to the config bundled with the extension, so the
// file shipped with the repo actually does something instead of being a dead document.
const CONFIG_PATH = path.join(os.homedir(), '.pi', 'cwl', 'config.json');
const BUNDLED_CONFIG_PATH = path.join(_EXT_DIR, 'config.json');
const LOG_PATH = path.join(os.homedir(), '.pi', 'cwl', 'cwl.log');
/** See debugLog: the debug log rotates once instead of growing without bound. */
const MAX_LOG_BYTES = 2_000_000;
// Traced compaction and episode graph of each session, resumed on restart.
const STATE_DIR = path.join(os.homedir(), '.pi', 'cwl', 'state');

/**
 * Deep-merge: the levels must be merged individually, otherwise a partial
 * config.json (e.g. { levels: { stripBulkOutput: false } }) would replace
 * the whole levels block and lose the other three flags.
 */
/**
 * Rejects a value that is not usable as the field it is bound to.
 *
 * Without this the merge below accepts ANY type, and the eviction policy breaks
 * silently: `thresholdRatio: "high"` made `tokenBudget * thresholdRatio` NaN, so
 * `currentTokens <= NaN` was permanently false and the extension evicted on
 * EVERY turn. A user typo, not a crash, and nothing told the user.
 */
function validNumber(v: unknown, min: number, max: number, fallback: number): number {
  if (typeof v !== 'number' || !Number.isFinite(v)) return fallback;
  if (v < min || v > max) return fallback;
  return v;
}

function validBool(v: unknown, fallback: boolean): boolean {
  return typeof v === 'boolean' ? v : fallback;
}

function loadConfig(): CwlConfig {
  for (const candidate of [CONFIG_PATH, BUNDLED_CONFIG_PATH]) {
    try {
      const raw = fs.readFileSync(candidate, 'utf8');
      const rawParsed = JSON.parse(raw);
      const user = (rawParsed ?? {}) as Partial<CwlConfig> & { levels?: Record<string, unknown> };
      const levels = (user.levels ?? {}) as Partial<CwlConfig['levels']>;
      return {
        // Both scalars are range-checked: tokenBudget is a token count, so it
        // must stay positive; thresholdRatio is a fraction, so 0 < r <= 1.
        // Out-of-range input falls back to the default instead of silently
        // disabling (r too big) or inverting (r garbage) the whole policy.
        tokenBudget: validNumber(user.tokenBudget, 1, Number.MAX_SAFE_INTEGER, DEFAULT_CONFIG.tokenBudget),
        thresholdRatio: validNumber(user.thresholdRatio, 0, 1, DEFAULT_CONFIG.thresholdRatio),
        protectedTurns: validNumber(user.protectedTurns, 0, 10_000, DEFAULT_CONFIG.protectedTurns),
        looseLeaves: validNumber(user.looseLeaves, 0, 10_000, DEFAULT_CONFIG.looseLeaves),
        nodeCapacity: validNumber(user.nodeCapacity, 1, 10_000, DEFAULT_CONFIG.nodeCapacity),
        // At least 2: a merge of one node into itself is not a merge.
        mergeNodesAt: validNumber(user.mergeNodesAt, 2, 10_000, DEFAULT_CONFIG.mergeNodesAt),
        gate: validBool(user.gate, DEFAULT_CONFIG.gate),
        levels: {
          stripReasoning: validBool(levels.stripReasoning, DEFAULT_CONFIG.levels.stripReasoning),
          stripBulkOutput: validBool(levels.stripBulkOutput, DEFAULT_CONFIG.levels.stripBulkOutput),
          stripIntermediate: validBool(levels.stripIntermediate, DEFAULT_CONFIG.levels.stripIntermediate),
          removeEpisode: validBool(levels.removeEpisode, DEFAULT_CONFIG.levels.removeEpisode),
        },
        showWidget: validBool(user.showWidget, DEFAULT_CONFIG.showWidget),
        debug: validBool(user.debug, DEFAULT_CONFIG.debug),
      };
    } catch {
      // try the next candidate
    }
  }
  return { ...DEFAULT_CONFIG, levels: { ...DEFAULT_CONFIG.levels } };
}

function debugLog(cfg: CwlConfig, msg: string) {
  if (!cfg.debug) return;
  try {
    fs.mkdirSync(path.dirname(LOG_PATH), { recursive: true });
    // `appendFileSync` has no cap of its own, and this log is meant to be LEFT
    // ON while chasing a bug in a long session — which is exactly when it grows
    // fastest. The state files already have a prune (pruneStateFiles); the log
    // had nothing at all. One generation is kept, so the file is bounded at 2x
    // MAX_LOG_BYTES and the previous run is still readable.
    try {
      if (fs.statSync(LOG_PATH).size > MAX_LOG_BYTES) fs.renameSync(LOG_PATH, `${LOG_PATH}.1`);
    } catch { /* no log yet, or not readable: nothing to rotate */ }
    fs.appendFileSync(LOG_PATH, `[${new Date().toISOString()}] ${msg}\n`);
  } catch { /* not critical */ }
}

// ---------------------------------------------------------------------------
// Token estimation (approximate, no tokenizer call)
// ---------------------------------------------------------------------------

/** Token estimate: ~4 characters per token for English prose; it overestimates for code. */
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
// Content helpers (real shape: array of blocks)
// ---------------------------------------------------------------------------

/** Extracts the concatenated text from a string or an array of content blocks. */
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

/** Number of text characters actually present in the message. */
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
  /** Names of the expl episodes this act depends on. */
  dependencies: string[];
  /**
   * toolCallId of the `delimiter` result that opened the episode: that is the
   * RELIABLE anchor. The index is not, because the cursor only updates
   * in the context hook (before the LLM call) while the tool runs within the turn.
   */
  startToolCallId: string;
  /** Same as startToolCallId, for the close. */
  endToolCallId: string | null;
  /** Diagnostic index, not used for the mapping. */
  startIdx: number;
  /** Diagnostic index, not used for the mapping. */
  endIdx: number | null;
  /** Description supplied by the agent on close (expl only). */
  description: string;
  /** Stripping level currently applied. */
  level: StripLevel;
  /** Opening timestamp. */
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

  /** Closed episodes whose content still exists in the context. */
  recoverable(): Episode[] {
    return this.episodes.filter(e => e.endIdx !== null && e.level !== 'removed');
  }

  /** Closed episodes at ANY level: needed to map the indices. */
  closed(): Episode[] {
    return this.episodes.filter(e => e.endIdx !== null);
  }

  /** true if an episode with this name already exists (prevents cycles). */
  has(name: string) { return this.nameSet.has(name); }

  /** Dependents not yet fully evicted. */
  hasLiveDependents(name: string): boolean {
    return this.episodes.some(ep =>
      ep.name !== name &&
      ep.dependencies.includes(name) &&
      ep.level !== 'removed'
    );
  }

  /** Full reset (new session). */
  reset() {
    this.episodes = [];
    this.activeNames.clear();
    this.nameSet.clear();
  }

  /** Serialises for persistence. */
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
  /** Next message index: derived from the REAL message count. */
  messageCursor: number;
  /** Number of messages seen in the last context hook (to compute the delta). */
  lastSeenMessages: number;
  /** Tokens estimated at the next check. */
  lastMeasuredTokens: number;
  /**
   * Diagnostic, taken every turn: how many addressable messages carry a stable
   * id. See stableIdOf — with 0 here, two identical texts share one address.
   */
  addrEligible: number;
  addrWithId: number;
  /**
   * Closed episodes the eviction can no longer locate (their delimiters left the
   * list). Reported, not assumed: see the hook.
   */
  unlocatable: number;
  unlocatableSeen: number;
  /**
   * Closed episodes located by DEDUCTION: their start anchor was carried away by
   * a native compaction, so the range was derived instead of read. Same reason
   * as above — measured, reported, never assumed.
   */
  deduced: number;
  deducedSeen: number;
  /** Spans compressed by the LLM: hash of the first/last message + summary. */
  spans: CompressedSpan[];
  /** Nodes of level 1: containers of leaves, chronological. */
  nodes: SpanNode[];
  /**
   * The graveyard: the anchors of the spans the state dropped.
   *
   * Capped: a leaf nobody asks for, for long enough, stops being recoverable by id — and
   * that has to be DICHIARATO (see `cwl_open`), because a silent limit is the promise
   * broken quietly.
   */
  graves: Grave[];
  /**
   * The OLD node — the pit — once the agent has written its riassuntone.
   *
   * It holds young nodes instead of leaves, so `cwl_open` can still page through
   * them one by one; what its leaves lose is the CONTEXT: their micros are no
   * longer injected individually, because the riassuntone stands for all of them.
   */
  oldNode: OldNode | null;
  /** Text hashes of the messages seen: anchor for cwl_compress. */
  knownHashes: Set<string>;
  /** BM25 index of the transcript, built lazily on the first search. */
  recallIndex: Recall.Bm25Index | null;
  /** Turn counter, incremented in turn_end. */
  turns: number;
  /** Turn the context first went over budget without relief; -1 when under. */
  overBudgetSince: number;
  /** Turn the gate was armed, -1 when inactive. */
  gateArmedTurn: number;
  /** Failed attempts of the active gate. */
  gateAttempts: number;
  /** Turn of the last failed gate, used for the cooldown. */
  lastGateViolationTurn: number;
  /**
   * Endpoints of the largest range `cwl_compress_range` may compress right now,
   * recomputed on every context hook. Deliberately NOT persisted: they describe
   * the CURRENT message list, and a stale pair reloaded in another process would
   * point at addresses that no longer exist.
   */
  rangeStartHash: string | null;
  rangeEndHash: string | null;
  /** Tokens held by that range: shown in the status and in the demand. */
  rangeTokens: number;
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
    addrEligible: 0,
    addrWithId: 0,
    unlocatable: 0,
    unlocatableSeen: -1,
    deduced: 0,
    deducedSeen: -1,
    spans: [],
    nodes: [],
    graves: [],
    oldNode: null,
    knownHashes: new Set(),
    recallIndex: null,
    turns: 0,
    overBudgetSince: -1,
    gateArmedTurn: -1,
    gateAttempts: 0,
    lastGateViolationTurn: -1,
    rangeStartHash: null,
    rangeEndHash: null,
    rangeTokens: 0,
  };
}

/**
 * Identity of the session, used to keep one state per session.
 *
 * The transcript path comes first: it is unique per session AND stable across a
 * restart, which is what makes the persisted state resumable. The old key was
 * `${cwd}::${sid}` and, when the session id was missing, collapsed to
 * `::` or the literal `default`: every session without an id shared ONE state,
 * and a `session_start` in the same cwd wiped the previous graph.
 */
function sessionKey(ctx: ExtensionContext | null | undefined): string {
  try {
    // SAFETY: sessionManager is declared on ExtensionContext, but a degraded
    // context may omit it, so the probe stays optional.
    const sm = ctx?.sessionManager as
      | { getSessionFile?: () => string | undefined; getSessionId?: () => string }
      | undefined;
    const file = typeof sm?.getSessionFile === 'function' ? sm.getSessionFile() : undefined;
    if (typeof file === 'string' && file) return file;
    const cwd = typeof ctx?.cwd === 'string' ? ctx.cwd : '';
    const sid = typeof sm?.getSessionId === 'function' ? sm.getSessionId() : '';
    if (sid) return `${cwd}::${sid}`;
    return anonymousKey(ctx);
  } catch {
    return anonymousKey(ctx);
  }
}

// When a context carries no session id at all, each context object gets its own
// generated key. Two anonymous sessions can then never share state; the old
// fallback ('default') made them collide by construction.
const anonymousKeys = new WeakMap<object, string>();
function anonymousKey(ctx: unknown): string {
  if (ctx !== null && typeof ctx === 'object') {
    const o = ctx as object;
    const existing = anonymousKeys.get(o);
    if (existing) return existing;
    const fresh = `anon::${randomUUID()}`;
    anonymousKeys.set(o, fresh);
    return fresh;
  }
  return `anon::${randomUUID()}`;
}

const states = new Map<string, CwlState>();
const configs = new Map<string, CwlConfig>();

// Recall module, dynamically loaded in session_start: it holds the BM25
// index of the transcript. Null until the load has happened.
let recall: typeof Recall | null = null;

// ---------------------------------------------------------------------------
// Persistence of the session state (episode graph + traced compressions)
// ---------------------------------------------------------------------------

/**
 * On-disk shape of one session's state.
 *
 * What is saved: the episode graph and the compressions requested through
 * `cwl_compress`. Those two are DECISIONS taken by the agent and cannot be
 * recomputed from the transcript, so losing them throws away real work. What is
 * NOT saved: the BM25 index (derived data, rebuilt from the transcript), the node
 * structure of the index — `refreshNodes` DERIVES it from the leaves and their
 * micros on every turn, and a second copy on disk would be a second source of
 * truth — and the token cursors beyond the counters we display.
 */
interface PersistedState {
  version: number;
  key: string;
  savedAt: number;
  graph: { episodes: Episode[] };
  spans: CompressedSpan[];
  /** The graveyard. Optional on load: an older state file simply has none. */
  graves?: Grave[];
  /** The pit. Persisted because it carries the agent's riassuntone, which no rule can recompute. */
  oldNode?: OldNode | null;
  totalEvictions: number;
  totalEvictedTokens: number;
  lastEvictionTurn: number;
  lastMeasuredTokens: number;
  knownHashes: string[];
  /** Gate bookkeeping. Optional on load: an older state file simply has none. */
  turns?: number;
  overBudgetSince?: number;
  gateArmedTurn?: number;
  gateAttempts?: number;
  lastGateViolationTurn?: number;
}

const STATE_VERSION = 1;
/** Cap on persisted hashes: they are an anchor for cwl_compress, not an archive. */
const MAX_PERSISTED_HASHES = 2000;
/** State files older than this are removed at session start. */
const STATE_MAX_AGE_MS = 14 * 24 * 60 * 60 * 1000;

/** One file per session, named by the hash of the key (the key is a path). */
function statePath(key: string): string {
  return path.join(STATE_DIR, `${createHash('sha256').update(key).digest('hex').slice(0, 32)}.json`);
}

function saveState(key: string, st: CwlState): void {
  try {
    fs.mkdirSync(STATE_DIR, { recursive: true });
    const payload: PersistedState = {
      version: STATE_VERSION,
      key,
      savedAt: Date.now(),
      graph: { episodes: st.graph.all },
      spans: st.spans,
      graves: st.graves.slice(-GRAVE_MAX),
      oldNode: st.oldNode,
      totalEvictions: st.totalEvictions,
      totalEvictedTokens: st.totalEvictedTokens,
      lastEvictionTurn: st.lastEvictionTurn,
      lastMeasuredTokens: st.lastMeasuredTokens,
      knownHashes: [...st.knownHashes].slice(-MAX_PERSISTED_HASHES),
      turns: st.turns,
      overBudgetSince: st.overBudgetSince,
      gateArmedTurn: st.gateArmedTurn,
      gateAttempts: st.gateAttempts,
      lastGateViolationTurn: st.lastGateViolationTurn,
    };
    // Atomic write: a crash mid-write must not leave a truncated file that then
    // fails to parse on resume and silently loses the whole state.
    const target = statePath(key);
    const tmp = `${target}.tmp`;
    fs.writeFileSync(tmp, JSON.stringify(payload));
    fs.renameSync(tmp, target);
  } catch { /* persistence is best effort: it must never break a turn */ }
}

function loadPersistedState(key: string): CwlState | null {
  try {
    const raw = readFileOrNull(statePath(key));
    if (raw === null) return null;
    const data = JSON.parse(raw) as Partial<PersistedState>;
    if (data.version !== STATE_VERSION) return null;
    const st = newState();
    st.graph = EpisodeGraph.fromJSON(data.graph);
    st.spans = Array.isArray(data.spans) ? (data.spans as CompressedSpan[]) : [];
    st.graves = Array.isArray(data.graves) ? (data.graves as Grave[]) : [];
    st.oldNode = data.oldNode && typeof data.oldNode.id === 'string' ? data.oldNode : null;
    st.totalEvictions = typeof data.totalEvictions === 'number' ? data.totalEvictions : 0;
    st.totalEvictedTokens = typeof data.totalEvictedTokens === 'number' ? data.totalEvictedTokens : 0;
    st.lastEvictionTurn = typeof data.lastEvictionTurn === 'number' ? data.lastEvictionTurn : -1;
    st.lastMeasuredTokens = typeof data.lastMeasuredTokens === 'number' ? data.lastMeasuredTokens : 0;
    st.turns = typeof data.turns === 'number' ? data.turns : 0;
    st.overBudgetSince = typeof data.overBudgetSince === 'number' ? data.overBudgetSince : -1;
    st.gateArmedTurn = typeof data.gateArmedTurn === 'number' ? data.gateArmedTurn : -1;
    st.gateAttempts = typeof data.gateAttempts === 'number' ? data.gateAttempts : 0;
    st.lastGateViolationTurn = typeof data.lastGateViolationTurn === 'number' ? data.lastGateViolationTurn : -1;
    if (Array.isArray(data.knownHashes)) {
      st.knownHashes = new Set(data.knownHashes.filter((h): h is string => typeof h === 'string'));
    }
    return st;
  } catch {
    return null;
  }
}

/** Removes state files for sessions that have not come back. Best effort. */
function pruneStateFiles(): void {
  try {
    for (const f of fs.readdirSync(STATE_DIR)) {
      if (!f.endsWith('.json')) continue;
      const p = path.join(STATE_DIR, f);
      try {
        if (Date.now() - fs.statSync(p).mtimeMs > STATE_MAX_AGE_MS) fs.rmSync(p, { force: true });
      } catch { /* single file: skip it */ }
    }
  } catch { /* no state dir yet */ }
}

function getState(key: string): CwlState {
  let st = states.get(key);
  if (!st) {
    // Resume a previous run of this session when one is on disk, instead of
    // starting from an empty graph.
    st = loadPersistedState(key) ?? newState();
    states.set(key, st);
  }
  return st;
}

function getConfig(key: string): CwlConfig {
  let cf = configs.get(key);
  if (!cf) { cf = loadConfig(); configs.set(key, cf); }
  return cf;
}

/** Forgets the state in MEMORY. The on-disk copy is written separately. */
function dropState(key: string) {
  states.delete(key);
  configs.delete(key);
}

// ---------------------------------------------------------------------------
// Stripping (real shape: array of content blocks)
// ---------------------------------------------------------------------------

/** Threshold above which a single text block gets reduced. */
const STRIP_BLOCK_CHARS = 2000;

/**
 * Characters of an episode returned by `cwl_recall_episode` when `full` is not
 * set. Same idea as arc_recall's 500-char preview: a recall is a DECISION of the
 * agent, and the default must not be able to blow the context it just freed.
 */
const EPISODE_PREVIEW_CHARS = 4000;

// ---------------------------------------------------------------------------
// Budget gate (the agent-driving channel)
// ---------------------------------------------------------------------------

/** customType of the injected demand, so it can be replaced instead of stacked. */
const GATE_CUSTOM_TYPE = 'cwl-budget-gate';

/** True when the message is the budget demand this extension injected. */
function isGateMessage(m: AgentMessage): boolean {
  // SAFETY: read-only field probe (customType); the AgentMessage union does not declare it.
  return (m as unknown as RealMessage).customType === GATE_CUSTOM_TYPE;
}
/** Turns over budget before the agent is asked to act on its own. */
const GATE_AFTER_TURNS = 2;
/** How many turns the agent gets before the demand is dropped. */
const GATE_MAX_ATTEMPTS = 3;
/** Turns of silence after a failed demand, so it does not nag forever. */
const GATE_COOLDOWN_TURNS = 5;

/**
 * Reduces a message according to the requested stripping level.
 *
 * Handles the real pi-ai shape:
 *   - ToolResultMessage: content: (Text|Image)[]
 *   - AssistantMessage : content: (Text|Thinking|ToolCall)[]
 *   - UserMessage      : content: string | (Text|Image)[]
 *
 * Strategy:
 *   - reasoning  -> removes the { type: "thinking" } blocks
 *   - bulk/intermediate -> truncates long text blocks, preserves ToolCall
 *   - images and toolCall are NEVER touched
 */
function stripToolResult(msg: AgentMessage, level: StripLevel): AgentMessage {
  // SAFETY: AgentMessage is a union over the real pi-ai shapes; RealMessage is the
  // structural probe of it, plus an index signature for the extra fields.
  const m = msg as unknown as RealMessage & Record<string, unknown>;

  // Case: string content (UserMessage)
  if (typeof m.content === 'string') {
    if (level === 'reasoning') return msg; // no reasoning in a string
    if (m.content.length > STRIP_BLOCK_CHARS) {
      // SAFETY: only `content` changes (string -> string); every other field keeps
      // the original message shape.
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

    // reasoning: drop the thinking blocks
    if (level === 'reasoning' && part.type === 'thinking') {
      changed = true;
      return { type: 'text', text: t('strippedReasoning') } satisfies RealContentBlock;
    }

    // toolCall: untouchable, action semantics
    if (part.type === 'toolCall') return part;

    // images: untouchable
    if (part.type === 'image') return part;

    // long text: truncate
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
  // SAFETY: `content` keeps the same block-array shape; only the text/thinking
  // blocks are rewritten, so the result is still a valid message.
  return { ...m, content: reduced } as unknown as AgentMessage;
}

// ---------------------------------------------------------------------------
// Eviction policy (deterministic, LLM-free)
// ---------------------------------------------------------------------------

const LEVEL_ORDER: StripLevel[] = ['none', 'reasoning', 'bulk', 'intermediate', 'removed'];

/**
 * Maps the stripping level to the matching key in cfg.levels.
 * Without this map, cfg.levels['bulk'] was undefined and the
 * bulk/intermediate levels were ALWAYS dropped.
 */
const LEVEL_CONFIG_KEY: Record<StripLevel, keyof CwlConfig['levels'] | null> = {
  none: null,
  reasoning: 'stripReasoning',
  bulk: 'stripBulkOutput',
  intermediate: 'stripIntermediate',
  removed: 'removeEpisode',
};

/**
 * The index range each closed episode REALLY occupies in this list, anchored by
 * toolCallId.
 *
 * The eviction pass deliberately does not trust `ep.startIdx`: it is a cursor
 * into a list that keeps changing, and a stale cursor made eviction impossible
 * whenever the context got shorter. `delimiter` results are real messages, so
 * their position is exact. The PLAN must measure on the same range the eviction
 * will touch, or its numbers describe a stretch of conversation nobody will
 * evict.
 *
 * An anchor that cannot be found means the episode cannot be located, so the
 * eviction skips it (falling back to `messages.length` used to stretch the span
 * to the end of the context and evict messages belonging to no episode at all —
 * including the agent's own memory card). The plan skips it too, instead of
 * inventing a range for it.
 *
 * ONE exception, and it points in one direction only. A native compaction cuts a
 * PREFIX: Pi keeps `firstKeptEntryId` and everything after it, and the summary
 * is prepended. So the only anchor a cut can carry away is the START one, and an
 * episode that was still open when the cut happened loses its start while its
 * end survives. Its surviving content is then everything the list still holds up
 * to that end: `{from: 0, to: end, deduced: true}`.
 *
 * The opposite deduction (`end` lost, `start` alive -> `[start, len-1]`) is
 * REFUSED, and the reason is structural rather than cautious: a prefix cut
 * cannot take the end away while leaving the start, so that layout has no
 * explanation here, and inventing a range for an unexplained layout is exactly
 * how this extension would evict what it cannot account for.
 *
 * MEASURED before writing the branch (189 transcripts, 202 native compactions,
 * 7 closed episodes): 5 episodes sat entirely before the cut — their BOTH
 * anchors were gone, and no deduction can help those — 2 sat entirely after it,
 * and 0 spanned it. So this branch fixes nothing that was observed: it is
 * insurance for the layout the cut makes possible. `deduced` travels with the
 * range for one reason only: the log can then say that a range was DERIVED
 * rather than read, which is the difference between a measurement and an
 * assumption.
 */
function episodeRanges(
  messages: AgentMessage[],
  episodes: Episode[],
): Map<string, { from: number; to: number; deduced: boolean }> {
  const posByToolCallId = new Map<string, number>();
  messages.forEach((m, i) => {
    // SAFETY: toolCallId exists on the real tool-result messages; the public
    // AgentMessage union does not declare it.
    const id = (m as unknown as RealMessage).toolCallId;
    if (typeof id === 'string') posByToolCallId.set(id, i);
  });
  const out = new Map<string, { from: number; to: number; deduced: boolean }>();
  for (const ep of episodes) {
    const to = ep.endToolCallId !== null ? posByToolCallId.get(ep.endToolCallId) : undefined;
    if (to === undefined) continue;
    const from = posByToolCallId.get(ep.startToolCallId);
    if (from === undefined) {
      // The prefix cut took the start anchor. With the whole prefix gone there is
      // nothing older than index 0 left to claim, and an `end` AT 0 would claim
      // nothing at all.
      if (to <= 0) continue;
      out.set(ep.name, { from: 0, to: Math.min(to, messages.length - 1), deduced: true });
      continue;
    }
    if (to <= from) continue;
    out.set(ep.name, { from, to: Math.min(to, messages.length - 1), deduced: false });
  }
  return out;
}

/** Tokens the messages in [from..to] occupy right now. */
function rangeTokens(messages: AgentMessage[], from: number, to: number): number {
  let total = 0;
  for (let i = from; i <= to; i++) total += estimateMessageTokens(messages[i]);
  return total;
}

/**
 * Tokens a level would ACTUALLY free on this range, measured with the SAME
 * primitive the eviction uses (`stripToolResult`).
 *
 * The table of fractions this replaces (reasoning 0.2, bulk 0.5, intermediate
 * 0.35) was invented, and two of its numbers contradicted the code: the eviction
 * applies `bulk` and `intermediate` as ONE transformation — it groups them in a
 * single set and calls `stripToolResult(msg, 'bulk')` for both — so giving them
 * different estimates was fiction. Only `removed` had a measured justification
 * (211-token episode: 205 gone, 6 left), and here even that is measured.
 */
function levelSavingTokens(messages: AgentMessage[], from: number, to: number, level: StripLevel): number {
  if (level === 'removed') return rangeTokens(messages, from, to);
  // `intermediate` is not a separate transformation in the eviction pass.
  const prim: StripLevel = level === 'intermediate' ? 'bulk' : level;
  let freed = 0;
  for (let i = from; i <= to; i++) {
    const before = estimateMessageTokens(messages[i]);
    const after = estimateMessageTokens(stripToolResult(messages[i], prim));
    freed += Math.max(0, before - after);
  }
  return freed;
}

function levelEnabled(cfg: CwlConfig, level: StripLevel): boolean {
  const key = LEVEL_CONFIG_KEY[level];
  if (!key) return false;
  return cfg.levels[key] === true;
}

/**
 * Runs one eviction pass: walks the graph, selects the most recoverable
 * target and computes the required stripping level.
 *
 * Returns an array of actions (empty if the budget is already satisfied).
 */
function runEvictionPass(
  cfg: CwlConfig,
  graph: EpisodeGraph,
  currentTokens: number,
  targetTokens: number,
  messages: AgentMessage[],
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

  // Measured on the SAME ranges the eviction will touch (episodeRanges), not on
  // `ep.startIdx`.
  const ranges = episodeRanges(messages, graph.closed());
  let projected = currentTokens;

  for (const target of ordered) {
    if (projected <= targetTokens) break;

    const range = ranges.get(target.name);
    // No range: the eviction will skip this episode too. Counting a saving here
    // would lower `projected` for something that will never be freed, and the
    // pass would stop before reaching the target.
    if (!range) continue;

    // Find the minimum level that brings us back under budget
    for (const level of ['reasoning', 'bulk', 'intermediate', 'removed'] as StripLevel[]) {
      if (level === 'reasoning' && target.type !== 'expl') continue;

      const action = computeStripActionWith(cfg, target, level);
      if (action) {
        const saved = levelSavingTokens(messages, range.from, range.to, level);
        // A level that frees NOTHING is not a plan: escalate to the next one
        // instead of stopping here with an action that cannot help.
        if (saved <= 0 && level !== 'removed') continue;
        target.level = level;
        projected = Math.max(0, projected - saved);
        actions.push({ ...action, estimatedTokens: rangeTokens(messages, range.from, range.to) });
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
      return { episode: ep.name, level, instruction: `Remove the thinking blocks from episode "${ep.name}". Keep tool calls and results.` };
    case 'bulk':
      return { episode: ep.name, level, instruction: `Remove the bulk enumerable tool output from episode "${ep.name}". Keep the actions.` };
    case 'intermediate':
      return { episode: ep.name, level, instruction: `Remove the intermediate results from episode "${ep.name}". Keep only the conclusions.` };
    case 'removed':
      return { episode: ep.name, level, instruction: ep.type === 'expl'
        ? `Remove the episode "${ep.name}" completely. Replace it with: "${ep.description || '(no description)'}"`
        : `Remove the action episode "${ep.name}" completely.` };
    default:
      return null;
  }
}

// ---------------------------------------------------------------------------
// Extension entry
// ---------------------------------------------------------------------------

/**
 * Fallback with no episodes: reduces the reasoning blocks of assistant messages.
 * It is the safest recovery: thinking blocks are the model's internal trace
 * — not results, not actions, not user input — and ARC does not
 * touch them. Returns null if there is nothing to remove.
 */
/**
 * Share of the list the safety window is allowed to cover.
 *
 * A window counted in USER TURNS can swallow the whole list once Pi's own
 * compaction has collapsed it. Measured in two real sessions: the context sat at
 * 455k tokens against a 68k threshold while the extension did NOTHING, because
 * 10 turns WAS the entire conversation after compaction. All three compaction
 * paths died at once — `cwl_compress_range` answered "nothing left to
 * compress", the reasoning safety net found every message protected, and the
 * budget gate asked for the impossible in a loop.
 */
const MAX_PROTECTED_SHARE = 0.5;

/**
 * Index from which the messages are PROTECTED.
 *
 * Everything at index >= this value is never evicted nor stripped. The last
 * `turns` user turns are counted, and a turn begins at a user message.
 *
 * The window is CAPPED at `MAX_PROTECTED_SHARE` of the list. Protecting the
 * recent work must never mean protecting all of it: that turns the safety
 * guarantee into the reason the context grows without bound.
 */
/**
 * Is this message the START of an exchange — a place where a turn begins?
 *
 * A `user` message is the obvious one, but not the only one. In an autonomous session the
 * operator writes NOTHING for days: the turns are the cron wake-ups and the memory card
 * refreshes. Counting only `user` made the window start at the last prompt the operator
 * typed, so everything after it — days of autonomous work — stayed protected and no
 * compaction path could touch anything. MEASURED in a live session: 348.650t protected out
 * of 436.837t, i.e. 80% of the context — and NOT because the degenerate branch below fired
 * (that session has 33 `user` messages, so the normal branch did its job): the NORMAL branch
 * was measuring the wrong thing. The last four `user` turns of an autonomous session are
 * days of machine work, and everything BEFORE the window was already compressed (88.187t
 * left, of which 85.491t of summaries inside the spans).
 *
 * A FOREIGN injected message counts like a prompt. Our OWN do not: a compression summary
 * sits where the compressed messages were, and the index demand rides at the END of the
 * list, so counting either would move the window onto the wrong thing.
 */
function isTurnBoundary(m: AgentMessage): boolean {
  // SAFETY: read-only probe of optional fields; the union does not expose them.
  const probe = m as unknown as { role?: unknown; customType?: unknown };
  if (probe.role === 'user') return true;
  if (probe.role !== 'custom') return false;
  return typeof probe.customType !== 'string' || !probe.customType.startsWith('cwl-');
}

function protectedFromIndex(messages: AgentMessage[], turns: number): number {
  if (turns <= 0) return messages.length;
  let seen = 0;
  let byTurns = 0;
  // Two boundaries IN A ROW are ONE turn. A wake-up, a card refresh and another extension's
  // notice arrive together at the start of the SAME exchange, and counting them apart made
  // "four turns" mean a turn and a half — the opposite of what a safety window is for.
  // Walking backwards, "in a row" is the message we just passed.
  let afterBoundary = false;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (!isTurnBoundary(messages[i])) { afterBoundary = false; continue; }
    if (!afterBoundary) {
      seen++;
      if (seen > turns) { byTurns = i + 1; break; }
    }
    afterBoundary = true;
  }
  // Normal case: the window is honoured EXACTLY as configured. The cap must not
  // touch this, or `protectedTurns` would stop meaning what it says.
  if (byTurns > 0) return byTurns;
  // DEGENERATE case: the list holds fewer user turns than the window, so the
  // window would cover EVERYTHING. That is not a safety window any more: it is
  // the reason the context grows without bound, because all three compaction
  // paths die together. Measured in real sessions: 469k tokens against a 68k
  // threshold, `cwl_compress_range` answering "nothing to compress", the
  // reasoning net finding every message protected, and the gate looping on an
  // impossible demand. Here the OLDEST part is freed so progress is always
  // possible.
  return Math.floor(messages.length * MAX_PROTECTED_SHARE);
}

/**
 * Endpoints of the largest range `cwl_compress_range` may compress right now:
 * the user/assistant messages BEFORE the safety window that no existing span
 * already covers.
 *
 * The extension computes the ADDRESS because the agent cannot. Leaving it to the
 * model did not work: measured in a real session, the budget gate asked for
 * `startHash`/`endHash` three times and got nothing back, because an opaque hash
 * cannot be mapped back to a message by the very agent that is supposed to pick
 * it. The division that does work: the extension picks the addresses, the model
 * writes the summary — the only part only the model can do.
 */
function compressibleRange(
  messages: AgentMessage[],
  spans: CompressedSpan[],
  protectedTurns: number,
): { startHash: string; endHash: string; tokens: number } | null {
  const floor = protectedFromIndex(messages, protectedTurns);

  // One resolution for every caller (see locateSpans). This used to be a copy
  // that claimed to be "the same resolution applySpans performs" while NOT
  // extending the range over the tool results that follow an assistant: the
  // `covered` set was smaller than what the span actually removes.
  const covered = new Set<number>();
  for (const { from, to } of locateSpans(messages, spans).resolved) {
    for (let i = from; i <= to; i++) covered.add(i);
  }

  let first = -1;
  let last = -1;
  let tokens = 0;
  for (let i = 0; i < Math.min(floor, messages.length); i++) {
    // SAFETY: read-only field probe (role); the union does not expose it.
    const role = (messages[i] as unknown as RealMessage).role;
    if (role !== 'user' && role !== 'assistant') continue;
    if (covered.has(i)) continue;
    // An empty body hashes to sha256(""), so every empty message would share one
    // address. They are not valid endpoints.
    if (!textOf(messages[i]).trim()) continue;
    if (first < 0) first = i;
    last = i;
    tokens += estimateMessageTokens(messages[i]);
  }
  if (first < 0 || last <= first) return null;
  return {
    startHash: addressOf(messages[first]),
    endHash: addressOf(messages[last]),
    tokens,
  };
}

/**
 * Stores the range the agent may ask to compress, and LOGS the decision.
 *
 * The log line is the instrument every future diagnosis of "why is nothing on
 * offer" depends on, so BOTH places that compute the range must emit it. This
 * hook has two exits that matter — the normal path and the spans branch — and the
 * first version of the line covered only one of them: its silence proved
 * nothing, and that gap cost a whole diagnosis.
 */
function storeRange(
  st: CwlState,
  cf: CwlConfig,
  range: ReturnType<typeof compressibleRange>,
  messages: AgentMessage[],
  currentTokens?: number,
  trigger?: number,
): void {
  st.rangeStartHash = range?.startHash ?? null;
  st.rangeEndHash = range?.endHash ?? null;
  st.rangeTokens = range?.tokens ?? 0;
  // The spans branch runs before `trigger` exists, so fall back to the numbers
  // the state already carries.
  const tokens = currentTokens ?? st.lastMeasuredTokens;
  const trig = trigger ?? cf.tokenBudget * cf.thresholdRatio;
  debugLog(cf, `RANGE ${range
    ? `${range.startHash}..${range.endHash} (~${range.tokens}t)`
    : 'none'} | ${messages.length} msgs, ${tokens}t vs trigger ${Math.round(trig)}t, ${st.spans.length} span(s)`);
}

function globalReasoningStrip(
  messages: AgentMessage[],
  floor: number = messages.length,
): { kept: AgentMessage[]; changed: number } | null {
  const kept: AgentMessage[] = [];
  let changed = 0;
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    // The safety window is never touched, whatever is inside it.
    if (i >= floor) { kept.push(msg); continue; }
    // SAFETY: read-only field probe (role/content); the union does not expose them.
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

/**
 * A compressed span: the messages from `startHash` to `endHash` are replaced
 * in the active context by the summary alone. The original stays in Pi's JSONL
 * transcript, which is append-only: compression is therefore LOSSLESS, and
 * `cwl_recall` can retrieve the text in full.
 */
/**
 * Stable id of a compressed span — a "leaf" of the index.
 *
 * The two anchors hashed together, so it survives restarts and does not shift
 * when the list moves: a summary lives ONLY in the state, and this id is how
 * `cwl_open` asks for one back without walking the transcript.
 */
function spanId(startHash: string, endHash: string): string {
  return `sp-${hashText(`${startHash}|${endHash}`).slice(0, 8)}`;
}

/**
 * The id of a span, derived from its anchors when the persisted state predates
 * ids. ONE definition: `cwl_open`, the injected notice and the tool that writes
 * the compression must all agree on what a leaf is called, or the agent is handed
 * an id that no tool accepts.
 */
function idOfSpan(sp: CompressedSpan): string {
  return sp.id ?? spanId(sp.startHash, sp.endHash);
}

/**
 * A NODE of level 1: a container that grows up to 30 leaves.
 *
 * Why it exists: the head of the context must have a BOUNDED shape. Five leaves
 * stay loose — whole bodies, the working set — and everything older is absorbed,
 * its body already out of the context (the micro took its place). What a node adds
 * is a PLACE to gather those leaves, with an id to open.
 *
 * It re-orders nothing: leaves enter in chronological order and never move, so the
 * head stays stable — and stable means CACHE.
 */
/**
 * The OLD node — the pit.
 *
 * It holds YOUNG NODES, not leaves, so `cwl_open` can still page through them one
 * by one; what its leaves lose is the context, because the riassuntone stands for
 * all of them. The riassuntone is the one piece of this design no rule can
 * recompute, so it — and only it — is persisted.
 */
interface OldNode {
  /** `nd-`-style id of the pit itself. */
  id: string;
  /** Ids of the young nodes it absorbed, oldest first. */
  nodes: string[];
  /** The agent's synthesis. Empty means the merge has not happened yet. */
  summary: string;
  at: number;
}

/** How many leaves a page lists (a node's micros, or a pit's hot leaves). */
const NODE_PAGE_MAX = 30;

/**
 * What `applySpans` needs to know about the pit: its header, and the leaves whose
 * micros must STOP being injected.
 *
 * `null` when there is no riassuntone yet: the merge takes effect only once the
 * synthesis EXISTS, because a pit without one would make those leaves disappear from
 * the context with nothing standing for them — the content would still be in the
 * state and in the transcript, and the agent's thread would be cut.
 */
interface PitView {
  id: string;
  /** How many young nodes are behind the synthesis. */
  nodes: number;
  summary: string;
  /** The leaves it stands for. */
  leaves: Set<string>;
}

/** Builds the view above from the state. See `PitView` for why the summary gates it. */
function pitView(st: CwlState): PitView | null {
  const pit = st.oldNode;
  if (!pit || !pit.summary) return null;
  const inPit = new Set(pit.nodes);
  const leaves = new Set<string>();
  for (const nd of st.nodes) {
    if (!inPit.has(nd.id)) continue;
    for (const id of nd.leaves) leaves.add(id);
  }
  return { id: pit.id, nodes: pit.nodes.length, summary: pit.summary, leaves };
}

/** How many dropped spans stay recoverable by id. */
const GRAVE_MAX = 200;

/** How many waiting leaf ids the micro demand names in one turn: a few is enough to start. */
const MICRO_DEMAND_IDS = 8;

/**
 * The label a compression may ALREADY carry: trimmed, or absent when empty.
 *
 * Whoever writes a compression summary has just read the messages it summarises, so the
 * ~200-word micro is nearly free at that moment. Asking for it later means a second pass over
 * a body that by then lives only in the state — and a batch of five leaves to label. Arriving
 * here, the leaf is ready for a node the moment it is born and `leavesDue` never has to fire
 * for it. It stays OPTIONAL: a compression without a label is valid, and the demand picks it
 * up later.
 */
const microOrUndefined = (v: unknown): string | undefined => {
  const s = String(v ?? '').trim();
  return s.length > 0 ? s : undefined;
};

/**
 * What is left of a span the state DROPPED: enough to find its original again.
 *
 * A pruned span's summary is gone for good — it lived only in the state — but the
 * messages it replaced are still in the append-only transcript, and the stable ids are
 * what find them. Without this record the leaf would be unreachable by id: the id is a
 * HASH of the two anchors, so it cannot be turned back into them.
 */
interface Grave {
  id: string;
  startSid: string;
  endSid: string;
  at: number;
}

interface SpanNode {
  /** Stable id: `nd-` + 8 hex of the first leaf. */
  id: string;
  /** Leaf ids, chronological. A leaf belongs to AT MOST ONE node. */
  leaves: string[];
  at: number;
}

/**
 * Decides which leaves a node owns and how many are still waiting for a micro.
 *
 * The rule, in order: the LAST `LOOSE_LEAVES` leaves by creation stay loose; every
 * older leaf that HAS a micro enters the current node (a new one when that is full);
 * an older leaf WITHOUT a micro is COUNTED as waiting — its body is still in the
 * context, and that is a cost to declare, not to hide.
 *
 * A leaf that left the state (pruned: its anchors are gone) cannot stay in a node,
 * or the node would describe material that no longer exists.
 */
function refreshNodes(
  st: CwlState,
  cf: CwlConfig,
): { formed: number; waiting: number; waitingIds: string[]; due: number } {
  const leaves = [...st.spans].sort((a, b) => a.at - b.at);
  const byId = new Map(leaves.map((l) => [idOfSpan(l), l]));
  // A leaf that left the state (pruned) cannot stay, and neither can a leaf whose
  // micro was REMOVED: un-absorbing puts its body back in the context, so it must
  // come out of the node it no longer belongs to. One condition covers both.
  for (const nd of st.nodes) nd.leaves = nd.leaves.filter((id) => Boolean(byId.get(id)?.micro));
  st.nodes = st.nodes.filter((nd) => nd.leaves.length > 0);

  // The nodes the pit absorbed are SETTLED: they keep their leaves and nothing new
  // enters them. That is exactly what lets their micros leave the context — the
  // riassuntone stands for all of them.
  const inPit = new Set(st.oldNode?.nodes ?? []);
  const young = st.nodes.filter((nd) => !inPit.has(nd.id));
  const settled = new Set(st.nodes.flatMap((nd) => nd.leaves));
  const loose = new Set(leaves.slice(-cf.looseLeaves).map((l) => idOfSpan(l)));
  let formed = 0;
  let waiting = 0;
  // WHICH leaves are waiting, not just how many: without the ids the demand can only say
  // "some leaves" and the agent would have to guess which, or go read the state file.
  const waitingIds: string[] = [];
  for (const leaf of leaves) {
    const id = idOfSpan(leaf);
    if (loose.has(id) || settled.has(id)) continue;
    if (!leaf.micro) { waiting++; waitingIds.push(id); continue; }
    let current = young[young.length - 1];
    if (!current || current.leaves.length >= cf.nodeCapacity) {
      current = { id: `nd-${hashText(id).slice(0, 8)}`, leaves: [], at: Date.now() };
      st.nodes.push(current);
      young.push(current);
      formed++;
    }
    current.leaves.push(id);
  }
  return { formed, waiting, waitingIds, due: young.length >= cf.mergeNodesAt ? 1 : 0 };
}

interface CompressedSpan {
  startHash: string;
  endHash: string;
  /** Stable id, see `spanId`. */
  id: string;
  summary: string;
  /**
   * The MICRO-summary: what stands in the context once the leaf has been
   * absorbed. The body above stays whole — this design's promise is that
   * nothing is lost, and `cwl_open` reads the body back in full.
   */
  micro?: string | null;
  /**
   * How many times this leaf was OPENED, and when was the last time.
   *
   * USAGE, not a decision — and not derivable, so it lives with the span in the state
   * and survives a reload. It only ever ORDERS a page: no counter deletes anything,
   * because "never opened" is not "useless". A leaf used while it was FRESH left no
   * trace, and that is correct: its body was in the context already and needed no help.
   * What the counter records is exactly the other case, the one that matters here — the
   * leaves reopened AFTER they aged, the ones that look stale but are still in use.
   */
  opens?: number;
  lastOpen?: number;
  /**
   * The STABLE IDS of the two anchors: `stableIdOf` of the messages they point at.
   *
   * Kept because they are the only thing that can find this range again in the
   * transcript once the state has dropped the span. The hashes above cannot: `addressOf`
   * mixes the id with the TEXT, and this extension strips reasoning from the messages it
   * compacts — `contentToText` includes `part.thinking` — so a hash computed from the
   * transcript can differ from the one computed from the context. A timestamp cannot.
   */
  startSid?: string;
  endSid?: string;
  at: number;
  /**
   * Whether this span's saving has already been added to `totalEvictedTokens`.
   *
   * Applying a span is IDEMPOTENT and happens on EVERY turn: Pi rebuilds the
   * list from its append-only transcript, so the originals come back each time
   * and the span has to be re-applied to keep the context compressed. Counting
   * is NOT idempotent. Measured before this flag existed: one span, three
   * identical turns, 427 -> 1708 tokens "saved", exactly four times the truth.
   * That counter is the only evidence the operator has that this works, so it
   * must not grow on its own.
   */
  counted?: boolean;
}

/** Stable hash of a message's text: 12 hex chars, like ARC ids. */
function hashText(text: string): string {
  return createHash('sha256').update(text).digest('hex').slice(0, 12);
}

/** Role of a message, or '' when it has no usable role. */
function roleOf(m: unknown): string {
  // SAFETY: read-only probe of an optional field; the union does not expose it.
  const r = (m as { role?: unknown })?.role;
  return typeof r === 'string' ? r : '';
}

/** ids of the tool calls an assistant message carries, in order. */
function callIdsOf(m: AgentMessage): string[] {
  // SAFETY: read-only probe of an optional field; the union does not expose it.
  const c = (m as unknown as { content?: unknown }).content;
  if (!Array.isArray(c)) return [];
  const out: string[] = [];
  for (const b of c) {
    // SAFETY: read-only probe of block fields; blocks are untyped here.
    const blk = b as { type?: unknown; id?: unknown };
    if (blk && blk.type === 'toolCall' && typeof blk.id === 'string') out.push(blk.id);
  }
  return out;
}

/** id of the tool call a result answers, or '' when the message is not a result. */
function resultCallId(m: AgentMessage): string {
  // SAFETY: read-only probe of an optional field; the union does not expose it.
  const id = (m as unknown as { toolCallId?: unknown }).toolCallId;
  return typeof id === 'string' ? id : '';
}

/**
 * Stable identity of a single message, when Pi gives us one.
 *
 * MEASURED why it is needed: in a real session 303 assistant messages carry the
 * text "*", 51 carry "🌙", 40 carry the same 100-character sentence. Hashing the
 * TEXT alone collapses every one of them onto a SINGLE address, and a map keyed
 * by that hash keeps the last occurrence — so a span created on the first one
 * resolved on the last, and the range the agent asked to compress was not the
 * range that got replaced. The transcript carries a timestamp on every message
 * (assistant: epoch ms) and it survives into this hook, so that is what
 * separates two identical texts.
 */
function stableIdOf(m: AgentMessage): string {
  // SAFETY: read-only probes of optional fields; the union does not expose them.
  const o = m as unknown as { timestamp?: unknown; id?: unknown };
  if (typeof o.timestamp === 'number' && Number.isFinite(o.timestamp)) return `t${o.timestamp}`;
  if (typeof o.timestamp === 'string' && o.timestamp) return `t${o.timestamp}`;
  return typeof o.id === 'string' && o.id ? `i${o.id}` : '';
}

/**
 * Address of a message: its text, plus a stable id when there is one.
 *
 * Without an id it degrades to the text hash alone — which is also the address
 * of every span created before this change, so those keep resolving.
 */
function addressOf(m: AgentMessage): string {
  const id = stableIdOf(m);
  const text = textOf(m);
  return id ? hashText(`${id}|${text}`) : hashText(text);
}

/**
 * The two address maps for one message list.
 *
 * `exact` holds addresses built by `addressOf` (unique when a stable id exists);
 * `legacy` holds the text-only hash, which is what spans created before this
 * change carry. Callers look in `exact` first, so old spans keep resolving and
 * new ones cannot land on the wrong message.
 *
 * An empty body is not an address in either map: sha256("") would map every
 * empty message onto one entry — 1407 of them measured in the largest session.
 */
function addressMaps(messages: AgentMessage[]): { exact: Map<string, number>; legacy: Map<string, number> } {
  const exact = new Map<string, number>();
  const legacy = new Map<string, number>();
  messages.forEach((m, i) => {
    // SAFETY: read-only field probe (role); the union does not expose it.
    const role = (m as unknown as RealMessage).role;
    if (role !== 'user' && role !== 'assistant') return;
    const text = textOf(m);
    if (!text.trim()) return;
    exact.set(addressOf(m), i);
    legacy.set(hashText(text), i);
  });
  return { exact, legacy };
}

/**
 * Every tool RESULT must have its tool CALL, and every CALL its result: the
 * provider rejects the whole request otherwise. Measured on real sessions, the
 * two symptoms are `400 status code (no body)` (the scrocco-llm gateway, which
 * says nothing) and `No tool call found for function call output with call_id
 * ...` (Codex, which names the orphan).
 *
 * The pairing must NOT be inferred from position. Measured over the session
 * corpus (187 files, 35041 results): 5 results are separated from their own
 * assistant by ANOTHER assistant, because Pi writes the next assistant before
 * the batch of results is recorded. A rule like "swallow the toolResults that
 * follow the last message of the range" cannot see those, so the guarantee
 * lives HERE, by id, over the final list.
 *
 * The two directions cannot interfere: a result is dropped only when NO call
 * carries its id, and a call is stripped only when no result carries its id.
 */
function repairToolPairs(kept: AgentMessage[]): { kept: AgentMessage[]; dropped: number; stripped: number } {
  const calls = new Set<string>();
  for (const m of kept) for (const id of callIdsOf(m)) calls.add(id);
  const results = new Set<string>();
  for (const m of kept) {
    const id = resultCallId(m);
    if (id) results.add(id);
  }

  const out: AgentMessage[] = [];
  let dropped = 0;
  let stripped = 0;
  for (const m of kept) {
    // A result whose call did not survive is unusable: the model cannot connect
    // it to anything, and the provider rejects the whole request.
    const rid = resultCallId(m);
    if (rid && !calls.has(rid)) { dropped++; continue; }
    const ids = callIdsOf(m);
    const orphans = ids.filter((id) => !results.has(id));
    if (orphans.length > 0) {
      // Strip the calls whose result is gone, and drop the message when nothing
      // else is left. The deterministic eviction path already does this from
      // the other direction (H1, droppedToolCallIds).
      // SAFETY: read-only probe of an optional field; the union does not expose it.
      const c = (m as unknown as { content?: unknown }).content;
      const filtered = Array.isArray(c)
        ? c.filter((b) => {
            // SAFETY: read-only probe of block fields; blocks are untyped here.
            const blk = b as { type?: unknown; id?: unknown };
            return !(blk && blk.type === 'toolCall' && typeof blk.id === 'string' && orphans.includes(blk.id));
          })
        : c;
      if (!Array.isArray(filtered) || filtered.length === 0) { stripped++; continue; }
      // SAFETY: the same message, with a filtered content list.
      out.push({ ...(m as object), content: filtered } as unknown as AgentMessage);
      stripped++;
      continue;
    }
    out.push(m);
  }
  return { kept: out, dropped, stripped };
}

/** Locates the session JSONL transcript, looking it up by key suffix. */
/** Reads a file or returns null: missing and unreadable must not fail. */
/** Concatenable text from a message (string or list of blocks). */
function textOf(m: unknown): string {
  const c = (m as { content?: unknown }).content;
  if (typeof c === 'string') return c;
  if (!Array.isArray(c)) return '';
  let out = '';
  for (const b of c as { text?: string; thinking?: string }[]) {
    if (typeof b?.text === 'string') out += b.text;
    else if (typeof b?.thinking === 'string') out += b.thinking;
  }
  return out;
}

function readFileOrNull(p: string): string | null {
  try { return fs.readFileSync(p, 'utf-8'); } catch { return null; }
}

/**
 * Resolves a session key to the JSONL transcript on disk.
 *
 * The key can be either the transcript PATH itself — `sessionKey()` prefers
 * `sessionManager.getSessionFile()` — or a bare session id, the older shape.
 * The previous version handled only the bare id: it built `_${key}.jsonl` and
 * looked for a file ending with it. With a path as key the suffix became
 * `_/home/.../sessione.jsonl.jsonl`, which matches nothing, so `cwl_recall`
 * answered "transcript not found" while the transcript was right there on disk.
 */
function findTranscript(sessionKey: string): string | null {
  if (path.isAbsolute(sessionKey) && sessionKey.endsWith('.jsonl') && fs.existsSync(sessionKey)) {
    return sessionKey;
  }
  // `<cwd>::<sid>` and `<sid>` both resolve by the `_<sid>.jsonl` suffix.
  const id = sessionKey.includes('::') ? sessionKey.split('::')[1] : sessionKey;
  const sessionsDir = path.join(os.homedir(), '.pi', 'agent', 'sessions');
  const suffix = `_${id}.jsonl`;
  let dirs: string[];
  try { dirs = fs.readdirSync(sessionsDir); } catch { return null; }
  for (const dir of dirs) {
    const full = path.join(sessionsDir, dir);
    try {
      for (const f of fs.readdirSync(full)) {
        if (f.endsWith(suffix)) return path.join(full, f);
      }
    } catch { /* unreadable directory: try the next one */ }
  }
  return null;
}

/**
 * Concatenates the readable parts of a content block array: text, thinking and
 * the tool calls (with their arguments), so a recalled episode shows WHAT was
 * done and not only what was said.
 */
function blocksToText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const out: string[] = [];
  for (const part of content as RealContentBlock[]) {
    if (!part || typeof part !== 'object') continue;
    if (typeof part.text === 'string') out.push(part.text);
    else if (typeof part.thinking === 'string') out.push(`[thinking] ${part.thinking}`);
    else if (part.type === 'toolCall') {
      out.push(`[toolCall ${part.name ?? '?'}] ${JSON.stringify(part.arguments ?? {})}`);
    }
  }
  return out.join('\n');
}

/**
 * Reads one transcript record.
 *
 * Two shapes exist in the wild: the session log nests the message under
 * `message`, while subagent artifacts keep `role`/`text` at the top level. Both
 * are accepted, so the same reader works on either.
 */
function transcriptRecordText(
  rec: unknown,
): { role: string; toolCallId?: string; text: string; sid: string } | null {
  if (!rec || typeof rec !== 'object') return null;
  const r = rec as Record<string, unknown>;
  const src = (r.message ?? r) as RealMessage;
  const role = typeof src?.role === 'string' ? src.role : '';
  if (!role) return null;
  const toolCallId = typeof src.toolCallId === 'string' ? src.toolCallId : undefined;
  const text = blocksToText(src.content) || (typeof r.text === 'string' ? r.text : '');
  // The SAME identity `stableIdOf` gives a live message: one definition, because a
  // lookup that disagrees with the writer finds nothing and does not say why.
  // SAFETY: real transcript records are messages in every field that matters here, but
  // the union Pi declares is narrower than the JSON on disk.
  const sid = stableIdOf(src as unknown as AgentMessage);
  return { role, toolCallId, text, sid };
}

/**
 * Reads an episode back from the append-only transcript.
 *
 * The anchors are the toolCallIds of the two `delimiter` results: they are real
 * records in the transcript, so the span they delimit is exact and does NOT
 * shift with the context (which is why the episode stores them instead of the
 * indices). Returns null when the OPENING anchor is missing — a dangling
 * pointer, reported as such rather than as an empty episode.
 */
function extractEpisodeText(
  raw: string,
  startToolCallId: string,
  endToolCallId: string | null,
): string | null {
  const parts: string[] = [];
  let started = false;
  for (const line of raw.split(/\r?\n/)) {
    if (!line) continue;
    let rec: unknown;
    try { rec = JSON.parse(line); } catch { continue; }
    const info = transcriptRecordText(rec);
    if (!info) continue;
    if (!started) {
      if (info.toolCallId === startToolCallId) started = true;
      continue;
    }
    if (endToolCallId !== null && info.toolCallId === endToolCallId) break;
    if (info.text.trim()) parts.push(`[${info.role}] ${info.text}`);
  }
  return started ? parts.join('\n\n') : null;
}

/**
 * Reads back the ORIGINAL messages between two anchors, by STABLE ID.
 *
 * Why the ids and not the text hashes: `addressOf` mixes the id with the TEXT, and this
 * extension strips reasoning from the messages it compacts — `contentToText` includes
 * `part.thinking` — so a hash computed from the transcript can differ from the one
 * computed from the context, and the lookup would fail without saying why. The timestamp
 * cannot drift. MEASURED: that difference is exactly what makes a hash-based lookup
 * unreliable, and it is why a span carries `startSid`/`endSid` as well.
 *
 * Returns null when the OPENING anchor is missing: a dangling pointer, reported as such
 * rather than as an empty range.
 */
function transcriptRangeText(raw: string, startSid: string, endSid: string | null): string | null {
  const parts: string[] = [];
  let started = false;
  for (const line of raw.split(/\r?\n/)) {
    if (!line) continue;
    let rec: unknown;
    try { rec = JSON.parse(line); } catch { continue; }
    const info = transcriptRecordText(rec);
    if (!info || !info.sid) continue;
    if (!started) {
      if (info.sid !== startSid) continue;
      started = true;
    }
    // The anchors here ARE content (unlike an episode's delimiters): include both, or the
    // two messages the span began and ended on would be missing from its own original.
    if (info.text.trim()) parts.push(`[${info.role}] ${info.text}`);
    if (endSid !== null && info.sid === endSid) break;
  }
  return started ? parts.join('\n\n') : null;
}

/**
 * Resolves every span against one message list: the ONLY place that turns two
 * addresses into a range.
 *
 * Three copies of this logic used to live in this file — compressibleRange,
 * applySpans and declareFloor — and they drifted: only applySpans extended the
 * range over the tool results that follow an assistant, so `covered`, the applied
 * set and the floor count disagreed about the same span. compressibleRange even
 * carried the comment "Same resolution applySpans performs" while not doing it.
 *
 * Two rules live here and nowhere else:
 *  - extend FORWARD over consecutive `toolResult` messages;
 *  - drop a span contained in another (a summary inside a summary loses twice).
 *
 * An endpoint that no longer resolves means the history it described is not in
 * this list any more: the span is returned in `dead` so the caller can drop it.
 */
function locateSpans(
  messages: AgentMessage[],
  spans: CompressedSpan[],
): { resolved: { sp: CompressedSpan; from: number; to: number }[]; dead: CompressedSpan[]; overlapped: number } {
  // address -> position index, computed once.
  const { exact, legacy } = addressMaps(messages);
  const findPos = (h: string): number | undefined => exact.get(h) ?? legacy.get(h);

  // Nested spans make no sense: if one contains another, the outermost one
  // wins. A summary inside a summary loses information twice.
  const dead: CompressedSpan[] = [];
  const resolved = spans
    .map((sp) => {
      const from = findPos(sp.startHash);
      const to = findPos(sp.endHash);
      // Both endpoints gone: the history this span replaced is not in the list
      // any more (native compaction replaced it, or the transcript grew past it).
      // Collect it so the caller can drop it: its summary is thousands of
      // characters of dead weight, re-saved in the state on every turn.
      if (from === undefined || to === undefined) { dead.push(sp); return null; }
      // Backfill the stable ids the first time the endpoints are visible: this is the
      // only place that has BOTH the span and the messages it points at, so it is the
      // only place that can write down what will be needed to find it again later.
      if (!sp.startSid) sp.startSid = stableIdOf(messages[from]);
      if (!sp.endSid) sp.endSid = stableIdOf(messages[to]);
      // An endpoint is always a user/assistant message, but the message right
      // after an assistant is usually its tool RESULT (role 'toolResult').
      // Replacing the assistant while keeping that result leaves a tool_result
      // whose tool_use no longer exists, and the provider rejects the whole
      // request with `400 status code (no body)` — blocking the session.
      // MEASURED on a real session: the range ended on an assistant carrying a
      // toolCall at index 1296 while its 6396-char toolResult sat just outside,
      // at 1297. The deterministic eviction path already guards the opposite
      // direction (H1, droppedToolCallIds); the spans path had no guard at all.
      // Extending forward is the right direction: the result is part of the
      // same exchange being summarised, so its tokens land in `original` too.
      // But this only covers the CONTIGUOUS layout: the pairing can never be
      // trusted to position (see repairToolPairs, which is the actual guarantee).
      // This is the faithful path for the common case.
      let end = Math.max(from, to);
      while (end + 1 < messages.length && roleOf(messages[end + 1]) === 'toolResult') end++;
      return { sp, from, to: end };
    })
    .filter((x): x is { sp: CompressedSpan; from: number; to: number } => x !== null);

  // A span contained in another can never apply on its own: the outer summary
  // already covers its region, and a summary inside a summary describes less
  // than what is already there. Dropping it is right; dropping it SILENTLY was
  // not. This used to be a single `.filter(...)`, so an inner span ended up in
  // NEITHER `resolved` nor `dead`: not applied, not declared anywhere, and its
  // summary stayed in the state — thousands of characters re-saved on every
  // turn, accounted for by no line of the log. MEASURED in a sandbox with three
  // spans in the state (two inside the third): the log said `SPANS applied: 1`
  // and nothing else, while the state kept all three.
  // Equal ranges count as containment: the FIRST one wins, the others are pruned.
  // Sorted by start and, for two spans with the same start, the WIDER one first:
  // after that a single sweep is enough. A span is contained in an earlier one
  // exactly when its `to` is not greater than the greatest `to` seen so far, and
  // putting the wider first is what makes the equal-start case fall on the right
  // side of that comparison. O(n log n) once, instead of a nested scan on a path
  // that runs on every turn.
  const ordered = resolved.sort((a, b) => a.from - b.from || b.to - a.to);
  const survivors: typeof ordered = [];
  let widestTo = -1;
  for (const x of ordered) {
    if (widestTo >= x.to) { dead.push(x.sp); continue; }
    survivors.push(x);
    widestTo = x.to;
  }

  // A PARTIAL overlap — two survivors sharing at least one message — cannot be
  // fixed from here: dropping either one would bring back the messages that only
  // IT covers, undoing the compression for that region. What it is, then, is a
  // redundancy: the shared messages are described by two summaries. And it has to
  // be SAID, because nothing else in the pipeline can see it — `covered` is a
  // set, and `declareFloor`'s four numbers add up by construction (`outside` is
  // the residual), so a double count leaves no trace there either.
  // Survivors are sorted by start and none contains another, so a shared region
  // always shows up between two ADJACENT intervals: one pass, no nested scan.
  let overlapped = 0;
  for (let i = 1; i < survivors.length; i++) {
    if (survivors[i].from <= survivors[i - 1].to) overlapped++;
  }
  return { resolved: survivors, dead, overlapped };
}

/**
 * Applies the spans compressed by the LLM: the messages between startHash and
 * endHash are replaced by the summary alone. The original stays in the JSONL
 * transcript, which is append-only: compression is therefore LOSSLESS and
 * cwl_recall can retrieve the text in full.
 */
function applySpans(
  messages: AgentMessage[],
  spans: CompressedSpan[],
  pit: PitView | null,
  demanda: string | null,
): {
  kept: AgentMessage[];
  applied: number;
  saved: number;
  newApplied: number;
  newSaved: number;
  pairDropped: number;
  pairStripped: number;
  /**
   * Positions in `kept` of the messages that are there BECAUSE of a span: the
   * injected summaries and whatever survived inside a span (a user turn is
   * inviolable even when a span covers it).
   *
   * They are recorded HERE because here is the only place that knows: the loop
   * below pushes them one by one. Re-deriving them later is impossible, and that
   * is not a guess: a span whose closing anchor is an assistant REMOVES that
   * anchor, so a second resolution on the compressed list finds nothing — which
   * is exactly how `0t inside the spans` was born, with 27 spans applied and
   * `0 of 27 spans located` printed in a live session.
   *
   * Empty when the pair repair dropped an orphan result: then every later
   * position moved, and a stale position is worse than none.
   */
  insideOut: number[];
  /** Spans whose endpoints are no longer in the list: they can never apply. */
  dead: CompressedSpan[];
  /** Pairs of spans that share at least one message (a partial overlap). */
  overlapped: number;
} {
  // The demand rides at the END of the list: it is an instruction for NOW, not a piece
  // of the history. It is built FIRST, because it is appended on every return —
  // including the two that change nothing, or a turn with no resolvable span would
  // swallow the request and the agent would never hear it.
  // SAFETY: Pi accepts `custom` in the context hook although the AgentMessage union
  // does not declare it — the same contract as the compression notices below.
  const demandMsg: AgentMessage | null = demanda
    ? ({ role: 'custom', customType: 'cwl-demand', content: demanda, display: false, timestamp: Date.now() } as unknown as AgentMessage)
    : null;
  /** Appends the demand, when there is one. */
  const out = (list: AgentMessage[]): AgentMessage[] => (demandMsg ? [...list, demandMsg] : list);

  if (spans.length === 0) return { kept: out(messages), applied: 0, saved: 0, newApplied: 0, newSaved: 0, pairDropped: 0, pairStripped: 0, insideOut: [], dead: [], overlapped: 0 };

  const { resolved, dead, overlapped } = locateSpans(messages, spans);

  if (resolved.length === 0) return { kept: out(messages), applied: 0, saved: 0, newApplied: 0, newSaved: 0, pairDropped: 0, pairStripped: 0, insideOut: [], dead, overlapped: 0 };

  const replaced = new Set<number>();
  const injected: (AgentMessage | null)[] = [];
  let saved = 0;
  let newSaved = 0;
  let newApplied = 0;
  /** The pit injects ONE block, at its first leaf: the others inject nothing at all. */
  let pitBlockDone = false;

  /**
   * Inside a span these roles are NOT replaced by the summary: the operator's own
   * turns (`user`), the directives (`system`/`developer`) and the injected
   * messages (`custom`, which is where the span's own summary lives) stay where
   * they are. Defined ONCE, and used both by the saving computation below and by
   * the reconstruction of the list further down: if the two ever disagree, the
   * saving becomes a number that claims space this list never freed. It did: 749t
   * declared against 298t really lost on a 12-message probe.
   */
  const keptInsideSpan = (r: string | undefined): boolean =>
    r === 'user' || r === 'system' || r === 'developer' || r === 'custom';

  for (const { sp, from, to } of resolved) {
    // What counts is what is really taken AWAY, not the size of the range: the
    // roles that survive keep costing tokens in the list this function returns.
    let removed = 0;
    for (let i = from; i <= to; i++) {
      replaced.add(i);
      if (!keptInsideSpan(roleOf(messages[i]))) removed += estimateMessageTokens(messages[i]);
    }
    // A leaf the OLD node holds is no longer injected one by one: the riassuntone
    // stands for it. Its messages still leave the context — that IS the saving — and
    // the FIRST pit leaf carries the synthesis in their place. The others bring no
    // block at all, so what they free is the whole `removed`, wrapper included.
    if (pit && pit.leaves.has(idOfSpan(sp))) {
      if (pitBlockDone) {
        saved += removed;
        if (!sp.counted) { sp.counted = true; newSaved += removed; newApplied++; }
        injected.push(null);
        continue;
      }
      pitBlockDone = true;
      // SAFETY: same contract as the per-leaf notice below — Pi accepts `custom` in the
      // context hook although the AgentMessage union does not declare it.
      const buildPit = (claim: number): AgentMessage => ({
        role: 'custom',
        customType: 'cwl-compressed',
        content: t('oldHead')(pit.id, pit.nodes, claim) + pit.summary,
        display: false,
        timestamp: Date.now(),
      } as unknown as AgentMessage);
      let gainPit = Math.max(0, removed - estimateMessageTokens(buildPit(0)));
      for (let k = 0; k < 3; k++) {
        const next = Math.max(0, removed - estimateMessageTokens(buildPit(gainPit)));
        if (next === gainPit) break;
        gainPit = next;
      }
      saved += gainPit;
      if (!sp.counted) { sp.counted = true; newSaved += gainPit; newApplied++; }
      injected.push(buildPit(gainPit));
      continue;
    }
    // SAFETY: Pi accepts the custom role in the context hook although the
    // AgentMessage union does not declare it; the extra keys are its contract.
    const build = (claim: number): AgentMessage => ({
      role: 'custom',
      customType: 'cwl-compressed',
      content: t('compressedNotice')(sp.startHash, sp.endHash, claim, idOfSpan(sp)) + (sp.micro ?? sp.summary),
      display: false,
      timestamp: Date.now(),
    } as unknown as AgentMessage);
    // The saving is what is removed MINUS what takes its place, and what takes its
    // place is the WHOLE injected message: the wrapper the agent keeps reading
    // ("[CWL ...] (~N token risparmiati)") costs tokens too. Counting only
    // `sp.summary` made this row claim 749t (and, with the surviving `user` turns
    // already subtracted, 377t) against the 298t the context really lost on a
    // 12-message probe. ONE number, used by the totals, by the log row and by the
    // notice inside the injected message. It is a fixed point because the claim is
    // printed INSIDE that same message; it converges at once (only the digits
    // change) and the extra passes cost nothing.
    let gain = Math.max(0, removed - estimateMessageTokens(build(0)));
    for (let k = 0; k < 3; k++) {
      const next = Math.max(0, removed - estimateMessageTokens(build(gain)));
      if (next === gain) break;
      gain = next;
    }
    const injectedMsg = build(gain);
    saved += gain;
    // Count it once, HERE, where the real gain is known. Every later turn marks
    // it `counted`, so the caller leaves the totals alone.
    if (!sp.counted) { sp.counted = true; newSaved += gain; newApplied++; }
    injected.push(injectedMsg);
  }

  const kept: AgentMessage[] = [];
  // Recorded here, where the positions are BORN: everything this loop pushes
  // because of a span (its summary, and whatever survives inside it) becomes a
  // position the floor report can read later instead of trying to guess it.
  const insideOut: number[] = [];
  messages.forEach((m, i) => {
    // Inject the summary in place of the first compressed message.
    const startsSpan = resolved.find((r) => r.from === i);
    if (startsSpan) {
      // `null` for every pit leaf but the first: the riassuntone already stands for it.
      const inj = injected[resolved.indexOf(startsSpan)];
      if (inj) { insideOut.push(kept.length); kept.push(inj); }
    }
    if (!replaced.has(i)) { kept.push(m); return; }
    // A span covers every index between its two endpoints, whatever their role,
    // while the endpoints themselves are always user/assistant messages. So a
    // user turn (or a system/developer message, or another extension's custom
    // content) can sit inside the range: dropping it would delete instructions
    // the main eviction path protects explicitly ("Principle 3: user turns are
    // inviolable"). Keep them; only the unprotected content is replaced.
    // SAFETY: read-only probe of an optional field, undefined for other roles.
    const role = (m as unknown as RealMessage).role;
    if (keptInsideSpan(role)) {
      insideOut.push(kept.length);
      kept.push(m);
    }
  });

  // The pair invariant is restored by id on the final list: the range
  // arithmetic above can split a call from its result, and position cannot be
  // trusted to detect every layout.
  const repaired = repairToolPairs(kept);
  return {
    kept: out(repaired.kept),
    applied: resolved.length,
    saved,
    newApplied,
    newSaved,
    pairDropped: repaired.dropped,
    pairStripped: repaired.stripped,
    insideOut: repaired.dropped === 0 ? insideOut : [],
    dead,
    overlapped,
  };
}

export default function (pi: ExtensionAPI) {
  // ---- Tools -------------------------------------------------------------

  pi.registerTool({
    name: 'delimiter',
    label: 'CWL Delimiter',
    description:
      'Marks the boundaries of a CWL episode. Types: "expl" (exploration: searches, ' +
      'reads, orientation — the content is not needed after the inference) and "act" ' +
      '(action: writes, edits, executions — persistent effects, first candidate ' +
      'for eviction). When you open an "act", declare the explorations it depends on. ' +
      'When you close an "expl", give the description of what you learned: ' +
      'it is the only content that survives eviction.',
    promptSnippet: t('snippets').delimiter,
    promptGuidelines: I18N[LANG].guidelines,
    parameters: Type.Object({
      action: Type.Union([Type.Literal('start'), Type.Literal('end')], {
        description: t('params').action,
      }),
      name: Type.Optional(Type.String({
        description: t('params').name,
      })),
      type: Type.Optional(Type.Union([Type.Literal('expl'), Type.Literal('act')], {
        description: t('params').type,
      })),
      dependencies: Type.Optional(Type.Array(Type.String(), {
        description: t('params').dependencies,
      })),
      description: Type.Optional(Type.String({
        description: t('params').description,
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
            details: { ok: false, error: 'missing-params' },
          };
        }
        if (st.graph.has(params.name)) {
          return {
            content: [{ type: 'text', text: t('duplicateName')(params.name) }],
            details: { ok: false, error: 'duplicate-name' },
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
            details: { ok: false, error: 'invalid-dependencies' },
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
            details: { ok: false, error: 'missing-params' },
          };
        }
        const ep = st.graph.close(params.name, params.description ?? '', st.messageCursor, toolCallId);
        if (ep) saveState(key, st);
        if (!ep) {
          return {
            content: [{ type: 'text', text: t('notFoundOrClosed')(params.name) }],
            details: { ok: false, error: 'episode-not-found' },
          };
        }
        debugLog(cf, `CLOSE ${ep.type} "${ep.name}" level=${ep.level} endIdx=${ep.endIdx}`);

        let msg = t('episodeClosed')(ep.type, ep.name);
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
    description: t('params').statusDesc,
    promptSnippet: t('snippets').status,
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
        t('statusMeasured')(st.lastMeasuredTokens.toLocaleString()),
        t('statusEpisodes')(g.count, active.length, closed.length, stripped.length),
        t('statusEvictions')(st.totalEvictions, st.totalEvictedTokens.toLocaleString()),
      ];
      // The range cwl_compress_range would take right now: shown so the operator
      // can see WHAT the extension would compress before it is gone.
      if (st.rangeStartHash && st.rangeEndHash) {
        lines.push(t('statusRange')(st.rangeTokens, st.rangeStartHash, st.rangeEndHash));
      }
      // An address must be UNIQUE, or a span resolves on the wrong message. These
      // two numbers are the measurement of that, not a promise.
      if (st.addrEligible > 0) {
        lines.push(t('statusAddresses')(st.addrEligible, st.addrWithId));
      }
      lines.push(t('statusSpans')(st.spans.length));
      if (st.unlocatable > 0) {
        lines.push(t('statusUnlocatable')(st.unlocatable));
      }
      if (active.length > 0) {
        lines.push(t('statusActive')(active.map(e => `${e.name}(${e.type})`).join(', ')));
      }
      if (stripped.length > 0) {
        lines.push(t('statusStripped')(stripped.map(e => `${e.name}[${e.level}]`).join(', ')));
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
          addrEligible: st.addrEligible,
          addrWithId: st.addrWithId,
          spans: st.spans.length,
          unlocatable: st.unlocatable,
        },
      };
    },
  });

  pi.registerTool({
    name: 'cwl_compress',
    label: 'CWL Compress',
    description:
      t('tools').compressDesc,
    promptSnippet: t('snippets').compress,
    parameters: Type.Object({
      startHash: Type.String({ description: t('tools').compressStart, }),
      endHash: Type.String({ description: t('tools').compressEnd, }),
      summary: Type.String({
        description:
          t('tools').compressSummary,
      }),
      micro: Type.Optional(Type.String({ description: t('tools').compressMicro })),
    }),
    async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
      const key = sessionKey(ctx);
      const st = getState(key);
      const cf = getConfig(key);

      if (!params.startHash || !params.endHash || !params.summary) {
        return { content: [{ type: 'text', text: t('compressMissingParams') }], details: { ok: false, error: 'missing-params' } };
      }
      const seen = st.knownHashes;
      // Revoke means "undo a span I already asked for": it must be decided on
      // st.spans, NOT on knownHashes. knownHashes holds the hash of every
      // user/assistant message in the context, so a *valid* request (both
      // endpoints are real messages) always matched the old guard and was
      // always answered with a revoke — the push below was unreachable, and the
      // compression feature was dead while reporting success.
      const alreadyCompressed = st.spans.some(
        (sp) => sp.startHash === params.startHash && sp.endHash === params.endHash,
      );
      if (alreadyCompressed) {
        st.spans = st.spans.filter((sp) => !(sp.startHash === params.startHash && sp.endHash === params.endHash));
        saveState(key, st);
        debugLog(cf, `COMPRESS revoked ${params.startHash}..${params.endHash}`);
        return { content: [{ type: 'text', text: t('compressRevoked')(params.startHash, params.endHash) }], details: { ok: true, revoked: true } };
      }
      const from = seen.has(params.startHash);
      const to = seen.has(params.endHash);
      if (!from || !to) {
        return {
          content: [{ type: 'text', text: t('compressUnknownHash')(from, to) }],
          details: { ok: false, error: 'unknown-hash', startFound: from, endFound: to },
        };
      }
      st.spans.push({
        startHash: params.startHash,
        endHash: params.endHash,
        id: spanId(params.startHash, params.endHash),
        summary: params.summary,
        micro: microOrUndefined(params.micro),
        at: Date.now(),
      });
      // Durable immediately: the agent asked for this compression, so it must
      // survive a restart even if the process is killed before the next turn ends.
      saveState(key, st);
      debugLog(cf, `COMPRESS applied ${params.startHash}..${params.endHash}`);
      return {
        content: [{ type: 'text', text: t('compressApplied')(params.startHash, params.endHash) }],
        details: { ok: true, spans: st.spans.length },
      };
    },
  });

  pi.registerTool({
    name: 'cwl_compress_range',
    label: 'CWL Compress Range',
    description:
      t('tools').compressRangeDesc,
    promptSnippet: t('snippets').compressRange,
    parameters: Type.Object({
      summary: Type.String({ description: t('tools').compressRangeSummary }),
      micro: Type.Optional(Type.String({ description: t('tools').compressMicro })),
    }),
    async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
      const key = sessionKey(ctx);
      const st = getState(key);
      const cf = getConfig(key);

      if (!params.summary || !params.summary.trim()) {
        return { content: [{ type: 'text', text: t('compressRangeNoSummary') }], details: { ok: false, error: 'missing-summary' } };
      }
      // The address comes from the LAST context hook, which is the state the
      // agent was looking at when it decided to call this. Deliberately not
      // recomputed here: this tool has no access to the message list, and an
      // address guessed from stale data is exactly the failure it replaces.
      if (!st.rangeStartHash || !st.rangeEndHash) {
        return { content: [{ type: 'text', text: t('compressRangeNothing') }], details: { ok: false, error: 'nothing-to-compress' } };
      }
      const startHash = st.rangeStartHash;
      const endHash = st.rangeEndHash;
      const tokens = st.rangeTokens;
      st.spans.push({ startHash, endHash, id: spanId(startHash, endHash), summary: params.summary, micro: microOrUndefined(params.micro), at: Date.now() });
      // Spend the address: the next hook recomputes it on the smaller list, so a
      // second call cannot compress the same range twice.
      st.rangeStartHash = null;
      st.rangeEndHash = null;
      st.rangeTokens = 0;
      saveState(key, st);
      debugLog(cf, `COMPRESS-RANGE applied ${startHash}..${endHash} (~${tokens}t)`);
      return {
        content: [{ type: 'text', text: t('compressRangeApplied')(startHash, endHash, tokens) }],
        details: { ok: true, spans: st.spans.length, tokens },
      };
    },
  });

  pi.registerTool({
    name: 'cwl_recall',
    label: 'CWL Recall',
    description:
      t('tools').recallDesc,
    promptSnippet: t('snippets').recall,
    parameters: Type.Object({
      query: Type.String({ description: t('tools').recallQuery, }),
      limit: Type.Optional(Type.Number({ description: t('tools').recallLimit, minimum: 1, maximum: 50 })),
    }),
    async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
      const key = sessionKey(ctx);
      if (!recall) {
        return { content: [{ type: 'text', text: t('recallNotLoaded') }], details: { ok: false, error: 'not-loaded' } };
      }
      const file = findTranscript(key);
      if (!file) {
        return { content: [{ type: 'text', text: t('recallNoTranscript') }], details: { ok: false, error: 'no-transcript' } };
      }
      const raw = readFileOrNull(file);
      if (raw === null) {
        return { content: [{ type: 'text', text: t('recallUnreadable') }], details: { ok: false, error: 'unreadable' } };
      }
      const st = getState(key);
      // Incremental update: pass the existing index back in. Building it once and
      // freezing it (the previous `?? recall.indexTranscript(raw, null)`) meant
      // the index never saw a message written after the first recall — exactly
      // the content most likely to be needed — while reporting a stale size.
      // indexTranscript already supports this: it reuses `existing` and only
      // adds the records it has not seen.
      const idx = recall.indexTranscript(raw, st.recallIndex);
      st.recallIndex = idx;
      const hits = idx.search(params.query, params.limit ?? 5);
      if (hits.length === 0) {
        return { content: [{ type: 'text', text: t('recallNoMatch')(params.query) }], details: { ok: true, hits: 0, indexed: idx.size } };
      }
      const body = hits.map((h, i) => `[${i + 1}] ${h.role} score=${h.score.toFixed(2)}\n${h.preview}`).join('\n\n');
      return {
        content: [{ type: 'text', text: t('recallFound')(hits.length, idx.size, params.query, body) }],
        details: { ok: true, hits: hits.length, indexed: idx.size, ids: hits.map((h) => h.id) },
      };
    },
  });

  pi.registerTool({
    name: 'cwl_recall_episode',
    label: 'CWL Recall Episode',
    description:
      t('tools').recallEpisodeDesc,
    promptSnippet: t('snippets').recallEpisode,
    parameters: Type.Object({
      name: Type.String({ description: t('tools').recallEpisodeName }),
      full: Type.Optional(Type.Boolean({ description: t('tools').recallEpisodeFull })),
    }),
    async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
      const key = sessionKey(ctx);
      const st = getState(key);
      // The name is the address: the state is per session, so it already scopes
      // the lookup. No index is used, because indices shift as the context is
      // evicted — the toolCallId anchors do not.
      const ep = st.graph.all.find((e) => e.name === params.name);
      if (!ep) {
        return { content: [{ type: 'text', text: t('episodeRecallNotKnown')(params.name) }], details: { ok: false, error: 'episode-not-found' } };
      }
      if (ep.endToolCallId === null) {
        return { content: [{ type: 'text', text: t('episodeRecallStillOpen')(params.name) }], details: { ok: false, error: 'episode-still-open' } };
      }
      const file = findTranscript(key);
      if (!file) {
        return { content: [{ type: 'text', text: t('recallNoTranscript') }], details: { ok: false, error: 'no-transcript' } };
      }
      const raw = readFileOrNull(file);
      if (raw === null) {
        return { content: [{ type: 'text', text: t('recallUnreadable') }], details: { ok: false, error: 'unreadable' } };
      }
      const text = extractEpisodeText(raw, ep.startToolCallId, ep.endToolCallId);
      if (text === null) {
        return { content: [{ type: 'text', text: t('episodeRecallAnchorLost')(params.name) }], details: { ok: false, error: 'anchor-lost' } };
      }
      if (!text.trim()) {
        return { content: [{ type: 'text', text: t('episodeRecallEmpty')(params.name) }], details: { ok: false, error: 'empty' } };
      }
      const full = params.full === true;
      const tokens = estimateTokens(text);
      const body = full ? text : text.slice(0, EPISODE_PREVIEW_CHARS) + t('episodeRecallTruncatedHint');
      return {
        content: [{ type: 'text', text: t('episodeRecallFound')(params.name, tokens, body) }],
        details: { ok: true, episode: params.name, tokens, chars: text.length, full },
      };
    },
  });

  /**
   * cwl_open: read back a compressed span (a "leaf") IN FULL.
   *
   * A summary lives ONLY in the state — the transcript holds the ORIGINAL
   * messages, not the summaries — so this is the only way to see again what a
   * compression set aside, without re-reading the source.
   *
   * It does NOT truncate, on purpose: the operator's models have 1M-token
   * windows, and an agent that wants to see something must SEE it. The tool's
   * only duty is to declare the size BEFORE handing it over, so the reader knows
   * what it is about to pay.
   */
  pi.registerTool({
    name: 'cwl_open',
    label: 'CWL Open',
    description: t('tools').openDesc,
    parameters: Type.Object({
      id: Type.String({ description: t('tools').openId }),
    }),
    async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
      const key = sessionKey(ctx);
      const st = getState(key);
      const cf = getConfig(key);
      const wanted = String(params.id ?? '').trim();
      // The OLD node first: its page is the riassuntone plus what it holds — one line
      // per young node inside, with the SHAPE (how many leaves, and the first and last
      // leaf id), so a reader can choose what to open without opening it.
      if (st.oldNode && st.oldNode.id === wanted) {
        const inPit = new Set(st.oldNode.nodes);
        const righe = st.nodes
          .filter((nd) => inPit.has(nd.id))
          .map((nd) => `- ${nd.id}: ${nd.leaves.length} leaf/leaves (${nd.leaves[0]} .. ${nd.leaves[nd.leaves.length - 1]})`)
          .join('\n');
        // The page's second half: the leaves CONSULTED, most opened first. The rule the
        // operator asked for — not "the most recent", which are still in the young nodes
        // and in the loose leaves anyway (their reasoning trail is already in the
        // context), but the ones that LOOK stale and are still being reached for.
        // With an empty history the order degrades to the most recently absorbed, which
        // is a sane default rather than a criterion. Every one is cap 30, like a node.
        const pitLeaves = st.nodes
          .filter((nd) => inPit.has(nd.id))
          .flatMap((nd) => nd.leaves)
          .map((id) => st.spans.find((s) => idOfSpan(s) === id))
          .filter((s): s is CompressedSpan => Boolean(s));
        const hot = [...pitLeaves]
          .sort((a, b) => (b.opens ?? 0) - (a.opens ?? 0) || (b.lastOpen ?? 0) - (a.lastOpen ?? 0) || b.at - a.at)
          .slice(0, NODE_PAGE_MAX);
        const hotLines = hot
          .map((s) => `- ${idOfSpan(s)} (opened ${s.opens ?? 0}x): ${s.micro ?? '(no micro yet)'}`)
          .join('\n');
        const body = `${st.oldNode.summary}\n\n${righe}\n\n${t('oldHot')(hot.length, pitLeaves.length)}\n${hotLines}`;
        const tokens = estimateTokens(body);
        debugLog(cf, `OPEN ${st.oldNode.id}: riassuntone + ${st.oldNode.nodes.length} node(s), ${pitLeaves.length} leaf/leaves inside, most opened ${hot[0]?.opens ?? 0}x (${hot.filter((s) => (s.opens ?? 0) > 0).length} ever opened) — ${tokens}t`);
        return {
          content: [{ type: 'text', text: t('oldPage')(st.oldNode.id, st.oldNode.nodes.length, tokens, body) }],
          details: { ok: true, id: st.oldNode.id, kind: 'old', nodes: st.oldNode.nodes.length, leaves: pitLeaves.length, opened: hot.filter((s) => (s.opens ?? 0) > 0).length, tokens },
        };
      }
      // A NODE next: its page is the micros of its leaves, each with the id that
      // opens it. Same rule as a leaf: the size is declared before it is handed over.
      const nd = st.nodes.find((n) => n.id === wanted);
      if (nd) {
        const lines = nd.leaves
          .map((leafId) => {
            const leaf = st.spans.find((s) => idOfSpan(s) === leafId);
            return leaf ? `- ${leafId}: ${leaf.micro ?? '(no micro yet)'}` : `- ${leafId}: (leaf gone)`;
          })
          .join('\n');
        const tokens = estimateTokens(lines);
        debugLog(cf, `OPEN ${nd.id}: ${nd.leaves.length} leaf/leaves, ${tokens}t of micros`);
        return {
          content: [{ type: 'text', text: t('nodePage')(nd.id, nd.leaves.length, tokens, lines) }],
          details: { ok: true, id: nd.id, kind: 'node', leaves: nd.leaves.length, tokens, chars: lines.length },
        };
      }
      // Spans written before ids existed have none in the persisted state:
      // `idOfSpan` derives it, exactly as the injected notice does.
      const sp = st.spans.find(
        (s) => idOfSpan(s) === wanted || s.startHash === wanted || s.endHash === wanted,
      );
      if (!sp) {
        // Not in the state. Maybe it was DROPPED — native compaction took its anchors —
        // and then the summary is gone for good while the ORIGINAL is still in the
        // append-only transcript. The graveyard kept the stable ids that find it: the id
        // itself could not, being a hash of the anchors.
        const grave = st.graves.find((g) => g.id === wanted);
        if (grave) {
          const transcriptPath = findTranscript(key);
          const raw = transcriptPath ? readFileOrNull(transcriptPath) : null;
          const original = raw ? transcriptRangeText(raw, grave.startSid, grave.endSid) : null;
          if (original && original.trim()) {
            const tokens = estimateTokens(original);
            debugLog(cf, `OPEN ${wanted}: ORIGINAL from the transcript, ${tokens}t (the summary went with the state)`);
            return {
              content: [{ type: 'text', text: t('openOriginal')(wanted, tokens, original) }],
              details: { ok: true, id: wanted, kind: 'original', dropped: true, tokens, chars: original.length },
            };
          }
          // Declared, never silent: the ids were there and found nothing, so the content
          // is only reachable by MEANING from here on.
          return {
            content: [{ type: 'text', text: t('openOriginalLost')(wanted) }],
            details: { ok: false, error: 'original-lost', id: wanted, kind: 'original' },
          };
        }
        return {
          content: [{ type: 'text', text: t('openMissing')(wanted) }],
          details: { ok: false, error: 'unknown-id', id: wanted, spans: st.spans.length },
        };
      }
      const id = idOfSpan(sp);
      const body = sp.summary;
      const tokens = estimateTokens(body);
      const when = new Date(sp.at).toISOString().slice(0, 16).replace('T', ' ');
      // Usage, not policy: this only orders the pit's page (see the branch above).
      sp.opens = (sp.opens ?? 0) + 1;
      sp.lastOpen = Date.now();
      saveState(key, st);
      debugLog(cf, `OPEN ${id}: ${tokens}t of body, ${body.length} chars (open #${sp.opens})`);
      return {
        content: [{ type: 'text', text: t('openFound')(id, tokens, when) + body }],
        details: { ok: true, id, tokens, chars: body.length },
      };
    },
  });

  /**
   * cwl_micro: absorb a leaf — its body leaves the context, its micro stays.
   *
   * This is the lever the whole project started from. MEASURED in a live session:
   * 52.027t of the context were the extension's OWN summaries (30 of them, mean
   * 1.707t) and no path could touch them, because `keptInsideSpan` and the eviction
   * applier both keep `role: 'custom'` — the role a summary is injected with. The
   * extension could WRITE a summary and could not ABSORB an old one.
   *
   * The body is never overwritten: the micro is a separate field, so `cwl_open`
   * keeps returning all of it. A tool that wrote the micro INTO the body would make
   * the promise "nothing is lost" false with one line.
   *
   * An EMPTY text removes the micro and puts the whole body back in the context:
   * an absorption nobody can undo is a one-way door, and this one has a way back.
   */
  pi.registerTool({
    name: 'cwl_micro',
    label: 'CWL Micro',
    description: t('tools').microDesc,
    parameters: Type.Object({
      id: Type.String({ description: t('tools').microId }),
      text: Type.String({ description: t('tools').microText }),
    }),
    async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
      const key = sessionKey(ctx);
      const st = getState(key);
      const cf = getConfig(key);
      const wanted = String(params.id ?? '').trim();
      const micro = String(params.text ?? '').trim();
      const sp = st.spans.find(
        (s) => idOfSpan(s) === wanted || s.startHash === wanted || s.endHash === wanted,
      );
      if (!sp) {
        return {
          content: [{ type: 'text', text: t('openMissing')(wanted) }],
          details: { ok: false, error: 'unknown-id', id: wanted, spans: st.spans.length },
        };
      }
      const id = idOfSpan(sp);
      sp.micro = micro ? micro : null;
      saveState(key, st);
      const shorter = micro.length > 0 && micro.length < sp.summary.length;
      debugLog(cf, micro
        ? `MICRO ${id}: ${micro.length} chars in place of ${sp.summary.length}${shorter ? '' : ' — NOT SHORTER, the context does not shrink'}`
        : `MICRO ${id}: removed, the body is back in the context`);
      return {
        content: [{ type: 'text', text: t('microSet')(id, micro.length, sp.summary.length, shorter) }],
        details: { ok: true, id, microChars: micro.length, bodyChars: sp.summary.length, shorter },
      };
    },
  });

  /**
   * cwl_old: writes the RIASSUNTONE and merges the oldest young nodes into the old
   * node — the one place where this index SAVES something the extension could not
   * touch before.
   *
   * MEASURED in a live session: 52.027t of the context were the extension's OWN
   * summaries (30 leaves, mean 1.707t), and every one of them was protected by rules
   * written for another extension's content — `keptInsideSpan` and the eviction
   * applier both keep `role: 'custom'`, which is the role a summary is injected with.
   * Sixty micros become ONE synthesis.
   *
   * The extension does not write it: it has no model. It says the merge is due; the
   * agent writes the synthesis; this tool records it. The nodes stay in the state and
   * their leaves stay readable — what they leave is the CONTEXT.
   */
  pi.registerTool({
    name: 'cwl_old',
    label: 'CWL Old Node',
    description: t('tools').oldDesc,
    parameters: Type.Object({
      text: Type.String({ description: t('tools').oldText }),
    }),
    async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
      const key = sessionKey(ctx);
      const st = getState(key);
      const cf = getConfig(key);
      const text = String(params.text ?? '').trim();
      if (!text) {
        return {
          content: [{ type: 'text', text: t('tools').oldText }],
          details: { ok: false, error: 'missing-text' },
        };
      }
      refreshNodes(st, cf);
      const inPit = new Set(st.oldNode?.nodes ?? []);
      const young = st.nodes.filter((nd) => !inPit.has(nd.id));
      if (young.length < cf.mergeNodesAt) {
        return {
          content: [{ type: 'text', text: t('oldNotDue')(young.length, cf.mergeNodesAt) }],
          details: { ok: false, error: 'not-due', young: young.length, mergeNodesAt: cf.mergeNodesAt },
        };
      }
      // All but the NEWEST young node: the newest is the one still filling up, and
      // the pit is where the old material goes.
      const absorbed = young.slice(0, young.length - 1);
      const ids = absorbed.flatMap((nd) => nd.leaves);
      const microChars = ids.reduce(
        (n, id) => n + (st.spans.find((s) => idOfSpan(s) === id)?.micro?.length ?? 0),
        0,
      );
      const pit: OldNode = st.oldNode ?? {
        id: `old-${hashText(absorbed[0].id).slice(0, 8)}`,
        nodes: [],
        summary: '',
        at: Date.now(),
      };
      pit.nodes.push(...absorbed.map((nd) => nd.id));
      pit.summary = text;
      pit.at = Date.now();
      st.oldNode = pit;
      saveState(key, st);
      const tokens = estimateTokens(text);
      debugLog(cf, `OLD ${pit.id}: absorbed ${absorbed.length} node(s), ${ids.length} leaf/leaves; ${microChars} chars of micros -> a synthesis of ${tokens}t`);
      return {
        content: [{ type: 'text', text: t('oldSet')(pit.id, absorbed.length, ids.length, microChars, tokens) }],
        details: { ok: true, id: pit.id, nodes: absorbed.length, leaves: ids.length, microChars, tokens },
      };
    },
  });

  // ---- Hooks -------------------------------------------------------------

  pi.on('session_start', async (_event, ctx) => {
    const key = sessionKey(ctx);
    // Do NOT wipe the state: getState resumes the PERSISTED decision (episode
    // graph + traced compressions) when a file exists for this session, and
    // creates a fresh state only when it does not. The previous
    // `states.set(key, newState())` threw away the whole graph on every start.
    const st = getState(key);
    configs.set(key, loadConfig());
    pruneStateFiles();

    // recall.mjs: BM25 indexer for the transcript. Same cache-busting as the
    // helper: static imports would stay cached after /reload.
    try {
      const recallUrl = new URL('./recall.mjs', import.meta.url);
      const src = readFileOrNull(fileURLToPath(recallUrl));
      if (src !== null) {
        recallUrl.searchParams.set(
          'version',
          createHash('sha256').update(src).digest('hex').slice(0, 16),
        );
        recall = await import(recallUrl.href) as typeof Recall;
      }
    } catch (err) {
      debugLog(getConfig(key), `recall.mjs not loaded: ${String(err)}`);
    }

    debugLog(getConfig(key), `SESSION START — ${st.graph.count} episodes, ${st.spans.length} spans resumed`);
  });

  /**
   * Eviction lives in the 'context' hook: it is the only point where Pi lets you
   * REWRITE the message list that will be sent to the provider. turn_end can
   * only observe, so computing there would never have reduced anything.
   */
  pi.on('context', async (event, ctx) => {
    const key = sessionKey(ctx);
    const st = getState(key);
    const cf = getConfig(key);

    const eventMessages: AgentMessage[] = event.messages;
    if (!eventMessages || eventMessages.length === 0) return;

    // Drop the demand injected on the PREVIOUS call before anything else: it is
    // re-added only while the gate is armed, so it can never pile up turn after
    // turn. Measuring AFTER this also keeps the demand from inflating the very
    // number it is about.
    // `let`, not `const`: the span branch below REASSIGNS it to the COMPRESSED
    // list when the spans did not bring the context back under the trigger, so
    // that the rest of the hook (trigger check, `reasoningFallback`,
    // `runEvictionPass`) runs on the list the agent will really see. Every
    // index-based decision below — the episode ranges and `safetyFloor` — is
    // resolved against whatever list this variable holds, so the two must never
    // disagree.
    let messages: AgentMessage[] = eventMessages.filter((m) => !isGateMessage(m));

    // Diagnostic, and free: it only reads a field, it does not hash. It answers
    // ONE question without a debug log — do real messages carry a stable id? —
    // which decides whether addressOf can tell two identical texts apart.
    st.addrEligible = 0;
    st.addrWithId = 0;
    for (const m of messages) {
      // SAFETY: read-only field probe (role); the union does not expose it.
      const role = (m as unknown as RealMessage).role;
      if (role !== 'user' && role !== 'assistant') continue;
      if (!textOf(m).trim()) continue;
      st.addrEligible++;
      if (stableIdOf(m)) st.addrWithId++;
    }
    const droppedGate = messages.length !== eventMessages.length;

    // Update the cursor with the REAL message count (not turns).
    // Tool results are appended during the turn, so the message count is
    // the only reliable basis for episode indices.
    st.messageCursor = messages.length; // recomputed, not accumulated

    // Hash anchoring: the LLM points at the messages to compress by the hash
    // of their text. Indices shift every turn, hashes do not.
    for (const m of messages) {
      // SAFETY: read-only probe of an optional field, undefined for other roles.
      const role = (m as unknown as RealMessage).role;
      if (role !== 'user' && role !== 'assistant') continue;
      st.knownHashes.add(hashText(textOf(m)));
    }

    const g = st.graph;

    // 1. Measure the real context
    let currentTokens = 0;
    for (const m of messages) {
      currentTokens += estimateMessageTokens(m);
    }
    st.lastMeasuredTokens = currentTokens;

    const trigger = cf.tokenBudget * cf.thresholdRatio;

    /**
     * Says OUT LOUD why the context is still above the trigger, with four numbers
     * that ADD UP to the active context exactly: it is arithmetic, not a story.
     *
     * The gate already refuses to demand what it cannot deliver (it tests
     * `canClose`/`canCompress`); what it never said is WHY there is nothing left to
     * deliver. MEASURED on a real session: 184.490t against a 68.000t trigger with
     * only 6.307t of compressible range — the last 10 user turns (the window the
     * operator asked for) and the content the spans keep hold everything else. An
     * operator reading "over budget" cannot tell a broken extension from an
     * inviolable context, and that ambiguity is what this row removes.
     *
     * `inside the spans` is what a span KEEPS in place — the operator's turns, the
     * system/developer directives, other extensions' custom messages. It still
     * costs tokens, and `compressibleRange` treats a span's whole interval as
     * covered, so a later compression can never reach it.
     */
    const declareFloor = (list: AgentMessage[], tokens: number): void => {
      if (tokens <= trigger) return;
      const floor = protectedFromIndex(list, cf.protectedTurns);
      // The spans recorded where their content ended up, in the list they built
      // (see applySpans.insideOut). Re-resolving is impossible here: this list has
      // already been through the spans, so a span whose END anchor was an
      // assistant does not contain that anchor any more — the span removed it.
      // MEASURED in a live session: `0t inside the spans` with 27 spans applied,
      // printed as `0 of 27 spans located here`.
      // So the recorded positions are used when the list still has the same
      // LENGTH they were recorded against, and the row says which of the two
      // sources it counted from — never a bare zero.
      const record = spanInside && spanInside.len === list.length ? spanInside.set : null;
      const located = record ? [] : locateSpans(list, st.spans).resolved;
      const inSpans = record ?? new Set<number>();
      if (!record) {
        for (const { from, to } of located) {
          for (let i = from; i <= to; i++) inSpans.add(i);
        }
      }
      const countedFrom = record ? `counted from ${st.spans.length} of ${st.spans.length} spans` : `counted from ${located.length} of ${st.spans.length} spans`;
      let protectedTokens = 0;
      let spanTokens = 0;
      let outside = 0;
      // What the spans are actually HOLDING, by kind. Inside a span only
      // user/system/developer/custom survive (plus the injected summary), and a
      // user turn is inviolable (Principle 3): so this breakdown decides whether
      // the `covered` set is blocking anything that COULD be compressed at all,
      // or whether the space inside the spans is unrecoverable by construction.
      // The total alone cannot answer that: `48623t inside the spans` was
      // MEASURED on a real session and says nothing about whose text it is.
      let spanSummaries = 0;
      let spanUser = 0;
      let spanOther = 0;
      list.forEach((m, i) => {
        const t = estimateMessageTokens(m);
        if (i >= floor) protectedTokens += t;
        else if (inSpans.has(i)) {
          spanTokens += t;
          // SAFETY: read-only probe of an optional field; the union does not
          // declare `role`, and `undefined` simply falls through to `other`.
          const role = (m as unknown as RealMessage).role;
          // SAFETY: same probe for `customType`, which only custom messages
          // carry: reading it as unknown and comparing it to a string cannot
          // throw, and anything else lands in `other`.
          const customType = (m as unknown as { customType?: unknown }).customType;
          if (customType === 'cwl-compressed') spanSummaries += t;
          else if (role === 'user') spanUser += t;
          else spanOther += t;
        } else outside += t;
      });
      // `free` is a SUBSET of `outside` (below the floor and not covered by a
      // span), so it is subtracted from it: the four numbers must partition the
      // context EXACTLY, and the test checks that identity.
      const free = Math.min(st.rangeTokens, outside);
      debugLog(cf, `CONTEXT ${tokens}t still above trigger ${Math.round(trigger)}t: ${protectedTokens}t in the protected window (last ${cf.protectedTurns} user turns), ${spanTokens}t inside the spans (${countedFrom}), ${free}t freely compressible, ${outside - free}t elsewhere`);
      // The three parts sum to `spanTokens` BY CONSTRUCTION, and the test checks
      // that against the row above: two lines, one number, and a sum that cannot
      // be talked around. Without this, "48.623t inside the spans" is a number
      // that cannot decide anything.
      if (spanTokens > 0) {
        debugLog(cf, `SPANS content: ${spanTokens}t inside the spans = ${spanSummaries}t of summaries + ${spanUser}t of user turns + ${spanOther}t of other roles`);
      }
    };

    /**
     * Final step of the hook: decides whether to ask the AGENT to compact.
     *
     * The gate is verified by EFFECT — the measured context — not by the agent's
     * word. anti-amnesia asks for a `[CARD OK]` token to be echoed back; here the
     * thing we want (fewer tokens) is directly measurable, so it is measured, and
     * a hallucinated confirmation earns nothing.
     */
    /**
     * Durability without side effects.
     *
     * The hook must persist what cannot be recomputed — the stable ids written when a span
     * first resolves, the graveyards the prune records — or a restart throws them away.
     * But a session that never uses CWL must leave NOTHING on disk, and that is a tested
     * contract (`tests/state-persistence`). The state file existing is exactly that
     * difference: the tools write it the first time the agent compresses anything, and
     * from then on the hook keeps it fresh. Saving unconditionally broke that contract the
     * first time it ran, which is why the condition is written down here and not assumed.
     */
    const persistIfUsed = (): void => {
      if (fs.existsSync(statePath(key))) saveState(key, st);
    };

    const finish = (list: AgentMessage[]): { messages: AgentMessage[] } => {
      const after = list.reduce((s: number, m: AgentMessage) => s + estimateMessageTokens(m), 0);
      st.lastMeasuredTokens = after;
      persistIfUsed();
      if (after <= trigger) {
        // Effect achieved: the gate has nothing left to ask for.
        st.overBudgetSince = -1;
        st.gateArmedTurn = -1;
        return { messages: list };
      }
      // Still over budget. Remember since when, so turn_end knows when to ask.
      if (st.overBudgetSince < 0) st.overBudgetSince = st.turns;
      declareFloor(list, after);
      if (!cf.gate || st.gateArmedTurn < 0) return { messages: list };
      // Only demand what the extension can actually deliver. In a real session the
      // gate asked to compact while ALL four episodes were already closed and
      // cwl_compress_range answered "nothing left to compress": it demanded the
      // impossible once per turn, burning the very context it was trying to save.
      const canClose = st.graph.active().length > 0;
      const canCompress = st.rangeStartHash !== null;
      if (!canClose && !canCompress) return { messages: list };
      // SAFETY: Pi accepts the custom role in the context hook although the
      // AgentMessage union does not declare it; the extra keys are its contract.
      return {
        messages: [...list, {
          role: 'custom',
          customType: GATE_CUSTOM_TYPE,
          content: t('gateDemand')(
            after.toLocaleString(),
            Math.round(trigger).toLocaleString(),
            cf.protectedTurns,
            canClose,
            canCompress,
          ),
          display: false,
          timestamp: Date.now(),
        } as unknown as AgentMessage],
      };
    };

    // Episodes the eviction can no longer locate, said out loud instead of
    // assumed. Their delimiters are gone from the list (native compaction
    // replaced that history), so the eviction skips them — correctly: without
    // anchors it cannot know what to touch. But a `removed` one is ALSO excluded
    // from `recoverable()`, so the state claims it was evicted while nobody can
    // verify its content is gone. MEASURED in a live session: 4 of 4.
    const closedEps = st.graph.closed();
    if (closedEps.length > 0) {
      const locatable = episodeRanges(messages, closedEps);
      st.unlocatable = closedEps.filter((ep) => !locatable.has(ep.name)).length;
      st.deduced = [...locatable.values()].filter((r) => r.deduced).length;
      if (st.unlocatable !== st.unlocatableSeen) {
        st.unlocatableSeen = st.unlocatable;
        debugLog(cf, `EPISODES unlocatable: ${st.unlocatable} of ${closedEps.length} (their anchors left the context)`);
      }
      if (st.deduced !== st.deducedSeen) {
        st.deducedSeen = st.deduced;
        debugLog(cf, `EPISODES deduced: ${st.deduced} episode(s) lost their START anchor to a native compaction (the cut takes a prefix); their range was derived, not read`);
      }
    } else {
      st.unlocatable = 0;
      st.deduced = 0;
    }

    // Set when the span branch renewed the stored range on the ORIGINAL list:
    // the tail must not recompute it on the compressed one, where the span
    // endpoints no longer exist and `covered` would come out empty — the same
    // region would be offered again.
    let rangeStoredBySpans = false;

    // Where the spans put their content, in the list they built: `applySpans`
    // records it while it pushes and nobody can recover it afterwards (a span
    // removes its own closing anchor, so a second resolution is blind). `len`
    // guards the positions: the reasoning strip rewrites message for message and
    // keeps the length, the eviction removes messages and changes it.
    let spanInside: { set: Set<number>; len: number } | null = null;

    // The spans compressed by the LLM are ALWAYS applied, not only above the
    // threshold: the agent decides when to compress, not the extension estimate.
    // The INDEX runs here, BEFORE the spans are applied: which leaves a node owns, how
    // many are still waiting for a micro, and whether a merge is due. Before, because
    // the injection must see the structure of THIS turn — running it in `finish` meant
    // the injection worked on the previous turn's nodes. It is idempotent: it moves
    // nothing that is already owned. What changes is said out loud; a line per turn
    // would be noise, and noise hides bugs.
    const nodiPrima = st.nodes.map((nd) => `${nd.id}:${nd.leaves.length}`).join(',');
    const piano = refreshNodes(st, cf);
    const nodiDopo = st.nodes.map((nd) => `${nd.id}:${nd.leaves.length}`).join(',');
    if (nodiDopo !== nodiPrima || piano.waiting > 0) {
      debugLog(cf, `NODES: ${st.nodes.length} node(s) [${nodiDopo || 'none'}]${piano.formed > 0 ? `, formed ${piano.formed}` : ''}, ${piano.waiting} leaf/leaves waiting for a micro — their body is still in the context`);
    }
    // The merge needs a synthesis only the AGENT can write. Until this demand existed it
    // reached the log and not the agent: the extension knew, the operator could read it,
    // and the only one able to write the riassuntone never called cwl_old. The mechanism
    // was tested and could not fire in a real session — so the demand goes where the
    // compression demand already goes: into the context.
    const giovani = st.nodes.filter((nd) => !new Set(st.oldNode?.nodes ?? []).has(nd.id)).length;
    const richiestaMerge = piano.due > 0 ? t('indexDue')(giovani, cf.mergeNodesAt) : null;
    if (richiestaMerge) {
      debugLog(cf, `OLD NODE due: ${cf.mergeNodesAt}+ young nodes — the oldest ones should merge: write the riassuntone with cwl_old`);
    }
    // The FIRST step of the index needs the agent too, and it was the step with NO voice: the
    // leaves without a micro were listed in the LOG and nothing in the context asked for them,
    // so with zero nodes `piano.due` is 0, the merge demand never appears, and the mechanism
    // cannot start at all. MEASURED live, right after the reload that made the index run:
    // `NODES: 0 node(s) [none], 40 leaf/leaves waiting for a micro — their body is still in
    // the context` at every single turn, 45 spans, 90.415t of summaries still in the context,
    // and no demand anywhere. The request names a FEW ids instead of all forty: it stays
    // small, it rides at the end of every turn until the work is done, and the agent does the
    // rest next turn.
    const richiestaMicro =
      piano.waiting > 0
        ? t('leavesDue')(piano.waiting, piano.waitingIds.slice(0, MICRO_DEMAND_IDS).join(' '))
        : null;
    if (richiestaMicro) {
      debugLog(cf, `MICRO due: ${piano.waiting} leaf/leaves waiting for a micro — asked in the context (${Math.min(piano.waiting, MICRO_DEMAND_IDS)} id(s) named)`);
    }
    const demanda =
      [richiestaMicro, richiestaMerge].filter((d): d is string => d !== null).join('\n\n') || null;

    if (st.spans.length > 0) {
      const applied = applySpans(messages, st.spans, pitView(st), demanda);
      // A span whose endpoints left the context can never apply again, and each
      // one carries a summary of thousands of characters that the state re-saves
      // on every turn. Pruned here — outside the `applied > 0` guard, because the
      // case that matters is when they are ALL dead — and said out loud.
      if (applied.dead.length > 0) {
        // Before dropping them, write down what can find their original again: the stable
        // ids of their anchors. The SUMMARY is lost for good — it lived only in the state
        // — but the messages it replaced are still in the append-only transcript, and the
        // id alone could not find them: it is a HASH of the anchors.
        for (const sp of applied.dead) {
          if (sp.startSid && sp.endSid) {
            st.graves.push({ id: idOfSpan(sp), startSid: sp.startSid, endSid: sp.endSid, at: Date.now() });
          }
        }
        st.graves = st.graves.slice(-GRAVE_MAX);
        st.spans = st.spans.filter((s) => !applied.dead.includes(s));
        debugLog(cf, `SPANS pruned: ${applied.dead.length} span(s) that can never apply again — endpoints gone, or contained in another span (${st.spans.length} left, ${st.graves.length} still recoverable by id from the transcript)`);
      }
      // A partial overlap leaves both spans applied on purpose (dropping one would
      // bring back what only it covers), but the shared messages are then inside
      // two summaries. Nothing else in the pipeline can see that, so it is said here.
      if (applied.overlapped > 0) {
        debugLog(cf, `SPANS overlap: ${applied.overlapped} pair(s) of spans share a message — both stay applied, but that message is described by two summaries`);
      }
      if (applied.applied > 0) {
        // Only spans counted for the FIRST time move the totals. The others are
        // re-applications: the context really is that much smaller, but it was
        // already paid for on the turn the span was created.
        st.totalEvictions += applied.newApplied;
        st.totalEvictedTokens += applied.newSaved;
        debugLog(cf, applied.newApplied > 0
          ? `SPANS applied: ${applied.applied} (new ${applied.newApplied}), saved ${applied.newSaved}t`
          : `SPANS re-applied: ${applied.applied}, nothing new to count`);
        // The repair drops and strips messages so that toolCall and toolResult
        // stay paired. A silent drop is how a bug hides: say it out loud.
        if (applied.pairDropped > 0 || applied.pairStripped > 0) {
          debugLog(cf, `PAIR REPAIR: dropped ${applied.pairDropped} orphan tool result(s), stripped ${applied.pairStripped} orphan tool call(s) — the range had split a pair`);
        }
        // Spans applied. This used to return here, and that made the rest of the
        // hook unreachable — including the ONLY place that computes the NEXT
        // compressible range (line ~2290). `finish` and `turn_end` both read
        // `st.rangeStartHash` to decide whether there is anything to ask for,
        // and `cwl_compress_range` REFUSES while it is null (it deliberately
        // does not recompute: it has no message list). So as long as any span
        // resolved, the address was never renewed and the agent could not
        // compress a second time.
        //
        // Recomputed HERE, on the ORIGINAL list: the span endpoints must still
        // resolve, or `covered` would be empty and the same region would be
        // offered again.
        const nextRange = compressibleRange(messages, st.spans, cf.protectedTurns);
        storeRange(st, cf, nextRange, messages);
        rangeStoredBySpans = true;

        // THE RETURN HERE USED TO BE UNCONDITIONAL, and that switched the rest of
        // the hook off for good: `applied.applied > 0` is true on EVERY turn
        // while one span resolves (a span must be re-applied every time, or the
        // provider gets the uncompressed history back), so returning here made
        // the trigger check, `reasoningFallback` and `runEvictionPass`
        // unreachable for the rest of the session. MEASURED on a real session:
        // `SPANS re-applied: 3, nothing new to count` + `RANGE none | 178 msgs,
        // 133852t vs trigger 68000t` on every turn, with ZERO `EVICTION`, ZERO
        // `CONTEXT`, ZERO `no safe candidate` — 133k tokens against a 68k trigger,
        // one episode with evictable content, and the extension did nothing at
        // all. Returning is right only when the spans ALREADY did the job.
        const afterSpans = applied.kept.reduce((s: number, m: AgentMessage) => s + estimateMessageTokens(m), 0);
        if (afterSpans <= trigger) return finish(applied.kept);
        debugLog(cf, `SPANS applied (${afterSpans}t) still above trigger ${Math.round(trigger)}t: the episode pass and the fallback still run`);
        // Fall through on the COMPRESSED list, not the original one: `applySpans`
        // REPLACED the messages inside every span with its summary, so a range
        // resolved on the original list would point the eviction at messages that
        // no longer exist there.
        messages = applied.kept;
        spanInside = { set: new Set(applied.insideOut), len: applied.kept.length };
        currentTokens = afterSpans;
      }
    }

    if (currentTokens <= trigger) {
      // Under budget: nothing to compact and nothing to ask for. This is also the
      // ONLY way the gate closes — by effect, never by a confirmation token.
      st.overBudgetSince = -1;
      st.gateArmedTurn = -1;
      debugLog(cf, `CONTEXT ${currentTokens}t under threshold ${Math.round(trigger)}t: no eviction`);
      // The demand from the previous call must still be removed even here.
      // AND the state is persisted even here: this exit does NOT go through `finish`, which
      // is where the other save lives, and by now the turn may have pruned spans (writing
      // graveyards) or resolved one for the first time (writing stable ids). A save that
      // depends on WHICH exit the turn took is a save that can be skipped in silence —
      // measured: the file kept the pre-prune snapshot while the memory had the grave, so
      // `cwl_open` could not find what it had just written.
      persistIfUsed();
      return droppedGate ? { messages } : undefined;
    }

    // The last `protectedTurns` user turns are inviolable: compaction must never
    // destroy the context the agent is working on.
    const safetyFloor = protectedFromIndex(messages, cf.protectedTurns);

    // Addresses of the largest range the agent may ask to compress. Recomputed
    // here because this hook is the only place that sees the real message list.
    if (!rangeStoredBySpans) {
      const range = compressibleRange(messages, st.spans, cf.protectedTurns);
      storeRange(st, cf, range, messages, currentTokens, trigger);
    }

    /**
     * Level A — the safety net: strip reasoning blocks, ahead of the safety
     * window.
     *
     * It runs whenever we are still over budget AFTER the episode pass, not only
     * when the graph is empty. The old `if (g.isEmpty)` gate meant that as soon
     * as ONE episode existed, the biggest and safest reclaim available was
     * disabled for the whole session: MEASURED on a real context, assistant
     * thinking blocks are 43% of it (280k of 650k tokens), and in a controlled
     * probe all 4 thinking blocks outside an episode survived while the same 4
     * were removed when no episode existed.
     */
    const reasoningFallback = (list: AgentMessage[]): AgentMessage[] | null => {
      // Honour the level switch. The old fallback called the strip unguarded, so
      // `stripReasoning: false` did not actually turn it off; that mattered little
      // when the fallback only ran in a graph with no episodes, and matters a lot
      // now that it is the main path. With the level off, the safety net is off:
      // the operator's switch has to mean something.
      if (!levelEnabled(cf, 'reasoning')) return null;
      const floor = protectedFromIndex(list, cf.protectedTurns);
      const out = globalReasoningStrip(list, floor);
      if (!out) return null;
      const before = list.reduce((s: number, m: AgentMessage) => s + estimateMessageTokens(m), 0);
      const after = out.kept.reduce((s: number, m: AgentMessage) => s + estimateMessageTokens(m), 0);
      st.totalEvictions++;
      st.totalEvictedTokens += Math.max(0, before - after);
      debugLog(cf, `FALLBACK reasoning-strip: ${out.changed} messages, ${before}t -> ${after}t`);
      if (ctx?.hasUI) {
        ctx.ui.notify(t('fallbackNotice')(out.changed, before, after), 'info');
      }
      return out.kept;
    };

    // No episodes at all: episodes are the targeted path, so the safety net is
    // the only one left.
    if (g.isEmpty) {
      const stripped = reasoningFallback(messages);
      if (stripped) return finish(stripped);
      debugLog(cf, 'CONTEXT above threshold but no episode AND no reasoning strip possible');
      return finish(messages);
    }

    // 2. Deterministic policy: compute what to evict and at which level
    const actions = runEvictionPass(cf, g, currentTokens, trigger, messages);
    if (actions.length === 0) {
      // No episode is safe to touch: the safety net still is.
      const stripped = reasoningFallback(messages);
      if (stripped) return finish(stripped);
      debugLog(cf, `CONTEXT ${currentTokens}t above threshold but no safe candidate: context untouched`);
      return finish(messages);
    }

    // 3. Actually APPLY: rebuild the message list evicting the segments
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

    // Map message index -> episode, through the ONE helper that knows how to
    // locate an episode in this list (episodeRanges, by toolCallId anchors).
    // The plan measures on the same ranges: two different notions of "where the
    // episode is" is how a plan describes a stretch of conversation the
    // eviction never touches.
    // Uses g.closed() (all closed episodes, including those at level='removed'):
    // with recoverable() the episodes already marked 'removed' would NEVER be
    // mapped and the full eviction would never fire.
    const episodeAt = new Map<number, Episode>();
    const episodesByName = new Map(g.closed().map((ep) => [ep.name, ep]));
    // Closed episodes keep the ORDER THEY WERE OPENED IN, and an episode can only
    // be deduced when its start anchor predates the compaction cut — so a deduced
    // range is always written BEFORE the located ranges of the episodes opened
    // after the cut, and those overwrite it on the indices they legitimately
    // claim. That is why the loop needs no special ordering: reordering it would
    // be code no test could break.
    for (const [name, range] of episodeRanges(messages, g.closed())) {
      const ep = episodesByName.get(name);
      if (!ep) continue;
      for (let i = range.from; i <= range.to; i++) episodeAt.set(i, ep);
    }

    const kept: AgentMessage[] = [];
    let dropped = 0;
    let truncated = 0;
    let removedTokens = 0;
    let truncatedTokens = 0;
    /** Native summaries that a FULL eviction would have taken, and did not. */
    let summariesSaved = 0;
    // toolCallIds whose tool result is being dropped: the assistant message that
    // carries the matching toolCall must not keep it, or the conversation has a
    // call with no result (an invalid request for the provider).
    const droppedToolCallIds = new Set<string>();
    // One re-entry marker per episode, not one per evicted message: a four-message
    // episode used to emit four identical markers.
    const markedEpisodes = new Set<string>();

    messages.forEach((msg, idx) => {
      const ep = episodeAt.get(idx);
      // SAFETY: read-only probe of an optional field; it is undefined for
      // messages that are not user/assistant/toolResult.
      const role = (msg as unknown as RealMessage).role;

      // Principle 3: user turns are inviolable. But `user` is not enough: `system`
      // and `developer` carry the role instructions and the available tools.
      // Evicting them leaves the agent with no action schema.
      // `custom` is protected too: other extensions inject their own content with
      // that role (pi-anti-amnesia's memory card is role:'custom'), and it sits
      // outside any episode of this extension.
      if (role === 'user' || role === 'system' || role === 'developer' || role === 'custom') {
        kept.push(msg);
        return;
      }

      // Pi's own compaction summary is NOT a user turn: it carries the role
      // 'compactionSummary' (pi/dist/core/messages.js, createCompactionSummaryMessage),
      // and a branch summary carries 'branchSummary'. Both are the ONLY copy of
      // the history native compaction replaced — the transcript holds it, the
      // provider does not — so they are inviolable in the same sense user turns
      // are. They were simply missing from the list of protected roles, and that
      // omission is what a deduced range starting at index 0 would have reached.
      // Counted only when a FULLY evicted episode really claimed their index:
      // otherwise the number would claim a save the `!ep` branch already gave.
      if (role === 'compactionSummary' || role === 'branchSummary') {
        if (ep && evictFull.has(ep.name)) summariesSaved++;
        kept.push(msg);
        return;
      }

      // Safety window: the most recent turns are inviolable whatever the episode
      // mapping says. This is checked BEFORE the episode branches, so a `removed`
      // level cannot reach into the window either.
      if (idx >= safetyFloor) {
        kept.push(msg);
        return;
      }

      if (!ep) {
        kept.push(msg);
        return;
      }

      if (evictFull.has(ep.name)) {
        removedTokens += estimateMessageTokens(msg);
        // SAFETY: toolCallId exists on the real tool-result messages, but the
        // public AgentMessage union does not declare it.
        const droppedId = (msg as unknown as RealMessage).toolCallId;
        if (typeof droppedId === 'string') droppedToolCallIds.add(droppedId);
        // ONE re-entry marker per episode — emitted for `act` too. The marker
        // used to carry only the prose description, so an evicted `act` (whose
        // description is always empty) left NOTHING behind, and an `expl`
        // without a description lost its content for good. It now carries the
        // episode NAME, which is the address: the transcript is append-only and
        // the episode can be read back with cwl_recall_episode.
        if (!markedEpisodes.has(ep.name)) {
          markedEpisodes.add(ep.name);
          // SAFETY: Pi accepts the custom role in the context hook although the
          // AgentMessage union does not declare it; the extra keys are its own
          // custom-message contract.
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
        // Count only if the strip REALLY reduced something: otherwise the notice
        // would claim a saving that did not happen, and the operator would believe
        // eviction is working while it moves nothing.
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
        // Count only if the strip REALLY reduced something: otherwise the notice
        // would claim a saving that did not happen, and the operator would believe
        // eviction is working while it moves nothing.
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
      // Nothing in the episodes was reducible: the safety net is still there.
      const stripped = reasoningFallback(messages);
      if (stripped) return finish(stripped);
      debugLog(cf, 'EVICTION: no message actually reducible, context left untouched');
      return finish(messages);
    }

    // H1: strip the toolCall blocks whose tool result was just dropped. The
    // episode range starts at the tool RESULT, so the assistant message carrying
    // the opening toolCall sits before it and always survived — leaving an
    // orphan call that makes the request invalid.
    if (droppedToolCallIds.size > 0) {
      for (let i = 0; i < kept.length; i++) {
        // SAFETY: read-only field probe (content); the union does not expose it.
        const m = kept[i] as unknown as RealMessage;
        if (!Array.isArray(m?.content)) continue;
        const blocks = m.content as RealContentBlock[];
        const hasOrphan = blocks.some(
          (b) => b && b.type === 'toolCall' && typeof (b as { id?: string }).id === 'string'
            && droppedToolCallIds.has((b as { id: string }).id),
        );
        if (!hasOrphan) continue;
        const filtered = blocks.filter(
          (b) => !(b && b.type === 'toolCall' && typeof (b as { id?: string }).id === 'string'
            && droppedToolCallIds.has((b as { id: string }).id)),
        );
        if (filtered.length === 0) {
          // Nothing left: an assistant message with an empty content array is not
          // a valid turn, so drop the whole message instead of keeping a husk.
          kept.splice(i, 1);
          i--;
          continue;
        }
        // SAFETY: only `content` changes; the message keeps its role and every
        // other field, so it is still a valid message for the provider.
        kept[i] = { ...(m as object), content: filtered } as unknown as AgentMessage;
      }
    }

    if (summariesSaved > 0) {
      debugLog(cf, `SUMMARY GUARD: ${summariesSaved} native summary message(s) sat inside an evicted range and were kept — they are the only copy of the history that was compacted`);
    }

    st.totalEvictions++;
    st.lastEvictionTurn = st.messageCursor;
    const afterTokens = kept.reduce((s: number, m: AgentMessage) => s + estimateMessageTokens(m), 0);
    // The saving is MEASURED, not estimated. `currentTokens - afterTokens` is the
    // same quantity the arrow prints and the same one the user's notice receives;
    // the plan's estimate (`removedTokens + truncatedTokens`) is a DIFFERENT
    // number, and printing the two side by side without saying so is what this
    // row did: `32 msg removed, 0 reduced, 79683t -> 58987t (saved 21053t)`
    // declares 21053 while the eviction freed 20696. The estimate is off in BOTH
    // directions, and the signs name the causes: it counts the messages as they
    // were BEFORE the pass, so it ignores the re-entry MARKER the pass ADDS to
    // `kept` (too high) and the orphan `toolCall` blocks it prunes from
    // SURVIVING messages (too low); for strips it measures raw text
    // (`contentToText`, i.e. `text` and `thinking` characters) instead of the
    // serialized message, so the keys, the `type` field and the escaping of every
    // block stay out of the count — negligible for one huge `text` block,
    // ~4% across many small ones.
    const measuredSaved = Math.max(0, currentTokens - afterTokens);
    const planEstimate = removedTokens + truncatedTokens;
    st.totalEvictedTokens += measuredSaved;
    debugLog(cf, `EVICTION applied: ${dropped} msg removed, ${truncated} reduced, ${currentTokens}t -> ${afterTokens}t (saved ${measuredSaved}t)`);
    // Kept, and said out loud, because the gap is the diagnostic of the pass (it
    // is how the reporter learned the marker and the orphan pruning were never
    // counted). Two rows, never two meanings in one.
    if (planEstimate !== measuredSaved) {
      debugLog(cf, `EVICTION accounting: il piano stimava ${planEstimate}t, il misurato e' ${measuredSaved}t (${planEstimate > measuredSaved ? '+' : ''}${planEstimate - measuredSaved}t)`);
    }

    if (ctx?.hasUI) {
      ctx.ui.notify(
        t('evictionNotice')(dropped, truncated, currentTokens.toLocaleString(), afterTokens.toLocaleString()),
        'info',
      );
    }

    // Still over budget after the episode pass: the safety net closes the gap in
    // the SAME turn instead of waiting for the episode pass to run dry.
    if (afterTokens > trigger) {
      const stripped = reasoningFallback(kept);
      if (stripped) return finish(stripped);
    }

    return finish(kept);
  });

  pi.on('turn_end', async (_event, ctx) => {
    // Observer: the message cursor is updated in the context hook,
    // which is the only point with access to the real message array.
    const key = sessionKey(ctx);
    const st = states.get(key);
    if (!st) return;
    const cf = getConfig(key);

    st.turns += 1;

    // The gate is ARMED here, not in the context hook: arming is a decision about
    // elapsed turns, and the hook must only render the demand for the state it
    // finds. The demand is dropped — with a cooldown — after a few unanswered
    // turns, so a model that ignores it is not nagged forever: the safety net in
    // the context hook is what actually bounds the context.
    // Arm only when there is something the agent can actually DO. Otherwise the
    // demand is unsatisfiable BY CONSTRUCTION and just burns context: measured in
    // a real session, where it asked every turn while no episode was open and no
    // compressible range existed.
    const actionable = st.graph.active().length > 0 || st.rangeStartHash !== null;
    if (cf.gate && actionable && st.overBudgetSince >= 0 && (st.turns - st.overBudgetSince) >= GATE_AFTER_TURNS) {
      const inCooldown = st.lastGateViolationTurn >= 0 &&
        (st.turns - st.lastGateViolationTurn) < GATE_COOLDOWN_TURNS;
      if (!inCooldown) {
        if (st.gateArmedTurn < 0) {
          st.gateArmedTurn = st.turns;
          st.gateAttempts = 0;
          debugLog(cf, `GATE armed at turn ${st.turns} (over budget since turn ${st.overBudgetSince})`);
        } else if (st.gateArmedTurn < st.turns) {
          st.gateAttempts += 1;
          if (st.gateAttempts >= GATE_MAX_ATTEMPTS) {
            st.lastGateViolationTurn = st.turns;
            st.gateArmedTurn = -1;
            debugLog(cf, `GATE dropped after ${GATE_MAX_ATTEMPTS} attempts; cooldown ${GATE_COOLDOWN_TURNS} turns`);
            if (ctx?.hasUI) ctx.ui.notify(t('gateGiveUp')(GATE_MAX_ATTEMPTS), 'warning');
          }
        }
      }
    }

    // Persist the DECISIONS (episode graph + traced compressions) at the end of
    // every turn. They cannot be recomputed from the transcript, so a crash or
    // a restart must not throw them away. Sessions that never used CWL write
    // nothing.
    if (!st.graph.isEmpty || st.spans.length > 0) saveState(key, st);
  });

  pi.on('session_shutdown', async (_event, ctx) => {
    const key = sessionKey(ctx);
    const st = getState(key);
    debugLog(getConfig(key), `SESSION SHUTDOWN — total evictions: ${st.totalEvictions}`);
    // Save BEFORE dropping the in-memory copy, otherwise the graph and the
    // compressed spans are gone at the next start (the bug this fixes).
    if (!st.graph.isEmpty || st.spans.length > 0) saveState(key, st);
    dropState(key);
  });
}
