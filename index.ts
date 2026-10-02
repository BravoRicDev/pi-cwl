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
  /**
   * The announcement that opens an INHERITED memory (`cwl_adopt`), injected as the very
   * first block. It exists because nothing else says it: the blocks that follow describe
   * the past they hold, but they do not say that the past is ANOTHER transcript's — and a
   * session that is not told goes looking for its own history on disk instead of reading
   * the index it was given.
   */
  inheritedHead: (name: string, from: string, leaves: number, nodes: number) => string;
  episodeClosed: (type: string, name: string) => string;
  statusMeasured: (tokens: string, extra?: string) => string;
  statusActive: (list: string) => string;
  statusStripped: (list: string) => string;
  /** UI notice when falling back to the global reasoning strip. */
  /** UI notice after an eviction pass. */
  evictionNotice: (dropped: number, truncated: number, from: string, to: string) => string;
  /** Strings of the cwl_compress tool. */
  compressMissingParams: string;
  compressRevoked: (start: string, end: string) => string;
  compressUnknownHash: (from: boolean, to: boolean) => string;
  compressApplied: (start: string, end: string) => string;
  /** cwl_compress_range: the extension picks the range, the model writes the text. */
  compressRangeApplied: (start: string, end: string, tokens: number, leafId: string) => string;
  compressRangeNothing: string;
  compressRangeNoSummary: string;
  /** Status line for the currently compressible range. */
  statusRange: (tokens: number, start: string, end: string) => string;
  statusAddresses: (eligible: number, withId: number) => string;
  statusSpans: (n: number) => string;
  /** The names of the topics, so the catalogue is reachable without opening every node. */
  statusTopics: (n: number, names: string) => string;
  /** Ids of the nodes that are NOT topics (the pit, the buffer): the two an agent needs most. */
  statusIds: (ids: string) => string;
  /** #1 — the leaf ids of the buffer and of the young nodes, only when explicitly asked. */
  statusLeafIds: (ids: string) => string;
  statusLooseIds: (ids: string) => string;
  statusLeafIdsHint: string;
  /** A merge that would free less than it costs: refused, with the numbers said. */
  oldTooSmall: (leaves: number, microChars: number, needChars: number) => string;
  /** The index shape in one line: the TUI widget and `cwl_status` show the same one. */
  indexLine: (pitNodes: number, pitLeaves: number, topicNodes: number, topicLeaves: number, bufferNodes: number, bufferLeaves: number, plainNodes: number, plainLeaves: number, loose: number, headTokens: string, evictions: number, savedTokens: string) => string;
  /** Grouping leaves into a topic node, and the four reasons to refuse it. */
  groupRefused: (why: string, detail: string) => string;
  groupTooSmall: (leaves: number, microChars: number, needChars: number) => string;
  groupCreated: (name: string, id: string, leaves: number, microChars: number) => string;
  /** A topic born INSIDE the old node: catalogued there, and the pit's synthesis is NOT touched. */
  pitTopicBorn: (name: string, id: string, leaves: number) => string;
  groupAdded: (name: string, id: string, leaves: number) => string;
  /** Said when a topic absorbs other NODES: what left the head, and what stays readable. */
  groupAbsorbed: (name: string, id: string, nodes: number, leaves: number, blocks: number) => string;
  /** The header of the ONE block a topic node injects, where its first leaf used to be. */
  topicHead: (id: string, name: string, leaves: number, saved: number) => string;
  /** The shape of what a parent topic holds, MEASURED by the code and never written by the agent. */
  topicHolds: (nodes: number, leaves: number) => string;
  /** Said after a merge that carried the immutable descriptions of the topics it absorbed. */
  oldTopicsConcatenated: (topics: number, chars: number) => string;
  /** Said when a description is rewritten inside the pit: the context does not move. */
  groupDescriptionUpdated: () => string;
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
  /** The old node: the synthesis that replaces the micros, and what it has merged. */
  oldNodeSet: (id: string, nodes: number, leaves: number, microChars: number, tokens: number) => string;
  oldNotDue: (young: number, need: number) => string;
  /** A DRY RUN: what a merge WOULD do, with the numbers, and nothing recorded. */
  oldDryRun: (pass: boolean, destination: string, nodes: number, leaves: number, freedChars: number, needChars: number) => string;
  /** A DRY RUN for `cwl_group` when it would ADD leaves to a topic that already exists. */
  groupDryRunAdd: (id: string, name: string, leaves: number, nodes: number, microChars: number, needChars: number) => string;
  /** #7 — the batch: all the groups validated first, then all applied, or none. */
  groupManyDone: (groups: number, already: number, leaves: number, nodes: number) => string;
  /** A DRY RUN for `cwl_group` when it would BORN a new topic. */
  groupDryRunNew: (id: string, name: string, leaves: number, nodes: number, microChars: number, needChars: number) => string;
  oldHead: (id: string, nodes: number, tokens: number) => string;
  /** Said when the block below is the DESCRIPTIONS rather than the synthesis, so the agent knows
   *  where the narrative went. See `PitView.body`. */
  oldHeadDescriptions: (id: string) => string;
  /** The old-node page: the merge summary and the shape of what it holds. */
  oldPage: (id: string, nodes: number, tokens: number, body: string) => string;
  /** One entry of the old node's catalogue: a topic inside the pit, named and tasted. */
  oldTopicLine: (id: string, name: string, shape: string, taste: string) => string;
  /** The heading of the "consulted leaves" section inside the pit page. */
  oldHot: (listed: number, total: number) => string;
  /** The header of the "superseded syntheses" block on the pit page. */
  oldSupersededHead: (count: number) => string;
  /** One superseded synthesis, listed on the pit page with its own address. */
  oldSupersededLine: (id: string, chars: number) => string;
  /** The page of one superseded synthesis: the text `cwl_old` replaced. */
  oldSupersededPage: (id: string, chars: number, text: string) => string;
  /** The request to write the merge summary: it is the only trigger the agent sees. */
  indexDue: (young: number, need: number) => string;
  /** The invitation to open a topic while the buffer's leaves can still be moved. */
  topicDue: (id: string, leaves: number, microChars: number, needChars: number) => string;
  /** The leaves waiting for a micro: the request that starts the index. */
  leavesDue: (missing: number, ids: string) => string;
  /** A leaf that was pruned: its summary is lost, the original comes back from the transcript. */
  openOriginal: (id: string, tokens: number, body: string) => string;
  openOriginalLost: (id: string) => string;
  /** cwl_open: the summary left the RAM (the leaf is in the pit) and the disk record is gone. */
  openBodyLost: (id: string) => string;
  /** The memories a session can fork, and the fork itself. See `cwl_memories` / `cwl_adopt`. */
  memoriesEmpty: () => string;
  memoriesList: (rows: string) => string;
  memoriesRow: (name: string, leaves: number, nodes: number, pit: string) => string;
  adoptDone: (name: string, from: string, leaves: number, nodes: number) => string;
  adoptNotFound: (from: string, names: string) => string;
  adoptBusy: (name: string) => string;
  adoptEmpty: (from: string) => string;
  adoptOtherSession: string;
  /** The slash commands: the fork in the operator's hands, with no agent in between. */
  cmdMemoriesDesc: () => string;
  cmdAdoptDesc: () => string;
  cmdPickSource: () => string;
  cmdPickName: () => string;
  cmdCancelled: () => string;
  cmdNoAdoptable: () => string;
  /** /cwl_save and /cwd_save: operator-triggered full compaction with zero protected turns. */
  cmdSaveDesc: () => string;
  cmdSaveNothing: () => string;
  cmdSaveTriggered: (tokens: number) => string;
  cmdSavePrompt: (note?: string) => string;
  cmdSaveDemand: (note?: string) => string;
  /** cwl_micro: the body leaves the context, the micro takes its place. */
  microSet: (id: string, microChars: number, bodyChars: number, shorter: boolean) => string;
  /** Budget gate: the agent-driving channel. */
  gateDemand: (current: string, budget: string, turns: number, canClose: boolean, canCompress: boolean) => string;
  gateGiveUp: (attempts: number) => string;
  /** Texts that end up in the LLM context. */
  snippets: { delimiter: string; status: string; compress: string; recall: string; recallEpisode: string; compressRange: string; memories: string; adopt: string };
  /** Texts of the autonomous tools. */
  tools: {
    compressDesc: string; compressStart: string; compressEnd: string; compressSummary: string;
    recallDesc: string; recallQuery: string; recallLimit: string;
    recallEpisodeDesc: string; recallEpisodeName: string; recallEpisodeFull: string;
    memoriesDesc: string; adoptDesc: string; adoptFromDesc: string; adoptAsDesc: string;
    openDesc: string;
    openId: string;
    microDesc: string;
    microId: string;
    microText: string;
    oldDesc: string;
    oldText: string;
    oldDryRun: string;
    groupDryRun: string;
    groupManyDesc: string;
    groupGroups: string;
    compressRangeDesc: string; compressRangeSummary: string; compressMicro: string;
    compressCovered: (start: string, end: string) => string;
    microOver: (where: string, chars: number, ceiling: number) => string;
    groupDesc: string; groupLeaves: string; groupNodes: string; groupNode: string; groupName: string; groupText: string; groupPit: string;
  };
  /** Parameter descriptions: read by the LLM on every invocation. */
  params: {
    delimiterDesc: string; action: string; name: string; type: string;
    dependencies: string; description: string; statusDesc: string;
  };
  /**
   * The CONSULTATION tools: the index is findable instead of guessable. Every ID
   * an agent needs must be reachable from here, or it will be hunted by trial.
   */
  consult: {
    findDesc: string; findQuery: string; findScope: string; findLimit: string; findSnippet: string;
    findFound: (n: number, query: string, body: string) => string;
    findNoMatch: (query: string) => string;
    findNotLoaded: string;
    mapDesc: string; mapNode: string; mapSnippet: string;
    mapHeader: (nodes: number, leaves: number, chars: string) => string;
    mapLine: (id: string, kind: string, name: string, leaves: number, children: number, chars: number) => string;
    mapLeaves: (ids: string) => string;
    nodeDesc: string; nodeId: string; nodeSnippet: string;
    nodeLimit: string; nodeCursor: string; pendingLimit: string; pendingCursor: string;
    nodeHeader: (id: string, kind: string, name: string, leaves: number, chars: number) => string;
    nodeLeaf: (id: string, micro: string) => string;
    nodeChild: (ids: string) => string;
    nodeNotFound: (id: string) => string;
    pendingDesc: string; pendingSnippet: string;
    pendingHeader: (nodes: number, leaves: number, chars: string, need: string) => string;
    pendingLine: (id: string, kind: string, leaves: number, chars: number) => string;
    pendingEmpty: string;
    pendingPitHeader: (nodes: number, leaves: number) => string;
    pendingPitLine: (id: string, leaves: number) => string;
    pendingPitHint: string;
    pageInfo: (shown: number, total: number, next: string) => string;
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
      `To reopen the WHOLE summary, call exactly cwl_open({id: "${id}"}). ` +
      `If you need the original text, call cwl_recall with keywords from that content.\n\n`,
    inheritedHead: (name, from, leaves, nodes) => `[CWL · INHERITED MEMORY] This session starts with an INHERITED MEMORY: ` +
      `"${name}", forked from "${from}" — ${leaves} archived leaf/leaves and ${nodes} node(s) copied here from ANOTHER transcript.\n` +
      `This session has NO earlier messages of its own: the index at the TOP of this context IS its past — ` +
      `the pit's synthesis, the topics' descriptions, the labels of the leaves no topic holds. ` +
      `Read it BEFORE exploring: cwl_map for the shape with ids, cwl_node to page a node's leaves, ` +
      `cwl_open to read one leaf or the pit whole, cwl_recall to search them by keywords.\n` +
      `Do NOT go looking on the filesystem or in old session logs for what this memory already says. ` +
      `An archived leaf is never compressed nor pruned again: for those leaves the summary is the ONLY copy that exists.`,
    episodeClosed: (type, name) => `Episode ${type} "${name}" closed.`,
    statusMeasured: (tokens, extra) => `Measured context tokens: ~${tokens}${extra ? ` (${extra})` : ''}`,
    statusActive: (list) => `Active: ${list}`,
    statusStripped: (list) => `Stripped: ${list}`,
    evictionNotice: (dropped, truncated, from, to) => `CWL: ${dropped} evicted, ${truncated} reduced (${from} -> ${to} tokens).`,
    compressMissingParams: 'startHash, endHash and summary are all required.',
    compressRevoked: (start, end) => `Span ${start}..${end} restored to full text.`,
    compressUnknownHash: (from, to) => `Unknown hash. startHash found: ${from}, endHash found: ${to}. Copy them verbatim from the context.`,
    compressApplied: (start, end) => `Compressed ${start}..${end} into your summary. The original stays on disk: recover it with cwl_recall.`,
    compressRangeApplied: (start, end, tokens, leafId) => `Compressed ${start}..${end} (~${tokens} tokens) into your summary. The original stays in the transcript: recover it with cwl_recall_episode or cwl_recall. This range became the leaf ${leafId}: cwl_open("${leafId}") reads it whole, and cwl_group takes its id to group it with the leaves born in the same turn.`,
    compressRangeNothing: 'Nothing left to compress: everything remaining is either inside the protected window or already compressed.',
    compressRangeNoSummary: 'The summary is required: it is the ONLY part of this call you have to write yourself.',
    statusRange: (tokens, start, end) => `Compressible range: ~${tokens} tokens (${start}..${end})`,
    statusAddresses: (eligible, withId) => `Addresses: ${withId}/${eligible} endpoint messages carry a stable id`,
    statusSpans: (n) => `Compressed spans held: ${n}`,
    statusTopics: (n, names) => `Topics (${n}): ${names}`,
    statusIds: (ids) => `Node ids: ${ids}`,
    statusLeafIds: (ids) => `Leaf ids: ${ids}`,
    statusLooseIds: (ids) => `Loose leaves: ${ids}`,
    statusLeafIdsHint: 'Also list the LEAF ids of the buffer and of the young nodes, plus the loose leaves. Off by default: cwl_status is called often and the buffer alone can hold fourteen leaves, while the ids are one cwl_map away.',
    oldTopicLine: (id, name, shape, taste) => `- ${id} \u00b7 TOPIC "${name}": ${shape} \u2014 ${taste}`,
    oldTooSmall: (leaves, microChars, needChars) =>
      `\n\nNothing was recorded: the ${leaves} leaf/leaves that would leave the context hold ${microChars} characters, and a merge must free at least ${needChars}. A merge COSTS a synthesis: the pit's summary is rewritten, so what leaves has to be worth more than what replaces it. Compress more first, or merge when the nodes are full.`,
    indexLine: (pitNodes, pitLeaves, topicNodes, topicLeaves, bufferNodes, bufferLeaves, plainNodes, plainLeaves, loose, headTokens, evictions, savedTokens) =>
      `pit ${pitNodes}n/${pitLeaves}l │ topics ${topicNodes}n/${topicLeaves}l │ buffer ${bufferNodes}n/${bufferLeaves}l │ ordinary ${plainNodes}n/${plainLeaves}l │ loose ${loose} │ head ~${headTokens}t │ ${evictions} evict │ ${savedTokens} saved`,
    groupRefused: (why, detail) =>
      `\n\nGrouping refused (${why}${detail ? `: ${detail}` : ''}). Nothing was recorded. Leaves can be taken from the buffer, from the loose ones, and from ordinary nodes (a TOPIC is not a source: its description stands for its leaves), and they must be consecutive in time. A leaf already inside the old node cannot come back (the pit's synthesis stands for it), and a leaf without a micro would not appear in any head: write the micro first. A description is IMMUTABLE while its topic is OUTSIDE the old node, because there it IS the index; inside the old node it can be rewritten, and there it moves nothing.`,
    groupTooSmall: (leaves, microChars, needChars) =>
      `\n\nNo topic was born: the ${leaves} leaf/leaves hold ${microChars} characters, and a topic must free at least ${needChars}. A topic is born COLLAPSED — its description replaces the labels of those leaves from this moment on — so it has to be worth more than the description that replaces them (~3,600 characters times the ratio, and never less than the absolute floor). Group more leaves, or leave them in the buffer until the topic is big enough.`,
    groupCreated: (name, id, leaves, microChars) =>
      `Topic "${name}" born as ${id}: ${leaves} leaf/leaves (${microChars} characters of labels) now stand behind your description, and their bodies stay readable with cwl_open. More leaves can be added to it at any time, and it costs nothing: the description does not change.`,
    pitTopicBorn: (name, id, leaves) =>
      `Topic "${name}" born INSIDE the old node as ${id}: ${leaves} leaf/leaves catalogued there. The pit's synthesis was NOT touched, and nothing in the index moved, because those labels were already hidden behind it. You can add more leaves to it and rewrite its description: inside the pit that no longer touches the index.`,
    groupAdded: (name, id, leaves) =>
      `Leaf/leaves added to the topic "${name}" (${id}), which now holds ${leaves}.`,
    groupAbsorbed: (name, id, nodes, leaves, blocks) =>
      `\n\n"${name}" (${id}) now CONTAINS ${nodes} node(s) and ${leaves} leaf/leaves in all: their ${blocks} block(s) left the head and the parent's description stands for them. Nothing was deleted — cwl_open("<child id>") returns any of them whole.`,
    groupDescriptionUpdated: () =>
      `\n\nThe description was rewritten, and this changes NOTHING in the index: the node is inside the old node, whose synthesis was written before these leaves arrived. The pit's synthesis is a frozen snapshot; what you have just written is the living copy, readable with cwl_open.`,
    topicHead: (id, name, leaves, saved) =>
      `[CWL \u00b7 TOPIC "${name}" (${id}) \u2014 ${leaves} leaf/leaves stand behind this description, which never changes (~${saved} tokens saved). To open this topic, call exactly cwl_open({id: "${id}"}); to open one leaf, call cwl_open({id: "<leaf-id>"}).\n\n`,
    topicHolds: (nodes, leaves) =>
      `(holds ${nodes} node(s), ${leaves} leaf/leaves in all \u2014 open them with cwl_open("<id>"))\n`,
    oldTopicsConcatenated: (topics, chars) =>
      `\n\nThe immutable description(s) of ${topics} topic node(s) were glued to this synthesis, word for word (${chars} characters): they are not rewritten and not to be rewritten.`,
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
    oldNodeSet: (id, nodes, leaves, microChars, tokens) => `Old node ${id}: merged ${nodes} node(s), ${leaves} leaf/leaves (${microChars} chars of micros) into a synthesis of ~${tokens} tokens. They stay readable: cwl_open("${id}") lists the nodes inside.`,
    oldNotDue: (young, need) => `No merge: ${young} young node(s), the merge starts at ${need}. Nothing was recorded.`,
    oldDryRun: (pass, destination, nodes, leaves, freedChars, needChars) => `DRY RUN, nothing was recorded. Destination: ${destination}. It would merge ${nodes} node(s) and ${leaves} leaf/leaves, freeing ${freedChars} characters; the merge needs ${needChars}. Verdict: ${pass ? 'it would go through' : 'it would be REFUSED'}.`,
    groupManyDone: (groups, already, leaves, nodes) => `Batch of ${groups} group(s): ${groups - already} applied, ${already} already satisfied (nothing to do), ${leaves} leaf/leaves moved and ${nodes} node(s) absorbed. Validated ALL of them before touching anything, so nothing was applied half way.`,
    groupDryRunAdd: (id, name, leaves, nodes, microChars, needChars) => `DRY RUN, nothing was recorded. Topic ${id} "${name}" would receive ${leaves} leaf/leaves and ${nodes} node(s); it holds ${microChars} characters of labels against a floor of ${needChars}.`,
    groupDryRunNew: (id, name, leaves, nodes, microChars, needChars) => `DRY RUN, nothing was recorded. A new topic would be born as ${id} "${name}", taking ${leaves} leaf/leaves and ${nodes} node(s); it holds ${microChars} characters of labels against a floor of ${needChars}.`,
    oldHead: (id, nodes, tokens) => `[CWL OLD NODE ${id} — ${nodes} older node(s) merged behind this synthesis (~${tokens} tokens). To open this pit and list its contents, call exactly cwl_open({id: "${id}"}); open a listed node or leaf with cwl_open({id: "<id-from-the-page>"}).]\n\n`,
    oldHeadDescriptions: (id) => `[CWL OLD NODE — what follows is NOT the merge synthesis but the shorter topic descriptions held by the pit. To read the full synthesis and list its contents, call cwl_open({id: "${id}"}).]\n\n`,
    oldPage: (id, nodes, tokens, body) => `[CWL old node ${id} — ${nodes} node(s) inside, ~${tokens} tokens. The synthesis first, then one line per node with its SHAPE; cwl_open("<node id>") opens one, and its leaves open in full.]\n\n${body}`,
    oldHot: (listed, total) => `--- Most consulted leaves (${listed} of ${total} in the old node; nothing was deleted, this is only the reading order) ---`,
    oldSupersededHead: (count) => `--- Syntheses this one replaced (${count}, newest first): open one with cwl_open("<id>.s1") — cwl_old overwrites the synthesis instead of extending it, so these are kept readable rather than lost ---`,
    oldSupersededLine: (id, chars) => `- ${id}: ${chars} chars`,
    openBodyLost: (id) => `Leaf "${id}" has no body: its summary left the memory (the leaf is inside the pit) and the disk record is gone. If the leaf has a micro, that is what remains of it.`,
    memoriesEmpty: () => 'No CWL memory exists yet.',
    memoriesList: (rows) => `CWL memories:\n${rows}`,
    memoriesRow: (name, leaves, nodes, pit) => `- ${name} │ ${leaves} leaf/leaves │ ${nodes} node(s) │ pit: ${pit}`,
    adoptDone: (name, from, leaves, nodes) => `Memory "${name}" created by forking "${from}": ${leaves} leaf/leaves, ${nodes} node(s) copied as ARCHIVED. They carry no anchors in this session, so they will never be compressed again and never pruned — their summaries are the only copy. Episodes did not travel. The memory's index is injected at the top of the context and announces itself at the end of it, until this session has leaves of its own.`,
    adoptNotFound: (from, names) => `No CWL memory matches "${from}". The memories that exist are: ${names}.`,
    adoptBusy: (name) => `The memory "${name}" is already this session's memory: there is nothing to fork.`,
    adoptEmpty: (from) => `The memory "${from}" has no leaves: there is nothing to fork.`,
    adoptOtherSession: 'A memory cannot be adopted while it belongs to a LIVING session: that session keeps writing its own copy and the two would overwrite each other. Close it first, or adopt from a session file path.',
    cmdMemoriesDesc: () => 'List the CWL memories on this machine, by name and size.',
    cmdAdoptDesc: () => 'Fork a memory into this session: pick it from the list, then name the fork.',
    cmdPickSource: () => 'Pick the memory to fork:',
    cmdPickName: () => 'Name for the fork (empty keeps the default):',
    cmdCancelled: () => 'Nothing was adopted.',
    cmdNoAdoptable: () => 'No adoptable memory on this machine.',
    cmdSaveDesc: () => 'Force total compaction (zero protected turns) into a final leaf for inheritance or topic change.',
    cmdSaveNothing: () => 'Nothing left to compact: all recent messages are already inside leaves or not enough messages exist.',
    cmdSaveTriggered: (tokens) => `Total compaction triggered: ~${tokens} recent tokens will be compressed into the final leaf.`,
    cmdSavePrompt: (note) => `[/cwl_save COMMAND] Perform full compaction of the entire recent history (including the latest turn, zero protected turns) by calling cwl_compress_range with an accurate summary and micro.${note ? `\nOperator note: "${note}"` : ''}`,
    cmdSaveDemand: (note) => `TOTAL COMPACTION REQUESTED (/cwl_save): The operator requested full compaction of the recent history into a final leaf (zero protected turns). Call cwl_compress_range(summary="...", micro="...") detailing all work done so far and final status.${note ? ` Note: "${note}"` : ''}`,
    oldSupersededPage: (id, chars, text) => `[CWL superseded synthesis ${id} — ${chars} chars. This is a synthesis that the CURRENT one of the pit replaced; cwl_old overwrites instead of extending, so the archive keeps the text it would otherwise have erased. Nothing else refers to it.]\n\n${text}`,
    indexDue: (young, need) => `[CWL INDEX] ${young} node(s) of the index are due to merge (a merge starts at ${need}). Call cwl_old with the merge summary: your synthesis replaces the content of the oldest nodes — their topics' descriptions and their labels — and their leaves stay readable with cwl_open.`,
    topicDue: (id, leaves, microChars, needChars) => `[CWL TOPIC] the buffer (${id}) holds ${leaves} leaf/leaves, and their micros are ${microChars} characters — enough for the ${needChars} a topic must free. Open a topic NOW with cwl_group: pass those leaves, a name, and a description that already covers their FUTURE use, and one description then stands for all of them. This is the last moment it is possible: a leaf that enters a node can never be moved again, and the window closes with it.`,
    leavesDue: (missing, ids) => `[CWL INDEX] ${missing} leaf/leaves still have their BODY in the context and wait for a micro: ${ids} — call cwl_micro with the id and a micro of ~960 characters (≈240 tokens, ~160 words). The body leaves the context, the micro stands for it, and only then can the leaf enter a node. And while you label them: leaves that belong together can be grouped into a TOPIC with cwl_group — one description then stands for all of them, their labels leave the context, and the topic counts as a node like any other. Do it NOW, while those leaves still have their BODY in the context: once a leaf is inside a node it cannot be moved any more, and the window closes. It is the cheapest saving you can make.`,
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
          'cwl_compress_range(summary="<whole pieces, not a digest>", micro="<label of ~960 characters, ≈240 tokens>") ' +
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
      memories: 'cwl_memories: the memories that can be adopted, by name',
      adopt: 'cwl_adopt: forks another memory into this session',
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
      microText: 'The micro-summary that REPLACES the body in the context. The body is NOT touched: cwl_open still returns all of it. Write the pieces that matter; ~960 characters (≈240 tokens) is the size this design is built for — and the ceiling is measured: going over it is said in the result.',
      oldDesc: 'Writes the RIASSUNTONE and merges the oldest young nodes into the old node: their micros leave the context and the synthesis stands for all of them. The nodes and their leaves stay readable — cwl_open pages through them.',
      oldText: 'The synthesis (merge summary) that replaces, in the context, the content of the merged nodes: the descriptions of their TOPIC nodes and the labels of their ordinary ones — in short, what would otherwise be injected. Write WHOLE PIECES: it is what the agent will see instead of them, so SUMMARISE that content; do not list it, and do not glue it in. If the pit already holds a synthesis, cwl_open shows it: CARRY IT FORWARD — a merge REPLACES the pit synthesis, it does not add to it. If the pit content ends up SHORTER than your synthesis, the nodes enter the context as they are, in chronological order, and your text is not paid.',
      groupDesc: 'Groups leaves into a TOPIC node, which is born COLLAPSED: the name and the description you write now stand for its leaves in the index from this moment on, so the description must already cover the FUTURE use of the topic. Leaves can be taken from the buffer (the last node, attached to the leaves still open) and — unless `groupBeyondBuffer` is off — from any ORDINARY node: the leaves of an ordinary node are still in the head as micros, so the window does not close when the round ends. They must be a CHRONOLOGICAL BLOCK, because the topic is one block in the index: the nodes in between may be taken whole, and what the involved nodes keep is re-partitioned around the topic. A TOPIC is never a source — its description stands for its leaves. When the topic already exists, pass `node` and the description is NOT rewritten — that is what makes adding leaves later free.',
      groupLeaves: 'Ids of the leaves to group or move: the buffer\'s, the loose ones, and — unless `groupBeyondBuffer` is off — the leaves of ordinary nodes. Each must already carry a micro, and they must be consecutive in time.',
      groupNodes: 'Ids of NODES to absorb into the topic: a topic can CONTAIN other topics. Containment only goes BACKWARD in time (a node absorbs only nodes that come after it), and the buffer is never absorbed. The absorbed node keeps its leaves and its description and stays readable with cwl_open: what changes is that its block is no longer injected, because the parent description stands for it.',
      groupNode: 'Id of an existing TOPIC node to add the leaves to. Omit it to create a new topic, which then requires `name` and `description`.',
      groupName: 'Short name of the new topic, written ONCE: it is part of the index prefix, so it never changes. e.g. "login-otp".',
      groupText: 'Description of the new topic, written ONCE: it must already cover the future use of the topic, because it is what replaces the labels of its leaves in the index and it is never rewritten WHILE THE TOPIC IS OUTSIDE THE OLD NODE. The one exception: a topic that is already inside the old node may have its description rewritten, because there it no longer touches the index. Mandatory when creating.',
      groupPit: 'Catalogue these leaves INSIDE the old node instead of creating a topic at the frontier: every leaf must already be in the pit (a leaf at the frontier is refused), the minimum is 3 leaves, and there is no size guard, because the pit synthesis is left exactly as it is and nothing in the index moves. A pit topic is a way to NAME material that is already archived, not a way to save tokens. To put order in an archive that ALREADY exists, pass the id of a pit topic as `node` together with `pit: true`: the leaves move into it and its description may be rewritten. A node left empty is dropped, and every leaf has to end up somewhere: the pit synthesis is never touched either way.',
      oldDryRun: 'Preview the merge without doing it: dryRun: true reports the destination, the nodes and leaves involved, what would leave the context and what the merge needs, and records NOTHING.',
      groupDryRun: 'Preview the grouping without doing it: dryRun: true reports the destination, the leaves and nodes involved, the label characters against the floor, and records NOTHING. The validations run exactly as in a real call, so a dry run also tells you which refusal you would hit.',
      groupManyDesc: 'Apply SEVERAL groupings as ONE operation: every group is validated first (with the same code path as a single call), and only if they all pass is anything applied. All or nothing, and idempotent: repeating the same request does not duplicate, because a topic born from a set of leaves gets an id derived from those leaves, so the second request finds it already satisfied and says so.',
      groupGroups: 'The groups to apply, in order. Each one takes the same fields as a single call: leaves, nodes, node, name, description, pit.',
      recallEpisodeFull: 'false (default) returns a truncated preview; true returns the whole episode.',
      memoriesDesc: 'Lists the CWL memories that exist on this machine, by NAME, with their size. A memory is an archive of leaves, nodes and a pit that outlives the session that wrote it. Useful together with cwl_adopt: adopt needs a name.',
      adoptDesc: 'Forks ANOTHER memory into this session: the leaves, the nodes, the pit and the micros are copied into a NEW memory of its own (the source is never modified). The copied leaves are ARCHIVED — they carry no anchors in this transcript, so they can never be compressed again, they are never pruned, and their summary is the only copy that exists. Episodes do NOT travel: this session starts with an empty episode graph. Give a name with `as` so the fork can be found later.',
      adoptFromDesc: 'The memory to fork: its NAME as cwl_memories shows it, or the path of a session file.',
      adoptAsDesc: 'The name to give the new memory. Defaults to the name of the source plus a suffix.',
      compressRangeDesc: 'Compresses the OLDEST usable range of the conversation into your summary. YOU DO NOT pick the range and you do not need any hash: the extension already computed the address and holds it. Call it when an eviction marker or the budget demand tells you to compact, and write a summary good enough to keep working without re-reading the originals. Nothing inside the protected window is touched.',
      compressRangeSummary: 'The summary that REPLACES the compressed range. Write WHOLE PIECES, not a digest: it must be enough to keep working without re-reading them. Include paths, file names, function names, numeric values, and what you decided and why.',
      compressMicro: 'Your LABEL of ~960 characters (≈240 tokens) for this leaf: what it contains, detailed enough that the index can show it instead of the body. Write it HERE, while you have the messages in front of you — the leaf is then ready for a node and nothing will have to ask you for it later. A compression without a label stays valid: the extension will ask for it when the leaf is due to join a node.',
      compressCovered: (start: string, end: string) =>
        `Refused: the region ${start}..${end} is already inside a leaf of the index. Compressing it again would describe the same messages a second time, and the two descriptions would drift apart. Open the existing leaf with cwl_open, pick a region that is not covered, or use cwl_compress_range and let the extension choose.`,
      microOver: (where: string, chars: number, ceiling: number) =>
        `\n\nStored, but ${where} is ${chars} characters (~${Math.round(chars / 4)} tokens) against a ceiling of ${ceiling} (~350 tokens). The top of the index is made of these labels: measured on 42 real ones they cost ~722t each instead of ~300t, and the top cost 30k instead of ~12k. The label stays as you wrote it — next time write it shorter: the density that matters lives in the body, and cwl_open still returns all of it.`,
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
    consult: {
      findDesc: 'Searches the CWL memories and the transcript by keywords and returns the IDs. Use it BEFORE opening anything: a memory is found, not guessed. scope="index" (default) searches the labels of the leaves and the descriptions of the topics; scope="transcript" searches the conversation; scope="both" searches everything.',
      findQuery: 'Keywords of what you remember: a topic, a decision, a file name.',
      findScope: '"index" (labels and topic descriptions), "transcript" (the conversation), or "both".',
      findLimit: 'How many results (default 8, max 50).',
      findSnippet: 'cwl_find: find a memory or a message by keywords and get its ID (use before cwl_open)',
      findFound: (n, query, body) => `${n} result(s) for "${query}":\n${body}`,
      findNoMatch: (query) => `No result for "${query}". Try fewer or different words: labels are written in the language of the work.`,
      findNotLoaded: 'The recall index is not loaded in this session yet.',
      mapDesc: 'The map of the CWL index WITH the IDs: every node, its kind (pit, topic, buffer, legacy), how many leaves and children it holds, and what it costs in the head. Leaf IDs are listed only for the OPEN nodes (the buffer and the loose leaves); for a topic, read its leaves with cwl_node.',
      mapNode: 'Optional node id: show only that subtree. Without it, the whole index.',
      mapSnippet: 'cwl_map: the map of the CWL index with the node IDs',
      mapHeader: (nodes, leaves, chars) => `${nodes} node(s), ${leaves} leaf/leaves, head ~${chars}t`,
      mapLine: (id, kind, name, leaves, children, chars) => `${id} │ ${kind}${name ? ` "${name}"` : ''} │ ${leaves} leaf/leaves${children > 0 ? ` │ holds ${children} node(s)` : ''} │ ${chars} chars`,
      mapLeaves: (ids) => `  leaves: ${ids}`,
      nodeDesc: 'One node of the CWL index in full: metadata, the whole description, the nodes it holds, and its leaves as ID plus the first line of the label. Works on ANY node: a topic, a legacy node, the buffer, the pit.',
      nodeId: 'The node id, as shown by cwl_map or cwl_status.',
      nodeLimit: 'How many leaves to show (default: all). Use it with cursor to walk a long topic without pulling all of it into the context.',
      nodeCursor: 'Where to start: the nextCursor of the previous page (default: 0).',
      pendingLimit: 'How many nodes to show (default: all). Use it with cursor.',
      pendingCursor: 'Where to start: the nextCursor of the previous page (default: 0).',
      nodeSnippet: 'cwl_node: one CWL node in full, with the IDs of its leaves',
      nodeHeader: (id, kind, name, leaves, chars) => `${id} │ ${kind}${name ? ` "${name}"` : ''} │ ${leaves} leaf/leaves │ ${chars} chars`,
      nodeLeaf: (id, micro) => `  ${id} │ ${micro}`,
      nodeChild: (ids) => `holds: ${ids}`,
      nodeNotFound: (id) => `No node with id "${id}". List them with cwl_map.`,
      pendingDesc: 'What is NOT yet in a topic: the young legacy nodes and the buffer, with their leaves and the total characters. It is the inventory for creating a topic: if the total is below the needed size, a topic is refused.',
      pendingSnippet: 'cwl_pending: the leaves still to be ordered (young nodes + buffer), with the character count',
      pendingHeader: (nodes, leaves, chars, need) => `${nodes} node(s) still to order, ${leaves} leaf/leaves, ${chars} chars (a topic needs ~${need})`,
      pendingLine: (id, kind, leaves, chars) => `${id} │ ${kind} │ ${leaves} leaf/leaves │ ${chars} chars`,
      pendingEmpty: 'Nothing left to order: every leaf is inside a topic.',
      pendingPitHeader: (nodes, leaves) => `Already in the archive and still without a topic of their own: ${nodes} node(s), ${leaves} leaf/leaves.`,
      pendingPitLine: (id, leaves) => `  ${id} │ ${leaves} leaf/leaves`,
      pendingPitHint: 'To give these a name, call cwl_group with pit: true (and node: <an existing pit topic> to move them into one): nothing in the index moves and no synthesis is written, so there is no size guard — the minimum is 3 leaves.',
      pageInfo: (shown, total, next) => `  ... ${shown} of ${total} shown. Continue with cursor: ${next}`,
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
      `Per riaprire il riepilogo INTERO, chiama esattamente cwl_open({id: "${id}"}). ` +
      `Se ti serve il testo originale, chiama cwl_recall con parole chiave di quel contenuto.\n\n`,
    inheritedHead: (name, from, leaves, nodes) => `[CWL · MEMORIA EREDITATA] Questa sessione parte con una MEMORIA EREDITATA: ` +
      `"${name}", forketta da "${from}" — ${leaves} foglia/e archiviate e ${nodes} nodo/i copiati qui da UN'ALTRA trascrizione.\n` +
      `Questa sessione NON ha messaggi precedenti propri: l'indice in TESTA a questo contesto e' il suo passato — ` +
      `la sintesi del pozzo, le descrizioni dei topic, le etichette delle foglie che nessun topic tiene. ` +
      `Leggilo PRIMA di esplorare: cwl_map per la forma con gli id, cwl_node per paginare le foglie di un nodo, ` +
      `cwl_open per una foglia o il pozzo intero, cwl_recall per cercarli per parole chiave.\n` +
      `NON andare a cercare nel filesystem o nei vecchi log di sessione cio' che questa memoria gia' dice. ` +
      `Una foglia archiviata non viene piu' compressa ne' potato: per quelle foglie il riassunto e' l'UNICA copia che esiste.`,
    episodeClosed: (type, name) => `Episodio ${type} "${name}" chiuso.`,
    statusMeasured: (tokens, extra) => `Token contesto misurati: ~${tokens}${extra ? ` (${extra})` : ''}`,
    statusActive: (list) => `Attivi: ${list}`,
    statusStripped: (list) => `Stripped: ${list}`,
    evictionNotice: (dropped, truncated, from, to) => `CWL: ${dropped} evictati, ${truncated} ridotti (${from} -> ${to} token).`,
    compressMissingParams: 'startHash, endHash e summary sono tutti obbligatori.',
    compressRevoked: (start, end) => `Span ${start}..${end} ripristinato al testo integrale.`,
    compressUnknownHash: (from, to) => `Hash sconosciuto. startHash trovato: ${from}, endHash trovato: ${to}. Copiali verbatim dal contesto.`,
    compressApplied: (start, end) => `Compresso ${start}..${end} nel tuo riepilogo. L'originale resta su disco: recuperalo con cwl_recall.`,
    compressRangeApplied: (start, end, tokens, leafId) => `Compresso ${start}..${end} (~${tokens} token) nel tuo riepilogo. L'originale resta nel transcript: recuperalo con cwl_recall_episode o cwl_recall. Questo intervallo e' diventato la foglia ${leafId}: cwl_open("${leafId}") la legge intera, e cwl_group prende il suo id per raggrupparla con le foglie nate nello stesso giro.`,
    compressRangeNothing: "Non resta niente da comprimere: cio' che rimane e' dentro la finestra protetta oppure gia' compresso.",
    compressRangeNoSummary: "Il riassunto e' obbligatorio: e' l'UNICA parte di questa chiamata che devi scrivere tu.",
    statusRange: (tokens, start, end) => `Intervallo comprimibile: ~${tokens} token (${start}..${end})`,
    statusAddresses: (eligible, withId) => `Indirizzi: ${withId}/${eligible} messaggi-endpoint con un id stabile`,
    statusSpans: (n) => `Span di compressione tenuti: ${n}`,
    statusTopics: (n, names) => `Topic (${n}): ${names}`,
    statusIds: (ids) => `Id dei nodi: ${ids}`,
    statusLeafIds: (ids) => `Id delle foglie: ${ids}`,
    statusLooseIds: (ids) => `Foglie sciolte: ${ids}`,
    statusLeafIdsHint: 'Elenca anche gli ID delle FOGLIE del buffer e dei nodi giovani, piu\' le foglie sciolte. Spento di default: cwl_status viene chiamato spesso e il solo buffer puo\' tenere quattordici foglie, mentre gli id sono a un cwl_map di distanza.',
    oldTopicLine: (id, name, shape, taste) => `- ${id} \u00b7 TOPIC "${name}": ${shape} \u2014 ${taste}`,
    oldTooSmall: (leaves, microChars, needChars) =>
      `\n\nNon e\' stato registrato niente: le ${leaves} foglia/e che uscirebbero dal contesto tengono ${microChars} caratteri, e un accorpamento deve liberarne almeno ${needChars}. Un accorpamento COSTA una sintesi: la sintesi del pozzo viene riscritta, quindi cio\' che esce deve valere piu\' di cio\' che lo sostituisce. Comprimi altro prima, o accorpa quando i nodi sono pieni.`,
    indexLine: (pitNodes, pitLeaves, topicNodes, topicLeaves, bufferNodes, bufferLeaves, plainNodes, plainLeaves, loose, headTokens, evictions, savedTokens) =>
      `pozzo ${pitNodes}n/${pitLeaves}f │ topic ${topicNodes}n/${topicLeaves}f │ buffer ${bufferNodes}n/${bufferLeaves}f │ giovani ${plainNodes}n/${plainLeaves}f │ sciolte ${loose} │ testa ~${headTokens}t │ ${evictions} eviction │ ${savedTokens} risparmiati`,
    groupRefused: (why, detail) =>
      `\n\nRaggruppamento rifiutato (${why}${detail ? `: ${detail}` : ''}). Non e' stato registrato niente. Si possono raggruppare o spostare le foglie del buffer, quelle sciolte e quelle dei nodi ordinari (un TOPIC non e' una sorgente: la sua descrizione sta per le sue foglie), e devono essere consecutive nel tempo. Una foglia gia' dentro il nodo vecchio non puo' tornare indietro (la sintesi del pozzo sta per lei), e una foglia senza micro non comparirebbe in nessuna testa: scrivi prima il micro. Una descrizione e' IMMUTABILE finche' il suo topic e' FUORI dal nodo vecchio, perche' li' e' l'indice; dentro il nodo vecchio si puo' riscrivere, e li' non muove niente.`,
    groupTooSmall: (leaves, microChars, needChars) =>
      `\n\nNessun topic e' nato: le ${leaves} foglia/e tengono ${microChars} caratteri, e un topic deve liberarne almeno ${needChars}. Un topic nasce GIA' COLLASSATO — la sua descrizione sostituisce le etichette di quelle foglie da questo momento — quindi deve valere piu' della descrizione che le sostituisce (~3.600 caratteri per il rapporto, e mai meno del minimo assoluto). Raggruppa piu' foglie, oppure lasciale nel buffer finche' il topic non e' abbastanza grande.`,
    groupCreated: (name, id, leaves, microChars) =>
      `Topic "${name}" nato come ${id}: ${leaves} foglia/e (${microChars} caratteri di etichette) ora stanno dietro la tua descrizione, e i loro corpi restano leggibili con cwl_open. Si possono aggiungere altre foglie in qualsiasi momento, e non costa niente: la descrizione non cambia.`,
    pitTopicBorn: (name, id, leaves) =>
      `Topic "${name}" nato DENTRO il nodo vecchio come ${id}: ${leaves} foglia/e catalogate li'. La sintesi del pozzo NON e' stata toccata, e nell'indice non si e' mosso niente, perche' quelle etichette erano gia' nascoste dietro di essa. Puoi aggiungergli altre foglie e riscriverne la descrizione: dentro il pozzo non tocca piu' l'indice.`,
    groupAdded: (name, id, leaves) =>
      `Foglia/e aggiunte al topic "${name}" (${id}), che ora ne tiene ${leaves}.`,
    groupAbsorbed: (name, id, nodes, leaves, blocks) =>
      `\n\n"${name}" (${id}) ora CONTIENE ${nodes} nodo/i e ${leaves} foglia/e in tutto: i loro ${blocks} blocco/chi sono usciti dalla testa e la descrizione del genitore sta per loro. Niente e' stato cancellato — cwl_open("<id figlio>") li restituisce interi.`,
    groupDescriptionUpdated: () =>
      `\n\nLa descrizione e' stata riscritta, e questo NON cambia niente nell'indice: il nodo e' dentro il nodo vecchio, la cui sintesi e' stata scritta prima che queste foglie arrivassero. La sintesi del pozzo e' un'istantanea congelata; quella che hai appena scritto e' la copia viva, leggibile con cwl_open.`,
    topicHead: (id, name, leaves, saved) =>
      `[CWL \u00b7 TOPIC "${name}" (${id}) \u2014 ${leaves} foglia/e stanno dietro questa descrizione, che non cambia mai (~${saved} token risparmiati). Per aprire questo topic, chiama esattamente cwl_open({id: "${id}"}); per aprire una foglia, chiama cwl_open({id: "<id-foglia>"}).\n\n`,
    topicHolds: (nodes, leaves) =>
      `(contiene ${nodes} nodo/i, ${leaves} foglia/e in tutto \u2014 aprili con cwl_open("<id>"))\n`,
    oldTopicsConcatenated: (topics, chars) =>
      `\n\nLe descrizioni immutabili di ${topics} nodo/i topic sono state incollate a questa sintesi, parola per parola (${chars} caratteri): non vengono riscritte e non vanno riscritte.`,
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
    oldNodeSet: (id, nodes, leaves, microChars, tokens) => `Nodo vecchio ${id}: accorpati ${nodes} nodo/i, ${leaves} foglia/e (${microChars} caratteri di micro) in una sintesi di ~${tokens} token. Restano leggibili: cwl_open("${id}") elenca i nodi dentro.`,
    oldNotDue: (young, need) => `Nessun accorpamento: ${young} nodo/i giovane/i, si accorpa da ${need} in su. Non e' stato registrato niente.`,
    oldDryRun: (pass, destination, nodes, leaves, freedChars, needChars) => `PROVA, non e' stato registrato niente. Destinazione: ${destination}. Accorperebbe ${nodes} nodo/i e ${leaves} foglia/e, liberando ${freedChars} caratteri; l'accorpamento ne richiede ${needChars}. Esito: ${pass ? 'passerebbe' : 'sarebbe RIFIUTATO'}.`,
    groupManyDone: (groups, already, leaves, nodes) => `Lotto di ${groups} gruppo/i: ${groups - already} applicati, ${already} gia' soddisfatti (niente da fare), ${leaves} foglia/e spostate e ${nodes} nodo/i assorbiti. Validati TUTTI prima di toccare qualcosa, quindi niente e' stato applicato a meta'.`,
    groupDryRunAdd: (id, name, leaves, nodes, microChars, needChars) => `PROVA, non e' stato registrato niente. Il topic ${id} "${name}" riceverebbe ${leaves} foglia/e e ${nodes} nodo/i; tiene ${microChars} caratteri di etichette contro una soglia di ${needChars}.`,
    groupDryRunNew: (id, name, leaves, nodes, microChars, needChars) => `PROVA, non e' stato registrato niente. Nascerebbe un topic nuovo come ${id} "${name}", prendendo ${leaves} foglia/e e ${nodes} nodo/i; tiene ${microChars} caratteri di etichette contro una soglia di ${needChars}.`,
    oldHead: (id, nodes, tokens) => `[CWL NODO VECCHIO ${id} — ${nodes} nodo/i piu' vecchi accorpati dietro questa sintesi (~${tokens} token). Per aprire il pozzo e vedere cosa contiene, chiama esattamente cwl_open({id: "${id}"}); apri un nodo o una foglia elencata con cwl_open({id: "<id-dalla-pagina>"}).]\n\n`,
    oldHeadDescriptions: (id) => `[CWL NODO VECCHIO — quello che segue NON e' la sintesi del riassuntone ma le descrizioni piu' corte dei topic contenuti nel pozzo. Per leggere la sintesi intera e vedere cosa contiene, chiama cwl_open({id: "${id}"}).]\n\n`,
    oldPage: (id, nodes, tokens, body) => `[CWL nodo vecchio ${id} — ${nodes} nodo/i dentro, ~${tokens} token. Prima la sintesi, poi una riga per nodo con la sua FORMA; cwl_open("<id nodo>") ne apre uno, e le sue foglie si aprono intere.]\n\n${body}`,
    oldHot: (listed, total) => `--- Foglie piu' consultate (${listed} di ${total} nel nodo vecchio; niente e' stato cancellato, questo e' solo l'ordine di lettura) ---`,
    oldSupersededHead: (count) => `--- Sintesi sostituite da questa (${count}, dalla piu' recente): aprine una con cwl_open("<id>.s1") — cwl_old sostituisce la sintesi invece di estenderla, quindi queste restano leggibili invece di andare perse ---`,
    oldSupersededLine: (id, chars) => `- ${id}: ${chars} caratteri`,
    openBodyLost: (id) => `La foglia "${id}" non ha corpo: il suo riassunto e' uscito dalla memoria (la foglia e' dentro il pozzo) e il record su disco non c'e' piu'. Se la foglia ha una micro, e' quello che ne resta.`,
    memoriesEmpty: () => 'Non esiste ancora nessuna memoria CWL.',
    memoriesList: (rows) => `Memorie CWL:\n${rows}`,
    memoriesRow: (name, leaves, nodes, pit) => `- ${name} │ ${leaves} foglia/e │ ${nodes} nodo/i │ pozzo: ${pit}`,
    adoptDone: (name, from, leaves, nodes) => `Memoria "${name}" creata forkando "${from}": ${leaves} foglia/e, ${nodes} nodo/i copiati come ARCHIVIATI. Non hanno ancoraggi in questa sessione, quindi non verranno mai piu\' compresse ne\' potate — i loro riassunti sono l\'unica copia. Gli episodi non sono viaggiati. La memoria viene iniettata in testa al contesto da adesso in poi.`,
    adoptNotFound: (from, names) => `Nessuna memoria CWL corrisponde a "${from}". Quelle che esistono sono: ${names}.`,
    adoptBusy: (name) => `La memoria "${name}" e\' gia\' la memoria di questa sessione: non c\'e\' niente da forkare.`,
    adoptEmpty: (from) => `La memoria "${from}" non ha foglie: non c\'e\' niente da forkare.`,
    adoptOtherSession: 'Una memoria non si adotta mentre appartiene a una sessione VIVA: quella continua a scrivere la sua copia e le due si sovrascriverebbero. Chiudila prima, oppure adotta dal path del file di sessione.',
    cmdMemoriesDesc: () => 'Elenca le memorie CWL di questa macchina, per nome e dimensione.',
    cmdAdoptDesc: () => 'Forka una memoria in questa sessione: sceglila dalla lista, poi dai un nome al fork.',
    cmdPickSource: () => 'Scegli la memoria da forkare:',
    cmdPickName: () => 'Nome per il fork (vuoto = resta quello di default):',
    cmdCancelled: () => 'Non e\' stato adottato niente.',
    cmdNoAdoptable: () => 'Nessuna memoria adottabile su questa macchina.',
    cmdSaveDesc: () => 'Forza la compattazione totale (zero turni protetti) in una foglia finale per eredità o cambio topic.',
    cmdSaveNothing: () => 'Nessun messaggio recente da compattare: tutta la cronologia e\' gia\' dentro le foglie o non ci sono abbastanza messaggi.',
    cmdSaveTriggered: (tokens) => `Compattazione totale avviata: ~${tokens} token recenti verranno compressi nella foglia finale.`,
    cmdSavePrompt: (note) => `[COMANDO /cwl_save] Esegui subito la compattazione totale dell'intera cronologia recente (fino all'ultimo turno, zero turni protetti) chiamando cwl_compress_range con summary e micro accurati.${note ? `\nNota dell'operatore: "${note}"` : ''}`,
    cmdSaveDemand: (note) => `RICHIESTA DI COMPATTAZIONE TOTALE (/cwl_save): L'operatore ha richiesto di compattare l'intera cronologia recente in un'unica foglia (zero turni protetti). Chiama cwl_compress_range(summary="...", micro="...") descrivendo in dettaglio tutto il lavoro svolto finora e lo stato finale.${note ? ` Nota: "${note}"` : ''}`,
    oldSupersededPage: (id, chars, text) => `[CWL sintesi sostituita ${id} — ${chars} caratteri. E' una sintesi che quella ATTUALE del pozzo ha sostituito; cwl_old sostituisce invece di estendere, quindi l'archivio tiene il testo che altrimenti avrebbe cancellato. Nient'altro la referenzia.]\n\n${text}`,
    indexDue: (young, need) => `[CWL INDICE] ${young} nodo/i dell'indice sono da accorpare (si accorpa da ${need} in su). Chiama cwl_old col riassuntone: la tua sintesi sostituisce il contenuto dei nodi piu' vecchi — le descrizioni dei loro topic e i micro dei loro nodi sparsi — e le loro foglie restano leggibili con cwl_open.`,
    topicDue: (id, leaves, microChars, needChars) => `[CWL TOPIC] il buffer (${id}) tiene ${leaves} foglia/e, e i loro micro sono ${microChars} caratteri — abbastanza per i ${needChars} che un topic deve liberare. Apri un topic ADESSO con cwl_group: passa quelle foglie, un nome e una descrizione che copra gia' il loro uso FUTURO, e una descrizione sola sta per tutte. E' l'ultimo momento in cui si puo': una foglia che entra in un nodo non si sposta piu', e la finestra si chiude con lei.`,
    leavesDue: (missing, ids) => `[CWL INDICE] ${missing} foglia/e hanno ancora il CORPO nel contesto e aspettano un micro: ${ids} — chiama cwl_micro con l'id e un micro di ~960 caratteri (≈240 token, ~160 parole). Il corpo esce dal contesto, il micro lo rappresenta, e solo allora la foglia puo' entrare in un nodo. E mentre le etichetti: le foglie che vanno insieme si possono raggruppare in un TOPIC con cwl_group — una descrizione sola sta per tutte, le loro etichette escono dal contesto, e il topic conta come nodo come tutti gli altri. Fallo ADESSO, finche' quelle foglie hanno ancora il CORPO nel contesto: una volta entrata in un nodo, una foglia non si sposta piu' e la finestra si chiude. E' il risparmio piu' economico che hai.`,
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
          'cwl_compress_range(summary="<pezzi interi, non un sommario>", micro="<etichetta di ~960 caratteri, ≈240 token>") ' +
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
      memories: 'cwl_memories: le memorie adottabili, per nome',
      adopt: 'cwl_adopt: fork di un\'altra memoria in questa sessione',
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
      microText: "Il micro-riassunto che SOSTITUISCE il corpo nel contesto. Il corpo NON viene toccato: cwl_open lo restituisce ancora tutto. Scrivi i pezzi che contano; ~960 caratteri (≈240 token) e' la misura per cui questo design e' costruito — e il tetto e' misurato: sfondarlo viene detto nel risultato.",
      oldDesc: 'Scrive il RIASSUNTONE e accorpa i nodi giovani piu\' vecchi nel nodo vecchio: i loro micro escono dal contesto e la sintesi sta per tutti. I nodi e le loro foglie restano leggibili — cwl_open li pagina.',
      oldText: 'La sintesi (riassuntone) che sostituisce nel contesto il contenuto dei nodi accorpati: le descrizioni dei loro nodi TOPIC e i micro dei loro nodi sparsi — insomma, quello che altrimenti verrebbe iniettato. Scrivi PEZZI INTERI: e\' quello che l\'agente vedra\' al posto loro, quindi RIASSUMI quel contenuto; non elencarlo e non incollarlo. Se il pozzo ha gia\' una sintesi, cwl_open la mostra: PORTALA AVANTI — un accorpamento la SOSTITUISCE, non ci si aggiunge. Se il contenuto del pozzo risulta piu\' corto della tua sintesi, nel contesto entrano i nodi cosi\' come sono, in ordine cronologico, e il tuo testo non si paga.',
      groupDesc: 'Raggruppa le foglie in un nodo TOPIC, che nasce GIA\' COLLASSATO: il nome e la descrizione che scrivi adesso stanno per le sue foglie nell\'indice da questo momento, quindi la descrizione deve coprire GIA\' l\'uso futuro del topic. Si possono raggruppare o spostare le foglie del buffer (l\'ultimo nodo, attaccato alle foglie ancora aperte) e — se `groupBeyondBuffer` e\' attivo — quelle di qualsiasi nodo ORDINARIO: le foglie di un nodo ordinario sono ancora in testa come micro, quindi la finestra non si chiude a fine round. Devono essere un BLOCCO CRONOLOGICO, perche\' il topic e\' un blocco solo nell\'indice: i nodi in mezzo si possono prendere interi, e cio\' che i nodi coinvolti tengono viene ripartito attorno al topic. Un TOPIC non e\' mai una sorgente — la sua descrizione sta per le sue foglie. Quando il topic esiste gia\', passa `node` e la descrizione NON viene riscritta: e\' questo che rende gratis l\'aggiunta di foglie.',
      groupLeaves: 'Id delle foglie da raggruppare o spostare: quelle del buffer, quelle sciolte e — se `groupBeyondBuffer` e\' attivo — quelle dei nodi ordinari. Ognuna deve gia\' avere un micro, e devono essere consecutive nel tempo.',
      groupNodes: 'Id dei NODI da assorbire nel topic: un topic puo\' CONTENERE altri topic. Il contenimento va SOLO all\'indietro nel tempo (un nodo assorbe solo nodi che stanno dopo di lui) e il buffer non si assorbe mai. Il nodo assorbito tiene le sue foglie e la sua descrizione e resta leggibile con cwl_open: cambia solo che il suo blocco non viene piu\' iniettato, perche\' la descrizione del genitore sta per lui.',
      groupNode: 'Id di un nodo TOPIC esistente a cui aggiungere le foglie. Omesso, crea un topic nuovo, che allora richiede `name` e `description`.',
      groupName: 'Nome breve del topic nuovo, scritto UNA VOLTA: fa parte del prefisso dell\'indice, quindi non cambia mai. Es. "login-otp".',
      groupText: 'Descrizione del topic nuovo, scritta UNA VOLTA: deve coprire gia\' l\'uso futuro del topic, perche\' e\' quello che sostituisce le etichette delle sue foglie nell\'indice e non viene mai riscritta FINCHE\' IL TOPIC E\' FUORI DAL NODO VECCHIO. Unica eccezione: un topic gia\' dentro il nodo vecchio puo\' avere la descrizione riscritta, perche\' li\' non tocca piu\' l\'indice. Obbligatoria alla creazione.',
      groupPit: 'Cataloga queste foglie DENTRO il nodo vecchio invece di creare un topic sulla frontiera: ogni foglia deve essere gia\' nel pozzo (una foglia sulla frontiera viene rifiutata), il minimo sono 3 foglie, e non c\'e\' guard di dimensione, perche\' la sintesi del pozzo resta esattamente com\'e\' e nell\'indice non si muove niente. Un topic nel pozzo serve a DARE UN NOME a materiale gia\' archiviato, non a risparmiare token. Per mettere ordine in un archivio che esiste GIA\', passa l\'id di un topic del pozzo come `node` insieme a `pit: true`: le foglie si spostano dentro di esso e la sua descrizione si puo\' riscrivere. Un nodo lasciato vuoto viene eliminato, e ogni foglia deve finire da qualche parte: la sintesi del pozzo in nessuno dei due casi viene toccata.',
      oldDryRun: 'Anteprima dell\'accorpamento senza farlo: dryRun: true riporta la destinazione, i nodi e le foglie coinvolte, cosa uscirebbe dal contesto e cosa serve all\'accorpamento, e NON registra niente.',
      groupDryRun: 'Anteprima del raggruppamento senza farlo: dryRun: true riporta la destinazione, le foglie e i nodi coinvolti, i caratteri di etichette contro la soglia, e NON registra niente. Le validazioni girano esattamente come in una chiamata vera, quindi la prova dice anche quale rifiuto incontreresti.',
      groupManyDesc: 'Applica PIU\' raggruppamenti come UNA sola operazione: ogni gruppo viene validato prima (con lo stesso percorso di codice di una chiamata singola), e solo se passano tutti si applica qualcosa. Tutto o niente, e idempotente: ripetere la stessa richiesta non duplica, perche\' un topic nato da un insieme di foglie prende un id derivato da quelle foglie, quindi la seconda richiesta lo trova gia\' soddisfatto e lo dice.',
      groupGroups: 'I gruppi da applicare, in ordine. Ognuno prende gli stessi campi di una chiamata singola: leaves, nodes, node, name, description, pit.',
      recallEpisodeFull: 'false (default) restituisce un estratto troncato; true restituisce l\'episodio intero.',
      memoriesDesc: 'Elenca le memorie CWL che esistono su questa macchina, per NOME, con la loro dimensione. Una memoria e\' un archivio di foglie, nodi e pozzo che sopravvive alla sessione che l\'ha scritto. Utile insieme a cwl_adopt: l\'adozione vuole un nome.',
      adoptDesc: 'Fa il fork di UN\'ALTRA memoria dentro questa sessione: foglie, nodi, pozzo e micro vengono copiati in una memoria NUOVA (la sorgente non viene mai modificata). Le foglie copiate sono ARCHIVIATE: non hanno ancoraggi in questo transcript, quindi non verranno mai piu\' compresse ne\' potate, e il loro riassunto e\' l\'unica copia che esiste. Gli episodi NON viaggiano: questa sessione parte con un grafo vuoto. Da\' un nome con `as` per poter ritrovare il fork.',
      adoptFromDesc: 'La memoria da forkare: il suo NOME come lo mostra cwl_memories, oppure il path di un file di sessione.',
      adoptAsDesc: 'Il nome da dare alla nuova memoria. Default: il nome della sorgente piu\' un suffisso.',
      compressRangeDesc: "Comprime nel tuo riassunto l'intervallo PIU' VECCHIO utilizzabile della conversazione. NON scegli tu l'intervallo e non ti serve nessun hash: l'estensione ha gia' calcolato e tiene l'indirizzo. Chiamalo quando un marker di eviction o la richiesta di budget ti dicono di compattare, e scrivi un riassunto che basti a lavorare senza rileggere gli originali. Nulla dentro la finestra protetta viene toccato.",
      compressRangeSummary: "Il riassunto che SOSTITUISCE l'intervallo compresso. Scrivi PEZZI INTERI, non un sommario: deve bastare a lavorare senza rileggere. Includi path, nomi di file, nomi di funzione, valori numerici e cosa hai scelto e perche'.",
      compressMicro: "La tua ETICHETTA di ~960 caratteri (≈240 token) per questa foglia: cosa contiene, con dettaglio sufficiente perche' l'indice la possa mostrare al posto del corpo. Scrivila QUI, mentre hai i messaggi davanti — la foglia e' cosi' pronta per un nodo e nessuno dovra' chiedertela dopo. Una compressione senza etichetta resta valida: l'estensione te la chiedera' quando la foglia dovra' entrare in un nodo.",
      compressCovered: (start: string, end: string) =>
        `Rifiutato: la regione ${start}..${end} e' gia' dentro una foglia dell'indice. Comprimerla di nuovo descriverebbe due volte gli stessi messaggi, e le due descrizioni divergerebbero. Apri la foglia esistente con cwl_open, scegli una regione non coperta, oppure usa cwl_compress_range e lascia che l'intervallo lo scelga l'estensione.`,
      microOver: (where: string, chars: number, ceiling: number) =>
        `\n\nRegistrata, ma ${where} e' ${chars} caratteri (~${Math.round(chars / 4)} token) contro un tetto di ${ceiling} (~350 token). La cima dell'indice e' fatta di queste etichette: misurate su 42 reali costavano ~722t l'una invece di ~300t, e la cima 30k invece di ~12k. L'etichetta resta quella che hai scritto — la prossima volta scrivila piu' corta: la densita' che conta sta nel corpo, e cwl_open lo restituisce ancora tutto.`,
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
    consult: {
      findDesc: 'Cerca nelle memorie CWL e nel transcript per parole chiave e restituisce gli ID. Usalo PRIMA di aprire qualcosa: una memoria si trova, non si indovina. scope="index" (default) cerca le etichette delle foglie e le descrizioni dei topic; scope="transcript" cerca la conversazione; scope="both" cerca in entrambi.',
      findQuery: 'Parole chiave di cio\' che ricordi: un argomento, una decisione, un nome di file.',
      findScope: '"index" (etichette e descrizioni dei topic), "transcript" (la conversazione), oppure "both".',
      findLimit: 'Quanti risultati (default 8, massimo 50).',
      findSnippet: 'cwl_find: trova una memoria o un messaggio per parole chiave e d\u00e0 il suo ID (usalo prima di cwl_open)',
      findFound: (n, query, body) => `${n} risultato/i per "${query}":\n${body}`,
      findNoMatch: (query) => `Nessun risultato per "${query}". Prova con meno parole o parole diverse: le etichette sono scritte nella lingua del lavoro.`,
      findNotLoaded: 'L\'indice di recall non \u00e8 ancora caricato in questa sessione.',
      mapDesc: 'La mappa dell\'indice CWL CON gli ID: ogni nodo, il suo tipo (pozzo, topic, buffer, legacy), quante foglie e quanti nodi contiene, e quanto costa in testa. Gli id delle foglie compaiono solo per i nodi APERTI (il buffer e le foglie sciolte); per un topic, leggi le sue foglie con cwl_node.',
      mapNode: 'Id opzionale di un nodo: mostra solo quel sottoalbero. Senza, tutto l\'indice.',
      mapSnippet: 'cwl_map: la mappa dell\'indice CWL con gli id dei nodi',
      mapHeader: (nodes, leaves, chars) => `${nodes} nodo/i, ${leaves} foglia/e, testa ~${chars}t`,
      mapLine: (id, kind, name, leaves, children, chars) => `${id} \u2502 ${kind}${name ? ` "${name}"` : ''} \u2502 ${leaves} foglia/e${children > 0 ? ` \u2502 contiene ${children} nodo/i` : ''} \u2502 ${chars} caratteri`,
      mapLeaves: (ids) => `  foglie: ${ids}`,
      nodeDesc: 'Un nodo dell\'indice CWL per intero: metadati, descrizione completa, i nodi che contiene, e le sue foglie come ID piu\' la prima riga dell\'etichetta. Funziona su QUALSIASI nodo: un topic, un nodo legacy, il buffer, il pozzo.',
      nodeId: 'L\'id del nodo, come lo mostrano cwl_map o cwl_status.',
      nodeLimit: 'Quante foglie mostrare (default: tutte). Usalo con cursor per scorrere un topic lungo senza tirarlo tutto nel contesto.',
      nodeCursor: 'Da dove partire: il nextCursor della pagina precedente (default: 0).',
      pendingLimit: 'Quanti nodi mostrare (default: tutti). Usalo con cursor.',
      pendingCursor: 'Da dove partire: il nextCursor della pagina precedente (default: 0).',
      nodeSnippet: 'cwl_node: un nodo CWL per intero, con gli ID delle sue foglie',
      nodeHeader: (id, kind, name, leaves, chars) => `${id} \u2502 ${kind}${name ? ` "${name}"` : ''} \u2502 ${leaves} foglia/e \u2502 ${chars} caratteri`,
      nodeLeaf: (id, micro) => `  ${id} \u2502 ${micro}`,
      nodeChild: (ids) => `contiene: ${ids}`,
      nodeNotFound: (id) => `Nessun nodo con id "${id}". Elencali con cwl_map.`,
      pendingDesc: 'Cio\' che NON \u00e8 ancora in un topic: i nodi young legacy e il buffer, con le loro foglie e i caratteri totali. \u00c8 l\'inventario per creare un topic: se il totale \u00e8 sotto la soglia, il topic viene rifiutato.',
      pendingSnippet: 'cwl_pending: le foglie ancora da mettere in ordine (nodi young + buffer), con i caratteri',
      pendingHeader: (nodes, leaves, chars, need) => `${nodes} nodo/i ancora da ordinare, ${leaves} foglia/e, ${chars} caratteri (un topic ne serve ~${need})`,
      pendingLine: (id, kind, leaves, chars) => `${id} \u2502 ${kind} \u2502 ${leaves} foglia/e \u2502 ${chars} caratteri`,
      pendingEmpty: 'Non resta niente da ordinare: ogni foglia \u00e8 dentro un topic.',
      pendingPitHeader: (nodes, leaves) => `Gia' nel pozzo e ancora senza un topic proprio: ${nodes} nodo/i, ${leaves} foglia/e.`,
      pendingPitLine: (id, leaves) => `  ${id} \u2502 ${leaves} foglia/e`,
      pendingPitHint: 'Per dargli un nome chiama cwl_group con pit: true (e node: <un topic del pozzo> per spostarle dentro): nell\'indice non si muove niente e non si scrive nessuna sintesi, quindi non c\'e\' guard di dimensione — il minimo sono 3 foglie.',
      pageInfo: (shown, total, next) => `  ... ${shown} di ${total} mostrate. Continua con cursor: ${next}`,
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
  /**
   * Whether `cwl_group` may take leaves that already sit inside an ORDINARY node, not just
   * the buffer's or the loose ones. On by default, and disableable: the leaves of an ordinary
   * node are still in the head as micros (a TOPIC stands for its leaves with one description,
   * an ordinary node injects them), so they are in fact loose — and the window that today
   * closes when the round ends is the reason a topic had to be born as the last act of a
   * round. What it costs is the index prefix: the topic is born at the position of the first
   * node involved, so the head is rewritten from there. Turn it off to keep the old rule.
   */
  groupBeyondBuffer: boolean;
  nodeCapacity: number;
  mergeNodesAt: number;
  /** How many times the synthesis a merge frees must outweigh the one it writes. */
  mergeMinRatio: number;
  /** And the absolute floor, in characters: below this a merge is refused whatever the ratio says. */
  mergeMinChars: number;
  /**
   * Past how many leaves of the BUFFER the agent is invited to open a topic.
   *
   * The buffer is the LAST node and the only one whose leaves can still be moved:
   * a leaf that has entered a node can never be moved again, so a topic that is not
   * born while its leaves are still in the buffer is lost for good. This is why the
   * invitation exists at all — the index itself closes the window it depends on.
   * It is a preference, like `nodeCapacity`: the number only decides WHEN the agent
   * is told, and the size guard decides whether the group is worth a description.
   */
  topicInviteAt: number;
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
  // The compression trigger is `tokenBudget * thresholdRatio`, and it is set to
  // land at 100.000t. The budget IS the trigger and the ratio is 1, so the number
  // the operator reasons about appears literally in the config instead of hiding
  // inside a product. Note that `thresholdRatio` is validated in (0, 1]: an
  // attempt to express 100k as `80_000 * 1.25` would be silently rejected and
  // fall back on the default, which is the worst possible failure mode for a
  // setting that decides when compaction starts.
  tokenBudget: 100_000,
  thresholdRatio: 1,
  // Three turns, not ten. The window is a CONTINUOUS RUN of the list: the last N
  // user turns PLUS everything between and after them — every assistant message,
  // reasoning block, tool call and tool output. An operator who writes little
  // ends up protecting mostly the agent's own text, and MEASURED on a real
  // session ten turns left only ~6.307t compressible out of ~184.000t: the safety
  // window WAS the budget, and the extension could not say so.
  //
  // It went 4 -> 3 for a reason worth recording: the protected window is a FLOOR,
  // not a ceiling. On a session whose turns are dense, the floor alone can sit
  // above the trigger — measured at ~111.792t against a 68.000t trigger, with
  // `evictable: 0` and the index head at only ~4.344t. The extension then asks
  // to compact something that does not exist, and an agent that does not
  // understand the floor burns turns hunting for it. Lowering N is the only knob
  // that moves that floor.
  protectedTurns: 3,
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
  looseLeaves: 4,
  nodeCapacity: 30,
  mergeNodesAt: 3,
  // 1.8, not 3: the ratio is the guard a merge AND a topic both pass, and at 3 it asked a
  // topic for 3 x 3,600 = 10,800 characters of micro before one could be born — high
  // enough that topics only appeared for the biggest rounds, or late. At 1.8 the floor is
  // 6,480. MEASURED consequence, accepted deliberately: the same formula is `cwl_old`'s
  // guard (against the REAL synthesis length, floored at MERGE_SYNTHESIS_CHARS), so merges
  // into the pit become 40% easier too.
  mergeMinRatio: 1.8,
  mergeMinChars: 6_000,
  // 18 of the 30 leaves a node holds. Early enough that the leaves are still in the
  // buffer and can be moved, late enough that the group already passes the size guard
  // with the micros it holds. Raising it trades the invitation for a fuller node.
  topicInviteAt: 18,
  gate: true,
  levels: {
    stripReasoning: true,
    stripBulkOutput: true,
    stripIntermediate: true,
    removeEpisode: true,
  },
  groupBeyondBuffer: true,
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
        // At least 1: the agent can archive EARLIER than the index asks (and then it
        // takes the node in progress too). A merge of one node into itself is still not
        // a merge: with `young.length === 1` that node is all there is, and the size
        // guards below decide whether it is worth a synthesis.
        mergeNodesAt: validNumber(user.mergeNodesAt, 1, 10_000, DEFAULT_CONFIG.mergeNodesAt),
        mergeMinRatio: validNumber(user.mergeMinRatio, 0, 1_000, DEFAULT_CONFIG.mergeMinRatio),
        mergeMinChars: validNumber(user.mergeMinChars, 0, 10_000_000, DEFAULT_CONFIG.mergeMinChars),
        topicInviteAt: validNumber(user.topicInviteAt, 0, 10_000, DEFAULT_CONFIG.topicInviteAt),
        gate: validBool(user.gate, DEFAULT_CONFIG.gate),
        levels: {
          stripReasoning: validBool(levels.stripReasoning, DEFAULT_CONFIG.levels.stripReasoning),
          stripBulkOutput: validBool(levels.stripBulkOutput, DEFAULT_CONFIG.levels.stripBulkOutput),
          stripIntermediate: validBool(levels.stripIntermediate, DEFAULT_CONFIG.levels.stripIntermediate),
          removeEpisode: validBool(levels.removeEpisode, DEFAULT_CONFIG.levels.removeEpisode),
        },
        groupBeyondBuffer: validBool(user.groupBeyondBuffer, DEFAULT_CONFIG.groupBeyondBuffer),
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
// Token estimation (dynamic calibration based on provider usage)
// ---------------------------------------------------------------------------

/** Default char-to-token ratio (4.0 preserves byte-length fixture math until real usage arrives). */
const DEFAULT_CHAR_TOKEN_RATIO = 4.0;
const MIN_CHAR_TOKEN_RATIO = 1.8;
const MAX_CHAR_TOKEN_RATIO = 5.0;

/** Token estimate: characters divided by char-to-token ratio (default 3.2, or dynamically calibrated). */
function estimateTokens(text: string, ratio: number = DEFAULT_CHAR_TOKEN_RATIO): number {
  const r = typeof ratio === 'number' && Number.isFinite(ratio) && ratio > 0 ? ratio : DEFAULT_CHAR_TOKEN_RATIO;
  return Math.ceil(text.length / r);
}

function estimateMessageTokens(msg: unknown, ratio: number = DEFAULT_CHAR_TOKEN_RATIO): number {
  try {
    const s = JSON.stringify(msg);
    return estimateTokens(s, ratio);
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
   * The OLD node — the pit — once the agent has written its merge summary.
   *
   * It holds young nodes instead of leaves, so `cwl_open` can still page through
   * them one by one; what its leaves lose is the CONTEXT: their micros are no
   * longer injected individually, because the merge summary stands for all of them.
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
   * Whether the demand was APPENDED to the context in this turn. `turn_end` counts an
   * attempt only when this is true: a turn where the demand was withheld is not a
   * failure of the agent, and charging it as one is how the gate gave up on a request
   * it had never delivered.
   */
  demandShown: boolean;
  /**
   * Why the demand was withheld the last time it was impossible, or '' when the
   * last check found something doable. Not bookkeeping for its own sake: the
   * silence of a withheld demand proved nothing once, and a diagnosis died on it.
   */
  gateWithheld: string;
  /**
   * Endpoints of the largest range `cwl_compress_range` may compress right now,
   * recomputed on every context hook. Deliberately NOT persisted: they describe
   * the CURRENT message list, and a stale pair reloaded in another process would
   * point at addresses that no longer exist.
   */
  rangeStartHash: string | null;
  rangeEndHash: string | null;
  /**
   * Timestamp of the last assistant message whose `usage` was already logged. The
   * context hook runs once per REQUEST, so a tool loop would log the same request
   * many times: the timestamp is what makes the row one per request.
   */
  lastUsageTs: number;
  /**
   * The event the NEXT request follows: `new-leaf` when a range was just applied,
   * `pit-rewritten` when the merge rewrote the pit, `none` otherwise. It is what a
   * CACHE row needs to be chargeable: without it, a write is a number without a
   * cause.
   */
  lastEvent: string;
  /**
   * Turns since the last leaf was written. A cache miss is the PRICE of that leaf,
   * and the price only means something next to how often we paid it.
   */
  turnsSinceCompress: number;
  /**
   * Dynamically calibrated char-to-token ratio, learned from assistant usage reports.
   * Defaults to 3.2 (realistic for Italian prose, code and JSON).
   */
  charTokenRatio: number;
  /**
   * Incompressible system overhead (System Prompt + Tool JSON Schemas) in tokens,
   * derived from `promptTokens - estimatedMessageTokens` when an assistant usage is observed.
   */
  systemOverheadTokens: number;
  /**
   * Last total context tokens reported by provider/Pi (promptTokens + output or getContextUsage).
   */
  providerContextTokens: number;
  /** Calibration baseline bookkeeping for Δchars / Δtokens calculation. */
  lastCalibChars: number;
  lastCalibTokens: number;
  /**
   * The NAME of this memory, so a memory can be found and adopted by a human-readable name
   * instead of a hash of a session path. Optional: a state written before names existed simply
   * has none, and `defaultMemoryName` derives one when it is saved.
   */
  memoryName?: string;
  /**
   * The name of the memory this one was ADOPTED from, when it was. A fork carries this so the
   * two branches can be told apart later; it is a trace, never a link — the fork owns its own
   * leaves from the moment it is created.
   */
  importedFrom?: string;
  /**
   * Where the DEEP bodies live: leaf id -> [byte offset, byte length] in the append-only
   * bodies file. Only the leaves INSIDE the pit have their summary moved there (the level-0
   * rule: everything outside the pit keeps micro + full summary in RAM). A summary, once
   * written, is never rewritten, so an offset never moves.
   */
  bodies: Map<string, [number, number]>;
  /** Tokens held by that range: shown in the status and in the demand. */
  rangeTokens: number;
  /**
   * The FIRST loose leaf: an index into `st.spans`. The loose set is everything from
   * here to the end, and this number only ever GROWS — that monotonicity is the whole
   * point. Before it, "loose" was `spans.slice(-looseLeaves)`, a window that follows
   * the tail: a leaf that left it could walk back IN as soon as four newer leaves
   * existed, which would put its body back in the context and pay for it twice.
   *
   * It is also where batching lives. Closing a leaf in micro is not a write: it is this
   * frontier moving. Advancing it by ONE per new leaf closes one leaf at a time — one
   * invalidation each. Advancing it so that only the newest `LOOSE_AFTER_BATCH` stay
   * open closes several at once, and the invalidation is already paid by the leaf that
   * was just written, so the closing itself is FREE.
   */
  looseFrom: number;
  /**
   * Set by /cwl_save (or /cwd_save): forces total compaction with zero protected
   * turns on the next context pass, so the entire history up to the end enters a final leaf.
   */
  forceAllNext?: boolean;
  /** Optional note provided by the operator to /cwl_save. */
  forceAllNote?: string;
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
    gateWithheld: '',
    demandShown: false,
    rangeStartHash: null,
    rangeEndHash: null,
    lastUsageTs: 0,
    lastEvent: 'none',
    turnsSinceCompress: -1,
    charTokenRatio: DEFAULT_CHAR_TOKEN_RATIO,
    systemOverheadTokens: 0,
    providerContextTokens: 0,
    lastCalibChars: 0,
    lastCalibTokens: 0,
    bodies: new Map(),
    rangeTokens: 0,
    // 0 = every span is loose. A state restored from disk clamps it (see
    // `loadPersistedState`); a fresh state has no spans, so 0 is the honest value.
    looseFrom: 0,
    forceAllNext: false,
    forceAllNote: undefined,
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
  /** The memory's name, and where it came from when it is a fork. See `CwlState`. */
  name?: string;
  importedFrom?: string;
  /**
   * The pid of the session that wrote this state. It is what makes adoption safe: a memory
   * whose owner is STILL ALIVE cannot be forked, because that session keeps saving its own
   * copy and the two branches would overwrite each other.
   */
  ownerPid?: number;
  /** The deep bodies, leaf id -> [offset, len] in the append-only bodies file. Optional. */
  bodies?: Record<string, [number, number]>;
  /** The pit. Persisted because it carries the agent's merge summary, which no rule can recompute. */
  oldNode?: OldNode | null;
  /**
   * The node structure: which leaves are grouped, and the name/description of the topics.
   * NOTHING here is recomputable. `refreshNodes` can re-form nodes out of the leaves, but it
   * cannot invent a topic's name; and without the pit's node ids the old-node page comes back
   * EMPTY while the leaves it absorbed are re-formed as young ones — the head grows again
   * under a synthesis that already stands for them.
   */
  nodes?: SpanNode[];
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
  /** Optional on load: a state written before this field existed has none. */
  gateWithheld?: string;
  /** Optional on load: a state written before this field existed has none. */
  looseFrom?: number;
  charTokenRatio?: number;
  systemOverheadTokens?: number;
  providerContextTokens?: number;
}

const STATE_VERSION = 1;
/** Cap on persisted hashes: they are an anchor for cwl_compress, not an archive. */
const MAX_PERSISTED_HASHES = 2000;
/** State files older than this are removed at session start. */
const STATE_MAX_AGE_MS = 14 * 24 * 60 * 60 * 1000;

/** One file per session, named by the hash of the key (the key is a path). */
/**
 * Is that process still running? Signal 0 asks the kernel without delivering anything.
 *
 * It is what makes adoption safe: a stored `ownerPid` that is still alive means the session
 * that wrote that memory is STILL writing it, and a fork would give two writers to one past.
 */
function isPidAlive(pid: number): boolean {
  if (!Number.isInteger(pid) || pid <= 0) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

/** The append-only store of the deep bodies, next to the state file that indexes it. */
function bodiesPath(key: string): string {
  return statePath(key).replace(/\.json$/, '.bodies.jsonl');
}

/**
 * The leaves the PIT stands for, computed straight from the state.
 *
 * Not `pitView`: that one returns null while the synthesis is empty, but a leaf can be inside
 * the pit before (or without) a synthesis, and its body must still be movable. The pit's
 * nodes own the leaves; nothing else matters.
 */
function pitLeafIds(st: CwlState): Set<string> {
  const inPit = new Set(st.oldNode?.nodes ?? []);
  const out = new Set<string>();
  for (const nd of st.nodes) {
    if (!inPit.has(nd.id)) continue;
    for (const id of nd.leaves) out.add(id);
  }
  return out;
}

/** One leaf's body record, read at its stored offset: ONE read, never the whole file. */
function readBody(key: string, st: CwlState, id: string): { text: string; micro?: string } | null {
  const rec = st.bodies.get(id);
  if (!rec) return null;
  try {
    const fd = fs.openSync(bodiesPath(key), 'r');
    try {
      const buf = Buffer.alloc(rec[1]);
      const n = fs.readSync(fd, buf, 0, rec[1], rec[0]);
      const parsed = JSON.parse(buf.toString('utf8', 0, n)) as { text?: string; micro?: string };
      if (typeof parsed.text !== 'string' || !parsed.text) return null;
      return { text: parsed.text, micro: typeof parsed.micro === 'string' ? parsed.micro : undefined };
    } finally { fs.closeSync(fd); }
  } catch { return null; }
}

/**
 * The level-0 rule, applied: the bodies of the leaves INSIDE the pit leave the RAM and move
 * to the append-only bodies file. Idempotent, and ordered so that nothing can be lost: the
 * line is appended and its offset recorded BEFORE the in-RAM summary is blanked, so a crash
 * at any point leaves the summary in one of the two places.
 */
function moveLeafBodiesToDisk(key: string, st: CwlState): void {
  const ids = pitLeafIds(st);
  if (ids.size === 0) return;
  let fd: number | null = null;
  try {
    for (const sp of st.spans) {
      const id = idOfSpan(sp);
      if (!ids.has(id)) continue;
      if (!sp.summary) continue; // already moved, or never had a body
      if (st.bodies.has(id)) continue;
      if (fd === null) fd = fs.openSync(bodiesPath(key), 'a');
      const line = `${JSON.stringify({ id, text: sp.summary })}\n`;
      const offset = fs.fstatSync(fd).size;
      fs.writeSync(fd, line);
      st.bodies.set(id, [offset, Buffer.byteLength(line)]);
      sp.summary = '';
    }
  } catch { /* on failure the summary stays in RAM and the next save retries */ }
  finally {
    if (fd !== null) { try { fs.closeSync(fd); } catch { /* already closed */ } }
  }
}

function statePath(key: string): string {
  return path.join(STATE_DIR, `${createHash('sha256').update(key).digest('hex').slice(0, 32)}.json`);
}

/**
 * A readable name for a memory that was never named: the session file without its extension.
 *
 * The state is keyed by a PATH, and a path is not something anyone can say out loud; the name
 * is what makes a memory findable (see `cwl_memories`) and adoptable (see `cwl_adopt`). It is
 * derived, not invented: a fork names itself after the memory it came from.
 */
function defaultMemoryName(key: string): string {
  const base = key.split(/[\\/]/).pop() ?? key;
  const clean = base.replace(/\.jsonl$/, '').replace(/[^a-zA-Z0-9._-]/g, '-');
  return clean.slice(0, 48) || 'memory';
}

function saveState(key: string, st: CwlState): void {
  try {
    fs.mkdirSync(STATE_DIR, { recursive: true });
    // The level-0 rule: before writing, move the pit's leaf bodies out of the state. Append
    // FIRST, blank AFTER — whatever the crash order, the summary survives in one of the two.
    moveLeafBodiesToDisk(key, st);
    const payload: PersistedState = {
      version: STATE_VERSION,
      key,
      savedAt: Date.now(),
      name: st.memoryName ?? defaultMemoryName(key),
      importedFrom: st.importedFrom,
      ownerPid: process.pid,
      bodies: Object.fromEntries(st.bodies),
      graph: { episodes: st.graph.all },
      spans: st.spans,
      looseFrom: st.looseFrom,
      graves: st.graves.slice(-GRAVE_MAX),
      oldNode: st.oldNode,
      nodes: st.nodes,
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
      gateWithheld: st.gateWithheld,
      charTokenRatio: st.charTokenRatio,
      systemOverheadTokens: st.systemOverheadTokens,
      providerContextTokens: st.providerContextTokens,
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
    // The frontier only ever GROWS, and a state written before this field existed has
    // none. Defaulting to the OLD rule's boundary is what keeps an upgrade from
    // opening every leaf body at once (and paying to cache it). `DEFAULT_CONFIG` is
    // read rather than a literal so the boundary and the config cannot drift.
    st.looseFrom = typeof data.looseFrom === 'number'
      ? Math.max(0, Math.min(data.looseFrom, st.spans.length))
      : Math.max(0, st.spans.length - DEFAULT_CONFIG.looseLeaves);
    st.graves = Array.isArray(data.graves) ? (data.graves as Grave[]) : [];
    st.oldNode = data.oldNode && typeof data.oldNode.id === 'string' ? data.oldNode : null;
    // Restore the node structure BEFORE anything rebuilds it: `refreshNodes` keeps the nodes
    // it finds and only prunes the leaves that lost their micro, so a topic's name, its
    // description and its place in the pit survive the reload, and the leaves it holds are
    // not re-formed as young ones.
    if (Array.isArray(data.nodes)) {
      st.nodes = data.nodes.filter((nd) => typeof nd.id === 'string' && Array.isArray(nd.leaves));
    }
    st.totalEvictions = typeof data.totalEvictions === 'number' ? data.totalEvictions : 0;
    st.totalEvictedTokens = typeof data.totalEvictedTokens === 'number' ? data.totalEvictedTokens : 0;
    st.lastEvictionTurn = typeof data.lastEvictionTurn === 'number' ? data.lastEvictionTurn : -1;
    st.lastMeasuredTokens = typeof data.lastMeasuredTokens === 'number' ? data.lastMeasuredTokens : 0;
    st.charTokenRatio = typeof data.charTokenRatio === 'number' && Number.isFinite(data.charTokenRatio) && data.charTokenRatio >= MIN_CHAR_TOKEN_RATIO && data.charTokenRatio <= MAX_CHAR_TOKEN_RATIO
      ? data.charTokenRatio
      : DEFAULT_CHAR_TOKEN_RATIO;
    st.systemOverheadTokens = typeof data.systemOverheadTokens === 'number' && Number.isFinite(data.systemOverheadTokens) && data.systemOverheadTokens >= 0
      ? data.systemOverheadTokens
      : 0;
    st.providerContextTokens = typeof data.providerContextTokens === 'number' && Number.isFinite(data.providerContextTokens) && data.providerContextTokens >= 0
      ? data.providerContextTokens
      : 0;
    st.turns = typeof data.turns === 'number' ? data.turns : 0;
    st.memoryName = typeof data.name === 'string' && data.name ? data.name : undefined;
    st.importedFrom = typeof data.importedFrom === 'string' && data.importedFrom ? data.importedFrom : undefined;
    st.bodies = new Map<string, [number, number]>();
    if (data.bodies && typeof data.bodies === 'object') {
      for (const [k, v] of Object.entries(data.bodies)) {
        if (Array.isArray(v) && v.length === 2 && typeof v[0] === 'number' && typeof v[1] === 'number') {
          st.bodies.set(k, [v[0], v[1]]);
        }
      }
    }
    st.overBudgetSince = typeof data.overBudgetSince === 'number' ? data.overBudgetSince : -1;
    st.gateArmedTurn = typeof data.gateArmedTurn === 'number' ? data.gateArmedTurn : -1;
    st.gateAttempts = typeof data.gateAttempts === 'number' ? data.gateAttempts : 0;
    st.lastGateViolationTurn = typeof data.lastGateViolationTurn === 'number' ? data.lastGateViolationTurn : -1;
    st.gateWithheld = typeof data.gateWithheld === 'string' ? data.gateWithheld : '';
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
/** customType of the demand appended to context (index, leaves, topics, force-save). */
const DEMAND_CUSTOM_TYPE = 'cwl-demand';
/**
 * customType of the announcement that opens an INHERITED memory (`cwl_adopt`). It is its own
 * type and not one of the notices above because it says something about the SESSION, not
 * about the budget or the context: it is injected at the END like the demand, and it stops
 * being injected on its own once this session has leaves of its own — see the caller.
 */
const INHERITED_CUSTOM_TYPE = 'cwl-inherited';

/** True when the message is the budget demand this extension injected. */
function isGateMessage(m: AgentMessage): boolean {
  // SAFETY: read-only field probe (customType); the AgentMessage union does not declare it.
  return (m as unknown as RealMessage).customType === GATE_CUSTOM_TYPE;
}
/** True when the message is a general demand injected by this extension. */
function isDemandMessage(m: AgentMessage): boolean {
  // SAFETY: read-only field probe (customType); the AgentMessage union does not declare it.
  return (m as unknown as RealMessage).customType === DEMAND_CUSTOM_TYPE;
}
/**
 * True when the message is the announcement of an inherited memory. It needs its own filter
 * for the same reason the demand has one: an injected message comes BACK in the list of the
 * next turn, so without this the announcement would stack one copy per turn — the tail would
 * grow a banner every turn, which is exactly what the gate paid for once already.
 */
function isInheritedMessage(m: AgentMessage): boolean {
  // SAFETY: read-only field probe (customType); the AgentMessage union does not declare it.
  return (m as unknown as RealMessage).customType === INHERITED_CUSTOM_TYPE;
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
/**
 * Position of every tool call result in the list, by call id. Episode anchors ARE
 * tool calls, so this map is what turns an anchor into a position — one
 * implementation for every reader, because two copies of it drift.
 */
function toolCallPositions(messages: AgentMessage[]): Map<string, number> {
  const posByToolCallId = new Map<string, number>();
  messages.forEach((m, i) => {
    // SAFETY: toolCallId exists on the real tool-result messages; the public
    // AgentMessage union does not declare it.
    const id = (m as unknown as RealMessage).toolCallId;
    if (typeof id === 'string') posByToolCallId.set(id, i);
  });
  return posByToolCallId;
}

function episodeRanges(
  messages: AgentMessage[],
  episodes: Episode[],
): Map<string, { from: number; to: number; deduced: boolean }> {
  const posByToolCallId = toolCallPositions(messages);
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
/**
 * What the LAST request paid for the PREFIX, read from the assistant message it
 * produced. `usage` is not declared on the message union, so this is a read-only
 * field probe like the ones already used for `role` and `customType`.
 *
 * WHY IT MATTERS: the provider's prompt cache is a PREFIX cache. Every leaf this
 * extension writes lands where the compressed content used to be, and a merge
 * lands at the very FRONT of the conversation: both invalidate everything AFTER
 * them, and the next request pays a cache WRITE where it used to pay a READ.
 * Nothing in this extension could see that cost, so the whole design rested on a
 * bet nobody was measuring. This is the INSTRUMENT, not a fix: it changes no
 * behaviour and no message.
 *
 * The scan goes BACKWARD and stops at the first assistant message that carries a
 * usage, because the newest one is not guaranteed to have it. A stale row cannot
 * be logged twice in a row: the caller dedupes on `ts`.
 */
function lastUsageOf(messages: AgentMessage[]): {
  ts: number; cacheRead: number; cacheWrite: number; input: number; output: number;
} | null {
  const num = (v: unknown): number => (typeof v === 'number' && Number.isFinite(v) ? v : 0);
  for (let i = messages.length - 1; i >= 0; i--) {
    // SAFETY: read-only probes on fields the AgentMessage union does not declare.
    const probe = messages[i] as unknown as { role?: unknown; usage?: unknown; timestamp?: unknown };
    if (probe.role !== 'assistant') continue;
    const u = probe.usage as
      | { cacheRead?: unknown; cacheWrite?: unknown; input?: unknown; output?: unknown }
      | undefined;
    if (!u || typeof u !== 'object') continue;
    return {
      ts: typeof probe.timestamp === 'number' ? probe.timestamp : 0,
      cacheRead: num(u.cacheRead),
      cacheWrite: num(u.cacheWrite),
      input: num(u.input),
      output: num(u.output),
    };
  }
  return null;
}

function compressibleRange(
  messages: AgentMessage[],
  spans: CompressedSpan[],
  protectedTurns: number,
  ratio: number = DEFAULT_CHAR_TOKEN_RATIO,
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
    tokens += estimateMessageTokens(messages[i], ratio);
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
 * by one; what its leaves lose is the context, because the merge summary stands for
 * all of them. The merge summary is the one piece of this design no rule can
 * recompute, so it — and only it — is persisted.
 */
interface OldNode {
  /** `nd-`-style id of the pit itself. */
  id: string;
  /** Ids of the young nodes it absorbed, oldest first. */
  nodes: string[];
  /** The agent's synthesis. Empty means the merge has not happened yet. */
  summary: string;
  /**
   * The syntheses this pit's CURRENT one replaced, newest first, bounded by
   * SUPERSEDED_KEEP. `cwl_old` is the only destructive tool of the set: it
   * OVERWRITES the synthesis instead of extending it, so a rewrite that does not
   * carry the previous text forward would erase the only copy the archive has.
   * They stay readable on their own pages, addressed as `<pit id>.s1`, `.s2`, ...
   */
  superseded?: string[];
  at: number;
}

/** How many leaves a page lists (a node's micros, or a pit's hot leaves). */
const NODE_PAGE_MAX = 30;
// How many syntheses of a pit stay readable after `cwl_old` replaced them. The archive is
// the only place those texts exist, so the newest few stay openable instead of being
// overwritten; the bound keeps both the state file and the pit page from growing at every
// merge. Newest first: `.s1` is the synthesis replaced last.
const SUPERSEDED_KEEP = 3;
// How much of a topic's description the OLD node page shows. A pit topic can be born tiny
// and numerous (no synthesis refacing there), so the page has to stay a page: the name and
// a taste here, and cwl_open on the topic itself for the rest.
const PIT_TOPIC_TASTE = 200;
// A topic born inside the old node needs at least this many leaves. A floor, not a saving rule:
// there is no size guard there (nothing in the index moves), but one or two leaves are a leaf
// list, not a catalogue entry.
const PIT_TOPIC_MIN_LEAVES = 3;

/**
 * The synthesis one merge writes when the pit has none yet: ~900 tokens. MEASURED, not
 * guessed: the merges made by hand replaced 88,806 characters of micros with ~3,400
 * characters of synthesis.
 */
const MERGE_SYNTHESIS_CHARS = 3_600;

/** Id of the TUI widget, and the last line drawn: a widget is UI, not a message. */
const WIDGET_ID = 'cwl-index';
const widgetLines = new Map<string, string>();

/**
 * The shape of the index as one tuple, so `t('indexLine')` can take it positionally.
 * Read-only on purpose: `cwl_status` and the widget both ask here, and a status query
 * must not create nodes.
 */
function indexShape(
  st: CwlState,
  cf: typeof DEFAULT_CONFIG,
): [number, number, number, number, number, number, number, number, number, string, number, string] {
  const inPit = containedNodes(st, st.oldNode?.nodes ?? []);
  const pit = st.nodes.filter((nd) => inPit.has(nd.id));
  const young = st.nodes.filter((nd) => !inPit.has(nd.id));
  // The young nodes SPLIT in three, because the head pays for them in three different ways and
  // mixing them made the line unreadable: a topic injects ONE description (its leaves left the
  // head the moment it was born), the BUFFER is the working area the next round lands in, an
  // ordinary node injects the labels of its leaves. Reported apart, a grouping SHOWS UP as
  // leaves moving from `ordinary` to `topics` instead of the total going up — which is what the
  // operator asked to see. Together the three cover every node that is not in the pit.
  const buffer = st.nodes[st.nodes.length - 1];
  const youngOthers = young.filter((nd) => nd !== buffer);
  const youngTopics = youngOthers.filter((nd) => Boolean(nd.description));
  const youngPlain = youngOthers.filter((nd) => !nd.description);
  // `slice(-0)` is `slice(0)`: see `refreshNodes`. Zero loose leaves means zero, not all.
  const looseIds = new Set(looseSpansOf(st, cf).map((s) => idOfSpan(s)));
  // The head is what the context PAYS for the index: a topic node costs its ONE
  // description, not the labels of its leaves, and a loose leaf costs its own label (its
  // body is still in the context but the label is injected all the same). Counting only the
  // micros of the leaves inside a node undercounted the head by every loose leaf.
  const microOf = new Map(st.spans.map((s) => [idOfSpan(s), s.micro?.length ?? 0]));
  // A node CONTAINED in another injects nothing: its parent's description stands for it, so
  // charging its micros here would overcount the head by everything inside it.
  const childIds = new Set(st.nodes.flatMap((nd) => nd.children ?? []));
  // The PIT is not a node of `st.nodes`: it lives in `st.oldNode`, and its block is ONE
  // synthesis injected once, standing for every leaf it holds. Counting the pit's nodes
  // instead — which is what this did — charged their descriptions and their labels and never
  // charged the synthesis, which is bigger than both together. MEASURED on the live state:
  // 17.599 characters counted against 23.057 actually injected, i.e. 76% of the truth, short
  // by ~1.364 token. The operator's eye caught it ("head does not look realistic, it is much
  // more") before any test did.
  let headChars = pitView(st)?.body.length ?? 0;
  for (const nd of st.nodes) {
    if (childIds.has(nd.id) || inPit.has(nd.id)) continue;
    if (nd.description) headChars += nd.description.length;
    else for (const id of nd.leaves) headChars += microOf.get(id) ?? 0;
  }
  for (const id of looseIds) headChars += microOf.get(id) ?? 0;
  return [
    pit.length,
    pit.reduce((n, nd) => n + nd.leaves.length, 0),
    youngTopics.length,
    youngTopics.reduce((n, nd) => n + nd.leaves.length, 0),
    buffer ? 1 : 0,
    buffer?.leaves.length ?? 0,
    youngPlain.length,
    youngPlain.reduce((n, nd) => n + nd.leaves.length, 0),
    looseIds.size,
    Math.round(headChars / (st.charTokenRatio || DEFAULT_CHAR_TOKEN_RATIO)).toLocaleString(),
    st.totalEvictions,
    st.totalEvictedTokens.toLocaleString(),
  ];
}

/**
 * What `applySpans` needs to know about the pit: its header, and the leaves whose
 * micros must STOP being injected.
 *
 * `null` when there is no merge summary yet: the merge takes effect only once the
 * synthesis EXISTS, because a pit without one would make those leaves disappear from
 * the context with nothing standing for them — the content would still be in the
 * state and in the transcript, and the agent's thread would be cut.
 */
interface PitView {
  id: string;
  /** How many young nodes are behind the synthesis. */
  nodes: number;
  summary: string;
  /**
   * What the head injects for the pit: `summary`, or the descriptions it contains when those
   * are shorter. `cwl_old` writes the synthesis as the agent's text PLUS every absorbed topic's
   * description concatenated verbatim, so the synthesis contains the descriptions and can
   * never be shorter than their sum. MEASURED on the live state: 9.899 characters of synthesis
   * against 4.441 characters of what the pit holds (3.300 of descriptions, 1.141 of labels) —
   * the archive cost 2.2 times its own contents. The operator asked for the shorter of the two,
   * accepting the cache break when the choice flips.
   */
  body: string;
  /** Which of the two `body` is, so the block and the log can say it out loud. */
  mode: 'synthesis' | 'descriptions';
  /** The size of the candidate that lost, so the switch can be logged with its numbers. */
  otherChars: number;
  /** The leaves it stands for. */
  leaves: Set<string>;
  /**
   * When the pit was born. It travels in the view so the injected block can carry a STATIC
   * trace of its own event instead of `Date.now()`: the block sits in the middle of the list,
   * and a value regenerated per turn would make it differ from itself.
   */
  at: number;
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
  // The pit's OWN content, put back together the way `cwl_old` wrote it into the synthesis: a
  // topic's description wrapped exactly as the merge wrapped it, an ordinary node's labels. It
  // keeps every piece of material the pit holds — only the agent's narrative is left out, and
  // that one stays whole on the pit page, reachable with `cwl_open` on the pit id.
  const byId = new Map(st.nodes.map((nd) => [nd.id, nd]));
  const microOf = new Map(st.spans.map((s) => [idOfSpan(s), s.micro ?? '']));
  const childIds = new Set(st.nodes.flatMap((nd) => nd.children ?? []));
  const parts: string[] = [];
  for (const id of pit.nodes) {
    const nd = byId.get(id);
    // A node contained in another injects nothing: its parent's description stands for it.
    if (!nd || childIds.has(nd.id)) continue;
    if (nd.description) parts.push(`[topic "${nd.name ?? nd.id}"] ${nd.description}`);
    else for (const leaf of nd.leaves) {
      const micro = microOf.get(leaf);
      if (micro) parts.push(micro);
    }
  }
  const internal = parts.join('\n\n');
  const useInternal = internal.length > 0 && internal.length < pit.summary.length;
  return {
    id: pit.id,
    nodes: pit.nodes.length,
    summary: pit.summary,
    body: useInternal ? internal : pit.summary,
    mode: useInternal ? 'descriptions' : 'synthesis',
    otherChars: useInternal ? pit.summary.length : internal.length,
    leaves,
    at: pit.at,
  };
}

/**
 * What `applySpans` needs to know about the TOPIC nodes: which topic each leaf belongs to.
 *
 * A topic keeps its description in the head INSTEAD of the labels of its leaves, so the
 * applier has to recognise the leaves by their topic. Only the nodes that carry a
 * description are in the map: an ordinary node is still injected one label at a time, as it
 * always was, and a state without topics behaves exactly as before.
 */
/**
 * What the head needs to know about a topic: the node that injects the ONE block, and the
 * shape of what that block stands for.
 */
interface TopicView {
  node: SpanNode;
  /** Nodes it CONTAINS (transitively), and the leaves inside them. Measured here, so the
   *  shape line in the block can never lie. */
  heldNodes: number;
  heldLeaves: number;
}

function topicView(st: CwlState): Map<string, TopicView> {
  // A leaf inside a CONTAINED node must map to the ROOT of the containment: the root's block
  // is injected at the root's FIRST leaf, which comes before the child's leaves (containment
  // only goes backward in time), so by the time the child's leaves are reached the root is
  // already done and they inject nothing. That is exactly what makes a nested topic free.
  const parent = new Map<string, string>();
  for (const nd of st.nodes) for (const child of nd.children ?? []) parent.set(child, nd.id);
  const byId = new Map(st.nodes.map((nd) => [nd.id, nd]));
  const rootOf = (id: string): string => {
    let cur = id;
    // The hop bound is a guard, not a rule: a cycle would otherwise loop forever, and a
    // cycle is impossible by construction (a node absorbs only nodes AFTER it).
    for (let hop = 0; hop < 64; hop++) {
      const up = parent.get(cur);
      if (!up) return cur;
      cur = up;
    }
    return cur;
  };
  const view = new Map<string, TopicView>();
  for (const nd of st.nodes) {
    if (!nd.description) continue;
    const root = byId.get(rootOf(nd.id));
    if (!root) continue;
    const subtree = containedNodes(st, root.children ?? []);
    const entry: TopicView = {
      node: root,
      heldNodes: subtree.size,
      heldLeaves: st.nodes.filter((c) => subtree.has(c.id)).reduce((n, c) => n + c.leaves.length, 0),
    };
    for (const id of nd.leaves) view.set(id, entry);
  }
  return view;
}

/** How many dropped spans stay recoverable by id. */
const GRAVE_MAX = 200;

/** How many waiting leaf ids the micro demand names in one turn: a few is enough to start. */
const MICRO_DEMAND_IDS = 8;

/**
 * The label a compression may ALREADY carry: trimmed, or absent when empty.
 *
 * Whoever writes a compression summary has just read the messages it summarises, so the
 * ~960-character (≈240-token) micro is nearly free at that moment. Asking for it later means a second pass over
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
  /**
   * A TOPIC node: the name and the description the agent wrote AT BIRTH, and they never
   * change. The description stands for the leaves in the head from the moment the node is
   * born, and it is written to cover the FUTURE use of the topic — which is exactly what
   * makes adding leaves later free: their micros leave the context, the description stays
   * where it is, and no synthesis is written a second time. Rewriting it would move the
   * prefix of the index, so it is IMMUTABLE by construction.
   */
  name?: string;
  description?: string;
  /**
   * The nodes this one CONTAINS: a topic can hold other topics. Containment only goes
   * BACKWARD in time — a node absorbs only nodes that come after it in `st.nodes` — so
   * the position of the parent, and everything before it in the index, never move.
   *
   * The head still pays ONE block per node: a parent injects its description (plus a
   * shape line the code computes) and its children inject NOTHING at all. Depth can be
   * unlimited exactly because depth is never injected, and the children stay readable:
   * `cwl_open("<child id>")` is the same page as any other node.
   */
  children?: string[];
  at: number;
}

/**
 * The nodes a set of roots CONTAINS, transitively: the closure of `children`.
 *
 * It is the set of nodes whose leaves are already spoken for by a description higher up
 * — the pit's synthesis, or a parent topic's — so nothing inside them may be injected or
 * re-formed leaf by leaf. The transitivity is the whole point: a pit node that contains
 * other nodes does not list them in `st.oldNode.nodes`, and without the closure its
 * children would be counted as young, re-formed, and pruned as if the archive did not
 * hold them.
 */
function containedNodes(st: CwlState, roots: Iterable<string>): Set<string> {
  const byId = new Map(st.nodes.map((nd) => [nd.id, nd]));
  const seen = new Set<string>();
  const stack = [...roots];
  while (stack.length) {
    const id = stack.pop() as string;
    if (seen.has(id)) continue;
    seen.add(id);
    for (const child of byId.get(id)?.children ?? []) stack.push(child);
  }
  return seen;
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
/**
 * The leaves whose WHOLE BODY stays in the context: the working set.
 *
 * One function, because three copies of the rule used to exist — `indexShape`,
 * `refreshNodes` and the widget — and a rule duplicated three times is three rules.
 * They all sliced `-looseLeaves` from the tail, which re-opened a leaf as soon as
 * enough newer leaves existed.
 *
 * `looseLeaves` is the CEILING of the window, not its size: the frontier is what
 * decides. When a new leaf pushes the count past the ceiling, the caller below jumps
 * the frontier so that only `LOOSE_AFTER_BATCH` remain — closing three at once, in
 * the same pass that wrote the leaf, which is what makes it free.
 */
const LOOSE_AFTER_BATCH = 2;

/** How many leaves stay open after a batch: never MORE than the ceiling allows. */
function looseAfterBatch(cf: CwlConfig): number {
  return Math.max(0, Math.min(LOOSE_AFTER_BATCH, cf.looseLeaves));
}

function looseSpansOf(st: CwlState, cf: CwlConfig): CompressedSpan[] {
  if (cf.looseLeaves <= 0) return [];
  const from = Math.max(0, Math.min(st.looseFrom, st.spans.length));
  return st.spans.slice(from);
}

/**
 * Moves the frontier after a leaf was written, and returns how many leaves that closed.
 *
 * WHY IT IS FREE: the leaf that was just written landed where the compressed content
 * used to be, so the provider is ALREADY going to rewrite everything after that point.
 * A frontier jump moves no message and inserts nothing: it changes which bodies the
 * NEXT render includes, in the same pass. Closing one leaf per turn would pay a
 * separate invalidation for each; closing them together pays nothing extra.
 */
function advanceLooseFrontier(st: CwlState, cf: CwlConfig): number {
  if (cf.looseLeaves <= 0) {
    st.looseFrom = st.spans.length;
    return 0;
  }
  const open = st.spans.length - st.looseFrom;
  if (open <= cf.looseLeaves) return 0;
  // The batch target is capped by the ceiling. `looseLeaves` is a budget the operator
  // sets, and a batch that left MORE leaves open than that would be this function
  // overruling the config: measured, with `looseLeaves: 1` it kept 2 open and stole a
  // leaf from the buffer node, turning a 19-leaf buffer into 18.
  const keep = looseAfterBatch(cf);
  const closed = open - keep;
  st.looseFrom = st.spans.length - keep;
  return closed;
}

function refreshNodes(
  st: CwlState,
  cf: CwlConfig,
): { formed: number; waiting: number; waitingIds: string[]; due: number } {
  const leaves = [...st.spans].sort((a, b) => a.at - b.at);
  const byId = new Map(leaves.map((l) => [idOfSpan(l), l]));
  // A leaf that left the state (pruned) cannot stay, and neither can a leaf whose
  // micro was REMOVED: un-absorbing puts its body back in the context, so it must
  // come out of the node it no longer belongs to. One condition covers both.
  //
  // The PIT's nodes are the exception, and it is not a nuance: their leaves are spoken
  // for by the merge SUMMARY, not by their micros, so a micro that disappears must not
  // empty a topic. It did: the topic lost its leaves, the emptied node died, its id left
  // `st.oldNode.nodes`, and the pit came back as a summary with no nodes while its
  // leaves were re-formed as young ones — the archive's contents walking back into the
  // head, which is the opposite of what the pit is for. A leaf that is not in the state
  // at all still leaves the node, pit or not: nothing could read it back.
  // Transitive: a node the pit holds may itself CONTAIN nodes, and those are spoken for by
  // the same synthesis. Without the closure a child of a pit node would be treated as young
  // and its leaves would walk back into the head.
  const inPitIds = containedNodes(st, st.oldNode?.nodes ?? []);
  for (const nd of st.nodes) {
    nd.leaves = nd.leaves.filter((id) => {
      const leaf = byId.get(id);
      if (!leaf) return false; // pruned: the node can no longer stand for it
      return inPitIds.has(nd.id) ? true : Boolean(leaf.micro);
    });
  }
  // A node that lost its leaves dies — EXCEPT the LAST one, the buffer attached to the
  // open leaves. It survives even at zero leaves: the agent can move every leaf of the
  // buffer into a topic, and if the buffer died with them the first node would become a
  // topic node, which must never hold that position (it cannot act as a buffer). The
  // buffer keeps its id, so the head of the index does not move either.
  const buffer = st.nodes[st.nodes.length - 1];
  // A node that CONTAINS nodes must not die while its children are alive: its description
  // still stands for them, and killing it would send its whole subtree back into the head —
  // the same failure the pit's exception above prevents, one level down. The children's
  // lists are pruned FIRST, so a parent whose children all died dies in the same pass.
  const aliveIds = new Set(st.nodes.map((nd) => nd.id));
  for (const nd of st.nodes) {
    if (nd.children?.length) nd.children = nd.children.filter((id) => id !== nd.id && aliveIds.has(id));
  }
  st.nodes = st.nodes.filter(
    (nd) => nd.leaves.length > 0 || nd === buffer || (nd.children ?? []).length > 0,
  );
  // The pit's node list is a CLAIM about what the archive holds, and the old-node page prints
  // its length. A node that died — emptied by a re-cataloguing, or pruned — must leave the list
  // too, or the page starts counting nodes that do not exist.
  if (st.oldNode) {
    const alive = new Set(st.nodes.map((nd) => nd.id));
    st.oldNode.nodes = st.oldNode.nodes.filter((id) => alive.has(id));
  }

  // The nodes the pit absorbed are SETTLED: they keep their leaves and nothing new
  // enters them. That is exactly what lets their micros leave the context — the
  // merge summary stands for all of them.
  const inPit = inPitIds;
  const young = st.nodes.filter((nd) => !inPit.has(nd.id));
  const settled = new Set(st.nodes.flatMap((nd) => nd.leaves));
  // `slice(-0)` is `slice(0)`: with `looseLeaves: 0` the whole list would be loose, so no leaf
  // would ever enter a node and the buffer itself would not exist. Zero means zero.
  const loose = new Set(looseSpansOf(st, cf).map((l) => idOfSpan(l)));
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
  // THE SIGNAL IS ANNOUNCED ONLY WHEN THE MERGE WOULD PASS ITS OWN SIZE GUARD. Counting the
  // young nodes is not enough, and the difference is not cosmetic: the nudge kept asking while
  // `cwl_old` refused, and the agent spent a turn writing a synthesis that was thrown away.
  // MEASURED in one session: nine nudges, every one of them refused. A signal that does not
  // survive the check it announces is noise, and noise trains the reader to ignore it.
  const budget = mergeBudget(st, cf);
  if (young.length >= cf.mergeNodesAt && !budget.ok) {
    debugLog(cf, `OLD NODE not-due: ${young.length} young node(s), ${budget.freedChars} chars would leave, need ${budget.needChars} (synthesis ~${budget.synthesisChars}, ratio ${cf.mergeMinRatio}, floor ${cf.mergeMinChars})`);
  }
  return { formed, waiting, waitingIds, due: young.length >= cf.mergeNodesAt && budget.ok ? 1 : 0 };
}

/**
 * What a young -> archive merge would FREE, and what it must free. ONE place, because two
 * callers have to agree: `cwl_old` refuses on these numbers and the index nudge announces the
 * merge on them. They disagreed once, and the nudge became noise.
 *
 * `absorbed` — WHEN the index is due (`young >= mergeNodesAt`) the NEWEST young node stays out:
 * it is the one still filling up, and the pit is where the old material goes. Archiving EARLIER
 * than that — fewer nodes than the index needs to ask — takes everything, the node in progress
 * included. That is what makes `cwl_old` usable to put away material you no longer need WITHOUT
 * losing it: an absorbed node is SETTLED (it keeps its leaves and stops growing), the leaves
 * that come next form a fresh node, and every leaf stays readable through `cwl_open`.
 *
 * `freedChars` — WHAT LEAVES THE CONTEXT, and it is a MEASURE, not a guess: a TOPIC node costs
 * its one description (the labels of its leaves left the head the moment it was born), while an
 * ordinary node costs the labels of its leaves. The pit's own synthesis is replaced too, so it
 * leaves as well. Summing the micros of a topic's leaves would count again what had already
 * gone, and would make the guard approve a merge that frees nothing.
 *
 * `needChars` — the size guard, because a merge COSTS a synthesis. DECIDED with the operator and
 * measured against the real numbers (~1,200 characters per label, ~3,600 for a synthesis): the
 * merge must be worth `mergeMinRatio` times what it writes, and never less than `mergeMinChars`
 * (6,000, about five leaves). MEASURED on the merges made by hand: 60 leaves left 88,806
 * characters of micros and the synthesis that replaced them was ~3,400 (a ratio of 26).
 */
function mergeBudget(
  st: CwlState,
  cf: CwlConfig,
): { absorbed: SpanNode[]; freedChars: number; needChars: number; synthesisChars: number; ok: boolean } {
  const inPit = new Set(st.oldNode?.nodes ?? []);
  const young = st.nodes.filter((nd) => !inPit.has(nd.id));
  const absorbed = young.length >= cf.mergeNodesAt && young.length > 1
    ? young.slice(0, young.length - 1)
    : young;
  const freedChars = (pitView(st)?.body.length ?? 0) + absorbed.reduce((n, nd) => {
    if (nd.description) return n + nd.description.length;
    return n + nd.leaves.reduce((m, id) => m + (st.spans.find((s) => idOfSpan(s) === id)?.micro?.length ?? 0), 0);
  }, 0);
  const synthesisChars = Math.max(pitView(st)?.body.length ?? 0, MERGE_SYNTHESIS_CHARS);
  const needChars = Math.max(Math.round(cf.mergeMinRatio * synthesisChars), cf.mergeMinChars);
  return { absorbed, freedChars, needChars, synthesisChars, ok: freedChars >= needChars };
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
   * A leaf that came from ANOTHER transcript (an adopted/forked memory) and can therefore
   * never resolve here: its anchors belong to a session this one does not have.
   *
   * It is not a DEAD leaf, and the difference is the whole point. A dead leaf lost its
   * anchors IN THIS session: the summary is a cache of an original that is still in the
   * append-only transcript, so dropping it is safe (that is what `st.graves` is for). An
   * ARCHIVED leaf has no original anywhere this session can reach — the summary IS the only
   * copy. Pruning it would not free dead weight, it would DELETE the memory.
   *
   * So it is skipped by `locateSpans` (neither resolved nor dead) and it is never pruned.
   */
  archived?: boolean;
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
 * `_/home/.../session.jsonl.jsonl`, which matches nothing, so `cwl_recall`
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
      // An ARCHIVED leaf belongs to another transcript: it can never resolve here, and it
      // must never be counted as dead either — `dead` is what the caller prunes, and
      // pruning an archived leaf would delete the only copy of its summary. Skipped before
      // the lookup so it is neither. See `CompressedSpan.archived`.
      if (sp.archived) return null;
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
  topics: Map<string, TopicView>,
  demand: string | null,
  /**
   * The announcement of an INHERITED memory, or null when this session's memory is its own.
   * Built by the CALLER, because the name and the origin of the memory live in the state and
   * not in this function's arguments. It is passed as text, not as a flag, so that the wording
   * stays where the rest of the wording is.
   */
  inheritedHead: string | null,
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
  const demandMsg: AgentMessage | null = demand
    ? ({ role: 'custom', customType: DEMAND_CUSTOM_TYPE, content: demand, display: false, timestamp: Date.now() } as unknown as AgentMessage)
    : null;
  /**
   * The announcement of an INHERITED memory, when there is one to make. It is an INJECTION
   * and it rides at the END, beside the demand — the operator's placement, and the reason is
   * measurable: at the TOP it would be part of the PREFIX, so dropping it later (the sentence
   * is true only while this session has no history of its own) would invalidate every cached
   * token after it. In the tail it is free to appear, and free to go. It also reads in the
   * right order: the index it talks about is above, and this is the note that explains it.
   */
  const inheritedMsg: AgentMessage | null = inheritedHead
    ? ({ role: 'custom', customType: INHERITED_CUSTOM_TYPE, content: inheritedHead, display: false, timestamp: Date.now() } as unknown as AgentMessage)
    : null;
  /** Appends the announcement and the demand, in that order: identity first, then the ask. */
  const out = (list: AgentMessage[]): AgentMessage[] =>
    inheritedMsg || demandMsg
      ? [...list, ...(inheritedMsg ? [inheritedMsg] : []), ...(demandMsg ? [demandMsg] : [])]
      : list;

  if (spans.length === 0) return { kept: out(messages), applied: 0, saved: 0, newApplied: 0, newSaved: 0, pairDropped: 0, pairStripped: 0, insideOut: [], dead: [], overlapped: 0 };

  const { resolved, dead, overlapped } = locateSpans(messages, spans);

  // THE INHERITED MEMORY. An adopted memory (see `cwl_adopt`) came from ANOTHER transcript,
  // so its leaves never resolve here — `locateSpans` skips them — and the blocks that normally
  // stand where the compressed content used to be have nowhere to stand. Without this they
  // would be invisible: the memory would live on disk and never reach the model. They are
  // injected AT THE TOP, chronological order (the pit first, then the topics, then the leaves
  // no topic holds): a prefix that changes only when an event changes it, which is the position
  // the cache prefers. Claim 0, because nothing is removed HERE, so the fixed point the normal
  // path iterates is already at its answer.
  const inherited: AgentMessage[] = [];
  /** The leaves that resolved HERE: a block that has one of them has a place to stand. */
  const resolvedIds = new Set(resolved.map((r) => idOfSpan(r.sp)));
  /** The pit already injected at the top: the loop must not inject it a second time. */
  const pitDoneAtTop = pit !== null && ![...pit.leaves].some((id) => resolvedIds.has(id));
  /** The same, one flag per TOPIC node. */
  const topicsDoneAtTop = new Set<string>();
  {
    const covered = new Set<string>(topics.keys());
    if (pit) {
      for (const id of pit.leaves) covered.add(id);
      // A block goes to the top ONLY when it has no resolvable leaf: with no inherited
      // material at all every block has one, so nothing is prepended and the behaviour of a
      // normal session is untouched.
      if (pitDoneAtTop) inherited.push({
        role: 'custom',
        customType: 'cwl-compressed',
        content: t('oldHead')(pit.id, pit.nodes, 0) + (pit.mode === 'descriptions' ? t('oldHeadDescriptions')(pit.id) : '') + pit.body,
        display: false,
        timestamp: pit.at,
      } as unknown as AgentMessage);
    }
    for (const tv of topics.values()) {
      if (topicsDoneAtTop.has(tv.node.id)) continue;
      if (tv.node.leaves.some((id) => resolvedIds.has(id))) continue;
      topicsDoneAtTop.add(tv.node.id);
      const standsFor = tv.node.leaves.length + tv.heldLeaves;
      inherited.push({
        role: 'custom',
        customType: 'cwl-compressed',
        content: t('topicHead')(tv.node.id, tv.node.name ?? tv.node.id, standsFor, 0)
          + (tv.heldNodes ? t('topicHolds')(tv.heldNodes, tv.heldLeaves) : '')
          + String(tv.node.description ?? ''),
        display: false,
        timestamp: tv.node.at,
      } as unknown as AgentMessage);
    }
    for (const sp of spans) {
      if (!sp.archived) continue;
      const id = idOfSpan(sp);
      if (covered.has(id)) continue;
      inherited.push({
        role: 'custom',
        customType: 'cwl-compressed',
        content: t('compressedNotice')(sp.startHash, sp.endHash, 0, id) + (sp.micro ?? sp.summary),
        display: false,
        timestamp: sp.at,
      } as unknown as AgentMessage);
    }
  }

  if (resolved.length === 0) return { kept: out([...inherited, ...messages]), applied: 0, saved: 0, newApplied: 0, newSaved: 0, pairDropped: 0, pairStripped: 0, insideOut: [], dead, overlapped: 0 };

  const replaced = new Set<number>();
  const injected: (AgentMessage | null)[] = [];
  let saved = 0;
  let newSaved = 0;
  let newApplied = 0;
  /** The pit injects ONE block, at its first leaf: the others inject nothing at all. */
  let pitBlockDone = pitDoneAtTop;
  /** The same, one flag per TOPIC node: one description where its first leaf used to be. */
  const topicDone = topicsDoneAtTop;

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
    // A leaf the OLD node holds is no longer injected one by one: the merge summary
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
        content: t('oldHead')(pit.id, pit.nodes, claim) + (pit.mode === 'descriptions' ? t('oldHeadDescriptions')(pit.id) : '') + pit.body,
        display: false,
        // STATIC TRACE: the time of the EVENT that created this block, never `Date.now()`.
        // This block sits in the MIDDLE of the list, so a value that changes per turn makes
        // the block differ from itself and invalidates every cached token after it. The
        // provider payload carries only `role` + `content` (measured on both builders), so a
        // drifting timestamp does not break the cache TODAY — but the trace of an event must
        // say when the event happened, not when the turn was rendered.
        timestamp: pit.at,
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
    // A TOPIC node keeps its IMMUTABLE description in the head instead of the labels of its
    // leaves: the FIRST leaf of the topic carries that one description, and the others bring
    // no block at all — so what they free is the whole `removed`, wrapper included. Without
    // this, grouping nine leaves would still show nine labels and the nesting would save
    // nothing at all. Same shape as the pit above, and the same fixed point for the claim.
    const topic = topics.get(idOfSpan(sp));
    if (topic) {
      if (topicDone.has(topic.node.id)) {
        saved += removed;
        if (!sp.counted) { sp.counted = true; newSaved += removed; newApplied++; }
        injected.push(null);
        continue;
      }
      topicDone.add(topic.node.id);
      // The claim is printed INSIDE the message, so the saving is a fixed point: compute the
      // TEXT, not the message (the AgentMessage union does not expose `content`), exactly as
      // the pit block above does.
      // The SHAPE of what it holds is measured in `topicView`, not written by the agent: a
      // parent topic's block must say how much is inside it, and a number the code measures is
      // always true. It costs no cache of its own — absorbing a LATER node already removes
      // content after this block, so everything from here on is invalidated anyway.
      const standsFor = topic.node.leaves.length + topic.heldLeaves;
      const topicText = (claim: number): string =>
        t('topicHead')(topic.node.id, topic.node.name ?? topic.node.id, standsFor, claim)
        + (topic.heldNodes ? t('topicHolds')(topic.heldNodes, topic.heldLeaves) : '')
        + String(topic.node.description ?? '');
      // SAFETY: the same contract as the per-leaf notice and the pit block below — Pi
      // accepts `custom` in the context hook although the AgentMessage union does not
      // declare it, and the extra key `customType` is how the notice is recognised again.
      const buildTopic = (claim: number): AgentMessage => ({
        role: 'custom',
        customType: 'cwl-compressed',
        content: topicText(claim),
        display: false,
        // STATIC TRACE: the node's own birth, for the same reason as the pit block above.
        timestamp: topic.node.at,
      } as unknown as AgentMessage);
      let gainTopic = estimateTokens(topicText(0));
      for (let i = 0; i < 8; i++) {
        const next = removed - estimateTokens(topicText(gainTopic));
        if (next === gainTopic) break;
        gainTopic = next;
      }
      saved += gainTopic;
      if (!sp.counted) { sp.counted = true; newSaved += gainTopic; newApplied++; }
      injected.push(buildTopic(gainTopic));
      continue;
    }
    // SAFETY: Pi accepts the custom role in the context hook although the
    // AgentMessage union does not declare it; the extra keys are its contract.
    const build = (claim: number): AgentMessage => ({
      role: 'custom',
      customType: 'cwl-compressed',
      content: t('compressedNotice')(sp.startHash, sp.endHash, claim, idOfSpan(sp)) + (sp.micro ?? sp.summary),
      display: false,
      // STATIC TRACE: the leaf's own birth, for the same reason as the pit block above.
      timestamp: sp.at,
    } as unknown as AgentMessage);
    // The saving is what is removed MINUS what takes its place, and what takes its
    // place is the WHOLE injected message: the wrapper the agent keeps reading
    // ("[CWL ...] (~N tokens saved)") costs tokens too. Counting only
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

  const kept: AgentMessage[] = [...inherited];
  // Recorded here, where the positions are BORN: everything this loop pushes
  // because of a span (its summary, and whatever survives inside it) becomes a
  // position the floor report can read later instead of trying to guess it.
  const insideOut: number[] = [];
  messages.forEach((m, i) => {
    // Inject the summary in place of the first compressed message.
    const startsSpan = resolved.find((r) => r.from === i);
    if (startsSpan) {
      // `null` for every pit leaf but the first: the merge summary already stands for it.
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

// ---------------------------------------------------------------------------
// Memories, shared by the tools and by the slash commands
// ---------------------------------------------------------------------------

/**
 * One memory on disk, as `cwl_memories` shows it and `cwl_adopt` forks it.
 *
 * `name` is the raw name a fork resolves by; `display` is the same name with the
 * `(LIVE)` marker the listing shows, which must never be used as an argument.
 */
interface MemoryEntry {
  name: string;
  display: string;
  leaves: number;
  nodes: number;
  pit: string;
  savedAt: number;
  alive: boolean;
  file: string;
  data: Partial<PersistedState>;
}

/** Every memory on disk, parsed once: a memory is found by the NAME inside the file. */
function listMemories(): MemoryEntry[] {
  let files: string[] = [];
  try { files = fs.readdirSync(STATE_DIR).filter((f) => f.endsWith('.json')); } catch { files = []; }
  const found: MemoryEntry[] = [];
  for (const f of files) {
    const raw = readFileOrNull(path.join(STATE_DIR, f));
    if (raw === null) continue;
    let data: Partial<PersistedState>;
    try { data = JSON.parse(raw) as Partial<PersistedState>; } catch { continue; }
    const leaves = Array.isArray(data.spans) ? data.spans.length : 0;
    const nodes = Array.isArray(data.nodes) ? data.nodes.length : 0;
    const name = typeof data.name === 'string' && data.name ? data.name : f.replace(/\.json$/, '');
    const pit = data.oldNode && typeof data.oldNode.id === 'string' ? data.oldNode.id : 'no';
    const alive = typeof data.ownerPid === 'number' && data.ownerPid !== process.pid && isPidAlive(data.ownerPid);
    found.push({
      name,
      display: alive ? `${name} (LIVE)` : name,
      leaves, nodes, pit, alive,
      savedAt: typeof data.savedAt === 'number' ? data.savedAt : 0,
      file: path.join(STATE_DIR, f),
      data,
    });
  }
  found.sort((a, b) => b.savedAt - a.savedAt);
  return found;
}

/**
 * The fork itself, shared by the tool and the slash command: resolve `from`
 * (name or path), refuse what cannot be forked, then copy.
 *
 * It is a plain function and not a tool body so that the two entry points cannot
 * drift: the guards and the copy are one implementation.
 */
function performAdopt(
  cf: CwlConfig,
  st: CwlState,
  key: string,
  from: string,
  as: string | undefined,
): { ok: boolean; text: string; details: Record<string, unknown> } {
  const all = listMemories();
  const mine = st.memoryName ?? defaultMemoryName(key);
  const nameOf = (d: Partial<PersistedState>, file: string): string =>
    typeof d.name === 'string' && d.name ? d.name : path.basename(file).replace(/\.json$/, '');

  let source: MemoryEntry | undefined = all.find((s) => s.name === from);
  if (!source) {
    // A path is the other way in: a state file, or the session file it belongs to.
    if (path.isAbsolute(from) && fs.existsSync(from)) {
      if (from.endsWith('.jsonl')) {
        const viaSession = loadPersistedState(from);
        const hit = viaSession ? all.find((s) => s.name === (viaSession.memoryName ?? '')) : undefined;
        source = hit;
      } else {
        const raw = readFileOrNull(from);
        if (raw !== null) {
          try {
            const data = JSON.parse(raw) as Partial<PersistedState>;
            const named = nameOf(data, from);
            source = { name: named, display: named, leaves: 0, nodes: 0, pit: 'no', savedAt: 0, alive: false, file: from, data };
          } catch { source = undefined; }
        }
      }
    }
  }
  if (!source) {
    const names = all.map((s) => s.name).join(', ') || '- none -';
    return { ok: false, text: t('adoptNotFound')(from, names), details: { ok: false, error: 'memory-not-found', from } };
  }
  const sourceName = nameOf(source.data, source.file);
  if (sourceName === mine) {
    return { ok: false, text: t('adoptBusy')(mine), details: { ok: false, error: 'already-this-memory', name: mine } };
  }
  // A memory whose owner is still alive keeps writing its own copy: forking it would give
  // two writers to one past, and the two branches would erase each other.
  const aliveOwner = typeof source.data.ownerPid === 'number' && source.data.ownerPid !== process.pid
    && isPidAlive(source.data.ownerPid) && source.file !== statePath(key);
  if (aliveOwner) {
    return { ok: false, text: t('adoptOtherSession'), details: { ok: false, error: 'memory-in-use', from: sourceName, pid: source.data.ownerPid } };
  }
  const leaves = Array.isArray(source.data.spans) ? source.data.spans : [];
  if (leaves.length === 0) {
    return { ok: false, text: t('adoptEmpty')(sourceName), details: { ok: false, error: 'memory-empty', from: sourceName } };
  }

  // THE FORK. The leaves are copied and marked ARCHIVED: they carry anchors of another
  // transcript, so `locateSpans` skips them (never resolved, never dead) and the pruning
  // can never touch them — the summary IS the copy that exists from here on. The usage
  // counters are dropped on purpose: how often a leaf was opened is the history of a
  // session that is not this one.
  st.spans = leaves.map((sp) => ({ ...sp, archived: true, counted: true, opens: undefined, lastOpen: undefined }));
  st.nodes = Array.isArray(source.data.nodes) ? source.data.nodes : [];
  st.oldNode = source.data.oldNode ?? null;
  st.looseFrom = Math.max(0, st.spans.length - DEFAULT_CONFIG.looseLeaves);
  // The graveyard points at a transcript this session does not have, and the episodes are
  // positions in the old message list: neither travels.
  st.graves = [];
  st.graph = new EpisodeGraph();
  st.totalEvictions = 0;
  st.totalEvictedTokens = 0;
  st.memoryName = as && as.trim() ? as.trim() : `${sourceName}--fork`;
  st.importedFrom = sourceName;
  // The fork COPIES the bodies, not a pointer to them: the operator chose copied over
  // shared, so the branch survives the deletion of the source. The copy is byte-identical,
  // so the offsets the source indexed remain valid here unchanged.
  if (source.file) {
    const srcBodies = source.file.replace(/\.json$/, '.bodies.jsonl');
    if (fs.existsSync(srcBodies)) {
      try { fs.copyFileSync(srcBodies, bodiesPath(key)); } catch { /* the fork keeps what the index carries */ }
    }
  }
  st.bodies = new Map<string, [number, number]>();
  const srcMap = source.data.bodies;
  if (srcMap && typeof srcMap === 'object') {
    for (const [k, v] of Object.entries(srcMap)) {
      if (Array.isArray(v) && v.length === 2 && typeof v[0] === 'number' && typeof v[1] === 'number') {
        st.bodies.set(k, [v[0], v[1]]);
      }
    }
  }
  // A source older than the bodies store carries its pit summaries in the spans still:
  // this moves them into the FORK's file, exactly as a save would.
  moveLeafBodiesToDisk(key, st);
  saveState(key, st);
  debugLog(cf, `ADOPT: forked "${sourceName}" into "${st.memoryName}" — ${st.spans.length} leaf/leaves ARCHIVED, ${st.nodes.length} node(s), pit ${st.oldNode ? st.oldNode.id : 'none'}; episodes did not travel`);
  return {
    ok: true,
    text: t('adoptDone')(st.memoryName, sourceName, st.spans.length, st.nodes.length),
    details: { ok: true, name: st.memoryName, from: sourceName, leaves: st.spans.length, nodes: st.nodes.length, pit: st.oldNode ? st.oldNode.id : null },
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

/**
 * One node of the index, as every CONSULTATION tool sees it.
 *
 * `cwl_map`, `cwl_node`, `cwl_pending` and `cwl_find` all read the index through
 * `indexNodeViews`, so they can never disagree about what exists or about an id.
 */
interface IndexNodeView {
  id: string;
  name: string | null;
  kind: 'pit' | 'topic' | 'buffer' | 'legacy';
  leaves: string[];
  children: string[];
  /** What this node costs in the head: its description, or the micros of its leaves. */
  chars: number;
  description: string | null;
  /** The last node of the frontier: the working set, the only one that can become a topic. */
  isBuffer: boolean;
}

/** Every leaf the index knows, with the text a search can look at. */
interface LeafView {
  id: string;
  micro: string | null;
  chars: number;
  /** The node that owns it, or null when it is loose. */
  container: string | null;
}

/** A single-line rendering of a label: the consultation tools must stay readable. */
function firstLine(text: string | null | undefined, max = 90): string {
  if (!text) return '';
  const line = text.replace(/\s+/g, ' ').trim();
  return line.length > max ? `${line.slice(0, max - 1)}\u2026` : line;
}

function leafViews(st: CwlState): Map<string, LeafView> {
  const views = new Map<string, LeafView>();
  for (const sp of st.spans) {
    const id = idOfSpan(sp);
    views.set(id, { id, micro: sp.micro ?? null, chars: sp.micro?.length ?? 0, container: null });
  }
  for (const nd of st.nodes) {
    for (const id of nd.leaves) {
      const v = views.get(id);
      if (v) v.container = nd.id;
    }
  }
  return views;
}

/**
 * Every node of the index in the order the head injects them: the pit first, then
 * `st.nodes` chronologically. The pit reports the chars it REALLY costs (the shorter of
 * its synthesis and the descriptions it holds, see `pitView`), and a node that is HELD
 * by another costs nothing: its parent's block stands for it.
 */
function indexNodeViews(st: CwlState): IndexNodeView[] {
  const leaves = leafViews(st);
  const childIds = new Set(st.nodes.flatMap((nd) => nd.children ?? []));
  const bufferId = st.nodes.length > 0 ? st.nodes[st.nodes.length - 1].id : null;
  const views: IndexNodeView[] = [];
  const pit = st.oldNode;
  if (pit && pit.summary) {
    views.push({
      id: pit.id,
      name: null,
      kind: 'pit',
      leaves: [],
      children: [...pit.nodes],
      chars: pitView(st)?.body.length ?? pit.summary.length,
      description: null,
      isBuffer: false,
    });
  }
  for (const nd of st.nodes) {
    const held = childIds.has(nd.id);
    views.push({
      id: nd.id,
      name: nd.name ?? null,
      kind: nd.description ? 'topic' : nd.id === bufferId ? 'buffer' : 'legacy',
      leaves: [...nd.leaves],
      children: [...(nd.children ?? [])],
      chars: held
        ? 0
        : nd.description
          ? nd.description.length
          : nd.leaves.reduce((n, id) => n + (leaves.get(id)?.chars ?? 0), 0),
      description: nd.description ?? null,
      isBuffer: nd.id === bufferId,
    });
  }
  return views;
}

  pi.registerTool({
    name: 'cwl_status',
    label: 'CWL Status',
    description: t('params').statusDesc,
    promptSnippet: t('snippets').status,
    parameters: Type.Object({
      leafIds: Type.Optional(Type.Boolean({ description: t('statusLeafIdsHint') })),
    }),
    async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
      const key = sessionKey(ctx);
      const st = getState(key);
      const cf = getConfig(key);
      const g = st.graph;
      const active = g.active();
      const closed = g.recoverable();
      const stripped = closed.filter(e => e.level !== 'none');

      const extra = st.systemOverheadTokens > 0
        ? (LANG === 'it'
            ? `overhead fisso: ~${st.systemOverheadTokens.toLocaleString()}, ratio: ${st.charTokenRatio.toFixed(1)} c/t`
            : `fixed overhead: ~${st.systemOverheadTokens.toLocaleString()}, ratio: ${st.charTokenRatio.toFixed(1)} c/t`)
        : undefined;

      const lines = [
        t('statusHeader')(cf.tokenBudget.toLocaleString(), (cf.thresholdRatio * 100).toFixed(0)),
        t('statusMeasured')(st.lastMeasuredTokens.toLocaleString(), extra),
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
      // The shape of the index: which memories exist, how big they are, and what the head
      // costs. The TUI widget shows the same line — one measurement, two windows.
      lines.push(t('indexLine')(...indexShape(st, cf)));
      const topicNames = st.nodes
        .filter((nd) => nd.description)
        .map((nd) => (nd.name ? `${nd.name}(${nd.id})` : nd.id));
      if (topicNames.length > 0) lines.push(t('statusTopics')(topicNames.length, topicNames.join(', ')));
      // The pit and the buffer are the two ids an agent needs most often and could not see
      // anywhere. They are NOT topics, so they get their OWN line: appending them to the
      // topics made `Topics (1)` say something untrue on a fresh index, and four tests
      // caught it. Everything else is one `cwl_map` away.
      const nodeIds: string[] = [];
      if (st.oldNode?.summary) nodeIds.push(`${st.oldNode.id} (pit)`);
      const bufferNode = st.nodes.length > 0 ? st.nodes[st.nodes.length - 1] : null;
      if (bufferNode) nodeIds.push(`${bufferNode.id} (buffer)`);
      if (nodeIds.length > 0) lines.push(t('statusIds')(nodeIds.join(', ')));
      // #1 — THE LEAF IDS, only when asked. `cwl_status` is called often and the ids are the
      // part of it that grows with the session: the buffer alone can hold fourteen leaves,
      // and every one of them is a line the caller may never read. Off by default, so every
      // existing caller pays exactly what it paid before.
      if (params.leafIds === true) {
        const young = st.nodes.filter((nd) => !nd.description);
        const shown = young.map((nd) => `${nd.id}: ${nd.leaves.join(' ')}`);
        if (shown.length > 0) lines.push(t('statusLeafIds')(shown.join(' | ')));
        // The loose leaves are the ones with no owner at all: they are the newest material
        // and the easiest to lose track of, so they get their own line.
        const owned = new Set(young.flatMap((nd) => nd.leaves));
        const loose = [...st.spans]
          .sort((a, b) => a.at - b.at)
          .map((s) => idOfSpan(s))
          .filter((id) => !owned.has(id) && !st.nodes.some((nd) => nd.leaves.includes(id)));
        if (loose.length > 0) lines.push(t('statusLooseIds')(loose.join(' ')));
      }
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
      // A hand-picked range must respect the coverage the OFFERED path already enforces: an
      // interval that intersects a live leaf would describe those messages a second time, and
      // the two descriptions would drift apart. `cwl_compress_range` cannot do this; this tool
      // could, and did: it validated the two hashes against `knownHashes` and pushed.
      const cov = liveCoverage(ctx, st.spans);
      const request = `${params.startHash}..${params.endHash}`;
      if (cov === null) {
        debugLog(cf, `COMPRESS coverage: not verifiable on this context — ${request} let through, NOT checked`);
      } else {
        const covered = cov.covers(params.startHash, params.endHash);
        if (covered === 'unknown') {
          debugLog(
            cf,
            `COMPRESS coverage: endpoints of ${request} are not placeable on the ${cov.messages} message(s) read — let through, NOT checked`,
          );
        } else if (covered) {
          debugLog(
            cf,
            `COMPRESS refused ${request}: inside a live leaf (${cov.spans} span(s) resolved, ${cov.messages} message(s) read)`,
          );
          return {
            content: [{ type: 'text', text: t('tools').compressCovered(params.startHash, params.endHash) }],
            details: { ok: false, error: 'covered-range', spansResolved: cov.spans, messagesRead: cov.messages },
          };
        }
      }
      st.spans.push({
        startHash: params.startHash,
        endHash: params.endHash,
        id: spanId(params.startHash, params.endHash),
        summary: params.summary,
        micro: microOrUndefined(params.micro),
        at: Date.now(),
      });
      st.forceAllNext = false;
      st.forceAllNote = undefined;
      // Durable immediately: the agent asked for this compression, so it must
      // survive a restart even if the process is killed before the next turn ends.
      saveState(key, st);
      debugLog(cf, `COMPRESS applied ${params.startHash}..${params.endHash}`);
      return {
        content: [{ type: 'text', text: t('compressApplied')(params.startHash, params.endHash) + overCeiling(`${params.startHash}..${params.endHash}`, microOrUndefined(params.micro), cf) }],
        details: { ok: true, spans: st.spans.length },
      };
    },
  });

  /**
   * The coverage the OFFERED path already enforces, brought to the hand-picked one.
   *
   * `compressibleRange` skips every index inside a resolved span, so `cwl_compress_range` can
   * never describe the same messages twice. `cwl_compress` had no such check AND no way to get
   * one: it never sees the message list, and its own comment says so. But Pi's session manager
   * is what the hook's context is built from, and `buildContextEntries` belongs to the
   * read-only API an extension receives, so the tool can rebuild the same ORDER of addresses.
   * An interval comparison needs nothing more than the order: the absolute indices may differ
   * from the hook's list, the relative position of two endpoints cannot.
   *
   * `null` means the list is not readable at all. The caller must then DECLARE that the range
   * was not checked — quietly accepting it as if it had been is the silence this project keeps
   * paying for.
   */
  function liveCoverage(
    ctx: ExtensionContext | null | undefined,
    spans: CompressedSpan[],
  ): { messages: number; spans: number; covers: (a: string, b: string) => boolean | 'unknown' } | null {
    try {
      const sm = ctx?.sessionManager as
        | { buildContextEntries?: () => Array<{ type?: string; message?: AgentMessage }> }
        | undefined;
      if (typeof sm?.buildContextEntries !== 'function') return null;
      // Only `type: 'message'` entries carry a message: `compaction` and `branchSummary`
      // entries carry a `summary` instead, and Pi skips them the same way when it builds the
      // context. Dropping them shifts every index by the same amount, which is precisely what
      // an interval comparison tolerates.
      const messages = sm
        .buildContextEntries()
        .filter((e) => !!e && e.type === 'message' && !!e.message)
        .map((e) => e.message as AgentMessage);
      if (messages.length === 0) return null;
      const { exact, legacy } = addressMaps(messages);
      const ranges = locateSpans(messages, spans).resolved.map((r) => [r.from, r.to] as [number, number]);
      const place = (h: string): number | undefined => exact.get(h) ?? legacy.get(h);
      return {
        messages: messages.length,
        spans: ranges.length,
        covers: (a, b) => {
          const i = place(a);
          const j = place(b);
          if (i === undefined || j === undefined) return 'unknown';
          const low = Math.min(i, j);
          const hi = Math.max(i, j);
          return ranges.some(([from, to]) => low <= to && from <= hi);
        },
      };
    } catch {
      return null;
    }
  }

  /**
   * The ceiling on one label, and the overrun said out loud.
   *
   * MEASURED on 42 real labels: ~722t each (~1.900 characters) against the ~300t this design
   * assumed, so the top of the index cost 30k instead of ~12k. The prompt asked for "about 200
   * words" and bound nothing: 42 labels came out at ~300 words each. A ceiling in CHARACTERS is
   * what a model can count while it writes, hence the number below — and the overrun is
   * DECLARED, in the tool result and in the log, rather than refused: refusing would block the
   * work, and a label that is too long is still a label. What is not acceptable is silence.
   */
  const MAX_MICRO_CHARS = 1400;
  function overCeiling(where: string, micro: string | undefined, cf: ReturnType<typeof getConfig>): string {
    const n = (micro ?? '').trim().length;
    if (!micro || n <= MAX_MICRO_CHARS) return '';
    debugLog(
      cf,
      `MICRO over ceiling: ${where} is ${n} characters (~${Math.round(n / 4)}t) against a ceiling of ${MAX_MICRO_CHARS} (~350t)`,
    );
    return t('tools').microOver(where, n, MAX_MICRO_CHARS);
  }

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
      // The leaf id belongs in the ANSWER too: the RICHIAMO marker injected into the context
      // carries it, but a turn that compresses twice has to be able to group the leaves it
      // has just born, and it cannot read a marker it has not seen yet.
      const leafId = spanId(startHash, endHash);
      st.spans.push({ startHash, endHash, id: leafId, summary: params.summary, micro: microOrUndefined(params.micro), at: Date.now() });
      // Spend the address: the next hook recomputes it on the smaller list, so a
      // second call cannot compress the same range twice.
      st.rangeStartHash = null;
      st.rangeEndHash = null;
      st.rangeTokens = 0;
      st.forceAllNext = false;
      st.forceAllNote = undefined;
      // The event the NEXT request follows: this leaf landed where the compressed
      // content used to be, so everything after it is rewritten and the provider
      // will pay a cache WRITE instead of a READ. Recorded BEFORE the save, or a
      // restart would lose the cause while keeping the effect.
      st.lastEvent = 'new-leaf';
      st.turnsSinceCompress = 0;
      // The leaf just written is the SECOND event of this pass: the frontier move rides
      // on the invalidation it already causes (see `advanceLooseFrontier`).
      const closed = advanceLooseFrontier(st, cf);
      saveState(key, st);
      if (closed > 0) debugLog(cf, `LOOSE frontier: ${closed} leaf/leaves closed in the same pass, ${st.spans.length - st.looseFrom} still open`);
      debugLog(cf, `COMPRESS-RANGE applied ${startHash}..${endHash} (~${tokens}t)`);
      return {
        content: [{ type: 'text', text: t('compressRangeApplied')(startHash, endHash, tokens, leafId) + overCeiling(`${startHash}..${endHash}`, microOrUndefined(params.micro), cf) }],
        details: { ok: true, spans: st.spans.length, tokens, id: leafId },
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
    name: 'cwl_memories',
    label: 'CWL Memories',
    description: t('tools').memoriesDesc,
    promptSnippet: t('snippets').memories,
    parameters: Type.Object({}),
    async execute(_toolCallId, _params, _signal, _onUpdate, _ctx) {
      // Reading every state file is the ONE heavy read of the set, and it is deliberate: a
      // memory is found by NAME, and the name lives inside the file. It is an explicit act,
      // never a per-turn cost.
      const found = listMemories();
      if (found.length === 0) {
        return { content: [{ type: 'text', text: t('memoriesEmpty')() }], details: { ok: true, memories: 0, names: [] as string[] } };
      }
      const rows = found.map((m) => t('memoriesRow')(m.display, m.leaves, m.nodes, m.pit)).join('\n');
      return {
        content: [{ type: 'text', text: t('memoriesList')(rows) }],
        details: { ok: true, memories: found.length, names: found.map((m) => m.display) },
      };
    },
  });

  pi.registerTool({
    name: 'cwl_adopt',
    label: 'CWL Adopt',
    description: t('tools').adoptDesc,
    promptSnippet: t('snippets').adopt,
    parameters: Type.Object({
      from: Type.String({ description: t('tools').adoptFromDesc }),
      as: Type.Optional(Type.String({ description: t('tools').adoptAsDesc })),
    }),
    async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
      const key = sessionKey(ctx);
      const st = getState(key);
      const cf = getConfig(key);
      const from = String(params.from ?? '').trim();
      const as = params.as === undefined ? undefined : String(params.as);
      const res = performAdopt(cf, st, key, from, as);
      return { content: [{ type: 'text', text: res.text }], details: res.details };
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
      // The OLD node first: its page is the merge summary plus what it holds — one line
      // per young node inside, with the SHAPE (how many leaves, and the first and last
      // leaf id), so a reader can choose what to open without opening it.
      if (st.oldNode && st.oldNode.id === wanted) {
        const inPit = new Set(st.oldNode.nodes);
        const righe = st.nodes
          .filter((nd) => inPit.has(nd.id))
          .map((nd) => {
            const shape = `${nd.leaves.length} leaf/leaves (${nd.leaves[0]} .. ${nd.leaves[nd.leaves.length - 1]})`;
            // A TOPIC inside the pit is a catalogue entry: its name and a taste of its
            // description, so the page can be read WITHOUT opening every node. Its id stays
            // because that is what cwl_group takes to add more leaves to it.
            if (!nd.description) return `- ${nd.id}: ${shape}`;
            const taste = nd.description.length > PIT_TOPIC_TASTE ? `${nd.description.slice(0, PIT_TOPIC_TASTE)}...` : nd.description;
            return t('oldTopicLine')(nd.id, nd.name ?? '', shape, taste);
          })
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
        // The syntheses this one replaced stay OPENABLE, a page each: `<pit id>.s1` is the
        // one replaced last. Listed here with their size, so the pit page stays a page while
        // nothing the archive held is lost. This is the "superseded by" the operator asked
        // for: the CURRENT synthesis is the one in the context, the others are one call away.
        const supersededLines = (st.oldNode.superseded ?? [])
          .map((old, i) => t('oldSupersededLine')(`${st.oldNode?.id ?? ''}.s${i + 1}`, old.length))
          .join('\n');
        const body = `${st.oldNode.summary}\n\n${righe}\n\n${t('oldHot')(hot.length, pitLeaves.length)}\n${hotLines}`
          + (supersededLines ? `\n\n${t('oldSupersededHead')(st.oldNode.superseded?.length ?? 0)}\n${supersededLines}` : '');
        const tokens = estimateTokens(body);
        debugLog(cf, `OPEN ${st.oldNode.id}: merge summary + ${st.oldNode.nodes.length} node(s), ${pitLeaves.length} leaf/leaves inside, most opened ${hot[0]?.opens ?? 0}x (${hot.filter((s) => (s.opens ?? 0) > 0).length} ever opened) — ${tokens}t`);
        return {
          content: [{ type: 'text', text: t('oldPage')(st.oldNode.id, st.oldNode.nodes.length, tokens, body) }],
          details: { ok: true, id: st.oldNode.id, kind: 'old', nodes: st.oldNode.nodes.length, leaves: pitLeaves.length, opened: hot.filter((s) => (s.opens ?? 0) > 0).length, tokens },
        };
      }
      // A superseded synthesis has its own page: `<pit id>.s1` is the most recently replaced
      // one. Read on demand, which is why the pit page can afford to stay small.
      const superseded = /^(.*)\.s(\d+)$/.exec(wanted);
      if (superseded && st.oldNode && st.oldNode.id === superseded[1]) {
        const old = st.oldNode.superseded?.[Number(superseded[2]) - 1];
        if (old) {
          debugLog(cf, `OPEN ${wanted}: superseded synthesis, ${old.length} chars`);
          return {
            content: [{ type: 'text', text: t('oldSupersededPage')(wanted, old.length, old) }],
            details: { ok: true, id: wanted, kind: 'superseded', chars: old.length, tokens: estimateTokens(old) },
          };
        }
      }
      // A NODE next: its page is the micros of its leaves, each with the id that
      // opens it. Same rule as a leaf: the size is declared before it is handed over.
      const nd = st.nodes.find((n) => n.id === wanted);
      if (nd) {
        // A TOPIC node carries what the agent wrote at its BIRTH: the name and the
        // description that stand for these leaves in the index. Both are immutable (a
        // rewritten description would move the prefix of the index), so this page is the
        // only place to read them back — and the only proof that the leaves below are
        // held by a topic rather than by an ordinary node.
        const head = nd.description ? `TOPIC "${nd.name ?? nd.id}": ${nd.description}\n` : '';
        // A node that CONTAINS nodes prints them the way the pit's page does: id, shape, and
        // a taste of the description, so the page stays a page. Opening one is the same call.
        const childLines = (nd.children ?? [])
          .map((childId) => st.nodes.find((n) => n.id === childId))
          .filter((child): child is SpanNode => Boolean(child))
          .map((child) => {
            const shape = `${child.leaves.length} leaf/leaves (${child.leaves[0] ?? '-'} .. ${child.leaves[child.leaves.length - 1] ?? '-'})`;
            if (!child.description) return `- ${child.id}: ${shape}`;
            const taste = child.description.length > PIT_TOPIC_TASTE ? `${child.description.slice(0, PIT_TOPIC_TASTE)}...` : child.description;
            return t('oldTopicLine')(child.id, child.name ?? '', shape, taste);
          })
          .join('\n');
        const lines = head + (childLines ? `${childLines}\n` : '') + nd.leaves
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
        // Not in the state: it was DROPPED — its endpoints left the context for good. The
        // operator's rule says micro + summary must ALWAYS remain and the original transcript
        // may go away: the summary (and the micro, when it had one) were appended to the body
        // store when the span was pruned, and that store is asked BEFORE declaring anything
        // lost. The graveyard still finds the ORIGINAL text when the transcript has it.
        const droppedAt = (id: string, when: string): { content: { type: 'text'; text: string }[]; details: Record<string, unknown> } | null => {
          const stored = readBody(key, st, id);
          if (!stored) return null;
          const tokens = estimateTokens(stored.text);
          const microLine = stored.micro ? `\n\n[MICRO]\n${stored.micro}` : '';
          return {
            content: [{ type: 'text', text: t('openFound')(id, tokens, when) + stored.text + microLine }],
            details: { ok: true, id, kind: 'body', dropped: true, tokens, chars: stored.text.length, hasMicro: Boolean(stored.micro) },
          };
        };
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
          const fromBody = droppedAt(wanted, new Date(grave.at).toISOString().slice(0, 16).replace('T', ' '));
          if (fromBody) return fromBody;
          // Declared, never silent: neither the transcript nor the body store has it.
          return {
            content: [{ type: 'text', text: t('openOriginalLost')(wanted) }],
            details: { ok: false, error: 'original-lost', id: wanted, kind: 'original' },
          };
        }
        // No grave either (the anchors left before the span ever resolved): the body store
        // is the only place left, and the rule says it must be asked.
        const fromBody = droppedAt(wanted, '');
        if (fromBody) return fromBody;
        return {
          content: [{ type: 'text', text: t('openMissing')(wanted) }],
          details: { ok: false, error: 'unknown-id', id: wanted, spans: st.spans.length },
        };
      }
      const id = idOfSpan(sp);
      // A leaf absorbed into the PIT may have its body on disk: the summary leaves the RAM
      // when it enters the archive, and it is read back on demand — one read at the stored
      // offset, never the whole file.
      const stored = sp.summary ? null : readBody(key, st, id);
      const body = sp.summary || (stored ? stored.text : '');
      if (!body) {
        return {
          content: [{ type: 'text', text: t('openBodyLost')(id) }],
          details: { ok: false, error: 'body-lost', id },
        };
      }
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
        content: [{ type: 'text', text: t('microSet')(id, micro.length, sp.summary.length, shorter) + overCeiling(id, micro, cf) }],
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
      dryRun: Type.Optional(Type.Boolean({ description: t('tools').oldDryRun })),
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
      if (young.length === 0) {
        return {
          content: [{ type: 'text', text: t('oldNotDue')(young.length, cf.mergeNodesAt) }],
          details: {
            ok: false, error: 'not-due', code: 'no-young-nodes',
            young: young.length, mergeNodesAt: cf.mergeNodesAt,
            constraint: { young: young.length, mergeNodesAt: cf.mergeNodesAt },
            allowed: ['group-leaves-into-a-topic', 'wait-for-more-nodes'],
          },
        };
      }
      // WHEN the index is due (young >= mergeNodesAt) the NEWEST young node stays out: it
      // is the one still filling up, and the pit is where the old material goes. When the
      // agent archives EARLIER than that — fewer nodes than the index needs to ask — it
      // takes everything, the node in progress included. That is what makes `cwl_old`
      // usable to put away material you no longer need WITHOUT losing it: an absorbed node
      // is SETTLED (it keeps its leaves and stops growing), the leaves that come next form
      // a fresh node, and every leaf stays readable through `cwl_open`.
      const budget = mergeBudget(st, cf);
      const absorbed = budget.absorbed;
      const ids = absorbed.flatMap((nd) => nd.leaves);
      const microChars = ids.reduce(
        (n, id) => n + (st.spans.find((s) => idOfSpan(s) === id)?.micro?.length ?? 0),
        0,
      );
      // The two guards on the SIZE, because a merge COSTS a synthesis. DECIDED with the
      // operator and measured against the real numbers (~1,200 characters per label, ~3,600
      // for a synthesis): the young -> archive merge must be worth at least 3x what it
      // writes, and never less than `mergeMinChars` (6,000, about five leaves). MEASURED on
      // the merges made by hand: 60 leaves left 88,806 characters of micros and the synthesis
      // that replaced them was ~3,400 (a ratio of 26). A two-leaf node leaves ~5,800 and buys
      // a synthesis with it: that one has to be refused, and said out loud.
      // WHAT LEAVES THE CONTEXT, and it is a MEASURE, not a guess: a TOPIC node costs its one
      // description — the labels of its leaves left the head the moment it was born — while an
      // ordinary node costs the labels of its leaves. The pit's own synthesis is replaced too,
      // so it leaves as well. Summing the micros of a topic's leaves would count again what had
      // already gone, and would make the guard approve a merge that frees nothing.
      const { freedChars, needChars, synthesisChars } = budget;
      // #2 — THE DRY RUN, placed HERE on purpose: after the budget is measured and before
      // anything is written. A dry run that could only report a merge it would approve would
      // be half useful — the case worth predicting is the one that gets REFUSED, because that
      // is the one that costs a second call today. Nothing above this line mutates the state
      // (`mergeBudget` only measures), so there is no snapshot to put back.
      if (params.dryRun === true) {
        return {
          content: [{
            type: 'text',
            text: t('oldDryRun')(
              freedChars >= needChars,
              st.oldNode?.id ?? 'a new pit',
              absorbed.length, ids.length, freedChars, needChars,
            ),
          }],
          details: {
            ok: true, dryRun: true, wouldProceed: freedChars >= needChars,
            destination: st.oldNode?.id ?? null, nodes: absorbed.map((nd) => nd.id),
            leaves: ids.length, microChars, freedChars, needChars, synthesisChars,
          },
        };
      }
      if (freedChars < needChars) {
        debugLog(cf, `OLD not-due: ${ids.length} leaf/leaves would leave ${freedChars} chars, need ${needChars} (synthesis ~${synthesisChars}, ratio ${cf.mergeMinRatio}, floor ${cf.mergeMinChars})`);
        return {
          content: [{ type: 'text', text: t('oldTooSmall')(ids.length, freedChars, needChars) }],
          details: {
            ok: false, error: 'too-small', code: 'synthesis-too-expensive',
            leaves: ids.length, microChars, freedChars, needChars, synthesisChars,
            constraint: { freedChars, needChars, synthesisChars, ratio: cf.mergeMinRatio, floor: cf.mergeMinChars },
            allowed: ['write-a-shorter-synthesis', 'merge-more-nodes', 'lower-mergeMinRatio', 'lower-mergeMinChars'],
          },
        };
      }
      const pit: OldNode = st.oldNode ?? {
        id: `old-${hashText(absorbed[0].id).slice(0, 8)}`,
        nodes: [],
        summary: '',
        at: Date.now(),
      };
      pit.nodes.push(...absorbed.map((nd) => nd.id));
      // The synthesis is the agent's text ALONE. It used to carry the absorbed topics'
      // descriptions concatenated verbatim, on the argument that re-describing immutable text
      // through an LLM pays twice and can drift. That argument was right about the risk and wrong
      // about the cost: MEASURED, it made the pit's block 9.899 characters against 4.441 of
      // content — the archive cost 2.2 times what it held — and no synthesis could ever be
      // shorter than the descriptions it contained, so `cwl_old` could never actually compress.
      // The rule the operator gave: a merge SUMMARISES the descriptions, it does not append
      // pieces. Nothing is lost: a description that leaves the head is whole on its topic page,
      // and the pit's own content comes back into the head as it is — chronological, no LLM —
      // whenever it is shorter than the synthesis (see `pitView`).
      // Keep what this merge REPLACES: `cwl_old` overwrites the synthesis, so for a rewrite
      // that does not carry the previous text forward the pit page would have been the only
      // place it ever lived. Newest first, bounded (see SUPERSEDED_KEEP).
      if (pit.summary) pit.superseded = [pit.summary, ...(pit.superseded ?? [])].slice(0, SUPERSEDED_KEEP);
      pit.summary = text;
      pit.at = Date.now();
      st.oldNode = pit;
      // The merge lands at the very FRONT of the conversation: it invalidates the
      // whole prefix, not just a suffix, which is why it must be rare and big.
      st.lastEvent = 'pit-rewritten';
      saveState(key, st);
      const tokens = estimateTokens(text);
      debugLog(cf, `OLD ${pit.id}: absorbed ${absorbed.length} node(s), ${ids.length} leaf/leaves; ${freedChars} chars leave the head -> a synthesis of ${tokens}t`);
      return {
        content: [{
          type: 'text',
          text: t('oldNodeSet')(pit.id, absorbed.length, ids.length, freedChars, tokens),
        }],
        details: { ok: true, id: pit.id, nodes: absorbed.length, leaves: ids.length, microChars, freedChars, tokens },
      };
    },
  });

  /**
   * `cwl_group` — how the LLM classifies leaves into a TOPIC node.
   *
   * A topic node is born COLLAPSED: `name` and `description` are written at birth and
   * never change, and the description stands for its leaves in the head from that moment
   * on. The description has to cover the FUTURE use of the topic — that is exactly what
   * makes adding leaves later free: their micros leave the head, the description stays
   * where it is, and no synthesis is written a second time. Rewriting it would move the
   * prefix of the index, so it is immutable by construction.
   *
   * The rules, all of them REFUSALS rather than silent fixes:
   *  - only the leaves of the BUFFER can be grouped or moved: it is the last node, the one
   *    attached to the leaves that are still open. Older material stays where it is;
   *  - a leaf already inside the old node cannot be grouped: the pit's synthesis stands
   *    for it, and pulling it back would describe the same content twice;
   *  - a leaf without a micro cannot be grouped: it would not appear in any head;
   *  - a topic can never be the FIRST node, so it is inserted BEHIND the buffer;
   *  - the size guard is the one a merge passes too: `mergeMinRatio` x the synthesis
   *    estimate, and never less than `mergeMinChars`.
   *
   * The micros are NOT deleted. They are what the state knows about the leaves; the head
   * simply stops showing them, because the description of the topic stands in their place.
   */
  // #7 — the batch reuses THIS tool's execute, so the validations and the mutations live in
  // ONE place and cannot drift apart. The reference is assigned right after the registration
  // and read only at call time, so it is never null when the batch uses it.
  // #7 — the batch reuses THIS tool's own execute, so the validations and the mutations live in
  // ONE place and cannot drift apart. The reference is taken from a NAMED function expression
  // (`execute: async function executeGroup(...)`), and that detail is the whole trick: a method
  // shorthand has no name to refer to, and pulling the object literal out into a `const` to get
  // one DESTROYS its contextual type — MEASURED, that produced eleven TS7006 errors, turned
  // `params` into `any` and collapsed the inference of every return value in the tool.
  let groupExec: ((...args: unknown[]) => Promise<{ content: Array<{ type: string; text: string }>; details: Record<string, unknown> }>) | null = null;

  pi.registerTool({
    name: 'cwl_group',
    label: 'CWL Group',
    description: t('tools').groupDesc,
    parameters: Type.Object({
      leaves: Type.Optional(Type.Array(Type.String(), { description: t('tools').groupLeaves })),
      nodes: Type.Optional(Type.Array(Type.String(), { description: t('tools').groupNodes })),
      node: Type.Optional(Type.String({ description: t('tools').groupNode })),
      name: Type.Optional(Type.String({ description: t('tools').groupName })),
      description: Type.Optional(Type.String({ description: t('tools').groupText })),
      pit: Type.Optional(Type.Boolean({ description: t('tools').groupPit })),
      dryRun: Type.Optional(Type.Boolean({ description: t('tools').groupDryRun })),
      groups: Type.Optional(Type.Array(Type.Object({
        leaves: Type.Optional(Type.Array(Type.String())),
        nodes: Type.Optional(Type.Array(Type.String())),
        node: Type.Optional(Type.String()),
        name: Type.Optional(Type.String()),
        description: Type.Optional(Type.String()),
        pit: Type.Optional(Type.Boolean()),
      }), { description: t('tools').groupGroups })),
    }),
    execute: async function executeGroup(_toolCallId, params, _signal, _onUpdate, ctx) {
      groupExec = executeGroup as unknown as typeof groupExec;
      const key = sessionKey(ctx);
      const st = getState(key);
      const cf = getConfig(key);
      refreshNodes(st, cf);

      // #2 — THE DRY RUN. `dryRun: true` runs EVERY validation and every decision of a real
      // call — so the prediction is the real one, refusals included — and then puts the state
      // back. The snapshot is taken HERE, right after `refreshNodes` has normalised the nodes,
      // so putting it back cannot undo work that a real call would have done anyway. It holds
      // ONLY the two plain-data fields this tool can mutate (`nodes` and `oldNode`): a JSON
      // round trip of the whole state would flatten `graph`, which is an instance and is not
      // touched here. The state is the LIVE object the next tool call reads (`getState` caches
      // it in a map), so putting it back is not optional: a dry run that left a trace would be
      // worse than no dry run at all.
      const dry = params.dryRun === true;
      const snapshot = dry
        ? {
            nodes: JSON.parse(JSON.stringify(st.nodes)) as SpanNode[],
            oldNode: st.oldNode ? (JSON.parse(JSON.stringify(st.oldNode)) as OldNode) : null,
          }
        : null;
      const restore = (): void => {
        if (!snapshot) return;
        st.nodes = snapshot.nodes;
        st.oldNode = snapshot.oldNode;
      };

      // #6 — STRUCTURED ERRORS. A refusal already carried `why` and `detail` as prose; a caller
      // that has to FIX its call needs the pieces, not the sentence: the CODE of the refusal,
      // the IDs involved, the CONSTRAINT that was violated and the values that would be
      // ALLOWED. All machine-readable, no semantic interpretation. The old fields stay exactly
      // where they were, so nothing that already reads them changes.
      const refuse = (why: string, detail: string, extra?: { ids?: string[]; constraint?: Record<string, unknown>; allowed?: string[] }) => ({
        content: [{ type: 'text' as const, text: t('groupRefused')(why, detail) }],
        details: {
          ok: false, error: 'group-refused', code: why, why, detail,
          ids: extra?.ids ?? [],
          ...(extra?.constraint ? { constraint: extra.constraint } : {}),
          ...(extra?.allowed ? { allowed: extra.allowed } : {}),
        },
      });

      // #7 — THE BATCH: several groupings as ONE operation. All or nothing, and idempotent.
      //
      // The validator is the DRY RUN OF THIS SAME TOOL, and that is the whole design: there is
      // one code path, so the batch cannot disagree with a single call. Phase 1 runs every group
      // with `dryRun: true` and mutates NOTHING; only if all of them pass does phase 2 apply
      // them. A batch that validated as it applied would leave a half-built index behind on the
      // first refusal, which is the failure this exists to prevent.
      if (Array.isArray(params.groups)) {
        const groups = params.groups as Array<Record<string, unknown>>;
        if (groups.length === 0) return refuse('no-groups', '', { allowed: ['pass-at-least-one-group'] });
        if (!groupExec) return refuse('batch-unavailable', '', { allowed: ['call-cwl_group-one-group-at-a-time'] });
        const plans: Array<{ group: Record<string, unknown>; already: string | null }> = [];
        // IDEMPOTENCE, and it is DETERMINISTIC: a topic born from a set of leaves gets an id
        // DERIVED from those leaves, so repeating the same request computes the same id. If a
        // node with that id already holds them, the request is ALREADY SATISFIED and re-applying
        // it would be the duplicate this rule forbids. Same for `node: <target>`: if every leaf
        // is already inside the target, there is nothing left to move.
        const satisfiedBy = (group: Record<string, unknown>, dry: Record<string, unknown>): string | null => {
          const leaves = Array.isArray(group.leaves) ? (group.leaves as string[]) : [];
          if (leaves.length === 0) return null;
          // WHERE TO LOOK. A dry run that PASSED names the destination in `id`. A dry run that
          // FAILED does not — and the failure that means "already done" is `leaf-in-a-topic`,
          // which names the topic holding the leaves in its own `constraint.owner`. Without
          // this second source the idempotence rule would only ever fire on a call that was
          // going to succeed anyway, which is exactly the case that does not need it.
          const constraint = dry.constraint && typeof dry.constraint === 'object'
            ? (dry.constraint as Record<string, unknown>)
            : {};
          const owner = typeof constraint.owner === 'string' ? constraint.owner : null;
          const wanted = typeof group.node === 'string' ? group.node : (owner ?? String(dry.id ?? ''));
          const target = st.nodes.find((nd) => nd.id === wanted);
          if (!target) return null;
          return leaves.every((leafId) => target.leaves.includes(leafId)) ? target.id : null;
        };
        for (let i = 0; i < groups.length; i++) {
          const probe = await groupExec('batch', { ...groups[i], dryRun: true }, _signal, _onUpdate, ctx);
          if (probe.details.ok === true) {
            plans.push({ group: groups[i], already: null });
            continue;
          }
          const code = String(probe.details.code ?? probe.details.error ?? 'refused');
          // ALREADY SATISFIED is not a failure: it is the answer to a request repeated twice,
          // and the whole point of idempotence is that the second answer is not an error.
          const done = satisfiedBy(groups[i], probe.details);
          if (!done) {
            return refuse('batch-refused', `#${i} (${code})`, {
              ids: Array.isArray(probe.details.ids) ? (probe.details.ids as string[]) : [],
              constraint: { index: i, code, detail: probe.details.detail ?? '', group: groups[i] },
              allowed: Array.isArray(probe.details.allowed) ? (probe.details.allowed as string[]) : [],
            });
          }
          plans.push({ group: groups[i], already: done });
        }
        // PHASE 2 — APPLY. Everything passed validation, so this loop has nothing left to
        // refuse: the only failure it can produce is one a bug would cause. Each group saves its
        // own state, so even a crash mid-way leaves a consistent prefix rather than a torn one.
        let movedLeaves = 0;
        let movedNodes = 0;
        let alreadyCount = 0;
        for (let i = 0; i < plans.length; i++) {
          if (plans[i].already) { alreadyCount += 1; continue; }
          const res = await groupExec('batch', { ...plans[i].group, dryRun: false }, _signal, _onUpdate, ctx);
          if (res.details.ok !== true) {
            return refuse('batch-partial', `#${i}`, {
              constraint: { index: i, applied: i, code: res.details.code ?? res.details.error ?? 'refused' },
              allowed: ['call-cwl_group-one-group-at-a-time-to-finish-it'],
            });
          }
          movedLeaves += Number(res.details.added ?? res.details.leaves ?? 0);
          movedNodes += Array.isArray(res.details.absorbed) ? res.details.absorbed.length : 0;
        }
        return {
          content: [{ type: 'text', text: t('groupManyDone')(groups.length, alreadyCount, movedLeaves, movedNodes) }],
          details: {
            ok: true, groups: groups.length, applied: plans.length - alreadyCount,
            already: alreadyCount, leaves: movedLeaves, nodes: movedNodes,
          },
        };
      }

      const byId = new Map(st.spans.map((s) => [idOfSpan(s), s]));
      const nodeOf = (id: string): SpanNode | undefined => st.nodes.find((nd) => nd.leaves.includes(id));
      const inPit = new Set(st.oldNode?.nodes ?? []);
      const buffer = st.nodes[st.nodes.length - 1];
      const ids = (Array.isArray(params.leaves) ? params.leaves : []).filter((s) => typeof s === 'string');
      const nodeIds = (Array.isArray(params.nodes) ? params.nodes : []).filter((s) => typeof s === 'string');
      // `pit: true` — the topic is born INSIDE the old node, over leaves that are already
      // archived. Same tool, opposite operation: see the two rules in the loop below.
      const wantPit = params.pit === true;

      for (const id of ids) {
        const leaf = byId.get(id);
        if (!leaf) return refuse('unknown-leaf', id, { ids: [id], allowed: ['cwl_map', 'cwl_find', 'cwl_pending'] });
        if (!leaf.micro) return refuse('leaf-without-micro', id, { ids: [id], allowed: ['cwl_micro'] });
        const owner = nodeOf(id);
        // TWO OPPOSITE OPERATIONS, one tool. At the frontier a topic collates leaves that are
        // still open, and the pit is off limits for it. Inside the pit it is the other way
        // round: the leaves must ALREADY be there — the pit's synthesis stands for them, and
        // this only gives them a name — and the buffer rule does not apply, because nothing in
        // the index is moving. A pit topic is a catalogue entry, not a saving.
        if (wantPit) {
          if (!owner || !inPit.has(owner.id)) return refuse('leaf-not-in-the-pit', `${id} (${owner ? owner.id : 'loose'})`, { ids: [id], constraint: { owner: owner ? owner.id : null, pit: true }, allowed: ['cwl_old', 'pit:false'] });
          continue;
        }
        if (owner && inPit.has(owner.id)) return refuse('leaf-in-the-pit', `${id} (${owner.id})`, { ids: [id], constraint: { owner: owner.id, pit: true }, allowed: ['pit:true'] });
        if (owner && owner !== buffer) {
          if (!cf.groupBeyondBuffer) return refuse('leaf-not-in-the-buffer', `${id} (${owner.id})`, { ids: [id], constraint: { owner: owner.id, buffer: buffer ? buffer.id : null }, allowed: ['groupBeyondBuffer'] });
          // A TOPIC is never a source. Its leaves are NOT in the head — one description stands
          // for all of them — so taking one out would make that description a lie about what
          // the topic holds, and the leaf would come back into the head as a micro.
          if (owner.description) return refuse('leaf-in-a-topic', `${id} (${owner.id})`, { ids: [id], constraint: { owner: owner.id, hasDescription: true }, allowed: ['pick-leaves-of-a-node-with-no-description'] });
        }
      }
      // THE TOPIC IS ONE BLOCK IN A CHRONOLOGICAL INDEX, so its leaves must be a chronological
      // BLOCK too. Scattered NODES are fine — the ones in between are emptied, and what they keep
      // is re-partitioned around the topic — but scattered LEAVES are not: a leaf left between two
      // of the topic's own leaves has nowhere to go that is both inside the topic's span and
      // outside it. Checked only in the new mode: under the old rule the buffer and the loose
      // leaves are the newest ones, so they are consecutive by construction.
      const uniqueIds = [...new Set(ids)];
      const chrono = [...st.spans].sort((a, b) => a.at - b.at).map((s) => idOfSpan(s));
      const order = new Map(chrono.map((leafId, i) => [leafId, i]));
      const contiguity = (extra: string[]): string | null => {
        const set = [...new Set([...uniqueIds, ...extra])];
        if (set.length < 2) return null;
        const pos = set.map((leafId) => order.get(leafId) ?? 0).sort((a, b) => a - b);
        const span = pos[pos.length - 1] - pos[0] + 1;
        return span === pos.length ? null : `${set.length} leaf/leaves over ${span} positions`;
      };
      if (cf.groupBeyondBuffer && !wantPit && !params.node) {
        const broken = contiguity([]);
        if (broken) return refuse('leaves-not-contiguous', broken, { ids: uniqueIds, constraint: { contiguity: broken }, allowed: ['pick-leaves-that-are-consecutive-in-time'] });
      }

      if (params.node) {
        const target = st.nodes.find((nd) => nd.id === params.node);
        if (!target) return refuse('unknown-node', String(params.node));
        if (!target.description) return refuse('not-a-topic', target.id);
        // ABSORBING NODES (the containment). Two rules carry the whole safety of the feature:
        //   1. BACKWARD ONLY — a node absorbs only nodes that come AFTER it in `st.nodes`, so
        //      the parent's position, and everything before it in the index, never move.
        //   2. THE BUFFER IS NEVER ABSORBED — the last node is the one attached to the open
        //      leaves and it must stay a plain node: a topic cannot act as a buffer.
        // The absorbed node keeps its leaves and its description and stays readable with
        // cwl_open: what changes is that its block is no longer injected, because the parent's
        // description stands for it now. The cost is declared, never hidden.
        const targetIdx = st.nodes.indexOf(target);
        const absorbed: SpanNode[] = [];
        for (const id of nodeIds) {
          const child = st.nodes.find((nd) => nd.id === id);
          if (!child) return refuse('unknown-node', id);
          if (child === target) return refuse('node-contains-itself', id);
          if (child === buffer) return refuse('node-is-the-buffer', id);
          if (st.nodes.indexOf(child) < targetIdx) return refuse('only-backward', `${child.id} is older than ${target.id}`);
          absorbed.push(child);
        }
        // A topic INSIDE the pit is still a catalogue the agent keeps using, so it may keep
        // receiving leaves: the historical archive gets used, and long sessions stay
        // catalogued instead of piling everything into one buffer. THE TRADE-OFF, said out
        // loud: the pit's synthesis was written BEFORE these leaves arrived, so the synthesis
        // does not describe them — the topic's own description is the living copy, and both
        // are readable with cwl_open.
        const inThePit = inPit.has(target.id);
        const rewrite = (params.description ?? '').trim();
        // Immutable while the topic is OUTSIDE the pit: there the description IS the head, and
        // rewriting it would move the prefix of the index. Inside the pit it no longer touches
        // the context, so the exception is safe, and an updated description is what keeps a
        // topic usable when relevant leaves join it later.
        if (rewrite && !inThePit) return refuse('description-is-immutable', target.id);
        if (rewrite) target.description = rewrite;
        if (ids.length === 0 && !rewrite && absorbed.length === 0) return refuse('no-leaves', target.id);
        const moving = new Set(ids);
        for (const nd of st.nodes) if (nd !== target) nd.leaves = nd.leaves.filter((x) => !moving.has(x));
        target.leaves.push(...ids);
        if (absorbed.length) {
          const kids = new Set(target.children ?? []);
          for (const child of absorbed) kids.add(child.id);
          target.children = [...kids];
        }
        // #2 — THE DRY RUN. Every validation of this branch has already run, so the
        // prediction is the real one, refusals included. Nothing is written and the state is
        // put back from the snapshot taken at the top.
        if (dry) {
          const floor = Math.max(Math.round(cf.mergeMinRatio * MERGE_SYNTHESIS_CHARS), cf.mergeMinChars);
          const labelChars = ids.reduce((n, leafId) => n + (byId.get(leafId)?.micro?.length ?? 0), 0);
          restore();
          return {
            content: [{
              type: 'text' as const,
              text: t('groupDryRunAdd')(target.id, target.name ?? target.id, ids.length, absorbed.length, labelChars, floor),
            }],
            details: {
              ok: true, dryRun: true, id: target.id, name: target.name, wouldAdd: ids.length,
              wouldAbsorb: absorbed.map((c) => c.id), leaves: target.leaves.length,
              inPit: inThePit, descriptionUpdated: Boolean(rewrite),
              microChars: labelChars, needChars: floor,
            },
          };
        }
        saveState(key, st);
        debugLog(cf, `GROUP ${target.id} "${target.name ?? ''}": +${ids.length} leaf/leaves (now ${target.leaves.length})${absorbed.length ? `, +${absorbed.length} node(s) absorbed (${absorbed.map((c) => c.id).join(', ')})` : ''}${inThePit ? ', the node is in the pit' : ''}${rewrite ? `, description rewritten (${rewrite.length} chars)` : ''}`);
        return {
          content: [{
            type: 'text' as const,
            text: t('groupAdded')(target.name ?? target.id, target.id, target.leaves.length)
              + (absorbed.length
                ? t('groupAbsorbed')(
                    target.name ?? target.id, target.id, absorbed.length,
                    absorbed.reduce((n, c) => n + c.leaves.length, 0), absorbed.length,
                  )
                : '')
              + (rewrite ? t('groupDescriptionUpdated')() : ''),
          }],
          details: { ok: true, id: target.id, name: target.name, leaves: target.leaves.length, added: ids.length, absorbed: absorbed.map((c) => c.id), inPit: inThePit, descriptionUpdated: Boolean(rewrite) },
        };
      }

      const name = (params.name ?? '').trim();
      const description = (params.description ?? '').trim();
      if (!name || !description) return refuse('name-and-description-required', '');
      // At the FRONTIER a new topic cannot be born from NODES: a parent must already exist to
      // hold its position, so the containment there is always `node: <the parent>` +
      // `nodes: [<the child>]`. Inside the pit a new parent over nodes is fine.
      if (nodeIds.length > 0 && !wantPit) return refuse('nodes-need-a-topic', `${nodeIds.length} node(s)`);
      if (ids.length === 0 && nodeIds.length === 0) return refuse('no-leaves', '');
      if (!buffer) return refuse('no-buffer', '');
      const microChars = ids.reduce((n, id) => n + (byId.get(id)?.micro?.length ?? 0), 0);
      // A topic born INSIDE the pit over NODES: the same containment, with the parent created
      // here. The absorbed nodes must sit AFTER the insertion point (the parent is placed right
      // after the last node already in the pit), which is the backward rule, checked instead of
      // assumed. Their blocks leave the head and the new description stands for them.
      const absorbedNew: SpanNode[] = [];
      if (wantPit && nodeIds.length > 0 && st.oldNode) {
        const lastPitIdx = st.nodes.reduce((idx, nd, i) => (inPit.has(nd.id) ? i : idx), -1);
        for (const id of nodeIds) {
          const child = st.nodes.find((nd) => nd.id === id);
          if (!child) return refuse('unknown-node', id);
          if (child === buffer) return refuse('node-is-the-buffer', id);
          if (st.nodes.indexOf(child) <= lastPitIdx) return refuse('only-backward', `${child.id} is not after the pit`);
          absorbedNew.push(child);
        }
      }
      // A topic INSIDE the old node has no size guard, because nothing in the index changes:
      // the pit's synthesis already stands for those leaves, and the name only makes them
      // findable again. The floor is a sanity rule, not a saving one: one or two leaves are a
      // leaf list, not a catalogue entry.
      if (wantPit) {
        if (!st.oldNode) return refuse('no-pit', '');
        const held = ids.length + absorbedNew.reduce((n, c) => n + c.leaves.length, 0);
        if (held < PIT_TOPIC_MIN_LEAVES) return refuse('pit-too-few', `${held} of ${PIT_TOPIC_MIN_LEAVES}`);
      }
      const needChars = Math.max(Math.round(cf.mergeMinRatio * MERGE_SYNTHESIS_CHARS), cf.mergeMinChars);
      if (!wantPit && microChars < needChars) {
        debugLog(cf, `GROUP not-born: ${ids.length} leaf/leaves hold ${microChars} chars, need ${needChars}`);
        return {
          content: [{ type: 'text', text: t('groupTooSmall')(ids.length, microChars, needChars) }],
          details: {
            ok: false, error: 'too-small', code: 'labels-too-small',
            leaves: ids.length, microChars, needChars,
            ids: uniqueIds, constraint: { microChars, needChars, ratio: cf.mergeMinRatio, floor: cf.mergeMinChars },
            allowed: ['group-more-leaves', 'lower-mergeMinRatio', 'lower-mergeMinChars'],
          },
        };
      }
      const base = hashText([...ids, ...nodeIds].join('|'));
      let id = `nd-${base.slice(0, 8)}`;
      for (let n = 2; st.nodes.some((nd) => nd.id === id); n++) id = `nd-${base.slice(0, 8)}-${n}`;
      const moving = new Set(uniqueIds);
      // WHERE THE TOPIC LANDS, and what happens to what it does NOT take. The involved nodes are
      // the ones the taken leaves come from; the topic is born at the position of the FIRST of
      // them, so everything before it in the index stays where it is. What those nodes keep is
      // re-partitioned AROUND the topic — the older leftovers in front, the newer behind, chunked
      // by `nodeCapacity` — because the planner would otherwise append them to the NEWEST node,
      // moving them in time. A node left empty is dropped by `refreshNodes` itself; the buffer is
      // not, and that is why the topic never takes its place.
      const involved = st.nodes.filter((nd) => nd.leaves.some((x) => moving.has(x)));
      const firstTaken = Math.min(...uniqueIds.map((leafId) => order.get(leafId) ?? 0));
      const older: string[] = [];
      const newer: string[] = [];
      for (const nd of involved) {
        for (const x of nd.leaves) {
          if (moving.has(x)) continue;
          ((order.get(x) ?? 0) < firstTaken ? older : newer).push(x);
        }
      }
      const chunkNodes = (list: string[]): SpanNode[] => {
        const out: SpanNode[] = [];
        for (let i = 0; i < list.length; i += cf.nodeCapacity) {
          const slice = list.slice(i, i + cf.nodeCapacity);
          const stem = `nd-${hashText(slice.join('|')).slice(0, 8)}`;
          let chunkId = stem;
          for (let n = 2; st.nodes.some((nd) => nd.id === chunkId) || out.some((nd) => nd.id === chunkId); n++) {
            chunkId = `${stem}-${n}`;
          }
          out.push({ id: chunkId, leaves: slice, at: Date.now() });
        }
        return out;
      };
      const insertAt = involved.length > 0
        ? st.nodes.indexOf(involved[0])
        : Math.max(0, st.nodes.length - 1);
      const olderNodes = chunkNodes(older);
      const newerNodes = chunkNodes(newer);
      for (const nd of st.nodes) nd.leaves = nd.leaves.filter((x) => !moving.has(x));
      const node: SpanNode = {
        id,
        leaves: [...uniqueIds],
        name,
        description,
        children: absorbedNew.length ? absorbedNew.map((c) => c.id) : undefined,
        at: Date.now(),
      };
      if (wantPit && st.oldNode) {
        // Inside the pit the topic joins the ARCHIVE's own node list, right after the last node
        // already there. The pit's synthesis is left byte for byte as it was: no synthesis is
        // written, and the prefix of the index — the pit block is the FIRST thing in it — does
        // not move. That is the whole saving of this operation, and its whole point.
        st.oldNode.nodes.push(id);
        const lastPit = st.nodes.reduce((idx, nd, i) => (inPit.has(nd.id) ? i : idx), -1);
        st.nodes.splice(lastPit + 1, 0, node);
      } else {
        st.nodes.splice(insertAt, 0, ...olderNodes, node, ...newerNodes);
      }
      // #2 — the dry run for a NEW topic. Everything is decided by now: the id it would get,
      // where it would land, what it would take. The state is put back and nothing is written.
      if (dry) {
        restore();
        return {
          content: [{
            type: 'text' as const,
            text: t('groupDryRunNew')(id, name, uniqueIds.length, absorbedNew.length, microChars, needChars),
          }],
          details: {
            ok: true, dryRun: true, id, name, leaves: uniqueIds.length,
            absorb: absorbedNew.map((c) => c.id), microChars, needChars, pit: wantPit,
          },
        };
      }
      saveState(key, st);
      debugLog(cf, `GROUP ${id} "${name}": born ${wantPit ? 'INSIDE the pit' : 'at the frontier'} with ${ids.length} leaf/leaves, ${microChars} chars of micros, description of ${description.length} chars`);
      return {
        content: [{ type: 'text', text: wantPit
          ? t('pitTopicBorn')(name, id, ids.length)
          : t('groupCreated')(name, id, ids.length, microChars) }],
        details: { ok: true, id, name, leaves: ids.length, microChars, descriptionChars: description.length, pit: wantPit },
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
    // that the rest of the hook (trigger check, block compression,
    // `runEvictionPass`) runs on the list the agent will really see. Every
    // index-based decision below — the episode ranges and `safetyFloor` — is
    // resolved against whatever list this variable holds, so the two must never
    // disagree.
    let messages: AgentMessage[] = eventMessages.filter((m) => !isGateMessage(m) && !isInheritedMessage(m) && !isDemandMessage(m));

    // MEASUREMENT of the cache — the instrument this design never had. The provider
    // cache is a PREFIX cache, and every leaf written here lands where the
    // compressed content used to be while a merge lands at the very FRONT: both
    // invalidate everything after them, and the next request pays a WRITE where it
    // used to pay a READ. This row charges that cost to the event that caused it,
    // which is the only way to tell whether batching writes is worth anything.
    // ONE row per REQUEST: a tool loop calls this hook many times for the same
    // request, and the timestamp is what tells the calls apart.
    const usage = lastUsageOf(messages);
    if (usage && usage.ts !== st.lastUsageTs) {
      st.lastUsageTs = usage.ts;
      const prefix = usage.cacheRead + usage.cacheWrite + usage.input;
      const hit = prefix > 0 ? Math.round((usage.cacheRead / prefix) * 100) : 0;
      st.providerContextTokens = prefix + usage.output;

      // Online calibration of charTokenRatio and systemOverheadTokens based on real provider numbers
      let totalMsgChars = 0;
      for (const m of messages) {
        try { totalMsgChars += JSON.stringify(m).length; } catch { /* skip */ }
      }

      if (prefix > 0 && totalMsgChars > 0) {
        if (st.lastCalibTokens > 0 && prefix > st.lastCalibTokens && totalMsgChars > st.lastCalibChars) {
          const deltaChars = totalMsgChars - st.lastCalibChars;
          const deltaTokens = prefix - st.lastCalibTokens;
          if (deltaTokens >= 300) {
            const measuredRatio = deltaChars / deltaTokens;
            if (measuredRatio >= MIN_CHAR_TOKEN_RATIO && measuredRatio <= MAX_CHAR_TOKEN_RATIO) {
              st.charTokenRatio = Math.round((0.7 * st.charTokenRatio + 0.3 * measuredRatio) * 100) / 100;
            }
            st.lastCalibChars = totalMsgChars;
            st.lastCalibTokens = prefix;
          }
        } else if (st.lastCalibTokens === 0) {
          st.lastCalibChars = totalMsgChars;
          st.lastCalibTokens = prefix;
        }

        const estimatedMsgTokens = Math.ceil(totalMsgChars / st.charTokenRatio);
        st.systemOverheadTokens = Math.max(0, prefix - estimatedMsgTokens);
      }

      debugLog(cf,
        `CACHE read=${usage.cacheRead} write=${usage.cacheWrite} input=${usage.input}`
        + ` output=${usage.output} hit=${hit}% after=${st.lastEvent}`
        + ` since-compress=${st.turnsSinceCompress}`
        + ` ratio=${st.charTokenRatio.toFixed(2)} overhead=${st.systemOverheadTokens}t`);
      // Charged once: the event explains THIS row and no other.
      st.lastEvent = 'none';
    }

    const ctxUsage = typeof (ctx as unknown as { getContextUsage?: () => { tokens: number; contextWindow: number; percent: number } }).getContextUsage === 'function'
      ? (ctx as unknown as { getContextUsage: () => { tokens: number; contextWindow: number; percent: number } }).getContextUsage()
      : null;
    if (ctxUsage && typeof ctxUsage.tokens === 'number' && ctxUsage.tokens > 0) {
      st.providerContextTokens = ctxUsage.tokens;
    }

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
      currentTokens += estimateMessageTokens(m, st.charTokenRatio);
    }
    st.lastMeasuredTokens = st.providerContextTokens > 0
      ? Math.max(currentTokens, st.providerContextTokens)
      : currentTokens + st.systemOverheadTokens;

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
      const msgTokens = list.reduce((s: number, m: AgentMessage) => s + estimateMessageTokens(m, st.charTokenRatio), 0);
      const after = st.providerContextTokens > 0
        ? Math.max(msgTokens, st.providerContextTokens)
        : msgTokens + st.systemOverheadTokens;
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
      // Only demand what the extension can actually deliver — and MEASURE it on
      // THIS list instead of trusting the state: an ask no call can satisfy costs
      // the agent a turn, distracts it and dirties the context. Two ways to be
      // impossible, both seen in a live session:
      //  - an ACTIVE episode whose anchors the native compaction took: MEASURED as
      //    "EPISODES unlocatable: 4 of 4". Ending it frees nothing, because the
      //    eviction cannot locate the content it would remove;
      //  - a stored RANGE whose endpoints are no longer in the list the agent will
      //    see: the leaf would be born and pruned right after ("SPANS pruned"), so
      //    the call would answer ok and free nothing.
      // `turn_end` arms the gate on the static pair and that is enough: withholding
      // here leaves it armed, so the demand fires as soon as it is doable.
      const activeEps = st.graph.active();
      // Closing an OPEN episode is judged by its ANCHOR, not by `episodeRanges`:
      // that function resolves closed episodes (it needs the end anchor, which an
      // open one does not have yet) and would call every open episode impossible.
      // Two things make the ask useless: an episode begun at the very end of the
      // list (closing it would save nothing — `to <= from`), and no material at all.
      // A LOST start anchor is not a reason to stay silent: closing it makes the
      // range deduced from 0, which the eviction can act on.
      const anchors = activeEps.length > 0 ? toolCallPositions(list) : null;
      const canClose = activeEps.some((ep) => {
        const from = anchors?.get(ep.startToolCallId);
        return from === undefined ? list.length >= 2 : from < list.length - 1;
      });
      const startH = st.rangeStartHash;
      const endH = st.rangeEndHash;
      // Option B: canCompress is true only if start/end exist AND there is actual material (rangeTokens > 0)
      const canCompress = startH !== null && endH !== null && st.rangeTokens > 0;
      if (!canClose && !canCompress) {
        const why =
          activeEps.length > 0
            ? 'the only open episode(s) begin at the end of the list: closing them frees nothing'
            : (st.systemOverheadTokens > 0 && msgTokens <= trigger
              ? `context ${after}t is over trigger ${Math.round(trigger)}t due to incompressible system overhead (~${st.systemOverheadTokens}t): history (~${msgTokens}t) is within limits`
              : 'no episode is open and no compressible range is available (remaining history is inside safety floor or already compressed)');
        if (st.gateWithheld !== why) {
          st.gateWithheld = why;
          debugLog(cf, `GATE withheld: ${why} — the demand would ask for something no call can do`);
        }
        return { messages: list };
      }
      st.gateWithheld = '';
      // The demand is DELIVERED now, and that is the only thing that makes a turn
      // count as an attempt: `turn_end` reads this flag. Without it the gate charged
      // the agent for turns in which the request was never shown — measured: it gave
      // up with "unanswered for 3 turns" right after a withheld turn.
      st.demandShown = true;
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
    const nodesBefore = st.nodes.map((nd) => `${nd.id}:${nd.leaves.length}`).join(',');
    const plan = refreshNodes(st, cf);
    const nodesAfter = st.nodes.map((nd) => `${nd.id}:${nd.leaves.length}`).join(',');
    if (nodesAfter !== nodesBefore || plan.waiting > 0) {
      debugLog(cf, `NODES: ${st.nodes.length} node(s) [${nodesAfter || 'none'}]${plan.formed > 0 ? `, formed ${plan.formed}` : ''}, ${plan.waiting} leaf/leaves waiting for a micro — their body is still in the context`);
    }
    // The merge needs a synthesis only the AGENT can write. Until this demand existed it
    // reached the log and not the agent: the extension knew, the operator could read it,
    // and the only one able to write the merge summary never called cwl_old. The mechanism
    // was tested and could not fire in a real session — so the demand goes where the
    // compression demand already goes: into the context.
    const young = st.nodes.filter((nd) => !new Set(st.oldNode?.nodes ?? []).has(nd.id)).length;
    const mergeRequest = plan.due > 0 ? t('indexDue')(young, cf.mergeNodesAt) : null;
    // The topic invitation exists because the INDEX ITSELF closes the window it depends on:
    // a leaf inside a node can never be moved again, so a topic that is not born while its
    // leaves are still in the buffer is lost for good. Until now the agent learned the size
    // guard by trying and being refused, one turn and one failed call at a time; here it is
    // told BEFORE, with every number it needs to act.
    const buffer = st.nodes[st.nodes.length - 1];
    const microCharsById = new Map(st.spans.map((s) => [idOfSpan(s), (s.micro ?? '').length]));
    const bufferMicroChars = buffer
      ? buffer.leaves.reduce((n, id) => n + (microCharsById.get(id) ?? 0), 0)
      : 0;
    const topicNeedChars = Math.max(Math.round(cf.mergeMinRatio * MERGE_SYNTHESIS_CHARS), cf.mergeMinChars);
    // The invitation fires ONLY when the guard would pass: `bufferMicroChars >= topicNeedChars`.
    // One that cannot be acted on would teach the agent nothing and cost a demand per turn.
    // `!buffer.name` only guards a state written by an older version — a topic can never be
    // the first node, because it cannot act as a buffer.
    const topicRequest =
      buffer && !buffer.name && buffer.leaves.length > cf.topicInviteAt && bufferMicroChars >= topicNeedChars
        ? t('topicDue')(buffer.id, buffer.leaves.length, bufferMicroChars, topicNeedChars)
        : null;
    if (topicRequest && buffer) {
      debugLog(cf, `TOPIC due: the buffer ${buffer.id} holds ${buffer.leaves.length} leaf/leaves (${bufferMicroChars} chars of micros, need ${topicNeedChars}) — asked in the context`);
    }
    // The index shape belongs in the TUI, NOT in the context: a widget is UI, it costs no
    // tokens and it cannot nudge the agent. `showWidget: false` turns it off, and the
    // option is finally read by somebody: it was declared, defaulted and validated, and
    // until now nothing drew anything.
    if (ctx.hasUI) {
      const line = cf.showWidget ? `CWL \u25b8 ${t('indexLine')(...indexShape(st, cf))}` : '';
      if (widgetLines.get(key) !== line) {
        widgetLines.set(key, line);
        ctx.ui.setWidget(WIDGET_ID, line ? [line] : undefined, { placement: 'belowEditor' });
      }
    }
    if (mergeRequest) {
      debugLog(cf, `OLD NODE due: ${cf.mergeNodesAt}+ young nodes — the oldest ones should merge: write the merge summary with cwl_old`);
    }
    // The FIRST step of the index needs the agent too, and it was the step with NO voice: the
    // leaves without a micro were listed in the LOG and nothing in the context asked for them,
    // so with zero nodes `plan.due` is 0, the merge demand never appears, and the mechanism
    // cannot start at all. MEASURED live, right after the reload that made the index run:
    // `NODES: 0 node(s) [none], 40 leaf/leaves waiting for a micro — their body is still in
    // the context` at every single turn, 45 spans, 90.415t of summaries still in the context,
    // and no demand anywhere. The request names a FEW ids instead of all forty: it stays
    // small, it rides at the end of every turn until the work is done, and the agent does the
    // rest next turn.
    const microRequest =
      plan.waiting > 0
        ? t('leavesDue')(plan.waiting, plan.waitingIds.slice(0, MICRO_DEMAND_IDS).join(' '))
        : null;
    if (microRequest) {
      debugLog(cf, `MICRO due: ${plan.waiting} leaf/leaves waiting for a micro — asked in the context (${Math.min(plan.waiting, MICRO_DEMAND_IDS)} id(s) named)`);
    }
    const saveRequest = st.forceAllNext ? t('cmdSaveDemand')(st.forceAllNote) : null;
    const demand =
      [saveRequest, microRequest, mergeRequest, topicRequest].filter((d): d is string => d !== null).join('\n\n') || null;

    if (st.spans.length > 0 || demand !== null || Boolean(st.importedFrom)) {
      // An inherited memory announces itself, and the announcement is an INJECTION at the END
      // (see `inheritedMsg`): without it the agent does not know that the index at the top of
      // its context is NOT its own past, and goes looking for its history on disk — the real
      // cost of that omission was a session that spent ~100k tokens working out why it existed.
      // It speaks only while this session has NO leaves of its own: that is exactly the window
      // in which the sentence is true ("this session has no earlier messages of its own"), and
      // afterwards it stops by itself, in the one position where stopping is free.
      const inheritedNotice = st.importedFrom && !st.spans.some((sp) => !sp.archived)
        ? t('inheritedHead')(
            st.memoryName ?? defaultMemoryName(key),
            st.importedFrom,
            st.spans.filter((sp) => sp.archived).length,
            st.nodes.length,
          )
        : null;
      const applied = applySpans(messages, st.spans, pitView(st), topicView(st), demand, inheritedNotice);
      // A span whose endpoints left the context can never apply again, and each
      // one carries a summary of thousands of characters that the state re-saves
      // on every turn. Pruned here — outside the `applied > 0` guard, because the
      // case that matters is when they are ALL dead — and said out loud.
      if (applied.dead.length > 0) {
        // A dead span can never apply again. The OLD contract dropped the summary, on the
        // argument that the original is still in the append-only transcript. The operator's
        // rule changed that: MICRO + SUMMARY must always remain, the original transcript may
        // go away. So the body (and the micro, when there is one) are appended to the body
        // store FIRST — one record per leaf — and only then the span leaves the state.
        // Whatever the failure order, nothing is lost: append before drop.
        let fd: number | null = null;
        try {
          for (const sp of applied.dead) {
            const id = idOfSpan(sp);
            if (!sp.summary || st.bodies.has(id)) continue;
            if (fd === null) fd = fs.openSync(bodiesPath(key), 'a');
            const line = `${JSON.stringify({ id, text: sp.summary, micro: sp.micro ?? undefined })}\n`;
            const offset = fs.fstatSync(fd).size;
            fs.writeSync(fd, line);
            st.bodies.set(id, [offset, Buffer.byteLength(line)]);
          }
        } finally {
          if (fd !== null) { try { fs.closeSync(fd); } catch { /* already closed */ } }
        }
        for (const sp of applied.dead) {
          if (sp.startSid && sp.endSid) {
            st.graves.push({ id: idOfSpan(sp), startSid: sp.startSid, endSid: sp.endSid, at: Date.now() });
          }
        }
        st.graves = st.graves.slice(-GRAVE_MAX);
        st.spans = st.spans.filter((s) => !applied.dead.includes(s));
        debugLog(cf, `SPANS pruned: ${applied.dead.length} span(s) that can never apply again — endpoints gone, or contained in another span (${st.spans.length} left, ${st.bodies.size} bodie(s) preserved on disk, ${st.graves.length} still recoverable by id from the transcript)`);
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
        const protTurns = st.forceAllNext ? 0 : cf.protectedTurns;
        const nextRange = compressibleRange(messages, st.spans, protTurns, st.charTokenRatio);
        storeRange(st, cf, nextRange, messages);
        rangeStoredBySpans = true;

        // THE RETURN HERE USED TO BE UNCONDITIONAL, and that switched the rest of
        // the hook off for good: `applied.applied > 0` is true on EVERY turn
        // while one span resolves (a span must be re-applied every time, or the
        // provider gets the uncompressed history back), so returning here made
        // the trigger check, block compression and `runEvictionPass`
        // unreachable for the rest of the session. MEASURED on a real session:
        // `SPANS re-applied: 3, nothing new to count` + `RANGE none | 178 msgs,
        // 133852t vs trigger 68000t` on every turn, with ZERO `EVICTION`, ZERO
        // `CONTEXT`, ZERO `no safe candidate` — 133k tokens against a 68k trigger,
        // one episode with evictable content, and the extension did nothing at
        // all. Returning is right only when the spans ALREADY did the job.
        const afterSpans = applied.kept.reduce((s: number, m: AgentMessage) => s + estimateMessageTokens(m, st.charTokenRatio), 0);
        if (afterSpans <= trigger) return finish(applied.kept);
        debugLog(cf, `SPANS applied (${afterSpans}t) still above trigger ${Math.round(trigger)}t: the episode pass and the fallback still run`);
        // Fall through on the COMPRESSED list, not the original one: `applySpans`
        // REPLACED the messages inside every span with its summary, so a range
        // resolved on the original list would point the eviction at messages that
        // no longer exist there.
        messages = applied.kept;
        spanInside = { set: new Set(applied.insideOut), len: applied.kept.length };
        currentTokens = afterSpans;
      } else if (applied.kept.length !== messages.length) {
        // Nothing resolved, and the list still grew: the only two ways are the INHERITED
        // memory (an adopted fork has no resolvable leaves, so `applied.applied` stays 0
        // while `kept` carries its blocks at the top) and the index demand riding at the
        // end. Either way `kept` IS the list to show — and the measured tokens must
        // follow it, or the trigger check below reads the pre-injection count.
        messages = applied.kept;
        currentTokens = applied.kept.reduce((s: number, m: AgentMessage) => s + estimateMessageTokens(m, st.charTokenRatio), 0);
      }
    }

    const totalContext = st.lastMeasuredTokens;
    if (currentTokens <= trigger) {
      if (st.forceAllNext && !rangeStoredBySpans) {
        const range = compressibleRange(messages, st.spans, 0, st.charTokenRatio);
        storeRange(st, cf, range, messages, currentTokens, trigger);
      }
      if (totalContext > trigger && st.systemOverheadTokens > 0) {
        // Option B: total context exceeds trigger only because of incompressible system overhead.
        // History is within limits: nothing to evict, and gate must be withheld.
        const why = `context ${totalContext}t is over trigger ${Math.round(trigger)}t due to incompressible system overhead (~${st.systemOverheadTokens}t): history (~${currentTokens}t) is within limits`;
        if (st.gateWithheld !== why) {
          st.gateWithheld = why;
          debugLog(cf, `GATE withheld: ${why}`);
        }
      } else {
        // Truly under budget: nothing to compact and nothing to ask for.
        st.overBudgetSince = -1;
        st.gateArmedTurn = -1;
        debugLog(cf, `CONTEXT ${currentTokens}t under threshold ${Math.round(trigger)}t: no eviction`);
      }
      persistIfUsed();
      return (droppedGate || st.forceAllNext || demand !== null) ? { messages } : undefined;
    }

    const protTurns = st.forceAllNext ? 0 : cf.protectedTurns;
    // The last `protectedTurns` user turns are inviolable: compaction must never
    // destroy the context the agent is working on.
    const safetyFloor = protectedFromIndex(messages, protTurns);

    // Addresses of the largest range the agent may ask to compress. Recomputed
    // here because this hook is the only place that sees the real message list.
    if (!rangeStoredBySpans) {
      const range = compressibleRange(messages, st.spans, protTurns, st.charTokenRatio);
      storeRange(st, cf, range, messages, currentTokens, trigger);
    }

    // No episodes at all: with no episodes, we do not perform global reasoning
    // strips on live messages because modifying historical turns destroys the
    // provider's prefix cache across the entire conversation.
    if (g.isEmpty) {
      debugLog(cf, 'CONTEXT above threshold but no episode: context untouched (reasoning fallback disabled for cache preservation)');
      return finish(messages);
    }

    // 2. Deterministic policy: compute what to evict and at which level
    const actions = runEvictionPass(cf, g, currentTokens, trigger, messages);
    if (actions.length === 0) {
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
            // STATIC TRACE: the marker stands where the episode's content used to be, so it
            // is re-pushed at every pass; `ep.openedAt` is when the episode opened, and it
            // does not move. `Date.now()` would have made the same marker a different message
            // on every turn.
            timestamp: ep.openedAt,
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
      // No reducible episode: leave history unchanged and offer block compression.
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
      debugLog(cf, `EVICTION accounting: the plan estimated ${planEstimate}t, the measured saving is ${measuredSaved}t (${planEstimate > measuredSaved ? '+' : ''}${planEstimate - measuredSaved}t)`);
    }

    if (ctx?.hasUI) {
      ctx.ui.notify(
        t('evictionNotice')(dropped, truncated, currentTokens.toLocaleString(), afterTokens.toLocaleString()),
        'info',
      );
    }

    // Do not rewrite unrelated historical reasoning. Explicit block compression
    // owns that reduction; changing an older message may invalidate its suffix.

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
    // One more turn at the price the last leaf set. It is `-1` until the first
    // leaf exists, because "turns since" a thing that never happened is not 0.
    if (st.turnsSinceCompress >= 0) st.turnsSinceCompress += 1;

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
          // ONE attempt = one turn in which the demand was really SHOWN and went
          // unanswered. Counting the turn regardless is what produced "unanswered for
          // 3 turns" immediately after a turn in which the gate had withheld the
          // demand: the agent was blamed for ignoring a request it never received.
          if (st.demandShown) {
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
    }
    // Spent by the turn that just ended, whether or not it counted as an attempt.
    st.demandShown = false;

    // Persist the DECISIONS (episode graph + traced compressions) at the end of
    // every turn. They cannot be recomputed from the transcript, so a crash or
    // a restart must not throw them away. Sessions that never used CWL write
    // nothing.
    if (!st.graph.isEmpty || st.spans.length > 0) saveState(key, st);
  });

  // -------------------------------------------------------------------------
  // The CONSULTATION tools. They exist because an id that cannot be FOUND is an
  // id that will be hunted by trial: the index is walked once, here, and every
  // tool below reads it through `indexNodeViews`, so they cannot disagree.
  // -------------------------------------------------------------------------

  pi.registerTool({
    name: 'cwl_find',
    label: 'CWL Find',
    description: t('consult').findDesc,
    promptSnippet: t('consult').findSnippet,
    parameters: Type.Object({
      query: Type.String({ description: t('consult').findQuery }),
      scope: Type.Optional(Type.String({ description: t('consult').findScope })),
      limit: Type.Optional(Type.Number({ description: t('consult').findLimit, minimum: 1, maximum: 50 })),
    }),
    async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
      const key = sessionKey(ctx);
      const st = getState(key);
      const cf = getConfig(key);
      const scope = (params.scope ?? 'index').toLowerCase();
      const limit = params.limit ?? 8;
      if (!recall) {
        return { content: [{ type: 'text', text: t('consult').findNotLoaded }], details: { ok: false, error: 'not-loaded' } };
      }
      const rows: string[] = [];
      const ids: string[] = [];
      let indexed = 0;

      // HALF ONE: the index. The labels of the leaves and the descriptions of the
      // topics — the half that was missing entirely, because a leaf inside a topic
      // had no other way in than opening the topic, i.e. guessing it first.
      if (scope === 'index' || scope === 'both') {
        const idx = new recall.Bm25Index();
        const leaves = leafViews(st);
        for (const nd of indexNodeViews(st)) {
          const text = `${nd.name ?? ''} ${nd.description ?? ''} ${nd.kind}`;
          if (!text.trim()) continue;
          idx.add(nd.id, {
            id: nd.id,
            role: nd.kind,
            ts: 0,
            preview: firstLine(`${nd.kind}${nd.name ? ` "${nd.name}"` : ''}${nd.description ? `: ${nd.description}` : ''}`, 160),
            hash: nd.id,
          }, text);
        }
        for (const v of leaves.values()) {
          if (!v.micro) continue;
          idx.add(v.id, {
            id: v.id,
            role: 'leaf',
            ts: 0,
            preview: firstLine(v.micro, 160),
            hash: v.id,
          }, v.micro);
        }
        indexed = idx.size;
        for (const hit of idx.search(params.query, limit)) {
          ids.push(hit.id);
          const container = leaves.get(hit.id)?.container;
          rows.push(`[${rows.length + 1}] ${hit.id} \u2502 ${hit.role}${container ? ` \u2502 in ${container}` : ''} \u2502 score=${hit.score.toFixed(2)}\n${hit.preview}`);
        }
      }

      // HALF TWO: the transcript. Same engine, different corpus: the messages.
      if (scope === 'transcript' || scope === 'both') {
        const file = findTranscript(key);
        const raw = file ? readFileOrNull(file) : null;
        if (raw !== null) {
          const idx = recall.indexTranscript(raw, scope === 'both' ? null : st.recallIndex);
          if (scope !== 'both') st.recallIndex = idx;
          indexed += idx.size;
          for (const hit of idx.search(params.query, limit)) {
            ids.push(hit.id);
            rows.push(`[${rows.length + 1}] ${hit.id} \u2502 ${hit.role} \u2502 score=${hit.score.toFixed(2)}\n${hit.preview}`);
          }
        }
      }

      debugLog(cf, `FIND scope=${scope} query="${params.query}" indexed=${indexed} hits=${rows.length}`);
      if (rows.length === 0) {
        return {
          content: [{ type: 'text', text: t('consult').findNoMatch(params.query) }],
          details: { ok: true, hits: 0, indexed, scope, ids: [] },
        };
      }
      return {
        content: [{ type: 'text', text: t('consult').findFound(rows.length, params.query, rows.join('\n\n')) }],
        details: { ok: true, hits: rows.length, indexed, scope, ids },
      };
    },
  });

  pi.registerTool({
    name: 'cwl_map',
    label: 'CWL Map',
    description: t('consult').mapDesc,
    promptSnippet: t('consult').mapSnippet,
    parameters: Type.Object({
      node: Type.Optional(Type.String({ description: t('consult').mapNode })),
    }),
    async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
      const key = sessionKey(ctx);
      const st = getState(key);
      const cf = getConfig(key);
      let nodes = indexNodeViews(st);
      if (params.node) {
        // A subtree: the node asked for, plus the nodes it holds, transitively.
        const wanted = new Set<string>([params.node]);
        let grew = true;
        while (grew) {
          grew = false;
          for (const nd of nodes) {
            if (wanted.has(nd.id)) continue;
            if (nd.children.some((c) => wanted.has(c))) { wanted.add(nd.id); grew = true; }
          }
        }
        nodes = nodes.filter((nd) => wanted.has(nd.id));
        if (nodes.length === 0) {
          return { content: [{ type: 'text', text: t('consult').nodeNotFound(params.node) }], details: { ok: false, error: 'node-not-found' } };
        }
      }
      const totalLeaves = nodes.reduce((n, nd) => n + nd.leaves.length, 0);
      const headChars = nodes.reduce((n, nd) => n + nd.chars, 0);
      const lines = [t('consult').mapHeader(nodes.length, totalLeaves, Math.round(headChars / (st.charTokenRatio || DEFAULT_CHAR_TOKEN_RATIO)).toLocaleString())];
      for (const nd of nodes) {
        lines.push(t('consult').mapLine(nd.id, nd.kind, nd.name ?? '', nd.leaves.length, nd.children.length, nd.chars));
        // The leaves are named ONLY where the head already pays for them (a node with
        // no description injects its micros). Listing a topic's leaves here would put
        // back the very ids that the description was written to replace.
        if (!nd.description && nd.leaves.length > 0) {
          lines.push(t('consult').mapLeaves(nd.leaves.join(' ')));
        }
      }
      // The loose leaves are not nodes, but they ARE part of the index: the head injects
      // their labels, and they are the working set. Naming their ids here is the difference
      // between a map of the NODES and a map of the INDEX.
      const looseIds = looseSpansOf(st, cf).map((sp) => idOfSpan(sp));
      if (looseIds.length > 0) lines.push(t('consult').mapLeaves(looseIds.join(' ')));
      debugLog(cf, `MAP nodes=${nodes.length} leaves=${totalLeaves}${params.node ? ` node=${params.node}` : ''}`);
      return {
        content: [{ type: 'text', text: lines.join('\n') }],
        details: { ok: true, nodes: nodes.length, leaves: totalLeaves, ids: nodes.map((nd) => nd.id) },
      };
    },
  });

  /**
   * A PAGE of a list, for the tools that can return a long one. `limit` is 0 or absent by
   * default, and THAT is the point: the default does not truncate, so nothing that called
   * these tools before sees a different answer. `cursor` is a plain offset — deterministic,
   * and it survives the list changing under it because the caller also gets the total.
   */
  function paginate<T>(items: T[], limit: unknown, cursor: unknown): { page: T[]; total: number; next: number | null } {
    const total = items.length;
    const start = Math.max(0, Math.floor(Number(cursor) || 0));
    const lim = Math.max(0, Math.floor(Number(limit) || 0));
    if (lim === 0) return { page: items.slice(start), total, next: null };
    const page = items.slice(start, start + lim);
    const next = start + page.length < total ? start + page.length : null;
    return { page, total, next };
  }

  pi.registerTool({
    name: 'cwl_node',
    label: 'CWL Node',
    description: t('consult').nodeDesc,
    promptSnippet: t('consult').nodeSnippet,
    parameters: Type.Object({
      id: Type.String({ description: t('consult').nodeId }),
      limit: Type.Optional(Type.Number({ description: t('consult').nodeLimit })),
      cursor: Type.Optional(Type.Number({ description: t('consult').nodeCursor })),
    }),
    async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
      const key = sessionKey(ctx);
      const st = getState(key);
      const cf = getConfig(key);
      const views = indexNodeViews(st);
      const nd = views.find((v) => v.id === params.id);
      if (!nd) {
        return { content: [{ type: 'text', text: t('consult').nodeNotFound(params.id) }], details: { ok: false, error: 'node-not-found', id: params.id } };
      }
      const leaves = leafViews(st);
      const lines = [t('consult').nodeHeader(nd.id, nd.kind, nd.name ?? '', nd.leaves.length, nd.chars)];
      if (nd.description) lines.push(nd.description);
      if (nd.children.length > 0) lines.push(t('consult').nodeChild(nd.children.join(' ')));
      const page = paginate(nd.leaves, params.limit, params.cursor);
      for (const id of page.page) {
        const v = leaves.get(id);
        lines.push(t('consult').nodeLeaf(id, v?.micro ? firstLine(v.micro) : '(no label yet)'));
      }
      if (page.next !== null) lines.push(t('consult').pageInfo(page.page.length, page.total, String(page.next)));
      debugLog(cf, `NODE ${nd.id} kind=${nd.kind} leaves=${nd.leaves.length} chars=${nd.chars}`);
      return {
        content: [{ type: 'text', text: lines.join('\n') }],
        details: {
          ok: true, id: nd.id, kind: nd.kind, leaves: nd.leaves.length, chars: nd.chars,
          leafIds: nd.leaves, children: nd.children,
          shownIds: page.page, shown: page.page.length, total: page.total, nextCursor: page.next,
        },
      };
    },
  });

  pi.registerTool({
    name: 'cwl_pending',
    label: 'CWL Pending',
    description: t('consult').pendingDesc,
    promptSnippet: t('consult').pendingSnippet,
    parameters: Type.Object({
      limit: Type.Optional(Type.Number({ description: t('consult').pendingLimit })),
      cursor: Type.Optional(Type.Number({ description: t('consult').pendingCursor })),
    }),
    async execute(_toolCallId, params, _signal, _onUpdate, ctx) {
      const key = sessionKey(ctx);
      const st = getState(key);
      const cf = getConfig(key);
      // ONLY what still has to be ordered: the nodes with no description (their micros
      // are injected as they are) plus the buffer. A topic is already ordered, and the
      // pit is the archive.
      const open = indexNodeViews(st).filter((nd) => nd.kind !== 'pit' && !nd.description);
      const need = Math.max(Math.round(cf.mergeMinRatio * MERGE_SYNTHESIS_CHARS), cf.mergeMinChars);
      const totalLeaves = open.reduce((n, nd) => n + nd.leaves.length, 0);
      const totalChars = open.reduce((n, nd) => n + nd.chars, 0);
      // #3 — WHERE THE LEAVES ARE, because it decides WHICH operation to use and the two are
      // not interchangeable. The list below is the FRONTIER: `cwl_group` without `pit` takes
      // those leaves and BORNS a topic, which moves the index and pays the size guard. What
      // this section names is the PIT: leaves already archived and still unnamed, which
      // `pit: true` catalogues WITHOUT moving anything and WITHOUT a size guard. Saying which
      // is which is the whole point: an agent that does not know where a leaf sits spends a
      // call finding out, and the refusal it gets back does not say what to do instead.
      const pitRoots = st.oldNode?.nodes ?? [];
      const pitAll = new Set<string>([...pitRoots, ...(pitRoots.length > 0 ? containedNodes(st, pitRoots) : [])]);
      const pitOrdinary = st.nodes.filter((nd) => pitAll.has(nd.id) && !nd.description);
      const pitLeaves = pitOrdinary.reduce((n, nd) => n + nd.leaves.length, 0);
      const pitLines: string[] = pitOrdinary.length > 0
        ? [
            t('consult').pendingPitHeader(pitOrdinary.length, pitLeaves),
            ...pitOrdinary.map((nd) => t('consult').pendingPitLine(nd.id, nd.leaves.length)),
            t('consult').pendingPitHint,
          ]
        : [];
      if (open.length === 0) {
        return {
          content: [{ type: 'text', text: [t('consult').pendingEmpty, ...pitLines].join('\n') }],
          details: { ok: true, nodes: 0, leaves: 0, chars: 0, need, pit: pitOrdinary.map((nd) => nd.id) },
        };
      }
      const lines = [t('consult').pendingHeader(open.length, totalLeaves, totalChars.toLocaleString(), need.toLocaleString())];
      // #4 — the page. `limit` absent means ALL of it, so the answer is what it always was.
      const page = paginate(open, params.limit, params.cursor);
      for (const nd of page.page) {
        lines.push(t('consult').pendingLine(nd.id, nd.kind, nd.leaves.length, nd.chars));
        lines.push(t('consult').mapLeaves(nd.leaves.join(' ')));
      }
      if (page.next !== null) lines.push(t('consult').pageInfo(page.page.length, page.total, String(page.next)));
      lines.push(...pitLines);
      debugLog(cf, `PENDING nodes=${open.length} leaves=${totalLeaves} chars=${totalChars} need=${need}`);
      return {
        content: [{ type: 'text', text: lines.join('\n') }],
        details: {
          ok: true, nodes: open.length, leaves: totalLeaves, chars: totalChars, need,
          enough: totalChars >= need, ids: open.map((nd) => nd.id),
          total: page.total, shown: page.page.length, nextCursor: page.next,
          shownIds: page.page.map((nd) => nd.id),
          pit: pitOrdinary.map((nd) => nd.id),
        },
      };
    },
  });

  // ---- Slash commands ----------------------------------------------------
  // The same two acts the tools expose, but in the OPERATOR's hands: forking a
  // memory is a decision about which past to continue, and it belongs to whoever
  // is at the keyboard, not to the model.

  pi.registerCommand('cwl_memories', {
    description: t('cmdMemoriesDesc')(),
    handler: async (_args, ctx) => {
      if (!ctx.hasUI) return;
      const found = listMemories();
      if (found.length === 0) {
        ctx.ui.notify(t('memoriesEmpty')(), 'warning');
        return;
      }
      const rows = found.map((m) => t('memoriesRow')(m.display, m.leaves, m.nodes, m.pit)).join('\n');
      ctx.ui.notify(t('memoriesList')(rows), 'info');
    },
  });

  pi.registerCommand('cwl_adopt', {
    description: t('cmdAdoptDesc')(),
    handler: async (_args, ctx) => {
      if (!ctx.hasUI) return;
      const key = sessionKey(ctx);
      const st = getState(key);
      const cf = getConfig(key);
      // A LIVE memory is not offered: forking it would give two writers to one past.
      const adoptable = listMemories().filter((m) => !m.alive);
      if (adoptable.length === 0) {
        ctx.ui.notify(t('cmdNoAdoptable')(), 'warning');
        return;
      }
      const choice = await ctx.ui.select(t('cmdPickSource')(), adoptable.map((m) => m.name));
      if (choice === undefined) {
        ctx.ui.notify(t('cmdCancelled')(), 'info');
        return;
      }
      const entered = await ctx.ui.input(t('cmdPickName')(), `${choice}--fork`);
      if (entered === undefined) {
        ctx.ui.notify(t('cmdCancelled')(), 'info');
        return;
      }
      const res = performAdopt(cf, st, key, choice, entered.trim() || undefined);
      ctx.ui.notify(res.text, res.ok ? 'info' : 'error');
    },
  });

  const saveHandler = async (args: string | undefined, ctx: ExtensionContext) => {
    const key = sessionKey(ctx);
    const st = getState(key);
    const sm = ctx?.sessionManager as
      | { buildContextEntries?: () => Array<{ type?: string; message?: AgentMessage }> }
      | undefined;
    const messages = typeof sm?.buildContextEntries === 'function'
      ? sm.buildContextEntries()
          .filter((e) => !!e && e.type === 'message' && !!e.message)
          .map((e) => e.message as AgentMessage)
      : [];
    const checkRange = messages.length > 0 ? compressibleRange(messages, st.spans, 0) : null;
    if (!checkRange) {
      if (ctx.hasUI) ctx.ui.notify(t('cmdSaveNothing')(), 'warning');
      return;
    }
    st.forceAllNext = true;
    st.forceAllNote = typeof args === 'string' && args.trim() ? args.trim() : undefined;
    saveState(key, st);
    if (ctx.hasUI) {
      ctx.ui.notify(t('cmdSaveTriggered')(checkRange.tokens), 'info');
    }
    const promptText = t('cmdSavePrompt')(st.forceAllNote);
    const piSender = pi as unknown as { sendUserMessage?: (msg: string) => Promise<unknown> | void };
    if (typeof piSender.sendUserMessage === 'function') {
      await piSender.sendUserMessage(promptText);
    }
  };

  pi.registerCommand('cwl_save', {
    description: t('cmdSaveDesc')(),
    handler: saveHandler,
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
