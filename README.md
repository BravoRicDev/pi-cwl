# CWL (Context Window Lifecycle)

Structured context eviction for long-horizon Pi agents, based on arXiv:2606.11213.

## Panoramica
La compattazione tradizionale del contesto tramite summarization a soglia fissa introduce tre problemi critici per gli agenti autonomi:
1. **Perdita di informazione critica** (dettagli di codice, errori di compilazione, specifiche).
2. **Allucinazioni indotte dalla sintesi** (il modello riassume male il proprio passato).
3. **Interruzioni di metà turno** (blocchi bloccanti mentre l'agente sta eseguendo una catena di task).

**CWL** risolve questo problema trattando la sessione come un grafo di episodi tipizzati (`expl` e `act`) delimitati dall'agente tramite il tool `delimiter`. Quando il budget di token viene superato, una politica deterministica e LLM-free applica un'eviction graduata basata sulla recuperabilità degli episodi e sulle dipendenze causali.

---

## Principi Architetturali
- **Zero LLM Overhead per l'Eviction:** Nessuna chiamata al modello per comprimere; la compressione è puramente algoritmica e basata su metriche strutturali.
- **Inviolabilità dell'Utente:** I turni dell'utente e del system prompt non vengono mai toccati (Principio 3).
- **Separazione Episodica:**
  - `expl` (Esplorazione): Ricerche, listing, letture di orientamento. Alla chiusura, l'agente fornisce una descrizione concisa: **questo è l'unico contenuto che sopravvive all'eviction**.
  - `act` (Azione): Scritture, edit, esecuzioni di comandi. Gli effetti sono persistenti nell'ambiente, quindi sono i primi candidati all'eviction graduata (stripping di reasoning, bulk output, rimozione completa).
- **Dipendenze Causali:** Un episodio esplorativo non può essere evictato finché un'azione che dipende da esso è ancora attiva nel contesto.

---

## Crediti e Riconoscimenti
Questo progetto è un'implementazione indipendente e originale di:
- **Paper:** "Beyond Compaction: Structured Context Eviction for Long-Horizon Agents" (arXiv:2606.11213)
- **Autori del Paper:** Andrew Semenov, Svyatoslav Dorofeev (2026)

L'architettura e il concetto originale di CWL appartengono agli autori del paper. Questa estensione è stata progettata e scritta da zero in TypeScript per integrarsi nativamente con l'ecosistema Pi Agent. Non contiene codice di terze parti o derivato da repository degli autori.

Per citare formalmente questo lavoro, fai riferimento al file `CITATION.cff`.

---

## Installazione e Configurazione
L'estensione viene caricata automaticamente da Pi tramite symlink in `~/.pi/agent/extensions/pi-cwl`.

### Configurazione (`~/.pi/cwl/config.json`)
```json
{
  "tokenBudget": 80000,
  "thresholdRatio": 0.85,
  "levels": {
    "stripReasoning": true,
    "stripBulkOutput": true,
    "stripIntermediate": true,
    "removeEpisode": true
  },
  "showWidget": true,
  "debug": false
}
```

---

## Tool Disponibili
1. **`delimiter`**: Segna l'inizio o la fine di un episodio.
   - `action`: `"start"` | `"end"`
   - `name`: Nome univoco dell'episodio.
   - `type`: `"expl"` | `"act"` (per `start`)
   - `dependencies`: Array di nomi di episodi `expl` da cui un `act` dipende.
   - `description`: Sintesi dell'apprendimento (obbligatorio alla chiusura di un `expl`).
2. **`cwl_status`**: Mostra lo stato corrente del grafo, i token stimati, gli episodi attivi e le eviction eseguite.
