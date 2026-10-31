"""
Phase 4 — deterministic safety net for the LLM router.

Why this exists: backend/router_sketch.py asks a small local LLM (llama3:8b)
to classify each question as sql / rag / both. In evaluation it sometimes
sent plain document questions ("...based on the return policy?") to SQL
only, so the RAG tool never ran and the user got a wrong answer — while the
same question asked directly of the RAG pipeline was answered correctly.

This module doesn't replace the LLM. It checks the question for EXPLICIT
signals that it refers to documents, and if the LLM's route would leave the
documents out, it adds them. Design rules that keep it safe:

  1. It is additive: it adds document retrieval (sql -> both, rag -> both)
     and never changes "both". The worst a false positive costs is extra
     latency, never a lost answer (the synthesizer already copes with an
     irrelevant source).
  2. ONE deliberate exception: sql/unknown -> rag (SQL dropped) happens only
     when the question names a document explicitly (a word like policy /
     handbook / FAQ, or an indexed document's name) AND contains no database
     vocabulary whatsoever. Then there is nothing for SQL to answer, and
     running it just adds ~20s and an unrelated generated query. Looser
     phrasing ("what happens if...") never triggers this; it gets "both".
  3. Cue words that are really DATA words in the active database are
     ignored. If the connected database has a `policies` table, then
     "How many policies were sold?" is a data question, not a document one.
     The same applies to uploaded-document names that match table names.

It is pure Python with no dependencies, so every rule is covered by plain
unit tests (tests/test_router_rules.py) — no LLM or embeddings needed.
"""
import re

# Words that, on their own, mean "this is about a document".
_DOC_WORD_RE = re.compile(
    r"\b(policy|policies|handbook|handbooks|faq|faqs|manual|manuals|guideline|guidelines|"
    r"procedure|procedures|document|documents|documentation|terms and conditions)\b",
    re.IGNORECASE,
)

# Phrasings typical of policy / procedure questions even without a doc word.
_DOC_PHRASE_RE = re.compile(
    r"\b(according to|as per|per the|as stated in|as described in|as outlined in)\b"
    r"|\bwhat happens (if|when)\b"
    r"|\b(am i|are we|are they|is it) (allowed|permitted|eligible)\b"
    r"|\bcan (an?|the) (employee|staff|customer|user|manager)\b"
    r"|\bnon-?refundable\b|\brefundable\b|\beligible for\b"
    r"|\bwho (do|should|can) (i|we) (contact|ask|call|escalate)\b|\bescalat(e|ion)\b"
    r"|\bhow long do (i|we|customers?) have\b",
    re.IGNORECASE,
)

# Aggregate / ranking language: the question wants numbers from rows.
# "how many days/weeks/..." is deliberately NOT a data cue: it asks about a
# duration in a policy ("how many days until a refund?"), not a row count.
_DATA_STRONG_RE = re.compile(
    r"\b(how many (?!days|weeks|months|years|hours|minutes|business)"
    r"|number of|count of|total|average|avg|sum of|highest|lowest|maximum|minimum"
    r"|top \d+|top|most|least|how much|list all)\b",
    re.IGNORECASE,
)

_UUID_SUFFIX_RE = re.compile(r"-[0-9a-f]{8}$")


def stem(word: str) -> str:
    """Tiny plural-insensitive normalizer applied to BOTH sides of every
    comparison, so exact linguistic correctness doesn't matter — only
    consistency does (policies/policy, genres/genre, invoices/invoice)."""
    w = re.sub(r"[^a-z0-9]", "", word.lower())
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("sses"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _tokens(text: str) -> list:
    return [stem(t) for t in re.findall(r"[A-Za-z0-9]+", text)]


def _doc_name_phrases(doc_ids, schema_stems: set) -> list:
    """Each indexed document's readable name as a list of stems, e.g.
    'returns_policy-a453105b' -> ['return', 'policy']. Names made entirely
    of database vocabulary, and very short single words, are skipped so a
    PDF called 'customers.pdf' can't hijack every question about customers."""
    phrases = []
    for doc_id in doc_ids or []:
        base = _UUID_SUFFIX_RE.sub("", str(doc_id))
        words = [stem(w) for w in re.split(r"[-_\s]+", base) if w]
        if not words:
            continue
        if all(w in schema_stems for w in words):
            continue
        if len(words) == 1 and len(words[0]) < 5:
            continue
        phrases.append(words)
    return phrases


def _contains_sequence(tokens: list, phrase: list) -> bool:
    n = len(phrase)
    return any(tokens[i:i + n] == phrase for i in range(len(tokens) - n + 1))


def analyze(question: str, doc_ids=None, schema_terms=None) -> dict:
    """Extract routing cues. Returns:
      doc_cue:  the matched text if the question refers to documents, else None
      doc_cue_explicit: True if a document is named outright (policy word or
                an indexed document's name), False for loose phrasing
      data_strong: question asks for counts / rankings / aggregates
      data_weak:   question mentions an entity of the active database
    """
    schema_stems = {stem(t) for t in (schema_terms or [])}
    tokens = _tokens(question)

    doc_cue = None
    explicit = False  # True when a document is named outright (vs. policy-style phrasing)
    for m in _DOC_WORD_RE.finditer(question):
        if stem(m.group(1)) not in schema_stems:  # 'policies' table => not a doc cue
            doc_cue, explicit = m.group(1), True
            break
    if doc_cue is None:
        for phrase in _doc_name_phrases(doc_ids, schema_stems):
            if _contains_sequence(tokens, phrase):
                doc_cue, explicit = " ".join(phrase), True
                break
    if doc_cue is None:
        m = _DOC_PHRASE_RE.search(question)
        if m:
            doc_cue = m.group(0)

    return {
        "doc_cue": doc_cue,
        "doc_cue_explicit": explicit,
        "data_strong": bool(_DATA_STRONG_RE.search(question)),
        "data_weak": any(t in schema_stems for t in tokens),
    }


def refine_route(llm_route: str, question: str, doc_ids=None, schema_terms=None):
    """Return (final_route, note). `note` is None when the LLM's route is
    kept, otherwise a human-readable explanation of the adjustment."""
    cues = analyze(question, doc_ids, schema_terms)
    if cues["doc_cue"] is None:
        return llm_route, None
    if llm_route == "both":
        return llm_route, None  # already includes documents

    if cues["data_strong"]:
        new_route = "both"
    elif llm_route in ("sql", "unknown") and (cues["data_weak"] or not cues["doc_cue_explicit"]):
        new_route = "both"  # data vocabulary present, or only loose phrasing: keep SQL, add documents
    else:
        new_route = "rag"   # explicit document reference, zero database vocabulary

    if new_route == llm_route:
        return llm_route, None
    note = (
        f"Router adjusted '{llm_route}' to '{new_route}': the question explicitly "
        f"refers to documents ({cues['doc_cue']!r})."
    )
    return new_route, note