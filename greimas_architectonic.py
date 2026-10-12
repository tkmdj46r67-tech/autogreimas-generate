#!/usr/bin/env python3
"""
Architectonic Greimas Generator — Version 10

Changes from V9:
  * Cycle 1 default model changed from phi4-mini to qwen2.5:7b.

Usage:
    python greimas_architectonic.py
    python greimas_architectonic.py --s1 being
    python greimas_architectonic.py --s1 being --s2 seeming   # skips cycle 1
    python greimas_architectonic.py --no-cycle1 --s1 being
    python greimas_architectonic.py --cycle1-model llama3.2:3b

Requirements:
    pip install ollama
    Running Ollama instance (local or remote).
"""

import sys
import json
import re
import time
import argparse
from datetime import datetime
from pathlib import Path

try:
    import ollama
except ImportError:
    print("Install with: python3 -m pip install ollama")
    sys.exit(1)

# =========================================================
# CONFIG
# =========================================================

DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_CYCLE1_MODEL = "qwen2.5:7b"
DEFAULT_VALIDATOR_MODEL = "phi4-mini"
FAST_MODE_MODEL = "phi4-mini"

TEMPERATURE = 0.3
CYCLE1_TEMPERATURE = 0.9
NUM_PREDICT = 1200
MAX_RETRIES = 3
MAX_UNIQUENESS_TRIES = 4
CYCLE1_N_CANDIDATES = 5
CYCLE1_N_REQUEST = 10

EXIT_WORDS = {"q", "quit", "exit"}

def _is_exit(text):
    return text.strip().lower() in EXIT_WORDS

def _exit_cleanly(msg="Exiting."):
    print(msg)
    sys.exit(0)

# =========================================================
# MODEL MENUS
# =========================================================

MODEL_MENUS = {
    "production": [
        ("phi4:14b", "Phi-4 (14B) — larger Phi; strong abstract reasoning; partial GPU offload"),
        ("qwen2.5:14b", "Qwen 2.5 (14B) — strong structured output; partial GPU offload"),
        ("mistral-nemo:12b", "Mistral Nemo (12B) — strong instruction follower; partial offload"),
        ("llama3.1:8b", "Llama 3.1 (8B) — best all-around 8B; full GPU offload on RX 580 via Vulkan"),
        ("qwen2.5:7b", "Qwen 2.5 (7B) — fast, good structured generation; full GPU"),
        ("mistral:7b", "Mistral (7B) — reliable classic; full GPU"),
        ("CUSTOM", "Enter a custom model name"),
    ],
    "iteration": [
        ("qwen2.5:7b", "Qwen 2.5 (7B) — good speed/quality balance for iteration"),
        ("llama3.1:8b", "Llama 3.1 (8B) — strong general purpose"),
        ("phi4-mini", "Phi-4 Mini (3.8B) — fast; good structured reasoning"),
        ("llama3.2:3b", "Llama 3.2 (3B) — very fast; lower quality"),
        ("CUSTOM", "Enter a custom model name"),
    ],
}

TIER_ALIASES = {
    "1": "production", "production": "production", "prod": "production", "p": "production",
    "2": "iteration",  "iteration": "iteration",  "iter": "iteration",  "i": "iteration",
}

# =========================================================
# GLOBAL STATE
# =========================================================

MODEL = DEFAULT_MODEL
VALIDATOR_MODEL = DEFAULT_VALIDATOR_MODEL
COLLISION_LOG = []
USED_TERMS = {}
REGISTRATION_ORDER = []

# =========================================================
# OLLAMA HELPERS
# =========================================================

def list_ollama_models():
    try:
        result = ollama.list()
        models = result.get("models", [])
        names = []
        for m in models:
            n = m.get("name") or m.get("model")
            if n:
                names.append(n)
        return sorted(names)
    except Exception as e:
        print(f"Could not list Ollama models: {e}")
        return []

def find_present_variant(name, available):
    if not available:
        return None
    if name in available:
        return name
    base = name.split(":")[0]
    matches = [m for m in available if m.split(":")[0] == base]
    return matches[0] if matches else None

# =========================================================
# TERM REGISTRY
# =========================================================

def normalize(term):
    if not term:
        return ""
    return re.sub(r'[^a-z0-9]+', '', term.lower())

def register_term(term, location, role):
    norm = normalize(term)
    if not norm:
        return
    entries = USED_TERMS.setdefault(norm, [])
    for e in entries:
        if e[1] == location and e[2] == role:
            return
    entries.append((term, location, role))
    if len(entries) == 1:
        REGISTRATION_ORDER.append((norm, term))

def is_collision(term, exclude_locations=None):
    norm = normalize(term)
    if not norm:
        return True
    if norm in USED_TERMS:
        entries = USED_TERMS[norm]
        if exclude_locations is None:
            return True
        if any(e[1] not in exclude_locations for e in entries):
            return True
    return False

def all_exclusion_terms(exclude_norms=None, extra_terms=None):
    exclude_norms = exclude_norms or set()
    extra_terms = set(extra_terms or ())
    seen = set()
    out = []
    for term in extra_terms:
        n = normalize(term)
        if n and n not in exclude_norms and n not in seen:
            seen.add(n)
            out.append(term)
    for norm, surface in REGISTRATION_ORDER:
        if norm in exclude_norms or norm in seen:
            continue
        seen.add(norm)
        out.append(surface)
    return out

def exclusion_block(exclude_norms=None, extra_terms=None):
    terms = all_exclusion_terms(exclude_norms=exclude_norms, extra_terms=extra_terms)
    if not terms:
        return ""
    if len(terms) > 40:
        terms = terms[-40:]
    return (
        "UNIQUENESS CONSTRAINT: Every term you output must be a NEW word or short "
        "phrase that does not already exist anywhere in the architectonic. "
        "The following terms have already been used and MUST NOT be reused "
        "(as-is or as close variants): " + ", ".join(terms) + "."
    )

# =========================================================
# LLM HELPERS
# =========================================================

def extract_json_object(text):
    if not text:
        return ""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    if start == -1:
        return text
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]

def llm_json(prompt, required_keys, label, retries=MAX_RETRIES,
             model_override=None, temperature_override=None):
    active_model = model_override or MODEL
    active_temp = (temperature_override if temperature_override is not None
                   else TEMPERATURE)
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = ollama.chat(
                model=active_model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": active_temp, "num_predict": NUM_PREDICT},
            )
            raw = resp["message"]["content"]
            data = json.loads(extract_json_object(raw))
            missing = [k for k in required_keys if k not in data]
            if missing:
                raise ValueError(f"missing keys: {missing}")
            return data
        except Exception as e:
            last_err = str(e)
            print(f"  retry {attempt}/{retries} ({label}): {last_err}")
            time.sleep(1)
    raise RuntimeError(f"LLM failed after {retries} retries ({label}): {last_err}")

# =========================================================
# STRUCTURAL VALIDATION (cheap, no LLM)
# =========================================================

def structural_check_square(s1, s2, ns2, ns1):
    terms = [s1, s2, ns2, ns1]
    for t in terms:
        if not t or not t.strip():
            return False, "empty term"
    for t in terms:
        if len(t.split()) > 4:
            return False, f"term too long: '{t}'"
        if len(t) > 40:
            return False, f"term too long: '{t}'"
    norms = [normalize(t) for t in terms]
    if len(set(norms)) != 4:
        return False, "terms are not all distinct"
    ns2_norm = normalize(ns2)
    if ns2_norm.startswith("non") and ns2_norm[3:] == normalize(s2):
        return False, f"~S2 is a literal negation of S2"
    ns1_norm = normalize(ns1)
    if ns1_norm.startswith("non") and ns1_norm[3:] == normalize(s1):
        return False, f"~S1 is a literal negation of S1"
    return True, "ok"

# =========================================================
# CYCLE 1 — FULL-SQUARE SKELETON GENERATION
# =========================================================

PROMPT_CYCLE1_SQUARE = """You are given the first pole of a Greimas square:

S1 = "{s1}"

Context: {context_note}

Propose exactly {n} candidate Greimas squares. Each candidate is a complete
inner square: (S2, ~S2, ~S1). S1 is fixed.

Structural rules — every candidate must satisfy ALL of them:
- S2 is the CONTRARY of S1: a positive term, the same kind of thing as S1.
  Do NOT use a negation of S1 (that would be ~S1, a different slot).
- ~S2 is the CONTRADICTORY of S2: exactly one of S2 and ~S2 holds for any
  subject. Prefer a semantic term; fall back to "non-X"/"un-X" only if no
  better word exists.
- ~S1 is the CONTRADICTORY of S1, AND it is the ANTITHESIS of ~S2.
  Concretely: ~S1 is to ~S2 what S2 is to S1 (a contrary relation). ~S1 is
  NOT a literal negation of S1.
- All four terms (S1, S2, ~S2, ~S1) must be pairwise distinct, different
  words, not synonyms of each other.
- Every term: 1-3 words, lowercase.

Diversity across the {n} candidates: explore DIFFERENT senses of opposition.
Vary the register (abstract / concrete / experiential) and the lexical
family of S2. Do not produce near-duplicates.

Return JSON with exactly this key:
{{"squares": [
  {{"S2": "...", "~S2": "...", "~S1": "..."}},
  ...
]}} with exactly {n} entries.
"""

CONTEXT_NOTES = {
    "base": "S1 is the first pole of the top-level (Layer 1) Greimas square. S2 will be its contrary, the second positive pole of the same square.",
    "internal": "S1 is the first internal pole of a subordinate Greimas square that will decompose a parent concept. S2 will be the second internal pole, on the same axis.",
}

def cycle1_generate_squares(s1, seed, n=CYCLE1_N_REQUEST, cycle1_model=None,
                            context=None, extra_note=None):
    if cycle1_model is None:
        cycle1_model = DEFAULT_CYCLE1_MODEL
    if context is None:
        context = "base" if seed == "A1" else "internal"
    note = CONTEXT_NOTES.get(context, CONTEXT_NOTES["base"])
    if extra_note:
        note = note + " " + extra_note
    prompt = PROMPT_CYCLE1_SQUARE.format(s1=s1, n=n, context_note=note)
    try:
        data = llm_json(
            prompt, ["squares"], "cycle1",
            model_override=cycle1_model,
            temperature_override=CYCLE1_TEMPERATURE,
        )
        raw = data.get("squares", [])
        if not isinstance(raw, list):
            raw = []
        out = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            s2 = str(item.get("S2", "")).strip()
            ns2 = str(item.get("~S2", "")).strip()
            ns1 = str(item.get("~S1", "")).strip()
            if not s2 or not ns2 or not ns1:
                continue
            out.append({"S2": s2, "~S2": ns2, "~S1": ns1})
        return out
    except Exception as e:
        print(f"  cycle 1 LLM failed: {e}")
        return []

def cycle1_collect_valid(s1, seed, cycle1_model, n_keep=CYCLE1_N_CANDIDATES,
                         max_attempts=2):
    collected = []
    seen_sigs = set()
    last_note = None
    for attempt in range(1, max_attempts + 1):
        raw = cycle1_generate_squares(
            s1, seed, n=CYCLE1_N_REQUEST, cycle1_model=cycle1_model,
            extra_note=last_note)
        for cand in raw:
            sig = (normalize(cand["S2"]), normalize(cand["~S2"]), normalize(cand["~S1"]))
            if sig in seen_sigs:
                continue
            ok, _reason = structural_check_square(s1, cand["S2"], cand["~S2"], cand["~S1"])
            if not ok:
                continue
            seen_sigs.add(sig)
            collected.append(cand)
            if len(collected) >= n_keep:
                return collected[:n_keep]
        if attempt < max_attempts:
            last_note = ("Previously proposed candidates failed structural checks "
                         "(non-distinct terms, literal negations, or too long). "
                         "Please propose completely fresh candidates with these "
                         "issues corrected.")
    return collected[:n_keep]

# ---------------------------------------------------------
# Override string parser
# ---------------------------------------------------------

def parse_override_string(s):
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        return None, f"expected 3 comma-separated terms, got {len(parts)}"
    s2, ns2, ns1 = parts
    if not s2 or not ns2 or not ns1:
        return None, "all three terms must be non-empty"
    if len(s2) > 40 or len(ns2) > 40 or len(ns1) > 40:
        return None, "one or more terms are too long"
    return {"S2": s2, "~S2": ns2, "~S1": ns1}, None

# ---------------------------------------------------------
# Cycle 1 interactive runner
# ---------------------------------------------------------

def cycle1_run(s1, seed, cycle1_model):
    print()
    print("=" * 64)
    print("Cycle 1 — proposing candidate squares")
    print(f"  S1:    '{s1}'")
    print(f"  Model: {cycle1_model} (lightweight, for divergence)")
    print("=" * 64)

    candidates = cycle1_collect_valid(s1, seed, cycle1_model)

    while True:
        print()
        if candidates:
            for i, c in enumerate(candidates, 1):
                print(f"  [{i}] S2 = {c['S2']:<22s}  ~S2 = {c['~S2']:<22s}  ~S1 = {c['~S1']}")
        else:
            print("  (no structurally valid candidates at this time)")

        print("  [s] specify S2 only; let the generator fill in ~S2 and ~S1")
        print("  [o] override with custom string, format: S2, ~S2, ~S1")
        print("  [r] regenerate candidates")
        print("  [b] back to S1 entry")
        print("  [q] exit")

        default_choice = "1" if candidates else "r"
        sel = input(f"Select [1-{len(candidates)}/s/o/r/b/q] (default: {default_choice}): ").strip().lower()

        if _is_exit(sel):
            _exit_cleanly("Exiting.")
        if not sel:
            sel = default_choice
        if sel == "b":
            return ("back", None)
        if sel == "r":
            candidates = cycle1_collect_valid(s1, seed, cycle1_model)
            continue
        if sel == "s":
            print()
            print("  Enter S2 only. The generator model will produce ~S2 and ~S1.")
            s2 = input("  S2 (or [q] to exit): ").strip()
            if _is_exit(s2):
                _exit_cleanly("Exiting.")
            if not s2:
                print("  Empty S2; try again.")
                continue
            if len(s2) > 40 or len(s2.split()) > 4:
                print(f"  S2 too long or too wordy; try again.")
                continue
            if normalize(s2) == normalize(s1):
                print(f"  S2 cannot be identical to S1.")
                continue
            return ("s2_only", {"S2": s2})
        if sel == "o":
            print()
            print("  Enter three comma-separated terms in the order: S2, ~S2, ~S1")
            print("  Example: servitude, non-servitude, non-freedom")
            raw = input("  > ").strip()
            if _is_exit(raw):
                _exit_cleanly("Exiting.")
            parsed, err = parse_override_string(raw)
            if err:
                print(f"  Parse error: {err}. Try again.")
                continue
            ok, reason = structural_check_square(s1, parsed["S2"], parsed["~S2"], parsed["~S1"])
            if not ok:
                print(f"  Structural problem: {reason}.")
                print(f"  You can try again, or type 'f' to force-accept.")
                conf = input("  Retry (any key) or force [f]? ").strip().lower()
                if conf != "f":
                    continue
            return ("ok", parsed)
        try:
            idx = int(sel) - 1
            if 0 <= idx < len(candidates):
                return ("ok", candidates[idx])
        except ValueError:
            pass
        print("  Invalid selection.")

# =========================================================
# PROMPTS — REVERSE FIBRATION
# =========================================================

PROMPT_REVERSE_FIBRATION = """Given two terms that are the two positive internal poles of some concept:
- Internal S1' = "{x}"
- Internal S2' = "{y}"

These two terms are the two positive contraries into which a parent concept internally polarizes. For example:
- Internal poles "substance" and "appearance" → parent concept = "being"
- Internal poles "existence" and "vitality" → parent concept = "life"
- Internal poles "negative freedom" and "positive freedom" → parent concept = "freedom"

Propose the parent concept for the pair "{x}" and "{y}".

Requirements:
- Single word or short phrase (1-3 words).
- A genuine unifying concept: both "{x}" and "{y}" are aspects of it.
- Must NOT be a synonym of "{x}" or "{y}".
- Must be a positive term, not a negation.
- {exclusion_note}

Return JSON with exactly: {{"parent": "..."}}
"""

PROMPT_CHECK_UNIFIER = """Is "{parent}" a genuine unifying concept for the pair ("{x}", "{y}")?

Meaning: are "{x}" and "{y}" both genuine internal aspects of "{parent}"?

Answer "yes" or "no" with one sentence.

Return JSON with exactly: {{"pass": true/false, "explanation": "..."}}
"""

PROMPT_CONTINUE_A_FROM_S1 = """A Greimas square A is being constructed around a seme.

- A's S1 = "{s1}"

Generate A's S2, the contrary of S1 (the second positive pole of A).

Rules:
- S2 must be a positive term, not a negation.
- S1 and S2 must be contraries: both can hold together, both can fail together, one can hold alone.
- S2 must be the same kind of thing as S1.
- Must not be a synonym of S1 or of the internal poles.
- {exclusion_note}

Return JSON with exactly: {{"S2": "..."}}
"""

# =========================================================
# PROMPTS — LAYER 1
# =========================================================

PROMPT_L1_NEG_S2 = """You are given S2 = "{s2}" (the antithesis of S1 = "{s1}").

Generate ~S2, the CONTRADICTORY of S2.

Rules:
- ~S2 and S2 are contradictories: exactly one holds for any subject.
- Simple negation of S2 ("non-X" / "un-X") if no better term exists.
- Single word or short phrase (1-2 words).
- Must not be a synonym of S2 or S1.
- {exclusion_note}

Return JSON with exactly: {{"~S2": "..."}}
"""

PROMPT_L1_ANTITHESIS = """You are given:
- S1 = "{s1}"
- S2 = "{s2}"
- ~S2 = "{neg_s2}"

Generate ~S1, the CONTRADICTORY of S1.

CRITICAL: ~S1 is NOT a simple negation of S1. It is the ANTITHESIS of ~S2.
Just as S2 is a contrary of S1, ~S1 is a contrary of ~S2.

Derivation chain:
S1 --(antithesis)--> S2 --(negation)--> ~S2 --(antithesis)--> ~S1

Consequences:
- ~S1 and S1 are contradictories.
- ~S1 and ~S2 are contraries.
- ~S1 implies S2.

Rules:
- Single word or short phrase (1-2 words).
- Must not be a synonym of S1, S2, or ~S2.
- {exclusion_note}

Return JSON with exactly: {{"~S1": "..."}}
"""

PROMPT_L1_M1 = """Greimas square inner semes:
- S1 = "{s1}", S2 = "{s2}", ~S2 = "{neg_s2}", ~S1 = "{neg_s1}"

Generate the four M1 meta-terms. Each M1 is the union of two adjacent inner semes:

M11 (complex axis)  = S1 + S2   = "{s1}" + "{s2}"
M12 (neutral axis)  = ~S1 + ~S2 = "{neg_s1}" + "{neg_s2}"
M13 (positive deixis) = S1 + ~S2 = "{s1}" + "{neg_s2}"
M14 (negative deixis) = S2 + ~S1 = "{s2}" + "{neg_s1}"

Rules:
- Each: single word or short phrase (1-2 words).
- Each: a positive, evocative unifying concept — NOT a concatenation.
- No duplicates or near-duplicates.
- {exclusion_note}

Return JSON with exactly these keys:
{{"M11": "...", "M12": "...", "M13": "...", "M14": "..."}}
"""

PROMPT_L1_M2 = """Greimas square inner semes:
- S1 = "{s1}", S2 = "{s2}", ~S2 = "{neg_s2}", ~S1 = "{neg_s1}"

Generate the four M2 meta-terms. Each contains exactly THREE of the four semes:

M21 = S1 + S2 + ~S2  (omits ~S1)
M22 = S1 + S2 + ~S1  (omits ~S2)
M23 = S1 + ~S1 + ~S2 (omits S2)
M24 = S2 + ~S1 + ~S2 (omits S1)

Reference: in the veridictory square the four M2 terms are revelation, paradox, subterfuge, deception — evocative single words, not concatenations.

Rules:
- Each: single word or short phrase (1-2 words).
- Each: positive, evocative — a real concept, not "X Y Z" concatenation.
- No duplicates.
- {exclusion_note}

Return JSON with exactly these keys:
{{"M21": "...", "M22": "...", "M23": "...", "M24": "..."}}
"""

PROMPT_L1_MIXED = """Greimas square inner semes:
- S1 = "{s1}", S2 = "{s2}", ~S2 = "{neg_s2}", ~S1 = "{neg_s1}"

Generate the two mixed terms — the joins of the contradictories of each axis:

Mixed1 = S1 + ~S1 = "{s1}" + "{neg_s1}"
Mixed2 = S2 + ~S2 = "{s2}" + "{neg_s2}"

These represent "belonging to this axis at all, without committing to a side."

Rules:
- Each: single word or short phrase.
- STRUCTURAL REQUIREMENT: each name MUST begin with "Mixed ".
- No duplicates.
- {exclusion_note}

Return JSON with exactly:
{{"Mixed1": "...", "Mixed2": "..."}}
"""

# =========================================================
# PROMPTS — LAYER 2
# =========================================================

PROMPT_L2_INNER = """The concept "{parent_term}" is a seme in a Greimas square. Decompose it into its own internal Greimas square.

Identify the two contrary poles into which "{parent_term}" internally polarizes.

Examples:
- "being" internally polarizes into "substance" and "appearance"
- "freedom" internally polarizes into "negative freedom" and "positive freedom"

CRITICAL — the two poles MUST be contraries in the strict sense:
- Both can hold at once; neither can hold; one can hold alone.

Contradictories (exactly one holds), near-synonyms, and unrelated concepts are all WRONG.

Both poles must be specific aspects of "{parent_term}", not separate concepts or successors.

Rules:
- Neither may be a synonym of "{parent_term}" itself.
- {exclusion_note}

Return JSON with exactly:
{{"S1_prime": "...", "S2_prime": "..."}}
"""

PROMPT_L2_NEG_S2 = """The concept "{parent_term}" has been decomposed. Its positive internal poles are:
- S1' = "{s1_prime}"
- S2' = "{s2_prime}"

Generate ~S2', the CONTRADICTORY of S2'.

Rules:
- ~S2' and S2' are contradictories.
- Simple negation of S2' if no better term exists.
- Single word or short phrase.
- {exclusion_note}

Return JSON with exactly: {{"~S2_prime": "..."}}
"""

PROMPT_L2_ANTITHESIS = """The concept "{parent_term}" is being decomposed.
- S1' = "{s1_prime}"
- S2' = "{s2_prime}"
- ~S2' = "{neg_s2_prime}"

Generate ~S1', the CONTRADICTORY of S1'.

CRITICAL: ~S1' is the ANTITHESIS of ~S2', NOT a simple negation of S1'.

Consequences:
- ~S1' and S1' are contradictories.
- ~S1' and ~S2' are contraries.
- ~S1' implies S2'.

Rules:
- Single word or short phrase.
- {exclusion_note}

Return JSON with exactly: {{"~S1_prime": "..."}}
"""

PROMPT_L2_M1 = """Internal square of "{parent_term}":
- S1' = "{s1_prime}", S2' = "{s2_prime}"
- ~S2' = "{neg_s2_prime}", ~S1' = "{neg_s1_prime}"

Generate the four M1 meta-terms:

M11' = S1' + S2';   M12' = ~S1' + ~S2'
M13' = S1' + ~S2';  M14' = S2' + ~S1'

Rules:
- Each: single word or short phrase (1-2 words).
- Each: a positive, evocative concept — NOT a concatenation.
- No duplicates.
- {exclusion_note}

Return JSON:
{{"M11_prime": "...", "M12_prime": "...", "M13_prime": "...", "M14_prime": "..."}}
"""

PROMPT_L2_M2 = """Internal square of "{parent_term}":
- S1' = "{s1_prime}", S2' = "{s2_prime}"
- ~S2' = "{neg_s2_prime}", ~S1' = "{neg_s1_prime}"

Generate the four M2 meta-terms:

M21' = S1' + S2' + ~S2'  (omits ~S1')
M22' = S1' + S2' + ~S1'  (omits ~S2')
M23' = S1' + ~S1' + ~S2' (omits S2')
M24' = S2' + ~S1' + ~S2' (omits S1')

Rules:
- Each: single word or short phrase.
- Each: positive, evocative — a real concept, not concatenation.
- No duplicates.
- {exclusion_note}

Return JSON:
{{"M21_prime": "...", "M22_prime": "...", "M23_prime": "...", "M24_prime": "..."}}
"""

PROMPT_L2_MIXED = """Internal square of "{parent_term}":
- S1' = "{s1_prime}", S2' = "{s2_prime}"
- ~S2' = "{neg_s2_prime}", ~S1' = "{neg_s1_prime}"

Generate the two mixed terms:

Mixed1' = S1' + ~S1'
Mixed2' = S2' + ~S2'

Rules:
- STRUCTURAL REQUIREMENT: each name MUST begin with "Mixed ".
- No duplicates.
- {exclusion_note}

Return JSON:
{{"Mixed1_prime": "...", "Mixed2_prime": "..."}}
"""

# =========================================================
# PROMPT — VALIDATION
# =========================================================

PROMPT_VALIDATE_SQUARE = """Check consistency of a Greimas square:

- S1 = "{s1}", S2 = "{s2}"
- ~S2 = "{neg_s2}", ~S1 = "{neg_s1}"

Check each:
Rule 1: S1 and S2 are CONTRARIES. Both can hold together, both can fail together, one can hold alone. (NOT antonyms. NOT contradictories. NOT synonyms.)
Rule 2: S1 and ~S1 are contradictories.
Rule 3: S2 and ~S2 are contradictories.
Rule 4: ~S2 implies S1.
Rule 5: ~S1 implies S2.
Rule 6: ~S1 and ~S2 are contraries.

Return JSON with exactly: "rule1", "rule2", "rule3", "rule4", "rule5", "rule6".
Each containing {{"pass": true/false, "explanation": "..."}}
"""

# =========================================================
# GENERATION HELPERS
# =========================================================

def generate_with_uniqueness(prompt_fn, required_keys, label,
                             exclude_norms=None, check_keys=None,
                             max_tries=MAX_UNIQUENESS_TRIES):
    if check_keys is None:
        check_keys = required_keys
    forced = set()
    last_data = None
    last_collisions = []
    for attempt in range(1, max_tries + 1):
        extra = list(forced)
        note = exclusion_block(exclude_norms=exclude_norms, extra_terms=extra)
        prompt = prompt_fn(note)
        try:
            data = llm_json(prompt, required_keys, f"{label}:{attempt}")
        except Exception as e:
            print(f"  x {label} attempt {attempt}: {e}")
            continue
        collisions = []
        for k in check_keys:
            val = data.get(k, "").strip()
            if is_collision(val, exclude_locations={label}):
                collisions.append((k, val))
        if not collisions:
            for k in required_keys:
                register_term(data[k].strip(), label, k)
            return data
        for k, val in collisions:
            print(f"  ! {label}.{k}='{val}' collides; forcing exclusion")
            forced.add(val)
        last_data = data
        last_collisions = collisions

    print(f"  ! {label}: EXHAUSTED retries. Logging {len(last_collisions)} collision(s).")
    for k, val in last_collisions:
        COLLISION_LOG.append({"label": label, "key": k, "term": val})
    if last_data:
        for k in required_keys:
            register_term(last_data[k].strip(), label, k)
    return last_data

def fix_mixed_terms(square):
    m1_keys = ["M11", "M12", "M13", "M14"]
    m1_norms = {normalize(square[k]) for k in m1_keys if k in square}

    for slot in ["Mixed1", "Mixed2"]:
        if slot not in square:
            continue
        t = square[slot].strip()
        if normalize(t) in m1_norms:
            new_t = f"Mixed {t}"
            print(f"  [fix] {slot}='{t}' collides with M1; renamed to '{new_t}'")
            square[slot] = new_t

    if (square.get("Mixed1") and square.get("Mixed2")
            and normalize(square["Mixed1"]) == normalize(square["Mixed2"])):
        square["Mixed2"] = f"Mixed {square['Mixed2']}"
        print(f"  [fix] Mixed2 renamed to '{square['Mixed2']}'")

    return square

# =========================================================
# REVERSE FIBRATION
# =========================================================

def infer_parent_concept(x, y, seme_role):
    exclude_norms = {normalize(x), normalize(y)}
    while True:
        print(f"\n[reverse-fibration] Inferring parent seme for ({x}, {y}) → A's {seme_role}")
        def p(note):
            return PROMPT_REVERSE_FIBRATION.format(x=x, y=y, exclusion_note=note)
        try:
            data = llm_json(p(""), ["parent"], f"reverse:{seme_role}")
            proposed = data["parent"].strip()
        except Exception as e:
            print(f"  LLM failed: {e}")
            proposed = None

        if proposed:
            try:
                check = llm_json(
                    PROMPT_CHECK_UNIFIER.format(parent=proposed, x=x, y=y),
                    ["pass", "explanation"], f"check_unifier:{seme_role}")
                expl = check.get("explanation", "")
            except Exception:
                expl = "(could not validate)"

            print(f"\n  LLM proposal: parent = '{proposed}'")
            if expl:
                print(f"  Plausibility: {expl}")
            print(f"\n  This will be A's {seme_role}.")
            print("  Options: [c] Confirm / [o] Override / [r] Regenerate / [q] Abort")
            choice = input("  Choice [c/o/r/q]: ").strip().lower()

            if _is_exit(choice):
                _exit_cleanly("Exiting.")
            if choice == "c":
                register_term(proposed, f"A:{seme_role}", seme_role)
                return proposed
            if choice == "o":
                override = input("  Enter your own parent term (or [q] to exit): ").strip()
                if _is_exit(override):
                    _exit_cleanly("Exiting.")
                if not override:
                    continue
                if normalize(override) in exclude_norms:
                    print("  That is one of the internal poles; try again.")
                    continue
                register_term(override, f"A:{seme_role}", seme_role)
                return override
            if choice == "r":
                continue
        else:
            choice = input("  [o] Override / [q] Abort: ").strip().lower()
            if _is_exit(choice):
                _exit_cleanly("Exiting.")
            if choice == "o":
                override = input("  Enter your own parent term (or [q] to exit): ").strip()
                if _is_exit(override):
                    _exit_cleanly("Exiting.")
                if override:
                    register_term(override, f"A:{seme_role}", seme_role)
                    return override
            _exit_cleanly("Exiting.")

def infer_a_s2_from_s1(s1):
    print(f"\n[generate] A's S2 from S1 = '{s1}'")
    def p(note):
        return PROMPT_CONTINUE_A_FROM_S1.format(s1=s1, exclusion_note=note)
    data = generate_with_uniqueness(
        p, ["S2"], "A:S2", exclude_norms={normalize(s1)})
    return data["S2"].strip()

# =========================================================
# LAYER 1 GENERATION
# =========================================================

def generate_layer1_square(s1, s2, skeleton=None):
    square = {"S1": s1, "S2": s2}
    s_norms = {normalize(s1), normalize(s2)}
    skeleton = skeleton or {}

    if skeleton.get("~S2"):
        square["~S2"] = skeleton["~S2"].strip()
        print(f"  ~S2 = '{square['~S2']}' (from cycle 1)")
        register_term(square["~S2"], "L1:~S2", "~S2")
    else:
        print("  ~S2")
        def p1(note):
            return PROMPT_L1_NEG_S2.format(s1=s1, s2=s2, exclusion_note=note)
        square["~S2"] = generate_with_uniqueness(
            p1, ["~S2"], "L1:~S2", exclude_norms=s_norms)["~S2"].strip()

    if skeleton.get("~S1"):
        square["~S1"] = skeleton["~S1"].strip()
        print(f"  ~S1 = '{square['~S1']}' (from cycle 1)")
        register_term(square["~S1"], "L1:~S1", "~S1")
    else:
        print("  ~S1")
        def p2(note):
            return PROMPT_L1_ANTITHESIS.format(
                s1=s1, s2=s2, neg_s2=square["~S2"], exclusion_note=note)
        square["~S1"] = generate_with_uniqueness(
            p2, ["~S1"], "L1:~S1", exclude_norms=s_norms)["~S1"].strip()

    inner_norms = {normalize(square[k]) for k in ["S1", "S2", "~S2", "~S1"]}

    print("  M1 batch")
    def p3(note):
        return PROMPT_L1_M1.format(
            s1=square["S1"], s2=square["S2"],
            neg_s2=square["~S2"], neg_s1=square["~S1"], exclusion_note=note)
    m1 = generate_with_uniqueness(
        p3, ["M11", "M12", "M13", "M14"], "L1:M1", exclude_norms=inner_norms)
    for k in ["M11", "M12", "M13", "M14"]:
        square[k] = m1[k].strip()

    m1_norms = {normalize(square[k]) for k in ["M11", "M12", "M13", "M14"]}

    print("  M2 batch")
    def p4(note):
        return PROMPT_L1_M2.format(
            s1=square["S1"], s2=square["S2"],
            neg_s2=square["~S2"], neg_s1=square["~S1"], exclusion_note=note)
    m2 = generate_with_uniqueness(
        p4, ["M21", "M22", "M23", "M24"], "L1:M2",
        exclude_norms=inner_norms | m1_norms)
    for k in ["M21", "M22", "M23", "M24"]:
        square[k] = m2[k].strip()

    m2_norms = {normalize(square[k]) for k in ["M21", "M22", "M23", "M24"]}

    print("  Mixed batch")
    def p5(note):
        return PROMPT_L1_MIXED.format(
            s1=square["S1"], s2=square["S2"],
            neg_s2=square["~S2"], neg_s1=square["~S1"], exclusion_note=note)
    mix = generate_with_uniqueness(
        p5, ["Mixed1", "Mixed2"], "L1:Mixed",
        exclude_norms=inner_norms | m1_norms | m2_norms)
    square["Mixed1"] = mix["Mixed1"].strip()
    square["Mixed2"] = mix["Mixed2"].strip()

    square = fix_mixed_terms(square)
    return square

# =========================================================
# LAYER 2 GENERATION
# =========================================================

def generate_layer2_complete(parent_term, parent_key, fixed_square=None):
    label_prefix = f"L2[{parent_key}]"
    print(f"\n[L2] Decomposing {parent_key} = '{parent_term}'")

    square = {}
    fixed_square = fixed_square or {}

    if fixed_square.get("S1") and fixed_square.get("S2"):
        square["S1"] = fixed_square["S1"]
        square["S2"] = fixed_square["S2"]
        print(f"  using fixed inner pair: S1'='{square['S1']}', S2'='{square['S2']}'")
    else:
        print("  inner pair (S1', S2')")
        def p1(note):
            return PROMPT_L2_INNER.format(parent_term=parent_term, exclusion_note=note)
        pair = generate_with_uniqueness(
            p1, ["S1_prime", "S2_prime"], f"{label_prefix}:inner")
        square["S1"] = pair["S1_prime"].strip()
        square["S2"] = pair["S2_prime"].strip()

    inner_norms = {normalize(square["S1"]), normalize(square["S2"])}

    if fixed_square.get("~S2"):
        square["~S2"] = fixed_square["~S2"]
        print(f"  ~S2' = '{square['~S2']}' (from cycle 1)")
        register_term(square["~S2"], f"{label_prefix}:~S2", "~S2")
    else:
        print("  ~S2'")
        def p2(note):
            return PROMPT_L2_NEG_S2.format(
                parent_term=parent_term,
                s1_prime=square["S1"], s2_prime=square["S2"], exclusion_note=note)
        square["~S2"] = generate_with_uniqueness(
            p2, ["~S2_prime"], f"{label_prefix}:~S2",
            exclude_norms=inner_norms)["~S2_prime"].strip()

    if fixed_square.get("~S1"):
        square["~S1"] = fixed_square["~S1"]
        print(f"  ~S1' = '{square['~S1']}' (from cycle 1)")
        register_term(square["~S1"], f"{label_prefix}:~S1", "~S1")
    else:
        print("  ~S1'")
        def p3(note):
            return PROMPT_L2_ANTITHESIS.format(
                parent_term=parent_term,
                s1_prime=square["S1"], s2_prime=square["S2"],
                neg_s2_prime=square["~S2"], exclusion_note=note)
        square["~S1"] = generate_with_uniqueness(
            p3, ["~S1_prime"], f"{label_prefix}:~S1",
            exclude_norms=inner_norms)["~S1_prime"].strip()

    full_inner_norms = {normalize(square[k]) for k in ["S1", "S2", "~S2", "~S1"]}

    print("  M1' batch")
    def p4(note):
        return PROMPT_L2_M1.format(
            parent_term=parent_term,
            s1_prime=square["S1"], s2_prime=square["S2"],
            neg_s2_prime=square["~S2"], neg_s1_prime=square["~S1"],
            exclusion_note=note)
    m1 = generate_with_uniqueness(
        p4, ["M11_prime", "M12_prime", "M13_prime", "M14_prime"],
        f"{label_prefix}:M1", exclude_norms=full_inner_norms)
    square["M11"] = m1["M11_prime"].strip()
    square["M12"] = m1["M12_prime"].strip()
    square["M13"] = m1["M13_prime"].strip()
    square["M14"] = m1["M14_prime"].strip()

    m1_norms = {normalize(square[k]) for k in ["M11", "M12", "M13", "M14"]}

    print("  M2' batch")
    def p5(note):
        return PROMPT_L2_M2.format(
            parent_term=parent_term,
            s1_prime=square["S1"], s2_prime=square["S2"],
            neg_s2_prime=square["~S2"], neg_s1_prime=square["~S1"],
            exclusion_note=note)
    m2 = generate_with_uniqueness(
        p5, ["M21_prime", "M22_prime", "M23_prime", "M24_prime"],
        f"{label_prefix}:M2", exclude_norms=full_inner_norms | m1_norms)
    square["M21"] = m2["M21_prime"].strip()
    square["M22"] = m2["M22_prime"].strip()
    square["M23"] = m2["M23_prime"].strip()
    square["M24"] = m2["M24_prime"].strip()

    m2_norms = {normalize(square[k]) for k in ["M21", "M22", "M23", "M24"]}

    print("  Mixed' batch")
    def p6(note):
        return PROMPT_L2_MIXED.format(
            parent_term=parent_term,
            s1_prime=square["S1"], s2_prime=square["S2"],
            neg_s2_prime=square["~S2"], neg_s1_prime=square["~S1"],
            exclusion_note=note)
    mix = generate_with_uniqueness(
        p6, ["Mixed1_prime", "Mixed2_prime"], f"{label_prefix}:Mixed",
        exclude_norms=full_inner_norms | m1_norms | m2_norms)
    square["Mixed1"] = mix["Mixed1_prime"].strip()
    square["Mixed2"] = mix["Mixed2_prime"].strip()

    square = fix_mixed_terms(square)

    return {
        "parent_seme": parent_key,
        "parent_term": parent_term,
        "square": square,
    }

# =========================================================
# VALIDATION
# =========================================================

def validate_square(square, label, validator_model=None):
    print(f"\n[validate:{label}] using {validator_model or VALIDATOR_MODEL}")
    prompt = PROMPT_VALIDATE_SQUARE.format(
        s1=square["S1"], s2=square["S2"],
        neg_s2=square["~S2"], neg_s1=square["~S1"])
    try:
        return llm_json(prompt,
                        ["rule1", "rule2", "rule3", "rule4", "rule5", "rule6"],
                        f"validate:{label}",
                        model_override=validator_model or VALIDATOR_MODEL)
    except Exception as e:
        print(f"  validation failed: {e}")
        return {f"rule{i}": {"pass": False, "explanation": "validation error"}
                for i in range(1, 7)}

def all_rules_pass(validation_dict):
    if not validation_dict:
        return False
    for k, v in validation_dict.items():
        if isinstance(v, dict) and not v.get("pass"):
            return False
    return True

# =========================================================
# LATTICE BUILDER
# =========================================================

def build_lattice(square):
    s1 = square["S1"]; s2 = square["S2"]
    ns2 = square["~S2"]; ns1 = square["~S1"]
    m11 = square["M11"]; m12 = square["M12"]
    m13 = square["M13"]; m14 = square["M14"]
    m21 = square["M21"]; m22 = square["M22"]
    m23 = square["M23"]; m24 = square["M24"]
    mix1 = square["Mixed1"]; mix2 = square["Mixed2"]

    inner_slots = [("S1", s1), ("S2", s2), ("~S1", ns1), ("~S2", ns2)]
    pair_slots = [
        ("M11", m11, [s1, s2]), ("M12", m12, [ns1, ns2]),
        ("M13", m13, [s1, ns2]), ("M14", m14, [s2, ns1]),
        ("Mixed1", mix1, [s1, ns1]), ("Mixed2", mix2, [s2, ns2]),
    ]
    m2_slots = [
        ("M21", m21, [s1, s2, ns2]), ("M22", m22, [s1, s2, ns1]),
        ("M23", m23, [ns1, ns2, s1]), ("M24", m24, [ns1, ns2, s2]),
    ]

    def m2_terms_containing(extent):
        return [t for _, t, e in m2_slots if all(x in e for x in extent)]

    concepts = []; slot_to_id = {}; cid = 0
    slot_to_id["bottom"] = cid
    concepts.append({"id": cid, "label": "bottom", "short": "Bottom",
                     "name": "Bottom — empty extent, all four M2 attributes",
                     "extent": [], "intent": [t for _, t, _ in m2_slots]})
    cid += 1

    for slot, term in inner_slots:
        slot_to_id[slot] = cid
        concepts.append({"id": cid, "label": term, "short": slot,
                         "name": f"{term} — inner term {slot}",
                         "extent": [term], "intent": m2_terms_containing([term])})
        cid += 1

    for slot, term, extent in pair_slots:
        slot_to_id[slot] = cid
        kind = "mixed" if slot.startswith("Mix") else "M1"
        concepts.append({"id": cid, "label": term,
                         "short": f"{'Mix' if kind == 'mixed' else 'M1'}:{term}",
                         "name": f"{term} — {kind} meta-term of {', '.join(extent)}",
                         "extent": list(extent), "intent": m2_terms_containing(extent)})
        cid += 1

    for slot, term, extent in m2_slots:
        slot_to_id[slot] = cid
        concepts.append({"id": cid, "label": term, "short": f"M2:{term}",
                         "name": f"{term} — M2 meta-term containing {', '.join(extent)}",
                         "extent": list(extent), "intent": [term]})
        cid += 1

    slot_to_id["top"] = cid
    concepts.append({"id": cid, "label": "top", "short": "Top",
                     "name": "Top — all inner terms, no shared attribute",
                     "extent": [t for _, t in inner_slots], "intent": []})
    cid += 1

    edges = []; edge_set = set()
    def add_edge(u, l):
        uu = slot_to_id[u]; ll = slot_to_id[l]
        key = (uu, ll)
        if key not in edge_set:
            edge_set.add(key); edges.append([uu, ll])

    for m2_slot, _, _ in m2_slots:
        add_edge("top", m2_slot)
    for m2_slot, _, m2_ext in m2_slots:
        for pair_slot, _, pair_ext in pair_slots:
            if all(x in m2_ext for x in pair_ext):
                add_edge(m2_slot, pair_slot)
    for pair_slot, _, pair_ext in pair_slots:
        for inner_slot, inner_term in inner_slots:
            if inner_term in pair_ext:
                add_edge(pair_slot, inner_slot)
    for inner_slot, _ in inner_slots:
        add_edge(inner_slot, "bottom")

    return {"concepts": concepts, "edges": edges,
            "inner_ids": {term: slot_to_id[slot] for slot, term in inner_slots},
            "top_id": slot_to_id["top"], "bottom_id": slot_to_id["bottom"]}

# =========================================================
# ARCHITECTONIC LATTICE
# =========================================================

def build_architectonic(layer1_lattice, layer2_data):
    concepts = []; edges = []; interface_map = {}
    A = layer1_lattice
    a_id_map = {}
    a_seme_ids = set(A["inner_ids"].values())
    a_bottom_id = A["bottom_id"]

    for c in A["concepts"]:
        if c["id"] in a_seme_ids:
            continue
        new_id = f"A:{c['id']}"
        new_c = dict(c); new_c["id"] = new_id
        new_c["layer"] = 1; new_c["origin"] = "A"
        if c["id"] == a_bottom_id:
            new_c["role"] = "shared_bottom"
        elif c["id"] == A["top_id"]:
            new_c["role"] = "architectonic_top"
        else:
            new_c["role"] = "layer_1_internal"
        concepts.append(new_c)
        a_id_map[c["id"]] = new_id

    b_id_maps = {}
    for seme_key, data in layer2_data.items():
        B = data["lattice"]; b_id_maps[seme_key] = {}
        for c in B["concepts"]:
            if c["id"] == B["bottom_id"]:
                continue
            new_id = f"{seme_key}:{c['id']}"
            new_c = dict(c); new_c["id"] = new_id
            new_c["layer"] = 2; new_c["origin"] = seme_key
            if c["id"] == B["top_id"]:
                new_c["role"] = "interface"
                new_c["name"] = f"Fully-articulated {data['parent_term']}"
                new_c["short"] = data["parent_term"]
                new_c["label"] = f"{data['parent_term']}*"
            else:
                new_c["role"] = "layer_2_internal"
            concepts.append(new_c)
            b_id_maps[seme_key][c["id"]] = new_id
        interface_map[seme_key] = b_id_maps[seme_key][B["top_id"]]

    for edge in A["edges"]:
        from_id, to_id = edge
        if from_id in a_seme_ids or to_id in a_seme_ids:
            continue
        edges.append([a_id_map[from_id], a_id_map[to_id]])

    a_seme_id_to_key = {}
    for seme_term, old_id in A["inner_ids"].items():
        for k, data in layer2_data.items():
            if data["parent_term"] == seme_term:
                a_seme_id_to_key[old_id] = k
                break

    for edge in A["edges"]:
        from_id, to_id = edge
        if to_id in a_seme_ids:
            seme_key = a_seme_id_to_key[to_id]
            edges.append([a_id_map[from_id], interface_map[seme_key]])

    for seme_key, data in layer2_data.items():
        B = data["lattice"]; b_bottom_old = B["bottom_id"]
        for edge in B["edges"]:
            from_id, to_id = edge
            if to_id == b_bottom_old:
                if from_id == B["top_id"]:
                    continue
                edges.append([b_id_maps[seme_key][from_id], a_id_map[a_bottom_id]])
            elif from_id == b_bottom_old:
                continue
            else:
                edges.append([b_id_maps[seme_key][from_id], b_id_maps[seme_key][to_id]])

    dedup = []; seen = set()
    for e in edges:
        key = (e[0], e[1])
        if key not in seen:
            seen.add(key); dedup.append(e)

    return {"concepts": concepts, "edges": dedup, "interface_map": interface_map,
            "a_id_map": a_id_map, "b_id_maps": b_id_maps}

# =========================================================
# FULL RUN
# =========================================================

SEME_ROLES = ["S1", "S2", "~S1", "~S2"]

def run_full_generation(seed, seed_s1, seed_s2, seed_skeleton=None):
    layer2 = {}
    fixed_B = None

    if seed == "A1":
        a_s1 = seed_s1
        a_s2 = seed_s2
        register_term(a_s1, "A:S1", "S1")
        register_term(a_s2, "A:S2", "S2")
    else:
        role_map = {"1": "S1", "2": "S2", "3": "~S1", "4": "~S2"}
        seme_role = role_map[seed[1:]]
        b_label = f"L2[{seme_role}]"
        register_term(seed_s1, f"{b_label}:S1", "S1")
        register_term(seed_s2, f"{b_label}:S2", "S2")

        parent = infer_parent_concept(seed_s1, seed_s2, seme_role)

        if seme_role == "S1":
            a_s1 = parent
            a_s2 = infer_a_s2_from_s1(a_s1)
            register_term(a_s2, "A:S2", "S2")
        elif seme_role == "S2":
            a_s2 = parent
            a_s1 = infer_a_s2_from_s1(a_s2)
            register_term(a_s1, "A:S1", "S1")
        else:
            if seme_role == "~S1":
                a_s1 = infer_a_s2_from_s1(parent); a_s2 = parent
                register_term(a_s1, "A:S1", "S1")
            else:
                a_s2 = infer_a_s2_from_s1(parent); a_s1 = parent
                register_term(a_s1, "A:S1", "S1")
            register_term(a_s2, "A:S2", "S2")

        fixed_B = (seme_role, {
            "S1": seed_s1,
            "S2": seed_s2,
            "~S2": seed_skeleton.get("~S2") if seed_skeleton else None,
            "~S1": seed_skeleton.get("~S1") if seed_skeleton else None,
        })

    print(f"\n[L1] Generating layer 1 (S1='{a_s1}', S2='{a_s2}')...")

    if seed == "A1" and seed_skeleton:
        layer1_square = generate_layer1_square(a_s1, a_s2, skeleton=seed_skeleton)
    else:
        layer1_square = generate_layer1_square(a_s1, a_s2)
    layer1_lattice = build_lattice(layer1_square)

    for seme_key in SEME_ROLES:
        parent_term = layer1_square[seme_key]
        if fixed_B and fixed_B[0] == seme_key:
            b_data = generate_layer2_complete(
                parent_term, seme_key, fixed_square=fixed_B[1])
        else:
            b_data = generate_layer2_complete(parent_term, seme_key)
        b_data["lattice"] = build_lattice(b_data["square"])
        layer2[seme_key] = b_data

    print("\n[validate] running square validations...")
    validation = {"layer_1": validate_square(layer1_square, "layer_1")}
    for seme_key, data in layer2.items():
        validation[f"layer_2_{seme_key}"] = validate_square(
            data["square"], f"layer_2_{seme_key}")

    print("\n[architectonic] Building lattice C...")
    architectonic = build_architectonic(layer1_lattice, layer2)
    print(f"  concepts: {len(architectonic['concepts'])}")
    print(f"  edges:    {len(architectonic['edges'])}")

    return {"seed": seed, "seed_s1": seed_s1, "seed_s2": seed_s2,
            "layer1_square": layer1_square, "layer1_lattice": layer1_lattice,
            "layer2": layer2, "validation": validation,
            "architectonic": architectonic}

# =========================================================
# MODEL SELECTION
# =========================================================

def _flatten_recommended():
    seen = set(); out = []
    for tier_name in ("production", "iteration"):
        for model_name, desc in MODEL_MENUS[tier_name]:
            if model_name == "CUSTOM":
                continue
            if model_name in seen:
                continue
            seen.add(model_name)
            out.append((model_name, tier_name, desc))
    return out

def select_generator_model():
    print(); print("=" * 64)
    print("Quality mode — generator model selection")
    print("=" * 64)
    print("  [1] production   — larger models (7B-14B)")
    print("  [2] iteration    — medium models for fast iteration")
    print("  [3] list all     — show every model installed in Ollama")
    print("  [q] exit")

    tier_raw = input("Mode [1/2/3/q] (default: 2): ").strip().lower()
    if _is_exit(tier_raw):
        _exit_cleanly("Exiting.")
    if not tier_raw or tier_raw == "2":
        tier = "iteration"
    elif tier_raw in ("3", "list", "l"):
        return select_from_all()
    else:
        tier = TIER_ALIASES.get(tier_raw, "iteration")

    menu = MODEL_MENUS[tier]
    print(f"\n{tier.capitalize()} models:")
    for i, (name, desc) in enumerate(menu, 1):
        print(f"  [{i}] {name:<22s} — {desc}")

    sel = input(f"Select [1-{len(menu)}, q] (default: 1): ").strip().lower()
    if _is_exit(sel):
        _exit_cleanly("Exiting.")
    if not sel:
        name = menu[0][0]
    else:
        try:
            idx = int(sel) - 1
            name = menu[idx][0]
        except (ValueError, IndexError):
            print(f"Invalid selection; defaulting to {menu[0][0]}")
            name = menu[0][0]

    if name == "CUSTOM":
        name = input("Enter model name (or [q] to exit): ").strip()
        if _is_exit(name):
            _exit_cleanly("Exiting.")
        if not name:
            name = menu[0][0]
            print(f"Empty; defaulting to {name}")
    return name

def select_from_all():
    available = list_ollama_models()
    recommended = _flatten_recommended()
    recommended_names = {m for m, _, _ in recommended}

    printed = []; idx = 1
    print(); print("=" * 64)
    print("All Ollama models"); print("=" * 64)
    if not available:
        print("  (No models detected in local Ollama.)")

    print("Recommended:")
    for name, tier_name, desc in recommended:
        present = find_present_variant(name, available)
        mark = " ✓ installed" if present else "  (not installed)"
        print(f"  [{idx:>2}] {name:<22s} [{tier_name:<10s}] {desc}{mark}")
        printed.append({"name": name, "recommended": True, "present": present})
        idx += 1

    others = []
    for m in available:
        base = m.split(":")[0]
        if m in recommended_names or base in recommended_names:
            continue
        others.append(m)
    if others:
        print(); print("Other installed models:")
        for m in others:
            print(f"  [{idx:>2}] {m}")
            printed.append({"name": m, "recommended": False, "present": m})
            idx += 1

    print(); print("  [q] exit")
    sel = input(f"Select [1-{len(printed)}, q]: ").strip().lower()
    if _is_exit(sel):
        _exit_cleanly("Exiting.")
    if not sel:
        if printed:
            entry = printed[0]
        else:
            _exit_cleanly("Nothing to select. Exiting.")
    else:
        try:
            n = int(sel)
            if n < 1 or n > len(printed):
                raise ValueError
            entry = printed[n - 1]
        except ValueError:
            print("Invalid selection.")
            return select_from_all()

    if not entry["recommended"]:
        print()
        print(f"  '{entry['name']}' is not in our shortlist of recommended models.")
        print(f"  It may produce lower-quality results for this task.")
        conf = input("  Are you sure? [y/N]: ").strip().lower()
        if conf != "y":
            return select_from_all()

    return entry["name"]

def select_cycle1_model(args, current_model, available):
    if args.cycle1_model:
        return current_model

    print()
    print("=" * 64)
    print("Cycle 1 model")
    print("=" * 64)
    print(f"  [1] use the default quick model for this run: {current_model}")
    print("  [2] choose a different model from installed Ollama models")
    print("  [q] exit")

    sel = input("Choice [1/2/q] (default: 1): ").strip().lower()
    if _is_exit(sel):
        _exit_cleanly("Exiting.")
    if not sel or sel == "1":
        return current_model

    chosen = select_from_all()

    if available and chosen not in available:
        match = find_present_variant(chosen, available)
        if match:
            print(f"  Note: using '{match}' (variant of '{chosen}').")
            return match
        print(f"  Note: model '{chosen}' not found locally. It will be used as-is.")

    return chosen

def select_mode_and_models(args, available):
    global VALIDATOR_MODEL

    if args.model and args.cycle1_model and args.validator_model:
        return (args.model, args.cycle1_model, args.validator_model)

    print(); print("=" * 64)
    print("Run mode")
    print("=" * 64)
    print("  [1] quality — pick a large generator model; small fixed models")
    print("                for cycle 1 and validation")
    print("  [2] fast    — one small model for all three cycles")
    print("  [q] exit")

    mode_raw = input("Mode [1/2/q] (default: 2): ").strip().lower()
    if _is_exit(mode_raw):
        _exit_cleanly("Exiting.")
    if not mode_raw or mode_raw == "2":
        mode = "fast"
    elif mode_raw == "1":
        mode = "quality"
    else:
        mode = "fast"

    if mode == "fast":
        fast = args.model or FAST_MODE_MODEL
        cycle1 = args.cycle1_model or fast
        validator = args.validator_model or fast
        print(f"  Fast mode: all three cycles use '{fast}'")
        return (fast, cycle1, validator)

    generator = args.model or select_generator_model()

    if available and generator not in available:
        match = find_present_variant(generator, available)
        if match:
            print(f"  Note: using '{match}' (variant of '{generator}').")
            generator = match
        else:
            print(f"\n  Note: model '{generator}' not found locally.")
            resp = input("  Continue anyway? [y/N] (or [q] to exit): ").strip().lower()
            if _is_exit(resp):
                _exit_cleanly("Exiting.")
            if resp != "y":
                sys.exit(1)

    cycle1 = args.cycle1_model or DEFAULT_CYCLE1_MODEL
    validator = args.validator_model or DEFAULT_VALIDATOR_MODEL

    if not args.validator_model and normalize(validator) == normalize(generator):
        alt = "llama3.2:3b"
        if available and find_present_variant(alt, available):
            validator = alt
            print(f"  (Validator auto-switched to '{validator}' to differ from generator.)")
        else:
            print(f"  Note: validator defaulting to '{validator}' (same as generator).")

    if not args.cycle1_model and normalize(cycle1) == normalize(generator):
        alt = "llama3.2:3b"
        if available and find_present_variant(alt, available):
            cycle1 = alt
            print(f"  (Cycle 1 auto-switched to '{cycle1}' to differ from generator.)")

    VALIDATOR_MODEL = validator
    return (generator, cycle1, validator)

# =========================================================
# MAIN
# =========================================================

def main():
    global MODEL, VALIDATOR_MODEL

    if "--list-models" in sys.argv:
        names = list_ollama_models()
        if names:
            print("Available Ollama models:")
            for n in names:
                print(f"  {n}")
        else:
            print("No models found. Pull one with: ollama pull phi4-mini")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Architectonic Greimas generator (v10).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python greimas_architectonic.py
  python greimas_architectonic.py --s1 being
  python greimas_architectonic.py --s1 being --s2 seeming   # skips cycle 1
  python greimas_architectonic.py --no-cycle1 --s1 being
  python greimas_architectonic.py --cycle1-model llama3.2:3b

The --cycle1-model flag, if provided, suppresses the per-run cycle-1
model prompt inside path [1].
""",
    )
    parser.add_argument("--seed", choices=["A1","B1","B2","B3","B4"],
                        help="Which square to seed (default: A1)")
    parser.add_argument("--s1", help="S1 of the seeded square")
    parser.add_argument("--s2", help="S2 of the seeded square (skips cycle 1 if given)")
    parser.add_argument("--output", "-o", default="architectonic.json",
                        help="Output JSON (default: architectonic.json)")
    parser.add_argument("--model", "-m", default=None,
                        help="Main model for generation (skips the mode menu if given)")
    parser.add_argument("--cycle1-model", default=None,
                        help=f"Model for cycle 1 (default: {DEFAULT_CYCLE1_MODEL}); "
                             f"if set, skips the cycle-1 model prompt")
    parser.add_argument("--validator-model", default=None,
                        help=f"Model for validation (default: {DEFAULT_VALIDATOR_MODEL})")
    parser.add_argument("--no-cycle1", action="store_true",
                        help="Default to skipping cycle 1 (path [2])")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip LLM validation steps")
    parser.add_argument("--skip-layer2", action="store_true",
                        help="Generate only layer 1")
    args = parser.parse_args()

    available = list_ollama_models()

    while True:
        USED_TERMS.clear()
        REGISTRATION_ORDER.clear()
        COLLISION_LOG.clear()

        # ---- Step 1: seed ----
        if args.seed:
            seed = args.seed
        else:
            seed_raw = input(
                "\nWhich square to seed? [A1/B1/B2/B3/B4] (default A1, [q] to exit): "
            ).strip().upper()
            if _is_exit(seed_raw):
                _exit_cleanly("Exiting.")
            seed = seed_raw or "A1"
            if seed not in ["A1","B1","B2","B3","B4"]:
                print(f"Invalid seed: {seed}")
                sys.exit(1)

        # ---- Step 2: mode + model selection ----
        generator_model, cycle1_model, validator_model = select_mode_and_models(args, available)
        MODEL = generator_model
        VALIDATOR_MODEL = validator_model

        # ---- Step 3: S1 entry + path choice ----
        skeleton = None
        seed_s1 = None
        seed_s2 = None

        while True:
            if args.s1 and seed_s1 is None:
                seed_s1 = args.s1
            elif seed_s1 is None:
                if seed == "A1":
                    prompt_s1 = "Enter S1 (first positive pole, or [q] to exit): "
                else:
                    prompt_s1 = f"Enter S1 for {seed} (first internal pole, or [q] to exit): "
                seed_s1 = input(prompt_s1).strip()
                if _is_exit(seed_s1):
                    _exit_cleanly("Exiting.")
                if not seed_s1:
                    print("Error: S1 is required.")
                    sys.exit(1)

            if args.s2:
                seed_s2 = args.s2
                skeleton = None
                break

            print()
            print("How would you like to proceed?")
            print("  [1] Let the model propose candidate squares (cycle 1)")
            print("  [2] Enter S2 directly and skip cycle 1")
            print("  [q] exit")
            default_path = "2" if args.no_cycle1 else "1"
            path = input(f"Choice [1/2/q] (default: {default_path}): ").strip().lower()
            if _is_exit(path):
                _exit_cleanly("Exiting.")
            if not path:
                path = default_path

            if path == "2":
                seed_s2 = input("Enter S2 (or [q] to exit): ").strip()
                if _is_exit(seed_s2):
                    _exit_cleanly("Exiting.")
                if not seed_s2:
                    print("Error: S2 is required.")
                    sys.exit(1)
                skeleton = None
                break

            # path == "1": prompt for cycle 1 model, then run cycle 1
            cycle1_model = select_cycle1_model(args, cycle1_model, available)

            status, data = cycle1_run(seed_s1, seed, cycle1_model)
            if status == "back":
                seed_s1 = None
                seed_s2 = None
                skeleton = None
                continue
            if status == "s2_only":
                seed_s2 = data["S2"]
                skeleton = None
                break
            seed_s2 = data["S2"]
            skeleton = {"~S2": data["~S2"], "~S1": data["~S1"]}
            break

        if not seed_s2:
            print("Error: no S2 selected.")
            sys.exit(1)

        # ---- Step 4: generation ----
        print()
        print("=" * 64)
        print("Architectonic Greimas Generator (v10)")
        print(f"  Seed:      {seed}")
        print(f"  S1:        '{seed_s1}'")
        print(f"  S2:        '{seed_s2}'")
        if skeleton:
            print(f"  ~S2:       '{skeleton['~S2']}'")
            print(f"  ~S1:       '{skeleton['~S1']}'")
        print(f"  Cycle 1 model: {cycle1_model}")
        print(f"  Generator:     {generator_model}")
        print(f"  Validator:     {validator_model}")
        print("=" * 64)

        try:
            result = run_full_generation(seed, seed_s1, seed_s2, seed_skeleton=skeleton)
        except KeyboardInterrupt:
            print("\nInterrupted.")
            sys.exit(1)
        except Exception as e:
            print(f"\nGeneration failed: {e}")
            resp = input("Try again? [y/N] (or [q] to exit): ").strip().lower()
            if _is_exit(resp):
                _exit_cleanly("Exiting.")
            if resp == "y":
                args.s1 = None; args.s2 = None; args.seed = None
                continue
            sys.exit(1)

        if COLLISION_LOG:
            print(f"\n[collisions] {len(COLLISION_LOG)} unresolved term collision(s):")
            for c in COLLISION_LOG:
                print(f"  - {c['label']}.{c['key']} = '{c['term']}'")

        all_ok = True; failed_labels = []
        for label, vd in result["validation"].items():
            if not all_rules_pass(vd):
                all_ok = False; failed_labels.append(label)

        if all_ok:
            print("\nAll square-level validations passed.")
        else:
            print(f"\nValidation failed for: {', '.join(failed_labels)}")

        if args.skip_validation:
            break

        if not all_ok:
            print("\nOptions:")
            print("  [w] Write output anyway")
            print("  [r] Retry generation (keep same S1/S2)")
            print("  [c] Change S1/S2 and retry")
            print("  [q] Abort without writing")
            choice = input("  Choice [w/r/c/q]: ").strip().lower()
            if _is_exit(choice):
                _exit_cleanly("Exiting without writing.")
            if choice == "w":
                break
            elif choice == "r":
                continue
            elif choice == "c":
                args.s1 = None; args.s2 = None
                continue
            else:
                sys.exit(0)
        else:
            break

    output = {
        "metadata": {
            "seed": result["seed"],
            "seed_s1": result["seed_s1"],
            "seed_s2": result["seed_s2"],
            "cycle1_model": cycle1_model,
            "generator_model": MODEL,
            "validator_model": VALIDATOR_MODEL,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "collisions": COLLISION_LOG,
        },
        "layer_1": {
            "square": result["layer1_square"],
            "lattice": {
                "concepts": result["layer1_lattice"]["concepts"],
                "edges": result["layer1_lattice"]["edges"],
            },
        },
        "layer_2": {
            key: {
                "parent_seme": data["parent_seme"],
                "parent_term": data["parent_term"],
                "square": data["square"],
                "lattice": {
                    "concepts": data["lattice"]["concepts"],
                    "edges": data["lattice"]["edges"],
                },
            }
            for key, data in result["layer2"].items()
        },
        "validation": result["validation"],
        "architectonic": {
            "concepts": result["architectonic"]["concepts"],
            "edges": result["architectonic"]["edges"],
        },
    }

    Path(args.output).write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n✓ Wrote {args.output}")

if __name__ == "__main__":
    main()