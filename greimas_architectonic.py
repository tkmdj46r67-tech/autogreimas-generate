#!/usr/bin/env python3
"""
Architectonic Greimas Generator — Version 3

Changes from V2:
  * build_lattice keys concepts by SLOT (M11, Mixed1, ...) not by term
    string. Prevents silent edge corruption when two slots produce the
    same term.
  * Edges are deduplicated.
  * Mixed terms are structurally prevented from colliding with M1 terms.
  * Uniqueness fallback logs collisions rather than silently accepting them.
  * Three-tier model selection menu at startup: production / iteration / mini.

Usage:
    python greimas_architectonic.py
    python greimas_architectonic.py --s1 being --s2 seeming
    python greimas_architectonic.py --seed B1
    python greimas_architectonic.py --model phi4:14b
    python greimas_architectonic.py --list-models

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
TEMPERATURE = 0.3
NUM_PREDICT = 1200
MAX_RETRIES = 3
MAX_UNIQUENESS_TRIES = 4

# =========================================================
# MODEL MENUS
# =========================================================

MODEL_MENUS = {
    "production": [
        ("phi4:14b",
         "Phi-4 (14B) — the larger Phi; strong abstract reasoning; partial GPU offload"),
        ("qwen2.5:14b",
         "Qwen 2.5 (14B) — strong structured output; partial GPU offload"),
        ("mistral-nemo:12b",
         "Mistral Nemo (12B) — strong instruction follower; partial offload"),
        ("llama3.1:8b",
         "Llama 3.1 (8B) — best all-around 8B; full GPU offload on RX 580 via Vulkan"),
        ("qwen2.5:7b",
         "Qwen 2.5 (7B) — fast, good structured generation; full GPU"),
        ("mistral:7b",
         "Mistral (7B) — reliable classic; full GPU"),
        ("CUSTOM", "Enter a custom model name"),
    ],
    "iteration": [
        ("qwen2.5:7b",
         "Qwen 2.5 (7B) — good speed/quality balance for iteration"),
        ("llama3.1:8b",
         "Llama 3.1 (8B) — strong general purpose"),
        ("phi4-mini",
         "Phi-4 Mini (3.8B) — fast; good structured reasoning"),
        ("llama3.2:3b",
         "Llama 3.2 (3B) — very fast; lower quality"),
        ("CUSTOM", "Enter a custom model name"),
    ],
    "mini": [
        ("phi4-mini",
         "Phi-4 Mini (3.8B) — best small model for structured reasoning on Mac Mini"),
        ("llama3.2:3b",
         "Llama 3.2 (3B) — very fast"),
        ("qwen2.5:3b",
         "Qwen 2.5 (3B) — fast; decent structured output"),
        ("gemma2:2b",
         "Gemma 2 (2B) — smallest, fastest; weakest"),
        ("CUSTOM", "Enter a custom model name"),
    ],
}

TIER_ALIASES = {
    "1": "production", "production": "production", "prod": "production", "p": "production",
    "2": "iteration",  "iteration": "iteration",  "iter": "iteration",  "i": "iteration",
    "3": "mini",       "mini": "mini",            "fallback": "mini",   "m": "mini",
}

# =========================================================
# GLOBAL STATE
# =========================================================

MODEL = DEFAULT_MODEL
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

def llm_json(prompt, required_keys, label, retries=MAX_RETRIES):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = ollama.chat(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={
                    "temperature": TEMPERATURE,
                    "num_predict": NUM_PREDICT,
                },
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
- Each: a positive, evocative unifying concept — NOT a concatenation of the two components.
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

Reference: in the veridictory square (being, seeming, non-seeming, non-being), the four M2 terms are: revelation, paradox, subterfuge, deception. These are evocative single words, NOT concatenations of their components.

Rules:
- Each: single word or short phrase (1-2 words).
- Each: positive, evocative — a real concept, NOT "X Y Z" style concatenation.
- No duplicates.
- {exclusion_note}

Return JSON with exactly these keys:
{{"M21": "...", "M22": "...", "M23": "...", "M24": "..."}}
"""

PROMPT_L1_MIXED = """Greimas square inner semes:
- S1 = "{s1}", S2 = "{s2}", ~S2 = "{neg_s2}", ~S1 = "{neg_s1}"

Generate the two mixed terms — the joins of the contradictories of each axis:

Mixed1 = S1 + ~S1 = "{s1}" + "{neg_s1}"  (first axis, undifferentiated)
Mixed2 = S2 + ~S2 = "{s2}" + "{neg_s2}"  (second axis, undifferentiated)

These represent "belonging to this axis at all, without committing to a side."

Rules:
- Each: single word or short phrase.
- STRUCTURAL REQUIREMENT: each name MUST begin with "Mixed " (e.g. "Mixed being", "Mixed seeming"). This is required to distinguish mixed terms from M1 terms.
- No duplicates.
- {exclusion_note}

Return JSON with exactly:
{{"Mixed1": "...", "Mixed2": "..."}}
"""

# =========================================================
# PROMPTS — LAYER 2
# =========================================================

PROMPT_L2_INNER = """The concept "{parent_term}" is a seme in a Greimas square. You are now decomposing "{parent_term}" into its own internal Greimas square.

Identify the two contrary poles into which "{parent_term}" internally polarizes.

Examples:
- "being" internally polarizes into "substance" and "appearance"
- "freedom" internally polarizes into "negative freedom" and "positive freedom"
- "life" internally polarizes into "existence" and "vitality"

CRITICAL — the two poles MUST be contraries in the strict sense:
- A subject can have BOTH poles at once
- A subject can have NEITHER pole at once
- A subject can have ONE pole without the other

If the two poles are contradictories (exactly one must hold), they are WRONG. If they are near-synonyms, they are WRONG. If they are unrelated concepts, they are WRONG.

Both poles must be specific aspects of "{parent_term}", not separate concepts or successors of it.

Rules:
- S1_prime and S2_prime are the two positive internal poles.
- Neither may be a synonym of "{parent_term}" itself.
- {exclusion_note}

Return JSON with exactly:
{{"S1_prime": "...", "S2_prime": "..."}}
"""

PROMPT_L2_NEG_S2 = """The concept "{parent_term}" has been decomposed. Its two positive internal poles are:
- S1' = "{s1_prime}"
- S2' = "{s2_prime}"

Generate ~S2', the CONTRADICTORY of S2'.

Rules:
- ~S2' and S2' are contradictories: exactly one holds.
- Simple negation of S2' ("non-X" / "un-X") if no better term exists.
- Single word or short phrase.
- {exclusion_note}

Return JSON with exactly: {{"~S2_prime": "..."}}
"""

PROMPT_L2_ANTITHESIS = """The concept "{parent_term}" is being decomposed.
- S1' = "{s1_prime}"
- S2' = "{s2_prime}"
- ~S2' = "{neg_s2_prime}"

Generate ~S1', the CONTRADICTORY of S1'.

CRITICAL: ~S1' is NOT a simple negation of S1'. It is the ANTITHESIS of ~S2'.
Just as S2' is a contrary of S1', ~S1' is a contrary of ~S2'.

Derivation chain (internal to "{parent_term}"):
S1' --(antithesis)--> S2' --(negation)--> ~S2' --(antithesis)--> ~S1'

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

Generate the four M1 meta-terms of this internal square:

M11' (complex axis)  = S1' + S2'   = "{s1_prime}" + "{s2_prime}"
M12' (neutral axis)  = ~S1' + ~S2' = "{neg_s1_prime}" + "{neg_s2_prime}"
M13' (positive deixis) = S1' + ~S2' = "{s1_prime}" + "{neg_s2_prime}"
M14' (negative deixis) = S2' + ~S1' = "{s2_prime}" + "{neg_s1_prime}"

Rules:
- Each: single word or short phrase (1-2 words).
- Each: a positive, evocative unifying concept — NOT a concatenation.
- No duplicates.
- {exclusion_note}

Return JSON with exactly these keys:
{{"M11_prime": "...", "M12_prime": "...", "M13_prime": "...", "M14_prime": "..."}}
"""

PROMPT_L2_M2 = """Internal square of "{parent_term}":
- S1' = "{s1_prime}", S2' = "{s2_prime}"
- ~S2' = "{neg_s2_prime}", ~S1' = "{neg_s1_prime}"

Generate the four M2 meta-terms. Each contains exactly THREE of the four internal semes:

M21' = S1' + S2' + ~S2'  (omits ~S1')
M22' = S1' + S2' + ~S1'  (omits ~S2')
M23' = S1' + ~S1' + ~S2' (omits S2')
M24' = S2' + ~S1' + ~S2' (omits S1')

Rules:
- Each: single word or short phrase (1-2 words).
- Each: positive, evocative — a real concept, NOT "X Y Z" style concatenation.
- No duplicates.
- {exclusion_note}

Return JSON with exactly these keys:
{{"M21_prime": "...", "M22_prime": "...", "M23_prime": "...", "M24_prime": "..."}}
"""

PROMPT_L2_MIXED = """Internal square of "{parent_term}":
- S1' = "{s1_prime}", S2' = "{s2_prime}"
- ~S2' = "{neg_s2_prime}", ~S1' = "{neg_s1_prime}"

Generate the two mixed terms of the internal square:

Mixed1' = S1' + ~S1' = "{s1_prime}" + "{neg_s1_prime}"
Mixed2' = S2' + ~S2' = "{s2_prime}" + "{neg_s2_prime}"

Rules:
- STRUCTURAL REQUIREMENT: each name MUST begin with "Mixed ". This is required to distinguish mixed terms from M1 terms.
- No duplicates.
- {exclusion_note}

Return JSON with exactly:
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

# =========================================================
# MIXED TERM NORMALIZATION
# =========================================================

def fix_mixed_terms(square):
    """
    Ensure mixed terms do not collide with M1 terms.
    If a mixed term's normalized form equals an M1 term's normalized form,
    prefix it with "Mixed ".
    """
    m1_keys = ["M11", "M12", "M13", "M14"]
    m1_norms = {normalize(square[k]) for k in m1_keys if k in square}

    for slot in ["Mixed1", "Mixed2"]:
        if slot not in square:
            continue
        t = square[slot].strip()
        if normalize(t) in m1_norms:
            new_t = f"Mixed {t}"
            print(f"  [fix] {slot}='{t}' collides with an M1 term; "
                  f"renaming to '{new_t}'")
            square[slot] = new_t

    # Also ensure Mixed1 and Mixed2 are not identical to each other
    if (square.get("Mixed1") and square.get("Mixed2")
            and normalize(square["Mixed1"]) == normalize(square["Mixed2"])):
        square["Mixed2"] = f"Mixed {square['Mixed2']}"
        print(f"  [fix] Mixed2 renamed to '{square['Mixed2']}' (collision with Mixed1)")

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
                    ["pass", "explanation"],
                    f"check_unifier:{seme_role}")
                ok = bool(check.get("pass"))
                expl = check.get("explanation", "")
            except Exception:
                ok = True
                expl = "(could not validate)"

            print(f"\n  LLM proposal: parent = '{proposed}'")
            if expl:
                print(f"  Plausibility: {expl}")
            print(f"\n  This will be A's {seme_role}.")
            print("  Options: [c] Confirm / [o] Override / [r] Regenerate / [q] Abort")
            choice = input("  Choice [c/o/r/q]: ").strip().lower()

            if choice == "c":
                register_term(proposed, f"A:{seme_role}", seme_role)
                return proposed
            if choice == "o":
                override = input("  Enter your own parent term: ").strip()
                if not override:
                    continue
                if normalize(override) in exclude_norms:
                    print("  That is one of the internal poles; try again.")
                    continue
                register_term(override, f"A:{seme_role}", seme_role)
                return override
            if choice == "r":
                continue
            if choice == "q":
                print("Aborted by user.")
                sys.exit(0)
        else:
            print("  No proposal available.")
            choice = input("  [o] Override / [q] Abort: ").strip().lower()
            if choice == "o":
                override = input("  Enter your own parent term: ").strip()
                if override:
                    register_term(override, f"A:{seme_role}", seme_role)
                    return override
            sys.exit(0)

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

def generate_layer1_square(s1, s2):
    square = {"S1": s1, "S2": s2}
    s_norms = {normalize(s1), normalize(s2)}

    print("  ~S2")
    def p1(note):
        return PROMPT_L1_NEG_S2.format(s1=s1, s2=s2, exclusion_note=note)
    square["~S2"] = generate_with_uniqueness(
        p1, ["~S2"], "L1:~S2", exclude_norms=s_norms)["~S2"].strip()

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
            neg_s2=square["~S2"], neg_s1=square["~S1"],
            exclusion_note=note)
    m1 = generate_with_uniqueness(
        p3, ["M11", "M12", "M13", "M14"], "L1:M1", exclude_norms=inner_norms)
    for k in ["M11", "M12", "M13", "M14"]:
        square[k] = m1[k].strip()

    m1_norms = {normalize(square[k]) for k in ["M11", "M12", "M13", "M14"]}

    print("  M2 batch")
    def p4(note):
        return PROMPT_L1_M2.format(
            s1=square["S1"], s2=square["S2"],
            neg_s2=square["~S2"], neg_s1=square["~S1"],
            exclusion_note=note)
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
            neg_s2=square["~S2"], neg_s1=square["~S1"],
            exclusion_note=note)
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

def generate_layer2_complete(parent_term, parent_key, fixed_pair=None):
    label_prefix = f"L2[{parent_key}]"
    print(f"\n[L2] Decomposing {parent_key} = '{parent_term}'")

    square = {}

    if fixed_pair is not None:
        sq_s1, sq_s2 = fixed_pair
        print(f"  using fixed inner pair: S1'='{sq_s1}', S2'='{sq_s2}'")
        square["S1"] = sq_s1
        square["S2"] = sq_s2
    else:
        print("  inner pair (S1', S2')")
        def p1(note):
            return PROMPT_L2_INNER.format(parent_term=parent_term, exclusion_note=note)
        pair = generate_with_uniqueness(
            p1, ["S1_prime", "S2_prime"], f"{label_prefix}:inner")
        square["S1"] = pair["S1_prime"].strip()
        square["S2"] = pair["S2_prime"].strip()

    inner_norms = {normalize(square["S1"]), normalize(square["S2"])}

    print("  ~S2'")
    def p2(note):
        return PROMPT_L2_NEG_S2.format(
            parent_term=parent_term,
            s1_prime=square["S1"], s2_prime=square["S2"],
            exclusion_note=note)
    square["~S2"] = generate_with_uniqueness(
        p2, ["~S2_prime"], f"{label_prefix}:~S2",
        exclude_norms=inner_norms)["~S2_prime"].strip()

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
        p6, ["Mixed1_prime", "Mixed2_prime"],
        f"{label_prefix}:Mixed",
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

def validate_square(square, label):
    print(f"\n[validate] {label}")
    prompt = PROMPT_VALIDATE_SQUARE.format(
        s1=square["S1"], s2=square["S2"],
        neg_s2=square["~S2"], neg_s1=square["~S1"])
    try:
        return llm_json(prompt,
                        ["rule1", "rule2", "rule3", "rule4", "rule5", "rule6"],
                        f"validate:{label}")
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
# LATTICE BUILDER — REWRITTEN
# =========================================================

def build_lattice(square):
    """
    Build the B4 concept lattice from a square.
    Keyed by SLOT label, not by term string, so identical terms in
    different slots cannot corrupt edges.
    """
    s1 = square["S1"]; s2 = square["S2"]
    ns2 = square["~S2"]; ns1 = square["~S1"]
    m11 = square["M11"]; m12 = square["M12"]
    m13 = square["M13"]; m14 = square["M14"]
    m21 = square["M21"]; m22 = square["M22"]
    m23 = square["M23"]; m24 = square["M24"]
    mix1 = square["Mixed1"]; mix2 = square["Mixed2"]

    # Slot definitions (label, term, extent).
    inner_slots = [
        ("S1",  s1),
        ("S2",  s2),
        ("~S1", ns1),
        ("~S2", ns2),
    ]
    pair_slots = [
        ("M11",    m11, [s1, s2]),
        ("M12",    m12, [ns1, ns2]),
        ("M13",    m13, [s1, ns2]),
        ("M14",    m14, [s2, ns1]),
        ("Mixed1", mix1, [s1, ns1]),
        ("Mixed2", mix2, [s2, ns2]),
    ]
    m2_slots = [
        ("M21", m21, [s1, s2, ns2]),
        ("M22", m22, [s1, s2, ns1]),
        ("M23", m23, [ns1, ns2, s1]),
        ("M24", m24, [ns1, ns2, s2]),
    ]

    def m2_terms_containing(extent):
        out = []
        for _, m2_term, m2_ext in m2_slots:
            if all(x in m2_ext for x in extent):
                out.append(m2_term)
        return out

    concepts = []
    slot_to_id = {}
    cid = 0

    # Bottom
    slot_to_id["bottom"] = cid
    concepts.append({
        "id": cid, "label": "bottom", "short": "Bottom",
        "name": "Bottom — empty extent, all four M2 attributes",
        "extent": [], "intent": [term for _, term, _ in m2_slots],
    })
    cid += 1

    # Inner semes
    for slot, term in inner_slots:
        slot_to_id[slot] = cid
        concepts.append({
            "id": cid, "label": term, "short": slot,
            "name": f"{term} — inner term {slot}",
            "extent": [term],
            "intent": m2_terms_containing([term]),
        })
        cid += 1

    # Pairs (M1 + mixed)
    for slot, term, extent in pair_slots:
        slot_to_id[slot] = cid
        kind = "mixed" if slot.startswith("Mix") else "M1"
        concepts.append({
            "id": cid, "label": term,
            "short": f"{'Mix' if kind == 'mixed' else 'M1'}:{term}",
            "name": f"{term} — {kind} meta-term of {', '.join(extent)}",
            "extent": list(extent),
            "intent": m2_terms_containing(extent),
        })
        cid += 1

    # M2 terms
    for slot, term, extent in m2_slots:
        slot_to_id[slot] = cid
        concepts.append({
            "id": cid, "label": term, "short": f"M2:{term}",
            "name": f"{term} — M2 meta-term containing {', '.join(extent)}",
            "extent": list(extent),
            "intent": [term],
        })
        cid += 1

    # Top
    slot_to_id["top"] = cid
    concepts.append({
        "id": cid, "label": "top", "short": "Top",
        "name": "Top — all inner terms, no shared attribute",
        "extent": [t for _, t in inner_slots], "intent": [],
    })
    cid += 1

    # Edges (deduplicated by construction via edge_set)
    edges = []
    edge_set = set()

    def add_edge(upper_slot, lower_slot):
        u = slot_to_id[upper_slot]
        l = slot_to_id[lower_slot]
        key = (u, l)
        if key not in edge_set:
            edge_set.add(key)
            edges.append([u, l])

    # Top → each M2
    for m2_slot, _, _ in m2_slots:
        add_edge("top", m2_slot)

    # M2 → each pair whose extent is subset of M2's extent
    for m2_slot, _, m2_ext in m2_slots:
        for pair_slot, _, pair_ext in pair_slots:
            if all(x in m2_ext for x in pair_ext):
                add_edge(m2_slot, pair_slot)

    # Pair → each inner seme in its extent
    for pair_slot, _, pair_ext in pair_slots:
        for inner_slot, inner_term in inner_slots:
            if inner_term in pair_ext:
                add_edge(pair_slot, inner_slot)

    # Inner seme → bottom
    for inner_slot, _ in inner_slots:
        add_edge(inner_slot, "bottom")

    return {
        "concepts": concepts,
        "edges": edges,
        "inner_ids": {term: slot_to_id[slot] for slot, term in inner_slots},
        "top_id": slot_to_id["top"],
        "bottom_id": slot_to_id["bottom"],
    }

# =========================================================
# ARCHITECTONIC LATTICE
# =========================================================

def build_architectonic(layer1_lattice, layer2_data):
    concepts = []
    edges = []
    interface_map = {}

    A = layer1_lattice
    a_id_map = {}
    a_seme_ids = set(A["inner_ids"].values())
    a_bottom_id = A["bottom_id"]

    for c in A["concepts"]:
        if c["id"] in a_seme_ids:
            continue
        new_id = f"A:{c['id']}"
        new_c = dict(c)
        new_c["id"] = new_id
        new_c["layer"] = 1
        new_c["origin"] = "A"
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
        B = data["lattice"]
        b_id_maps[seme_key] = {}

        for c in B["concepts"]:
            if c["id"] == B["bottom_id"]:
                continue
            new_id = f"{seme_key}:{c['id']}"
            new_c = dict(c)
            new_c["id"] = new_id
            new_c["layer"] = 2
            new_c["origin"] = seme_key
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

    # A's internal edges (not touching semes or bottom)
    for edge in A["edges"]:
        from_id, to_id = edge
        if from_id in a_seme_ids or to_id in a_seme_ids:
            continue
        edges.append([a_id_map[from_id], a_id_map[to_id]])

    # A's edges that pointed to a seme now point to that seme's interface
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

    # B_i edges
    for seme_key, data in layer2_data.items():
        B = data["lattice"]
        b_bottom_old = B["bottom_id"]
        for edge in B["edges"]:
            from_id, to_id = edge
            if to_id == b_bottom_old:
                if from_id == B["top_id"]:
                    continue
                edges.append([b_id_maps[seme_key][from_id],
                              a_id_map[a_bottom_id]])
            elif from_id == b_bottom_old:
                continue
            else:
                edges.append([b_id_maps[seme_key][from_id],
                              b_id_maps[seme_key][to_id]])

    # Deduplicate edges
    dedup = []
    seen = set()
    for e in edges:
        key = (e[0], e[1])
        if key not in seen:
            seen.add(key)
            dedup.append(e)

    return {
        "concepts": concepts,
        "edges": dedup,
        "interface_map": interface_map,
        "a_id_map": a_id_map,
        "b_id_maps": b_id_maps,
    }

# =========================================================
# FULL RUN
# =========================================================

SEME_ROLES = ["S1", "S2", "~S1", "~S2"]

def run_full_generation(seed, seed_s1, seed_s2):
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
                a_s1 = infer_a_s2_from_s1(parent)
                a_s2 = parent
                register_term(a_s1, "A:S1", "S1")
            else:
                a_s2 = infer_a_s2_from_s1(parent)
                a_s1 = parent
                register_term(a_s1, "A:S1", "S1")
            register_term(a_s2, "A:S2", "S2")

        fixed_B = (seme_role, (seed_s1, seed_s2))

    print(f"\n[L1] Generating layer 1 (S1='{a_s1}', S2='{a_s2}')...")
    layer1_square = generate_layer1_square(a_s1, a_s2)
    layer1_lattice = build_lattice(layer1_square)

    for seme_key in SEME_ROLES:
        parent_term = layer1_square[seme_key]
        if fixed_B and fixed_B[0] == seme_key:
            b_data = generate_layer2_complete(
                parent_term, seme_key, fixed_pair=fixed_B[1])
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

    return {
        "seed": seed,
        "seed_s1": seed_s1,
        "seed_s2": seed_s2,
        "layer1_square": layer1_square,
        "layer1_lattice": layer1_lattice,
        "layer2": layer2,
        "validation": validation,
        "architectonic": architectonic,
    }

# =========================================================
# MODEL SELECTION UI
# =========================================================

def select_model(cli_model):
    """
    If --model was passed, use it and skip the menu.
    Otherwise, prompt the user for a tier and then a specific model.
    """
    if cli_model:
        return cli_model

    print()
    print("=" * 64)
    print("Model selection")
    print("=" * 64)
    print("  [1] production — larger models (7B-14B) for the PC with Vulkan GPU")
    print("  [2] iteration  — medium models for fast iteration")
    print("  [3] mini       — small models for Mac Mini fallback")
    tier_raw = input("Mode [1/2/3] (default: iteration): ").strip().lower()
    if not tier_raw:
        tier = "iteration"
    else:
        tier = TIER_ALIASES.get(tier_raw, "iteration")

    menu = MODEL_MENUS[tier]
    print(f"\n{tier.capitalize()} models:")
    for i, (name, desc) in enumerate(menu, 1):
        print(f"  [{i}] {name:<22s} — {desc}")

    sel = input(f"Select [1-{len(menu)}] (default: 1): ").strip()
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
        name = input("Enter model name: ").strip()
        if not name:
            name = menu[0][0]
            print(f"Empty; defaulting to {name}")
    return name

# =========================================================
# MAIN
# =========================================================

def main():
    global MODEL

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
        description="Architectonic Greimas generator (v3).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python greimas_architectonic.py
  python greimas_architectonic.py --s1 being --s2 seeming
  python greimas_architectonic.py --seed B1
  python greimas_architectonic.py --model phi4:14b --s1 life --s2 death
  python greimas_architectonic.py --list-models
""",
    )
    parser.add_argument("--seed", choices=["A1", "B1", "B2", "B3", "B4"],
                        help="Which square to seed (default: A1)")
    parser.add_argument("--s1", help="S1 of the seeded square")
    parser.add_argument("--s2", help="S2 of the seeded square")
    parser.add_argument("--output", "-o", default="architectonic.json",
                        help="Output JSON (default: architectonic.json)")
    parser.add_argument("--model", "-m", default=None,
                        help="Ollama model; skip the interactive menu if given")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip LLM validation steps")
    parser.add_argument("--skip-layer2", action="store_true",
                        help="Generate only layer 1")
    args = parser.parse_args()

    # Model selection
    MODEL = select_model(args.model)

    # Verify model exists locally (soft check)
    available = list_ollama_models()
    if available and MODEL not in available:
        base = MODEL.split(":")[0]
        matches = [m for m in available if m == MODEL or m.split(":")[0] == base]
        if matches:
            MODEL = matches[0]
            print(f"Note: using '{MODEL}' (matched from '{MODEL}')")
        else:
            print(f"\nNote: model '{MODEL}' not found in local Ollama.")
            print(f"Available: {', '.join(available[:10])}"
                  + (" …" if len(available) > 10 else ""))
            print(f"If using a remote Ollama instance, this is fine.")
            if input("Continue anyway? [y/N]: ").strip().lower() != "y":
                sys.exit(1)

    # Outer retry loop
    while True:
        USED_TERMS.clear()
        REGISTRATION_ORDER.clear()
        COLLISION_LOG.clear()

        seed = (args.seed or input(
            "\nWhich square to seed? [A1/B1/B2/B3/B4] (default A1): "
        ).strip().upper()) or "A1"
        if seed not in ["A1", "B1", "B2", "B3", "B4"]:
            print(f"Invalid seed: {seed}")
            sys.exit(1)

        default_s1 = args.s1
        default_s2 = args.s2

        if seed == "A1":
            prompt_s1 = "Enter S1 for A1 (first positive pole): "
            prompt_s2 = "Enter S2 for A1 (its contrary): "
        else:
            prompt_s1 = f"Enter S1 for {seed} (first positive internal pole): "
            prompt_s2 = f"Enter S2 for {seed} (second positive internal pole): "

        s1 = default_s1 or input(prompt_s1).strip()
        s2 = default_s2 or input(prompt_s2).strip()
        if not s1 or not s2:
            print("Error: both terms are required.")
            sys.exit(1)

        print("=" * 64)
        print("Architectonic Greimas Generator (v3)")
        print(f"  Seed:  {seed}")
        print(f"  S1:    '{s1}'")
        print(f"  S2:    '{s2}'")
        print(f"  Model: {MODEL}")
        print("=" * 64)

        try:
            result = run_full_generation(seed, s1, s2)
        except KeyboardInterrupt:
            print("\nInterrupted.")
            sys.exit(1)
        except Exception as e:
            print(f"\nGeneration failed: {e}")
            if input("Try again? [y/N]: ").strip().lower() == "y":
                args.s1 = None
                args.s2 = None
                continue
            sys.exit(1)

        # Report collisions
        if COLLISION_LOG:
            print(f"\n[collisions] {len(COLLISION_LOG)} unresolved term collision(s):")
            for c in COLLISION_LOG:
                print(f"  - {c['label']}.{c['key']} = '{c['term']}'")

        # Validation check
        all_ok = True
        failed_labels = []
        for label, vd in result["validation"].items():
            if not all_rules_pass(vd):
                all_ok = False
                failed_labels.append(label)

        if all_ok:
            print("\nAll square-level validations passed.")
        else:
            print(f"\nValidation failed for: {', '.join(failed_labels)}")

        if args.skip_validation:
            break

        if not all_ok:
            print("\nOptions:")
            print("  [w] Write output anyway")
            print("  [r] Retry generation")
            print("  [q] Abort without writing")
            choice = input("  Choice [w/r/q]: ").strip().lower()
            if choice == "w":
                break
            elif choice == "r":
                args.s1 = None
                args.s2 = None
                continue
            else:
                sys.exit(0)
        else:
            break

    # Write output
    output = {
        "metadata": {
            "seed": result["seed"],
            "seed_s1": result["seed_s1"],
            "seed_s2": result["seed_s2"],
            "model": MODEL,
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
