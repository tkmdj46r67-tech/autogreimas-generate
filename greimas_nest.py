#!/usr/bin/env python3
"""
Architectonic Greimas Generator — Version 2

User picks a starting square (A1 on layer 1, or B1-B4 on layer 2) and
enters the two positive poles of that square. The script then generates
the full architectonic: layer 1 (A) + four layer-2 squares (B1-B4) +
the architectonic lattice C.

If a B_i is seeded, the LLM proposes the parent seme of A's corresponding
axis, and the user confirms or overrides before generation proceeds.

Usage:
    python greimas_architectonic.py --s1 being --s2 seeming
    python greimas_architectonic.py --seed B1
    python greimas_architectonic.py --seed A1 --s1 life --s2 death -m phi4-mini

Requirements:
    pip install ollama
    Running Ollama instance.
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

MODEL = "llama3.2:3b"
TEMPERATURE = 0.3
NUM_PREDICT = 1200
MAX_RETRIES = 3
MAX_UNIQUENESS_TRIES = 4

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
# GLOBAL TERM REGISTRY
# =========================================================

USED_TERMS = {}
REGISTRATION_ORDER = []

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
# EXCLUSION BLOCK
# =========================================================

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
# PROMPTS — REVERSE FIBRATION (V2)
# =========================================================

PROMPT_REVERSE_FIBRATION = """Given two terms that are the two positive internal poles of some concept:
- Internal S1' = "{x}"
- Internal S2' = "{y}"

These two terms are the two positive contraries into which a parent concept internally polarizes. For example:
- Internal poles "substance" and "appearance" → parent concept = "being"
- Internal poles "existence" and "vitality" → parent concept = "life"
- Internal poles "negative freedom" and "positive freedom" → parent concept = "freedom"
- Internal poles "showing" and "shown" → parent concept = "seeming"

Propose the parent concept for the pair "{x}" and "{y}". This parent concept is a seme in a higher-order Greimas square; the two internal poles are its internal decomposition.

Requirements:
- Single word or short phrase (1-3 words).
- The parent must be a genuine unifying concept: both "{x}" and "{y}" are aspects of it.
- The parent must not be a synonym of "{x}" or "{y}".
- The parent must be a positive term, not a negation.
- {exclusion_note}

Return JSON with exactly: {{"parent": "..."}}
"""

PROMPT_CONTINUE_A_FROM_S1 = """A Greimas square A is being constructed around a seme.

- A's S1 = "{s1}"

Generate A's S2, the contrary of S1 (the second positive pole of A).

Rules:
- S2 must be a positive term, not a negation.
- S1 and S2 must be contraries: both can hold together, both can fail together, one can hold alone.
- S2 must be the same kind of thing as S1 (if S1 is a substance, S2 is a substance; if a process, a process).
- S2 must not be a synonym of S1 or of any of the internal poles.
- {exclusion_note}

Return JSON with exactly: {{"S2": "..."}}
"""

PROMPT_CHECK_UNIFIER = """Is "{parent}" a genuine unifying concept for the pair ("{x}", "{y}")?

Meaning: are "{x}" and "{y}" both genuine internal aspects of "{parent}"? Is "{parent}" a coherent concept that the two poles jointly elaborate?

Answer "yes" or "no" with one sentence of explanation.

Return JSON with exactly: {{"pass": true/false, "explanation": "..."}}
"""

# =========================================================
# PROMPTS — LAYER 1
# =========================================================

PROMPT_L1_NEG_S2 = """You are given S2 = "{s2}" (the antithesis of S1 = "{s1}").

Generate ~S2, the CONTRADICTORY of S2.

Rules:
- ~S2 and S2 are contradictories: for any subject, exactly one of the two holds.
- ~S2 should be a simple negation of S2 ("non-X" or "un-X" if no better term exists).
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
- S1 = "{s1}"
- S2 = "{s2}"
- ~S2 = "{neg_s2}"
- ~S1 = "{neg_s1}"

Generate the four M1 meta-terms. Each M1 is the union of two adjacent inner semes:

M11 (complex axis)  = S1 + S2   = "{s1}" + "{s2}"
M12 (neutral axis)  = ~S1 + ~S2 = "{neg_s1}" + "{neg_s2}"
M13 (positive deixis) = S1 + ~S2 = "{s1}" + "{neg_s2}"
M14 (negative deixis) = S2 + ~S1 = "{s2}" + "{neg_s1}"

Rules:
- Each: single word or short phrase (1-2 words).
- Each: a positive, evocative unifying concept.
- No duplicates or near-duplicates.
- Must not duplicate any inner seme.
- {exclusion_note}

Return JSON with exactly these keys:
{{"M11": "...", "M12": "...", "M13": "...", "M14": "..."}}
"""

PROMPT_L1_M2 = """Greimas square inner semes:
- S1 = "{s1}", S2 = "{s2}", ~S2 = "{neg_s2}", ~S1 = "{neg_s1}"

Generate the four M2 meta-terms. Each M2 contains exactly THREE of the four semes:

M21 = S1 + S2 + ~S2  (omits ~S1)
M22 = S1 + S2 + ~S1  (omits ~S2)
M23 = S1 + ~S1 + ~S2 (omits S2)
M24 = S2 + ~S1 + ~S2 (omits S1)

Reference: in the veridictory square (being, seeming, non-seeming, non-being), the M2 terms are: revelation, paradox, subterfuge, deception.

Rules:
- Each: single word or short phrase (1-2 words).
- Each: positive, evocative.
- No duplicates.
- Must not duplicate inner semes or M1 terms.
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
- Each: single word or short phrase. A good pattern is "Mixed [axis name]".
- No duplicates.
- Must not duplicate inner semes or meta-terms.
- {exclusion_note}

Return JSON with exactly:
{{"Mixed1": "...", "Mixed2": "..."}}
"""

# =========================================================
# PROMPTS — LAYER 2 (INTERNAL DECOMPOSITION)
# =========================================================

PROMPT_L2_INNER = """The concept "{parent_term}" is a seme in a Greimas square. You are now decomposing "{parent_term}" into its own internal Greimas square.

Identify the two contrary poles into which "{parent_term}" internally polarizes. These are the two INTERNAL contraries of "{parent_term}".

Examples:
- "being" internally polarizes into "substance" (underlying reality) and "appearance" (manifest surface).
- "freedom" internally polarizes into "negative freedom" (absence of constraint) and "positive freedom" (capacity for self-determination).
- "life" internally polarizes into "existence" (mere being alive) and "vitality" (intensity of life).

Requirements:
- S1_prime and S2_prime are the two positive internal poles.
- Both must be specific aspects of "{parent_term}", not separate concepts.
- Together they must exhaust "{parent_term}" — nothing about it should remain outside their union.
- They must be contraries: both can hold, both can fail, one can hold alone.
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
- ~S2' and S2' are contradictories: exactly one holds for any subject.
- Simple negation of S2' ("non-X" / "un-X") if no better term exists.
- Single word or short phrase.
- Must not duplicate any existing term in the architectonic.
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
- Must not duplicate any existing term.
- {exclusion_note}

Return JSON with exactly: {{"~S1_prime": "..."}}
"""

PROMPT_L2_M1 = """Internal square of "{parent_term}":
- S1' = "{s1_prime}"
- S2' = "{s2_prime}"
- ~S2' = "{neg_s2_prime}"
- ~S1' = "{neg_s1_prime}"

Generate the four M1 meta-terms of this internal square:

M11' (complex axis)  = S1' + S2'   = "{s1_prime}" + "{s2_prime}"
M12' (neutral axis)  = ~S1' + ~S2' = "{neg_s1_prime}" + "{neg_s2_prime}"
M13' (positive deixis) = S1' + ~S2' = "{s1_prime}" + "{neg_s2_prime}"
M14' (negative deixis) = S2' + ~S1' = "{s2_prime}" + "{neg_s1_prime}"

Rules:
- Each: single word or short phrase (1-2 words).
- Each: positive, evocative unifying concept.
- No duplicates.
- Must not duplicate any existing term in the architectonic.
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
- Each: positive, evocative.
- No duplicates.
- Must not duplicate any existing term in the architectonic.
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
- Pattern: "Mixed [axis name]".
- No duplicates.
- Must not duplicate any existing term.
- {exclusion_note}

Return JSON with exactly:
{{"Mixed1_prime": "...", "Mixed2_prime": "..."}}
"""

# =========================================================
# VALIDATION PROMPT
# =========================================================

PROMPT_VALIDATE_SQUARE = """Check consistency of a Greimas square:

- S1 = "{s1}", S2 = "{s2}"
- ~S2 = "{neg_s2}", ~S1 = "{neg_s1}"

Check each:
Rule 1: S1 and S2 are contraries.
Rule 2: S1 and ~S1 are contradictories.
Rule 3: S2 and ~S2 are contradictories.
Rule 4: ~S2 implies S1.
Rule 5: ~S1 implies S2.
Rule 6: ~S1 and ~S2 are contraries.

For each: "yes" or "no" plus one sentence.

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
    print(f"  ! {label}: accepting after {max_tries} tries")
    if last_data:
        for k in required_keys:
            register_term(last_data[k].strip(), label, k)
    return last_data

# =========================================================
# REVERSE FIBRATION (V2)
# =========================================================

def infer_parent_concept(x, y, seme_role):
    """
    Given the two internal positive poles of a seme, infer the parent
    seme. Show the LLM proposal to the user; allow confirm/override/retry.
    Returns the confirmed parent term.
    """
    exclude_norms = {normalize(x), normalize(y)}
    while True:
        print(f"\n[reverse-fibration] Inferring parent seme for ({x}, {y}) → A's {seme_role}")
        def p(note):
            return PROMPT_REVERSE_FIBRATION.format(
                x=x, y=y, exclusion_note=note)
        try:
            data = llm_json(p(""), ["parent"], f"reverse:{seme_role}")
            proposed = data["parent"].strip()
        except Exception as e:
            print(f"  LLM failed: {e}")
            proposed = None

        if proposed:
            # Validate the proposal.
            try:
                check = llm_json(
                    PROMPT_CHECK_UNIFIER.format(
                        parent=proposed, x=x, y=y),
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
            print("\n  This will be A's " + seme_role + ".")
            print("  Options:")
            print("    [c] Confirm and continue")
            print("    [o] Override with your own term")
            print("    [r] Regenerate (ask LLM again)")
            print("    [q] Abort")
            choice = input("  Choice [c/o/r/q]: ").strip().lower()

            if choice == "c":
                register_term(proposed, f"A:{seme_role}", seme_role)
                return proposed
            if choice == "o":
                override = input("  Enter your own parent term: ").strip()
                if not override:
                    print("  Empty; ignoring override.")
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
    """Given A's S1, generate A's S2."""
    print(f"\n[generate] A's S2 from S1 = '{s1}'")
    def p(note):
        return PROMPT_CONTINUE_A_FROM_S1.format(s1=s1, exclusion_note=note)
    data = generate_with_uniqueness(
        p, ["S2"], f"A:S2", exclude_norms={normalize(s1)})
    return data["S2"].strip()

# =========================================================
# LAYER 1 GENERATION
# =========================================================

def generate_layer1_square(s1, s2):
    """Generate layer-1 square given both S1 and S2."""
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
        p3, ["M11", "M12", "M13", "M14"], "L1:M1",
        exclude_norms=inner_norms)
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

    return square

def complete_layer1_from_s1(s1):
    """
    Given only A's S1, generate A's S2 and everything else.
    Used when a B_i has been seeded and the parent concept has been inferred.
    """
    s2 = infer_a_s2_from_s1(s1)
    return generate_layer1_square(s1, s2)

# =========================================================
# LAYER 2 GENERATION
# =========================================================

def generate_layer2_complete(parent_term, parent_key, fixed_pair=None):
    """
    Generate the internal square of parent_term.
    If fixed_pair is given as (S1', S2'), those two are used as the
    positive inner poles; only the remaining terms are generated.
    """
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
            return PROMPT_L2_INNER.format(
                parent_term=parent_term, exclusion_note=note)
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
            neg_s2_prime=square["~S2"],
            exclusion_note=note)
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
        f"{label_prefix}:M2",
        exclude_norms=full_inner_norms | m1_norms)
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

    inner = [s1, s2, ns2, ns1]
    m2_terms = [m21, m22, m23, m24]

    m2_extents = {
        m21: [s1, s2, ns2],
        m22: [s1, s2, ns1],
        m23: [ns1, ns2, s1],
        m24: [ns1, ns2, s2],
    }

    inner_intents = {t: [] for t in inner}
    for m2, ext in m2_extents.items():
        for t in ext:
            inner_intents[t].append(m2)

    m1_terms = [m11, m12, m13, m14]
    m1_extents = {
        m11: [s1, s2], m12: [ns1, ns2],
        m13: [s1, ns2], m14: [s2, ns1],
    }
    m1_intents = {t: [] for t in m1_terms}
    for m1, ext in m1_extents.items():
        for m2, m2ext in m2_extents.items():
            if all(t in m2ext for t in ext):
                m1_intents[m1].append(m2)

    mixed_extents = {mix1: [s1, ns1], mix2: [s2, ns2]}
    mixed_intents = {t: [] for t in [mix1, mix2]}
    for mix, ext in mixed_extents.items():
        for m2, m2ext in m2_extents.items():
            if all(t in m2ext for t in ext):
                mixed_intents[mix].append(m2)

    concepts = []
    cid = 0

    def add(label, short, name, extent, intent):
        nonlocal cid
        concepts.append({
            "id": cid, "label": label, "short": short,
            "name": name, "extent": list(extent), "intent": list(intent),
        })
        cid += 1
        return cid - 1

    bottom_id = add("bottom", "Bottom",
                    "Bottom — empty extent, all four M2 attributes",
                    [], list(m2_terms))

    inner_ids = {}
    inner_short = {s1: "S1", s2: "S2", ns2: "~S2", ns1: "~S1"}
    for t in inner:
        inner_ids[t] = add(t, inner_short[t],
                           f"{t} — inner term {inner_short[t]}",
                           [t], list(inner_intents[t]))

    pair_ids = {}
    for t in m1_terms:
        pair_ids[t] = add(t, f"M1:{t}",
                          f"{t} — M1 meta-term of {', '.join(m1_extents[t])}",
                          list(m1_extents[t]), list(m1_intents[t]))
    for t in [mix1, mix2]:
        pair_ids[t] = add(t, f"Mix:{t}",
                          f"{t} — mixed term of {', '.join(mixed_extents[t])}",
                          list(mixed_extents[t]), list(mixed_intents[t]))

    m2_ids = {}
    for t in m2_terms:
        m2_ids[t] = add(t, f"M2:{t}",
                        f"{t} — M2 meta-term containing {', '.join(m2_extents[t])}",
                        list(m2_extents[t]), [t])

    top_id = add("top", "Top",
                 "Top — all inner terms, no shared attribute",
                 list(inner), [])

    edges = []
    for t in m2_terms:
        edges.append([top_id, m2_ids[t]])
    for m2, ext in m2_extents.items():
        for pair, pext in list(m1_extents.items()) + list(mixed_extents.items()):
            if all(t in ext for t in pext):
                edges.append([m2_ids[m2], pair_ids[pair]])
    for pair, ext in list(m1_extents.items()) + list(mixed_extents.items()):
        for t in ext:
            edges.append([pair_ids[pair], inner_ids[t]])
    for t in inner:
        edges.append([inner_ids[t], bottom_id])

    return {
        "concepts": concepts,
        "edges": edges,
        "inner_ids": inner_ids,
        "top_id": top_id,
        "bottom_id": bottom_id,
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
    b_bottom_old_ids = {}
    b_top_old_ids = {}

    for seme_key, data in layer2_data.items():
        B = data["lattice"]
        b_id_maps[seme_key] = {}
        b_bottom_old_ids[seme_key] = B["bottom_id"]
        b_top_old_ids[seme_key] = B["top_id"]

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
                new_c["short"] = data['parent_term']
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
        B = data["lattice"]
        b_bottom_old = B["bottom_id"]
        for edge in B["edges"]:
            from_id, to_id = edge
            if to_id == b_bottom_old:
                if from_id == B["top_id"]:
                    continue
                edges.append([b_id_maps[seme_key][from_id], a_id_map[a_bottom_id]])
            elif from_id == b_bottom_old:
                continue
            else:
                edges.append([b_id_maps[seme_key][from_id],
                              b_id_maps[seme_key][to_id]])

    return {
        "concepts": concepts,
        "edges": edges,
        "interface_map": interface_map,
        "a_id_map": a_id_map,
        "b_id_maps": b_id_maps,
    }

# =========================================================
# FULL RUN
# =========================================================

SEME_ROLES = ["S1", "S2", "~S1", "~S2"]

def run_full_generation(seed, seed_s1, seed_s2):
    """
    Run one full generation pass. Returns a dict with all outputs.
    Does NOT call USED_TERMS.clear() — caller handles registry reset.
    """
    layer2 = {}
    fixed_B = None  # (seme_key, (S1', S2'))

    # ---- Determine A's S1 and S2 ----
    if seed == "A1":
        a_s1 = seed_s1
        a_s2 = seed_s2
        register_term(a_s1, "A:S1", "S1")
        register_term(a_s2, "A:S2", "S2")
    else:
        # seed is B1, B2, B3, B4; infer A's S_i
        seme_role = seed[1:]  # "1" -> "S1", "2" -> "S2", etc.
        # Map seed "B1" -> seme role "S1", "B2" -> "S2",
        # "B3" -> "~S1", "B4" -> "~S2" (my chosen convention).
        role_map = {"1": "S1", "2": "S2", "3": "~S1", "4": "~S2"}
        seme_role = role_map[seed[1:]]
        # Register the two seed terms as B_i's inner poles
        b_label = f"L2[{seme_role}]"
        register_term(seed_s1, f"{b_label}:S1", "S1")
        register_term(seed_s2, f"{b_label}:S2", "S2")
        # Infer the parent seme (which becomes A's seme_role)
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
            # seme_role is ~S1 or ~S2; we still need A's S1 and S2.
            # Simplest: infer a plausible A's S1 by treating the parent as
            # the negation-side seme, then work forward.
            # For simplicity, we treat the parent as A's S1 or S2 in a
            # working approximation, then regenerate the other side.
            # (Full generality would require more inference steps.)
            if seme_role == "~S1":
                a_s1 = infer_a_s2_from_s1(parent)  # treat as S2, get contrary
                a_s2 = parent
                register_term(a_s1, "A:S1", "S1")
            else:  # "~S2"
                a_s2 = infer_a_s2_from_s1(parent)
                a_s1 = parent
                register_term(a_s1, "A:S1", "S1")
            register_term(a_s2, "A:S2", "S2")

        # Remember the fixed B_i
        fixed_B = (seme_role, (seed_s1, seed_s2))

    # ---- Layer 1 ----
    print(f"\n[L1] Generating layer 1 (S1='{a_s1}', S2='{a_s2}')...")
    layer1_square = generate_layer1_square(a_s1, a_s2)
    layer1_lattice = build_lattice(layer1_square)

    # ---- Layer 2 ----
    for seme_key in SEME_ROLES:
        parent_term = layer1_square[seme_key]
        if fixed_B and fixed_B[0] == seme_key:
            b_data = generate_layer2_complete(
                parent_term, seme_key, fixed_pair=fixed_B[1])
        else:
            b_data = generate_layer2_complete(parent_term, seme_key)
        b_data["lattice"] = build_lattice(b_data["square"])
        layer2[seme_key] = b_data

    # ---- Validation ----
    print("\n[validate] running square validations...")
    validation = {"layer_1": validate_square(layer1_square, "layer_1")}
    for seme_key, data in layer2.items():
        validation[f"layer_2_{seme_key}"] = validate_square(
            data["square"], f"layer_2_{seme_key}")

    # ---- Architectonic ----
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
        description="Architectonic Greimas generator (v2).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python greimas_architectonic.py --s1 being --s2 seeming
  python greimas_architectonic.py --seed B1
  python greimas_architectonic.py --seed A1 --s1 life --s2 death
  python greimas_architectonic.py --list-models
""",
    )
    parser.add_argument("--seed", choices=["A1", "B1", "B2", "B3", "B4"],
                        help="Which square to seed (default: A1)")
    parser.add_argument("--s1", help="S1 of the seeded square")
    parser.add_argument("--s2", help="S2 of the seeded square")
    parser.add_argument("--output", "-o", default="architectonic.json",
                        help="Output JSON (default: architectonic.json)")
    parser.add_argument("--model", "-m", default=MODEL,
                        help=f"Ollama model (default: {MODEL})")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip LLM validation steps")
    args = parser.parse_args()

    MODEL = args.model

    available = list_ollama_models()
    if available and MODEL not in available:
        base = MODEL.split(":")[0]
        matches = [m for m in available if m == MODEL or m.split(":")[0] == base]
        if matches:
            MODEL = matches[0]
            print(f"Note: using '{MODEL}' (matched from '{args.model}')")
        else:
            print(f"Warning: model '{MODEL}' not found.")
            print(f"Available: {', '.join(available)}")
            if input("Continue? [y/N]: ").strip().lower() != "y":
                sys.exit(1)

    # ---- Outer retry loop ----
    while True:
        # Reset registry for a fresh run.
        USED_TERMS.clear()
        REGISTRATION_ORDER.clear()

        # ---- Ask for seed ----
        seed = (args.seed or input(
            "Which square to seed? [A1/B1/B2/B3/B4] (default A1): "
        ).strip().upper()) or "A1"
        if seed not in ["A1", "B1", "B2", "B3", "B4"]:
            print(f"Invalid seed: {seed}")
            sys.exit(1)

        # ---- Ask for terms ----
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
        print("Architectonic Greimas Generator (v2)")
        print(f"  Seed:  {seed}")
        print(f"  S1:    '{s1}'")
        print(f"  S2:    '{s2}'")
        print(f"  Model: {MODEL}")
        print("=" * 64)

        # ---- Generation ----
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

        # ---- Global validation check ----
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

    # ---- Write output ----
    output = {
        "metadata": {
            "seed": result["seed"],
            "seed_s1": result["seed_s1"],
            "seed_s2": result["seed_s2"],
            "model": MODEL,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
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
