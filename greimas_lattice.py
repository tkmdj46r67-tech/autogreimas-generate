#!/usr/bin/env python3
"""
Greimas Square + Lattice Generator

Given two positive terms S1 and S2 (the two poles of a contrary pair),
this script uses an LLM to generate a complete Greimas square following
strict derivation rules, then builds the associated B4 concept lattice.

Outputs JSON.

Usage:
    python greimas_lattice.py --s1 "being" --s2 "seeming" -o result.json
    python greimas_lattice.py  # interactive mode

Requirements:
    ollama (pip install ollama)
    A running Ollama instance with a model like llama3.2:3b
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
NUM_PREDICT = 800
MAX_RETRIES = 3
MAX_UNIQUENESS_TRIES = 4

# =========================================================
# TERM REGISTRY (uniqueness enforcement)
# =========================================================

USED_TERMS = {}          # norm -> list of (surface, location, role)
REGISTRATION_ORDER = []  # list of (norm, surface)

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
    """Return True if term is already registered elsewhere."""
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
    """Extract a JSON object from LLM output."""
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
    """Call the LLM and parse a JSON response."""
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
# PROMPTS
# =========================================================

PROMPT_NEG_S2 = """You are given S2 = "{s2}" (the antithesis of S1 = "{s1}").

Generate ~S2, the CONTRADICTORY of S2.

Rules:
- ~S2 and S2 are contradictories. For any subject, exactly one of the two holds.
- ~S2 should be a simple negation of S2. If no better semantic equivalent exists, use "non-X" or "un-X".
- ~S2 must be a single word or a short phrase (1-2 words).
- ~S2 must NOT be a synonym of S2.
- ~S2 must NOT be a synonym of S1 = "{s1}".
- {exclusion_note}

Return ONLY a JSON object with exactly this key: {{"~S2": "..."}}
"""

PROMPT_ANTITHESIS = """You are given:
- S1 = "{s1}" (positive pole of the first axis)
- S2 = "{s2}" (positive pole of the second axis)
- ~S2 = "{neg_s2}" (contradictory of S2)

Generate ~S1, the CONTRADICTORY of S1.

CRITICAL RULE: ~S1 is NOT a simple negation of S1. Instead, ~S1 is the ANTITHESIS of ~S2.
This means you must find a contrary of ~S2, just as S2 is a contrary of S1.

The derivation chain is:
S1 --(antithesis)--> S2 --(negation)--> ~S2 --(antithesis)--> ~S1

Structural consequences:
- ~S1 and S1 are contradictories (exactly one holds).
- ~S1 and ~S2 are contraries (both can fail together, both can hold together).
- ~S1 implies S2.

Requirements:
- ~S1 must be a single word or a short phrase (1-2 words).
- ~S1 must NOT be a synonym of any of: S1, S2, ~S2.
- {exclusion_note}

Return ONLY a JSON object with exactly this key: {{"~S1": "..."}}
"""

PROMPT_M1 = """You are given the four inner terms of a Greimas square:
- S1 = "{s1}" (first positive pole)
- S2 = "{s2}" (second positive pole, the antithesis of S1)
- ~S2 = "{neg_s2}" (the negation of S2)
- ~S1 = "{neg_s1}" (the antithesis of ~S2)

Generate names for the four M1 meta-terms. Each M1 term is the union of two adjacent inner terms (one edge of the inner square).

M11 (complex axis): S1 + S2 = "{s1}" + "{s2}"
M12 (neutral axis): ~S1 + ~S2 = "{neg_s1}" + "{neg_s2}"
M13 (positive deixis): S1 + ~S2 = "{s1}" + "{neg_s2}"
M14 (negative deixis): S2 + ~S1 = "{s2}" + "{neg_s1}"

Rules:
- Each name should be a single word or a short phrase (1-2 words).
- Each name should be a positive, evocative term that captures the combination.
- No two names should be identical or near-identical.
- The names should NOT duplicate any of the inner terms.
- {exclusion_note}

Return ONLY a JSON object with exactly these keys:
{{"M11": "...", "M12": "...", "M13": "...", "M14": "..."}}
"""

PROMPT_M2 = """You are given the four inner terms of a Greimas square:
- S1 = "{s1}"
- S2 = "{s2}"
- ~S2 = "{neg_s2}"
- ~S1 = "{neg_s1}"

Generate names for the four M2 meta-terms. Each M2 term is the union of two overlapping M1 terms. Each M2 term contains exactly three of the four inner terms.

M21 = S1 + S2 + ~S2 (omits ~S1)
M22 = S1 + S2 + ~S1 (omits ~S2)
M23 = ~S1 + ~S2 + S1 (omits S2)
M24 = ~S1 + ~S2 + S2 (omits S1)

For guidance, in the classic veridictory square (being, seeming, non-seeming, non-being), the M2 terms are: revelation, paradox, subterfuge, deception.

Rules:
- Each name should be a single word or a short phrase (1-2 words).
- Each name should be a positive, evocative term.
- No two names should be identical or near-identical.
- The names should NOT duplicate any of the inner terms or the M1 terms.
- {exclusion_note}

Return ONLY a JSON object with exactly these keys:
{{"M21": "...", "M22": "...", "M23": "...", "M24": "..."}}
"""

PROMPT_MIXED = """You are given the four inner terms of a Greimas square:
- S1 = "{s1}"
- S2 = "{s2}"
- ~S2 = "{neg_s2}"
- ~S1 = "{neg_s1}"

Generate names for the two mixed terms. Each mixed term is the join of the two contradictories of one axis.

Mixed1 = S1 ⊔ ~S1 = "{s1}" + "{neg_s1}" (the first axis, undifferentiated)
Mixed2 = S2 ⊔ ~S2 = "{s2}" + "{neg_s2}" (the second axis, undifferentiated)

These represent "either the positive or the negative pole of this axis, without commitment."

Rules:
- Each name should be a single word or a short phrase (1-2 words).
- A good pattern is "Mixed [axis name]" — e.g., "Mixed meaning" or "Mixed saying".
- No two names should be identical or near-identical.
- The names should NOT duplicate any of the inner terms, M1 terms, or M2 terms.
- {exclusion_note}

Return ONLY a JSON object with exactly these keys:
{{"Mixed1": "...", "Mixed2": "..."}}
"""

PROMPT_VALIDATE = """Check the following Greimas square for consistency:

S1 = "{s1}"
S2 = "{s2}"
~S2 = "{neg_s2}"
~S1 = "{neg_s1}"

Check each rule:

Rule 1: S1 and S2 are contraries (both can fail together, both can hold together, one can hold while the other fails).
Rule 2: S1 and ~S1 are contradictories (exactly one holds).
Rule 3: S2 and ~S2 are contradictories (exactly one holds).
Rule 4: ~S2 implies S1.
Rule 5: ~S1 implies S2.
Rule 6: ~S1 and ~S2 are contraries.

For each rule, answer "yes" or "no" and give a one-sentence explanation.

Return ONLY a JSON object with exactly these keys:
"rule1", "rule2", "rule3", "rule4", "rule5", "rule6"
Each containing {{"pass": true/false, "explanation": "..."}}.
"""

# =========================================================
# GENERATION PIPELINE
# =========================================================

def exclusion_block(exclude_norms=None, extra_terms=None):
    terms = all_exclusion_terms(exclude_norms=exclude_norms, extra_terms=extra_terms)
    if not terms:
        return ""
    return (
        "UNIQUENESS CONSTRAINT: Every term you output must be a NEW word that does not "
        "appear elsewhere. The following terms have already been used and MUST NOT "
        "appear in your response (as-is or as a close variant): " + ", ".join(terms) + "."
    )

def generate_with_uniqueness(prompt_fn, required_keys, label,
                             exclude_norms=None, check_keys=None,
                             max_tries=MAX_UNIQUENESS_TRIES):
    """
    Generate a group of terms with uniqueness enforcement.
    prompt_fn(extra_exclusions) -> prompt string
    Returns dict of generated terms (already registered).
    """
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
            print(f"  ✗ {label} attempt {attempt}: {e}")
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
            print(f"  ⚠ {label}.{k}='{val}' collides; forcing exclusion")
            forced.add(val)
        last_data = data
    print(f"  ⚠ {label}: accepting possible duplicates after {max_tries} tries")
    if last_data:
        for k in required_keys:
            register_term(last_data[k].strip(), label, k)
    return last_data

def generate_square(s1, s2):
    """Generate the full Greimas square given S1 and S2."""
    USED_TERMS.clear()
    REGISTRATION_ORDER.clear()

    register_term(s1, "inner", "S1")
    register_term(s2, "inner", "S2")

    square = {"S1": s1, "S2": s2}
    s_norms = {normalize(s1), normalize(s2)}

    # --- Step 1: ~S2 ---
    print("\n[1/5] Generating ~S2 (negation of S2)...")
    def prompt_neg(extra):
        return PROMPT_NEG_S2.format(
            s1=s1, s2=s2,
            exclusion_note=exclusion_block(exclude_norms=s_norms, extra_terms=[extra] if extra else None),
        )
    data = generate_with_uniqueness(
        prompt_neg, ["~S2"], "~S2",
        exclude_norms=s_norms, check_keys=["~S2"],
    )
    square["~S2"] = data["~S2"].strip()
    print(f"  ~S2 = '{square['~S2']}'")

    # --- Step 2: ~S1 ---
    print("\n[2/5] Generating ~S1 (antithesis of ~S2)...")
    def prompt_anti(extra):
        return PROMPT_ANTITHESIS.format(
            s1=s1, s2=s2, neg_s2=square["~S2"],
            exclusion_note=exclusion_block(exclude_norms=s_norms, extra_terms=[extra] if extra else None),
        )
    data = generate_with_uniqueness(
        prompt_anti, ["~S1"], "~S1",
        exclude_norms=s_norms, check_keys=["~S1"],
    )
    square["~S1"] = data["~S1"].strip()
    print(f"  ~S1 = '{square['~S1']}'")

    inner_norms = {normalize(square[k]) for k in ["S1", "S2", "~S2", "~S1"]}

    # --- Step 3: M1 ---
    print("\n[3/5] Generating M1 meta-terms...")
    def prompt_m1(extra):
        return PROMPT_M1.format(
            s1=square["S1"], s2=square["S2"],
            neg_s2=square["~S2"], neg_s1=square["~S1"],
            exclusion_note=exclusion_block(exclude_norms=inner_norms, extra_terms=[extra] if extra else None),
        )
    m1_keys = ["M11", "M12", "M13", "M14"]
    data = generate_with_uniqueness(
        prompt_m1, m1_keys, "M1",
        exclude_norms=inner_norms, check_keys=m1_keys,
    )
    for k in m1_keys:
        square[k] = data[k].strip()
        print(f"  {k} = '{square[k]}'")

    m1_norms = {normalize(square[k]) for k in m1_keys}

    # --- Step 4: M2 ---
    print("\n[4/5] Generating M2 meta-terms...")
    def prompt_m2(extra):
        return PROMPT_M2.format(
            s1=square["S1"], s2=square["S2"],
            neg_s2=square["~S2"], neg_s1=square["~S1"],
            exclusion_note=exclusion_block(exclude_norms=inner_norms | m1_norms,
                                           extra_terms=[extra] if extra else None),
        )
    m2_keys = ["M21", "M22", "M23", "M24"]
    data = generate_with_uniqueness(
        prompt_m2, m2_keys, "M2",
        exclude_norms=inner_norms | m1_norms, check_keys=m2_keys,
    )
    for k in m2_keys:
        square[k] = data[k].strip()
        print(f"  {k} = '{square[k]}'")

    m2_norms = {normalize(square[k]) for k in m2_keys}

    # --- Step 5: Mixed ---
    print("\n[5/5] Generating mixed terms...")
    def prompt_mix(extra):
        return PROMPT_MIXED.format(
            s1=square["S1"], s2=square["S2"],
            neg_s2=square["~S2"], neg_s1=square["~S1"],
            exclusion_note=exclusion_block(exclude_norms=inner_norms | m1_norms | m2_norms,
                                           extra_terms=[extra] if extra else None),
        )
    mix_keys = ["Mixed1", "Mixed2"]
    data = generate_with_uniqueness(
        prompt_mix, mix_keys, "Mixed",
        exclude_norms=inner_norms | m1_norms | m2_norms, check_keys=mix_keys,
    )
    for k in mix_keys:
        square[k] = data[k].strip()
        print(f"  {k} = '{square[k]}'")

    return square

def validate_square(square):
    """Ask the LLM to validate the square."""
    print("\n[validate] Checking square consistency...")
    prompt = PROMPT_VALIDATE.format(
        s1=square["S1"], s2=square["S2"],
        neg_s2=square["~S2"], neg_s1=square["~S1"],
    )
    try:
        data = llm_json(
            prompt,
            ["rule1", "rule2", "rule3", "rule4", "rule5", "rule6"],
            "validate",
        )
        return data
    except Exception as e:
        print(f"  validation failed: {e}")
        return {f"rule{i}": {"pass": False, "explanation": "validation error"}
                for i in range(1, 7)}

# =========================================================
# LATTICE BUILDER
# =========================================================

def build_lattice(square):
    """Build the B4 concept lattice from a complete Greimas square."""
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
        m21: [s1, s2, ns2],   # omits ns1
        m22: [s1, s2, ns1],   # omits ns2
        m23: [ns1, ns2, s1],  # omits s2
        m24: [ns1, ns2, s2],  # omits s1
    }

    inner_intents = {t: [] for t in inner}
    for m2, ext in m2_extents.items():
        for t in ext:
            inner_intents[t].append(m2)

    m1_terms = [m11, m12, m13, m14]
    m1_extents = {
        m11: [s1, s2],
        m12: [ns1, ns2],
        m13: [s1, ns2],
        m14: [s2, ns1],
    }
    m1_intents = {t: [] for t in m1_terms}
    for m1, ext in m1_extents.items():
        for m2, m2ext in m2_extents.items():
            if all(t in m2ext for t in ext):
                m1_intents[m1].append(m2)

    mixed_terms = [mix1, mix2]
    mixed_extents = {
        mix1: [s1, ns1],
        mix2: [s2, ns2],
    }
    mixed_intents = {t: [] for t in mixed_terms}
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
            "name": name, "extent": extent, "intent": intent,
        })
        cid += 1
        return cid - 1

    bottom_id = add("⊥", "Bottom",
                    "Bottom — empty extent, all four M2 attributes",
                    [], list(m2_terms))

    inner_ids = {}
    inner_short = {s1: "S1", s2: "S2", ns2: "~S2", ns1: "~S1"}
    for t in inner:
        inner_ids[t] = add(t, inner_short[t],
                           f"{t} — the inner term {inner_short[t]}",
                           [t], list(inner_intents[t]))

    pair_ids = {}
    for t in m1_terms:
        pair_ids[t] = add(t, f"M1:{t}",
                          f"{t} — the M1 meta-term combining {', '.join(m1_extents[t])}",
                          list(m1_extents[t]), list(m1_intents[t]))
    for t in mixed_terms:
        pair_ids[t] = add(t, f"Mix:{t}",
                          f"{t} — the mixed term combining {', '.join(mixed_extents[t])}",
                          list(mixed_extents[t]), list(mixed_intents[t]))

    m2_ids = {}
    for t in m2_terms:
        m2_ids[t] = add(t, f"M2:{t}",
                        f"{t} — the M2 meta-term containing {', '.join(m2_extents[t])}",
                        list(m2_extents[t]), [t])

    top_id = add("⊤", "Top",
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

    return {"concepts": concepts, "edges": edges}

# =========================================================
# MAIN
# =========================================================

def main():
    global MODEL
    parser = argparse.ArgumentParser(
        description="Generate a Greimas square and its B4 concept lattice."
    )
    parser.add_argument("--s1", help="First positive term (S1)")
    parser.add_argument("--s2", help="Second positive term (S2, the antithesis of S1)")
    parser.add_argument("--output", "-o", default="greimas_lattice.json",
                        help="Output JSON file (default: greimas_lattice.json)")
    parser.add_argument("--model", default=MODEL,
                        help=f"Ollama model name (default: {MODEL})")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip the LLM validation step")
    parser.add_argument("--allow-duplicates", action="store_true",
                        help="Allow duplicate terms (default: strict uniqueness)")
    args = parser.parse_args()

    MODEL = args.model

    s1 = args.s1
    s2 = args.s2
    if not s1:
        s1 = input("Enter S1 (first positive term): ").strip()
    if not s2:
        s2 = input("Enter S2 (antithesis of S1): ").strip()
    if not s1 or not s2:
        print("Error: S1 and S2 are required.")
        sys.exit(1)

    print("=" * 64)
    print("Greimas Square + Lattice Generator")
    print(f"  S1 = '{s1}'")
    print(f"  S2 = '{s2}'")
    print(f"  Model: {MODEL}")
    print("=" * 64)

    square = generate_square(s1, s2)

    validation = {}
    if not args.skip_validation:
        validation = validate_square(square)
        print("\nValidation results:")
        for k in sorted(validation.keys()):
            v = validation[k]
            status = "✓" if v.get("pass") else "✗"
            print(f"  {status} {k}: {v.get('explanation', '')}")

    print("\nBuilding lattice...")
    lattice = build_lattice(square)
    print(f"  {len(lattice['concepts'])} concepts, {len(lattice['edges'])} edges")

    output = {
        "metadata": {
            "s1_input": s1,
            "s2_input": s2,
            "model": MODEL,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "square": square,
        "validation": validation,
        "lattice": lattice,
    }

    Path(args.output).write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n✓ Wrote {args.output}")

if __name__ == "__main__":
    main()
