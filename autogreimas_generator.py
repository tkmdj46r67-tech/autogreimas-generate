#!/usr/bin/env python3
"""
Greimas Cascade Generator (v8)

L1: 16 diagrams. L1 D(n+1).S1 = L1 D(n).C2 (linear chain).
L2:  4 diagrams. S-terms taken from ~S2 of consecutive L1 diagrams.
L3:  1 diagram.  S-terms taken from  C2 of the four L2 diagrams.

Uniqueness strategy — a three-tier escalation for any diagram:
  1. Normal generation with uniqueness retries.
  2. Synonym substitution for still-colliding terms.
  3. Accept the duplicate (last resort), flagged in the output.

Model presets — chosen via CLI:
  --fast / -f          llama3.2:3b   (default)
  --quality / -q       llama3.1:8b
  --model <name>       explicit preset name
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
    print("Install with:  python3 -m pip install ollama")
    sys.exit(1)


# =========================================================
# MODEL PRESETS
# =========================================================
# Each preset bundles a model + tuned inference params.
# Edit this dict to add or swap presets.
MODEL_PRESETS = {
    "fast": {
        "model":       "llama3.2:3b",
        "temperature": 0.3,
        "num_predict": 800,
        "label":       "fast (llama3.2:3b)",
    },
    "quality": {
        "model":       "llama3.1:8b",
        "temperature": 0.3,
        "num_predict": 1000,
        "label":       "quality (llama3.1:8b)",
    },
}
DEFAULT_PRESET = "fast"

# These are populated by apply_preset() before any LLM call.
MODEL_NAME   = MODEL_PRESETS[DEFAULT_PRESET]["model"]
TEMPERATURE  = MODEL_PRESETS[DEFAULT_PRESET]["temperature"]
NUM_PREDICT  = MODEL_PRESETS[DEFAULT_PRESET]["num_predict"]


# =========================================================
# CONFIG (preset-independent)
# =========================================================
MAX_RETRIES           = 3    # retries for valid JSON
MAX_UNIQUENESS_TRIES  = 4    # normal attempts before synonym fallback
MAX_SYNONYM_TRIES     = 3    # synonym lookups per colliding term
MAX_EXCLUSION_TERMS   = 80   # cap on exclusion list size per prompt
STOP_TOKENS           = ["<|endoftext|>", "<|im_end|>", "</s>"]

KEYS_FULL    = ["S1", "S2", "~S1", "~S2", "M1", "M2", "M3", "M4", "C1", "C2", "C3", "C4"]
KEYS_PARTIAL = ["M1", "M2", "M3", "M4", "C1", "C2", "C3", "C4"]

# Status codes returned by diagram generators
STATUS_OK        = "ok"
STATUS_SYNONYM   = "synonym"
STATUS_DUPLICATE = "duplicate"


# =========================================================
# GLOBAL TERM REGISTRY
# =========================================================
USED_TERMS = {}            # norm -> list of (surface, location, role)
REGISTRATION_ORDER = []    # (norm, surface) in first-seen order


def normalize(term):
    if not term:
        return ""
    return re.sub(r'[^a-z0-9]+', '', term.lower())


def norm_duplicate(na, nb):
    if not na or not nb:
        return False
    if na == nb:
        return True
    if len(na) >= 4 and len(nb) >= 4:
        if na in nb and (len(nb) - len(na)) <= 2:
            return True
        if nb in na and (len(na) - len(nb)) <= 2:
            return True
    return False


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


def all_exclusion_terms(exclude_norms=None, extra_terms=None):
    exclude_norms = exclude_norms or set()
    extra_terms = set(extra_terms or ())

    pool = REGISTRATION_ORDER
    if MAX_EXCLUSION_TERMS and len(pool) > MAX_EXCLUSION_TERMS:
        pool = pool[-MAX_EXCLUSION_TERMS:]

    seen = set()
    out = []
    for term in extra_terms:
        n = normalize(term)
        if n and n not in exclude_norms and n not in seen:
            seen.add(n)
            out.append(term)
    for norm, surface in pool:
        if norm in exclude_norms or norm in seen:
            continue
        seen.add(norm)
        out.append(surface)
    return out


def term_collides(term, current_location):
    norm = normalize(term)
    if not norm:
        return True
    if norm in USED_TERMS:
        entries = USED_TERMS[norm]
        if any(e[1] != current_location for e in entries):
            return True
    for existing_norm in USED_TERMS:
        if existing_norm == norm:
            continue
        if norm_duplicate(norm, existing_norm):
            return True
    return False


def find_collisions(generated, current_location):
    collisions = {}
    seen = {}
    for role, term in generated.items():
        norm = normalize(term)
        if not norm:
            continue
        for prev_norm, prev_role in list(seen.items()):
            if norm_duplicate(norm, prev_norm):
                collisions[role] = ("internal", prev_role)
                break
        if role not in collisions:
            seen[norm] = role
    for role, term in generated.items():
        if role in collisions:
            continue
        norm = normalize(term)
        if not norm:
            continue
        if norm in USED_TERMS:
            entries = USED_TERMS[norm]
            ext = [e for e in entries if e[1] != current_location] or entries
            collisions[role] = ("external", f"{ext[0][1]}.{ext[0][2]}")
            continue
        for existing_norm, entries in USED_TERMS.items():
            if existing_norm == norm:
                continue
            if norm_duplicate(norm, existing_norm):
                ext = [e for e in entries if e[1] != current_location] or entries
                collisions[role] = ("external", f"{ext[0][1]}.{ext[0][2]}")
                break
    return collisions


def registry_size():
    return len(USED_TERMS)


# =========================================================
# PROMPTS
# =========================================================
PROMPT_FULL = """Construct a Greimas semiotic square for the concept: "{s1}"

S-positions:
- S1 = "{s1}"  (given)
- S2 = the contrary of S1 (direct opposition)
- ~S1 = the contradictory of S1 (e.g., "non-X")
- ~S2 = the contradictory of S2

M-positions (mediations):
- M1: synthesis encompassing S1 and S2
- M2: synthesis encompassing S2 and ~S2
- M3: synthesis encompassing ~S1 and ~S2
- M4: synthesis encompassing S1 and ~S1

C-positions (enclosures, between M values):
- C1: between M1 and M2
- C2: between M2 and M3
- C3: between M3 and M4
- C4: between M4 and M1

Example for "Truth":
S1=Being, S2=Seeming, ~S1=Non-being, ~S2=Non-seeming
M1=Truth, M2=Lie, M3=Falsehood, M4=Secret
C1=Paradox, C2=Deception, C3=Subterfuge, C4=Revelation
{exclusion_note}
Use concise terms (1-2 words). Avoid "not X" constructions where a robust
semantic equivalent exists.

Return ONLY a JSON object with exactly these keys:
"S1","S2","~S1","~S2","M1","M2","M3","M4","C1","C2","C3","C4".
"""

PROMPT_PARTIAL = """You are given four FIXED S-positions of a Greimas semiotic square.
Do NOT change these; they are inputs.

P1 = "{s1}"   (S1)
P2 = "{s2}"   (S2, the contrary of S1)
P3 = "{s3}"   (~S1, the contradictory of S1)
P4 = "{s4}"   (~S2, the contradictory of S2)

Generate the remaining 8 positions that complete the square.
{exclusion_note}
M-positions (mediations):
- M1: synthesis encompassing P1 and P2
- M2: synthesis encompassing P2 and P4
- M3: synthesis encompassing P3 and P4
- M4: synthesis encompassing P1 and P3

C-positions (enclosures, between M values):
- C1: between M1 and M2
- C2: between M2 and M3
- C3: between M3 and M4
- C4: between M4 and M1

Use concise terms (1-2 words). Return ONLY a JSON object with exactly these
keys: "M1","M2","M3","M4","C1","C2","C3","C4".
"""

PROMPT_SYNONYM = """The term "{term}" has already been used elsewhere in a semantic cascade, so we need a DIFFERENT word that carries the same meaning.

Provide a SINGLE synonym (or near-synonym) for "{term}" — a different word that could stand in the same structural role in a semiotic square.

The synonym MUST NOT be any of the following terms (they are already in use):
{exclusions}

Requirements:
- 1-2 words.
- Conceptually close to "{term}".
- Semantically appropriate for a formal structural role.

Return ONLY a JSON object with exactly this key: {{"synonym": "<word>"}}
"""

EXCLUSION_NOTE = """
UNIQUENESS CONSTRAINT:
Every term you output (except S1) must be a NEW word that does not appear
elsewhere in this cascade. The following terms have already been used and
MUST NOT appear in your response (as-is or as a close variant):

{terms}

If your first instinct would use one of these, choose a distinct synonym
or a different semantic pole that still fits the structural role.
"""


# =========================================================
# LLM CALL + JSON EXTRACTION
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
    last_raw = None

    for attempt in range(1, retries + 1):
        try:
            chat_kwargs = {
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "format": "json",
                "options": {
                    "temperature": TEMPERATURE,
                    "num_predict": NUM_PREDICT,
                    "stop": STOP_TOKENS,
                },
            }
            try:
                resp = ollama.chat(think=False, **chat_kwargs)
            except TypeError:
                resp = ollama.chat(**chat_kwargs)

            raw = resp["message"]["content"]
            last_raw = raw

            data = json.loads(extract_json_object(raw))
            missing = [k for k in required_keys if k not in data]
            if missing:
                raise ValueError(f"missing keys: {missing}")
            return data
        except Exception as e:
            last_err = str(e)
            print(f"      retry {attempt}/{retries} ({label}): {last_err}")
            if last_raw:
                preview = last_raw[:160].replace("\n", " ")
                print(f"         raw: {preview!r}")
            time.sleep(1)

    raise RuntimeError(f"LLM failed after {retries} retries ({label}): {last_err}")


def _exclusion_block(exclude_norms, extra_terms=None, label=""):
    terms = all_exclusion_terms(exclude_norms=exclude_norms,
                                 extra_terms=extra_terms)
    if not terms:
        return ""
    if label:
        print(f"      (exclusion list for {label}: {len(terms)} terms"
              f"{', ' + str(len(extra_terms)) + ' forced' if extra_terms else ''})")
    joined = ", ".join(terms)
    return EXCLUSION_NOTE.format(terms=joined)


# =========================================================
# SYNONYM FALLBACK (Strategy 2)
# =========================================================
def find_synonym(term, exclude_norms, extra_terms=None, max_tries=MAX_SYNONYM_TRIES):
    tried = {normalize(term)}
    extra = set(extra_terms or ())

    for attempt in range(1, max_tries + 1):
        exclusions = all_exclusion_terms(exclude_norms=exclude_norms,
                                          extra_terms=extra)
        for t_norm in list(tried):
            if t_norm in USED_TERMS:
                exclusions.append(USED_TERMS[t_norm][0][0])

        prompt = PROMPT_SYNONYM.format(
            term=term,
            exclusions=", ".join(exclusions) if exclusions else "(none)"
        )
        try:
            data = llm_json(prompt, ["synonym"], f"synonym:{term}")
        except Exception as e:
            print(f"      ✗ synonym attempt {attempt}/{max_tries} for "
                  f"'{term}': {e}")
            continue

        candidate = (data.get("synonym") or "").strip()
        norm = normalize(candidate)

        if not norm:
            continue
        if norm in tried:
            print(f"      ↻ synonym attempt {attempt}/{max_tries}: "
                  f"'{candidate}' already tried")
            continue
        tried.add(norm)

        if term_collides(candidate, "_synonym_lookup_"):
            print(f"      ↻ synonym attempt {attempt}/{max_tries}: "
                  f"'{candidate}' collides with registry")
            continue

        return candidate

    return None


def try_synonym_substitution(data, collisions, location,
                             exclude_norms, check_keys, forced_exclusions):
    new_data = dict(data)
    substituted_surfaces = set()

    for role in list(collisions.keys()):
        if role not in check_keys:
            continue

        original = data[role]
        extra = set(forced_exclusions) | substituted_surfaces
        syn = find_synonym(original, exclude_norms=exclude_norms,
                            extra_terms=extra)

        if syn is None:
            print(f"      ✗ no synonym found for {role}='{original}'")
            return None, False

        print(f"      ↻ {role}: '{original}' → synonym '{syn}'")
        new_data[role] = syn
        substituted_surfaces.add(syn)

    to_check = {k: new_data[k] for k in check_keys}
    new_collisions = find_collisions(to_check, location)
    if new_collisions:
        roles = ", ".join(new_collisions.keys())
        print(f"      ⚠ synonym substitution still collides at: {roles}")
        return None, False

    return new_data, True


# =========================================================
# DIAGRAM GENERATORS
# =========================================================
def gen_full_square(s1, location):
    register_term(s1, location, "S1")

    s1_norm = normalize(s1)
    s_norms = {s1_norm}
    forced_exclusions = set()
    last_data = None
    last_collisions = {}
    check_keys = [k for k in KEYS_FULL if k != "S1"]

    # -------- Tier 1 --------
    for attempt in range(1, MAX_UNIQUENESS_TRIES + 1):
        note = ""
        if attempt > 1:
            note = _exclusion_block(
                exclude_norms=s_norms,
                extra_terms=forced_exclusions,
                label=location,
            )

        try:
            data = llm_json(
                PROMPT_FULL.format(s1=s1, exclusion_note=note),
                KEYS_FULL,
                f"full:{location}",
            )
        except Exception as e:
            print(f"      ✗ {location}: {e}")
            continue

        data["S1"] = s1
        to_check = {k: data[k] for k in check_keys}
        collisions = find_collisions(to_check, location)

        if not collisions:
            for role, term in to_check.items():
                register_term(term, location, role)
            return data, STATUS_OK

        msgs = []
        for role, (kind, info) in collisions.items():
            msgs.append(f"{role}='{to_check[role]}' ({kind}: {info})")
            forced_exclusions.add(to_check[role])
        print(f"      ⚠ attempt {attempt}/{MAX_UNIQUENESS_TRIES}: "
              f"{'; '.join(msgs)}")

        last_data = data
        last_collisions = collisions

    if last_data is None:
        return None, STATUS_DUPLICATE

    # -------- Tier 2 --------
    print(f"      → entering synonym fallback for {location}")
    syn_data, ok = try_synonym_substitution(
        last_data, last_collisions, location,
        exclude_norms=s_norms,
        check_keys=check_keys,
        forced_exclusions=forced_exclusions,
    )
    if ok:
        for role, term in syn_data.items():
            if role != "S1":
                register_term(term, location, role)
        return syn_data, STATUS_SYNONYM

    # -------- Tier 3 --------
    print(f"      ✗ all fallbacks exhausted; accepting duplicates "
          f"for {location}")
    for role, term in last_data.items():
        if role != "S1":
            register_term(term, location, role)
    return last_data, STATUS_DUPLICATE


def gen_partial(s_terms, location):
    for role, term in s_terms.items():
        register_term(term, location, role)

    s_norms = {normalize(t) for t in s_terms.values()}
    forced_exclusions = set()
    last_data = None
    last_collisions = {}
    check_keys = list(KEYS_PARTIAL)

    # -------- Tier 1 --------
    for attempt in range(1, MAX_UNIQUENESS_TRIES + 1):
        note = ""
        if attempt > 1:
            note = _exclusion_block(
                exclude_norms=s_norms,
                extra_terms=forced_exclusions,
                label=location,
            )

        try:
            data = llm_json(
                PROMPT_PARTIAL.format(
                    s1=s_terms["S1"], s2=s_terms["S2"],
                    s3=s_terms["~S1"], s4=s_terms["~S2"],
                    exclusion_note=note,
                ),
                KEYS_PARTIAL,
                f"partial:{location}",
            )
        except Exception as e:
            print(f"      ✗ {location}: {e}")
            continue

        full = {**s_terms, **data}
        to_check = {k: data[k] for k in check_keys}
        collisions = find_collisions(to_check, location)

        if not collisions:
            for role, term in to_check.items():
                register_term(term, location, role)
            return full, STATUS_OK

        msgs = []
        for role, (kind, info) in collisions.items():
            msgs.append(f"{role}='{to_check[role]}' ({kind}: {info})")
            forced_exclusions.add(to_check[role])
        print(f"      ⚠ attempt {attempt}/{MAX_UNIQUENESS_TRIES}: "
              f"{'; '.join(msgs)}")

        last_data = full
        last_collisions = collisions

    if last_data is None:
        return None, STATUS_DUPLICATE

    # -------- Tier 2 --------
    print(f"      → entering synonym fallback for {location}")
    syn_data, ok = try_synonym_substitution(
        last_data, last_collisions, location,
        exclude_norms=s_norms,
        check_keys=check_keys,
        forced_exclusions=forced_exclusions,
    )
    if ok:
        for role in check_keys:
            register_term(syn_data[role], location, role)
        return syn_data, STATUS_SYNONYM

    # -------- Tier 3 --------
    print(f"      ✗ all fallbacks exhausted; accepting duplicates "
          f"for {location}")
    for role in check_keys:
        register_term(last_data[role], location, role)
    return last_data, STATUS_DUPLICATE


# =========================================================
# PERSISTENCE + UTIL
# =========================================================
def save(state, path):
    Path(path).write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def slugify(text):
    out = "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")
    return out or "concept"


def apply_status_flag(diagram, status):
    if status == STATUS_SYNONYM:
        diagram["_resolved_by_synonym"] = True
    elif status == STATUS_DUPLICATE:
        diagram["_unresolved_collisions"] = True


# =========================================================
# CASCADE
# =========================================================
def build_cascade(initial, output_path, preset_name):
    USED_TERMS.clear()
    REGISTRATION_ORDER.clear()

    state = {
        "metadata": {
            "initial_concept": initial,
            "model": MODEL_NAME,
            "model_preset": preset_name,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "structure": {"L1": 16, "L2": 4, "L3": 1, "total": 21},
            "settings": {
                "max_exclusion_terms": MAX_EXCLUSION_TERMS,
                "max_uniqueness_tries": MAX_UNIQUENESS_TRIES,
                "max_synonym_tries": MAX_SYNONYM_TRIES,
                "temperature": TEMPERATURE,
                "num_predict": NUM_PREDICT,
            },
            "uniqueness_strategy": (
                "1. normal retries; 2. synonym substitution; "
                "3. accept duplicate (flagged)"
            ),
            "relationships": {
                "L1": "L1 D(n+1).S1 = L1 D(n).C2 (with global uniqueness)",
                "L2 D1": "S1..S4 = ~S2 of L1 D1, D2, D3, D4",
                "L2 D2": "S1..S4 = ~S2 of L1 D5, D6, D7, D8",
                "L2 D3": "S1..S4 = ~S2 of L1 D9, D10, D11, D12",
                "L2 D4": "S1..S4 = ~S2 of L1 D13, D14, D15, D16",
                "L3 D1": "S1..S4 = C2 of L2 D1, D2, D3, D4",
            },
        },
        "L1": {},
        "L2": {},
        "L3": {},
    }

    # ---------------- LEVEL 1 ----------------
    print("=" * 72)
    print("LEVEL 1 — 16 diagrams (S1 of D(n+1) = C2 of D(n))")
    print(f"         Model: {MODEL_NAME}")
    print(f"         Uniqueness: retries → synonyms → accept-duplicate")
    print("=" * 72)

    current_s1 = initial
    for i in range(1, 17):
        location = f"L1.D{i}"
        print(f"\n[{location}] S1 = '{current_s1}'   "
              f"(registry: {registry_size()} terms)")
        t0 = time.time()
        d, status = gen_full_square(current_s1, location)
        if d is None:
            raise RuntimeError(f"Failed to generate {location}")

        apply_status_flag(d, status)
        state["L1"][f"D{i}"] = d
        save(state, output_path)

        flag = ""
        if status == STATUS_SYNONYM:
            flag = "  [via synonyms]"
        elif status == STATUS_DUPLICATE:
            flag = "  [duplicate accepted]"
        print(f"        ✓ {time.time() - t0:.1f}s   C2 = '{d['C2']}'{flag}")
        current_s1 = d["C2"]

    # ---------------- LEVEL 2 ----------------
    print("\n" + "=" * 72)
    print("LEVEL 2 — 4 diagrams (S-terms = ~S2 of consecutive L1 diagrams)")
    print("=" * 72)

    for i in range(1, 5):
        location = f"L2.D{i}"
        base = (i - 1) * 4
        s_terms = {
            "S1":  state["L1"][f"D{base + 1}"]["~S2"],
            "S2":  state["L1"][f"D{base + 2}"]["~S2"],
            "~S1": state["L1"][f"D{base + 3}"]["~S2"],
            "~S2": state["L1"][f"D{base + 4}"]["~S2"],
        }
        print(f"\n[{location}] from L1 D{base+1}..D{base+4} .~S2")
        print(f"        S1='{s_terms['S1']}'  S2='{s_terms['S2']}'  "
              f"~S1='{s_terms['~S1']}'  ~S2='{s_terms['~S2']}'")
        t0 = time.time()
        d, status = gen_partial(s_terms, location)
        apply_status_flag(d, status)
        state["L2"][f"D{i}"] = d
        save(state, output_path)
        flag = ""
        if status == STATUS_SYNONYM:
            flag = "  [via synonyms]"
        elif status == STATUS_DUPLICATE:
            flag = "  [duplicate accepted]"
        print(f"        ✓ {time.time() - t0:.1f}s   C2 = '{d['C2']}'{flag}")

    # ---------------- LEVEL 3 ----------------
    print("\n" + "=" * 72)
    print("LEVEL 3 — 1 diagram (S-terms = C2 of L2 D1..D4)")
    print("=" * 72)

    location = "L3.D1"
    s_terms = {
        "S1":  state["L2"]["D1"]["C2"],
        "S2":  state["L2"]["D2"]["C2"],
        "~S1": state["L2"]["D3"]["C2"],
        "~S2": state["L2"]["D4"]["C2"],
    }
    print(f"\n[{location}]")
    print(f"        S1='{s_terms['S1']}'  S2='{s_terms['S2']}'  "
          f"~S1='{s_terms['~S1']}'  ~S2='{s_terms['~S2']}'")
    t0 = time.time()
    d, status = gen_partial(s_terms, location)
    apply_status_flag(d, status)
    state["L3"]["D1"] = d
    save(state, output_path)
    flag = ""
    if status == STATUS_SYNONYM:
        flag = "  [via synonyms]"
    elif status == STATUS_DUPLICATE:
        flag = "  [duplicate accepted]"
    print(f"        ✓ {time.time() - t0:.1f}s{flag}")

    return state


# =========================================================
# SUMMARY
# =========================================================
def print_summary(state):
    print("\n" + "=" * 72)
    print("CASCADE COMPLETE")
    print(f"Total unique terms: {registry_size()}")
    print("=" * 72)

    def flag_for(d):
        if d.get("_unresolved_collisions"):
            return "  ⚠ duplicate"
        if d.get("_resolved_by_synonym"):
            return "  ↻ synonyms"
        return ""

    print("\nL1 (S1 → C2 chain):")
    for i in range(1, 17):
        d = state["L1"][f"D{i}"]
        print(f"  D{i:>2}: S1={d['S1']:<24} C2={d['C2']}{flag_for(d)}")

    print("\nL2 (S-terms from L1 ~S2):")
    for i in range(1, 5):
        d = state["L2"][f"D{i}"]
        print(f"  D{i}: S1={d['S1']:<20} S2={d['S2']:<20} "
              f"~S1={d['~S1']:<20} ~S2={d['~S2']}{flag_for(d)}")

    print("\nL3 (S-terms from L2 C2):")
    d = state["L3"]["D1"]
    print(f"  D1: S1={d['S1']:<20} S2={d['S2']:<20} "
          f"~S1={d['~S1']:<20} ~S2={d['~S2']}{flag_for(d)}")

    n_syn = sum(
        1 for level in ("L1", "L2", "L3")
        for d in state[level].values()
        if d.get("_resolved_by_synonym")
    )
    n_dup = sum(
        1 for level in ("L1", "L2", "L3")
        for d in state[level].values()
        if d.get("_unresolved_collisions")
    )
    print(f"\nFallbacks used: "
          f"{n_syn} via synonym substitution, {n_dup} with accepted duplicates")
    print("=" * 72)


# =========================================================
# ARGUMENT PARSING
# =========================================================
def parse_args():
    p = argparse.ArgumentParser(
        prog="greimas_cascade",
        description="Generate a Greimas cascade using a local Ollama model.",
    )
    p.add_argument(
        "concept",
        nargs="*",
        help="Initial concept for L1 D1 (omit to be prompted).",
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        "--model", "-m",
        choices=list(MODEL_PRESETS.keys()),
        default=None,
        help=f"Model preset (default: {DEFAULT_PRESET}).",
    )
    g.add_argument(
        "--fast", "-f",
        action="store_true",
        help="Shorthand for --model fast.",
    )
    g.add_argument(
        "--quality", "-q",
        action="store_true",
        help="Shorthand for --model quality.",
    )
    p.add_argument(
        "--suffix", "-s",
        default=None,
        help="Optional filename suffix (e.g. 'quality') to distinguish runs.",
    )
    return p.parse_args()


def apply_preset(name):
    """Overwrite module-level MODEL_NAME/TEMPERATURE/NUM_PREDICT globals."""
    global MODEL_NAME, TEMPERATURE, NUM_PREDICT
    preset = MODEL_PRESETS[name]
    MODEL_NAME  = preset["model"]
    TEMPERATURE = preset["temperature"]
    NUM_PREDICT = preset["num_predict"]
    return preset


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    args = parse_args()

    # Decide which preset to use. Priority: --quality > --fast > --model > default.
    if args.quality:
        preset_name = "quality"
    elif args.fast:
        preset_name = "fast"
    elif args.model:
        preset_name = args.model
    else:
        preset_name = DEFAULT_PRESET

    preset = apply_preset(preset_name)

    if args.concept:
        concept = " ".join(args.concept).strip()
    else:
        concept = input("Enter initial concept for L1 D1: ").strip()

    if not concept:
        print("No concept provided. Exiting.")
        sys.exit(1)

    suffix = f"_{args.suffix}" if args.suffix else ""
    output_path = f"greimas_cascade_{slugify(concept)}{suffix}.json"

    print(f"\nCascade for: '{concept}'")
    print(f"Model:       {preset['label']}")
    print(f"Output:      {output_path}")
    print(f"Structure:   16 + 4 + 1 = 21 diagrams")
    print(f"Uniqueness:  retries → synonyms → accept-duplicate")
    print(f"Progress saves to disk after every diagram.\n")

    try:
        state = build_cascade(concept, output_path, preset_name)
    except KeyboardInterrupt:
        print("\n\nInterrupted. Partial result saved to:")
        print(f"  {output_path}")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Cascade failed: {e}")
        print(f"  Partial result saved to: {output_path}")
        sys.exit(1)

    save(state, output_path)
    print(f"\n✓ Full cascade saved to: {output_path}")
    print_summary(state)
