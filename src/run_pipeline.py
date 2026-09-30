"""
Business Entity Resolution -- Amazon ML Challenge 2026
Runnable pipeline (extracted 1:1 from the validated entity_resolution_v13.ipynb).

This script mirrors the notebook top-to-bottom, in the same order the notebook
was executed and validated in, so behavior is identical -- it is not a rewrite.
The only changes from the notebook are: no %pip install magic (see
requirements.txt / README.md in this folder), AWS credentials/bucket are read from
plain environment variables instead of Colab Secrets.

Run: python src/run_pipeline.py
"""


import re
import os
import gc
import math
import time
import resource
import unicodedata
import boto3
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier   # v13: non-linear model (built into scikit-learn, no new install)
from sklearn.model_selection import train_test_split
from rapidfuzz import fuzz
from rapidfuzz import process as rf_process   # v13: cpdist = C-level, multi-threaded element-wise scoring for the NEW features

pd.set_option('display.max_colwidth', 80)
STRING_DTYPE = "string[pyarrow]"   # Arrow-backed strings: avoids per-object Python overhead

def print_mem(msg=""):
    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # Linux reports KB; this is a HIGH-WATER MARK (only ever goes up within a session)
    print(f"[mem] {msg:<50s} peak RSS so far: {peak_kb/1024:,.0f} MB")

print_mem("baseline, before any data touched")


# ==============================================================================
# AWS credentials -- read from plain environment variables (see README.md).
# On an AWS-hosted instance with an attached IAM role this is a no-op.
# ==============================================================================
if not os.environ.get("AWS_ACCESS_KEY_ID"):
    print("AWS_ACCESS_KEY_ID not set -- assuming an attached IAM role. Otherwise export "
          "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION before running.")

# ==============================================================================
# ## 1. Config — every tunable knob lives here
# ==============================================================================

BUCKET_NAME = os.environ.get("BUCKET_NAME", "your-bucket-name-here")   # <-- set via env var or edit here
if not os.environ.get("AWS_DEFAULT_REGION"):
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
PREFIX = "dataset/"
TRAIN_DIR = f"s3://{BUCKET_NAME}/dataset/train"
TEST_DIR = f"s3://{BUCKET_NAME}/dataset/test"

SAMPLE_MODE = False    # <-- real run: full train_s1 pool (capped via TRAIN_FIT_SAMPLE_N below) + full test_s1 (never subsampled)
SAMPLE_N = 20000       # only used when SAMPLE_MODE=True -- the S1 side is subsampled; the pool stays full so recall numbers stay meaningful

MAX_POOL_BLOCK_SIZE = 50000       # legacy safety-valve constant from patches 3/5-8 -- NO LONGER USED by candidate generation (patch 9 / bottleneck #31 removes the per-lookup rng.choice call this used to bound entirely; explode_key's O(1) array slice makes it unnecessary). Left defined only so older bottleneck-history comments still resolve.
MAX_CANDIDATES_PER_ENTITY = 60   # v13: was 50. 6 keys x 10 = every original key keeps its exact 10-candidate share; the composite key adds up to 10 more (usually far fewer -- its buckets are small)
CHUNK_SIZE = 20_000               # S1 rows processed per streaming batch. Deliberately NOT raised despite candidate-gen now being ~1000x cheaper per chunk -- this keeps memory behavior identical to the last verified-safe run while the timing bug is what's actually being fixed; raise this only AFTER a first clean full run, if you want fewer/larger chunks.
NEG_CAP_PER_CHUNK = 40_000        # v13: was 4000. Negatives sampled per chunk for training. A tree model needs to SEE hard negatives (candidates that share a blocking key but are a different business) -- 4000/chunk was ~10x fewer negatives than positives. 40k/chunk x 13 chunks ~= 0.5M negatives, ~100MB total: negligible vs the RAM already in use

SCAN_WINDOW_MULT = 4        # each key scans at most per_key_cap * SCAN_WINDOW_MULT (~40) ids of a row's bucket, taking up to per_key_cap (~10) not already taken by an earlier key -- identical semantics to patches 5-8, now vectorized
TRAIN_FIT_SAMPLE_N = 250_000 # Section 6 only needs enough rows to fit the classifier + tune the thresholds -- it does NOT need test-set scale, and only Section 10's TEST predictions are scored on the leaderboard
VAL_MAX_ENTITIES = 40_000   # caps how many S1 entities are routed to the "kept in full" validation side -- large enough for a stable recall/F0.5 estimate, tiny relative to full scale

BASE_KEY_COLS = ["key_name_sorted", "key_name_prefix", "key_name_fingerprint", "key_addr_sorted", "key_addr_digits"]  # the 5 real key columns built by build_keys -- expanded from 3 to 5 after the Section 3.5 ceiling check came back at 84.7% (below the 90% target) with the original 3 -- the keys themselves, not the index, were the limiting factor
# v13 (bottleneck #33): feature set expanded from 4 live features to 19.
#  PAIR features (one string pair at a time):
#    name_sim/addr_sim (token_sort_ratio) and jaccard_* are unchanged from v12 (identical values).
#    name_ratio (order-SENSITIVE edit similarity), name_partial (best substring match -> truncations/abbreviations),
#    name_tset/addr_tset (token_set_ratio: 100 when one side's tokens are a subset of the other's -> missing words),
#    addr_partial, postal_eq (longest 4+ digit run equal: 1 / differs: 0 / missing on either side: 0.5),
#    first_tok_eq (first name token identical), name_len_ratio (short/long length ratio).
#  GROUP features (relative to the SAME S1 entity's other candidates -- the strongest signal in entity resolution:
#    "is this the best-looking candidate this entity has?"): n_cands, name_gap/addr_gap/tset_gap (value minus the
#    entity's best value), sum_sim (name_sim+addr_sim), sum_gap, sum_rank (1 = best).
#  country_match DROPPED: it is 1.0 for every pair under country-partitioned blocking (a constant, zero signal).
#  Nothing is country-specific, so France (unseen in training) is scored exactly like US/India.
PAIR_FEATURES = ["name_sim", "addr_sim", "jaccard_name", "jaccard_addr",
                 "name_ratio", "name_partial", "name_tset", "addr_tset", "addr_partial",
                 "postal_eq", "first_tok_eq", "name_len_ratio"]
GROUP_FEATURES = ["n_cands", "name_gap", "addr_gap", "tset_gap", "sum_sim", "sum_gap", "sum_rank"]
FEATURES = PAIR_FEATURES + GROUP_FEATURES

# v13 (bottleneck #33): COMPOSITE blocking key = postal code AND 4-char name prefix, built purely
# from the two existing columns' integer category codes (no new strings, no build_keys change,
# ~100MB extra index at full scale). Why: postal codes and name prefixes are each shared by
# thousands of businesses at real scale, and a key only takes ~10 RANDOM members of a big bucket
# -- so a true match sharing both is usually missed. Their intersection is a small bucket, so it
# is found. It goes FIRST so later keys spend their share on other candidates. It can't lower the
# Section 3.5 ceiling (it's the intersection of two existing keys), it only recovers pairs the
# random cap was throwing away.
COMPOSITE_KEYS = {"key_postal_x_prefix": ("key_addr_digits", "key_name_prefix")}
KEY_COLS = ["key_postal_x_prefix"] + BASE_KEY_COLS
V12_FEATURES = ["name_sim", "addr_sim", "jaccard_name", "jaccard_addr"]   # only for the same-run v12-style baseline printed in Section 7

os.makedirs("cache", exist_ok=True)
os.makedirs("output", exist_ok=True)

REPLACEMENTS = {
    r"\bcorp\b": "corporation", r"\bltd\b": "limited", r"\bpvt\b": "private",
    r"\binc\b": "incorporated", r"\bco\b": "company", r"\b&\b": " and ",
    r"\brd\b": "road", r"\bst\b": "street", r"\bave\b": "avenue",
    r"\bapt\b": "apartment", r"\bblvd\b": "boulevard",
}

def normalize(s):
    """Scalar version -- used by build_keys (one-time, memory-sensitive; see its
    docstring) and as the reference normalize_series() was validated against. The
    per-chunk hot path (compute_features_chunk) uses normalize_series() below.

    PATCH 10 (bottleneck #32): transliterates accented characters (e.g. Cafe,
    e-acute -> e) to ASCII BEFORE the ascii-only regex strips everything else.
    The old order stripped accented letters outright -- "Societe Generale" with
    accents came out as "soci t g n rale" (whole letters deleted, not just the
    accent mark), scrambling word boundaries. This directly hurts the ~15% of
    the test set that is France, unseen in training, where accented characters
    are common. Confirmed on real examples (see project README bottleneck #32);
    unaccented text (all of train, most of US/India test) is unaffected."""
    if pd.isna(s):
        return ""
    s = str(s).lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^a-z0-9\s&]", " ", s)
    for pat, rep in REPLACEMENTS.items():
        s = re.sub(pat, rep, s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_series(s):
    """PATCH 9 (bottleneck #31): vectorized pandas .str equivalent of normalize()
    above -- operates on a whole Series via C-level regex passes instead of a
    per-row Python function call. Validated byte-identical to normalize() applied
    elementwise across 2000+ randomized strings plus handcrafted edge cases
    (empty, None/NaN, unicode, punctuation-only, hyphens, leading/trailing
    whitespace, repeated-token inputs) -- see project README bottleneck #31.

    PATCH 10 (bottleneck #32): the accent-transliteration step is per-element
    (unicodedata has no vectorized pandas equivalent) but only runs once per
    UNIQUE id already (compute_features_chunk dedupes before calling this), so
    the added cost is small -- benchmarked at well under a second per chunk."""
    # Forced to plain object dtype on purpose: on pyarrow-backed string dtypes pandas
    # routes .str.replace through pyarrow's RE2 engine, whose \s / \b semantics can
    # differ from Python's re. Object dtype = Python's re in every pandas version,
    # i.e. guaranteed-identical output to normalize() -- and it benchmarked fastest too.
    s = s.astype(object)
    s = s.where(s.notna(), "").astype(str).astype(object).str.lower()
    s = s.map(lambda x: unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("ascii"))
    s = s.str.replace(r"[^a-z0-9\s&]", " ", regex=True)
    for pat, rep in REPLACEMENTS.items():
        s = s.str.replace(pat, rep, regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s

def digits_only(s):
    """Extract a POSTAL-CODE-LIKE digit run, not every digit in the string.

    IMPORTANT FIX: the original version concatenated ALL digits (street number +
    unit + suite + postal code, etc.), which is close to unique per business --
    a near-unique key means groupby(...).indices builds a Python dict with
    millions of tiny-bucket entries, which is what crashed Section 4 (the same
    per-object dict overhead problem as the earlier build_id_codec bug, just
    recurring in a new column). Postal codes are typically the longest run of
    4+ consecutive digits in a US/India/France address (street numbers are
    usually shorter) -- this keeps genuine duplicate structure (many businesses
    really do share a postal code) instead of manufacturing a near-unique key.
    """
    if pd.isna(s):
        return ""
    runs = re.findall(r"\d{4,}", str(s))
    return max(runs, key=len) if runs else ""

def token_jaccard(a, b):
    """PATCH 10 (bottleneck #32): word-set overlap, independent of rapidfuzz's
    character-edit-distance signal -- catches cases where tokens are identical but
    reordered/repeated differently (already handled well by token_sort_ratio) AND
    cases where one string is a strict subset of the other's tokens (e.g. a
    truncated/abbreviated name), which token_sort_ratio can under-score. Cheap:
    benchmarked at <1s per 400k-pair chunk."""
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def pairwise_scores(scorer, a, b):
    """v13: element-wise rapidfuzz score for aligned lists a[i] vs b[i], scaled to [0,1].
    Uses rapidfuzz.process.cpdist (C-level, all CPU cores) when available (rapidfuzz>=3.6,
    what a fresh pip install of rapidfuzz gives today); falls back to a plain Python loop otherwise,
    so an older rapidfuzz can never break the run -- only make it slower."""
    if len(a) == 0:
        return np.empty(0, dtype=np.float32)
    try:
        return (rf_process.cpdist(a, b, scorer=scorer, workers=-1) / 100).astype(np.float32)
    except (AttributeError, TypeError):
        return np.array([scorer(x, y) / 100 for x, y in zip(a, b)], dtype=np.float32)

# ==============================================================================
# ## 2. Load train data, cache raw text to Parquet, build keys + id_codes
#
# **Load order matters here**: id_code is assigned *before* keys are built, so raw
# text can be cached to Parquet keyed by `id_code` and then dropped from the pool
# frame in the same step that builds the keys — the pool frame never holds both the
# full key set AND the full raw text at once.
# ==============================================================================

def load_source(path, usecols=("entity_id", "business_name", "business_address", "country")):
    df = pd.read_csv(path, sep="\t", dtype=STRING_DTYPE, usecols=list(usecols))
    df["country"] = df["country"].astype("category")
    return df

train_s1 = load_source(f"{TRAIN_DIR}/train_source1.tsv")
_s2 = load_source(f"{TRAIN_DIR}/train_source2.tsv")
_s3 = load_source(f"{TRAIN_DIR}/train_source3.tsv")
train_gt = pd.read_csv(f"{TRAIN_DIR}/train_ground_truth.tsv", sep="\t", dtype=STRING_DTYPE)

train_pool_all = pd.concat([_s2, _s3], ignore_index=True)
del _s2, _s3
gc.collect()
print("train_s1:", train_s1.shape, " pool (S2+S3):", train_pool_all.shape, " ground truth:", train_gt.shape)
print_mem("after loading (pyarrow strings + categorical country)")

# Dict-free id codec: pd.factorize preserves input order, so slice its output array
# directly instead of building a ~12M-entry Python dict + calling .map() (confirmed
# ~1.1GB and meaningfully slower in an earlier version).
def build_id_codec(*dfs):
    lengths = [len(d) for d in dfs]
    all_ids = pd.concat([d["entity_id"] for d in dfs], ignore_index=True)
    codes, uniques = pd.factorize(all_ids)
    del all_ids
    gc.collect()
    return codes, lengths, np.asarray(uniques)

_codes, _lengths, train_id_lookup = build_id_codec(train_s1, train_pool_all)
train_s1["id_code"] = _codes[:_lengths[0]].astype("int32")
train_pool_all["id_code"] = _codes[_lengths[0]:].astype("int32")
del _codes, _lengths
gc.collect()
print_mem("after id codec")

# Compact int32-indexed lookup arrays for O(1) vectorized feature lookups later,
# instead of touching the big frames per candidate pair.
n_total_ids = len(train_id_lookup)
train_country_by_idcode = np.empty(n_total_ids, dtype=object)
train_country_by_idcode[train_s1["id_code"].values] = train_s1["country"].values
train_country_by_idcode[train_pool_all["id_code"].values] = train_pool_all["country"].values

# Cache raw text to Parquet keyed by id_code BEFORE it's dropped from the pool frame.
# train_s1 (~2.2M rows) is small enough to just keep resident -- no cache needed for it.
train_pool_all[["id_code", "business_name", "business_address"]].to_parquet("cache/train_pool_text.parquet")
train_s1_text = train_s1.set_index("id_code")[["business_name", "business_address"]]
print_mem("after caching pool raw text to disk")

def build_keys(df, drop_text, label=""):
    """Adds the 5 categorical blocking-key columns. Mutates in place (no .copy()).
    Processes name-keys then address-keys sequentially so only one normalized
    Series is held at once per field. When drop_text=True, raw text is dropped
    afterward (already cached to Parquet by the caller).

    Deliberately still uses .apply(normalize), NOT normalize_series(): on the
    ~10.2M-row pool, normalize_series would transiently hold two full object-string
    copies of the column (~+0.8GB) in the section that OOM'd before (#14/#15). This
    runs once, was never the time bottleneck, and produces identical keys either
    way -- so its memory profile is kept exactly as in the last verified-safe run.

    key_name_sorted / key_name_prefix / key_addr_sorted: original 3 keys.
    key_name_fingerprint: sorted set of each name token's first 3 chars --
      more forgiving than key_name_prefix for typos anywhere except a token's
      very start, and still tolerant of word-order transposition.
    key_addr_digits: digits only from the RAW address (postal code / street
      number) -- survives word-level noise (abbreviations, transliteration,
      landmark substitution, reordering) that breaks the word-based address key.
    Added after the Section 3.5 ceiling check showed the original 3 keys capped
    achievable recall at 84.7%, regardless of any index/cap tuning."""
    name_norm_tmp = df["business_name"].apply(normalize)
    df["key_name_sorted"] = name_norm_tmp.apply(lambda s: "".join(sorted(s.split()))).astype("category")
    df["key_name_prefix"] = name_norm_tmp.str.replace(" ", "", regex=False).str[:4].astype("category")
    df["key_name_fingerprint"] = name_norm_tmp.apply(
        lambda s: "".join(sorted(t[:3] for t in s.split() if t))
    ).astype("category")
    del name_norm_tmp
    gc.collect()

    addr_norm_tmp = df["business_address"].apply(normalize)
    df["key_addr_sorted"] = addr_norm_tmp.apply(lambda s: "".join(sorted(s.split()))).astype("category")
    del addr_norm_tmp
    gc.collect()

    df["key_addr_digits"] = df["business_address"].apply(digits_only).apply(lambda s: "".join(sorted(s))).astype("category")
    gc.collect()

    if drop_text:
        df = df.drop(columns=["business_name", "business_address"])
        gc.collect()
    print_mem(f"build_keys({label}): done")
    return df

train_s1 = build_keys(train_s1, drop_text=False, label="train_s1")           # keep text -- small, needed later anyway
train_pool_all = build_keys(train_pool_all, drop_text=True, label="train_pool_all")  # drop text -- cached to Parquet already
gc.collect()
print_mem("after build_keys on both frames")

# ==============================================================================
# ## 3. Ground truth, in id_code space (not string-keyed)
# ==============================================================================

# Map ground-truth string IDs to int id_codes via a transient reverse-lookup,
# built once and deleted right after -- this avoids keeping a string-keyed dict
# around for the rest of the run.
_id_to_code = pd.Series(np.arange(len(train_id_lookup), dtype=np.int32), index=train_id_lookup)

gt_pairs = train_gt.copy()
gt_pairs["matched_entity_ids"] = gt_pairs["matched_entity_ids"].fillna("").astype(str).str.split(",")
gt_pairs = gt_pairs.explode("matched_entity_ids")
gt_pairs = gt_pairs[gt_pairs["matched_entity_ids"].notna() & (gt_pairs["matched_entity_ids"] != "")]
gt_pairs["s1_code"] = _id_to_code.reindex(gt_pairs["source1_entity_id"]).values
gt_pairs["cand_code"] = _id_to_code.reindex(gt_pairs["matched_entity_ids"]).values
del _id_to_code
gc.collect()

gt_map = defaultdict(set)   # int id_code -> set(int id_code) -- far cheaper than a string-keyed dict
for s1c, cc in zip(gt_pairs["s1_code"].values, gt_pairs["cand_code"].values):
    gt_map[int(s1c)].add(int(cc))

singleton_rate = 1 - (len(gt_map) / train_s1["id_code"].nunique())
print(f"Singleton rate in train ground truth: {singleton_rate:.1%}")
del gt_pairs
gc.collect()
print_mem("after ground truth (id_code space)")

# ==============================================================================
# ## 3.5. Key-overlap CEILING check — run this before trusting any recall number from Section 7
#
# This bypasses the index entirely: for a sample of *known true* (S1, match) pairs,
# it checks whether they share **any** of the 5 key values, with no size limit
# applied at all.
#
# This number is the **hard ceiling** on achievable recall with the current key
# definitions — no amount of index/cap tuning can exceed it, because if the true
# match doesn't share a key value with its S1 entity in the raw data, no index
# design will ever surface it as a candidate.
# ==============================================================================

def measure_key_coverage_ceiling(s1_df, pool_df, gt_map, key_cols, sample_size=5000, seed=42):
    s1_keys = s1_df.set_index("id_code")[list(key_cols)]
    pool_keys = pool_df.set_index("id_code")[list(key_cols)]

    pairs = [(s1c, cc) for s1c, matches in gt_map.items() for cc in matches]
    if len(pairs) > sample_size:
        idx = np.random.default_rng(seed).choice(len(pairs), size=sample_size, replace=False)
        pairs = [pairs[i] for i in idx]

    hits, checked = 0, 0
    per_key_hits = {col: 0 for col in key_cols}
    for s1c, cc in pairs:
        if s1c not in s1_keys.index or cc not in pool_keys.index:
            continue
        checked += 1
        s1_row = s1_keys.loc[s1c]
        c_row = pool_keys.loc[cc]
        any_hit = False
        for col in key_cols:
            if s1_row[col] == c_row[col]:
                per_key_hits[col] += 1
                any_hit = True
        if any_hit:
            hits += 1

    print(f"Key-overlap CEILING on {checked} true pairs: {hits/checked:.1%}")
    print("(bypasses the index entirely -- this is the max recall these key definitions can EVER achieve)")
    for col, h in per_key_hits.items():
        print(f"  {col}: {h/checked:.1%} of true pairs share this key alone")
    return hits / checked if checked else 0.0

_ceiling = measure_key_coverage_ceiling(train_s1, train_pool_all, gt_map, BASE_KEY_COLS)
if _ceiling < 0.90:
    print("\n>>> Ceiling is below 90% -- fixing the index alone will NOT be enough. "
          "A 6th blocking key would be needed.")

# ==============================================================================
# ## 4. Country-partitioned capped hash index (blocking)
#
# Built **once** from the full pool. For each country, for each of the 5 key types,
# maps `key_value -> array of pool id_codes`, stored as a CSR-style structure
# (sorted-by-code numpy arrays + boundary offsets) rather than a Python dict of
# one array per bucket — this is what keeps memory scaling with (row count +
# unique-value count) in raw bytes, not in Python objects, regardless of key
# cardinality or skew (see bottlenecks #19-#21).
#
# **Patch 9 addition**: each country's row order is shuffled once, up front, before
# any key's CSR structure is built (details in `build_pool_index`'s docstring below).
# This lets candidate generation take a plain array-slice prefix of any bucket
# instead of calling `rng.choice()` per lookup — removing the confirmed root cause
# of the ~180-200s/chunk cost from bottlenecks #26-#30.
# ==============================================================================

def build_pool_index(pool_df, key_cols, seed=987):
    """CSR-style index: for each country, for each key type, stores
    (categories, sorted unique codes, boundaries, sorted pool id_codes) as plain
    NUMPY ARRAYS -- NOT a Python dict of one array per bucket (see bottlenecks
    #19-#20 for why a dict-based or cardinality-gated design doesn't work here).

    PATCH 9 (bottleneck #31): each country's row order is permuted ONCE, up
    front, before building any key's structure. Every key's per-code grouping
    below is a STABLE sort by that key's code -- a stable sort preserves whatever
    relative order rows were in beforehand, so after this one shuffle, every
    bucket's internal order is an independent-looking random permutation of that
    bucket's true members. This means candidate generation (see
    generate_candidates_chunk / explode_key) can take the FIRST `sample_size` ids
    from any bucket via a plain O(1) array slice and get a valid uniform random
    subsample of an oversized bucket -- with NO rng.choice() call needed at
    lookup time. That per-lookup rng.choice call over the (potentially huge)
    bucket length was the confirmed root cause of the ~180-200s/chunk cost
    patches 5-8 chased (bottlenecks #26-#30) -- moving the randomness to this
    one-time O(pool size) shuffle removes it from the per-chunk, per-row path
    entirely, rather than trying to cache or shrink it further.

    Also carries forward bottleneck #21's fix: the empty/blank-key exclusion mask
    is computed entirely in int space (`cat.codes` + `categories.get_loc("")`),
    never decoding categories back to a boxed string array.
    """
    index = {}
    stats = {"keys_built": 0, "rows_indexed": 0}
    rng = np.random.default_rng(seed)
    for country, cdf in pool_df.groupby("country", observed=True):
        n = len(cdf)
        perm = rng.permutation(n)
        id_codes = cdf["id_code"].values[perm]
        col_index = {}
        for col in key_cols:
            if col in COMPOSITE_KEYS:
                # v13: composite = code_a * n_categories_b + code_b, int64, only where BOTH are non-empty
                a, b = COMPOSITE_KEYS[col]
                cat_a, cat_b = cdf[a].cat.categories, cdf[b].cat.categories
                ca = cdf[a].cat.codes.values[perm].astype(np.int64)
                cb = cdf[b].cat.codes.values[perm].astype(np.int64)
                ea = cat_a.get_loc("") if "" in cat_a else -1
                eb = cat_b.get_loc("") if "" in cat_b else -1
                non_empty = (ca != -1) & (ca != ea) & (cb != -1) & (cb != eb)
                categories = None
                all_codes = ca * len(cat_b) + cb
                del ca, cb
            else:
                categories = cdf[col].cat.categories
                all_codes = cdf[col].cat.codes.values[perm]
                empty_code = categories.get_loc("") if "" in categories else -1
                non_empty = (all_codes != -1) & (all_codes != empty_code)
            codes = all_codes[non_empty]
            ids_here = id_codes[non_empty]

            order = np.argsort(codes, kind="stable")   # stable -> preserves the shuffled order within each bucket
            sorted_codes = codes[order]
            sorted_ids = ids_here[order].astype(np.int32)
            unique_codes, starts = np.unique(sorted_codes, return_index=True)
            boundaries = np.append(starts, len(sorted_codes)).astype(np.int64)

            col_index[col] = {
                "categories": categories,        # pandas Index of strings -- for value -> code lookup
                "unique_codes": unique_codes,     # sorted int array, one entry per unique non-empty value
                "boundaries": boundaries,         # start offset per unique_codes entry, +1 trailing sentinel
                "sorted_ids": sorted_ids,         # pool id_codes, grouped contiguously by code, randomly ordered WITHIN each code
            }
            stats["keys_built"] += 1
            stats["rows_indexed"] += int(non_empty.sum())
        index[country] = col_index
        gc.collect()
        print_mem(f"  [index] country={country}: done")
    return index, stats


train_index, train_index_stats = build_pool_index(train_pool_all, KEY_COLS)
print("index stats:", train_index_stats)
print_mem("after building train pool index")

# ==============================================================================
# ## 5. Streaming candidate generation + feature computation (shared by train & test)
#
# One function, reused for train (with labels + training/validation sampling) and
# test (with scoring + output writing). Processes `CHUNK_SIZE` S1 rows at a time;
# nothing about a finished chunk is kept. Pool text for the current chunk's
# candidates is read back from Parquet in a small, filtered batch — never the full
# pool's text at once.
#
# **Patch 9 (bottleneck #31)**: candidate generation is now fully vectorized — no
# per-row Python loop anywhere in this section. See `generate_candidates_chunk`'s
# docstring for the exact algorithm and how it was validated.
# ==============================================================================

def explode_key(codes, cap, unique_codes, boundaries, sorted_ids):
    """Vectorized: for each row's resolved code, take up to `cap` pool ids from
    the front of that code's (build-time-shuffled, see build_pool_index) bucket --
    equivalent in distribution to a random `cap`-sample of the bucket, with no
    per-lookup rng.choice call.

    Validated against a brute-force per-row reference across 300+ randomized
    synthetic (rows, pool size, cardinality, cap) combinations before use here --
    exact match on every trial, including edge cases (empty input, empty index,
    out-of-range / missing codes)."""
    n = len(codes)
    if n == 0 or len(unique_codes) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int32)
    pos = np.searchsorted(unique_codes, codes)
    pos_c = np.clip(pos, 0, len(unique_codes) - 1)
    valid = (codes >= 0) & (pos < len(unique_codes)) & (unique_codes[pos_c] == codes)
    starts = np.where(valid, boundaries[pos_c], 0)
    ends = np.where(valid, boundaries[pos_c + 1], 0)
    eff_len = np.where(valid, np.minimum(ends - starts, cap), 0)
    total = int(eff_len.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int32)
    row_idx = np.repeat(np.arange(n, dtype=np.int64), eff_len)
    group_start = np.cumsum(eff_len) - eff_len
    within = np.arange(total, dtype=np.int64) - np.repeat(group_start, eff_len)
    flat_pos = np.repeat(starts, eff_len) + within
    cand_ids = sorted_ids[flat_pos]
    return row_idx, cand_ids


PAIR_SHIFT = 4294967296   # 2**32 -- packs (row_idx, cand_id) into one int64 (row_idx * 2**32 + cand_id) for fast set-membership via np.isin


def generate_candidates_chunk(s1_chunk, index, key_cols, max_candidates=MAX_CANDIDATES_PER_ENTITY,
                               scan_window_mult=SCAN_WINDOW_MULT, verbose=False):
    """PATCH 9 (bottleneck #31) -- vectorized; loops over the 5 KEYS, never over rows.
    Replaces the get_bucket/per-row-loop design from patches 6-8, which a direct
    diagnostic (patch 8, run against real data) confirmed cost ~145-199s/chunk on
    its own -- get_indexer (~10s) and sampling were both separately ruled out first.

    Reproduces the ORIGINAL per-row loop's output EXACTLY (validated row-by-row,
    in priority order, across 300 randomized trials with heavy cross-key overlap):
      for each key in KEY_COLS order, each row takes up to per_key_cap candidates
      from that key's scan window (first sample_size ids of its bucket) that were
      NOT already taken by an earlier key; then found[:max_candidates].
    So every key keeps its guaranteed share of the budget -- a coarse key with huge
    buckets (key_name_prefix) cannot crowd out the address keys that lifted the
    Section 3.5 ceiling from 84.7% to 94%.

    Per key: explode_key() pulls each row's window in one vectorized pass;
    np.isin against the packed (row, cand) pairs already taken removes repeats;
    groupby(row).cumcount() < per_key_cap keeps the first per_key_cap NEW ones in
    window order. Oversized buckets are randomly subsampled for free because
    build_pool_index pre-shuffles each bucket's order (no rng.choice per lookup)."""
    per_key_cap = math.ceil(max_candidates / len(key_cols))
    sample_size = per_key_cap * scan_window_mult
    out_s1, out_cand = [], []

    for country, sub in s1_chunk.groupby("country", observed=True):
        country_index = index.get(country)
        if country_index is None:
            continue
        _t0 = time.time()
        id_codes = sub["id_code"].values
        n = len(sub)

        taken_rows, taken_cands = [], []
        taken_pair_keys = np.empty(0, dtype=np.int64)
        for col in key_cols:
            ci = country_index[col]
            if col in COMPOSITE_KEYS:
                # v13: resolve each component against the pool's categories, then combine exactly
                # like build_pool_index did; -1 (never matches) if either component is unknown
                a, b = COMPOSITE_KEYS[col]
                ra = country_index[a]["categories"].get_indexer(sub[a].astype(str)).astype(np.int64)
                rb = country_index[b]["categories"].get_indexer(sub[b].astype(str)).astype(np.int64)
                resolved = np.where((ra >= 0) & (rb >= 0), ra * len(country_index[b]["categories"]) + rb, -1)
            else:
                resolved = ci["categories"].get_indexer(sub[col].astype(str))
            r, c = explode_key(resolved, sample_size, ci["unique_codes"], ci["boundaries"], ci["sorted_ids"])
            if len(r) == 0:
                continue
            pair_keys = r * PAIR_SHIFT + c.astype(np.int64)
            if len(taken_pair_keys):
                fresh = ~np.isin(pair_keys, taken_pair_keys)
                r, c, pair_keys = r[fresh], c[fresh], pair_keys[fresh]
                if len(r) == 0:
                    continue
            rank = pd.Series(r).groupby(r).cumcount().values
            keep = rank < per_key_cap
            r, c, pair_keys = r[keep], c[keep], pair_keys[keep]
            taken_rows.append(r)
            taken_cands.append(c)
            taken_pair_keys = np.concatenate([taken_pair_keys, pair_keys])

        if taken_rows:
            all_r = np.concatenate(taken_rows)
            all_c = np.concatenate(taken_cands)
            rank = pd.Series(all_r).groupby(all_r).cumcount().values
            keep = rank < max_candidates   # safety net, same as the old found[:max_candidates]
            country_s1 = id_codes[all_r[keep]]
            country_cand = all_c[keep].astype(np.int32)
        else:
            country_s1 = np.empty(0, dtype=np.int32)
            country_cand = np.empty(0, dtype=np.int32)

        out_s1.append(country_s1)
        out_cand.append(country_cand)

        if verbose:
            print(f"    [candidates] country={country}: {n} rows -> {len(country_s1)} pairs -- "
                  f"vectorized gen {time.time()-_t0:.2f}s")

    if not out_s1:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    return np.concatenate(out_s1), np.concatenate(out_cand)


def compute_features_chunk(s1_ids, cand_ids, s1_text_indexed, pool_text_parquet_path, country_by_idcode, verbose=False):
    """PATCH 9/10: normalize() is now effectively called once per UNIQUE id on each
    side, not once per (s1, candidate) PAIR -- s1_ids/cand_ids both carry heavy
    repetition (each S1 entity repeats once per candidate it has; the candidate
    side was already deduplicated at the Parquet-read step but the old code then
    re-normalized it once per pair anyway). Uses normalize_series() (Section 1,
    vectorized pandas .str) instead of a per-row .apply(normalize) on top of the
    dedup. Both changes validated to produce identical output to the original
    scalar normalize() applied per pair (see project README, bottleneck #31)."""
    t0 = time.time()
    unique_cand_ids = np.unique(cand_ids)
    pool_text_chunk = pd.read_parquet(
        pool_text_parquet_path, filters=[("id_code", "in", unique_cand_ids.tolist())]
    ).set_index("id_code")
    t1 = time.time()

    unique_s1_ids = np.unique(s1_ids)
    s1_sub = s1_text_indexed.loc[unique_s1_ids]
    s1_name_norm = normalize_series(s1_sub["business_name"].astype(str))
    s1_addr_norm = normalize_series(s1_sub["business_address"].astype(str))
    cand_name_norm = normalize_series(pool_text_chunk["business_name"].astype(str))
    cand_addr_norm = normalize_series(pool_text_chunk["business_address"].astype(str))

    name_s1 = s1_name_norm.loc[s1_ids].values
    addr_s1 = s1_addr_norm.loc[s1_ids].values
    name_c = cand_name_norm.loc[cand_ids].values
    addr_c = cand_addr_norm.loc[cand_ids].values

    # v13: per-UNIQUE-id helpers (postal code, first name token, name length), then broadcast to pairs
    # (object dtype + to_numpy on purpose: plain numpy string comparisons in every pandas version)
    s1_postal = s1_sub["business_address"].astype(object).map(digits_only)
    cand_postal = pool_text_chunk["business_address"].astype(object).map(digits_only)
    post_s1 = s1_postal.loc[s1_ids].to_numpy(dtype=object)
    post_c = cand_postal.loc[cand_ids].to_numpy(dtype=object)
    ftok_s1 = s1_name_norm.str.split(" ").str[0].fillna("").loc[s1_ids].to_numpy(dtype=object)
    ftok_c = cand_name_norm.str.split(" ").str[0].fillna("").loc[cand_ids].to_numpy(dtype=object)
    len_s1 = s1_name_norm.str.len().loc[s1_ids].values.astype(np.float32)
    len_c = cand_name_norm.str.len().loc[cand_ids].values.astype(np.float32)
    del pool_text_chunk
    gc.collect()
    t_norm = time.time()

    name_s1, addr_s1, name_c, addr_c = list(name_s1), list(addr_s1), list(name_c), list(addr_c)
    f = {}
    # unchanged from v12 -- identical values (same scorer, same Python loop)
    f["name_sim"] = np.array([fuzz.token_sort_ratio(a, b) / 100 for a, b in zip(name_s1, name_c)], dtype=np.float32)
    f["addr_sim"] = np.array([fuzz.token_sort_ratio(a, b) / 100 for a, b in zip(addr_s1, addr_c)], dtype=np.float32)
    f["jaccard_name"] = np.array([token_jaccard(a, b) for a, b in zip(name_s1, name_c)], dtype=np.float32)
    f["jaccard_addr"] = np.array([token_jaccard(a, b) for a, b in zip(addr_s1, addr_c)], dtype=np.float32)
    # v13 new pair features
    f["name_ratio"] = pairwise_scores(fuzz.ratio, name_s1, name_c)
    f["name_partial"] = pairwise_scores(fuzz.partial_ratio, name_s1, name_c)
    f["name_tset"] = pairwise_scores(fuzz.token_set_ratio, name_s1, name_c)
    f["addr_tset"] = pairwise_scores(fuzz.token_set_ratio, addr_s1, addr_c)
    f["addr_partial"] = pairwise_scores(fuzz.partial_ratio, addr_s1, addr_c)
    both_postal = (post_s1 != "") & (post_c != "")
    f["postal_eq"] = np.where(both_postal, (post_s1 == post_c).astype(np.float32), np.float32(0.5)).astype(np.float32)
    f["first_tok_eq"] = ((ftok_s1 == ftok_c) & (ftok_s1 != "")).astype(np.float32)
    _mx = np.maximum(len_s1, len_c)
    f["name_len_ratio"] = np.where(_mx > 0, np.minimum(len_s1, len_c) / np.maximum(_mx, 1), 1.0).astype(np.float32)
    feats = pd.DataFrame(f)
    add_group_features(feats, s1_ids)
    t2 = time.time()

    if verbose:
        print(f"    [features] {len(s1_ids)} pairs, {len(unique_cand_ids)} unique candidates -- "
              f"parquet read {t1-t0:.1f}s, normalize (deduped+vectorized) {t_norm-t1:.1f}s, "
              f"{len(FEATURES)} features {t2-t_norm:.1f}s")

    return feats


def add_group_features(feats, s1_ids):
    """v13: features RELATIVE to the same S1 entity's other candidates (in place).
    Every candidate of an S1 entity is generated in the same chunk (chunks split S1
    rows, never an entity's candidates), and this runs on the FULL candidate set
    before any training-side negative subsampling, so the values are identical in
    train, validation and test."""
    g = pd.Series(s1_ids)
    feats["n_cands"] = g.groupby(s1_ids).transform("size").values.astype(np.float32)
    for col, out in (("name_sim", "name_gap"), ("addr_sim", "addr_gap"), ("name_tset", "tset_gap")):
        feats[out] = (feats[col].values - feats[col].groupby(s1_ids).transform("max").values).astype(np.float32)
    feats["sum_sim"] = (feats["name_sim"].values + feats["addr_sim"].values).astype(np.float32)
    feats["sum_gap"] = (feats["sum_sim"].values - feats["sum_sim"].groupby(s1_ids).transform("max").values).astype(np.float32)
    feats["sum_rank"] = feats["sum_sim"].groupby(s1_ids).rank(method="min", ascending=False).values.astype(np.float32)
    return feats


def iter_chunks(df, chunk_size):
    for start in range(0, len(df), chunk_size):
        yield df.iloc[start:start + chunk_size]

# ==============================================================================
# ## 6. Train pass: stream chunks, compute features + labels, sample a bounded training/validation set
#
# `train_s1` is capped to `TRAIN_FIT_SAMPLE_N` rows before this loop runs (Section 1)
# — only Section 10's TEST predictions are scored, so Section 6 doesn't need
# test-set scale. Every S1 entity in that capped set is pre-assigned to `train` or
# `val` (80/20 split on entity IDs). While streaming, `train`-side rows are
# downsampled (all positives kept, negatives capped per chunk); `val`-side rows are
# kept in full but explicitly capped to `VAL_MAX_ENTITIES` entities.
# ==============================================================================

s1_for_run = (train_s1.sample(n=min(SAMPLE_N, len(train_s1)), random_state=42)
              if SAMPLE_MODE else train_s1.sample(n=min(TRAIN_FIT_SAMPLE_N, len(train_s1)), random_state=42))

all_s1_ids = s1_for_run["id_code"].values
train_id_set, val_id_set = train_test_split(all_s1_ids, test_size=0.2, random_state=42)
train_id_set = set(train_id_set.tolist())

# VAL_MAX_ENTITIES actually enforced via an explicit membership check below (bottleneck #22's real fix)
val_id_set = set(list(val_id_set)[:VAL_MAX_ENTITIES])

train_fit_parts = []   # list of per-chunk feature DataFrames (FEATURES + s1_id, cand_id, label) -- train side, negatives downsampled
val_parts = []         # same shape -- val side, kept in full (capped at VAL_MAX_ENTITIES entities)

rng = np.random.default_rng(42)

_t_start = time.time()
_n_chunks = math.ceil(len(s1_for_run) / CHUNK_SIZE)
for _chunk_i, chunk in enumerate(iter_chunks(s1_for_run, CHUNK_SIZE)):
    _tc0 = time.time()
    s1_ids, cand_ids = generate_candidates_chunk(chunk, train_index, KEY_COLS, verbose=True)
    if len(s1_ids) == 0:
        print(f"  [train chunk {_chunk_i+1}/{_n_chunks}] 0 pairs, {time.time()-_tc0:.1f}s")
        continue
    feats = compute_features_chunk(
        s1_ids, cand_ids, train_s1_text, "cache/train_pool_text.parquet", train_country_by_idcode, verbose=True)
    label = np.array([1 if cc in gt_map.get(int(sid), ()) else 0 for sid, cc in zip(s1_ids, cand_ids)], dtype=np.int8)
    feats["s1_id"] = s1_ids
    feats["cand_id"] = cand_ids
    feats["label"] = label

    is_train_row = np.array([sid in train_id_set for sid in s1_ids])
    is_val_row = np.array([sid in val_id_set for sid in s1_ids])   # explicit check, not ~is_train_row (bottleneck #22)

    if is_val_row.any():
        val_parts.append(feats[is_val_row])

    if is_train_row.any():
        tr_idx = np.where(is_train_row)[0]
        pos_idx = tr_idx[label[tr_idx] == 1]
        neg_idx = tr_idx[label[tr_idx] == 0]
        if len(neg_idx) > NEG_CAP_PER_CHUNK:
            neg_idx = rng.choice(neg_idx, size=NEG_CAP_PER_CHUNK, replace=False)
        keep = np.concatenate([pos_idx, neg_idx])
        train_fit_parts.append(feats.iloc[keep])

    del s1_ids, cand_ids, feats, label
    gc.collect()
    _elapsed = time.time() - _t_start
    print(f"  [train chunk {_chunk_i+1}/{_n_chunks}] {time.time()-_tc0:.1f}s (elapsed {_elapsed:.1f}s, avg {_elapsed/(_chunk_i+1):.1f}s/chunk)")

print_mem("after streaming train pass (candidates + features + sampling)")

def assemble(parts, cols=tuple(["s1_id", "cand_id"] + FEATURES + ["label"])):
    if not parts:
        return pd.DataFrame(columns=list(cols))
    return pd.concat(parts, ignore_index=True)

train_fit = assemble(train_fit_parts)
val_scored = assemble(val_parts)
del train_fit_parts, val_parts
gc.collect()
print("train_fit:", train_fit.shape, " val_scored:", val_scored.shape)
print(train_fit["label"].value_counts())
print_mem("after assembling bounded train/val frames")

# ==============================================================================
# ## 7. Fit the model, tune the F_0.5 threshold on the validation set
# ==============================================================================

# ---- v13 (bottleneck #33): gradient-boosted trees instead of LogisticRegression ----
# HistGradientBoostingClassifier ships with scikit-learn (no install), trains in ~1-2 min on
# ~1M rows on Colab's 2 CPUs, and can learn non-linear rules a linear model can't (e.g. "high
# name_tset AND postal_eq=1 AND it's this entity's best candidate").
_t0 = time.time()
clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=63,
                                     min_samples_leaf=40, l2_regularization=1.0,
                                     early_stopping=True, validation_fraction=0.1,
                                     n_iter_no_change=20, random_state=42)
clf.fit(train_fit[FEATURES], train_fit["label"])
val_scored["proba"] = clf.predict_proba(val_scored[FEATURES])[:, 1]
print(f"HistGradientBoosting fit: {clf.n_iter_} trees, {time.time()-_t0:.1f}s on {len(train_fit):,} rows")

# ---- Vectorized macro-F_0.5 evaluator over ALL validation entities (incl. those with 0 candidates) ----
_val_ids = np.array(sorted(int(x) for x in val_id_set), dtype=np.int64)
_n_true = np.array([len(gt_map.get(int(s), ())) for s in _val_ids], dtype=np.float64)
_ent = np.searchsorted(_val_ids, val_scored["s1_id"].values.astype(np.int64))
_lab = val_scored["label"].values.astype(np.float64)
_E = len(_val_ids)

def macro_f05(pred_mask):
    """Same per-entity F_0.5 as the leaderboard: true matches missing from the candidate
    list count as misses; a true singleton scores 1 only if nothing is predicted."""
    pm = pred_mask.astype(np.float64)
    tp = np.bincount(_ent, weights=pm * _lab, minlength=_E)
    npred = np.bincount(_ent, weights=pm, minlength=_E)
    fp, fn = npred - tp, _n_true - tp
    denom = 1.25 * tp + 0.25 * fn + fp
    f = np.where(_n_true == 0, (npred == 0).astype(np.float64),
                 np.divide(1.25 * tp, denom, out=np.zeros_like(tp), where=denom > 0))
    return float(f.mean())

def is_group_top(proba, s1_ids):
    """True for the highest-probability candidate(s) of each S1 entity."""
    return proba >= pd.Series(proba).groupby(s1_ids).transform("max").values

# ---- v13 decision rule: accept a candidate if proba >= T_HI, OR it is its entity's top
# candidate and proba >= T_LO.  94% of entities have >=1 true match, and an empty prediction
# scores 0 for every one of them -- so the entity's best candidate deserves a much lower bar
# than its 2nd/3rd. T_LO == T_HI reproduces the v12 single-threshold rule exactly, and it is
# inside the grid, so validation can only pick the two-threshold rule if it scores higher.
_p = val_scored["proba"].values
_top = is_group_top(_p, val_scored["s1_id"].values)
best_score, T_HI, T_LO = -1.0, 0.5, 0.5
for t_hi in np.arange(0.05, 0.96, 0.025):
    for t_lo in np.arange(0.0, t_hi + 1e-9, 0.02):
        s = macro_f05((_p >= t_hi) | (_top & (_p >= t_lo)))
        if s > best_score:
            best_score, T_HI, T_LO = s, float(t_hi), float(t_lo)
_single = max((macro_f05(_p >= t), float(t)) for t in np.arange(0.05, 0.96, 0.025))
best_t = T_HI   # kept for backward compatibility with any cell/printout that references best_t

# ---- Same-run baseline: the v12 approach (LogisticRegression, 4 features, one threshold) ----
_lr = LogisticRegression(class_weight="balanced", max_iter=1000).fit(train_fit[V12_FEATURES], train_fit["label"])
_p_lr = _lr.predict_proba(val_scored[V12_FEATURES])[:, 1]
_v12 = max((macro_f05(_p_lr >= t), float(t)) for t in np.arange(0.05, 0.96, 0.025))
_oracle = macro_f05(_lab > 0)   # a PERFECT classifier on these candidates -> the ceiling blocking allows

print("\nValidation macro F_0.5 (same data, same candidates):")
print(f"  v12-style  LogReg, 4 features, 1 threshold          : {_v12[0]:.4f}  (t={_v12[1]:.3f})")
print(f"  v13 model  HistGB, {len(FEATURES)} features, 1 threshold        : {_single[0]:.4f}  (t={_single[1]:.3f})")
print(f"  v13 FULL   HistGB + top-candidate rule               : {best_score:.4f}  (T_HI={T_HI:.3f}, T_LO={T_LO:.3f})  <-- used for test")
print(f"  ceiling    perfect classifier on these candidates    : {_oracle:.4f}  (gap to this = blocking-limited)")
_pred_final = (_p >= T_HI) | (_top & (_p >= T_LO))
_pred_empty = np.bincount(_ent, weights=_pred_final.astype(np.float64), minlength=_E) == 0
print(f"Predicted singleton rate on val: {_pred_empty.mean():.3f}   (true singleton rate on val: {(_n_true == 0).mean():.3f})")
del _lr, _p_lr, _pred_final, _pred_empty

# Blocking recall on the same validation set -- your ceiling; matching can only hurt this, never fix it.
_val_groups = val_scored.groupby("s1_id")["cand_id"].apply(set)
hits = sum(len(gt_map.get(sid, set()) & _val_groups.get(sid, set())) for sid in gt_map if sid in val_id_set)
total = sum(len(v) for sid, v in gt_map.items() if sid in val_id_set)
print(f"Blocking recall on val: {hits/total:.1%}" if total else "No positive val entities to measure recall on.")

# ==============================================================================
# ## 8. Free train-stage memory before loading the (also large) test set
# ==============================================================================

for _name in ["train_s1", "train_pool_all", "train_s1_text", "train_index", "train_fit",
              "val_scored", "gt_map", "s1_for_run", "all_s1_ids", "train_id_set", "val_id_set",
              "train_country_by_idcode"]:
    globals().pop(_name, None)
gc.collect()
print_mem("after freeing train-stage memory")

# ==============================================================================
# ## 9. Apply the same pipeline to the TEST set — load, cache, index
# ==============================================================================

test_s1 = load_source(f"{TEST_DIR}/test_source1.tsv")
_ts2 = load_source(f"{TEST_DIR}/test_source2.tsv")
_ts3 = load_source(f"{TEST_DIR}/test_source3.tsv")
test_pool_all = pd.concat([_ts2, _ts3], ignore_index=True)
del _ts2, _ts3
gc.collect()

_tcodes, _tlengths, test_id_lookup = build_id_codec(test_s1, test_pool_all)
test_s1["id_code"] = _tcodes[:_tlengths[0]].astype("int32")
test_pool_all["id_code"] = _tcodes[_tlengths[0]:].astype("int32")
del _tcodes, _tlengths
gc.collect()

n_total_test_ids = len(test_id_lookup)
test_country_by_idcode = np.empty(n_total_test_ids, dtype=object)
test_country_by_idcode[test_s1["id_code"].values] = test_s1["country"].values
test_country_by_idcode[test_pool_all["id_code"].values] = test_pool_all["country"].values

test_pool_all[["id_code", "business_name", "business_address"]].to_parquet("cache/test_pool_text.parquet")
test_s1_text = test_s1.set_index("id_code")[["business_name", "business_address"]]

test_s1 = build_keys(test_s1, drop_text=False, label="test_s1")
test_pool_all = build_keys(test_pool_all, drop_text=True, label="test_pool_all")
gc.collect()

test_index, test_index_stats = build_pool_index(test_pool_all, KEY_COLS)
print("index stats:", test_index_stats)
print_mem("after test load + keys + index")

# ==============================================================================
# ## 10. Streaming test pass: candidates -> features -> score -> write to disk
#
# Writes `output/candidate_pairs.tsv` incrementally (append mode, one write per
# chunk) and a compact numeric temp file of matched (above-threshold) pairs — the
# 1-to-1 de-duplication in Section 11 reads that small temp file, not the full
# candidate set. Every test S1 entity is processed (never subsampled, even with
# `SAMPLE_MODE=False`) — every test entity needs a prediction to be scored.
#
# Watch the `projected total {…} min` figure printed after chunk 1: with candidate
# generation now vectorized, this should be dramatically lower than the ~9+ hour
# projections from before patch 9.
# ==============================================================================

s1_test_scope = (test_s1.sample(n=min(SAMPLE_N, len(test_s1)), random_state=42)
                  if SAMPLE_MODE else test_s1)
# NOT subsampled even in the real run -- every test entity needs a prediction to
# be scored on the leaderboard. test_s1_ids_scope in Section 11 must stay in sync
# with this (handled there via a SAMPLE_MODE check, not a manual reminder).

cand_pairs_path = "output/candidate_pairs.tsv"
matched_temp_path = "cache/matched_temp.parquet"
if os.path.exists(cand_pairs_path):
    os.remove(cand_pairs_path)

_matched_parts = []
_first_write = True
_t_start = time.time()
_n_chunks = math.ceil(len(s1_test_scope) / CHUNK_SIZE)

for _chunk_i, chunk in enumerate(iter_chunks(s1_test_scope, CHUNK_SIZE)):
    _tc0 = time.time()
    s1_ids, cand_ids = generate_candidates_chunk(chunk, test_index, KEY_COLS, verbose=True)

    # -- candidate_pairs.tsv: every S1 entity in this chunk, comma-joined candidates (empty if none) --
    chunk_ids = chunk["id_code"].values
    cand_by_entity = defaultdict(list)
    for sid, cid in zip(s1_ids, cand_ids):
        cand_by_entity[sid].append(cid)
    rows = pd.DataFrame({
        "source1_entity_id": test_id_lookup[chunk_ids],
        "candidate_entity_ids": [",".join(test_id_lookup[cand_by_entity.get(sid, [])]) for sid in chunk_ids],
    })
    rows.to_csv(cand_pairs_path, sep="\t", index=False, mode="a", header=_first_write)
    _first_write = False
    del rows, cand_by_entity

    if len(s1_ids) == 0:
        print(f"  [test chunk {_chunk_i+1}/{_n_chunks}] 0 pairs, {time.time()-_tc0:.1f}s")
        gc.collect()
        continue

    # -- scored candidates above threshold -> accumulate for 1-to-1 resolution --
    X = compute_features_chunk(
        s1_ids, cand_ids, test_s1_text, "cache/test_pool_text.parquet", test_country_by_idcode, verbose=True)
    proba = clf.predict_proba(X[FEATURES])[:, 1]
    # v13 decision rule (tuned in Section 7): all candidates >= T_HI, plus each entity's
    # top candidate if it is >= T_LO. Every candidate of an entity is in this chunk, so the
    # "top" computed here is the same as in validation.
    above = (proba >= T_HI) | (is_group_top(proba, s1_ids) & (proba >= T_LO))
    if above.any():
        _matched_parts.append((s1_ids[above], cand_ids[above], proba[above].astype(np.float32)))

    del s1_ids, cand_ids, X, proba, above
    gc.collect()

    _elapsed = time.time() - _t_start
    _avg = _elapsed / (_chunk_i + 1)
    print(f"  [test chunk {_chunk_i+1}/{_n_chunks}] {time.time()-_tc0:.1f}s "
          f"(elapsed {_elapsed:.1f}s, avg {_avg:.1f}s/chunk, projected total {_avg*_n_chunks/60:.1f} min)")

print_mem("after streaming test pass")

if _matched_parts:
    matched_all = pd.DataFrame({
        "s1_id": np.concatenate([p[0] for p in _matched_parts]),
        "cand_id": np.concatenate([p[1] for p in _matched_parts]),
        "proba": np.concatenate([p[2] for p in _matched_parts]),
    })
else:
    matched_all = pd.DataFrame(columns=["s1_id", "cand_id", "proba"])
del _matched_parts
gc.collect()
matched_all.to_parquet(matched_temp_path)
print("matched_all (pre-dedup):", matched_all.shape)
print_mem("after writing matched-candidates temp file")

# ==============================================================================
# ## 11. 1-to-1 de-duplication + final `matching_results.tsv`
#
# Reads back the small numeric temp file (not the full candidate set). If two
# Source-1 entities both claim the same candidate above threshold, only the
# higher-confidence claim survives -- a plain `groupby` + `idxmax` is enough at this
# size since it's only the matched subset, not the full candidate table.
# ==============================================================================

matched_all = pd.read_parquet(matched_temp_path)

if len(matched_all) > 0:
    # keep only the highest-proba claim per candidate (per-1-to-1 constraint)
    best_claim_idx = matched_all.groupby("cand_id")["proba"].idxmax()
    matched_dedup = matched_all.loc[best_claim_idx]
else:
    matched_dedup = matched_all

match_by_entity = matched_dedup.groupby("s1_id")["cand_id"].apply(list).to_dict()

# Self-resolving (bottleneck #28): equals test_s1 whenever SAMPLE_MODE=False, since
# s1_test_scope IS test_s1 in that case (Section 10) -- made explicit here instead
# of relying on a code comment telling a human to swap it by hand.
test_s1_ids_scope = test_s1["id_code"].values if not SAMPLE_MODE else s1_test_scope["id_code"].values
matching_results = pd.DataFrame({
    "source1_entity_id": test_id_lookup[test_s1_ids_scope],
    "matched_entity_ids": [",".join(test_id_lookup[match_by_entity.get(sid, [])]) for sid in test_s1_ids_scope],
})
matching_results.to_csv("output/matching_results.tsv", sep="\t", index=False)

print("Predicted singleton rate:", round((matching_results.matched_entity_ids == "").mean(), 3))

# PATCH 10 (bottleneck #32) diagnostic: singleton rate BROKEN OUT BY COUNTRY. France is
# unseen in training -- the classifier's threshold was tuned only on US/India validation
# data, so if France's singleton rate is far higher than US/India's, that points squarely
# at feature quality / threshold transfer for France, not a general blocking or model
# problem. Cheap (one groupby), highly informative for deciding where to spend remaining
# tuning time.
_singleton_by_country = (
    matching_results.assign(is_singleton=matching_results.matched_entity_ids == "")
    .assign(country=test_country_by_idcode[test_s1_ids_scope])
    .groupby("country")["is_singleton"].mean()
)
print("Predicted singleton rate by country:")
print(_singleton_by_country.round(3))

print_mem("after final 1-to-1 resolution + write")
matching_results.head()

# ==============================================================================
# ## 12. Self-check before you spend a real submission on it
# ==============================================================================

cp = pd.read_csv("output/candidate_pairs.tsv", sep="\t", dtype=str).fillna("")
mr = pd.read_csv("output/matching_results.tsv", sep="\t", dtype=str).fillna("")

checks = {
    "every test S1 id appears exactly once in candidate_pairs": cp["source1_entity_id"].is_unique,
    "every test S1 id appears exactly once in matching_results": mr["source1_entity_id"].is_unique,
    "candidate_pairs and matching_results cover the same S1 ids": set(cp.source1_entity_id) == set(mr.source1_entity_id),
}

# matches must be a subset of that row's own candidates
cp_map = dict(zip(cp.source1_entity_id, cp.candidate_entity_ids))
subset_ok = True
for sid, matches in zip(mr.source1_entity_id, mr.matched_entity_ids):
    if not matches:
        continue
    cand_set = set(cp_map.get(sid, "").split(","))
    if not set(matches.split(",")).issubset(cand_set):
        subset_ok = False
        break
checks["every matched id is a subset of that entity's own candidates"] = subset_ok

# global 1-to-1: no candidate id claimed by more than one S1 entity
all_matched = [m for matches in mr.matched_entity_ids if matches for m in matches.split(",")]
checks["no candidate id is matched to more than one S1 entity (1-to-1)"] = len(all_matched) == len(set(all_matched))

for name, ok in checks.items():
    print(("PASS" if ok else "FAIL"), "-", name)
