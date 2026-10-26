"""Uniform access to power-grid testbeds.

Phase-0 substrate: load any grid by a stable name into a pandapower net
behind ONE interface, so every downstream model/dataset is
testbed-agnostic. This module registers two sources behind the same
:func:`load_case` seam:

- the cases that ship with ``pandapower.networks`` (zero download,
  ``TESTBEDS`` registry below), and
- **PGLib-OPF v23.07** (:func:`list_pglib_cases`), an external clone at
  ``data/external/pglib-opf`` (66 base cases, 3-78,484 bus) loaded via
  ``pandapower.converter.matpower.from_mpc``. PGLib names are the file
  stem (e.g. ``"pglib_opf_case9241_pegase"``) and are dispatched by that
  ``pglib_opf_`` prefix, never mixed into ``TESTBEDS`` - they need a
  download and are not registry-consistency-checked the same way.
- **SimBench v1** (:func:`list_simbench_cases`), the ``simbench`` pip
  package's bundled benchmark grids, loaded via
  ``simbench.get_simbench_net``. Unlike PGLib, SimBench needs **no
  download and no network access at all** - the grid CSVs ship *inside*
  the installed package (``simbench.sb_dir/networks/``) and
  ``get_simbench_net`` only ever reads from there, so every registered
  code is always resolvable the moment ``simbench`` is importable (a
  declared ``environment.yml`` dependency). SimBench codes are
  hyphen-delimited (e.g. ``"1-LV-rural1--0-sw"``) with no distinguishing
  prefix PGLib-style dispatch could key on, so they are dispatched by
  **set membership** in :data:`SIMBENCH_CODES` instead, never mixed into
  ``TESTBEDS``.

  ACTIVSg loaders slot in behind the same seam next.

The registry deliberately covers a *realism spread*:

- ``classic``  IEEE / textbook transmission benchmarks (small, unrealistic
  but ubiquitous in the literature - good for method development).
- ``pegase``   PEGASE European synthetic transmission (89-9241 bus).
- ``rte``      RTE French transmission - real network topology (1888-6515 bus).
- ``other``    GB-reduced, Polish, Iceland, and the Illinois-200 synthetic
  geo-realistic grid.

PGLib cases are NOT part of ``FAMILIES``/``TESTBEDS`` (they need a download
and aren't registry-consistency-checked the same way); :func:`case_stats`
and :func:`inventory` report their pseudo-family as ``"pglib"`` regardless.

Note: ``case6495rte`` carries **6** ``ext_grid`` rows (its siblings have 1),
i.e. 6 slack/REF buses - fine for pandapower/``reference_pf`` (both handle
multiple REF buses structurally), but a footgun for any code that assumes
"the" single slack bus/angle.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from functools import lru_cache as _lru_cache
from pathlib import Path

import pandapower as pp
import pandapower.networks as nw

# External-download cases live here by default (gitignored `data/`); override
# with GRID_OPS_PGLIB_DIR for an OSC scratch path or an alternate clone.
PGLIB_DIR = Path(os.environ.get(
    "GRID_OPS_PGLIB_DIR",
    Path(__file__).resolve().parents[2] / "data" / "external" / "pglib-opf",
))

# --- registry -------------------------------------------------------------
# name -> family. The loader is ``getattr(pandapower.networks, name)`` and
# takes no arguments. Only no-arg transmission cases are registered here;
# parameterized distribution feeders (CIGRE / Kerber / mv_oberrhein) get a
# dedicated entry point later.
FAMILIES: dict[str, str] = {
    "classic": "IEEE / textbook transmission benchmark",
    "pegase": "PEGASE European synthetic transmission",
    "rte": "RTE French transmission (real topology)",
    "other": "GB-reduced / Polish / Iceland / Illinois-200 synthetic",
}

_CLASSIC = [
    "case4gs", "case5", "case6ww", "case9", "case11_iwamoto", "case14",
    "case24_ieee_rts", "case30", "case_ieee30", "case33bw", "case39",
    "case57", "case118", "case145", "case300",
]
_PEGASE = ["case89pegase", "case1354pegase", "case2869pegase", "case9241pegase"]
_RTE = ["case1888rte", "case2848rte", "case6470rte", "case6495rte", "case6515rte"]
_OTHER = ["GBreducednetwork", "GBnetwork", "case3120sp", "iceland", "case_illinois200"]

TESTBEDS: dict[str, str] = {
    **{n: "classic" for n in _CLASSIC},
    **{n: "pegase" for n in _PEGASE},
    **{n: "rte" for n in _RTE},
    **{n: "other" for n in _OTHER},
}


def list_testbeds(family: str | None = None) -> list[str]:
    """Return registered testbed names, optionally filtered by family."""
    if family is None:
        return list(TESTBEDS)
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; choose from {list(FAMILIES)}")
    return [n for n, fam in TESTBEDS.items() if fam == family]


_PGLIB_PREFIX = "pglib_opf_"


def list_pglib_cases() -> list[str]:
    """PGLib-OPF base-case names available under ``PGLIB_DIR``.

    Empty if the repo hasn't been cloned there yet (no error - callers can
    treat "0 cases" as "not downloaded"). Excludes the ``api/``/``sad/``
    stress-variant subdirectories; those need their own loader if wanted.
    """
    if not PGLIB_DIR.is_dir():
        return []
    return sorted(p.stem for p in PGLIB_DIR.glob(f"{_PGLIB_PREFIX}*.m"))


def _load_pglib(name: str) -> "pp.pandapowerNet":
    from pandapower.converter.matpower import from_mpc

    path = PGLIB_DIR / f"{name}.m"
    if not path.is_file():
        raise KeyError(
            f"pglib case {name!r} not found at {path}; clone PGLib-OPF v23.07 into "
            f"{PGLIB_DIR} (or set GRID_OPS_PGLIB_DIR) - see CLAUDE.md"
        )
    return from_mpc(str(path))


def load_case(name: str) -> "pp.pandapowerNet":
    """Load a testbed (by TESTBEDS registry name, PGLib case name, or
    SimBench code) into a pandapower net.

    The single seam every downstream consumer goes through. A
    ``pglib_opf_``-prefixed name dispatches to the PGLib-OPF loader
    (:func:`_load_pglib`); a name in :data:`SIMBENCH_CODES` dispatches to the
    SimBench loader (:func:`_load_simbench`); everything else must be a
    registered zero-download ``TESTBEDS`` name. An ACTIVSg loader will slot
    in here next.
    """
    if name.startswith(_PGLIB_PREFIX):
        return _load_pglib(name)
    if name in SIMBENCH_CODES:
        return _load_simbench(name)
    if name not in TESTBEDS:
        raise KeyError(
            f"unknown testbed {name!r}; {len(TESTBEDS)} registered "
            f"(e.g. {', '.join(list(TESTBEDS)[:5])}, ...), a "
            f"{_PGLIB_PREFIX}* name from list_pglib_cases(), or a SimBench "
            f"code from list_simbench_cases()"
        )
    loader = getattr(nw, name)
    return loader()


def _family_of(name: str) -> str:
    """TESTBEDS family, ``"pglib"``, or ``"simbench"`` for those names."""
    if name.startswith(_PGLIB_PREFIX):
        return "pglib"
    if name in SIMBENCH_CODES:
        return "simbench"
    return TESTBEDS[name]


# ---------------------------------------------------------------------------
# Grid-family taxonomy and size classes (the leave-one-out split axes)
# ---------------------------------------------------------------------------
# WHY THIS EXISTS. The held-out-grid experiment splits two ways, and they must
# be split SEPARATELY because every published zero-shot power-flow result
# confounds them: a model evaluated on an unseen grid is usually evaluated on a
# grid that is both structurally different AND a different size, so a failure
# cannot be attributed to either. `family_of` supports leave-one-FAMILY-out;
# `size_class_of` supports leave-one-SIZE-CLASS-out within a family.
#
# A "family" here means PROVENANCE: which synthetic-grid programme or real
# network a case came from. Cases inside a family share construction
# conventions (voltage levels, branch-parameter distributions, generator
# placement heuristics, load models), which is exactly the kind of regularity a
# surrogate can overfit to and a held-out family therefore tests.
#
# `TESTBEDS`' original four labels (classic/pegase/rte/other) are kept as-is so
# `list_testbeds(family=...)` does not change meaning; this taxonomy is a
# second, finer axis that also spans PGLib and SimBench.

_PGLIB_FAMILY_SUFFIX: dict[str, str] = {
    # suffix on the PGLib stem -> family. Order matters: longest match wins,
    # so `ieee_rts` and `ieee_dtc` resolve to `ieee` via the prefix check below.
    # `classic` is the IEEE-lineage benchmark set. It holds the true IEEE
    # cases plus the PJM 5-bus, EPRI 39-bus and PSERC 240-bus cases that the
    # literature conventionally groups with them; calling the family "ieee"
    # would put a per-family table's name at odds with its contents.
    "ieee": "classic", "pjm": "classic", "pserc": "classic", "epri": "classic",
    "pegase": "pegase",
    "rte": "rte",
    "goc": "goc",
    "epigrids": "epigrids",
    "sdet": "sdet",
    "snem": "snem",
    "activ": "activsg",
    "k": "polish",          # MATPOWER Polish cases: 2383wp_k, 3120sp_k, ...
    "lmbd": "misc", "as": "misc", "c": "misc",
}

_BUILTIN_FAMILY: dict[str, str] = {
    # IEEE / textbook transmission benchmarks
    **{n: "classic" for n in ("case4gs", "case5", "case6ww", "case9", "case11_iwamoto",
                              "case14", "case24_ieee_rts", "case30", "case_ieee30",
                              "case39", "case57", "case118", "case145", "case300")},
    **{n: "pegase" for n in ("case89pegase", "case1354pegase", "case2869pegase",
                             "case9241pegase")},
    **{n: "rte" for n in ("case1888rte", "case2848rte", "case6470rte",
                          "case6495rte", "case6515rte")},
    "case3120sp": "polish",
    "case_illinois200": "activsg",
    "case33bw": "distribution_feeder",   # 33-bus radial distribution feeder
    "GBreducednetwork": "gb", "GBnetwork": "gb",
    "iceland": "misc",
}

# Measured 2026-09-13 on this repo's own loaders, not taken from the literature.
# Branch/bus is (lines + transformers) / buses: a radial tree tends to 1.0 from
# below, a meshed transmission network sits well above it. Median R/X is over
# lines with x > 0.
#
#   family                 br/bus   median R/X
#   classic (case118)        1.58        0.280
#   pegase  (case1354)       1.47        0.156
#   rte     (case1888)       1.34        0.150
#   goc     (case500)        1.29        0.172
#   polish  (case3120sp)     1.18        0.309
#   epigrids(case5658)       1.14        0.349
#   simbench EHV  (no_sw)    1.85        0.100
#   simbench HV   (no_sw)    1.41        0.370
#   simbench MV   (no_sw)    1.06        2.184   <-- different regime
#   simbench LV   (no_sw)    0.93        2.570   <-- different regime
#
# The MV/LV grids are near-radial with an R/X ratio seven to seventeen times the
# transmission median, and their bus-type mix differs too (SimBench models
# distributed generation as `sgen`, i.e. negative load at PQ buses, so a 64-bus
# HV zone carries 3 generators and 103 sgen). Bottcher et al. found exactly this
# node-type shift, not size, was what broke their surrogate on SimBench.
REGIME: dict[str, str] = {
    "classic": "transmission", "pegase": "transmission", "rte": "transmission",
    "goc": "transmission", "polish": "transmission", "epigrids": "transmission",
    "sdet": "transmission", "snem": "transmission", "activsg": "transmission",
    "gb": "transmission", "misc": "transmission",
    "simbench_ehv": "transmission", "simbench_hv": "transmission",
    "simbench_mv": "distribution", "simbench_lv": "distribution",
    "distribution_feeder": "distribution",
}

# The leave-one-family-out ladder. Membership decisions, with reasons:
#
#  INCLUDED
#   classic, pegase, rte, goc, polish - the established transmission families.
#   epigrids                        - synthetic transmission, br/bus 1.14 and
#                                     R/X 0.349 put it inside the transmission
#                                     spread; it is also the only route above
#                                     30k buses. It is the newest and least
#                                     studied family, so holding it out is the
#                                     hardest fold, which is a feature.
#   simbench_ehv, simbench_hv       - meshed transmission (br/bus 1.85 and 1.41
#                                     on the no_sw representation). Use `no_sw`:
#                                     the `sw` variant inserts switch-only buses
#                                     that make br/bus read 0.33 and would
#                                     corrupt any structural statistic.
#
#  EXCLUDED from the ladder, kept as a separate labelled probe
#   simbench_mv, simbench_lv        - radial distribution. Including them would
#                                     confound topology shift with PHYSICS-REGIME
#                                     shift, which is the exact confound this
#                                     experiment exists to remove. Reported apart
#                                     as a transmission-to-distribution OOD probe,
#                                     which is a real question and nearly free.
#   sdet, snem, activsg, gb, misc   - too few cases to form a fold (1 to 3 each).
#                                     They stay in the corpus as extra in-fold
#                                     training data, never as a held-out family.
LADDER_FAMILIES: tuple[str, ...] = (
    "classic", "pegase", "rte", "goc", "polish", "epigrids",
    "simbench_ehv", "simbench_hv",
)
PROBE_FAMILIES: tuple[str, ...] = ("simbench_mv", "simbench_lv")

# The ONE SimBench switch representation the ladder (and every list built for a
# split or a corpus) uses, per the `no_sw` decision above. Each SimBench grid
# ships as both `sw` and `no_sw`; counting both put the same grid in the ladder
# twice, and in two different size classes (1-EHV-mixed is 571 buses as no_sw,
# 3,085 as sw). Enforced in `is_ladder_representation`, `list_by_family` and
# `inventory_by_split` since 2026-10-06 (audit null-baseline R1-F3; ladder
# 99 -> 90 cases).
LADDER_SIMBENCH_REP = "no_sw"


def is_ladder_representation(name: str) -> bool:
    """False only for a SimBench code in the non-ladder switch representation.

    Every PGLib and built-in case passes; a SimBench code passes only if it ends
    in ``-{LADDER_SIMBENCH_REP}``. This does not test family membership.
    """
    if name not in SIMBENCH_CODES:
        return True
    return name.rsplit("-", 1)[-1] == LADDER_SIMBENCH_REP


def _simbench_family(code: str) -> str:
    level = code.split("-")[1]
    return {"EHV": "simbench_ehv", "HV": "simbench_hv",
            "MV": "simbench_mv", "LV": "simbench_lv"}[level]


def family_of(name: str) -> str:
    """Provenance family of a registered case, across all three sources.

    This is the leave-one-family-out split key. See :data:`LADDER_FAMILIES` for
    which families form the ladder and why the others do not.
    """
    if name.startswith(_PGLIB_PREFIX):
        stem = name[len(_PGLIB_PREFIX):]            # e.g. "case2736sp_k"
        parts = stem.split("_")[1:]                 # drop "caseNNNN..."
        if not parts:
            return "misc"
        for key in ("ieee", "pegase", "rte", "goc", "epigrids", "sdet",
                    "snem", "activ", "pserc", "epri", "pjm"):
            if parts[0].startswith(key):
                return _PGLIB_FAMILY_SUFFIX[key]
        return _PGLIB_FAMILY_SUFFIX.get(parts[-1], "misc")
    if name in SIMBENCH_CODES:
        return _simbench_family(name)
    if name in _BUILTIN_FAMILY:
        return _BUILTIN_FAMILY[name]
    raise KeyError(f"unknown case {name!r}; not a PGLib name, SimBench code, or built-in")


def regime_of(name: str) -> str:
    """``"transmission"`` or ``"distribution"``. See :data:`REGIME`."""
    return REGIME[family_of(name)]


# Bus counts. PGLib encodes it in the name; the built-ins are measured once and
# pinned here (test_testbeds cross-checks); SimBench is loaded on demand because
# the count depends on the switch representation.
_BUILTIN_BUS: dict[str, int] = {
    "case4gs": 4, "case5": 5, "case6ww": 6, "case9": 9, "case11_iwamoto": 11,
    "case14": 14, "case24_ieee_rts": 24, "GBreducednetwork": 29, "case30": 30,
    "case_ieee30": 30, "case33bw": 33, "case39": 39, "case57": 57,
    "case89pegase": 89, "case118": 118, "case145": 145, "iceland": 189,
    "case_illinois200": 200, "case300": 300, "case1354pegase": 1354,
    "case1888rte": 1888, "GBnetwork": 2224, "case2848rte": 2848,
    "case2869pegase": 2869, "case3120sp": 3120, "case6470rte": 6470,
    "case6495rte": 6495, "case9241pegase": 9241, "case6515rte": 6515,
}

# Measured 2026-09-13. SimBench bus counts depend on the switch representation
# (`sw` keeps explicit switch-only buses, `no_sw` fuses them away), so they
# cannot be read off the code and are pinned here to keep split construction
# from paying 78 grid loads.
_SIMBENCH_BUS: dict[str, int] = {
    "1-EHV-mixed--0-no_sw": 571,
    "1-EHV-mixed--0-sw": 3085,
    "1-EHV-mixed--1-no_sw": 571,
    "1-EHV-mixed--1-sw": 3085,
    "1-EHV-mixed--2-no_sw": 571,
    "1-EHV-mixed--2-sw": 3085,
    "1-HV-mixed--0-no_sw": 64,
    "1-HV-mixed--0-sw": 306,
    "1-HV-mixed--1-no_sw": 94,
    "1-HV-mixed--1-sw": 355,
    "1-HV-mixed--2-no_sw": 114,
    "1-HV-mixed--2-sw": 377,
    "1-HV-urban--0-no_sw": 82,
    "1-HV-urban--0-sw": 372,
    "1-HV-urban--1-no_sw": 100,
    "1-HV-urban--1-sw": 402,
    "1-HV-urban--2-no_sw": 120,
    "1-HV-urban--2-sw": 428,
    "1-LV-rural1--0-no_sw": 15,
    "1-LV-rural1--0-sw": 15,
    "1-LV-rural1--1-no_sw": 15,
    "1-LV-rural1--1-sw": 15,
    "1-LV-rural1--2-no_sw": 15,
    "1-LV-rural1--2-sw": 15,
    "1-LV-rural2--0-no_sw": 97,
    "1-LV-rural2--0-sw": 97,
    "1-LV-rural2--1-no_sw": 97,
    "1-LV-rural2--1-sw": 97,
    "1-LV-rural2--2-no_sw": 97,
    "1-LV-rural2--2-sw": 97,
    "1-LV-rural3--0-no_sw": 129,
    "1-LV-rural3--0-sw": 129,
    "1-LV-rural3--1-no_sw": 129,
    "1-LV-rural3--1-sw": 129,
    "1-LV-rural3--2-no_sw": 129,
    "1-LV-rural3--2-sw": 129,
    "1-LV-semiurb4--0-no_sw": 44,
    "1-LV-semiurb4--0-sw": 44,
    "1-LV-semiurb4--1-no_sw": 44,
    "1-LV-semiurb4--1-sw": 44,
    "1-LV-semiurb4--2-no_sw": 44,
    "1-LV-semiurb4--2-sw": 44,
    "1-LV-semiurb5--0-no_sw": 111,
    "1-LV-semiurb5--0-sw": 111,
    "1-LV-semiurb5--1-no_sw": 111,
    "1-LV-semiurb5--1-sw": 111,
    "1-LV-semiurb5--2-no_sw": 111,
    "1-LV-semiurb5--2-sw": 111,
    "1-LV-urban6--0-no_sw": 59,
    "1-LV-urban6--0-sw": 59,
    "1-LV-urban6--1-no_sw": 59,
    "1-LV-urban6--1-sw": 59,
    "1-LV-urban6--2-no_sw": 59,
    "1-LV-urban6--2-sw": 59,
    "1-MV-comm--0-no_sw": 103,
    "1-MV-comm--0-sw": 107,
    "1-MV-comm--1-no_sw": 106,
    "1-MV-comm--1-sw": 110,
    "1-MV-comm--2-no_sw": 107,
    "1-MV-comm--2-sw": 111,
    "1-MV-rural--0-no_sw": 95,
    "1-MV-rural--0-sw": 97,
    "1-MV-rural--1-no_sw": 97,
    "1-MV-rural--1-sw": 99,
    "1-MV-rural--2-no_sw": 97,
    "1-MV-rural--2-sw": 99,
    "1-MV-semiurb--0-no_sw": 115,
    "1-MV-semiurb--0-sw": 117,
    "1-MV-semiurb--1-no_sw": 120,
    "1-MV-semiurb--1-sw": 122,
    "1-MV-semiurb--2-no_sw": 120,
    "1-MV-semiurb--2-sw": 122,
    "1-MV-urban--0-no_sw": 136,
    "1-MV-urban--0-sw": 144,
    "1-MV-urban--1-no_sw": 136,
    "1-MV-urban--1-sw": 144,
    "1-MV-urban--2-no_sw": 136,
    "1-MV-urban--2-sw": 144,
}

_SIZE_BINS: tuple[tuple[int, str], ...] = (
    (100, "tiny"), (1_000, "small"), (5_000, "medium"),
    (20_000, "large"), (10**9, "xlarge"),
)


@_lru_cache(maxsize=None)
def bus_count(name: str) -> int:
    """Bus count of a registered case, without loading it where possible.

    PGLib names carry the count; the built-ins and the SimBench codes are pinned
    in tables measured once (``test_testbeds`` cross-checks a sample against a
    live load, so a pandapower or simbench upgrade that changes a case is caught
    rather than silently reclassifying its size). The cache matters because
    ``size_class_of`` is called once per case per fold when a split is built.
    """
    import re
    if name.startswith(_PGLIB_PREFIX):
        m = re.search(r"case(\d+)", name)
        if not m:
            raise ValueError(f"cannot read a bus count out of {name!r}")
        return int(m.group(1))
    if name in _BUILTIN_BUS:
        return _BUILTIN_BUS[name]
    if name in SIMBENCH_CODES:
        if name in _SIMBENCH_BUS:
            return _SIMBENCH_BUS[name]
        return len(load_case(name).bus)          # depends on the switch representation
    raise KeyError(f"unknown case {name!r}")


def size_class_of(name: str) -> str:
    """Size class of a case: tiny / small / medium / large / xlarge.

    Bins are <100, <1k, <5k, <20k, and the rest. They are chosen so that the
    transmission ladder has at least two families represented in every class
    that is used as a held-out fold; `inventory_by_split()` prints the crosstab
    to check that before a sweep is designed.
    """
    n = bus_count(name)
    for hi, label in _SIZE_BINS:
        if n < hi:
            return label
    return "xlarge"


def list_by_family(family: str, *, include_pglib: bool = True,
                   include_simbench: bool = True,
                   all_switch_reps: bool = False) -> list[str]:
    """Every registered case in one family, sorted by bus count.

    SimBench codes come in the ladder representation only
    (:data:`LADDER_SIMBENCH_REP`) unless ``all_switch_reps`` is set, so each grid
    is listed once (2026-10-06, audit null-baseline R1-F3).
    """
    names = list(TESTBEDS)
    if include_pglib:
        names += list_pglib_cases()
    if include_simbench:
        names += list_simbench_cases()
    out = [n for n in names if family_of(n) == family
           and (all_switch_reps or is_ladder_representation(n))]
    return sorted(out, key=bus_count)


def list_families() -> list[str]:
    """Every family present in the registry, ladder families first."""
    seen = {family_of(n) for n in TESTBEDS}
    seen |= {family_of(n) for n in list_pglib_cases()}
    seen |= {family_of(n) for n in list_simbench_cases()}
    ladder = [f for f in LADDER_FAMILIES if f in seen]
    return ladder + sorted(seen - set(ladder))


def inventory_by_split(*, ladder_only: bool = True) -> dict[tuple[str, str], list[str]]:
    """Crosstab of (family, size class) -> case names.

    The thing to look at before designing a sweep: a leave-one-family-out fold
    is only meaningful if the remaining families still cover the held-out
    family's size classes, and a leave-one-size-class-out fold is only
    meaningful if that class has more than one family in it.

    With ``ladder_only`` (the default) SimBench grids appear once, in the
    :data:`LADDER_SIMBENCH_REP` representation: 90 cases as of 2026-10-06 (was 99,
    which counted the 9 SimBench EHV/HV grid-scenario pairs twice; audit
    null-baseline R1-F3). ``ladder_only=False`` keeps every registered code.
    """
    names = list(TESTBEDS) + list_pglib_cases() + list_simbench_cases()
    out: dict[tuple[str, str], list[str]] = {}
    for n in names:
        fam = family_of(n)
        if ladder_only and (fam not in LADDER_FAMILIES or not is_ladder_representation(n)):
            continue
        out.setdefault((fam, size_class_of(n)), []).append(n)
    return {k: sorted(v, key=bus_count) for k, v in sorted(out.items())}


# --- SimBench --------------------------------------------------------------
# SimBench's own official benchmark-grid catalog: 13 single voltage-level /
# grid-type zones x 3 scenarios (0/1/2 -- SimBench's own present-day /
# near-future / future scenario numbering; confirmed empirically that 1 and 2
# introduce battery `storage` elements the 0 scenario doesn't carry) x 2
# switch representations (``sw`` = explicit bus-bus switches, ``no_sw`` =
# switches pre-fused into the topology) = 78 codes. This is the
# ``lv_level="", all_data=False`` slice of
# ``simbench.collect_all_simbench_codes()`` -- i.e. every code EXCEPT the
# combined multi-voltage-level extracts (``EHVHV``, ``MVLV``, ... -- whose
# ``lv_grid`` attachment-point parameter blows the combinatorial space up to
# 246 total codes for comparatively niche multi-zone use cases) and the 3
# ``complete_data`` mega-grids (the full EHV-through-LV combined network,
# ~3000+ bus, a different research use case than a "testbed"). Hardcoded
# (mirrors how ``TESTBEDS`` itself is a static registry, not generated by
# introspecting ``pandapower.networks``) so importing this module never pays
# a ``simbench`` import; ``test_testbeds.py`` cross-checks this set against
# ``simbench.collect_all_simbench_codes()`` directly.
_SIMBENCH_VOLTAGE_TYPES = [
    ("EHV", "mixed"),
    ("HV", "mixed"), ("HV", "urban"),
    ("MV", "rural"), ("MV", "semiurb"), ("MV", "urban"), ("MV", "comm"),
    ("LV", "rural1"), ("LV", "rural2"), ("LV", "rural3"),
    ("LV", "semiurb4"), ("LV", "semiurb5"), ("LV", "urban6"),
]
_SIMBENCH_SCENARIOS = ("0", "1", "2")
_SIMBENCH_SWITCH_REPS = ("sw", "no_sw")

SIMBENCH_CODES: frozenset[str] = frozenset(
    f"1-{level}-{type_}--{scenario}-{sw}"
    for level, type_ in _SIMBENCH_VOLTAGE_TYPES
    for scenario in _SIMBENCH_SCENARIOS
    for sw in _SIMBENCH_SWITCH_REPS
)


def list_simbench_cases() -> list[str]:
    """The curated SimBench benchmark-grid codes registered here (78).

    Unlike :func:`list_pglib_cases`, this needs no directory/download check
    and is never empty: SimBench ships its grid CSVs bundled inside the
    installed ``simbench`` pip package, so every code here is always
    resolvable by :func:`_load_simbench` with no network access.
    """
    return sorted(SIMBENCH_CODES)


def _load_simbench(code: str) -> "pp.pandapowerNet":
    import simbench as sb

    return sb.get_simbench_net(code)


# --- inventory / analysis -------------------------------------------------
@dataclass
class CaseStats:
    """Size, topology, and AC power-flow summary for one testbed."""

    name: str
    family: str
    n_bus: int
    n_line: int
    n_trafo: int
    n_gen: int          # generators + ext_grid + static gens
    n_load: int
    load_mw: float      # total active load
    converged: bool
    solve_ms: float
    vm_min: float
    vm_max: float
    va_span_deg: float  # max - min bus voltage angle


def case_stats(name: str) -> CaseStats:
    """Load one testbed, run AC power flow, and summarize it.

    Power flow is run with ``numba=False`` for determinism / to silence the
    optional-dependency warning; install numba before bulk label generation.
    Non-convergence is captured, not raised, so the inventory never aborts.
    """
    net = load_case(name)
    n_gen = len(net.gen) + len(net.ext_grid) + len(net.sgen)
    load_mw = float(net.load.p_mw.sum()) if len(net.load) else 0.0

    converged, solve_ms = True, float("nan")
    vm_min = vm_max = va_span = float("nan")
    t0 = time.perf_counter()
    try:
        pp.runpp(net, numba=False)
        solve_ms = (time.perf_counter() - t0) * 1e3
        vm_min = float(net.res_bus.vm_pu.min())
        vm_max = float(net.res_bus.vm_pu.max())
        va_span = float(net.res_bus.va_degree.max() - net.res_bus.va_degree.min())
    except Exception:
        converged = False
        solve_ms = (time.perf_counter() - t0) * 1e3

    return CaseStats(
        name=name, family=_family_of(name),
        n_bus=len(net.bus), n_line=len(net.line), n_trafo=len(net.trafo),
        n_gen=n_gen, n_load=len(net.load), load_mw=load_mw,
        converged=converged, solve_ms=solve_ms,
        vm_min=vm_min, vm_max=vm_max, va_span_deg=va_span,
    )


def inventory(names: list[str] | None = None) -> list[CaseStats]:
    """Run :func:`case_stats` over every registered testbed (or a subset),
    ordered by bus count."""
    names = names or list(TESTBEDS)
    stats = [case_stats(n) for n in names]
    return sorted(stats, key=lambda s: s.n_bus)


def _print_inventory(stats: list[CaseStats]) -> None:
    hdr = (f"{'testbed':<18}{'family':<8}{'bus':>6}{'line':>6}{'traf':>6}"
           f"{'gen':>5}{'load':>6}{'load_MW':>10}{'conv':>6}{'ms':>9}"
           f"{'vm_min':>8}{'vm_max':>8}{'va_deg':>8}")
    print(hdr)
    print("-" * len(hdr))
    for s in stats:
        conv = "ok" if s.converged else "FAIL"
        print(f"{s.name:<18}{s.family:<8}{s.n_bus:>6}{s.n_line:>6}{s.n_trafo:>6}"
              f"{s.n_gen:>5}{s.n_load:>6}{s.load_mw:>10.1f}{conv:>6}{s.solve_ms:>9.1f}"
              f"{s.vm_min:>8.3f}{s.vm_max:>8.3f}{s.va_span_deg:>8.1f}")


if __name__ == "__main__":
    _print_inventory(inventory())
