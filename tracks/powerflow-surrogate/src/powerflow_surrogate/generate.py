"""Operating-point generation for the held-out-grid ladder.

One sample is: a grid, a perturbed operating point on it, and the solved voltage
state that the power-flow equations give at that point. The labels come from
``reference_pf.solve_reference``, so every sample is checked against a second
implementation (pandapower) rather than trusted on convergence, and every kept
label is verified as a root of our own equations. With pandapower's default
initialization the two share a start, so the check verifies the implementation,
not the choice of root (audit 2026-10-06, R1-F5; see solve_reference). See
``results/2026-09-13-family-taxonomy-and-oracle/report.md`` for why that check
exists: on ``case2848rte`` a flat-start solve converges to a residual of 1.4e-11
and is 1.01 pu away from the right answer.

Three design choices, each forced by a measurement rather than chosen:

1. **Topology is fixed within a shard**, and the case object is built once
   rather than reloaded per sample: reloading costs roughly 1.5 s against a
   per-sample cost in the hundreds of milliseconds, so it would dominate. N-1 and
   N-2 variants are generated as SEPARATE shards, each with its own fixed
   topology, which keeps that saving and still varies connectivity across the set.

   Measured per-sample cost, on a box at load 30 to 50: **184 ms at 118 buses,
   384 ms at 1,354**. Roughly half of that is pandapower solving the case itself,
   which point 2 below explains is not optional, and a further chunk is rebuilding
   the admittance matrix inside ``from_pandapower`` on every call. Caching the
   admittance matrix across a shard would cut the 118-bus figure by about half and
   is the first thing to do if the compute budget binds; it is deliberately not
   done yet, because the dominant term would remain.

2. **Every sample is checked against pandapower, not spot-checked.** That costs a
   full independent solve per sample, which is about half the per-sample time. It
   is not optional: the failure it catches is a solve that converges to a residual
   of 1e-11 on the wrong solution branch, and whether that happens depends on the
   operating point, not only on the topology, so checking one sample per shard
   would not cover the others.

3. **Load is perturbed with a global scale and a per-bus jitter**, not one or the
   other. A pure per-bus jitter averages out across a large grid and barely moves
   the operating point; a pure global scale traces a one-dimensional curve
   through state space. This is the same shape PGLearn and gridfm-datakit use.

4. **Cases whose shipped setpoints do not power-flow are dispatched first.**
   PGLib ships OPF benchmarks, so the base point need not satisfy the power-flow
   equations at all, and 16 of 53 ladder cases do not. ``runopp`` gives a feasible
   dispatch; note that it writes to ``net.res_gen`` and must be copied back into
   ``net.gen`` or the following power flow silently re-solves the original point.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandapower as pp

from grid_ops import reference_pf as _reference_pf
from grid_ops.reference_pf import (
    OracleDisagreement,
    OracleDiverged,
    OracleError,
    ReferenceSolverFailed,
    RootSelectionDisagreement,
    UncorroboratedLabel,
    UnverifiedLabel,
    solve_reference,
)
from grid_ops.testbeds import bus_count, family_of, load_case, size_class_of


# Target mode's default attempt cap, per usable sample asked for (R2-F6).
DEFAULT_ATTEMPTS_PER_TARGET = 50


@dataclass
class SampleStats:
    """What one generation shard produced, and what it cost."""

    case: str
    family: str
    size_class: str
    n_bus: int
    requested: int
    solved: int
    rejected_diverged: int
    rejected_disagreed: int
    seconds: float
    # Split out of rejected_diverged after the 2026-10-06 oracle audit (R1-F4):
    # pandapower itself failing on the operating point is a property of the point,
    # not of our solver. Before the split, all 4 "diverged" case2848rte samples in
    # the 2026-09-13 generation-yield table were this.
    rejected_reference_failed: int = 0
    # A candidate label that failed the residual check. Expected to stay 0; it is
    # counted so that a regression shows up as a number, not as a quiet drop.
    rejected_unverified: int = 0
    # Both solvers agreed, but on a root that is not the one connected to the base
    # operating point (solve_reference's root_check_v0). Counted apart from
    # rejected_disagreed, which stays "our start reached a different root than
    # pandapower". root_check_diverged counts KEPT labels whose base-point start
    # diverged, i.e. labels that passed agreement but got no root-selection check.
    rejected_root_check: int = 0
    # R2-F1: pandapower's root, verified as a root of our equations, that no start
    # of ours not seeded from pandapower reached (solve_reference corroborate=True).
    rejected_uncorroborated: int = 0
    root_check_diverged: int = 0
    root_check: bool = True
    target_usable: int | None = None
    hit_time_limit: bool = False
    # Target mode's attempt cap (R2-F6): set when the shard stopped because it had
    # made max_attempts attempts without reaching target_usable.
    hit_attempt_limit: bool = False
    max_attempts: int | None = None
    # R2-F4: an exception that is not a solver outcome (a bug, a data problem) is
    # counted here, never as rejected_diverged, with its exception class names.
    rejected_other_error: int = 0
    other_error_types: dict[str, int] = field(default_factory=dict)
    # R4-F2: a label was kept, but one of its OTHER starts raised a non-solver error
    # (warm_start_failed / other_error). Counted so a broken start path cannot hide
    # behind the start that succeeded; class names go into other_error_types too.
    kept_with_start_error: int = 0
    # The oracle rule's routes, counted per shard because they are part of the
    # flat-start reliability result rather than housekeeping: a case that needs
    # the reference restart often, or that drops samples, is telling us something
    # about that grid and not about the code.
    route_agreed: int = 0
    route_reference_restart: int = 0
    # Kept labels that implausible_reasons flags. Diagnostic only, never a drop:
    # the check no longer decides labels (R1-F1/F2). The old route_plausible_branch
    # counter is gone because that route no longer emits.
    kept_flagged_implausible: int = 0

    @property
    def yield_rate(self) -> float:
        return self.solved / self.requested if self.requested else 0.0

    @property
    def short_by(self) -> int:
        """Usable samples still owed against the target, if one was set.

        Non-zero means the shard ran out of wall time (``hit_time_limit``) or of
        attempts (``hit_attempt_limit``) before the target, and the case needs
        another task. It is not an error: on a grid near its
        loadability limit most sampled operating points do not solve, and the
        alternative to reporting the shortfall is silently returning a thinner
        dataset for exactly the grids the experiment cares most about.
        """
        return max(0, (self.target_usable or 0) - self.solved)


def apply_opf_dispatch(net) -> None:
    """Copy an OPF result back into the input tables.

    ``runopp`` writes to ``net.res_gen``; a following ``runpp`` reads ``net.gen``.
    Without this the power flow re-solves the ORIGINAL setpoints and fails exactly
    as it did before, which reads as "the dispatch did not help".
    """
    if len(net.gen) and len(getattr(net, "res_gen", [])):
        net.gen["p_mw"] = net.res_gen["p_mw"].values
        if "vm_pu" in net.res_bus.columns:
            net.gen["vm_pu"] = net.res_bus.vm_pu.loc[net.gen.bus.values].values
    if len(net.ext_grid) and "vm_pu" in net.res_bus.columns:
        net.ext_grid["vm_pu"] = net.res_bus.vm_pu.loc[net.ext_grid.bus.values].values
    # Controllable sgens are dispatched by the OPF too: PGLib's converter turns most
    # generators into them (90 of 92 on pglib_opf_case240_pserc). Leaving them at the
    # shipped P/Q made the following power flow re-solve a different point, which is
    # why the 09-13 report saw "OPF converges but the power flow still will not".
    # Found 2026-10-06 by the null-baseline fix (baselines._write_back_dispatch).
    if len(net.sgen) and "controllable" in net.sgen.columns and len(getattr(net, "res_sgen", [])):
        c = net.sgen.controllable.fillna(False).astype(bool).to_numpy()
        net.sgen.loc[c, "p_mw"] = net.res_sgen.p_mw.to_numpy()[c]
        net.sgen.loc[c, "q_mvar"] = net.res_sgen.q_mvar.to_numpy()[c]


def prepare_base_case(name: str):
    """Load a case and put it at an operating point the power flow can solve.

    Returns ``(net, how)`` where ``how`` is ``"as_shipped"``, ``"dispatched"`` or
    ``"slack_reenabled_and_dispatched"``. Raises if none of those works, so a case
    with no usable operating point is excluded loudly rather than silently
    contributing zero samples.
    """
    net = load_case(name)
    try:
        pp.runpp(net, numba=True)
        return net, "as_shipped"
    except Exception:
        pass
    # Two PGLib cases arrive with their only ext_grid out of service, so the
    # network has no voltage reference and both PF and OPF refuse.
    reenabled = False
    if len(net.ext_grid) and not net.ext_grid.in_service.all():
        net.ext_grid.loc[:, "in_service"] = True
        reenabled = True
        try:
            pp.runpp(net, numba=True)
            return net, "slack_reenabled"
        except Exception:
            pass
    pp.runopp(net, numba=True)          # raises if there is no feasible dispatch
    apply_opf_dispatch(net)
    pp.runpp(net, numba=True)           # raises if the dispatch still will not solve
    return net, ("slack_reenabled_and_dispatched" if reenabled else "dispatched")


def perturb_load(net, rng, *, global_lo=0.8, global_hi=1.2, jitter=0.1) -> None:
    """Scale every load by one global factor, then jitter each independently.

    In place, on ``net.load``. Reactive power is jittered separately from active
    so the power factor varies too; scaling both by the same draw would hold it
    fixed and shrink the reachable state space.
    """
    g = rng.uniform(global_lo, global_hi)
    n = len(net.load)
    net.load["p_mw"] = net.load["p_mw"].values * g * rng.uniform(1 - jitter, 1 + jitter, n)
    net.load["q_mvar"] = net.load["q_mvar"].values * g * rng.uniform(1 - jitter, 1 + jitter, n)


def generate_shard(
    case: str,
    n_samples: int | None = None,
    *,
    target_usable: int | None = None,
    max_seconds: float | None = None,
    max_attempts: int | None = None,
    seed: int = 0,
    global_lo: float = 0.8,
    global_hi: float = 1.2,
    jitter: float = 0.1,
    root_check: bool = True,
) -> tuple[list[dict], SampleStats]:
    """Generate one shard of solved operating points for a single topology.

    Two modes. Give ``n_samples`` to make a fixed number of ATTEMPTS, which is what
    a yield measurement wants. Give ``target_usable`` to keep attempting until that
    many samples have been KEPT, which is what dataset generation wants, since
    yield varies about fifteen-fold across the ladder: measured at the
    literature-standard perturbation, 30 of 30 attempts solve on case1354pegase
    against 2 of 15 on case6470rte, because the heavily loaded grids sit near their
    loadability limit. Asking for attempts would hand the thinnest datasets to
    exactly the grids the experiment cares most about.

    ``max_seconds`` bounds the target mode so a shard cannot overrun its
    allocation; a shard that stops early reports ``short_by``, and the case needs
    another task rather than being quietly accepted as complete.

    ``max_attempts`` caps target mode by attempts as well (R2-F6), so a grid whose
    samples never solve stops even with ``max_seconds=None``. ``None`` means
    ``DEFAULT_ATTEMPTS_PER_TARGET`` (50) x ``target_usable``: the lowest yield
    measured on the ladder is 2 of 15 (case6470rte), about 7.5 attempts per kept
    sample, so 50 leaves a wide margin. Stopping on it sets ``hit_attempt_limit``
    and reports ``short_by``. Attempt mode ignores it (``n_samples`` is the cap).

    A sample that diverges or that disagrees with the independent solver is dropped
    and counted, never returned: the yield rate is part of the result. The drop
    classes are disjoint and taken from ``solve_reference``'s exception, which is
    chosen over ALL starts it tried: a disagreement on the first start stays a
    disagreement even if the fallback start then diverges (R1-F4). Only
    :class:`OracleDiverged` counts as diverged (R2-F4); any exception that is not
    one of the oracle's rejection classes, and :class:`OracleError`, is counted in
    ``rejected_other_error`` with its exception class name in
    ``other_error_types``.

    On a topology with out-of-service or islanded buses (the planned N-1/N-2
    shards), ``vm_pu``/``va_deg`` are NaN at those buses: ``solve_reference``
    judges agreement on energized buses only and does not invent a voltage for a
    dead one (R2-F3).

    ``root_check`` (default True, NEW 2026-10-06: before that date every shard ran
    without it) solves the base operating point once through ``solve_reference``
    and passes that verified solution as ``root_check_v0`` for every sample, so a
    label both solvers agree on but which is not the root connected to the base
    point is dropped and counted as ``rejected_root_check``. Measured on
    case2848rte at the default perturbation it removes 4 of 25 (seed 0) and 5 of
    26 (seed 1) labels that the agreement check alone kept; on case1354pegase,
    case6470rte and case9241pegase it removes none. ``root_check=False`` does NOT
    reproduce the pre-root-check behaviour since R2-F1 (commit 664aa35,
    ``solve_reference(corroborate=True)``): with no base-point start, a route-2
    sample (pandapower's root, reached only by the restart from it) has nothing to
    corroborate it, so every such sample is dropped and counted as
    ``rejected_uncorroborated`` (audit R3-F2). The pre-R2-F1 rule is reachable only
    by calling ``solve_reference(corroborate=False)`` directly.
    """
    if (n_samples is None) == (target_usable is None):
        raise ValueError("give exactly one of n_samples (attempts) or target_usable (kept)")
    if target_usable is not None and max_attempts is None:
        max_attempts = DEFAULT_ATTEMPTS_PER_TARGET * max(target_usable, 1)
    if max_attempts is not None and max_attempts < 1:
        raise ValueError(f"max_attempts must be at least 1, got {max_attempts}")
    t0 = time.perf_counter()
    base, _how = prepare_base_case(case)
    # The base point's own verified label, in PYPOWER ordering, which is fixed
    # within a shard because the topology is. solve_reference raises if the base
    # point itself cannot be labelled, which excludes the case loudly.
    # Called through the module, not the name imported above, so a test that
    # replaces the per-sample oracle does not also replace the base-point label.
    v_base = _reference_pf.solve_reference(base).result.v if root_check else None
    p0 = base.load["p_mw"].copy()
    q0 = base.load["q_mvar"].copy()
    rng = np.random.default_rng(seed)

    samples: list[dict] = []
    diverged = disagreed = ref_failed = unverified = flagged = root_rej = root_div = 0
    uncorroborated = 0
    other = 0
    kept_err = 0
    other_types: dict[str, int] = {}
    routes: dict[str, int] = {}
    attempts = 0
    hit_limit = hit_attempt_limit = False
    while True:
        if target_usable is None:
            if attempts >= n_samples:
                break
        else:
            if len(samples) >= target_usable:
                break
            if max_seconds is not None and time.perf_counter() - t0 >= max_seconds:
                hit_limit = True
                break
            if attempts >= max_attempts:
                hit_attempt_limit = True
                break
        attempts += 1
        base.load["p_mw"] = p0
        base.load["q_mvar"] = q0
        perturb_load(base, rng, global_lo=global_lo, global_hi=global_hi, jitter=jitter)
        try:
            sol = solve_reference(base, root_check_v0=v_base)
        except RootSelectionDisagreement:
            root_rej += 1
            continue
        except UncorroboratedLabel:
            uncorroborated += 1
            continue
        except OracleDisagreement:
            disagreed += 1
            continue
        except ReferenceSolverFailed:
            ref_failed += 1
            continue
        except UnverifiedLabel:
            unverified += 1
            continue
        except OracleDiverged:
            diverged += 1
            continue
        except OracleError as exc:
            other += 1
            for name in exc.error_types or [type(exc).__name__]:
                other_types[name] = other_types.get(name, 0) + 1
            continue
        except Exception as exc:
            # Not an oracle outcome at all: count it under its own name, never as
            # a divergence (R2-F4: an injected KeyError used to read as "diverged").
            other += 1
            other_types[type(exc).__name__] = other_types.get(type(exc).__name__, 0) + 1
            continue
        start_errs = [a for a in (sol.attempts or [])
                      if a.get("outcome") in ("other_error", "warm_start_failed")]
        if start_errs:
            kept_err += 1
            for a in start_errs:
                name = a.get("error_type") or a["outcome"]
                other_types[name] = other_types.get(name, 0) + 1
        routes[sol.route] = routes.get(sol.route, 0) + 1
        if sol.implausible:
            flagged += 1
        if sol.root_check == "diverged":
            root_div += 1
        samples.append({
            "p_mw": base.load["p_mw"].values.astype(np.float32).copy(),
            "q_mvar": base.load["q_mvar"].values.astype(np.float32).copy(),
            "vm_pu": sol.result.vm.astype(np.float32),
            "va_deg": sol.result.va_deg.astype(np.float32),
        })
    stats = SampleStats(
        case=case, family=family_of(case), size_class=size_class_of(case),
        n_bus=bus_count(case), requested=attempts, solved=len(samples),
        rejected_diverged=diverged, rejected_disagreed=disagreed,
        seconds=time.perf_counter() - t0,
        rejected_reference_failed=ref_failed, rejected_unverified=unverified,
        rejected_root_check=root_rej, root_check_diverged=root_div, root_check=root_check,
        rejected_uncorroborated=uncorroborated,
        target_usable=target_usable, hit_time_limit=hit_limit,
        hit_attempt_limit=hit_attempt_limit,
        max_attempts=max_attempts if target_usable is not None else None,
        rejected_other_error=other, other_error_types=other_types,
        kept_with_start_error=kept_err,
        route_agreed=routes.get("agreed", 0),
        route_reference_restart=routes.get("agreed_after_reference_restart", 0),
        kept_flagged_implausible=flagged,
    )
    return samples, stats
