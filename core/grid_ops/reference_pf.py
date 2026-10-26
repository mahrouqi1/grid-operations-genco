"""Transparent reference AC power flow - polar Newton-Raphson.

A readable, instrumentable Newton-Raphson solver we control end to end.
Three jobs:

1. **Teaching** - the algorithm is laid out step by step (mismatch →
   Jacobian → Newton step → repeat), so it can drive a primer.
2. **Validation**: it reproduces pandapower's solution to ~1e-11 pu in voltage
   magnitude and ~1e-8 degrees in angle (its own converged power mismatch is
   ~1e-15 to 1e-12; see ``tests/test_reference_pf.py``, which asserts vm < 1e-8
   and angle < 1e-5 deg), so it
   is a trustworthy oracle to check derived quantities against - WHEN it
   converges. Flat-start Newton-Raphson (a *literal* ``v0``: |V|=1,
   angle=0 everywhere except the fixed slack reference) DIVERGES on the
   whole RTE family with a low minimum voltage - case6470rte,
   case6495rte (6 slack buses; verified the multi-REF handling is fine
   structurally, the divergence is unrelated), and case6515rte, all
   confirmed (mismatch grows geometrically past 1e13 by ``max_it=25``) - where pandapower's own solver converges fine on the identical
   Ybus/Sbus. See ``test_flat_start_diverges_on_*`` in
   ``tests/test_reference_pf.py``. The fix is :func:`dc_warm_start_va` +
   ``from_pandapower(..., warm_start="dc")``: a DC power-flow solve gets
   the angle STRUCTURE right (still assumes flat |V|=1, but that part was
   never the problem) and recovers Newton-Raphson convergence on all
   three cases in under 10 iterations, to the same precision as above, with
   zero change to the default (``warm_start="flat"``) path any existing
   caller/test uses. For bulk label generation on cases where flat start
   is known to diverge, prefer ``warm_start="dc"`` over calling pandapower
   directly - it keeps you on this solver's own instrumented Jacobian and
   mismatch trace.
3. **Substrate** - it records the per-iteration mismatch and exposes the
   Jacobian, which later phases need for warm-start / unrolled-solver
   experiments (things pandapower's compiled solver hides).

Everything runs in PYPOWER "internal" bus ordering so a pandapower net can
be ingested with no pandapower<->ppc index juggling (:func:`from_pandapower`).

The code assembles the Jacobian with the compact complex-derivative
formulation (PYPOWER's ``dSbus_dV``); it is algebraically identical to the
textbook scalar partials ∂P/∂θ, ∂P/∂|V|, ∂Q/∂θ, ∂Q/∂|V| spelled out in the
power-flow primer, just vectorized.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg  # noqa: F401 -- attaches sp.linalg.spsolve


@dataclass
class PFResult:
    """Outcome of a Newton-Raphson solve."""

    v: np.ndarray              # complex bus voltages (per unit)
    vm: np.ndarray             # |V|  (per unit)
    va_deg: np.ndarray         # voltage angle (degrees)
    converged: bool
    iterations: int
    mismatch_history: list[float]  # max |power mismatch| after each iteration (incl. start)


def injected_power(ybus: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Complex power injected at every bus: S = V ⊙ conj(Ybus V).

    This is the heart of the model - power is *quadratic* in voltage
    (a V times a current that is itself Ybus·V), which is exactly why the
    power-flow equations are nonlinear.
    """
    return v * np.conj(ybus @ v)


def _sparse_jacobian(ybus: sp.spmatrix, v: np.ndarray, pvpq: np.ndarray, pq: np.ndarray) -> sp.spmatrix:
    """Sparse Jacobian assembly - same block structure as the dense path in
    :func:`newton_raphson_pf`, built on PYPOWER's ``dSbus_dV_sparse`` (the
    same complex-derivative primitive, just reused rather than re-derived:
    a correctness-critical piece is safer sourced from tested code than
    hand-duplicated, per the "read the producer's own artifact" convention - the Newton loop, mismatch, and history stay ours).
    """
    from pandapower.pypower.dSbus_dV import dSbus_dV_sparse

    ds_dvm, ds_dva = dSbus_dV_sparse(ybus, v)
    ds_dva = ds_dva.tocsr()
    ds_dvm = ds_dvm.tocsr()
    return sp.bmat([
        [ds_dva[pvpq, :][:, pvpq].real, ds_dvm[pvpq, :][:, pq].real],
        [ds_dva[pq, :][:, pvpq].imag,   ds_dvm[pq, :][:, pq].imag],
    ], format="csc")


def newton_raphson_pf(
    ybus: np.ndarray | sp.spmatrix,
    sbus: np.ndarray,
    v0: np.ndarray,
    pv: np.ndarray,
    pq: np.ndarray,
    *,
    tol: float = 1e-10,
    max_it: int = 25,
) -> PFResult:
    """Solve the AC power-flow equations for the bus voltages.

    Parameters
    ----------
    ybus : (n, n) complex, dense ndarray OR scipy sparse matrix
        Bus admittance matrix (network topology + line/transformer physics).
        A sparse ``ybus`` (e.g. ``from_pandapower(net, sparse=True)``'s
        output) dispatches to a sparse Jacobian assembly + ``spsolve`` - the dense path below is UNCHANGED either way, so every case that
        validated to ~1e-11 against pandapower keeps doing so exactly as
        before. Sparse exists because the dense path is O(n^2) memory /
        O(n^3) per Newton step: ~49 s at 2869 bus, a projected ~9.6 GB /
        several minutes at 9241 bus (measured on this repo's cases) - the
        wall a real admittance matrix (a few nonzeros per row, however
        large n gets) never needs to hit.
    sbus : (n,) complex
        Specified net injection P+jQ at each bus (generation − load), per unit.
    v0 : (n,) complex
        Initial voltage guess (a "flat start" of 1∠0 with generator setpoints
        applied is the usual choice).
    pv, pq : int arrays
        Indices of PV (voltage-controlled) and PQ (load) buses. The remaining
        bus is the slack/reference. Unknowns solved for: angle at every
        non-slack bus, magnitude at every PQ bus.

    Returns
    -------
    PFResult with the converged voltages and the convergence trace.
    """
    sparse = sp.issparse(ybus)
    v = v0.astype(complex).copy()
    va = np.angle(v)
    vm = np.abs(v)
    pvpq = np.r_[pv, pq]

    def mismatch(v: np.ndarray) -> np.ndarray:
        # F = S_calculated − S_specified, keeping only the rows we solve:
        #   real power at every non-slack bus, reactive power at every PQ bus.
        ds = injected_power(ybus, v) - sbus
        return np.r_[ds[pvpq].real, ds[pq].imag]

    f = mismatch(v)
    history = [float(np.max(np.abs(f))) if f.size else 0.0]
    it = 0
    while history[-1] > tol and it < max_it:
        it += 1
        # --- Jacobian J = dF/d[angles, magnitudes] (complex-derivative form) ---
        if sparse:
            j = _sparse_jacobian(ybus, v, pvpq, pq)
            dx = sp.linalg.spsolve(j, -f)
        else:
            ibus = ybus @ v
            diag_v = np.diag(v)
            diag_i = np.diag(ibus)
            diag_vnorm = np.diag(v / np.abs(v))
            ds_dvm = diag_v @ np.conj(ybus @ diag_vnorm) + np.conj(diag_i) @ diag_vnorm
            ds_dva = 1j * diag_v @ np.conj(diag_i - ybus @ diag_v)
            j = np.block([
                [ds_dva[np.ix_(pvpq, pvpq)].real, ds_dvm[np.ix_(pvpq, pq)].real],
                [ds_dva[np.ix_(pq, pvpq)].imag,   ds_dvm[np.ix_(pq, pq)].imag],
            ])
            # --- Newton step: solve J · dx = −F, then update the unknowns ---
            dx = np.linalg.solve(j, -f)
        k = len(pvpq)
        va[pvpq] += dx[:k]
        vm[pq] += dx[k:]
        v = vm * np.exp(1j * va)
        f = mismatch(v)
        history.append(float(np.max(np.abs(f))) if f.size else 0.0)

    return PFResult(
        v=v, vm=np.abs(v), va_deg=np.degrees(np.angle(v)),
        converged=history[-1] <= tol, iterations=it, mismatch_history=history,
    )


# --- adapters -------------------------------------------------------------
def dc_warm_start_va(net) -> np.ndarray:
    """Bus voltage ANGLES (radians, PYPOWER-internal ordering) from
    pandapower's own DC power flow - a warm start for
    :func:`newton_raphson_pf` on cases where the flat start (angle=0
    everywhere) diverges.

    DC power flow still assumes flat |V|=1 pu and linearizes away reactive
    power entirely, but it DOES solve the real angle structure from the
    network's B-matrix and P injections - exactly the part a flat start
    guesses at (zero) and gets badly wrong on a case with a wide true
    angle span. It is not a full AC solution, just a much better guess for
    Newton-Raphson to iterate from.

    Runs on a private ``deepcopy`` of ``net`` so it never perturbs the
    caller's ``net`` / ``net.res_*`` / ``net._ppc`` (same "solve on a
    private copy" discipline as ``baselines.py``'s AC-vs-DC comparison - ``pp.rundcpp``/``pp.runpp`` both write results into ``net`` in place).

    Verified empirically (not assumed): the internal bus numbering
    ``net._pd2ppc_lookups["bus"]`` a DC solve produces is IDENTICAL to an
    AC solve's, on every case checked including case6495rte's 6 reference
    buses - pandapower's internal renumbering is a deterministic function
    of network topology / bus type, not of which solver ran, so this
    array lines up with :func:`from_pandapower`'s ``ybus``/``sbus``/``pv``/
    ``pq``/``ref`` with no re-indexing needed. Also verified: DC power flow
    honors a non-zero declared slack reference angle exactly (e.g.
    case118's +30°), matching what the flat-start path already sets at
    ``ref`` - so overwriting the WHOLE ``va`` array with this function's
    output (including at ``ref``) is safe.
    """
    import copy

    import pandapower as pp
    from pandapower.pypower.idx_bus import VA

    net_dc = copy.deepcopy(net)
    pp.rundcpp(net_dc, numba=False)
    return np.radians(net_dc._ppc["bus"][:, VA])


def from_pandapower(net, *, sparse: bool = False, warm_start: str = "flat",
                    pp_init: str = "auto") -> dict:
    """Extract (Ybus, Sbus, start V, bus-type indices, reference solution)
    from a pandapower net, all in PYPOWER-internal ordering.

    Runs pandapower's own power flow first so the ppc is built and its
    converged ``Vsol`` is available to validate our solver against.

    ``sparse=False`` (default) densifies Ybus exactly as before - every
    existing caller (tests, the primer's ``imshow`` heatmap) is unaffected.
    ``sparse=True`` keeps ``makeYbus``'s own sparse matrix, for
    :func:`newton_raphson_pf`'s sparse dispatch at scales where the dense
    path's O(n^2) memory / O(n^3)-per-iteration cost becomes the wall
    (measured: infeasible well before 9241 bus on this repo's cases).

    ``warm_start`` picks the ANGLE half of the returned ``v0``:
    ``"flat"`` (default, UNCHANGED - every non-reference angle starts at
    0, byte-identical to every existing caller/test) or ``"dc"`` (angles
    from :func:`dc_warm_start_va`, which recovers Newton-Raphson
    convergence on cases where flat-start diverges - case6470rte and its
    RTE-family siblings case6495rte/case6515rte, all verified). The
    MAGNITUDE half (flat 1.0 pu + generator setpoints) is identical either
    way - DC power flow doesn't model |V|, so there's nothing case-specific
    to gain by branching it.

    ``pp_init`` is passed to ``pp.runpp(init=...)`` and picks where PANDAPOWER's
    own Newton-Raphson starts. ``"auto"`` (default, UNCHANGED, the value every
    existing result was produced with) resolves to DC angles in pandapower 3.4 (audit 2026-10-06, R1-F5:
    ``init_va='dc'`` on every case checked), with load-bus |V| at the average
    generator setpoint (up to ~0.05 pu from our 1.0) and a 10-iteration cap
    against our 25 (audit R4-F1). That is CLOSE TO, not identical with, our
    ``warm_start="dc"``: both start from the same DC angles, so "our DC start
    agrees with pandapower" mostly checks the implementation, not the choice
    of root. ``"flat"`` makes pandapower start
    from 1.0 pu / zero angle instead, so an agreement with our DC start is then
    two genuinely different starts reaching the same root.

    The returned ``live`` mask (audit 2026-10-06, R2-F3) is True at every bus
    pandapower energizes and False at a bus that is out of service or islanded
    (``BUS_TYPE == NONE``; pandapower leaves those in the ppc with NaN voltage).
    A dead bus is in none of ``ref``/``pv``/``pq`` and has no admittance to any
    live bus, so it never enters our equations; ``v0`` carries a finite 1.0 pu
    placeholder there (the DC angle is NaN at a dead bus) so a dense ``Ybus @ v``
    cannot spread NaN into the live rows. On a net with every bus live, ``v0`` is
    unchanged. A failure inside the DC warm start raises :class:`WarmStartFailed`
    (R2-F4), so a caller can tell it from pandapower's own AC solve failing.
    """
    import pandapower as pp
    from pandapower.pypower.makeYbus import makeYbus
    from pandapower.pypower.makeSbus import makeSbus
    from pandapower.pypower.idx_bus import BUS_TYPE, VM, VA, REF, NONE, PV as PV_, PQ as PQ_
    from pandapower.pypower.idx_gen import VG, GEN_BUS, GEN_STATUS

    if warm_start not in ("flat", "dc"):
        raise ValueError(f"warm_start must be 'flat' or 'dc', got {warm_start!r}")
    if pp_init not in ("auto", "flat", "dc"):
        raise ValueError(f"pp_init must be 'auto', 'flat' or 'dc', got {pp_init!r}")

    pp.runpp(net, numba=False, init=pp_init)
    ppc = net._ppc
    base, bus, branch, gen = ppc["baseMVA"], ppc["bus"], ppc["branch"], ppc["gen"]

    ybus_sp = makeYbus(base, bus, branch)[0]
    ybus = ybus_sp if sparse else ybus_sp.toarray()
    sbus = makeSbus(base, bus, gen)
    n = bus.shape[0]
    ref = np.where(bus[:, BUS_TYPE] == REF)[0]
    pv = np.where(bus[:, BUS_TYPE] == PV_)[0]
    pq = np.where(bus[:, BUS_TYPE] == PQ_)[0]

    vm = np.ones(n)
    va = np.zeros(n)
    # Honor the case's slack reference angle (often 0, but e.g. case118 fixes
    # it at +30°). The slack angle is held fixed through the solve, so it sets
    # the global reference every other angle is measured against - get it wrong
    # and the whole solution is a rigid angle-offset away from pandapower's.
    va[ref] = np.radians(bus[ref, VA])
    on = gen[:, GEN_STATUS] > 0
    gbus = gen[on, GEN_BUS].astype(int)
    vm[gbus] = gen[on, VG]
    if warm_start == "dc":
        try:
            va = dc_warm_start_va(net)
        except Exception as exc:
            raise WarmStartFailed(
                f"DC warm start failed: {type(exc).__name__}: {exc}"[:300]) from exc
    v0 = vm * np.exp(1j * va)
    live = bus[:, BUS_TYPE] != NONE
    if not live.all():
        v0[~live] = 1.0 + 0.0j

    vsol = bus[:, VM] * np.exp(1j * np.radians(bus[:, VA]))
    return {"ybus": ybus, "sbus": sbus, "v0": v0, "pv": pv, "pq": pq,
            "ref": ref, "vsol": vsol, "base_mva": base, "live": live}


def two_bus_example() -> dict:
    """The textbook hand-solvable system used in the power-flow primer.

    Bus 0: slack, held at 1.0∠0°.
    Bus 1: PQ load drawing 0.5 + j0.2 per unit.
    One line, impedance z = 0.02 + j0.06 pu (no line charging), so the
    2×2 admittance matrix is exact and easy to write by hand:
        y = 1/z = 5 − j15
        Ybus = [[ y, −y], [−y, y]]
    Two unknowns (θ₁, |V₁|), two equations (ΔP₁, ΔQ₁).
    """
    z = 0.02 + 0.06j
    y = 1.0 / z
    ybus = np.array([[y, -y], [-y, y]], dtype=complex)
    sbus = np.array([0.0 + 0.0j, -0.5 - 0.2j], dtype=complex)  # bus 1 consumes
    v0 = np.array([1.0 + 0j, 1.0 + 0j], dtype=complex)         # flat start
    pv = np.array([], dtype=int)
    pq = np.array([1], dtype=int)
    return {"ybus": ybus, "sbus": sbus, "v0": v0, "pv": pv, "pq": pq,
            "ref": np.array([0])}




# ---------------------------------------------------------------------------
# The label-generation entry point
# ---------------------------------------------------------------------------

def power_mismatch(ybus, sbus: np.ndarray, v: np.ndarray, pv: np.ndarray,
                   pq: np.ndarray) -> float:
    """Max |power mismatch| of ``v`` under ``ybus``/``sbus``, per unit.

    Same rows as :func:`newton_raphson_pf` solves (and so the same masking): real
    power at every PV and PQ bus, reactive power at every PQ bus. Slack rows and
    PV reactive rows are free variables of the power-flow problem, not equations,
    so they are excluded. For a vector returned by ``newton_raphson_pf`` this is
    exactly its last ``mismatch_history`` entry; the point of having it separately
    is to check a vector that did NOT come out of our solver, before it is emitted.
    """
    pvpq = np.r_[pv, pq]
    ds = injected_power(ybus, v) - sbus
    f = np.r_[ds[pvpq].real, ds[pq].imag]
    return float(np.max(np.abs(f))) if f.size else 0.0


class WarmStartFailed(RuntimeError):
    """Building a start vector failed (today: pandapower's DC power flow inside
    :func:`dc_warm_start_va`). Not a solver outcome and not pandapower's AC solve
    failing, so :func:`solve_reference` records it as that start's outcome
    (``"warm_start_failed"``) and still tries the fallback start (R2-F4).
    """


class OracleRejection(RuntimeError):
    """``solve_reference`` refused to return a label. Base class of the reasons.

    ``attempts`` is the per-start record: one dict per start tried, each with
    ``start`` and ``outcome`` (``"diverged"``, ``"disagreed"``,
    ``"reference_failed"``, ``"warm_start_failed"`` or ``"other_error"``) plus the
    numbers behind it. The subclass raised is
    chosen from ALL starts, not the last one (audit 2026-10-06, R1-F4): a
    disagreement on any start outranks a later divergence, so a sample that reached
    a different root is never counted as one that merely failed to converge.
    """

    def __init__(self, message: str, attempts: list[dict] | None = None):
        super().__init__(message)
        self.attempts = list(attempts or [])


class OracleDisagreement(OracleRejection):
    """Newton-Raphson converged, but not to pandapower's solution.

    This is the dangerous failure, not divergence. The power-flow equations
    admit multiple solutions, so a solve can reach a power mismatch of 1e-11
    and still be sitting on a different branch than the reference solver. A
    label generator that gates on ``converged`` and the mismatch alone will emit
    that silently. Measured on this repo's own ladder (2026-09-13): from a flat
    start, ``case2848rte`` converges to a mismatch of 1.4e-11 with a maximum
    voltage-magnitude error of 1.01 pu and 100 degrees of angle error against
    pandapower. From a DC warm start the same case agrees to 1.8e-13 pu.
    """


class RootSelectionDisagreement(OracleDisagreement):
    """Both solvers agree on a root, but it is not the root connected to the base
    operating point (the ``root_check_v0`` start of :func:`solve_reference`).

    Found by the 2026-10-06 oracle rerun, beyond the audit's findings: on
    case2848rte at the generation perturbation, pandapower (default init) and our
    DC start agree on a low-voltage root (min |V| 0.51 to 0.66 pu, 116 to 155
    degrees across a plain line) on 4 of 25 (seed 0) and 5 of 26 (seed 1)
    solvable samples, while a load continuation from the base point reaches a
    different root up to 0.40 pu away. Agreement could not see this because the
    two solvers share a start (R1-F5). A subclass of OracleDisagreement so any
    caller that drops disagreements also drops these.
    """


class UncorroboratedLabel(OracleDisagreement):
    """Pandapower's root verifies as a root of our equations (route 2), but no start
    of ours that was not seeded from pandapower's solution reached it: not the
    other of dc/flat, and not the base-point start ``root_check_v0`` (absent, or it
    diverged). Dropped under ``corroborate=True`` (audit 2026-10-06, R2-F1): the
    restart from pandapower's vector converges in 0 or 1 iterations, so on its own
    it is one solver's root, verified, not two solvers' agreement. A subclass of
    OracleDisagreement because our own start did reach a different root.
    """


class OracleDiverged(OracleRejection):
    """Every start our solver could run failed to converge, and none disagreed.

    A singular Jacobian (``numpy.linalg.LinAlgError``) counts as a failure to
    converge: Newton-Raphson cannot take a step. A start whose warm-start vector
    could not be built is not a divergence; it is recorded as
    ``"warm_start_failed"`` and does not by itself make this class (R2-F4).
    """


class OracleError(OracleRejection):
    """An exception that is not a solver outcome stopped the label (R2-F4).

    Raised when any start hit an unexpected exception (``"other_error"``), or when
    no start could be run at all because every warm start failed. Each such
    attempt records ``error_type`` (the exception class name) and ``detail``;
    ``error_types`` lists them. It outranks every other class: an exception that
    may be a bug makes the sample's classification untrustworthy, so it is
    reported as an error and never folded into "diverged" or "disagreed".
    """

    @property
    def error_types(self) -> list[str]:
        return [a["error_type"] for a in self.attempts if "error_type" in a]


class ReferenceSolverFailed(OracleRejection):
    """pandapower itself did not converge, so there is no reference to agree with.

    Counted apart from :class:`OracleDiverged` (R1-F4): on ``case2848rte`` all four
    "diverged" samples in the 2026-09-13 generation-yield table were this, a
    property of the operating point (pandapower's solver gave up), not of ours.
    """


class UnverifiedLabel(OracleRejection):
    """A candidate label failed the residual check under our own Ybus/Sbus.

    Should be unreachable through routes 1 and 2 (both emit a vector our solver
    converged on). It exists so that any future route, or a regression, that tries
    to emit a vector which is not a root fails loudly instead of writing it.
    """


@dataclass
class ReferenceSolution:
    """A solved operating point, with the evidence that it is the right one."""

    result: "PFResult"
    vm_err: float          # max |vm - pandapower vm|, per unit
    va_err_deg: float      # max |angle - pandapower angle|, degrees, absolute (both use the case's own reference angle)
    warm_start: str        # which start produced it
    seconds: float
    route: str = "agreed"  # how the label was accepted; see solve_reference
    first_disagreement_vm: float | None = None   # vm_err at the first disagreement, if any
    first_disagreement_va_deg: float | None = None  # va_err at that disagreement (R2-F5)
    first_disagreement_start: str | None = None     # the start that disagreed (R2-F5)
    residual: float = float("nan")   # max |power mismatch| of the emitted vector, our Ybus/Sbus
    implausible: list[str] | None = None  # implausible_reasons() of the emitted label; diagnostic only
    attempts: list[dict] | None = None    # per-start record of the starts tried before this label
    root_check: str | None = None  # None (not run), "agreed" or "diverged"; see solve_reference
    # Buses pandapower leaves de-energized (out of service or islanded), PYPOWER
    # ordering. The label is NaN there; agreement is judged on the rest (R2-F3).
    dead_buses: np.ndarray | None = None


def implausible_reasons(vm, va_deg, ybus, *, vm_band=(0.5, 1.5),
                        max_angle_diff_deg: float = 90.0) -> list[str]:
    """Why a solved state does not look like an operating point, or [] if it does.

    **Diagnostic only.** Since the 2026-10-06 audit (R1-F1, F2) this no longer
    decides which label is emitted: :func:`solve_reference` records it on every
    label and on every dropped disagreement, and never uses it to choose between
    two solvers. It used to break ties, and calibrated at (0.7, 1.3) pu and 60
    degrees it flagged PANDAPOWER'S OWN agreed solution on 5 of the 37 solvable
    ladder cases (case145: 77.8 degrees across a branch; case1888rte: 1.301 pu;
    case6470rte / case6495rte / case6515rte: minimum 0.557 / 0.560 / 0.559 pu),
    which is every large RTE case, the family where disagreements happen.

    The defaults are now set from physics plus that measurement rather than from
    operating practice:

    * ``max_angle_diff_deg = 90``: for a mostly inductive branch the transferable
      power ``|V1||V2| sin(d) / X`` peaks at 90 degrees; beyond it the state is on
      the unstable side of the power-angle curve. The ladder maximum on an agreed
      solution is 77.8 degrees (case145).
    * ``vm_band = (0.5, 1.5)``: the spurious roots seen so far collapse to about
      0.02 pu (case2848rte, flat start), while agreed solutions reach 0.557 to
      1.301 pu. The band separates collapse from a heavily loaded but real state;
      it is not an operating-limit check (0.9 to 1.1 pu is the planning range).

    The rerun in ``results/2026-10-06-oracle-rerun/`` applies this to all 37
    agreed ladder solutions (0 flagged) and to the case2848rte spurious roots (all
    flagged). A label being flagged here is a reason to look, not a reason to drop.

    Connectivity comes from the admittance matrix's own sparsity pattern, so no
    extra topology argument is needed and the check cannot disagree with the
    matrix actually being solved. Angle differences are measured the short way
    round. Deliberately NOT checked: reversed flow on radial feeders, which has no
    well-defined expectation on a meshed transmission network.
    """
    why: list[str] = []
    lo, hi = vm_band
    if float(np.min(vm)) < lo:
        why.append(f"min |V| = {float(np.min(vm)):.3f} pu, below {lo}")
    if float(np.max(vm)) > hi:
        why.append(f"max |V| = {float(np.max(vm)):.3f} pu, above {hi}")
    yb = ybus.tocoo() if sp.issparse(ybus) else sp.coo_matrix(ybus)
    off = yb.row != yb.col
    rows, cols = yb.row[off], yb.col[off]
    if rows.size:
        d = np.abs(va_deg[rows] - va_deg[cols]) % 360.0
        d = np.minimum(d, 360.0 - d)          # angles are circular
        if float(d.max()) > max_angle_diff_deg:
            why.append(f"max angle across a branch = {float(d.max()):.1f} deg, "
                       f"above {max_angle_diff_deg}")
    return why


def _wrap_deg(d: np.ndarray) -> np.ndarray:
    """An angle difference in degrees, wrapped into [-180, 180)."""
    return (np.asarray(d) + 180.0) % 360.0 - 180.0


def solve_reference(
    net,
    *,
    warm_start: str = "dc",
    fallback: bool = True,
    vm_tol: float = 1e-6,
    va_tol_deg: float = 1e-3,
    sparse: bool = True,
    residual_tol: float = 1e-9,
    pp_init: str = "auto",
    root_check_v0: np.ndarray | None = None,
    corroborate: bool = True,
) -> ReferenceSolution:
    """Solve one operating point to oracle quality, or refuse to return a label.

    THE ORACLE RULE (2026-09-13; step 3 corrected after the 2026-10-06
    audit, R1-F1). A label is not "whatever converged"; it is a solution both
    solvers stand behind. Where they disagree, **pandapower is the reference**, and
    a label is emitted only if our own solver verifies pandapower's root. For each
    start in turn (``warm_start``, then the other one if ``fallback``):

    1. Solve from the start and accept only if it **agrees** with pandapower inside
       ``vm_tol`` and ``va_tol_deg`` (absolute angles, not re-referenced: both
       solvers pin every REF bus to the case's own reference angle; the angle
       difference is taken modulo 360 degrees, so a bus at the +/-180 degree
       branch cut is not a disagreement, audit R3-F5).
       Route ``"agreed"``.
    2. On disagreement, retry **from pandapower's own solution**. If our solver
       converges there and agrees, keep it. Route
       ``"agreed_after_reference_restart"``. Note what this route is: the ROOT was
       selected by pandapower alone (our starts did not reach it); our solver
       verifies that it is a root of our Ybus/Sbus to ``tol``. It is a pandapower
       label, independently verified as a solution, not independently found.
       With ``corroborate=True`` (default since R2-F1) that is not enough to emit:
       the candidate is held while the remaining start runs (if it agrees with
       pandapower outright, that is a genuine two-start agreement and the label
       goes out as route ``"agreed"``), and after the last start it is emitted
       only if the base-point start ``root_check_v0`` converges to the same root.
       Otherwise :class:`UncorroboratedLabel`. ``corroborate=False`` restores the
       pre-R2-F1 behaviour (pandapower selects, our solver only verifies).
       The disagreement that triggered it stays on the label (R2-F5): an
       ``attempts`` record with outcome ``"disagreed_then_restarted"`` (the start,
       its vm and va error, its min |V| and plausibility), and
       ``first_disagreement_vm`` / ``_va_deg`` / ``_start``.
    3. If they still disagree, **record and drop**. ``implausible_reasons`` of both
       candidates is written into the attempt record as a diagnostic, and the start
       counts as a disagreement. No label is emitted.

       Step 3 used to emit whichever candidate alone was "plausible". That emitted
       labels only one solver reached, and its pandapower-side branch returned
       ``vsol`` unverified: reaching step 3 means our Newton-Raphson, started ON
       ``vsol``, did not converge back to it, which for a well-conditioned
       Newton-Raphson means ``vsol`` is not a root of our equations (the audit's
       2-bus mock emitted a label with 0.54 pu of mismatch). The plausibility
       classifier was also miscalibrated (R1-F2). It could be made sound only by
       re-solving ``vsol`` under our equations, which is exactly step 2, so step 3
       can only ever add unverified or single-solver labels. Dropping it is the
       only version consistent with "never emit a label only one solver reached".
       No recorded result had taken this route (audit: all 60 sweep labels
       ``accepted_agreed``).
    4. If no start yields a label, raise. The exception is chosen from ALL starts
       (R1-F4) and from the exception TYPE, never from "something raised" (R2-F4):
       :class:`OracleError` if any start hit an unexpected exception
       (``"other_error"``, class name recorded) or no start could be run at all,
       else :class:`OracleDisagreement` if any start disagreed, else
       :class:`ReferenceSolverFailed` if pandapower's AC solve raised
       ``LoadflowNotConverged`` (or returned a non-finite voltage at an energized
       bus), else :class:`OracleDiverged` (a singular Jacobian counts as
       divergence). A failed DC warm start is that start's outcome
       (``"warm_start_failed"``) and the fallback start still runs. Each exception
       carries ``attempts``.

    **De-energized buses (R2-F3).** A bus pandapower leaves out of service or
    islanded (``from_pandapower``'s ``live`` mask) is NaN in pandapower's solution
    and is in none of our equations. Agreement, the root check and the emitted
    label's vm/va errors are judged on energized buses only; the label is NaN at
    the dead buses, which are listed in ``dead_buses``.

    **Root-selection check (optional, ``root_check_v0``).** Given a voltage vector
    in PYPOWER-internal ordering, normally the verified solution of the same grid
    at its BASE operating point, our solver is also run from it. If it converges
    to a different root than the candidate label, the sample is dropped with
    :class:`RootSelectionDisagreement`; if it agrees, ``root_check="agreed"``; if
    it diverges, the label is kept with ``root_check="diverged"`` (no evidence of
    another root, counted by the caller). This is the one start that does not
    share pandapower's initialization, and it is what catches a root both solvers
    agree on but which is not the operating point (see the exception's docstring
    for the measurement). One Newton solve from the base point gave the same
    verdict as a 20-step load continuation on every sample tested (rerun
    ``report.md``). ``generate_shard`` passes it by default; ``None`` here keeps
    the old behaviour for direct callers.

    **Every emitted label passes a residual check** (:func:`power_mismatch` under
    our Ybus/Sbus, same masking as ``newton_raphson_pf``) at ``residual_tol``;
    failure raises :class:`UnverifiedLabel`. The residual and
    ``implausible_reasons`` of the label are stored on the result.

    The caller is expected to count route 2 and every rejection class per family
    and size class: they are part of the flat-start reliability result, not
    housekeeping.

    Why any of this is needed: the power-flow equations have several roots, and a
    solve can reach a power mismatch of 1e-11 on the wrong one. On this repo's
    ladder at the shipped operating points, the flat start agrees with pandapower
    on 31 of 37 solvable cases and degrades with size (100 percent below 1,000
    buses, 80 percent at 1,000 to 5,000, 57 percent above 5,000); the DC warm start
    agrees on 37 of 37. During generation (0.8 to 1.2 global load scale, 10 percent
    jitter) on case2848rte, the flat start converges to the WRONG root on most
    samples pandapower can solve, and never to the right one: 18 of 26 at seed 1
    and 20 of 25 at seed 0, the rest diverging (``results/2026-10-06-oracle-rerun/``;
    the audit text said 21 of 25 for seed 0, its own output lists 20). The "1 in 30" quoted before that rerun
    was a different quantity: samples on which the DC start diverged and then the
    flat fallback went wrong, under the pre-rule code.

    **What agreement does and does not show (R1-F5).** With ``pp_init="auto"``
    (default, unchanged, what every existing result used) pandapower starts from
    DC angles with flat |V| and generator setpoints, the same point as our DC
    start, runs the same undamped Newton-Raphson on the same Ybus/Sbus. Agreement
    therefore verifies the IMPLEMENTATION (two codes, one answer), not the choice of
    root: it cannot detect a wrong root that pandapower itself reaches from that
    start. ``pp_init="flat"`` makes pandapower start from a different point, so
    agreement with our DC start is then two starts reaching one root; it is
    measured on the ladder in the rerun directory, and is not the default because
    existing results depend on the default. Neither is proof of physicality.

    The module default for ``from_pandapower`` stays ``"flat"`` deliberately: the
    teaching path and the primer depend on it being byte-identical. The safety
    belongs here, at the point where labels are produced.
    """
    from pandapower.auxiliary import LoadflowNotConverged

    def _err(r, d):
        # Compared over energized buses only (R2-F3): pandapower reports NaN at an
        # out-of-service or islanded bus, which is a data condition, not a
        # disagreement between solvers.
        # Angles are compared modulo 360 degrees (audit R3-F5): a bus at the +/-180
        # degree branch cut can come back as +180 from one solver and -180 from the
        # other. Still absolute angles, not re-referenced (R1-F7).
        m = d["live"]
        return (float(np.max(np.abs(r.vm[m] - np.abs(d["vsol"][m])))),
                float(np.max(np.abs(_wrap_deg(
                    r.va_deg[m] - np.degrees(np.angle(d["vsol"][m])))))))

    def _nr(d, v_start):
        """Our Newton-Raphson from ``v_start``, or None on a singular Jacobian
        (Newton cannot take a step, so it is a failure to converge; R2-F4)."""
        try:
            return newton_raphson_pf(d["ybus"], d["sbus"], v_start, d["pv"], d["pq"])
        except np.linalg.LinAlgError:
            return None

    def _emit(r, d, *, vm_err, va_err, start, seconds, route, require_root_check=False):
        live = d["live"]
        res = power_mismatch(d["ybus"], d["sbus"], r.v, d["pv"], d["pq"])
        if not np.isfinite(res) or res > residual_tol:
            attempts.append({"start": start, "outcome": "unverified", "route": route,
                             "residual": res})
            raise UnverifiedLabel(
                f"route {route} from {start} produced a vector with power mismatch "
                f"{res:.3e} pu under our Ybus/Sbus (tolerance {residual_tol:.0e})",
                attempts)
        implausible = implausible_reasons(r.vm, r.va_deg, d["ybus"])
        rc = None
        if root_check_v0 is not None:
            v_chk = np.asarray(root_check_v0, dtype=complex)
            if v_chk.shape != r.v.shape:
                raise ValueError(f"root_check_v0 has shape {v_chk.shape}, the grid has "
                                 f"{r.v.shape}: it must be in the same PYPOWER ordering")
            v_chk = np.where(live, v_chk, d["v0"])       # a NaN dead bus in a base label
            rr = _nr(d, v_chk)
            if rr is None or not rr.converged:
                rc = "diverged"
            else:
                dvm = float(np.max(np.abs(rr.vm[live] - r.vm[live])))
                dva = float(np.max(np.abs(_wrap_deg(rr.va_deg[live] - r.va_deg[live]))))
                if dvm < vm_tol and dva < va_tol_deg:
                    rc = "agreed"
                else:
                    attempts.append({
                        "start": start, "outcome": "root_check_disagreed", "route": route,
                        "vm_err": dvm, "va_err_deg": dva,
                        "label_min_vm": float(r.vm[live].min()),
                        "check_min_vm": float(rr.vm[live].min()),
                        "label_implausible": implausible,
                        "check_implausible": implausible_reasons(rr.vm, rr.va_deg, d["ybus"]),
                    })
                    raise RootSelectionDisagreement(
                        f"both solvers agree (route {route}) but the start from the base "
                        f"operating point reaches a different root: max |dvm| = {dvm:.3e} pu, "
                        f"max |dangle| = {dva:.3e} deg; label min |V| {r.vm[live].min():.3f}, "
                        f"base-connected root min |V| {rr.vm[live].min():.3f}", attempts)
        if require_root_check and rc != "agreed":
            attempts.append({"start": start, "outcome": "uncorroborated", "route": route,
                             "detail": f"base-point start {rc or 'not run'}"})
            raise UncorroboratedLabel(
                f"route {route}: pandapower's root is a root of our equations, but the "
                f"base-point start {rc or 'was not given'} and no other start of ours "
                f"reached it", attempts)
        dead = np.flatnonzero(~live)
        if dead.size:
            # The solver carried a placeholder there; the label must not.
            r.v[dead] = np.nan
            r.vm[dead] = np.nan
            r.va_deg[dead] = np.nan
        return ReferenceSolution(
            result=r, vm_err=vm_err, va_err_deg=va_err, warm_start=start,
            seconds=seconds, route=route,
            first_disagreement_vm=first.get("vm_err"),
            first_disagreement_va_deg=first.get("va_err_deg"),
            first_disagreement_start=first.get("start"),
            residual=res, implausible=implausible, attempts=list(attempts), root_check=rc,
            dead_buses=dead if dead.size else None)

    def _detail(exc):
        return f"{type(exc).__name__}: {exc}"[:300]

    order = [warm_start] + ([s for s in ("dc", "flat") if s != warm_start] if fallback else [])
    attempts: list[dict] = []
    first: dict = {}                 # the first disagreement: start, vm_err, va_err_deg
    ref_exc: Exception | None = None
    other_exc: Exception | None = None
    pending: dict | None = None      # a route-2 candidate awaiting corroboration (R2-F1)
    for start in order:
        try:
            d = from_pandapower(net, sparse=sparse, warm_start=start, pp_init=pp_init)
        except WarmStartFailed as exc:               # this start's vector, not pandapower's AC solve
            attempts.append({"start": start, "outcome": "warm_start_failed",
                             "error_type": type(exc.__cause__ or exc).__name__,
                             "detail": _detail(exc)})
            continue
        except LoadflowNotConverged as exc:          # pandapower itself could not solve it
            ref_exc = exc
            attempts.append({"start": start, "outcome": "reference_failed",
                             "detail": _detail(exc)})
            break
        except Exception as exc:                     # anything else: never a solver outcome
            other_exc = exc
            attempts.append({"start": start, "outcome": "other_error",
                             "error_type": type(exc).__name__, "detail": _detail(exc)})
            continue
        d.setdefault("live", np.ones(len(d["vsol"]), dtype=bool))
        if not np.isfinite(d["vsol"][d["live"]]).all():
            attempts.append({"start": start, "outcome": "reference_failed",
                             "detail": "pandapower returned a non-finite voltage at an "
                                       "energized bus"})
            break
        t0 = time.perf_counter()
        r = _nr(d, d["v0"])
        dt = time.perf_counter() - t0
        if r is None or not r.converged:
            attempts.append({"start": start, "outcome": "diverged",
                             "mismatch": float("nan") if r is None else r.mismatch_history[-1],
                             **({"detail": "singular Jacobian"} if r is None else {})})
            continue

        # Step 1: agree outright.
        vm_err, va_err = _err(r, d)
        if vm_err < vm_tol and va_err < va_tol_deg:
            return _emit(r, d, vm_err=vm_err, va_err=va_err, start=start, seconds=dt,
                         route="agreed")
        live = d["live"]
        ours_implausible = implausible_reasons(r.vm, r.va_deg, d["ybus"])
        if not first:
            first = {"start": start, "vm_err": vm_err, "va_err_deg": va_err}

        # Step 2: pandapower is the reference, so restart from its solution (the
        # placeholder, not NaN, at a dead bus).
        v_ref = np.where(live, d["vsol"], d["v0"])
        t1 = time.perf_counter()
        r2 = _nr(d, v_ref)
        dt2 = time.perf_counter() - t1
        vm2 = va2 = float("nan")
        if r2 is not None and r2.converged:
            vm2, va2 = _err(r2, d)
            if vm2 < vm_tol and va2 < va_tol_deg:
                # Keep the disagreement that triggered this route on the label (R2-F5).
                attempts.append({
                    "start": start, "outcome": "disagreed_then_restarted",
                    "vm_err": vm_err, "va_err_deg": va_err,
                    "ours_min_vm": float(r.vm[live].min()),
                    "ours_implausible": ours_implausible,
                    "restart_iterations": r2.iterations,
                })
                if not corroborate:
                    return _emit(r2, d, vm_err=vm2, va_err=va2, start="reference",
                                 seconds=dt + dt2, route="agreed_after_reference_restart")
                if pending is None:
                    pending = {"r": r2, "d": d, "vm": vm2, "va": va2, "seconds": dt + dt2}
                continue                 # let the remaining start try to reach it

        # Step 3: record the disagreement, with plausibility as a diagnostic. Drop.
        attempts.append({
            "start": start, "outcome": "disagreed", "vm_err": vm_err, "va_err_deg": va_err,
            "restart_converged": bool(r2 is not None and r2.converged), "restart_vm_err": vm2,
            "ours_implausible": ours_implausible,
            "pandapower_implausible": implausible_reasons(
                np.abs(v_ref), np.degrees(np.angle(v_ref)), d["ybus"]),
            "pandapower_residual": power_mismatch(d["ybus"], d["sbus"], v_ref,
                                                  d["pv"], d["pq"]),
        })

    # Step 2, corroboration (R2-F1): no start of ours reached pandapower's root, so
    # the base-point start, which is not seeded from pandapower, must.
    if pending is not None:
        if root_check_v0 is not None:
            return _emit(pending["r"], pending["d"], vm_err=pending["vm"],
                         va_err=pending["va"], start="reference",
                         seconds=pending["seconds"], route="agreed_after_reference_restart",
                         require_root_check=True)
        attempts.append({"start": "reference", "outcome": "uncorroborated",
                         "detail": "no root_check_v0 given and no start of ours reached "
                                   "pandapower's root"})

    # Step 4: no label. Classify from every start, not the last one.
    summary = "; ".join(
        f"{a['start']}: {a['outcome']}"
        + (f" (max |dvm| = {a['vm_err']:.3e} pu, max |dangle| = {a['va_err_deg']:.3e} deg)"
           if a["outcome"] == "disagreed" else "")
        + (f" (mismatch {a['mismatch']:.3e})" if a["outcome"] == "diverged" else "")
        + (f" ({a['detail']})" if a["outcome"] in ("reference_failed", "warm_start_failed",
                                                   "other_error", "uncorroborated") else "")
        for a in attempts) or "no start attempted"
    outcomes = {a["outcome"] for a in attempts}
    if "other_error" in outcomes:
        raise OracleError(f"no label: {summary}", attempts) from other_exc
    if "uncorroborated" in outcomes:
        raise UncorroboratedLabel(f"no label: {summary}", attempts)
    if "disagreed" in outcomes:
        raise OracleDisagreement(f"no label: {summary}", attempts)
    if "reference_failed" in outcomes:
        raise ReferenceSolverFailed(f"no label: {summary}", attempts) from ref_exc
    if "diverged" in outcomes:
        raise OracleDiverged(f"no label: {summary}", attempts)
    raise OracleError(f"no label: {summary}", attempts)
