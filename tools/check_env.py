#!/usr/bin/env python
"""Compare the running Python environment with the pinned lock (environment-genco.lock.yml).

    python tools/check_env.py            # exit 0 if every pinned pip package and python match, 1 otherwise

Checks: python major.minor, every `name==version` line of the pip section, and the conda-pinned packages
that expose python metadata (numpy, numba, ...). A package missing from the environment is a mismatch.
Set ALLOW_ENV_DRIFT=1 to report mismatches without failing (for a machine that cannot get the exact build).
"""
import os, re, sys
from importlib import metadata
from pathlib import Path

lock = Path(__file__).resolve().parents[1] / "environment-genco.lock.yml"
bad, ok, skipped = [], 0, 0
notes = []
want_py = None
for line in lock.read_text().splitlines():
    m = re.match(r"\s+- ([A-Za-z0-9_.-]+)=([^=\s]+)=\S+$", line)        # conda: name=version=build
    p = re.match(r"\s+- ([A-Za-z0-9_.-]+)==(\S+)$", line)               # pip: name==version
    q = m or p
    if not q:
        continue
    name, ver = q.group(1), q.group(2)
    if name == "python":
        want_py = ver
        continue
    try:
        have = metadata.version(name)
    except metadata.PackageNotFoundError:
        if p:
            bad.append(f"{name}: pinned {ver}, not installed")
        else:
            skipped += 1                       # conda package without python metadata (libs, compilers)
        continue
    if have == ver:
        ok += 1
    elif "+" not in ver and have.split("+", 1)[0] == ver:
        # PEP 440 local label only (e.g. pinned torch-scatter 2.1.2 built from source, installed 2.1.2+pt28cu128 from
        # the PyG wheel on a glibc >= 2.32 host, REPRODUCE.md step 1). A pin that carries a label (torch 2.8.0+cu128)
        # must still match exactly.
        ok += 1
        notes.append(f"{name}: pinned {ver}, installed {have} (local build label only; accepted)")
    else:
        bad.append(f"{name}: pinned {ver}, installed {have}")
if want_py and ".".join(want_py.split(".")[:2]) != f"{sys.version_info.major}.{sys.version_info.minor}":
    bad.append(f"python: pinned {want_py}, running {sys.version.split()[0]}")
print(f"check_env: {ok} pinned packages match, {skipped} conda-only packages not checkable, {len(bad)} mismatch(es)")
for b in bad:
    print("  MISMATCH", b)
for n in notes:
    print("  note", n)
sys.exit(0 if (not bad or os.environ.get("ALLOW_ENV_DRIFT") == "1") else 1)
