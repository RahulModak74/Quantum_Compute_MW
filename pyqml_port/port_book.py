#!/usr/bin/env python3
"""
Port + validate Dr. Frank Zickert's "Hands-On Quantum Machine Learning with Python, Vol. 1"
(PyQML) notebooks onto mwqc.

The book's repository has no license file, so its notebooks are NOT redistributed here.
This script works on YOUR local clone:

    git clone https://github.com/quantum-machine-learning/Hands-On-Quantum-Machine-Learning-With-Python-Vol-1 book_vol1
    python3 pyqml_port/port_book.py --book book_vol1 --out pyqml_port/ported      # write mwqc notebooks
    python3 pyqml_port/port_book.py --book book_vol1 --validate                   # run both, compare

Porting = rewriting the import lines only:
    from qiskit import QuantumCircuit, Aer, execute      -> from mwqc.qiskit_compat import ...
    from qiskit.visualization import plot_histogram      -> from mwqc.qiskit_compat import ...
    from qiskit.circuit.library import ZGate             -> from mwqc.qiskit_compat import ...

Validation runs every chapter twice from the same source, once on IBM's stack
(Qiskit 2.x + qiskit-aer, through pyqml_port/legacy_qiskit.py because the book uses the
pre-1.0 `execute`/`Aer` API) and once on mwqc. Every execute() result is recorded
and compared call-for-call:
    statevectors       -> fidelity |<a|b>|^2
    unitaries          -> max |Ua - Ub|
    exact probabilities (statevector backend get_counts) -> max |pa - pb|
    sampled counts (qasm) -> total-variation distance (expected ~0.01-0.05 from shot noise alone)
"""
import argparse
import contextlib
import io
import json
import os
import random
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)


def rewrite(src: str, target: str) -> str:
    mod = "mwqc.qiskit_compat" if target == "mwqc" else "legacy_qiskit"
    src = re.sub(r"^(\s*)from qiskit(\.visualization|\.circuit\.library)? import", rf"\1from {mod} import",
                 src, flags=re.M)
    src = re.sub(r"^(\s*)import qiskit\s*$", rf"\1import {mod} as qiskit", src, flags=re.M)
    src = re.sub(r"^(\s*)from qutip import.*$", r"\1pass  # qutip Bloch rendering not ported", src, flags=re.M)
    return src


def code_cells(nb_path):
    nb = json.load(open(nb_path))
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            src = "".join(c["source"])
            src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith(("%", "!")))
            yield src


def port(book, out):
    os.makedirs(out, exist_ok=True)
    for f in sorted(os.listdir(book)):
        if not f.endswith(".ipynb"):
            continue
        nb = json.load(open(os.path.join(book, f)))
        for c in nb["cells"]:
            if c["cell_type"] == "code":
                c["source"] = rewrite("".join(c["source"]), "mwqc").splitlines(keepends=True)
                c["outputs"], c["execution_count"] = [], None
        nb["cells"].insert(0, {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None,
                               "source": [f"import sys; sys.path.insert(0, {ROOT!r})  # mwqc\n"]})
        json.dump(nb, open(os.path.join(out, f), "w"), indent=1)
        print(f"ported {f}")


def run_chapter(nb_path, target, book_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.show = lambda *a, **k: None
    if target == "mwqc":
        import mwqc.qiskit_compat as M
    else:
        import legacy_qiskit as M
    M.RECORD = []
    random.seed(0)
    np.random.seed(0)
    ns = {"__name__": "__main__", "display": lambda *a, **k: None}
    buf = io.StringIO()
    errors = []
    cwd = os.getcwd()
    os.chdir(book_dir)
    t = time.time()
    try:
        for i, src in enumerate(code_cells(nb_path)):
            try:
                with contextlib.redirect_stdout(buf):
                    exec(compile(rewrite(src, target), f"{os.path.basename(nb_path)}[{i}]", "exec"), ns)
            except Exception as e:      # keep going; report
                errors.append(f"cell {i}: {type(e).__name__}: {str(e)[:160]}")
            plt.close("all")
    finally:
        os.chdir(cwd)
    return M.RECORD, errors, time.time() - t, buf.getvalue()


def compare(ra, rb):
    """ra: qiskit records, rb: mwqc records."""
    st = {"statevector_infidelity_max": 0.0, "prob_absdiff_max": 0.0, "unitary_absdiff_max": 0.0,
          "counts_tvd": [], "n_compared": 0}
    for a, b in zip(ra, rb):
        st["n_compared"] += 1
        if "statevector" in a and "statevector" in b:
            f = abs(np.vdot(a["statevector"], b["statevector"])) ** 2
            st["statevector_infidelity_max"] = max(st["statevector_infidelity_max"], 1 - f)
            pa, pb = np.abs(a["statevector"]) ** 2, np.abs(b["statevector"]) ** 2
            st["prob_absdiff_max"] = max(st["prob_absdiff_max"], float(np.max(np.abs(pa - pb))))
        elif "unitary" in a and "unitary" in b:
            Ua, Ub = np.asarray(a["unitary"]), np.asarray(b["unitary"])
            k = np.unravel_index(np.argmax(np.abs(Ua)), Ua.shape)
            ph = Ub[k] / Ua[k] if abs(Ua[k]) > 0 else 1
            st["unitary_absdiff_max"] = max(st["unitary_absdiff_max"], float(np.max(np.abs(Ua * ph - Ub))))
        elif "counts" in a and "counts" in b:
            ta, tb = sum(a["counts"].values()), sum(b["counts"].values())
            keys = set(a["counts"]) | set(b["counts"])
            st["counts_tvd"].append(0.5 * sum(abs(a["counts"].get(k, 0) / ta - b["counts"].get(k, 0) / tb)
                                              for k in keys))
    tv = st.pop("counts_tvd")
    st["counts_n"] = len(tv)
    st["counts_tvd_median"] = float(np.median(tv)) if tv else None
    st["counts_tvd_max"] = float(np.max(tv)) if tv else None
    return st


def validate(book, chapters):
    rows = []
    for f in sorted(os.listdir(book)):
        if not f.endswith(".ipynb"):
            continue
        ch = int(re.findall(r"\d+", f)[0])
        if chapters and ch not in chapters:
            continue
        nb = os.path.join(book, f)
        ra, ea, ta, oa = run_chapter(nb, "qiskit", book)
        rb, eb, tb, ob = run_chapter(nb, "mwqc", book)
        c = compare(ra, rb)
        row = {"chapter": ch, "executes_qiskit": len(ra), "executes_mwqc": len(rb),
               "errors_qiskit": ea, "errors_mwqc": eb, "seconds_qiskit": ta, "seconds_mwqc": tb, **c}
        rows.append(row)
        tvd = "-" if c["counts_tvd_median"] is None else f"{c['counts_tvd_median']:.3f}/{c['counts_tvd_max']:.3f}"
        print(f"ch{ch:02d}  executes q/m {len(ra):>4}/{len(rb):<4}  sv-infid {c['statevector_infidelity_max']:.1e}"
              f"  |dp| {c['prob_absdiff_max']:.1e}  |dU| {c['unitary_absdiff_max']:.1e}  counts TVD med/max {tvd}"
              f"  errors q/m {len(ea)}/{len(eb)}  time q/m {ta:.0f}s/{tb:.0f}s")
        for e in eb:
            if e not in ea:
                print(f"      mwqc-only error: {e}")
        for e in ea:
            if e not in eb:
                print(f"      qiskit-only error: {e}")
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    p = os.path.join(ROOT, "results", "pyqml_validation.json")
    json.dump(rows, open(p, "w"), indent=2, default=str)
    print(f"saved {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", required=True, help="local clone of the Vol-1 repository")
    ap.add_argument("--out", default=None, help="write mwqc-ported notebooks here")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--chapters", default="", help="e.g. 2,5,13")
    a = ap.parse_args()
    book = os.path.abspath(a.book)
    if a.out:
        port(book, a.out)
    if a.validate:
        validate(book, [int(x) for x in a.chapters.split(",") if x])


if __name__ == "__main__":
    main()
