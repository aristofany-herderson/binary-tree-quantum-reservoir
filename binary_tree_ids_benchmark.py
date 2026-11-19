"""
Intrusion-detection benchmark on CSE-CIC-IDS2018 windows restricted to
quantum-walk reservoirs on complete binary trees (EDB preprint).

Same pipeline as 12_ids_benchmark.py, with the walk grid fixed to the
`binarytree` family: CTQW only (the DTQW needs a regular graph), depths
2..4 (7, 15 and 31 vertices), the phase, qsample, Hamiltonian,
amplitude, basis and multibase encodings, the Laplacian and
adjacency Hamiltonians and two evolution times. A few classical models
run on the same windows as references (--baselines none drops them).

Copy this file to python/examples/ (next to 12_ids_benchmark.py and
_results.py) and run

    python binary_tree_ids_benchmark.py

With no arguments it reads dataset/balanced.csv (or dataset/data.csv when
the balanced file does not exist), prepares the windows once into a cache,
runs every (model, seed) task in parallel worker processes (one per CPU of
the node or of the SLURM allocation, each with one OpenMP thread), merges
the results and writes CSV, Markdown and LaTeX tables. A rerun skips the
finished tasks, so a job that hits its time limit can simply be resubmitted.

Inside a SLURM job array the tasks are split over the array as well: each
array task runs its share with its own workers, and the last one to finish
writes the tables (or run --merge afterwards).

    python binary_tree_ids_benchmark.py --list          # plan only
    python binary_tree_ids_benchmark.py --csv /scratch/ids/data.csv
    python binary_tree_ids_benchmark.py --merge         # tables only
    python binary_tree_ids_benchmark.py --synthetic --seeds 1 --depths 2 3 --max-train-windows 300

Outputs go to python/results/data/binary_tree_ids_benchmark/ (or --output).
The LaTeX files are also copied to documents/preprint-edb/paper/generated/
when that folder exists.
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
for candidate in (HERE, HERE.parents[1] / "python" / "examples"):
    if (candidate / "_results.py").exists():
        sys.path.insert(0, str(candidate))
        break

from _results import DEFAULT_CSV, data_dir  # noqa: E402

STEM = Path(__file__).stem
REPO = DEFAULT_CSV.parents[1]
PAPER_GENERATED = REPO / "documents" / "preprint-edb" / "paper" / "generated"
BALANCED_CSV = DEFAULT_CSV.with_name("balanced.csv")

DEPTHS = (2, 3, 4)
ENCODINGS = ("phase", "qsample", "hamiltonian", "amplitude", "basis", "multibase")
HAMILTONIANS = ("laplacian", "adjacency")
TIMES = (1.5, 3.0)
BASELINES = ("ESN-8", "ESN-100", "ESN-500", "RFF-256", "PCA-linear", "XGBoost")
ENCODING_PT = {
    "phase": "fase",
    "qsample": "qsample",
    "hamiltonian": "hamiltoniana",
    "amplitude": "amplitude",
    "basis": "base",
    "multibase": "multibase",
}


def env_int(name, default):
    value = os.environ.get(name, "")
    return int(value) if value.strip().isdigit() else default


def available_cpus():
    slurm = env_int("SLURM_CPUS_PER_TASK", 0)
    if slurm:
        return slurm
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    default_csv = BALANCED_CSV if BALANCED_CSV.exists() else DEFAULT_CSV
    p.add_argument("--csv", default=str(default_csv))
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--depths", nargs="+", type=int, default=list(DEPTHS))
    p.add_argument("--encodings", nargs="+", default=list(ENCODINGS))
    p.add_argument("--hamiltonians", nargs="+", default=list(HAMILTONIANS))
    p.add_argument("--times", nargs="+", type=float, default=list(TIMES))
    p.add_argument("--baselines", nargs="+", default=list(BASELINES),
                   help="reference models or groups of the registry; 'none' drops them")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--window", type=int, default=20)
    p.add_argument("--stride", type=int, default=20)
    p.add_argument("--split", default="episode_chronological")
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--max-train-windows", type=int, default=None)
    p.add_argument("--readout", default="ridge", choices=["ridge", "logistic"])
    p.add_argument("--pooling", default="last_mean")
    p.add_argument("--jobs", type=int, default=available_cpus(),
                   help="worker processes in this run (default: CPUs available)")
    p.add_argument("--threads", type=int, default=1,
                   help="OpenMP/BLAS threads per worker")
    p.add_argument("--load-workers", type=int, default=-1,
                   help="processes used to read the CSV (-1: all)")
    p.add_argument("--output", default=None, help="root of results/ (default python/results)")
    p.add_argument("--dataset-cache", default=None)
    p.add_argument("--array-index", type=int, default=env_int("SLURM_ARRAY_TASK_ID", 0))
    p.add_argument("--array-count", type=int, default=env_int("SLURM_ARRAY_TASK_COUNT", 1))
    p.add_argument("--list", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--merge", action="store_true")
    p.add_argument("--retry-errors", action="store_true")
    args = p.parse_args(argv)
    args.jobs = max(1, args.jobs)
    if args.baselines == ["none"]:
        args.baselines = []
    return args


def set_threads(threads):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(threads)


# --------------------------------------------------------------------- plan
def build_plan(args):
    from qrckit.benchmarking import GRID_GROUP, WalkGrid
    from qrckit.benchmarking.grid import register_walk_grid
    from qrckit.ids.benchmark import REGISTRY, IDSBenchmarkConfig

    grid = WalkGrid(
        dynamics=("ctqw",),
        graphs=("binarytree",),
        vertices=tuple(2 ** (d + 1) - 1 for d in args.depths),
        encodings=tuple(args.encodings),
        hamiltonians=tuple(args.hamiltonians),
        evolution_times=tuple(args.times),
    )
    config = IDSBenchmarkConfig(
        readout=args.readout,
        pooling=args.pooling,
        xgb_n_jobs=args.threads,
    )
    points, skipped = grid.split()
    points, too_small = fits_tree(points, config)
    skipped = list(skipped) + too_small
    registry = REGISTRY.copy()
    register_walk_grid(registry, points)
    models = registry.expand(list(args.baselines) + [GRID_GROUP])
    config = replace(config, models=tuple(models), seeds=tuple(range(args.seeds)))
    registry.validate(config.models)
    tree_models = set(registry.groups()[GRID_GROUP])
    return registry, config, skipped, tree_models


def fits_tree(points, config):
    """Drop basis and multibase points whose register does not fit the tree.

    A complete binary tree has 2**(d+1) - 1 vertices, one short of a power
    of two, so a register of 2**k states needs the next depth up.
    """
    import math

    need = {
        "basis": 2 ** (config.basis_bits * config.basis_pca),
        "multibase": 2 ** math.ceil(config.multibase_pca / config.multibase_features_per_qubit),
    }
    keep, dropped = [], []
    for point in points:
        states = need.get(point.encoding)
        if states is not None and states > point.vertices:
            dropped.append((point, f"{point.encoding} needs {states} vertices"))
        else:
            keep.append(point)
    return keep, dropped


def all_tasks(config):
    from qrckit.benchmarking import benchmark_tasks

    tasks = benchmark_tasks(config.models, config.seeds)
    # Largest trees first, so the slow tasks do not all start at the end.
    return sorted(tasks, key=lambda t: -_vertices(t[0]))


def _vertices(name):
    for token in name.split():
        if token.startswith("binarytree-"):
            return int(token.split("-")[1])
    return 0


def print_plan(config, skipped, tree_models, tasks, args):
    total = args.array_count * args.jobs
    print(f"tree models: {len(tree_models)}, baselines: {len(config.models) - len(tree_models)}")
    print(f"tasks: {len(tasks)} ({len(config.seeds)} seeds per model)")
    print(f"array tasks: {args.array_count}, workers per array task: {args.jobs}, "
          f"threads per worker: {args.threads} -> {total} shards, "
          f"at most {-(-len(tasks) // total)} tasks each")
    print(f"tree grid points skipped: {len(skipped)}")
    for reason, count in Counter(reason for _, reason in skipped).most_common():
        print(f"{count:>6}  {reason}")


# --------------------------------------------------------------------- data
def paths_for(args):
    out = data_dir(__file__, args.output)
    out.mkdir(parents=True, exist_ok=True)
    source = "synthetic" if args.synthetic else Path(args.csv).stem
    cache = Path(args.dataset_cache) if args.dataset_cache else (
        out / f"ids_{source}_w{args.window}s{args.stride}.pkl.gz"
    )
    return out, out / STEM, cache


def data_config(args):
    from qrckit.ids.data import IDSConfig

    return IDSConfig(
        window=args.window,
        stride=args.stride,
        split=args.split,
        max_rows=args.max_rows,
        max_train_windows=args.max_train_windows,
        workers=args.load_workers,
    )


def load_dataset(args, cache, prepare=True):
    from qrckit.benchmarking import WindowDataset
    from qrckit.ids.data import prepare_dataset
    from qrckit.ids.synthetic import make_synthetic_flows

    config = data_config(args)
    source = "synthetic" if args.synthetic else str(Path(args.csv).resolve())
    if cache.exists():
        dataset = WindowDataset.load(str(cache))
        same = (
            dataset.report.get("source") == source
            and _comparable(dataset.config) == _comparable(config)
        )
        if not same:
            raise SystemExit(
                f"{cache} was prepared from another source or IDSConfig; "
                "delete it or pass another --dataset-cache"
            )
        return dataset
    if not prepare:
        raise SystemExit(f"dataset cache {cache} is missing")
    flows = make_synthetic_flows() if args.synthetic else args.csv
    dataset = prepare_dataset(flows, config)
    dataset.report["source"] = source
    dataset.save(str(cache))
    return dataset


def _comparable(config):
    # The number of CSV reader processes does not change the windows.
    from dataclasses import asdict

    values = asdict(config)
    values.pop("workers", None)
    return values


# --------------------------------------------------------------------- run
def shard_file(prefix, index, total):
    return f"{prefix}_shard{index:04d}-of-{total:04d}.jsonl"


def run_shard(argv, index, total):
    """Worker process: one shard of the task list, with its own checkpoint."""
    args = parse_args(argv)
    set_threads(args.threads)
    from qrckit.benchmarking import shard_tasks
    from qrckit.ids.benchmark import run_ids_benchmark

    _, prefix, cache = paths_for(args)
    registry, config, _, _ = build_plan(args)
    # Skip what any earlier run finished, whatever its number of shards.
    finished, _ = merge(prefix, all_tasks(config), keep_errors=not args.retry_errors)
    finished = {(r.config["model"], r.seed) for r in finished}
    tasks = [t for t in shard_tasks(all_tasks(config), (index, total)) if t not in finished]
    if not tasks:
        return index, 0
    dataset = load_dataset(args, cache, prepare=False)
    start = time.time()

    def progress(name, seed):
        print(f"[shard {index}/{total}] {name} seed={seed} "
              f"({time.time() - start:.0f} s)", flush=True)

    # One call per model, with only the seeds of this shard; the runner does
    # not store models or seeds in the task configuration, so the shared
    # checkpoint stays consistent.
    done = 0
    by_model = defaultdict(list)
    for name, seed in tasks:
        by_model[name].append(seed)
    for name, seeds in by_model.items():
        single = replace(config, models=(name,), seeds=tuple(seeds))
        results = run_ids_benchmark(
            dataset,
            single,
            progress=progress,
            registry=registry,
            checkpoint=shard_file(prefix, index, total),
            on_error="record",
            retry_errors=args.retry_errors,
        )
        done += len(results)
    return index, done


def run_local(argv, args, prefix):
    total = args.array_count * args.jobs
    shards = [args.array_index * args.jobs + j for j in range(args.jobs)]
    if args.jobs == 1:
        run_shard(argv, shards[0], total)
        return
    set_threads(args.threads)
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.jobs, maxtasksperchild=1) as pool:
        pending = [pool.apply_async(run_shard, (argv, s, total)) for s in shards]
        for job in pending:
            index, count = job.get()
            print(f"shard {index} of {total} finished: {count} tasks", flush=True)


# --------------------------------------------------------------------- report
def merge(prefix, tasks, keep_errors=True):
    from qrckit.benchmarking import load_window_results

    paths = sorted(str(p) for p in prefix.parent.glob(f"{prefix.name}_shard*-of-*.jsonl"))
    wanted = set(tasks)
    results = [r for r in load_window_results(paths)
               if (r.config["model"], r.seed) in wanted
               and (keep_errors or "error" not in r.metrics)]
    found = {(r.config["model"], r.seed) for r in results}
    missing = [t for t in tasks if t not in found]
    return results, missing


def report(prefix, out, results, tasks, tree_models, missing, copy=True):
    from qrckit.ids.benchmark import (
        format_ids_table,
        save_ids_benchmark,
        summarize_ids_benchmark,
    )

    failed = sum("error" in r.metrics for r in results)
    paths = save_ids_benchmark(str(prefix), results)
    binary = summarize_ids_benchmark(results, "binary")
    binary.sort(key=lambda r: -(r["roc_auc"] if r["roc_auc"] == r["roc_auc"] else -1))
    print(format_ids_table(binary[:25], "Binary view (attack vs benign), top 25"))
    print(f"\ntasks: {len(results)} done, {failed} failed, {len(missing)} missing")
    latex = write_latex(out / "latex", binary, tree_models, len(results), failed,
                        len(missing))
    if copy and PAPER_GENERATED.exists():
        for f in latex:
            (PAPER_GENERATED / f.name).write_text(f.read_text(encoding="utf-8"),
                                                  encoding="utf-8")
        print(f"LaTeX tables copied to {PAPER_GENERATED}")
    print(paths)


def dec(x, digits=3):
    return f"{x:.{digits}f}".replace(".", "{,}")


def tex_name(row):
    enc = ENCODING_PT.get(row.get("encoding"), row.get("encoding"))
    h = "L" if row.get("hamiltonian") == "laplacian" else "-A"
    return (rf"árvore $n={row['num_vertices']}$, {enc}, $H={h}$, "
            rf"$\tau={dec(row['evolution_time'], 1)}$")


def write_latex(folder, binary, tree_models, n_done, n_failed, n_missing):
    folder.mkdir(parents=True, exist_ok=True)
    ok = [r for r in binary if not r.get("error") and r["roc_auc"] == r["roc_auc"]]
    trees = [r for r in ok if r["model"] in tree_models]
    refs = [r for r in ok if r["model"] not in tree_models]
    files = []

    # Top tree points and every reference, binary view.
    lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"Modelo & F1 & ROC-AUC $\pm$ DP & AP & FAR \\", r"\midrule"]
    for r in trees[:3]:
        lines.append(rf"\rowcolor{{qwrow}}{tex_name(r)} & {dec(r['f1'])} & "
                     rf"${dec(r['roc_auc'])} \pm {dec(r['roc_auc_std'])}$ & "
                     rf"{dec(r['ap'])} & {dec(r['far'])} \\")
    lines.append(r"\midrule")
    for r in refs:
        lines.append(rf"{r['model']} & {dec(r['f1'])} & "
                     rf"${dec(r['roc_auc'])} \pm {dec(r['roc_auc_std'])}$ & "
                     rf"{dec(r['ap'])} & {dec(r['far'])} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    files.append(folder / "tree_ids_top.tex")
    files[-1].write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Best ROC-AUC per encoding and size (max over Hamiltonian and time).
    sizes = sorted({r["num_vertices"] for r in trees})
    encodings = [e for e in ENCODINGS if any(r["encoding"] == e for r in trees)]
    lines = [r"\begin{tabular}{@{}l" + "r" * len(sizes) + "@{}}", r"\toprule",
             "Codificação & " + " & ".join(f"$n={n}$" for n in sizes) + r" \\",
             r"\midrule"]
    for e in encodings:
        cells = []
        for n in sizes:
            vals = [r["roc_auc"] for r in trees if r["encoding"] == e and r["num_vertices"] == n]
            cells.append(dec(max(vals)) if vals else "--")
        lines.append(f"{ENCODING_PT[e]} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    for h, label in (("laplacian", r"média, $H=L$"), ("adjacency", r"média, $H=-A$")):
        cells = []
        for n in sizes:
            vals = [r["roc_auc"] for r in trees if r["hamiltonian"] == h and r["num_vertices"] == n]
            cells.append(dec(sum(vals) / len(vals)) if vals else "--")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    files.append(folder / "tree_ids_axes.tex")
    files[-1].write_text("\n".join(lines) + "\n", encoding="utf-8")

    macros = {
        "Points": str(len(tree_models)),
        "Done": str(n_done),
        "Failed": str(n_failed),
        "Missing": str(n_missing),
    }
    if trees:
        macros.update(BestTree=tex_name(trees[0]), BestTreeAUC=dec(trees[0]["roc_auc"]),
                      BestTreeFAR=dec(trees[0]["far"]))
    if refs:
        macros.update(BestRef=refs[0]["model"], BestRefAUC=dec(refs[0]["roc_auc"]))
    esn = [r for r in refs if r["model"] == "ESN-100"]
    if esn:
        macros["ESNAUC"] = dec(esn[0]["roc_auc"])
    for h in ("laplacian", "adjacency"):
        vals = [r["roc_auc"] for r in trees if r["hamiltonian"] == h]
        if vals:
            key = "MeanL" if h == "laplacian" else "MeanA"
            macros[key] = dec(sum(vals) / len(vals))
    if trees:
        best_by_n = []
        for n in sizes:
            vals = [r["roc_auc"] for r in trees if r["num_vertices"] == n]
            best_by_n.append(rf"{dec(max(vals))} ($n={n}$)")
        macros["BestByN"] = "; ".join(best_by_n)
    lines = ["% Generated by binary_tree_ids_benchmark.py; do not edit."]
    lines += [rf"\newcommand{{\TreeIDS{k}}}{{{v}}}" for k, v in macros.items()]
    files.append(folder / "tree_ids_macros.tex")
    files[-1].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return files


# --------------------------------------------------------------------- main
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    set_threads(args.threads)
    registry, config, skipped, tree_models = build_plan(args)
    tasks = all_tasks(config)
    if args.list:
        print_plan(config, skipped, tree_models, tasks, args)
        return
    out, prefix, cache = paths_for(args)
    if not args.merge:
        print_plan(config, skipped, tree_models, tasks, args)
        if not cache.exists():
            print(f"preparing windows into {cache}", flush=True)
        from qrckit.ids.data import summarize_dataset

        dataset = load_dataset(args, cache)
        print(summarize_dataset(dataset), flush=True)
        del dataset
        if args.prepare_only:
            return
        run_local(argv, args, prefix)
    results, missing = merge(prefix, tasks)
    if missing and not args.merge:
        print(f"{len(missing)} tasks still missing (other array tasks may be "
              "running); run --merge once they finish")
        return
    report(prefix, out, results, tasks, tree_models, missing, copy=not args.synthetic)


if __name__ == "__main__":
    main()
