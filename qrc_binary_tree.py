"""
Quantum walks and QRC on a binary tree (EDB, binary tree application).

Part 1: continuous-time quantum walk (CTQW) on qk.BinaryTree, starting at
        the root, compared with the classical random walk on the same tree.
        Vertices use heap numbering: root 0, children of i are 2i+1, 2i+2.
Part 2: the same tree as a quantum reservoir (QRC) solving XOR, a task
        that no linear readout solves on the raw inputs.

Plots are saved next to this file, in plots/.

Usage (from the repository root, with qrckit built):
    PYTHONPATH=python python documents/preprint-edb/qrc_binary_tree.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import qrckit as qk

DEPTH = 4
PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def save(fig, name):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def tree_layout(depth):
    """(x, y) of each heap-numbered vertex, root on top."""
    xs, ys = [], []
    for v in range(2 ** (depth + 1) - 1):
        d = level_of(v)
        k = v - (2**d - 1)
        xs.append((k + 0.5) / 2**d)
        ys.append(-d)
    return np.array(xs), np.array(ys)


def level_of(v):
    """Tree level (distance from the root) of heap-numbered vertex v."""
    return int(np.floor(np.log2(v + 1)))


def level_mass(prob, depth):
    """Total probability on each level 0..depth."""
    levels = np.array([level_of(v) for v in range(len(prob))])
    return np.array([prob[levels == d].sum() for d in range(depth + 1)])


def classical_walk(depth, time):
    """Continuous-time classical walk exp(-L t) e_root on the same tree."""
    n = 2 ** (depth + 1) - 1
    A = np.zeros((n, n))
    for v in range(1, n):
        p = (v - 1) // 2
        A[v, p] = A[p, v] = 1.0
    L = np.diag(A.sum(axis=1)) - A
    w, U = np.linalg.eigh(L)
    e0 = np.zeros(n)
    e0[0] = 1.0
    return U @ (np.exp(-w * time) * (U.T @ e0))


def walks():
    graph = qk.BinaryTree(depth=DEPTH)
    print(f"Part 1: CTQW on {graph.label} ({graph.num_vertices} vertices), root 0")
    print("Probability per level (rows: time; columns: level 0..%d)\n" % DEPTH)
    header = "".join(f"{'L' + str(d):>8}" for d in range(DEPTH + 1))
    times = (0.5, 2.0, 5.0)
    fig, axes = plt.subplots(1, len(times), figsize=(12, 3.6), sharey=True)
    fig_tree, tree_axes = plt.subplots(1, len(times), figsize=(12, 3.6))
    xs, ys = tree_layout(DEPTH)
    edges = [
        ((xs[(v - 1) // 2], xs[v]), (ys[(v - 1) // 2], ys[v]))
        for v in range(1, len(xs))
    ]
    for ax, tax, t in zip(axes, tree_axes, times):
        q = qk.CTQW(graph, hamiltonian="laplacian", initial_position=0)
        q_prob = q.simulate(t).result.probabilities
        c_prob = classical_walk(DEPTH, t)
        print(f"t = {t}      {header}")
        for name, prob in (("quantum", q_prob), ("classical", c_prob)):
            row = "".join(f"{m:>8.3f}" for m in level_mass(prob, DEPTH))
            print(f"  {name:<10}{row}   (sum={prob.sum():.3f})")
        print()

        levels = np.arange(DEPTH + 1)
        ax.bar(levels - 0.2, level_mass(q_prob, DEPTH), 0.4, label="quantum")
        ax.bar(levels + 0.2, level_mass(c_prob, DEPTH), 0.4, label="classical")
        ax.set_title(f"t = {t}")
        ax.set_xlabel("tree level")
        for ex, ey in edges:
            tax.plot(ex, ey, color="0.8", zorder=1)
        sc = tax.scatter(xs, ys, c=q_prob, s=120, cmap="viridis", zorder=2)
        tax.set_title(f"quantum, t = {t}")
        tax.axis("off")
        fig_tree.colorbar(sc, ax=tax, fraction=0.046)
    axes[0].set_ylabel("probability on level")
    axes[0].legend()
    fig.suptitle(f"CTQW vs classical walk on BinaryTree(depth={DEPTH})")
    save(fig, "walk_levels.png")
    save(fig_tree, "walk_on_tree.png")


def xor_qrc():
    reservoir = qk.qrc.CTQWReservoir(
        graph=qk.BinaryTree(depth=DEPTH),
        evolution_time=1.5,
        input_dim=2,
        seed=99,
        scale=3.0,
    )
    model = qk.qrc.QRC(reservoir, ridge_lambda=1e-4)

    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0.0, 1.0, 1.0, 0.0])
    model.fit(X, y)
    pred = model.predict(X)

    print(f"Part 2: QRC on XOR, reservoir = CTQW on BinaryTree(depth={DEPTH})")
    print(f"features = {reservoir.num_features} (one per vertex), linear readout\n")
    print(f"{'x1':>4} {'x2':>4} {'target':>7} {'pred':>9} {'rounded':>8}")
    for (x1, x2), t, p in zip(X, y, pred):
        print(f"{x1:>4.0f} {x2:>4.0f} {t:>7.0f} {p:>9.4f} {np.round(p) + 0.0:>8.0f}")
    accuracy = np.mean(np.round(pred) == y)
    print(f"\nAccuracy: {accuracy:.0%}")

    labels = [rf"$({x1:.0f}, {x2:.0f})$" for x1, x2 in X]
    ax = qk.plot_qrc_features(
        X,
        model.transform(X),
        sample_labels=labels,
        title=rf"QRC/XOR: reservoir features (CTQW, BinaryTree({DEPTH}))",
    )
    save(ax.get_figure(root=True), "xor_features.png")
    ax = qk.plot_qrc_prediction_bars(
        y,
        pred,
        sample_labels=labels,
        title=f"QRC/XOR: linear readout (accuracy={accuracy:.0%})",
    )
    save(ax.get_figure(root=True), "xor_prediction.png")


if __name__ == "__main__":
    walks()
    xor_qrc()
