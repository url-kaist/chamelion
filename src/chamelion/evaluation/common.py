"""Shared point metrics, checksums and result figures."""
import hashlib

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def metrics(predicted, labels):
    valid = np.isin(labels, [0, 1, 2])
    prediction = predicted[valid].astype(bool)
    target = labels[valid] > 0
    return dict(
        tp=int(np.count_nonzero(prediction & target)),
        fp=int(np.count_nonzero(prediction & ~target)),
        fn=int(np.count_nonzero(~prediction & target)),
        tn=int(np.count_nonzero(~prediction & ~target)),
    )


def scores(counts):
    tp, fp, fn = (counts[name] for name in ("tp", "fp", "fn"))

    def divide(a, b):
        return a / b if b else None

    return {
        **counts,
        "iou": divide(tp, tp + fp + fn),
        "precision": divide(tp, tp + fp),
        "recall": divide(tp, tp + fn),
    }


def plot_points(ax, points, categories, colors, names):
    for category, (color, name) in enumerate(zip(colors, names)):
        indices = np.flatnonzero(categories == category)
        if len(indices) > 50000:
            indices = indices[np.linspace(0, len(indices) - 1, 50000).astype(int)]
        if len(indices):
            ax.scatter(
                points[indices, 0],
                points[indices, 1],
                s=0.5 if category == 0 else 2,
                color=color,
                label=name,
                rasterized=True,
            )
    ax.set_aspect("equal")
    ax.set_xlabel("Global X (m)")
    ax.set_ylabel("Global Y (m)")
    ax.legend(markerscale=4, fontsize=7)


def preview(
    destination,
    sequence,
    frame,
    scan,
    scan_gt,
    scan_pred,
    prior,
    map_gt,
    map_pred,
    caption="Raw model, logit > 0; no confidence gate, fusion or test-set tuning",
    prediction_title="Raw prediction",
):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for row, (points, gt, pred, kind, color) in enumerate(
        [
            (scan, scan_gt, scan_pred, "Scan / added change", "#1684dd"),
            (prior, map_gt, map_pred, "Prior map / removed change", "#df463d"),
        ]
    ):
        valid = np.isin(gt, [0, 1, 2])
        points, gt, pred = points[valid], gt[valid] > 0, pred[valid]
        for col, (classes, title) in enumerate([(gt, "Ground truth"), (pred, prediction_title)]):
            plot_points(
                axes[row, col],
                points,
                classes.astype(int),
                ["#b6bcc4", color],
                ["Static", "Changed"],
            )
            axes[row, col].set_title(f"{kind}: {title}")
        errors = np.zeros(len(gt), dtype=int)
        errors[gt & pred] = 1
        errors[~gt & pred] = 2
        errors[gt & ~pred] = 3
        plot_points(
            axes[row, 2],
            points,
            errors,
            ["#c5c9ce", "#179b62", "#e14742", "#287ed8"],
            ["True static", "True change", "False positive", "Missed change"],
        )
        axes[row, 2].set_title(f"{kind}: Errors")
        for ax in axes[row, 1:]:
            ax.set_xlim(axes[row, 0].get_xlim())
            ax.set_ylim(axes[row, 0].get_ylim())
    fig.suptitle(f"{sequence} | frame {frame}\n{caption}")
    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
