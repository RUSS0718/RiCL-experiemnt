"""Plot aggregate Motivation Study evidence; inputs come from the two offline replay scripts."""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from motivation_closeout import read


def draw(first, ms1):
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.titlepad": 7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "ricl-motivation-v1",
        }
    )
    fig, (a, b, c) = plt.subplots(
        3,
        1,
        figsize=(7.2, 5.1),
        layout="constrained",
        gridspec_kw={"height_ratios": [1.1, 1.45, 1.0]},
    )
    fig.get_layout_engine().set(h_pad=0.12)
    for ax in (a, b, c):
        ax.tick_params(axis="y", length=0)
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#e0e3e6", linewidth=0.5)

    colors = ("#156c82", "#b85d2d", "#c1a86a", "#e3e5e7")
    for row, key in enumerate(("ms2_a", "ms2_b")):
        pair = first[key]["paired"]
        prior = 0
        for field, label, color in zip(
            ("win", "loss", "tie", "missing"),
            ("Expected side wins", "Reversed", "Tie", "Missing"),
            colors,
            strict=True,
        ):
            n = pair["counts"].get(field, 0)
            a.barh(row, n, left=prior, height=0.58, color=color, label=label if not row else None)
            if n:
                a.text(
                    prior + n / 2,
                    row,
                    str(n),
                    va="center",
                    ha="center",
                    color="white" if field in ("win", "loss") else "#222222",
                )
            prior += n
        if prior != pair["coverage"]["denominator"]:
            raise ValueError("MS2 count does not match the fixed paired denominator")
    a.set(
        yticks=[0, 1], yticklabels=["MS2-A · label order", "MS2-B · historic outcome"], xlim=(0, 32)
    )
    a.set_title(
        "a   Frozen prefix scores: direction and missing pairs",
        loc="left",
        fontweight="bold",
        y=1.26,
    )
    a.invert_yaxis()
    a.set_xlabel("Planned original problem groups")
    a.legend(loc="lower left", bbox_to_anchor=(0, 1.01), frameon=False, ncols=4, fontsize=7)

    sources = ("GSM8K", "MATH", "OlympiadBench", "Omni-MATH")
    coverage = first["ms3_a"]["strata"]
    for row, (label, key) in enumerate(
        zip(sources, ("gsm8k", "math", "olympiadbench", "omnimath"), strict=True)
    ):
        n, total = (coverage[f"pb:{key}"][field] for field in ("n", "denominator"))
        b.barh(row, total, height=0.56, color="#e3e5e7")
        b.barh(row, n, height=0.56, color=colors[0])
        b.text(total + 0.16, row, f"{n}/{total}", va="center", color="#222222")
    b.set(yticks=range(4), yticklabels=sources, xlim=(0, 9.0), xticks=[0, 2, 4, 6, 8])
    b.invert_yaxis()
    b.set_title(
        "b   Frozen MS3-A: 17/32 complete across all three methods (gate 26/32)",
        loc="left",
        fontweight="bold",
    )
    b.set_xlabel("Groups evaluable by all methods")

    error = ms1["by_source_label"]["source_error"]
    for row, (label, key) in enumerate(
        (
            ("Error propagation", "error_propagation_ratio"),
            ("Reusable claims", "reusable_claim_ratio"),
        )
    ):
        m = error["summary"][key]
        ci = error["descriptive_path_bootstrap95"].get(key)
        if m["value"] is None or ci is None:
            c.text(0.04, row, "Not evaluable", va="center")
            continue
        lo, hi = ci
        c.hlines(row, lo, hi, colors=colors[0], linewidth=1.4)
        c.plot(m["value"], row, "o", color=colors[0], markersize=4.3)
        c.text(1.02, row, f"{m['numerator']}/{m['denominator']}", va="center")
    c.set(
        yticks=[0, 1],
        yticklabels=["Error propagation", "Reusable claims"],
        xlim=(0, 1.14),
        xticks=[0, 0.25, 0.5, 0.75, 1.0],
        xticklabels=["0", "25", "50", "75", "100"],
    )
    c.set_ylim(1.65, -0.65)
    c.set_title(
        f"c   MS1: confirmed fractions in {error['denominators']['annotation_paths']} public-error reference paths",
        loc="left",
        fontweight="bold",
    )
    c.set_xlabel("Fraction (%) · descriptive 95% problem-group bootstrap")
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("first", type=Path)
    p.add_argument("ms1", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    fig = draw(read(a.first), read(a.ms1))
    fig.savefig(a.output.with_suffix(".svg"), metadata={"Date": None})
    fig.savefig(a.output.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(a.output.with_suffix(".png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
