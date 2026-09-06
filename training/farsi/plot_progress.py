"""Render a run's progress.jsonl as a standalone HTML chart.

No plotting library and no network: the output is one self-contained file with
inline SVG, so you can scp it off the VM and open it anywhere.

Per-step metrics are extremely noisy here -- LSD draws `t` from a lognormal
every step, so consecutive values of flow_diag can differ several-fold. Each
panel therefore shows a faint raw trace under a rolling median, and the median
is the line to read.

    python -m training.farsi.plot_progress /mnt/data/runs/lsd_scratch_fa2
"""

import json
import logging
import statistics as st
from pathlib import Path

import typer
from typing_extensions import Annotated

logger = logging.getLogger("plot_progress")
app = typer.Typer(pretty_exceptions_show_locals=False)

W, H = 900, 220
PAD_L, PAD_R, PAD_T, PAD_B = 70, 20, 26, 40

# Different training objectives log different metrics -- a from-scratch/LSD run
# reports flow_diag/flow_loss/eos_loss, a depth-distillation run reports only
# distill_mse (see training/modules/model.py). Rendering the from-scratch panel
# list unconditionally against a distillation log produces three "no data"
# panels and hides the one metric that run actually has. The panel list is
# built per-file instead, from whichever of these keys are present.
METRIC_REGISTRY: dict[str, tuple[str, bool]] = {
    "flow_diag": ("flow_diag (raw flow-matching MSE)", True),
    "flow_loss": ("flow_loss (uncertainty-weighted; goes negative by design)", False),
    "eos_loss": ("eos_loss (length control)", True),
    # LSD's OWN self-distillation term (see training/modules/samplers.py) --
    # unrelated to the depth/CFG distillation that logs distill_mse. Computed
    # on only distill_prob (25% by default) of steps, so it is sparser and
    # noisier than the other panels.
    "flow_distill": ("flow_distill (LSD's self-distillation term)", True),
    "distill_mse": ("distill_mse (student vs. teacher backbone activations)", True),
    # On a scratch/LSD run "loss" = flow_loss + eos_loss_weight * eos_loss, the
    # actual optimized objective -- a real, distinct metric. On a distillation
    # run it is set to exactly distill_mse.detach() a second time (see
    # training/modules/model.py); main() drops it there rather than show the
    # same curve under two names.
    "loss": ("loss (combined training objective)", False),
    "grad_norm": ("grad_norm (clipped at 1.0 when applied)", True),
}
PANEL_ORDER = [
    "flow_diag",
    "flow_loss",
    "eos_loss",
    "flow_distill",
    "distill_mse",
    "loss",
    "grad_norm",
]

EXPLANATIONS = {
    "distill": (
        "<p class='explain'>This is a <strong>depth-distillation</strong> run: a smaller "
        "student learns to reproduce a larger teacher's backbone activations, rather than "
        "being trained on the flow-matching objective directly -- so <code>distill_mse</code> "
        "is the only loss there is, and it is read the same way as <code>flow_diag</code> on a "
        "from-scratch run: a falling <em>floor</em> under the noise means the student is still "
        "converging. WER and speaker similarity typically reach teacher parity by ~40k steps, "
        "with prosody continuing to settle after that. A rising <code>grad_norm</code> late in "
        "the schedule is not on its own a sign of trouble -- check it against whether "
        "<code>distill_mse</code>'s floor is still falling in the same window.</p>"
    ),
    "scratch": (
        "<p class='explain'>This is a <strong>from-scratch</strong> LSD run. Per-step values are "
        "extremely noisy -- <code>t</code> is redrawn from a lognormal every step, so consecutive "
        "readings of any metric can differ several-fold; the rolling median (blue) is the line to "
        "read, not the raw trace (grey). <code>flow_loss</code> is uncertainty-weighted and goes "
        "<strong>negative by design</strong> once the weighting network learns -- watch "
        "<code>flow_diag</code> instead, the raw unweighted flow-matching error. The acoustic "
        "quality transition (where a checkpoint stops sounding flat) typically only lifts off "
        "around 150-200k steps even once <code>flow_diag</code> has long since plateaued.</p>"
    ),
}


def read_progress(path: Path) -> tuple[list[dict], list[dict]]:
    train, valid = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue  # a torn last line after a preemption
            if d.get("type") == "train":
                train.append(d)
            elif d.get("type") == "valid":
                valid.append(d)

    # A resumed run re-appends earlier steps; keep the last value seen per step.
    def dedupe(rows: list[dict]) -> list[dict]:
        by_step: dict[int, dict] = {}
        for r in rows:
            by_step[r["step"]] = r
        return [by_step[s] for s in sorted(by_step)]

    return dedupe(train), dedupe(valid)


def series(rows: list[dict], key: str) -> list[tuple[int, float]]:
    out = []
    for r in rows:
        v = r.get(key, r.get("metrics", {}).get(key))
        if isinstance(v, (int, float)):
            out.append((r["step"], float(v)))
    return out


def subsample(pts: list[tuple[int, float]], cap: int = 1500) -> list[tuple[int, float]]:
    """Thin the faint raw trace: 400k steps of logs make a needlessly huge SVG."""
    if len(pts) <= cap:
        return pts
    stride = len(pts) // cap + 1
    return pts[::stride]


def rolling_median(pts: list[tuple[int, float]], window: int) -> list[tuple[int, float]]:
    if len(pts) < window:
        return pts
    out = []
    for i in range(0, len(pts) - window + 1, max(1, window // 4)):
        chunk = pts[i : i + window]
        out.append((chunk[len(chunk) // 2][0], st.median(v for _, v in chunk)))
    return out


def panel(
    title: str,
    raw: list[tuple[int, float]],
    med: list[tuple[int, float]],
    valid: list[tuple[int, float]],
    logy: bool,
) -> str:
    vals = [v for _, v in med] or [v for _, v in raw]
    if not vals:
        return f'<p class="empty">{title}: no data</p>'
    steps = [s for s, _ in raw] or [0]
    x0, x1 = min(steps), max(max(steps), 1)
    lo, hi = min(vals), max(vals)
    use_log = logy and lo > 0 and hi / max(lo, 1e-12) > 20
    if use_log:
        import math

        f = math.log10
        lo, hi = f(lo), f(hi)
    else:
        f = float
    if hi - lo < 1e-9:
        hi = lo + 1.0
    pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad

    def X(s: float) -> float:
        return PAD_L + (s - x0) / max(x1 - x0, 1) * (W - PAD_L - PAD_R)

    def Y(v: float) -> float:
        return PAD_T + (hi - f(v)) / (hi - lo) * (H - PAD_T - PAD_B)

    def path(pts: list[tuple[int, float]], cls: str) -> str:
        if not pts:
            return ""
        d = " ".join(
            f"{'M' if i == 0 else 'L'}{X(s):.1f},{Y(v):.1f}"
            for i, (s, v) in enumerate(pts)
            if not (use_log and v <= 0)
        )
        return f'<path class="{cls}" d="{d}"/>' if d else ""

    ticks = ""
    for i in range(5):
        v = lo + (hi - lo) * i / 4
        y = PAD_T + (hi - v) / (hi - lo) * (H - PAD_T - PAD_B)
        label = f"{10**v:.3g}" if use_log else f"{v:.3g}"
        ticks += (
            f'<line class="grid" x1="{PAD_L}" y1="{y:.1f}" x2="{W - PAD_R}" y2="{y:.1f}"/>'
            f'<text class="ytick" x="{PAD_L - 8}" y="{y + 4:.1f}">{label}</text>'
        )
    for i in range(6):
        s = x0 + (x1 - x0) * i / 5
        ticks += f'<text class="xtick" x="{X(s):.1f}" y="{H - 10}">{int(s):,}</text>'

    return (
        f"<h2>{title}{' <em>(log scale)</em>' if use_log else ''}</h2>"
        # width/height attributes (not just viewBox) give the SVG a guaranteed
        # intrinsic aspect ratio; without them "height:auto" in CSS has, in
        # practice, collapsed to 0 in some rendering contexts, and combined
        # with overflow:visible that painted the whole chart across the rest
        # of the page instead of clipping to its own box.
        f'<svg viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
        f'preserveAspectRatio="xMidYMid meet" role="img" aria-label="{title}">{ticks}'
        f"{path(raw, 'raw')}{path(med, 'med')}{path(valid, 'valid')}</svg>"
    )


@app.command()
def main(
    run_dir: Annotated[str, typer.Argument(help="a training run directory")],
    out: Annotated[
        str | None, typer.Option(help="output html (default: <run_dir>/progress.html)")
    ] = None,
    window: Annotated[int, typer.Option(help="rolling-median window, in logged points")] = 51,
) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    run = Path(run_dir)
    train, valid = read_progress(run / "progress.jsonl")
    if not train:
        raise SystemExit(f"no train entries in {run / 'progress.jsonl'}")
    logger.info(f"{len(train)} train points, {len(valid)} valid points")

    # Which metrics this particular run actually logged. distill_mse without
    # flow_diag means a depth-distillation run; anything else registered but
    # present goes in known order, and any further numeric key this file has
    # that the registry does not know about is still shown rather than
    # silently dropped, just without curated framing.
    present = {
        k for r in train for k, v in r.get("metrics", {}).items() if isinstance(v, (int, float))
    }
    if any(isinstance(r.get("grad_norm"), (int, float)) for r in train):
        present.add("grad_norm")

    is_distill = "distill_mse" in present and "flow_diag" not in present
    if is_distill:
        present.discard("loss")  # exact duplicate of distill_mse on this objective
    explain = EXPLANATIONS["distill" if is_distill else "scratch"]

    ordered = [k for k in PANEL_ORDER if k in present]
    ordered += sorted(present - set(ordered))
    if not ordered:
        raise SystemExit(f"no numeric metrics found in {run / 'progress.jsonl'}")
    logger.info(f"panels: {', '.join(ordered)}")

    body = ""
    for key in ordered:
        title, logy = METRIC_REGISTRY.get(key, (key, True))
        raw = series(train, key)
        body += panel(title, subsample(raw), rolling_median(raw, window), series(valid, key), logy)

    last = train[-1]
    head = (
        f"<h1>{run.name}</h1><p class='meta'>step {last['step']:,} &middot; "
        f"lr {last.get('lr', 0):.2e} &middot; {len(train):,} logged points &middot; "
        f"blue = rolling median (window {window}), grey = raw, orange = validation</p>"
        f"{explain}"
    )
    html = f"""<meta charset="utf-8"><title>{run.name} training progress</title>
<style>
 :root {{ --bg:#fff; --fg:#1a1a1a; --muted:#666; --grid:#e6e6e6;
          --raw:#c9d4e4; --med:#2b6cb0; --valid:#dd6b20; }}
 @media (prefers-color-scheme: dark) {{ :root:not([data-theme=light]) {{
   --bg:#14161a; --fg:#e8e8e8; --muted:#9aa; --grid:#2a2f36;
   --raw:#39445280; --med:#63a4e0; --valid:#f0904a; }} }}
 body {{ background:var(--bg); color:var(--fg); font:14px/1.5 system-ui,sans-serif;
         margin:0 auto; padding:24px; max-width:960px; }}
 h1 {{ font-size:20px; margin:0 0 4px; }}
 h2 {{ font-size:14px; font-weight:600; margin:22px 0 2px; }}
 h2 em {{ color:var(--muted); font-weight:400; }}
 .meta {{ color:var(--muted); margin:0 0 8px; }}
 .explain {{ margin:12px 0 20px; padding:10px 14px; border-radius:6px;
             background:color-mix(in srgb, var(--fg) 5%, transparent);
             border:1px solid var(--grid); font-size:13px; }}
 .explain code {{ font-size:12px; }}
 /* Explicit width/height on the <svg> itself (not just viewBox) give it a
    guaranteed intrinsic size; do not add overflow:visible back -- combined
    with any auto-height miscalculation that is what let a chart's raw trace
    paint across the whole page instead of clipping to its own box. */
 svg {{ display:block; width:100%; height:auto; }}
 .grid {{ stroke:var(--grid); stroke-width:1; }}
 .raw {{ fill:none; stroke:var(--raw); stroke-width:1; }}
 .med {{ fill:none; stroke:var(--med); stroke-width:2; }}
 .valid {{ fill:none; stroke:var(--valid); stroke-width:1.5; stroke-dasharray:4 3; }}
 .ytick {{ fill:var(--muted); font-size:11px; text-anchor:end; }}
 .xtick {{ fill:var(--muted); font-size:11px; text-anchor:middle; }}
 .empty {{ color:var(--muted); }}
</style>
{head}{body}"""
    dest = Path(out) if out else run / "progress.html"
    dest.write_text(html, encoding="utf-8")
    logger.info(f"wrote {dest.resolve()}")


if __name__ == "__main__":
    app()
