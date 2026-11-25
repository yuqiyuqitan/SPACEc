# Choosing between cyto3 and Cellpose-SAM

SPACEc's [`cell_segmentation`](api/modules) function runs on top of
[Cellpose](https://cellpose.readthedocs.io/), and supports two generations of
it:

- **cyto3** — the classic per-tissue Cellpose model zoo, shipped with
  cellpose 3.x.
- **Cellpose-SAM** (`cpsam`) — a single SAM-based generalist model, shipped
  with cellpose 4.x. It generalizes across tissues and markers without
  picking a model, and is largely diameter- and channel-invariant.

Which one you get is determined by **which cellpose package is installed**
in your Python environment — SPACEc does not bundle a specific version.

## TL;DR

| You install | `model=None` gives you | Legacy names (`cyto3`, `nuclei`) | v3 fine-tuned custom models |
| --- | --- | --- | --- |
| `cellpose>=3,<4` | `cyto3` | work natively | work |
| `cellpose>=4` | `cpsam` | remapped to `cpsam` (with a warning) | **not loadable** |

## How to switch

Check what's installed:

```python
import spacec as sp
sp.hf.set_cellpose_version(4, dry_run=True)  # prints current + target, changes nothing
```

Or from a shell:

```bash
python -c "import cellpose; print(cellpose.version)"
```

Switch, in-notebook:

```python
sp.hf.set_cellpose_version(4)   # or 3 / "cyto3" / "sam" / "cpsam"
# then restart your kernel — the running process already has the old
# cellpose/torch imported and will not pick up the change otherwise.
```

Or manually with pip:

```bash
pip install "cellpose>=4,<5"    # Cellpose-SAM
pip install "cellpose>=3,<4"    # classic cyto3
```

In either case, `sp.tl.cell_segmentation(..., model=None)` (the default)
auto-selects the right generalist model for whatever is installed — you
don't need to change your call when you switch versions.

## When to stay on cyto3

- You have a custom model fine-tuned on cellpose 3.x.
- You need reproducibility with previously published SPACEc runs.
- You're on a small GPU and can't afford Cellpose-SAM's larger memory
  footprint.

## When to move to Cellpose-SAM

- You don't want to tune `diameter` per dataset.
- You want better generalization across markers/tissues without picking a
  model.
- Your channel choice (membrane vs. cytoplasm vs. nucleus-only) matters
  less, since the model is largely channel-invariant.

## What changes in your call

```python
# Before (cyto3)
sp.tl.cell_segmentation(
    ...,
    model="cyto3",
    diameter=28,
)

# After (either version — model=None auto-selects)
sp.tl.cell_segmentation(
    ...,
    model=None,
    diameter=28,              # still used to rescale toward Cellpose-SAM's 30px training mean
    flow_threshold=0.4,       # new tuning knob, replaces most of what diameter tuned in cyto3
    cellprob_threshold=0.0,   # new tuning knob
)
```

`compartment` and `plot_predictions` are no-ops (leftover from the removed
Mesmer backend) regardless of which cellpose you use.

## Gotchas

- **First-run download.** Cellpose-SAM's weights are downloaded on first
  use (roughly a gigabyte) and cached locally afterward.
- **More VRAM.** Cellpose-SAM needs more GPU memory than cyto3. If you hit
  out-of-memory errors, lower `tile_size` (e.g. to 1024–2048) and/or
  `batch_size`.
- **Results will differ.** Cellpose-SAM is not a drop-in numerical match for
  cyto3 — don't mix segmentation from the two versions within one study.
- **Custom models don't cross versions.** A cellpose-3 fine-tuned checkpoint
  cannot be loaded by Cellpose-SAM (different backbone). Retrain on cpsam,
  or pin `cellpose<4`.
- This is unrelated to the separate Mesmer→Cellpose fallback: `seg_method`
  only ever runs Cellpose now (Mesmer was removed); `model` is what picks
  cyto3 vs. cpsam.
