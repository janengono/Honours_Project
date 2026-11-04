# Pain Causal-XAI: Stage 1 Preprocessing Pipeline

Implements Section 3.2.2 (Signal Preprocessing) of the proposal for BioVid Part B.

## Setup

```bash
cd pain-causal-xai
python -m venv venv && source venv/bin/activate   # or conda
pip install -r requirements.txt
```

Edit `configs/config.yaml` and set `paths.root` to your actual Part B folder.

## Before running on the full dataset: verify the smoke test first

```bash
cd src
python data_loading.py
```

This loads `Samples.csv` and prints the first trial's data. **Check the output
carefully** — specifically:
- Does `Samples.csv` parse into the right columns? (subject_id, class_id, subject_name, sample_name)
- Does the raw biosignal file get found and loaded?
- Is the trial duration close to 5.5s?

If any of this looks wrong, it's almost certainly a delimiter or column-naming
mismatch specific to your download — fix it in `data_loading.py` before
running the full pipeline, since debugging on 8,600 files is much slower
than debugging on 1.

## Run on a small subset first (recommended)

```bash
python build_dataset.py --limit 20
```

Inspect `data/processed/features.csv` — check for NaNs, check that class_id
values look sensible (0-4), check that the physiological features vary
sensibly between baseline (class 0) and high pain (class 4) trials.

## Run the full pipeline

```bash
python build_dataset.py
```

This will take a while — 87 subjects x ~100 trials x 3 signal-processing
pipelines. Expect this to run for a meaningful chunk of time (test the
`--limit 20` run's speed first and extrapolate).

Add `--save-waveforms` if you also want the cleaned/normalized signal arrays
saved (not just handcrafted features) — you'll want this for feeding raw
sequences into the multimodal Transformer described in Section 3.3.2, rather
than only handcrafted feature vectors.

## Output

`data/processed/features.csv` — one row per trial, columns:
- `subject_id`, `class_id`, `subject_name`, `sample_name`
- ECG: `ecg_mean`, `ecg_std`, `hrv_mean_rr_ms`, `hrv_sdnn_ms`, `n_r_peaks`, `heart_rate_bpm`
- EDA: `eda_mean/std/max/min/slope`, `eda_tonic_mean/slope`, `eda_phasic_mean/std/max`, `scr_count`
- EMG (x3 channels): `{channel}_rms_mean`, `{channel}_rms_max`, `{channel}_rms_std`

## Known limitations to mention in your report

- **HRV from a 5.5s window is short.** Standard HRV analysis (SDNN, RMSSD,
  frequency-domain measures) typically wants ≥30s-2min of clean signal for
  stable estimates. With ~5.5s you'll get very few R-peaks per trial (roughly
  5-8 heartbeats), so HRV features here are noisy proxies, not textbook HRV.
  Worth explicitly flagging as a methodological limitation, and considering
  whether to compute HRV across concatenated *consecutive* trials per subject
  instead, if your temporal ordering data supports that.
- **Per-subject EDA normalization requires all of a subject's trials to load
  successfully.** If some files are missing/corrupt, that subject's z-score
  baseline will be computed from fewer trials — check the `[WARN]` log lines.
- **EMG channel quality**: corrugator/zygomaticus electrodes are on the face,
  so on some subjects/trials these channels are noisier than trapezius —
  worth a quick visual sanity check on a few trials before trusting the RMS
  features blindly.

## Next steps (not yet built)

- SHAP + modality-ablation causal analysis (Section 3.4)

---

# Stage 3: Multimodal Transformer (Section 3.3)

## What it does

- **`model.py`** -- `PainTransformer`: three modality-specific encoders
  (ECG, EDA, EMG -- EMG's three channels are concatenated into one 12-dim
  branch, consistent with the proposal's three-encoder structure), each a
  small single-layer Transformer encoder with positional encoding. Their
  outputs are concatenated along the sequence axis and passed through a
  **hand-implemented** multi-head scaled dot-product attention fusion layer
  (Equation 3.5 written out explicitly, not hidden inside a library call),
  which returns attention weights directly -- needed for Section 3.4's
  explainability work later. A small classification head produces 5-class
  logits (BL, PA1-PA4).
- **`pain_dataset.py`** -- wraps `sequences.npz` as a PyTorch `Dataset`,
  splitting the flat 20-feature array into per-modality tensors by reading
  `feature_names`, so it never hardcodes column positions.
- **`train.py`** -- training loop using the subject-level train/val split
  from Stage 2, cross-entropy loss (Equation 3.7), early stopping on
  validation F1-macro, checkpointing.
- **`smoke_test.py`** -- runs the entire pipeline on synthetic random data
  of the correct shape, to catch bugs before you spend real training time.
  **Run this first.**

## Run it, in order

```bash
cd src
pip install -r ../requirements.txt      # adds torch, scikit-learn
python smoke_test.py                    # must pass before anything else
python train.py --epochs 30             # trains on your real data
```

If `smoke_test.py` fails, that's a bug in the architecture code itself and
should be fixed before touching real data -- paste the error and it can be
diagnosed without needing your actual dataset.

`train.py` accepts `--epochs`, `--batch-size`, `--lr`, `--d-model`,
`--n-heads`, `--n-layers`, `--dropout`, `--early-stop-patience` if you want
to experiment with the architecture size. Defaults are deliberately small
(`d_model=32`, 1 layer per encoder) since you have ~6,000 training trials --
a much larger model risks overfitting badly, especially per-LOSO-fold later
where each fold trains on even less varied data.

## What's NOT here yet

- `loso_train.py` -- repeats `train.py`'s training procedure across all 86
  LOSO folds for your final reported result. Expensive (86x the compute of
  one `train.py` run) -- build and validate the single-split version first.
- `evaluate.py` -- final held-out test set evaluation (touch this only once,
  after all architecture/hyperparameter decisions are locked in, or you risk
  quietly overfitting your choices to the test set).

## Bonus: binary BL-vs-PA4 comparison

`train_binary.py` trains the same architecture on just baseline vs. maximum
pain (no pain vs. severe pain), directly precedented by Albahdal et al. in
your own lit review (Section 2.4/2.5), who found binary classification
consistently outperforms multi-class on BioVid. This gives you a stronger,
cleaner model to anchor the Section 3.4 explainability analysis with more
confidence, while the harder 5-class model remains your main, more
clinically representative result. Report both -- the gap between them is
itself a legitimate finding, not a fallback.

```bash
python train_binary.py --epochs 40
```

Saves to `checkpoints/best_model_binary.pt`, separate from the 5-class
checkpoint so both remain available for the explainability stage.

---

# Stage 4: Causal & Explainability Analysis (Section 3.4)

## Modality ablation (Section 3.4.3)

`modality_ablation.py` masks (zeros out) one modality at a time during
inference and measures the resulting drop in validation F1-macro. A large
drop means the model depends heavily on that modality; a small drop means
it's redundant with the others or wasn't contributing much. This is a
direct implementation of the "intervention-based analysis" your proposal
describes in Section 3.4.3.

```bash
python modality_ablation.py --checkpoint ../checkpoints/best_model.pt
python modality_ablation.py --checkpoint ../checkpoints/best_model_binary.pt --binary
```

Run it against BOTH checkpoints -- comparing whether the same modality
matters most for the easier binary task and the harder 5-class task is
itself worth a paragraph in your discussion section.

## Coming next

- Consolidating all Section 3.4 outputs into your Results chapter

## SHAP feature attribution (Section 3.4.2, Equation 3.8)

`shap_analysis.py` wraps `PainTransformer` so the `shap` library can treat
it as a standard black-box model, then runs `GradientExplainer` to get
per-feature, per-class SHAP values. Prints the top-5 most important features
per class and a modality-level importance summary (which should broadly
agree with `modality_ablation.py`'s ranking -- if it strongly disagrees,
that disagreement is itself worth investigating and discussing).

```bash
pip install shap
python shap_analysis.py --checkpoint ../checkpoints/best_model.pt
python shap_analysis.py --checkpoint ../checkpoints/best_model_binary.pt --binary
```

**Heads up:** this was written without a local `shap` install to test
against (no internet access in the dev environment). The core model code
is solid, but if `explainer.shap_values(...)` throws an error, that's
likely a version-specific SHAP API quirk, not a deeper bug -- paste the
error back and it can be patched quickly.

## Attention-weight visualization (Section 3.3.2's interpretability claim)

`attention_analysis.py` extracts the fusion layer's attention weights
(Equation 3.5) on real validation trials, averages them per class, and
aggregates from token-level (30x30) down to modality-level (3x3): how much
does each modality's tokens attend to each other modality's, on average,
when the model processes a baseline trial vs. a severe-pain trial?

```bash
python attention_analysis.py --checkpoint ../checkpoints/best_model.pt
python attention_analysis.py --checkpoint ../checkpoints/best_model_binary.pt --binary
```

Compare the printed matrices across classes -- a shift in which modality
dominates attention as pain intensity increases is direct evidence for the
proposal's claim that different physiological systems respond differently
across pain states, and complements the modality-ablation results (which
show WHICH modality matters, while this shows HOW the model uses it
relative to the others).


---

# Stage 2: Sub-windowing + Splits

Implements Section 3.2.3 (deep-learning sequence input) and Section 3.5 /
3.4.4 (validation strategy, subject-independent evaluation).

## What it does

- **`segmentation.py`** cuts each 5.5s trial into overlapping sub-windows
  (default: 1.0s windows, 0.5s overlap → 10 windows/trial) and computes
  mean/std/min/max per channel per window, giving each trial a sequence
  shape of `(10, 20)` instead of one flat feature vector -- this is what
  the Transformer's attention mechanism operates over.
- **`build_sequences.py`** runs this across all 8,600 trials and saves
  `data/processed/sequences.npz` (`X`, `y`, `subject_ids`, `sample_names`, `feature_names`).
- **`splits.py`** provides subject-level train/val/test splitting (for fast
  iteration while developing the model) and Leave-One-Subject-Out fold
  generation (for your final reported result, per Section 3.5).

## Run it

```bash
cd src
python build_sequences.py --limit 20   # quick test first
python build_sequences.py              # full run once the test looks right
python splits.py                       # smoke test the splitting logic
```

## Why subject-level splitting, not trial-level

A subject's trials all share that person's baseline physiology. If the same
subject appears in both train and test, the model can partly learn to
recognize *the person* rather than the pain response, which inflates
accuracy in a way that won't generalize. This is directly the confound your
proposal's Section 3.4.4 discusses, so getting this right is part of your
contribution, not just a coding detail -- worth a paragraph in your report.

## Two evaluation protocols, two purposes

- **train/val/test** (`subject_level_split`): use this while you're building
  and debugging the Transformer -- fast, one train run per experiment.
- **LOSO** (`loso_folds`): 86 folds, one model trained per held-out subject.
  Expensive (86x the training cost) but this is the number your proposal
  commits to reporting as the final result. Budget real compute time for
  this once the architecture is finalized -- don't run it during early
  iteration.

