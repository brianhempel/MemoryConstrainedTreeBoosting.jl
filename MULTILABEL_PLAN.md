# Plan: Multi-Label Classification Support

Goal: train on `y` with `L` binary labels per datapoint (multi-*label*, i.e. independent
sigmoids per label — not mutually-exclusive multi-class softmax). Labels may have vastly
different frequencies, so also add an optional `equalize_label_losses` flag that reweights
each label's gradients so that, after the constant first tree, every label contributes
equal loss to training — otherwise the most entropic (most frequent) label dominates
split selection.

## 1. Architecture decision: one shared tree structure, vector-valued leaves

Two candidate designs:

**(a) Round-robin trees, one label per tree** (XGBoost-multiclass style). Smallest diff —
leaves stay scalar, the histogram kernel is untouched. But `X_binned` traffic is
essentially fixed at O(node datapoints × features), visited once per node (absent the
bug-prone same-depth sibling cleverness this library deliberately eschews), and (a)
needs `L`× as many trees — hence `L`× as many nodes — for the same predictive budget,
i.e. `L`× the fixed, dominant traffic (X_binned is the ~171 GB monster in the HRRR
case).

**(b) Shared tree structure, `L` Δscores per leaf** (multi-output trees). One pass over
`X_binned` per tree builds histograms for *all* labels simultaneously. Split gain is the
sum of per-label gains; each leaf then gets an independent optimal Δscore per label.
Constraint: all labels share the same split structure per tree. For correlated labels
(e.g. related hazard types) this is fine and even acts as regularization; boosting over
many trees lets different trees serve different labels anyway.

**Decision: (b).** Two reasons, and the second is decisive:

1. It amortizes each node's fixed `X_binned` visitation across all `L` labels — one
   node visit builds every label's histograms — instead of paying it `L` times.
2. **The intended labels are nested severity thresholds — e.g. EF0+, EF1+, EF2+
   tornadoes — and the downstream goal is conditional probabilities like
   P(EF2+ | EF0+) = P(EF2+) / P(EF0+).** That ratio is only reliable if both
   probabilities are estimated over the *same* partitions of feature space: with shared
   tree structure, every leaf carries all labels' Δscores fit on the same datapoints, so
   numerator and denominator move together. Separate per-label trees with different
   splits would make the ratio noisy garbage precisely in the sparse-data regions where
   it matters. So (a) is not merely slower — it fails the use case.

(a) remains the fallback only if the kernel generalization stalls — noted in Risks —
and would need revisiting the conditional-probability goal if ever used.

Note for nested labels: EF2+ ⊆ EF0+ means empirical rates in every leaf satisfy
rate(EF2+) ≤ rate(EF0+), and per-label Newton steps track those empirical rates, but
independent sigmoids give **no hard guarantee** that predicted P(EF2+) ≤ P(EF0+)
pointwise after many clamped, shrunk, L2-regularized steps accumulate. See Testing
(monotonicity diagnostic) and Risks. Also, nested thresholds make `equalize_label_losses`
directly load-bearing: EF2+ is far rarer than EF0+, so without it EF0+ dominates split
selection and the conditional's numerator is the worst-trained label.

### Why (b) is cheap here: the layout generalizes by construction

Current per-datapoint loss info (`∇losses_∇∇losses_weights`, stride 4):

```
[∇loss, ∇∇loss, weight, pad]
```

and each histogram bin has the *same* stride-4 layout, so the SIMD kernel core in
`_build_histogram_unrolled!` / `_build_2histograms_unrolled!` / `_build_3histograms_unrolled!`
is literally `vstorea(vloada(bin) + vloada(point_loss_info))`.

Generalized layout, stride `S = 4 * cld(2L + 2, 4)` floats:

```
[∇₁, ∇∇₁, ∇₂, ∇∇₂, …, ∇_L, ∇∇_L, weight, pad…]
```

per datapoint *and* per histogram bin. For `L = 1`, `S = 4` and this degenerates to the
existing layout exactly — the current code is the special case. The kernel inner body
becomes a loop of `S ÷ 4` `Vec{4,Float32}` add-accumulates instead of one; everything
else about the kernels (index math via a generalized `llw_base_i`, chunking, work
stealing, Float64 accumulator dumps) is unchanged in shape.

Where the cost lands: the actual cache optimization in this library is not about
`X_binned` (whose traffic is fixed per node) but about **loss-info residency** — loss
info is revisited *per feature*, which is why `make_a_chunk_of_histograms` iterates a
resident chunk of loss-info records against its work-chunk's features (the grid noted
in the comments around `src/MemoryConstrainedTreeBoosting.jl:1852`). Stride `S` grows
each record `S/4`×, so the per-feature loss-info reload traffic grows `S/4`× and the
rows-per-chunk must shrink ~`4/S`× to keep the working set resident (see Risks).
`X_binned` traffic stays 1× per tree — amortized over all `L` labels.

## 2. API surface

- `train(X, y; …)` / `train_on_binned(X_binned, y; …)`: accept `y :: AbstractMatrix`
  of size `data_count × L` (labels in columns, matching `X`'s row-per-datapoint
  convention). `y :: AbstractVector` keeps working and means `L = 1` — the entire
  existing API remains valid.
- `predict` / `predict_on_binned` / `load_unbinned_predictor`'s closure: return an
  `n × L` `Matrix{Prediction}` when `L > 1`, a `Vector` when the model was trained
  single-label (so existing callers see no change). `output_raw_scores` likewise.
- `starting_scores`: `n × L` matrix (or vector for `L = 1`).
- New config keys in `default_config`:
  - `equalize_label_losses = false` — see §4.
  - `label_weights = nothing` — optional explicit `Vector{Float32}` of length `L`,
    composable with `equalize_label_losses` (multiplied together if both given).
- **Label mask ("don't care" entries):** `NaN` in `y` marks a (datapoint, label) pair
  as contributing zero loss — see §5. No separate mask matrix, no new config key.
- `validation_y`: matrix, same convention, including `NaN` don't-cares.
- **Per-label early stopping (§6):** the validation callback becomes a callable
  `ValidationLossTracker` exposing per-label and global best iterations/losses after
  training; `predict` / `predict_on_binned` / `load_unbinned_predictor` gain a
  `label_tree_counts` argument so label ℓ can ignore trees past its own optimum.

Internally, store scores and y transposed/interleaved per datapoint (`L × n`) so that a
datapoint's `L` scores share a cache line, mirroring the loss-info interleave. Transpose
once at entry; document that `train_on_binned` callers passing huge `y` pay one transpose.

## 3. Core changes, file by file (all in `src/MemoryConstrainedTreeBoosting.jl`)

Thread a `label_count` (as `Val(L)` where it matters for codegen, plain `Int` elsewhere)
from `train_on_binned` down through `train_one_iteration` → `build_one_tree` →
`perhaps_split_tree`.

### Data structures
- `Leaf.Δscore :: Score` → `Δscores :: Vector{Score}` (length `L`). Same for
  `SplitCandidate.left_Δscore` / `right_Δscore` → `left_Δscores` / `right_Δscores`.
  `dont_split` needs to be constructed per-`L` (or lazily) instead of a global singleton.
- `FastNode.Δscore` → hold a leaf index into a separate `leaf_Δscores :: Matrix{Score}`
  (L × n_leaves), or an `NTuple` — decide during implementation; the apply loop then does
  `scores[:, i] .+= …` over `L`.
- `scale_leaf_Δscores`, `strip_tree_training_info`, `print_tree`,
  `feature_importance_by_absolute_delta_score` (sum abs over labels): mechanical updates.

### Layout helpers (~line 1143)
- `llw_base_i(i)` = `-3 + 4i` → `llw_base_i(i, S)` = `1 + S*(i-1)`; slot of `∇_ℓ` is
  `+2(ℓ-1)`, `∇∇_ℓ` is `+2ℓ-1` (0-based: `2ℓ-2`, `2ℓ-1`), weight at `+2L`.
- `llw_∇losses` / `llw_∇∇losses` / `llw_weights` views: take `S` and label index.

### Gradient computation (~line 1164)
- `compute_∇losses_∇∇losses!(y, scores, llw)`: loop labels inside the per-datapoint loop;
  `∇logloss` / `∇∇logloss` unchanged. Masked entries (`isnan(y)`) zero both ∇ and ∇∇
  branch-free: `m = ifelse(isnan(y_ℓi), 0f0, 1f0)` folded into the multiplier (cost: one
  compare per label per point). Per-label multiplier `λ_ℓ`
  (`label_weights × equalization`) is folded into **∇ and ∇∇ directly, not the weight
  slot** — the weight slot stays label-agnostic because it feeds
  `min_data_weight_in_leaf` and bin data-weight counts, which must keep meaning
  "datapoints", not "reweighted loss mass".
- `compute_weights!` / `bagged_weights!`: unchanged (one weight per datapoint; bagging
  scales all labels of a point together, which is the statistically right thing).

### Histogram kernels (~lines 1441–1686)
- Generalize the three unrolled builders to stride `S`: the `Vec{4}` load/add/store
  becomes a small `for v in 0:(S÷4 - 1)` loop (constant-unrolled via `Val{S}` /
  `@generated` so `L = 1` compiles to today's exact code — verify with `@code_native`).
- `consolidate_∇losses_∇∇losses_weights!` + its unrolled helper: same stride
  generalization (it copies whole per-point records; the 4-wide SIMD copy becomes
  `S÷4` copies).
- Histogram allocation math: `ScratchHistograms` histogram_size `4*max_bins` →
  `S*max_bins` (+ cache-line pad); same in `next_free_histogram` and
  `ScratchMemory` (`data_count*4` → `data_count*S`, and the consolidated scratch
  `*2 or *4` → `*S`-scaled equivalents). `scratch_accss` accumulator size likewise.
- `acc_hist!` / `acc_hist_final!`: already length-generic; no change beyond sizes.

### Split evaluation (~lines 1980–2125)
- `sum_histogram`: return per-label `(Σ∇_ℓ, Σ∇∇_ℓ)` plus shared `Σweight`. Keep it
  allocation-free: accumulate in `S÷4` `Vec{4}` lanes then unpack, or write into a
  caller-provided small scratch.
- `expected_Δloss_for_feature` and `best_split_for_feature`: the left-sums become
  per-label running sums (fixed-size, stack or per-thread scratch — no allocation in
  the bin loop). Gain at a candidate split:
  `Σ_ℓ [ -leaf_Δloss_ℓ + left_Δloss_ℓ + right_Δloss_ℓ ]`, each term via the existing
  `leaf_expected_Δloss` / `optimal_Δscore` per label (per-label clamp to
  `max_delta_score` as today). `min_data_weight_in_leaf` checks use the shared weight
  slot, unchanged.
- `sum_optimal_Δscore` (root leaf init) → per-label version; MPI sum sends `2L+1`
  floats instead of 2–3.
- Second-opinion logic (`find_best_split`): unchanged — it operates on the summed
  `features_expected_Δlosses`, which are already label-aggregated by the time it runs.

### Apply / predict
- `apply_tree!` / `_apply_tree!` / `apply_trees` / `load_unbinned_predictor`: scores
  become the `L`-interleaved buffer; the leaf hit adds `L` values. Keep a scalar `L = 1`
  method so single-label predict speed is untouched.
- `predict_on_binned`: `σ` per element; un-transpose to `n × L` at the boundary.

### Losses / callbacks
- `compute_mean_probability` → per-label vector (weighted); one MPI Allreduce of `L+1`
  values.
- `compute_mean_logloss`: per-label means (masked-aware, §5), combined via λ. The
  validation callback becomes a callable `ValidationLossTracker` struct implementing
  per-label early stopping — full design in §6. It needs the same `λ` used in training
  so the global-loss marker matches the training objective; note `train` builds the
  callback *before* `train_on_binned` computes base rates, so either hoist the
  base-rate/λ computation up into `train`, or make the tracker compute λ lazily on
  first call from the same y/weights formula (deterministic; MPI-safe since it derives
  from Allreduce'd sums).

### Initial constant tree (`train_on_binned`, ~line 834)
- Per-label base rate `π_ℓ = compute_mean_probability(y_ℓ, weights)`, initial leaf
  `Δscores[ℓ] = log(π_ℓ / (1 - π_ℓ))`.

### Save / load (~lines 270–322)
- `tree_to_dict` Leaf: `:delta_score => Float64` → `:delta_scores => Vector{Float64}`.
  Bump: write `:format_version => 2` and `label_count` at top level of the BSON.
  `load` accepts both: absent version / `:delta_score` scalar ⇒ `L = 1` model. `save`
  of an `L = 1` model may keep the old field for round-trip compat with old readers —
  decide; leaning "always write v2, keep v1 *reading* only".

### MPI (~lines 1011–1073)
- `copy_llwd_hist_to_llw_hist!` / inverse: strip padding generically — send `2L+1` of
  every `S` floats; `hist_size = 3*max_bins` → `(2L+1)*max_bins` in
  `mpi_sum_histograms!`.
- λ / base rates already derive from Allreduce'd sums ⇒ identical on all ranks; no new
  broadcast needed.

## 4. `equalize_label_losses` flag — semantics

After computing the per-label base rates `π_ℓ` for the constant first tree (this is the
"naive constant prediction"), the expected per-datapoint logloss of that constant model
is just the label's Bernoulli entropy (weighted by training weights only through `π_ℓ`
itself, which is already weight-aware):

```
H_ℓ = -π_ℓ·log(π_ℓ) - (1-π_ℓ)·log(1-π_ℓ)
```

With label masks (§5), all of this is computed over *unmasked* entries only: π_ℓ is the
weighted base rate among points where label ℓ is observed, and the label's total initial
loss is `T_ℓ = W_ℓ·H_ℓ` where `W_ℓ = Σᵢ wᵢ·maskᵢℓ` is the unmasked weight for label ℓ.
Equalize total contributions:

```
λ_ℓ = mean(T) / T_ℓ        (so Σ λ_ℓ·T_ℓ = Σ mean(T) — total initial loss preserved,
                            each label contributes exactly mean(T) of it)
```

With no masks, `W_ℓ` is the same for all labels and this reduces to `mean(H)/H_ℓ`.
The `W_ℓ` factor matters for the masked-conditional use case: EF2+-masked-to-EF0+ sees
far fewer points, and equalizing per-point entropy alone would under-weight it.

Rationale for `mean(H)/H_ℓ` rather than `1/H_ℓ`: keeps the total gradient magnitude —
and therefore the effective `learning_rate`, `l2_regularization`, and `max_delta_score`
scales — comparable to the unweighted run, so the flag doesn't silently change other
hyperparameters' meaning.

Application points:
1. Multiply `∇_ℓ` and `∇∇_ℓ` by `λ_ℓ` in `compute_∇losses_∇∇losses!` (both, so the
   Newton step `-Σ∇/Σ∇∇` for a pure single-label leaf is *unchanged* — reweighting
   shifts which splits win and how labels trade off inside shared leaves, without
   inflating step sizes for rare labels; the l2 term does shrink rare-label steps
   relatively more, which is acceptable and stabilizing).
2. Same `λ_ℓ` in training and validation loss reporting / early stopping (§3), else
   early stopping re-anchors to the entropic label. Report per-label unweighted losses
   alongside for interpretability.
3. When prior_trees are supplied (continued training), still compute λ from base rates
   of `y` — document that λ is a function of the *data*, not the prior model.
4. If `label_weights` is also given: use `label_weights .* λ`.
5. `L = 1`: flag is a no-op (λ = 1); warn if set.

Guard: if some `π_ℓ` is 0 or 1 (label never/always present among unmasked points), or
`W_ℓ = 0` (label fully masked), `T_ℓ = 0` ⇒ divide-by-zero. Error out with a clear
message naming the label index — a degenerate label is a data bug.

## 5. Label mask — per-(datapoint, label) "don't care"

Some (datapoint, label) pairs should contribute **zero loss** for that label while the
datapoint still trains the other labels normally. Motivating cases:

1. **Tropical cyclones and the wind label:** TC wind damage is not classified as severe
   thunderstorm wind, so there are no severe wind reports during TCs — those datapoints
   are unlabelable for wind, not negative. Mask wind during TCs; hail and tornado labels
   still train on those points.
2. **Directly training conditionals:** mask EF2+'s loss to the EF0+ points. Then the
   EF2+ head is trained only where the conditioning event occurred, so its output *is*
   P(EF2+ | EF0+) (its base rate is the conditional base rate, its gradients only ever
   see EF0+ points). Outside the EF0+ region its prediction is an extrapolation —
   meaningful exactly as "what the conditional would be, were EF0+ to occur here."

### Encoding

`NaN` in `y` (and `validation_y`) marks a don't-care entry. Rationale: zero extra
memory (no second n×L matrix riding along next to the giant `X_binned`), survives the
internal transpose for free, impossible to have a mask/y shape mismatch, and `NaN` can
never be a valid label value. `missing`-friendly convenience: accept
`Union{Missing,<:Real}` matrices at the API boundary and convert to `NaN` on transpose.

### Semantics (all falls out of ∇ = ∇∇ = 0 for masked entries)

- **Histograms / split gain:** masked entries add nothing to label ℓ's `Σ∇`/`Σ∇∇` bins,
  so they exert no pull on split selection or leaf Δscores for that label. No kernel
  changes beyond the multiplier in `compute_∇losses_∇∇losses!` (§3) — the hot path
  never sees the mask.
- **Weight slot / `min_data_weight_in_leaf`:** unchanged — the shared weight slot keeps
  counting the datapoint. A leaf can therefore satisfy the weight guard while having
  few unmasked points for some label; that label's leaf Δscore is then driven by its
  few unmasked points plus `l2_regularization` (which shrinks it toward 0 — i.e. toward
  no change — the right default for thin evidence). Document; no special handling.
- **Base rates / initial tree:** `compute_mean_probability` per label sums only unmasked
  entries (mask factor alongside the weight). The constant tree's `c_ℓ` is the logit of
  the unmasked base rate — for the conditional use case, exactly logit P(EF2+|EF0+).
- **Equalization:** uses `T_ℓ = W_ℓ·H_ℓ` — see §4.
- **Validation loss / early stopping:** `compute_mean_logloss` skips masked entries and
  divides by per-label unmasked weight, then combines labels with λ. A masked entry must
  contribute exactly 0 loss, not `logloss(NaN, ŷ)` = NaN.
- **Prediction:** unaffected — the model always predicts all `L` labels everywhere.
- **MPI:** nothing new; masks are local data like `y`, and all cross-rank quantities
  (base rates, `W_ℓ`, histograms) already flow through Allreduce'd sums.
- **Edge cases:** a point masked on *all* labels still contributes weight to
  `min_data_weight_in_leaf` and consumes a bagging draw — harmless, but document that
  fully-don't-care points are better dropped by the caller. A *label* masked everywhere
  is an error (degenerate-label guard, §4).

## 6. Per-label early stopping

Different labels exhaust their signal at different iteration counts — EF0+ might keep
improving through 1000 trees while EF2+'s validation loss bottoms out at 800. Since
trees are shared, we can't stop *training* per label, but we can record each label's
optimum and have prediction ignore later trees' contributions for that label.

### Tracking

Replace the `make_callback_to_track_validation_loss` closure with a callable struct
(`ValidationLossTracker`) so its findings are inspectable after training. Per iteration
it already computes the per-label validation losses (masked-aware, §5) to form the
λ-weighted combined loss (§4); it additionally records:

- `label_best_losses :: Vector{Loss}`, `label_best_iterations :: Vector{Int}` — argmin
  of each label's *raw* per-label validation loss. (λ scales a label's loss by a
  constant, so it cannot move a per-label argmin — per-label bests are
  equalization-independent.)
- `global_best_loss :: Loss`, `global_best_iteration :: Int` — argmin of the λ-weighted
  combined loss. **This is the "what if we'd stopped at the global minimum" marker**,
  recorded for post-hoc comparison; it does not affect the stop decision.

### Stop rule

Stop when **no label** has improved for `max_iterations_without_improvement`
iterations, i.e. when `current_iteration - maximum(label_best_iterations) ≥
max_iterations_without_improvement`. On stop, resize `trees` to keep everything through
`maximum(label_best_iterations)` (mind the off-by-one for the initial constant tree) —
trees after the last label's best helped nobody. For `L = 1` this is exactly today's
behavior. Trees between the global best and the last per-label best are kept: they were
still improving *some* label. MPI: per-label losses come out of the same Allreduce'd
sums, so all ranks see identical values and make identical decisions — no new
communication.

### Persistence & prediction

- `save` gains optional early-stopping metadata (top-level BSON fields in the v2
  format): `label_best_iterations`, `global_best_iteration`, and the corresponding
  losses. `load` returns them when present.
- `predict` / `predict_on_binned` / `load_unbinned_predictor` gain
  `label_tree_counts = nothing`: when given, label ℓ's score sums Δscores only from
  trees `1:label_tree_counts[ℓ]`. Implementation: preprocess — for trees beyond a
  label's count, apply a copy of the leaf Δscores with that label zeroed (per-tree
  setup cost, hot apply loop unchanged and branch-free). The caller can pass
  `label_best_iterations` for per-label stopping, or `fill(global_best_iteration, L)`
  to try the global-minimum variant post hoc — same saved model serves both.
- Optional convenience: `truncate_per_label(trees, counts)` returning a new tree vector
  with the zeroing baked in (non-destructive), for saving a final pruned model.

## 7. Implementation phases

1. **Plumbing + L=1 no-regression.** Accept matrix `y`, internal transpose, thread
   `label_count` through; everything still hardcoded to the stride-4 path when `L = 1`.
   Add a regression test asserting bit-identical trees vs. current master on a fixed
   seed (`test/` has BugHunt/TestHistogramBuilding to crib harnesses from).
2. **Layout generalization.** Stride-`S` loss info + histograms, generalized kernels
   with `Val{S}` unrolling; scalar reference implementation first, SIMD second, verified
   against each other (extend `test/TestHistogramBuilding.jl`). Check `@code_native`
   that `L = 1` still emits the old code.
3. **Vector leaves + split eval + apply/predict.** Multi-label trees end-to-end,
   single machine.
4. **save/load v2 + MPI stride changes.** Round-trip test; MPI test via
   `test_mpi.sbatch` / a 2-rank local run.
5. **Label mask + `equalize_label_losses` + `label_weights` + per-label early
   stopping.** These land together because they all live in
   `compute_∇losses_∇∇losses!`, `compute_mean_probability`, and the validation-loss
   path: NaN mask handling, masked-aware `T_ℓ = W_ℓ·H_ℓ` equalization,
   degenerate-label guard, `ValidationLossTracker` with per-label bests + global
   marker (§6), `label_tree_counts` predict support and metadata in save/load.
6. **Benchmarks & docs.** `test/ProfileHRRR.jl`-style run with `L ∈ {1, 3, 6}` to
   measure the loss-info bandwidth cost; README section with a multi-label example
   (including the TC-wind mask and the masked-conditional recipe).

Each phase should leave `master`'s existing single-label tests green.

## 8. Testing

- **Equivalence:** `L = 1` matrix-`y` run produces identical trees & predictions to
  vector-`y` run and to pre-change master (fixed seeds; bagging & feature sampling are
  seeded off `n_prior_trees`, so determinism holds).
- **Consistency:** with `L` copies of the *same* label and no equalization, every leaf's
  `Δscores` entries should be equal, and predictions should match the `L = 1` model
  trained with the same per-tree feature sampling (same seed ⇒ same feature subsets ⇒
  identical splits, since summed gains are `L×` the single-label gains).
- **Multi-label sanity:** synthetic data, two labels driven by disjoint features, check
  both labels' AUC/logloss beat the constant model.
- **Equalization:** rare label (π ≈ 0.001) + common label (π ≈ 0.4); without the flag the
  rare label's validation loss barely improves over constant; with the flag it does, and
  the *initial* weighted per-label loss contributions are equal by construction (assert
  `λ_ℓ·H_ℓ` all equal). Degenerate-label error test.
- **Histogram kernels:** stride-S SIMD vs. scalar reference, random data, all of
  `L ∈ {1, 2, 3, 5}` (odd/even to exercise padding).
- **Nested-label / conditional-probability diagnostic:** train on synthetic nested
  labels (y₂ ⊆ y₁); check (i) the fraction of validation points where predicted
  P(y₂) > P(y₁) is small and the violations are small in magnitude, and (ii) the
  conditional P(y₂)/P(y₁) is well-calibrated against the true conditional in bins.
  This is a report-style test (thresholds loose), guarding the primary use case.
- **Label mask:** (i) masking label ℓ at a set of points produces the same trees as
  physically possible comparison — check label ℓ's histograms/gradients match a run
  where those points' contributions are removed by hand; (ii) masked entries contribute
  exactly 0 to training and validation loss (no NaN leakage — run with `--check-bounds`
  and assert loss is finite); (iii) conditional recipe end-to-end: nested synthetic
  labels, mask y₂ to y₁'s positives, check the y₂ head's predictions on held-out
  y₁-positive points calibrate against the true conditional; (iv) fully-masked label
  errors out; (v) mask + `equalize_label_losses` uses unmasked weight `W_ℓ` (assert
  initial `λ_ℓ·T_ℓ` equal across labels).
- **Per-label early stopping:** synthetic data where label 2's signal is pure noise
  after few iterations while label 1 keeps improving — assert label 2's
  `label_best_iteration` ≪ label 1's, training continues until label 1 stalls, trees
  are truncated to `maximum(label_best_iterations)` (+ constant tree), and predictions
  with `label_tree_counts = label_best_iterations` match predictions from a model
  trained/stopped at each label's best iteration. Check the global marker: predictions
  with `fill(global_best_iteration, L)` match a global-stop run. `L = 1`: tracker
  behaves identically to the current closure (same stop iteration, same tree count).
- **save/load:** v2 round trip; loading a v1 file from a fixture.
- **MPI:** 2-rank run matches 1-rank run (existing pattern in `ValidationServer.jl` /
  `test_mpi.sbatch`).

## 9. Risks & open questions

- **Kernel regression risk (biggest):** the stride-4 kernels are tuned for loss-info
  cache residency — loss info is revisited once per feature, and `is_chunk_size` is
  sized so a chunk of loss-info records stays resident while the work-chunk's ≤ 12
  features stream `X_binned` through it. Stride S grows each record to `4S` bytes
  (vs 16), so scaling `is_chunk_size` by `4/S` holds the resident working set constant
  and is the principled first cut, not just a guess; the constants (20736 / 320 in
  `compute_histograms!`) should still be re-searched (`hyperoptimize_chunk_size.rb`
  exists for this) since histogram footprint also grows `S/4`× and shifts the L2/L3
  split. If perf at `L = 1` regresses and can't be recovered, fall back to keeping the
  dedicated stride-4 path (dispatch on `Val(4)`).
- **Shared-structure quality:** if labels turn out to want very different trees, design
  (a) (round-robin per-label trees) could be layered on later as a config option —
  it composes with everything here except it doesn't need vector leaves.
- **Memory:** loss-info scratch grows `S/4`× (e.g. `L = 6` ⇒ stride 16 ⇒ 4× today: for
  10M datapoints, 640 MB vs 160 MB). Histograms grow the same factor but are small.
  Document in README; this is inherent to design (b).
- **Monotonicity of nested labels:** not enforced by default. Notes on what does and
  doesn't work:
  - The constant tree is monotone for free: c_ℓ = logit(π_ℓ) and π₂ ≤ π₁ ⇒ c₂ ≤ c₁.
    Forcing the constants *equal* across labels does NOT help — the violations come
    from later trees. (If scores are pointwise *equal*, the next tree's steps are
    provably ordered: equal ŷ ⇒ equal ∇∇ and Σ∇₂ − Σ∇₁ = Σ(y₁ − y₂) ≥ 0 per leaf.
    But merely *ordered* scores can produce a violating step: a leaf mixing a point
    where label 1 is confidently-wrong-high with a point where both labels are
    uncertain-positive yields Δ₁ ≪ 0 < Δ₂, flipping the second point.)
  - **Per-leaf enforcement is provably sufficient:** score₂(x) − score₁(x) =
    (c₂ − c₁) + Σ_t (Δ₂,t − Δ₁,t) along x's leaf path, so ordered constants + ordered
    Δscores in every leaf of every tree ⇒ pointwise P₂ ≤ P₁ for all inputs, all
    horizons. `max_delta_score` clamping is monotone and λ_ℓ scales ∇ and ∇∇ equally,
    so neither disturbs it.
  - **Decision (2026-07-18): do NOT enforce monotonicity in training.** Per-leaf
    projection is sufficient but over-constraining: it forbids a tree from raising
    EF2+ more than EF0+ in a leaf even when the data demands it (e.g. correcting a
    region where earlier trees pushed the gap too wide). The needed condition is on
    the *sum* of leaf contributions, not each leaf, and clamping each summand throws
    away legitimate corrective moves.
  - Instead: **handle the conditional P(EF2+|EF0+) in post-processing**, out of scope
    for this library — e.g. clip the ratio at 1, or calibrate the conditional directly
    on held-out data. Keep the monotonicity diagnostic test as a report (it tells us
    how much post-processing has to correct), but it gates nothing.
- **Open:** exact `FastNode` representation for vector leaves (tuple vs. side table) —
  pick after microbenchmarking predict; whether `save` writes v1 format for `L = 1`
  models.
