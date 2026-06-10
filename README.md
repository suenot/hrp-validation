# When Does Hierarchical Risk Parity Beat Markowitz?

A reproducible, controlled comparison of Hierarchical Risk Parity (Lopez de
Prado 2016) against the Markowitz family (sample min-variance, Ledoit-Wolf
shrinkage min-variance, inverse variance, 1/N) **under known covariance**: the
true `Sigma` is drawn from one of four structure families, every allocator sees
only a finite sample of `T` observations, and every weight vector is scored
*exactly* on the truth (`w' Sigma w` vs the oracle min-variance floor). This
isolates estimation error -- the entire argument for HRP -- from everything else.

Motivated by a [marketmaker.cc](https://marketmaker.cc) blog post that promotes
an HRP-based pipeline as superior to classical mean-variance without controlled
evidence; this package is the controlled evidence, and the verdict is mixed.

> **Headline findings** (4,000 Gaussian + 800 Student-t simulated markets,
> N in {10, 30, 50, 100}, T/N in {0.5, 1, 2, 5, 10}):
>
> 1. **HRP beats raw sample min-variance exactly where the blog says** -- the
>    data-starved regime. At T/N <= 1 HRP's realized risk is lower in ~87% of
>    experiments (T/N = 1 medians: 16.8x oracle for sample MV vs 2.3x for HRP);
>    at T/N >= 5 the ranking flips almost completely (HRP wins ~3%).
> 2. **But the relevant competitor is shrinkage, not raw MV.** Ledoit-Wolf
>    min-variance beats HRP at *every* T/N in the factor-structured families
>    (HRP win rate 0.34 at T/N <= 1, 0.05 at T/N >= 5). HRP only edges LW out
>    in the unstructured-correlation family at small samples (win rate ~0.55).
> 3. **HRP is not a consistent min-variance estimator**: its risk ratio
>    plateaus around 1.4-3.2x oracle as T grows while MV/LW converge to ~1.1x.
> 4. **HRP ~ inverse variance unless clusters are asymmetric.** With equally
>    tight clusters IV beats HRP; HRP's win rate vs IV rises from 0.35 to 0.56
>    as cross-block tightness dispersion grows (hierarchical family, T/N <= 2).
> 5. **Linkage choice barely matters** for realized risk (the three
>    single/average/ward medians agree to within 0.02-0.06 per family, and the
>    distance-matrix convention -- condensed d vs the distance-of-distances
>    d-tilde of the original code listing -- moves them by at most 0.02) --
>    quasi-diagonalization, not the dendrogram flavor, does the work.
> 6. **Sharpe with estimated means is the real disaster** (DeMiguel et al.
>    2009, reproduced): at T/N <= 1 Markowitz fed a noisy mu-hat captures a
>    median ~10% of the oracle tangency Sharpe (negative in 28% of runs), vs
>    ~43-46% for mu-free HRP / equal weight.
>
> Bottom line: "HRP sidesteps Markowitz's inversion disease" is true against
> *raw* sample min-variance at small T/N, but plain Ledoit-Wolf shrinkage --
> equally inversion-free in spirit, one line of sklearn -- is the stronger
> default; HRP's edge is confined to small samples with messy, non-factor
> correlation structure or strongly asymmetric clusters.

Limitations (stated, not hidden): i.i.d. Gaussian returns (one multivariate
Student-t robustness batch, t_df in [4, 10], same qualitative ranking), constant
truth over the window, no transaction costs, long-only enforced only via the
crude clip-and-renormalize heuristic (not a QP).

## Reproduce everything

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/run_all.py            # ~2 min -> results/results.json + records.csv
python -m hrp_experiments.figures    # -> paper/figures/*.pdf (4 vector PDFs)
python -m pytest -q                  # 17 sanity tests
```

Deterministic given the seeds in `scripts/run_all.py` (SeedSequence-spawned
child RNGs). `--quick` runs a small smoke batch (~4 s).

## Layout

```
hrp_experiments/
  model.py        # true-covariance generator: one-factor / hierarchical /
                  # unstructured (Wishart) / equicorrelation; all knobs sampled
                  # and recorded (SAMPLING_RANGES), none hidden
  allocators.py   # 1/N, inverse variance, closed-form min-var (+ LO clip),
                  # Ledoit-Wolf min-var, HRP per Lopez de Prado 2016 (corr
                  # distance -> single linkage -> quasi-diag -> recursive
                  # bisection) under both distance conventions (condensed d +
                  # the original listing's distance-of-distances d-tilde),
                  # linkage variants, noisy-mu tangency direction
  simulate.py     # one experiment: truth -> finite sample -> all allocators ->
                  # exact true-risk / true-Sharpe / HHI / turnover scoring
  analysis.py     # headline grid, win rates, linkage sensitivity,
                  # concentration/turnover, Sharpe track, t robustness --
                  # everything stratified by structure family
  figures.py      # the paper's 4 figures
scripts/run_all.py
tests/            # pytest sanity checks (17 tests)
results/          # results.json + records.csv (generated)
paper/figures/    # fig_setup, fig_risk_vs_tn, fig_winrate, fig_sensitivity
```

## License

Code: [MIT](LICENSE). Paper text and figures: CC BY 4.0.
