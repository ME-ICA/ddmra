.. _methods_ref:

===========================
Methods and interpretation
===========================

``ddmra`` evaluates residual distance-dependent motion-related artifact in
resting-state fMRI connectivity estimates. The package is designed for
benchmarking denoising or preprocessing strategies, not for estimating a
neuroscientific connectivity effect. A good denoising strategy should reduce
associations between a run-level quality-control metric and functional
connectivity while preserving enough temporal degrees of freedom and data for
the downstream scientific analysis.

The workflow follows the broad evaluation logic used in resting-state fMRI
denoising benchmarks, where residual motion artifact is assessed with QC-FC
associations, distance dependence of those associations, scrubbing-related
connectivity changes, and accounting for data loss or temporal degrees of
freedom. See, for example, Power et al. (2012), Power et al. (2018), Ciric et
al. (2017), and Parkes et al. (2018).

The DDMRA analyses implemented here are based primarily on `Power et al.
(2018), PNAS <https://doi.org/10.1073/pnas.1720985115>`_. ``ddmra`` is not a
line-for-line reproduction of the original analysis scripts. It keeps the core
scientific target of the original method, namely evaluating whether
motion-related effects on functional connectivity vary with the physical
distance between regions, but makes several implementation choices intended to
make the analyses reusable for modern denoising comparisons.

References:

- `Power et al. (2012), NeuroImage <https://doi.org/10.1016/j.neuroimage.2011.10.018>`_
- `Power et al. (2018), PNAS <https://doi.org/10.1073/pnas.1720985115>`_
- `Ciric et al. (2017), NeuroImage <https://doi.org/10.1016/j.neuroimage.2017.03.020>`_
- `Parkes et al. (2018), NeuroImage <https://doi.org/10.1016/j.neuroimage.2017.12.073>`_


Differences from Power et al. (2018)
====================================

``ddmra`` differs from the original Power et al. DDMRA implementation in the
following ways.

Generalized inputs and atlases
    Power et al. evaluated a specific analysis setting. ``ddmra`` accepts
    arbitrary 4D NIfTI runs and either a local labels-image atlas or selected
    Nilearn coordinate atlases. The rationale is to support controlled
    comparisons among preprocessing or denoising pipelines, as long as all
    inputs are in the same space and resolution.

Scrubbing correlations are Fisher-z-transformed
    In the scrubbing analysis, ``ddmra`` Fisher-z-transforms the full and
    scrubbed correlation coefficients before subtracting and averaging them.
    The original Power implementation did not apply this transform in the same
    way. The rationale is to keep all implemented connectivity summaries on a
    Fisher-z scale and to avoid averaging raw correlations directly.

Scrubbing uses the opposite sign convention
    The original scrubbing contrast and the ``ddmra`` contrast have opposite
    signs. ``ddmra`` computes ``full time series - scrubbed time series``. The
    rationale is interpretive consistency: under this convention, larger
    positive local-distance effects point in the same artifact-like direction
    across QC-FC, high-low QC, and scrubbing summaries.

Inference is performed on smoothing-curve summaries
    ``ddmra`` treats edgewise ranks as diagnostics only. Inferential p-values
    are computed for the smoothing-curve intercept and slope against
    permutation null smoothing curves, using a plus-one finite-permutation
    correction. The rationale is to make inference target the reported
    distance-dependence summaries rather than treating edgewise ranks as
    p-values.

QC-FC can adjust for run-level covariates
    ``ddmra`` can residualize mean QC and edgewise connectivity with respect to
    run-level covariates before estimating QC-FC. The rationale is that
    denoising evaluations can be biased when motion is associated with site,
    age, diagnosis, acquisition, or other run-level variables.

Data loss and temporal degrees of freedom are explicit outputs
    ``ddmra`` writes ``run_denoising_summary.tsv`` with retained-run flags,
    volume counts, confound counts, nominal temporal degrees of freedom after
    confounds, and optional user-supplied denoising metrics. The rationale is
    that lower residual motion artifact is not automatically better if it is
    achieved by excessive volume loss or loss of temporal degrees of freedom.

Pipeline comparisons use paired label-swap tests
    The pipeline-comparison workflow adds direct pairwise tests that were not
    part of the original single-pipeline DDMRA method. These tests randomly
    swap pipeline labels within run and compare the pipeline difference in
    intercept and slope. The rationale is that pipelines are applied to the
    same runs, so direct paired inference is more appropriate than comparing
    independent per-pipeline p-values.


Single-pipeline workflow
========================

The :func:`ddmra.workflows.run_analyses` workflow takes a list of 4D NIfTI
files and a matching list of one-dimensional QC arrays, one array per run. The
QC array is usually framewise displacement, but may be another time-resolved
quality measure if it is defined for every volume in the run. All images should
be in the same space and resolution and should be compatible with the selected
atlas.

The workflow performs the following steps:

1. Validate that each QC array is one-dimensional, non-empty, and finite.
2. Extract ROI time series with either a labels-image atlas or a coordinate
   sphere atlas.
3. Compute ROI-to-ROI distances and sort edges from short to long distance.
4. Build z-transformed functional connectivity matrices for analyses that use
   run-level connectivity.
5. Drop runs with NaN ROI time series or zero-variance ROI time series.
6. Optionally identify multivariate connectivity outliers using PCA scores and
   a robust covariance estimator.
7. Compute the selected artifact analyses.
8. Smooth edgewise analysis values over distance and assess the smoothing
   curve intercept and slope against permutation null distributions.

The workflow writes ``run_denoising_summary.tsv`` for every run. This file is
important for interpretation because apparent denoising gains can be coupled to
data loss, temporal degrees of freedom, or complete removal of difficult runs.
Direct scientific comparisons of denoising strategies should consider these
columns alongside the artifact metrics.


QC-FC analysis
--------------

QC-FC measures the association between mean run QC and each functional
connectivity edge across runs. For each retained run, ``ddmra`` averages the
QC time series to one run-level value and computes Fisher-z-transformed ROI
correlations. For each edge, it then correlates run-level QC with the edge's
connectivity values across runs and Fisher-z-transforms that QC-FC
correlation.

If ``run_covariates`` are supplied, QC-FC is computed after residualizing both
the run-level QC vector and edgewise connectivity values with respect to those
covariates. This is useful when age, site, group, acquisition, or other
run-level factors could otherwise confound the QC-FC estimate.

Interpretation:

- Values near zero indicate little linear association between run quality and
  that connectivity edge.
- A distance-dependent curve with stronger short-distance effects than
  long-distance effects is consistent with residual motion-related artifact.
- QC-FC is not a measure of neural signal preservation. It should be interpreted
  with data-loss and reliability or validity benchmarks when available.


High-low QC analysis
--------------------

The high-low analysis splits retained runs by median mean QC. For each edge, it
subtracts the mean connectivity of the low-QC group from the mean connectivity
of the high-QC group.

Interpretation:

- Values reflect the edgewise difference between higher-motion and lower-motion
  runs.
- The analysis is intentionally simple and is useful as a complementary
  artifact benchmark.
- Because it depends on a median split, it should not be treated as a substitute
  for covariate-adjusted QC-FC when continuous QC information and covariates are
  important.


Scrubbing analysis
------------------

The scrubbing analysis compares connectivity before and after removing volumes
whose QC values exceed ``qc_thresh``. Within each run, the workflow computes
connectivity from the full time series and from the scrubbed time series, then
averages edgewise differences across retained runs. Runs are included in the
scrubbing analysis only when at least one volume is scrubbed and at least half
of the volumes remain.

``ddmra`` uses the sign convention ``full time series - scrubbed time series``.
This convention differs from the original Power et al. implementation, but it
keeps the direction of larger positive DDMRA effects similar across the
implemented analyses.

Interpretation:

- Larger effects indicate connectivity changes associated with removing
  high-QC volumes.
- Scrubbing results are conditional on the selected QC threshold and the
  availability of runs with both retained and scrubbed volumes.
- A method that reduces scrubbing-related effects by discarding many volumes
  should be evaluated together with temporal degrees of freedom and retained
  volume counts.


Distance smoothing, intercepts, and slopes
==========================================

All three analyses produce edgewise values ordered by ROI-to-ROI distance.
``ddmra`` smooths these values with a moving average over distance-sorted edges
and then averages values at identical distances. The smoothed curve is used for
summary inference.

Two scalar summaries are tested:

- ``intercept_35mm``: the smoothed curve value at 35 mm.
- ``slope_35_to_100mm``: the value at 35 mm minus the value at 100 mm.

The intercept is sensitive to the overall magnitude of local residual artifact.
The slope is sensitive to distance dependence, with larger positive values
indicating stronger local than long-distance effects under the package's sign
conventions.

The workflow tests these summaries against permutation null curves with the
plus-one finite-permutation correction, so the smallest possible p-value is
``1 / (n_iters + 1)``. The per-pipeline p-values answer whether a pipeline's
artifact summary is larger than expected under that pipeline's null model. They
do not directly answer whether one pipeline is better than another.


Output files from ``run_analyses``
==================================

``analysis_values.tsv.gz``
    Edgewise unsmoothed values for the selected analyses.

``smoothing_curves.tsv.gz``
    Distance-smoothed values used for intercept and slope summaries.

``null_smoothing_curves.npz``
    Permutation null smoothing curves for each selected analysis.

``ranks.tsv.gz``
    Diagnostic edgewise ranks of observed values against edgewise null values.
    These ranks are not p-values and should not be interpreted as inferential
    evidence.

``run_denoising_summary.tsv``
    Run-level accounting for input volumes, QC thresholding, confound counts,
    nominal temporal degrees of freedom after confounds, retention after data
    loading, retention for analysis, and optional user-provided denoising or
    data-loss metrics.

``log.tsv``
    Workflow messages, including retention counts and per-analysis intercept
    and slope p-values.

``analysis_results.png``
    Summary figure showing the available analysis curves.


Pipeline-comparison workflow
============================

The :func:`ddmra.workflows.run_pipeline_comparison` workflow directly supports
comparisons among processing pipelines. It accepts a TSV file or
:class:`pandas.DataFrame` with one row per run and one column per pipeline. Each
cell must contain a path to a 4D NIfTI file for that run and pipeline.

Example TSV:

.. code-block:: text

   preprocessed	XCP-D	tedana
   sub-01_preproc_bold.nii.gz	sub-01_xcpd_bold.nii.gz	sub-01_tedana_bold.nii.gz
   sub-02_preproc_bold.nii.gz	sub-02_xcpd_bold.nii.gz	sub-02_tedana_bold.nii.gz

Relative paths in a TSV are resolved relative to the TSV file. All selected
pipeline columns must have the same number of rows, and each row is assumed to
represent the same run across pipelines. The current implementation supports
NIfTI files only.

The workflow has two layers:

1. It runs :func:`ddmra.workflows.run_analyses` separately for each selected
   pipeline and writes each pipeline's outputs to a subdirectory.
2. By default, it performs direct pairwise statistical comparisons between
   pipelines.


Direct paired comparisons
-------------------------

Direct comparisons are performed for every selected pair of pipelines. For a
given pair, ``ddmra`` uses the intersection of runs retained for analysis by
both pipelines. It then recomputes the selected analysis curves on this paired
run set and compares the pipelines' smoothing-curve summaries.

For each analysis and pipeline pair, the observed difference is:

.. code-block:: text

   pipeline_1 summary - pipeline_2 summary

The null distribution is generated with paired run-wise pipeline-label swaps.
For each permutation, the two pipeline labels are randomly swapped or not
swapped within each run, and the pipeline difference is recomputed. This tests
the null hypothesis that the two pipeline outputs are exchangeable within run.
The procedure preserves:

- the run-level QC time series,
- run identity and pairing,
- the selected atlas and distance structure,
- the run set used for the pairwise comparison, and
- run-level covariates used in QC-FC adjustment.

This paired label-swap test is preferable to comparing two independent
per-pipeline p-values, because the pipelines are applied to the same runs and
their estimates are not independent.

Direct comparison p-values are two-sided and use the same plus-one
finite-permutation correction as the single-pipeline workflow. Increasing
``comparison_n_iters`` improves p-value resolution.


Interpreting pipeline-comparison results
----------------------------------------

``pipeline_pairwise_comparisons.tsv`` contains one row per pipeline pair,
analysis, and scalar contrast. Important columns include:

``pipeline_1`` and ``pipeline_2``
    The ordered pair being compared.

``analysis``
    One of ``qcrsfc``, ``highlow``, or ``scrubbing``.

``contrast``
    Either ``intercept_35mm`` or ``slope_35_to_100mm``.

``pipeline_1_value`` and ``pipeline_2_value``
    The paired-run summary values for the two pipelines.

``difference``
    ``pipeline_1_value - pipeline_2_value``.

``p_value``
    Two-sided paired label-swap p-value for the difference.

``n_paired_runs``
    Number of runs retained by both pipelines and used in the direct
    comparison.

For DDMRA artifact summaries, a lower positive intercept or slope is often
consistent with less residual distance-dependent artifact. However, users
should inspect the sign and shape of the full smoothing curves. If values cross
zero or if one pipeline changes the curve shape in a nonuniform way, the scalar
intercept and slope should be treated as summaries rather than complete
descriptions of performance.

The direct comparison outputs are:

``pipeline_comparison_summary.tsv``
    One row per pipeline with the pipeline output directory.

``pipeline_pairwise_comparisons.tsv``
    Pairwise intercept and slope differences with p-values.

``pipeline_pairwise_smoothing_curves.tsv.gz``
    Observed paired smoothing curves and distance-wise differences.

``pipeline_pairwise_nulls.npz``
    Null distributions for each pair, analysis, and scalar contrast.


Practical guidance
==================

- Use the same atlas, space, resolution, QC metric, and run order for all
  pipelines in a comparison.
- Prefer direct paired comparison p-values over informal comparisons of
  separate per-pipeline p-values.
- Report ``n_paired_runs`` and inspect ``run_denoising_summary.tsv`` for each
  pipeline.
- Treat data-loss and temporal degrees-of-freedom differences as part of the
  denoising result, not as incidental bookkeeping.
- Use a sufficiently large sample. QC-FC and high-low estimates are unstable
  in small samples, so ``ddmra`` warns when fewer than 30 runs are retained for
  these analyses and refuses to run with fewer than 10 (Parkes et al., 2018;
  Ciric et al., 2017).
- Use enough permutations for the inferential claim. With 10000 permutations,
  the minimum p-value is approximately 0.0001.
- Correct for multiple comparisons when making claims across many pipeline
  pairs, analyses, or contrasts.
- Do not interpret lower QC-FC or DDMRA values alone as proof of better neural
  signal preservation. Pair these metrics with reliability, identifiability,
  known network structure, or task/behavioral validity checks when those
  questions matter.
