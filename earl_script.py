"""
Generate distance-dependent motion-related artifact plots
The rank for the intercept (smoothing curve at 35mm) indexes general dependence
on motion (i.e., a mix of global and focal effects), while the rank for the
slope (difference in smoothing curve at 100mm and 35mm) indexes distance
dependence (i.e., focal effects).
"""
import os
import os.path as op
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import nibabel as nib
import pandas as pd
from nilearn import datasets
from bids.layout import BIDSLayout

# Distance-dependent motion-related artifact analysis code
import ddmra

sns.set_style('white')


def main(file_list, qc_list, out_dir='.', n_iters=10000, qc_thresh=0.2,
         earl=False, regress=False):
    """
    Run motion analyses and generate plots for input data.
    """
    if not op.isdir(out_dir):
        os.mkdir(out_dir)

    out_dir = op.join(out_dir, 'results_earl-{0}_regress-{1}'.format(str(earl), str(regress)))
    if not op.isdir(out_dir):
        os.mkdir(out_dir)
    res_file = op.join(out_dir, 'results_summary.txt')
    v1, v2 = 35, 100  # distances to evaluate
    n_lines = min((n_iters, 50))

    # Run analyses
    ddmra.run(file_list, qc_list, out_dir=out_dir, n_iters=n_iters,
              qc_thresh=qc_thresh, earl=earl, regress=regress)
    smc_sorted_dists = np.loadtxt(op.join(out_dir, 'smc_sorted_distances.txt'))
    all_sorted_dists = np.loadtxt(op.join(out_dir, 'all_sorted_distances.txt'))

    # QC:RSFC analysis
    # Assess significance
    qcrsfc_rs = np.loadtxt(op.join(
        out_dir, 'qcrsfc_analysis_values.txt'))
    qcrsfc_smc = np.loadtxt(op.join(
        out_dir, 'qcrsfc_analysis_smoothing_curve.txt'))
    perm_qcrsfc_smc = np.loadtxt(op.join(
        out_dir, 'qcrsfc_analysis_null_smoothing_curves.txt'))
    intercept = ddmra.get_val(smc_sorted_dists, qcrsfc_smc, v1)
    slope = (ddmra.get_val(smc_sorted_dists, qcrsfc_smc, v1) -
             ddmra.get_val(smc_sorted_dists, qcrsfc_smc, v2))
    perm_intercepts = ddmra.get_val(smc_sorted_dists, perm_qcrsfc_smc, v1)
    perm_slopes = (ddmra.get_val(smc_sorted_dists, perm_qcrsfc_smc, v1) -
                   ddmra.get_val(smc_sorted_dists, perm_qcrsfc_smc, v2))

    p_inter = ddmra.rank_p(intercept, perm_intercepts, tail='upper')
    p_slope = ddmra.rank_p(slope, perm_slopes, tail='upper')
    with open(res_file, 'w') as fo:
        fo.write('QCRSFC analysis results:\n')
        fo.write('\tIntercept = {0:.04f}, p = {1:.04f}\n'.format(intercept,
                                                                 p_inter))
        fo.write('\tSlope = {0:.04f}, p = {1:.04f}\n'.format(-1*slope,
                                                             p_slope))

    # Generate plot
    fig, ax = plt.subplots(figsize=(10, 14))
    sns.regplot(all_sorted_dists, qcrsfc_rs,
                ax=ax, scatter=True, fit_reg=False,
                scatter_kws={'color': 'red', 's': 2., 'alpha': 1})
    ax.axhline(0, xmin=0, xmax=200, color='black', linewidth=3)
    for i_line in range(n_lines):
        ax.plot(smc_sorted_dists, perm_qcrsfc_smc[i_line, :], color='black')
    ax.plot(smc_sorted_dists, qcrsfc_smc, color='white')
    ax.set_xlabel('Distance (mm)', fontsize=32)
    ax.set_ylabel('QC:RSFC r\n(QC = mean FD)', fontsize=32, labelpad=-30)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([-0.5, 0.5])
    ax.set_yticklabels([-0.5, 0.5], fontsize=32)
    ax.set_xticks([0, 50, 100, 150])
    ax.set_xticklabels([])
    ax.set_xlim(0, 160)
    ax.annotate('35 mm: {0:.04f}\n35-100 mm: {1:.04f}'.format(p_inter,
                                                              p_slope),
                xy=(1, 0), xycoords='axes fraction',
                xytext=(-20, 20), textcoords='offset pixels',
                horizontalalignment='right',
                verticalalignment='bottom', fontsize=32)
    fig.tight_layout()
    fig.savefig(op.join(out_dir, 'qcrsfc_analysis.png'), dpi=400)
    del qcrsfc_rs, qcrsfc_smc, perm_qcrsfc_smc

    # High-low motion analysis
    # Assess significance
    hl_corr_diff = np.loadtxt(op.join(
        out_dir, 'highlow_analysis_values.txt'))
    hl_smc = np.loadtxt(op.join(
        out_dir, 'highlow_analysis_smoothing_curve.txt'))
    perm_hl_smc = np.loadtxt(op.join(
        out_dir, 'highlow_analysis_null_smoothing_curves.txt'))
    intercept = ddmra.get_val(smc_sorted_dists, hl_smc, v1)
    slope = (ddmra.get_val(smc_sorted_dists, hl_smc, v1) -
             ddmra.get_val(smc_sorted_dists, hl_smc, v2))
    perm_intercepts = ddmra.get_val(smc_sorted_dists, perm_hl_smc, v1)
    perm_slopes = (ddmra.get_val(smc_sorted_dists, perm_hl_smc, v1) -
                   ddmra.get_val(smc_sorted_dists, perm_hl_smc, v2))

    p_inter = ddmra.rank_p(intercept, perm_intercepts, tail='upper')
    p_slope = ddmra.rank_p(slope, perm_slopes, tail='upper')
    with open(res_file, 'a') as fo:
        fo.write('High-low motion analysis results:\n')
        fo.write('\tIntercept = {0:.04f}, p = {1:.04f}\n'.format(intercept,
                                                                 p_inter))
        fo.write('\tSlope = {0:.04f}, p = {1:.04f}\n'.format(-1*slope,
                                                             p_slope))

    # Generate plot
    fig, ax = plt.subplots(figsize=(10, 14))
    sns.regplot(all_sorted_dists, hl_corr_diff,
                ax=ax, scatter=True, fit_reg=False,
                scatter_kws={'color': 'red', 's': 2, 'alpha': 1})
    ax.axhline(0, xmin=0, xmax=200, color='black', linewidth=3)
    for i in range(n_lines):
        ax.plot(smc_sorted_dists, perm_hl_smc[i, :], color='black')
    ax.plot(smc_sorted_dists, hl_smc, color='white')
    ax.set_xlabel('Distance (mm)', fontsize=32)
    ax.set_ylabel(r'High-low motion $\Delta$r', fontsize=32)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([-0.5, 0.5])
    ax.set_yticklabels([-0.5, 0.5], fontsize=32)
    ax.set_xticks([0, 50, 100, 150])
    ax.set_xticklabels([])
    ax.set_xlim(0, 160)
    ax.annotate('35 mm: {0:.04f}\n35-100 mm: {1:.04f}'.format(p_inter,
                                                              p_slope),
                xy=(1, 0), xycoords='axes fraction',
                xytext=(-20, 20), textcoords='offset pixels',
                horizontalalignment='right',
                verticalalignment='bottom', fontsize=32)
    fig.tight_layout()
    fig.savefig(op.join(out_dir, 'hilow_analysis.png'), dpi=400)
    del hl_corr_diff, hl_smc, perm_hl_smc

    # Scrubbing analysis
    mean_delta_r = np.loadtxt(op.join(
        out_dir, 'scrubbing_analysis_values.txt'))
    scrub_smc = np.loadtxt(op.join(
        out_dir, 'scrubbing_analysis_smoothing_curve.txt'))
    perm_scrub_smc = np.loadtxt(op.join(
        out_dir, 'scrubbing_analysis_null_smoothing_curves.txt'))

    # Assess significance
    intercept = ddmra.get_val(smc_sorted_dists, scrub_smc, v1)
    slope = (ddmra.get_val(smc_sorted_dists, scrub_smc, v1) -
             ddmra.get_val(smc_sorted_dists, scrub_smc, v2))

    perm_intercepts = ddmra.get_val(smc_sorted_dists, perm_scrub_smc, v1)
    perm_slopes = (ddmra.get_val(smc_sorted_dists, perm_scrub_smc, v1) -
                   ddmra.get_val(smc_sorted_dists, perm_scrub_smc, v2))
    p_inter = ddmra.rank_p(intercept, perm_intercepts, tail='upper')
    p_slope = ddmra.rank_p(slope, perm_slopes, tail='upper')
    with open(res_file, 'a') as fo:
        fo.write('Scrubbing analysis results:\n')
        fo.write('\tIntercept = {0:.04f}, p = {1:.04f}\n'.format(intercept,
                                                                 p_inter))
        fo.write('\tSlope = {0:.04f}, p = {1:.04f}\n'.format(-1*slope,
                                                             p_slope))

    # Generate scrubbing analysis plot
    fig, ax = plt.subplots(figsize=(10, 14))
    sns.regplot(all_sorted_dists, mean_delta_r,
                ax=ax, scatter=True, fit_reg=False,
                scatter_kws={'color': 'red', 's': 2, 'alpha': 1})
    ax.axhline(0, xmin=0, xmax=200, color='black', linewidth=3)
    for i in range(n_lines):
        ax.plot(smc_sorted_dists, perm_scrub_smc[i, :],
                color='black')
    ax.plot(smc_sorted_dists, scrub_smc,
            color='white')
    ax.set_xlabel('Distance (mm)', fontsize=32)
    ax.set_ylabel(r'Scrubbing $\Delta$r', fontsize=32)
    ax.set_ylim(-0.05, 0.05)
    ax.set_yticks([-0.05, 0.05])
    ax.set_yticklabels([-0.05, 0.05], fontsize=32)
    ax.set_xticks([0, 50, 100, 150])
    ax.set_xticklabels([])
    ax.set_xlim(0, 160)
    ax.annotate('35 mm: {0:.04f}\n35-100 mm: {1:.04f}'.format(p_inter,
                                                              p_slope),
                xy=(1, 0), xycoords='axes fraction',
                xytext=(-20, 20), textcoords='offset pixels',
                horizontalalignment='right',
                verticalalignment='bottom', fontsize=32)
    fig.tight_layout()
    fig.savefig(op.join(out_dir, 'scrubbing_analysis.png'), dpi=400)
    del mean_delta_r, scrub_smc, perm_scrub_smc


if __name__ == '__main__':
    """
    Run a test using 40 subjects from the ADHD dataset.
    """
    dset_dir = '/home/data/nbc/Sutherland_ACE/dset/'
    deriv_dir = '/home/data/nbc/Sutherland_ACE/derivatives/fmriprep-1.2.1/'
    fd_thresh = 0.2
    n_iters = 10000

    layout = BIDSLayout(dset_dir)
    subjects = layout.get_subjects(task='rest')
    motpars = []
    files = []
    for sub in subjects:
        preproc_file = op.join(
            deriv_dir,
            'sub-{0}/ses-S1/func/sub-{0}_ses-S1_task-rest_run-01_space-'
            'MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'.format(sub))
        files.append(preproc_file)
        conf_file = op.join(
            deriv_dir,
            'sub-{0}/ses-S1/func/sub-{0}_ses-S1_task-rest_run-01_desc-'
            'confounds_regressors.tsv'.format(sub))
        conf = pd.read_csv(conf_file, sep='\t')
        motpars_ = conf[['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']].values
        motpars.append(motpars_)

    for earl in [True, False]:
        for regress in [True, False]:
            main(files, motpars, n_iters=n_iters, qc_thresh=fd_thresh,
                 earl=earl, regress=regress)
