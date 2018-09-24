"""
Generate distance-dependent motion-related artifact plots
The rank for the intercept (smoothing curve at 35mm) indexes general dependence
on motion (i.e., a mix of global and focal effects), while the rank for the
slope (difference in smoothing curve at 100mm and 35mm) indexes distance
dependence (i.e., focal effects).
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import nibabel as nib
import pandas as pd
from nilearn import datasets

# Distance-dependent motion-related artifact analysis code
import ddmra

sns.set_style('white')


def get_fd(motion):
    # assuming rotations in degrees
    motion[:, :3] = motion[:, :3] * 50 * (np.pi/180.)
    motion = np.vstack((np.array([[0, 0, 0, 0, 0, 0]]),
                        np.diff(motion, axis=0)))
    fd = np.sum(np.abs(motion), axis=1)
    return fd


def main():
    # Constants
    n_subjects = 40 # 31
    fd_thresh = 0.2
    window = 1000
    v1, v2 = 35, 100  # distances to evaluate
    data = datasets.fetch_adhd(n_subjects=n_subjects)
    n_iters = 10000
    n_lines = min((n_iters, 50))
    res_file = 'results.txt'

    # Prepare data
    imgs = []
    fd_all = []
    for i in range(n_subjects):
        func = data.func[i]
        imgs.append(nib.load(func))
        conf = data.confounds[i]
        df = pd.read_csv(conf, sep='\t')
        motion = df[['motion-pitch', 'motion-roll', 'motion-yaw',
                     'motion-x', 'motion-y', 'motion-z']].values
        fd_all.append(get_fd(motion))

    # Run analyses
    results = ddmra.run(imgs, fd_all, n_iters=n_iters, qc_thresh=fd_thresh)

    # This fails due to object size
    # with open('results.pkl', 'wb') as fo:
    #     pickle.dump(results, fo)

    # QC:RSFC analysis
    # Assess significance
    intercept = ddmra.get_val(results['sorted_dists'], results['qcrsfc_smc'], v1)
    slope = (ddmra.get_val(results['sorted_dists'], results['qcrsfc_smc'], v1) -
             ddmra.get_val(results['sorted_dists'], results['qcrsfc_smc'], v2))
    perm_intercepts = ddmra.get_val(results['sorted_dists'], results['qcrsfc_null'], v1)
    perm_slopes = (ddmra.get_val(results['sorted_dists'], results['qcrsfc_null'], v1) -
                   ddmra.get_val(results['sorted_dists'], results['qcrsfc_null'], v2))

    p_inter = ddmra.rank_p(intercept, perm_intercepts, tail='upper')
    p_slope = ddmra.rank_p(slope, perm_slopes, tail='upper')
    with open(res_file, 'w') as fo:
        fo.write('QCRSFC analysis results:\n')
        fo.write('\tIntercept = {0:.04f}, p = {1:.04f}\n'.format(intercept, p_inter))
        fo.write('\tSlope = {0:.04f}, p = {1:.04f}\n'.format(-1*slope, p_slope))

    # Generate plot
    fig, ax = plt.subplots(figsize=(10, 14))
    sns.regplot(results['sorted_dists'], results['qcrsfc_y'],
                ax=ax, scatter=True, fit_reg=False,
                scatter_kws={'color': 'red', 's': 2., 'alpha': 1})
    ax.axhline(0, xmin=0, xmax=200, color='black', linewidth=3)
    for i in range(n_lines):
        ax.plot(results['sorted_dists'], results['qcrsfc_null'][i, :], color='black')
    ax.plot(results['sorted_dists'], results['qcrsfc_smc'], color='white')
    ax.set_xlabel('Distance (mm)', fontsize=32)
    ax.set_ylabel('QC:RSFC r\n(QC = mean FD)', fontsize=32, labelpad=-30)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([-0.5, 0.5])
    ax.set_yticklabels([-0.5, 0.5], fontsize=32)
    ax.set_xticks([0, 50, 100, 150])
    ax.set_xticklabels([])
    ax.set_xlim(0, 160)
    ax.annotate('35 mm: {0:.04f}\n35-100 mm: {1:.04f}'.format(p_inter, p_slope),
                xy=(1, 0), xycoords='axes fraction',
                xytext=(-20, 20), textcoords='offset pixels',
                horizontalalignment='right',
                verticalalignment='bottom', fontsize=32)
    fig.tight_layout()
    fig.savefig('sandbox/qcrsfc_analysis.png', dpi=400)

    # High-low motion analysis
    # Assess significance
    intercept = ddmra.get_val(results['sorted_dists'], results['hl_smc'], v1)
    slope = (ddmra.get_val(results['sorted_dists'], results['hl_smc'], v1) -
             ddmra.get_val(results['sorted_dists'], results['hl_smc'], v2))
    perm_intercepts = ddmra.get_val(results['sorted_dists'], results['hl_null'], v1)
    perm_slopes = (ddmra.get_val(results['sorted_dists'], results['hl_null'], v1) -
                   ddmra.get_val(results['sorted_dists'], results['hl_null'], v2))

    p_inter = ddmra.rank_p(intercept, perm_intercepts, tail='upper')
    p_slope = ddmra.rank_p(slope, perm_slopes, tail='upper')
    with open(res_file, 'a') as fo:
        fo.write('High-low motion analysis results:\n')
        fo.write('\tIntercept = {0:.04f}, p = {1:.04f}\n'.format(intercept, p_inter))
        fo.write('\tSlope = {0:.04f}, p = {1:.04f}\n'.format(-1*slope, p_slope))

    # Generate plot
    fig, ax = plt.subplots(figsize=(10, 14))
    sns.regplot(results['sorted_dists'], results['hl_y'],
                ax=ax, scatter=True, fit_reg=False,
                scatter_kws={'color': 'red', 's': 2, 'alpha': 1})
    ax.axhline(0, xmin=0, xmax=200, color='black', linewidth=3)
    for i in range(n_lines):
        ax.plot(results['sorted_dists'], results['hl_null'][i, :], color='black')
    ax.plot(results['sorted_dists'], results['hl_smc'], color='white')
    ax.set_xlabel('Distance (mm)', fontsize=32)
    ax.set_ylabel(r'High-low motion $\Delta$r', fontsize=32)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([-0.5, 0.5])
    ax.set_yticklabels([-0.5, 0.5], fontsize=32)
    ax.set_xticks([0, 50, 100, 150])
    ax.set_xticklabels([])
    ax.set_xlim(0, 160)
    ax.annotate('35 mm: {0:.04f}\n35-100 mm: {1:.04f}'.format(p_inter, p_slope),
                xy=(1, 0), xycoords='axes fraction',
                xytext=(-20, 20), textcoords='offset pixels',
                horizontalalignment='right',
                verticalalignment='bottom', fontsize=32)
    fig.tight_layout()
    fig.savefig('sandbox/hilow_analysis.png', dpi=400)

    # Scrubbing analysis
    # Assess significance
    intercept = ddmra.get_val(results['sorted_dists'], results['scrub_smc'], v1)
    slope = (ddmra.get_val(results['sorted_dists'], results['scrub_smc'], v1) -
             ddmra.get_val(results['sorted_dists'], results['scrub_smc'], v2))

    perm_intercepts = ddmra.get_val(results['sorted_dists'], results['scrub_null'], v1)
    perm_slopes = (ddmra.get_val(results['sorted_dists'], results['scrub_null'], v1) -
                   ddmra.get_val(results['sorted_dists'], results['scrub_null'], v2))
    p_inter = ddmra.rank_p(intercept, perm_intercepts, tail='upper')
    p_slope = ddmra.rank_p(slope, perm_slopes, tail='upper')
    with open(res_file, 'a') as fo:
        fo.write('Scrubbing analysis results:\n')
        fo.write('\tIntercept = {0:.04f}, p = {1:.04f}\n'.format(intercept, p_inter))
        fo.write('\tSlope = {0:.04f}, p = {1:.04f}\n'.format(-1*slope, p_slope))

    # Generate plot
    fig, ax = plt.subplots(figsize=(10, 14))
    sns.regplot(results['sorted_dists'], results['scrub_y'],
                ax=ax, scatter=True, fit_reg=False,
                scatter_kws={'color': 'red', 's': 2, 'alpha': 1})
    ax.axhline(0, xmin=0, xmax=200, color='black', linewidth=3)
    for i in range(n_lines):
        ax.plot(results['sorted_dists'], results['scrub_null'][i, :],
                color='black')
    ax.plot(results['sorted_dists'], results['scrub_smc'],
            color='white')
    ax.set_xlabel('Distance (mm)', fontsize=32)
    ax.set_ylabel(r'Scrubbing $\Delta$r', fontsize=32)
    ax.set_ylim(-0.05, 0.05)
    ax.set_yticks([-0.05, 0.05])
    ax.set_yticklabels([-0.05, 0.05], fontsize=32)
    ax.set_xticks([0, 50, 100, 150])
    ax.set_xticklabels([])
    ax.set_xlim(0, 160)
    ax.annotate('35 mm: {0:.04f}\n35-100 mm: {1:.04f}'.format(p_inter, p_slope),
                xy=(1, 0), xycoords='axes fraction',
                xytext=(-20, 20), textcoords='offset pixels',
                horizontalalignment='right',
                verticalalignment='bottom', fontsize=32)
    fig.tight_layout()
    fig.savefig('sandbox/scrubbing_analysis.png', dpi=400)


if __name__ == '__main__':
    main()
