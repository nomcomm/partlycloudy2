import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.spatial.distance import squareform
from scipy.stats import pearsonr, zscore
from scipy.fftpack import fft, ifft
import itertools as it
import sys
import logging

logger = logging.getLogger(__name__)

MAX_RANDOM_SEED = 2**32 - 1

# there functions are taken from the Princeton BrainIAK toolbox
# they are implemented separately here because the toolbox is still in progress, and some fundamental parts kept changing (e.g. the order of TR * voxel/region * subject for the ISC-input array). To avoid errors, I thus opted to go with separately implemented functions.



def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None):
    ax = ax if ax is not None else plt.gca()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)



def partly_isc(data, pairwise=False, summary_statistic=None, verbose=True):
    """Intersubject correlation
    For each voxel or ROI, compute the Pearson correlation between each
    subject's response time series and other subjects' response time series.
    If pairwise is False (default), use the leave-one-out approach, where
    correlation is computed between each subject and the average of the other
    subjects. If pairwise is True, compute correlations between all pairs of
    subjects. If summary_statistic is None, return N ISC values for N subjects
    (leave-one-out) or N(N-1)/2 ISC values for each pair of N subjects,
    corresponding to the upper triangle of the pairwise correlation matrix
    (see scipy.spatial.distance.squareform). Alternatively, supply either
    np.mean or np.median to compute summary statistic of ISCs (Fisher Z will
    be applied and inverted if using mean). Input data should be a list 
    where each item is a time-points by voxels ndarray for a given subject.
    Multiple input ndarrays must be the same shape. If a single ndarray is
    supplied, the last dimension is assumed to correspond to subjects. If 
    only two subjects are supplied, simply compute Pearson correlation
    (precludes averaging in leave-one-out approach, and does not apply
    summary statistic.) Output is an ndarray where the first dimension is
    the number of subjects or pairs and the second dimension is the number
    of voxels (or ROIs).
        
    The implementation is based on the following publication:
    
    .. [Hasson2004] "Intersubject synchronization of cortical activity 
    during natural vision.", U. Hasson, Y. Nir, I. Levy, G. Fuhrmann,
    R. Malach, 2004, Science, 303, 1634-1640.
    Parameters
    ----------
    data : list or ndarray
        fMRI data for which to compute ISC
        
    pairwise : bool, default: False
        Whether to use pairwise (True) or leave-one-out (False) approach
        
    summary_statistic : None
        Return all ISCs or collapse using np.mean or np.median
    Returns
    -------
    iscs : subjects or pairs by voxels ndarray
        ISC for each subject or pair (or summary statistic) per voxel
    """
    
    # Convert list input to 3d and check shapes
    if type(data) == list:
        data_shape = data[0].shape
        for i, d in enumerate(data):
            if d.shape != data_shape:
                raise ValueError("All ndarrays in input list "
                                 "must be the same shape!")
            if d.ndim == 1:
                data[i] = d[:, np.newaxis]
        data = np.dstack(data)

    # Convert input ndarray to 3d and check shape
    elif type(data) == np.ndarray:
        if data.ndim == 2:
            data = data[:, np.newaxis, :]            
        elif data.ndim == 3:
            pass
        else:
            raise ValueError("Input ndarray should have 2 "
                             f"or 3 dimensions (got {data.ndim})!")

    # Infer subjects, TRs, voxels and print for user to check
    n_subjects = data.shape[2]
    n_TRs = data.shape[0]
    n_voxels = data.shape[1]
    if verbose:
        print(f"Assuming {n_subjects} subjects with {n_TRs} time points "
              f"and {n_voxels} voxel(s) or ROI(s).")
    
    # Loop over each voxel or ROI
    voxel_iscs = []
    for v in np.arange(n_voxels):
        voxel_data = data[:, v, :].T
        if n_subjects == 2:
            iscs = pearsonr(voxel_data[0, :], voxel_data[1, :])[0]
            summary_statistic = None
            if verbose:
                print("Only two subjects! Simply computing Pearson correlation.")
        elif pairwise:
            iscs = squareform(np.corrcoef(voxel_data), checks=False)
        elif not pairwise:
            iscs = np.array([pearsonr(subject,
                                      np.mean(np.delete(voxel_data,
                                                        s, axis=0),
                                              axis=0))[0]
                    for s, subject in enumerate(voxel_data)])
        voxel_iscs.append(iscs)
    iscs = np.column_stack(voxel_iscs)
    
    # Summarize results (if requested)
    if summary_statistic == np.nanmean:
        iscs = np.tanh(np.nanmean(np.arctanh(iscs), axis=0))[np.newaxis, :]
    elif summary_statistic == np.mean:
        iscs = np.tanh(np.mean(np.arctanh(iscs), axis=0))[np.newaxis, :]
    elif summary_statistic == np.median:    
        iscs = summary_statistic(iscs, axis=0)[np.newaxis, :]
    elif not summary_statistic:
        pass
    else:
        raise ValueError("Unrecognized summary_statistic! Use None, np.median, or np.mean.")
    return iscs


def partly_phaseshift_isc(data, pairwise=False, summary_statistic=np.median,
                   n_shifts=1000, return_distribution=False, random_state=None):
    
    """Phase randomization for one-sample ISC test
    
    For each voxel or ROI, compute the actual ISC and p-values
    from a null distribution of ISCs where response time series
    are phase randomized prior to computing ISC. If pairwise,
    apply phase randomization to each subject and compute pairwise
    ISCs. If leave-one-out approach is used (pairwise=False), only
    apply phase randomization to the left-out subject in each iteration
    of the leave-one-out procedure. Input data should be a list where
    each item is a time-points by voxels ndarray for a given subject.
    Multiple input ndarrays must be the same shape. If a single ndarray is
    supplied, the last dimension is assumed to correspond to subjects.
    Returns the observed ISC and p-values (two-tailed test). Optionally
    returns the null distribution of ISCs computed on phase-randomized
    data.
    
    This implementation is based on the following publications:
    .. [Lerner2011] "Topographic mapping of a hierarchy of temporal
    receptive windows using a narrated story.", Y. Lerner, C. J. Honey,
    L. J. Silbert, U. Hasson, 2011, Journal of Neuroscience, 31, 2906-2915.
    .. [Simony2016] "Dynamic reconfiguration of the default mode network
    during narrative comprehension.", E. Simony, C. J. Honey, J. Chen, O.
    Lositsky, Y. Yeshurun, A. Wiesel, U. Hasson, 2016, Nature Communications,
    7, 12141.
    Parameters
    ----------
    data : list or dict, time series data for multiple subjects
        List or dictionary of response time series for multiple subjects
    pairwise : bool, default:False
        Indicator of pairwise or leave-one-out, should match iscs variable
    summary_statistic : numpy function, default:np.median
        Summary statistic, either np.median (default) or np.mean
        
    n_shifts : int, default:1000
        Number of randomly shifted samples
        
    return_distribution : bool, default:False
        Optionally return the bootstrap distribution of summary statistics
        
    random_state = int, None, or np.random.RandomState, default:None
        Initial random seed
    Returns
    -------
    observed : float, observed ISC (without time-shifting)
        Actual ISCs
    p : float, p-value
        p-value based on time-shifting randomization test
        
    distribution : ndarray, time-shifts by voxels (optional)
        Time-shifted null distribution if return_bootstrap=True
    """

    # Convert list input to 3d and check shapes
    if type(data) == list:
        data_shape = data[0].shape
        for i, d in enumerate(data):
            if d.shape != data_shape:
                raise ValueError("All ndarrays in input list "
                                 "must be the same shape!")
            if d.ndim == 1:
                data[i] = d[:, np.newaxis]
        data = np.dstack(data)

    # Convert input ndarray to 3d and check shape
    elif type(data) == np.ndarray:
        if data.ndim == 2:
            data = data[:, np.newaxis, :]            
        elif data.ndim == 3:
            pass
        else:
            raise ValueError("Input ndarray should have 2 "
                             f"or 3 dimensions (got {data.ndim})!")

    # Infer subjects, TRs, voxels and print for user to check
    n_subjects = data.shape[2]
    n_TRs = data.shape[0]
    n_voxels = data.shape[1]
    
    # Get actual observed ISC
    observed = partly_isc(data, pairwise=pairwise, summary_statistic=summary_statistic, verbose=False)
    
    # Iterate through randomized shifts to create null distribution
    distribution = []
    for i in np.arange(n_shifts):
        
        # Random seed to be deterministically re-randomized at each iteration
        if isinstance(random_state, np.random.RandomState):
            prng = random_state
        else:
            prng = np.random.RandomState(random_state)
            
        # Get randomized phase shifts
        if data.shape[0] % 2 == 0:
            # Why are we indexing from 1 not zero here? Vector is n_TRs / -1 long?
            pos_freq = np.arange(1, data.shape[0] // 2)
            neg_freq = np.arange(data.shape[0] - 1, data.shape[0] // 2, -1)
        else:
            pos_freq = np.arange(1, (data.shape[0] - 1) // 2 + 1)
            neg_freq = np.arange(data.shape[0] - 1, (data.shape[0] - 1) // 2, -1)

        phase_shifts = prng.rand(len(pos_freq), 1, n_subjects) * 2 * np.math.pi
        
        # In pairwise approach, apply all shifts then compute pairwise ISCs
        if pairwise:
        
            # Fast Fourier transform along time dimension of data
            fft_data = fft(data, axis=0)

            # Shift pos and neg frequencies symmetrically, to keep signal real
            fft_data[pos_freq, :, :] *= np.exp(1j * phase_shifts)
            fft_data[neg_freq, :, :] *= np.exp(-1j * phase_shifts)

            # Inverse FFT to put data back in time domain for ISC
            shifted_data = np.real(ifft(fft_data, axis=0))

            # Compute null ISC on shifted data for pairwise approach
            shifted_isc = partly_isc(shifted_data, pairwise=True,
                              summary_statistic=summary_statistic, verbose=False)
        
        # In leave-one-out, apply shift only to each left-out participant
        elif not pairwise:
            
            # Roll subject axis in phaseshifts for loop
            phase_shifts = np.rollaxis(phase_shifts, 2, 0)
            
            shifted_isc = []
            for s, shift in enumerate(phase_shifts):
                
                # Apply FFT to left-out subject
                fft_subject = fft(data[:, :, s], axis=0)
                
                # Shift pos and neg frequencies symmetrically, to keep signal real
                fft_subject[pos_freq, :] *= np.exp(1j * shift)
                fft_subject[neg_freq, :] *= np.exp(-1j * shift)

                # Inverse FFT to put data back in time domain for ISC
                shifted_subject = np.real(ifft(fft_subject, axis=0))

                # Compute ISC of shifted left-out subject against mean of N-1 subjects
                nonshifted_mean = np.mean(np.delete(data, s, 2), axis=2)
                loo_isc = partly_isc(np.dstack((shifted_subject, nonshifted_mean)), pairwise=False,
                              summary_statistic=None, verbose=False)
                shifted_isc.append(loo_isc)
                
            # Get summary statistics across left-out subjects
            if summary_statistic == np.mean:
                shifted_isc = np.tanh(np.mean(np.arctanh(np.dstack(shifted_isc)), axis=2))
            elif summary_statistic == np.nanmean:
                shifted_isc = np.tanh(np.nanmean(np.arctanh(np.dstack(shifted_isc)), axis=2))
            elif summary_statistic == np.median:
                shifted_isc = np.median(np.dstack(shifted_isc), axis=2)
                
        distribution.append(shifted_isc)
        
        # Update random state for next iteration
        random_state = np.random.RandomState(prng.randint(0, 2**32 - 1))
        
    # Convert distribution to numpy array
    distribution = np.vstack(distribution)
    assert distribution.shape == (n_shifts, n_voxels)

    # Get p-value for actual median from shifted distribution
    p = ((np.sum(np.abs(distribution) >= np.abs(observed), axis=0) + 1) /
          float((len(distribution) + 1)))[np.newaxis, :]
    
    if return_distribution:
        return observed, p, distribution
    elif not return_distribution:
        return observed, p
    
    
def loop_tisc_groups(input_array_filename, input_name, onsets, offsets):
    ts_array = np.load(input_array_filename)

    n_subjs = ts_array.shape[0]
    ts_data = np.swapaxes(ts_array, 0, 2)
    ts_data = np.swapaxes(ts_data, 0, 1)
    stacked_res = []
    stacked_res_cond_vec = []
    segment_iscs = np.zeros( 4)
    a = []
    b = []
    # loop over segments
    for curr_bin in range(4):#2,4, 1):
                curr_data = np.squeeze(ts_data[onsets[curr_bin]:offsets[curr_bin], :, : ])
                curr_res = partly_isc(curr_data, summary_statistic=np.mean, verbose = False)
                a.append(curr_res[0])
                curr_res = partly_isc(curr_data, pairwise=False, verbose = False)
                stacked_res.append(np.arctanh(partly_isc(curr_data, pairwise =False, verbose = False)))
                stacked_res_cond_vec.append((np.ones(curr_res.shape[0]) * curr_bin).T)

    ISC = np.asarray(stacked_res).flatten()
    cond = np.asarray(stacked_res_cond_vec).flatten()

    df = pd.DataFrame(ISC, columns = ['ISC'])
    df['Segment'] = cond
    sns.set(font_scale=2)
    sns.set_style('white')

    x = np.arange(segment_iscs.shape[0])

    fill_color = ['yellow', 'red','green','orange']

    plt.figure(figsize = (8,6))

    ax = sns.barplot(x="Segment", y="ISC", data=df,
                    palette = fill_color, alpha = 0.5, )
    plt.title(input_name)
    plt.xticks(x, ['P','RA','C','R']);
    plt.ylim([0, 0.5])
    plt.yticks(np.arange(0,0.51, 0.5));
    sns.despine()
    plt.show()
    
    



def compute_summary_statistic(iscs, summary_statistic='mean', axis=None):

    """Computes summary statistics for ISCs
    Computes either the 'mean' or 'median' across a set of ISCs. In the
    case of the mean, ISC values are first Fisher Z transformed (arctanh),
    averaged, then inverse Fisher Z transformed (tanh).
    The implementation is based on the work in [SilverDunlap1987]_.
    .. [SilverDunlap1987] "Averaging corrlelation coefficients: should
       Fisher's z transformation be used?", N. C. Silver, W. P. Dunlap, 1987,
       Journal of Applied Psychology, 72, 146-148.
       https://doi.org/10.1037/0021-9010.72.1.146
    Parameters
    ----------
    iscs : list or ndarray
        ISC values
    summary_statistic : str, default: 'mean'
        Summary statistic, 'mean' or 'median'
    axis : None or int or tuple of ints, optional
        Axis or axes along which the means are computed. The default is to
        compute the mean of the flattened array.
    Returns
    -------
    statistic : float or ndarray
        Summary statistic of ISC values
    """

    if summary_statistic not in ('mean', 'median'):
        raise ValueError("Summary statistic must be 'mean' or 'median'")

    # Compute summary statistic
    if summary_statistic == 'mean':
        statistic = np.tanh(np.nanmean(np.arctanh(iscs), axis=axis))
    elif summary_statistic == 'median':
        statistic = np.nanmedian(iscs, axis=axis)

    return statistic


def p_from_null(observed, distribution,
                side='two-sided', exact=False,
                axis=None):
    """Compute p-value from null distribution
    Returns the p-value for an observed test statistic given a null
    distribution. Performs either a 'two-sided' (i.e., two-tailed)
    test (default) or a one-sided (i.e., one-tailed) test for either the
    'left' or 'right' side. For an exact test (exact=True), does not adjust
    for the observed test statistic; otherwise, adjusts for observed
    test statistic (prevents p-values of zero). If a multidimensional
    distribution is provided, use axis argument to specify which axis indexes
    resampling iterations.
    The implementation is based on the work in [PhipsonSmyth2010]_.
    .. [PhipsonSmyth2010] "Permutation p-values should never be zero:
       calculating exact p-values when permutations are randomly drawn.",
       B. Phipson, G. K., Smyth, 2010, Statistical Applications in Genetics
       and Molecular Biology, 9, 1544-6115.
       https://doi.org/10.2202/1544-6115.1585
    Parameters
    ----------
    observed : float
        Observed test statistic
    distribution : ndarray
        Null distribution of test statistic
    side : str, default:'two-sided'
        Perform one-sided ('left' or 'right') or 'two-sided' test
    axis: None or int, default:None
        Axis indicating resampling iterations in input distribution
    Returns
    -------
    p : float
        p-value for observed test statistic based on null distribution
    """

    if side not in ('two-sided', 'left', 'right'):
        raise ValueError("The value for 'side' must be either "
                         "'two-sided', 'left', or 'right', got {0}".
                         format(side))

    n_samples = len(distribution)
    logger.info("Assuming {0} resampling iterations".format(n_samples))

    if side == 'two-sided':
        # Numerator for two-sided test
        numerator = np.sum(np.abs(distribution) >= np.abs(observed), axis=axis)
    elif side == 'left':
        # Numerator for one-sided test in left tail
        numerator = np.sum(distribution <= observed, axis=axis)
    elif side == 'right':
        # Numerator for one-sided test in right tail
        numerator = np.sum(distribution >= observed, axis=axis)

    # If exact test all possible permutations and do not adjust
    if exact:
        p = numerator / n_samples

    # If not exact test, adjust number of samples to account for
    # observed statistic; prevents p-value from being zero
    else:
        p = (numerator + 1) / (n_samples + 1)

    return p


def _check_isc_input(iscs, pairwise=False):

    """Checks ISC inputs for statistical tests
    Input ISCs should be n_subjects (leave-one-out approach) or
    n_pairs (pairwise approach) by n_voxels or n_ROIs array or a 1D
    array (or list) of ISC values for a single voxel or ROI. This
    function is only intended to be used internally by other
    functions in this module (e.g., bootstrap_isc, permutation_isc).
    Parameters
    ----------
    iscs : ndarray or list
        ISC values
    Returns
    -------
    iscs : ndarray
        Array of ISC values
    n_subjects : int
        Number of subjects
    n_voxels : int
        Number of voxels (or ROIs)
    """

    # Standardize structure of input data
    if type(iscs) == list:
        iscs = np.array(iscs)[:, np.newaxis]

    elif isinstance(iscs, np.ndarray):
        if iscs.ndim == 1:
            iscs = iscs[:, np.newaxis]

    # Check if incoming pairwise matrix is vectorized triangle
    if pairwise:
        try:
            test_square = squareform(iscs[:, 0])
            n_subjects = test_square.shape[0]
        except ValueError:
            raise ValueError("For pairwise input, ISCs must be the "
                             "vectorized triangle of a square matrix.")
    elif not pairwise:
        n_subjects = iscs.shape[0]

    # Infer subjects, voxels and print for user to check
    n_voxels = iscs.shape[1]
    logger.info("Assuming {0} subjects with and {1} "
                "voxel(s) or ROI(s) in bootstrap ISC test.".format(n_subjects,
                                                                   n_voxels))

    return iscs, n_subjects, n_voxels




def _threshold_nans(data, tolerate_nans):

    """Thresholds data based on proportion of subjects with NaNs
    Takes in data and a threshold value (float between 0.0 and 1.0) determining
    the permissible proportion of subjects with non-NaN values. For example, if
    threshold=.8, any voxel where >= 80% of subjects have non-NaN values will
    be left unchanged, while any voxel with < 80% non-NaN values will be
    assigned all NaN values and included in the nan_mask output. Note that the
    output data has not been masked and will be same shape as the input data,
    but may have a different number of NaNs based on the threshold.
    Parameters
    ----------
    data : ndarray (n_TRs x n_voxels x n_subjects)
        fMRI time series data
    tolerate_nans : bool or float (0.0 <= threshold <= 1.0)
        Proportion of subjects with non-NaN values required to keep voxel
    Returns
    -------
    data : ndarray (n_TRs x n_voxels x n_subjects)
        fMRI time series data with adjusted NaNs
    nan_mask : ndarray (n_voxels,)
        Boolean mask array of voxels with too many NaNs based on threshold
    """

    nans = np.all(np.any(np.isnan(data), axis=0), axis=1)

    # Check tolerate_nans input and use either mean/nanmean and exclude voxels
    if tolerate_nans is True:
        logger.info("ISC computation will tolerate all NaNs when averaging")

    elif type(tolerate_nans) is float:
        if not 0.0 <= tolerate_nans <= 1.0:
            raise ValueError("If threshold to tolerate NaNs is a float, "
                             "it must be between 0.0 and 1.0; got {0}".format(
                                tolerate_nans))
        nans += ~(np.sum(~np.any(np.isnan(data), axis=0), axis=1) >=
                  data.shape[-1] * tolerate_nans)
        logger.info("ISC computation will tolerate voxels with at least "
                    "{0} non-NaN values: {1} voxels do not meet "
                    "threshold".format(tolerate_nans,
                                       np.sum(nans)))

    else:
        logger.info("ISC computation will not tolerate NaNs when averaging")

    mask = ~nans
    data = data[:, mask, :]

    return data, mask





def _check_group_assignment(group_assignment, n_subjects):
    if type(group_assignment) == list:
        pass
    elif type(group_assignment) == np.ndarray:
        group_assignment = group_assignment.tolist()
    else:
        logger.info("No group assignment provided, "
                    "performing one-sample test.")

    if group_assignment and len(group_assignment) != n_subjects:
        raise ValueError("Group assignments ({0}) "
                         "do not match number of subjects ({1})!".format(
                                len(group_assignment), n_subjects))
    return group_assignment


def _get_group_parameters(group_assignment, n_subjects, pairwise=False):

    # Set up dictionary to contain group info
    group_parameters = {'group_assignment': group_assignment,
                        'n_subjects': n_subjects,
                        'group_labels': None, 'groups': None,
                        'sorter': None, 'unsorter': None,
                        'group_matrix': None, 'group_selector': None}

    # Set up group selectors for two-group scenario
    if group_assignment and len(np.unique(group_assignment)) == 2:
        group_parameters['n_groups'] = 2

        # Get group labels and counts
        group_labels = np.unique(group_assignment)
        groups = {group_labels[0]: group_assignment.count(group_labels[0]),
                  group_labels[1]: group_assignment.count(group_labels[1])}

        # For two-sample pairwise approach set up selector from matrix
        if pairwise:
            # Sort the group_assignment variable if it came in shuffled
            # so it's easier to build group assignment matrix
            sorter = np.array(group_assignment).argsort()
            unsorter = np.array(group_assignment).argsort().argsort()

            # Populate a matrix with group assignments
            upper_left = np.full((groups[group_labels[0]],
                                  groups[group_labels[0]]),
                                 group_labels[0])
            upper_right = np.full((groups[group_labels[0]],
                                   groups[group_labels[1]]),
                                  np.nan)
            lower_left = np.full((groups[group_labels[1]],
                                  groups[group_labels[0]]),
                                 np.nan)
            lower_right = np.full((groups[group_labels[1]],
                                   groups[group_labels[1]]),
                                  group_labels[1])
            group_matrix = np.vstack((np.hstack((upper_left, upper_right)),
                                      np.hstack((lower_left, lower_right))))
            np.fill_diagonal(group_matrix, np.nan)
            group_parameters['group_matrix'] = group_matrix

            # Unsort matrix and squareform to create selector
            group_parameters['group_selector'] = squareform(
                                        group_matrix[unsorter, :][:, unsorter],
                                        checks=False)
            group_parameters['sorter'] = sorter
            group_parameters['unsorter'] = unsorter

        # If leave-one-out approach, just user group assignment as selector
        else:
            group_parameters['group_selector'] = group_assignment

        # Save these parameters for later
        group_parameters['groups'] = groups
        group_parameters['group_labels'] = group_labels

    # Manage one-sample and incorrect group assignments
    elif not group_assignment or len(np.unique(group_assignment)) == 1:
        group_parameters['n_groups'] = 1

        # If pairwise initialize matrix of ones for sign-flipping
        if pairwise:
            group_parameters['group_matrix'] = np.ones((
                                            group_parameters['n_subjects'],
                                            group_parameters['n_subjects']))

    elif len(np.unique(group_assignment)) > 2:
        raise ValueError("This test is not valid for more than "
                         "2 groups! (got {0})".format(
                                len(np.unique(group_assignment))))
    else:
        raise ValueError("Invalid group assignments!")

    return group_parameters


def _permute_one_sample_iscs(iscs, group_parameters, i, pairwise=False,
                             summary_statistic='median', group_matrix=None,
                             exact_permutations=None, prng=None):

    """Applies one-sample permutations to ISC data
    Input ISCs should be n_subjects (leave-one-out approach) or
    n_pairs (pairwise approach) by n_voxels or n_ROIs array.
    This function is only intended to be used internally by the
    permutation_isc function in this module.
    Parameters
    ----------
    iscs : ndarray or list
        ISC values
    group_parameters : dict
        Dictionary of group parameters
    i : int
        Permutation iteration
    pairwise : bool, default: False
        Indicator of pairwise or leave-one-out, should match ISCs variable
    summary_statistic : str, default: 'median'
        Summary statistic, either 'median' (default) or 'mean'
    exact_permutations : list
        List of permutations
    prng = None or np.random.RandomState, default: None
        Initial random seed
    Returns
    -------
    isc_sample : ndarray
        Array of permuted ISC values
    """

    # Randomized sign-flips
    if exact_permutations:
        sign_flipper = np.array(exact_permutations[i])
    else:
        sign_flipper = prng.choice([-1, 1],
                                   size=group_parameters['n_subjects'],
                                   replace=True)

    # If pairwise, apply sign-flips by rows and columns
    if pairwise:
        matrix_flipped = (group_parameters['group_matrix'] * sign_flipper
                                                           * sign_flipper[
                                                                :, np.newaxis])
        sign_flipper = squareform(matrix_flipped, checks=False)

    # Apply flips along ISC axis (same across voxels)
    isc_flipped = iscs * sign_flipper[:, np.newaxis]

    # Get summary statistics on sign-flipped ISCs
    isc_sample = compute_summary_statistic(
                    isc_flipped,
                    summary_statistic=summary_statistic,
                    axis=0)

    return isc_sample


def _permute_two_sample_iscs(iscs, group_parameters, i, pairwise=False,
                             summary_statistic='median',
                             exact_permutations=None, prng=None):

    """Applies two-sample permutations to ISC data
    Input ISCs should be n_subjects (leave-one-out approach) or
    n_pairs (pairwise approach) by n_voxels or n_ROIs array.
    This function is only intended to be used internally by the
    permutation_isc function in this module.
    Parameters
    ----------
    iscs : ndarray or list
        ISC values
    group_parameters : dict
        Dictionary of group parameters
    i : int
        Permutation iteration
    pairwise : bool, default: False
        Indicator of pairwise or leave-one-out, should match ISCs variable
    summary_statistic : str, default: 'median'
        Summary statistic, either 'median' (default) or 'mean'
    exact_permutations : list
        List of permutations
    prng = None or np.random.RandomState, default: None
        Initial random seed
        Indicator of pairwise or leave-one-out, should match ISCs variable
    Returns
    -------
    isc_sample : ndarray
        Array of permuted ISC values
    """

    # Shuffle the group assignments
    if exact_permutations:
        group_shuffler = np.array(exact_permutations[i])
    elif not exact_permutations and pairwise:
        group_shuffler = prng.permutation(np.arange(
            len(np.array(group_parameters['group_assignment'])[
                            group_parameters['sorter']])))
    elif not exact_permutations and not pairwise:
        group_shuffler = prng.permutation(np.arange(
            len(group_parameters['group_assignment'])))

    # If pairwise approach, convert group assignments to matrix
    if pairwise:

        # Apply shuffler to group matrix rows/columns
        group_shuffled = group_parameters['group_matrix'][
                            group_shuffler, :][:, group_shuffler]

        # Unsort shuffled matrix and squareform to create selector
        group_selector = squareform(group_shuffled[
                                    group_parameters['unsorter'], :]
                                    [:, group_parameters['unsorter']],
                                    checks=False)

    # Shuffle group assignments in leave-one-out two sample test
    elif not pairwise:

        # Apply shuffler to group matrix rows/columns
        group_selector = np.array(
                    group_parameters['group_assignment'])[group_shuffler]

    # Get difference of within-group summary statistics
    # with group permutation
    isc_sample = (compute_summary_statistic(
                    iscs[group_selector == group_parameters[
                                            'group_labels'][0], :],
                    summary_statistic=summary_statistic,
                    axis=0) -
                  compute_summary_statistic(
                    iscs[group_selector == group_parameters[
                                            'group_labels'][1], :],
                    summary_statistic=summary_statistic,
                    axis=0))

    return isc_sample


def permutation_isc(iscs, group_assignment=None, pairwise=False,  # noqa: C901
                    summary_statistic='median', n_permutations=1000,
                    random_state=None):

    """Group-level permutation test for ISCs
    For ISCs from one or more voxels or ROIs, permute group assignments to
    construct a permutation distribution. Input is a list or ndarray of
    ISCs  for a single voxel/ROI, or an ISCs-by-voxels ndarray. If two groups,
    ISC values should stacked along first dimension (vertically), and a
    group_assignment list (or 1d array) of same length as the number of
    subjects should be provided to indicate groups. If no group_assignment
    is provided, one-sample test is performed using a sign-flipping procedure.
    Performs exact test if number of possible permutations (2**N for one-sample
    sign-flipping, N! for two-sample shuffling) is less than or equal to number
    of requested permutation; otherwise, performs approximate permutation test
    using Monte Carlo resampling. ISC values should either be N ISC values for
    N subjects in the leave-one-out approach (pairwise=False) or N(N-1)/2 ISC
    values for N subjects in the pairwise approach (pairwise=True). In the
    pairwise approach, ISC values should correspond to the vectorized upper
    triangle of a square corrlation matrix (scipy.stats.distance.squareform).
    Note that in the pairwise approach, group_assignment order should match the
    row/column order of the subject-by-subject square ISC matrix even though
    the input ISCs should be supplied as the vectorized upper triangle of the
    square ISC matrix. Returns the observed ISC and permutation-based p-value
    (two-tailed test), as well as the permutation distribution of summary
    statistic. According to Chen et al., 2016, this is the preferred
    nonparametric approach for controlling false positive rates (FPR) for
    two-sample tests. This approach may yield inflated FPRs for one-sample
    tests.
    The implementation is based on the work in [Chen2016]_.
    Parameters
    ----------
    iscs : list or ndarray, correlation matrix of ISCs
        ISC values for one or more voxels
    group_assignment : list or ndarray, group labels
        Group labels matching order of ISC input
    pairwise : bool, default: False
        Indicator of pairwise or leave-one-out, should match ISCs variable
    summary_statistic : str, default: 'median'
        Summary statistic, either 'median' (default) or 'mean'
    n_permutations : int, default: 1000
        Number of permutation iteration (randomizing group assignment)
    random_state = int, None, or np.random.RandomState, default: None
        Initial random seed
    Returns
    -------
    observed : float, ISC summary statistic or difference
        Actual ISC or group difference (excluding between-group ISCs)
    p : float, p-value
        p-value based on permutation test
    distribution : ndarray, permutations by voxels (optional)
        Permutation distribution if return_bootstrap=True
    """

    # Standardize structure of input data
    iscs, n_subjects, n_voxels = _check_isc_input(iscs, pairwise=pairwise)

    # Check for valid summary statistic
    if summary_statistic not in ('mean', 'median'):
        raise ValueError("Summary statistic must be 'mean' or 'median'")

    # Check match between group labels and ISCs
    group_assignment = _check_group_assignment(group_assignment,
                                               n_subjects)

    # Get group parameters
    group_parameters = _get_group_parameters(group_assignment, n_subjects,
                                             pairwise=pairwise)

    # Set up permutation type (exact or Monte Carlo)
    if group_parameters['n_groups'] == 1:
        if n_permutations < 2**n_subjects:
            logger.info("One-sample approximate permutation test using "
                        "sign-flipping procedure with Monte Carlo resampling.")
            exact_permutations = None
        else:
            logger.info("One-sample exact permutation test using "
                        "sign-flipping procedure with 2**{0} "
                        "({1}) iterations.".format(n_subjects,
                                                   2**n_subjects))
            exact_permutations = list(product([-1, 1], repeat=n_subjects))
            n_permutations = 2**n_subjects

    # Check for exact test for two groups
    else:
        if n_permutations < np.math.factorial(n_subjects):
            logger.info("Two-sample approximate permutation test using "
                        "group randomization with Monte Carlo resampling.")
            exact_permutations = None
        else:
            logger.info("Two-sample exact permutation test using group "
                        "randomization with {0}! "
                        "({1}) iterations.".format(
                                n_subjects,
                                np.math.factorial(n_subjects)))
            exact_permutations = list(permutations(
                np.arange(len(group_assignment))))
            n_permutations = np.math.factorial(n_subjects)

    # If one group, just get observed summary statistic
    if group_parameters['n_groups'] == 1:
        observed = compute_summary_statistic(
                        iscs,
                        summary_statistic=summary_statistic,
                        axis=0)[np.newaxis, :]

    # If two groups, get the observed difference
    else:
        observed = (compute_summary_statistic(
                        iscs[group_parameters['group_selector'] ==
                             group_parameters['group_labels'][0], :],
                        summary_statistic=summary_statistic,
                        axis=0) -
                    compute_summary_statistic(
                        iscs[group_parameters['group_selector'] ==
                             group_parameters['group_labels'][1], :],
                        summary_statistic=summary_statistic,
                        axis=0))
        observed = np.array(observed)

    # Set up an empty list to build our permutation distribution
    distribution = []

    # Loop through n permutation iterations and populate distribution
    for i in np.arange(n_permutations):

        # Random seed to be deterministically re-randomized at each iteration
        if exact_permutations:
            prng = None
        elif isinstance(random_state, np.random.RandomState):
            prng = random_state
        else:
            prng = np.random.RandomState(random_state)

        # If one group, apply sign-flipping procedure
        if group_parameters['n_groups'] == 1:
            isc_sample = _permute_one_sample_iscs(
                            iscs, group_parameters, i,
                            pairwise=pairwise,
                            summary_statistic=summary_statistic,
                            exact_permutations=exact_permutations,
                            prng=prng)

        # If two groups, set up group matrix get the observed difference
        else:
            isc_sample = _permute_two_sample_iscs(
                            iscs, group_parameters, i,
                            pairwise=pairwise,
                            summary_statistic=summary_statistic,
                            exact_permutations=exact_permutations,
                            prng=prng)

        # Tack our permuted ISCs onto the permutation distribution
        distribution.append(isc_sample)

        # Update random state for next iteration
        if not exact_permutations:
            random_state = np.random.RandomState(prng.randint(
                                                    0, MAX_RANDOM_SEED))

    # Convert distribution to numpy array
    distribution = np.array(distribution)

    # Get p-value for actual median from shifted distribution
    if exact_permutations:
        p = p_from_null(observed, distribution,
                        side='two-sided', exact=True,
                        axis=0)
    else:
        p = p_from_null(observed, distribution,
                        side='two-sided', exact=False,
                        axis=0)

    return observed, p, distribution



