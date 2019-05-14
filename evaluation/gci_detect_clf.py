# -*- coding: utf-8 -*-
import numpy as np
import logging
import os.path
import warnings
from scipy.io import wavfile
from collections import defaultdict
from pm import OnePm, Pm
from pm_compare import PM_Compare
from gci_utils import insert_trans_gci, seconds2samples, samples2seconds, sync_gci_to_samp_peak

"""
Glottal closure detection module.

This module contains code for glottal closure instant (GCI) detection. Scikit sklearn library is heavily used.

Attributes:
    logger (:obj:`logging.Logger`): Logger object
    T_MARK_DIST_TIME (float): Tine distance (sec) between "regular" GCI and possibly inserted transitional GCI

"""

logger = logging.getLogger('gci_detect')

# Time distance (s) between "regular" GCI and possibly inserted transitional GCI
T_MARK_DIST_TIME = 0.0005
# GCI_SYNC_LE_TIME = 0.002500
# GCI_SYNC_RI_TIME = 0.000250


class InfoArray(np.ndarray):
    """
    Subclass of Numpy array for storing an extra information within a Numpy array.

    An extra information `info` is stored to the Numpy array as a whole.

    """
    def __new__(cls, input_array, info):
        """
        New method.

        Args:
            input_array (array-like): Numpy array (ndarray instance).
            info (): Extra information stored within the Numpy array.

        Returns:
            object: The newly created instance of the `InfoArray` object

        """
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.info = info
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        """
        Finalize array.

        Args:
            obj (object): `InfoArray` object.

        """
        if obj is None:
            return
        self.info = getattr(obj, 'info', None)

    def __reduce__(self):
        """Pickle reduction method.

        State is encoded in the zeroth element of third element of the returned tuple, itself a tuple used to restore
        the state of the ndarray. This is always defined for numpy arrays.

        Notes:
            There is no :func:`__getstate__` method defined for numpy array. :func:`__reduce__` is used instead.

        See Also:
            http://numpy-discussion.10968.n7.nabble.com/Possible-to-pickle-new-state-in-NDArray-subclasses-td43772.html
            https://bitbucket.org/yt_analysis/yt/src/yt/yt/units/yt_array.py?fileviewer=file-view-default#yt_array.py-1250

        """
        np_ret = super(InfoArray, self).__reduce__()
        np_state = np_ret[2]
        state = ((self.info,) + np_state[:],)
        new_ret = np_ret[:2] + state + np_ret[3:]
        return new_ret

    def __setstate__(self, state, **kwargs):
        """Pickle setstate method.

        This is called inside :func:`pickle.load` and restores the state from the metadata extracted in __reduce__ and
        then serialized by pickle.

        Args:
            **kwargs (list): Helper list to fulfill signature of the superclass; not used explicitly.

        """
        super(InfoArray, self).__setstate__(state[1:], **kwargs)
        self.info = state[0]


class Utt:
    """
    Single utterance class.

    Contains indices to the original data examples (corresponding to the given utterance), (negative) peaks location in
    the utterance (in number of samples), peak-wise features (examples), peak-wise targets to corresponding examples (if
    available), waveform samples, reference GCIs (if available).

    Attributes:
        logger (:obj:`logging.Logger`): Logger object.

    """
    logger = logging.getLogger('gci_detect.Utt')

    def __init__(self, name, indices):
        """
        Init method.

        Args:
            name (str):                     The name of the utterance.
            indices (array-like):           Indices of data examples.

        """
        self._name = name
        self._indices = indices
        self._peaks = np.array([])
        self._samples = None
        self._ref_gcis = None
        self._samp_freq = None
        self._examples = []
        self._targets = []
        self.logger.debug('Utterance {} object created'.format(name))

    def __str__(self):
        return '{}: sf={} peaks={} indices={} GCIs={}'.format(self._name, self.samp_freq, self._peaks, self._indices,
                                                              len(self._ref_gcis))

    def __len__(self):
        return len(self._peaks)

    def feats_from_dataframe(self, df, utt_idx, peak_label, target_label='target'):
        """
        Read examples from a Pandas data frame.

        Args:
            df (:obj:`pandas.DataFrame`):   Data frame with rows as examples and columns as features.
            utt_idx (int):                  Utterance index
            peak_label (str):               Negative peak label. Examples are extracted around negative peak labels.
            target_label (str):             Target label (typically 1 or 0).

        """
        self._peaks = df[peak_label].values
        if len(self._indices) != len(self._peaks):
            raise RuntimeError('Number of peaks and indices must be equal')
        self._targets = df[target_label].values if target_label in df.columns else []
        # Check whether there are targets in data frame
        if target_label in df.columns:
            # Store targets out ofthe  data frame
            self._targets = df[target_label].values
            # Drop the targets in further feature processing
            drop_cols = [peak_label, target_label]
        else:
            self._targets = []
            drop_cols = [peak_label]
        # Remove non-feature columns (peak and possibly also target columns) and set up examples
        self._examples = [InfoArray(row, utt_idx)
                          for row in df.drop(drop_cols, axis=1).itertuples(index=False, name=None)]
        self.logger.debug('Utterance {} ({}) object read from {}'.format(self._name, utt_idx, df.head()))

    def read_samples(self, path):
        """Read waveform samples from a file.

        Args:
            path (str): Path to file with waveform (typically with wav extension).

        Returns:
            int: Sampling frequency

        """
        self._samp_freq, self._samples = wavfile.read(path)
        return self._samp_freq

    def read_gcis(self, path):
        """Read GCIs from a pitch-mark (text) file.

        Args:
            path (str): Path to file with pitch-marks (typically with pm extension).

        Returns:
            int: Number of pitch-marks (GCIs) read

        """
        self._ref_gcis = Pm(path)
        return len(self._ref_gcis)

    @property
    def name(self):
        """str: Utterance name."""
        return self._name

    # Indices to examples
    @property
    def indices(self):
        """array-like: Indices to examples"""
        return self._indices

    @indices.setter
    def indices(self, indices):
        self._indices = indices

    # Speech samples
    @property
    def samples(self):
        """:obj:`numpy.array`: Waveform samples."""
        return self._samples

    # Number of speech samples
    @property
    def n_samples(self):
        """int: Number of waveform samples."""
        return len(self._samples)

    # Reference GCIs (if available)
    @property
    def ref_gcis(self):
        """:obj:`Pm`: Pitch-marks object"""
        return self._ref_gcis

    # Peak indices in samples
    @property
    def peaks(self):
        """:obj:`numpy.array`: Array of peak positions (in samples) - int"""
        return self._peaks

    @peaks.setter
    def peaks(self, peaks):
        self._peaks = peaks

    # Peak indices in times
    @property
    def peak_times(self):
        """:obj:`numpy.array`: Array of peak positions (in seconds) - float"""
        return self._peaks/self.samp_freq

    # Sampling frequency
    @property
    def samp_freq(self):
        """int : Sampling frequency (Hz)"""
        return self._samp_freq

    @property
    def examples(self):
        """:obj:`list`: Examples (list of :obj:`InfoArray`)."""
        return self._examples

    @property
    def targets(self):
        return self._targets


class Utts:
    """Utterances class.

    Contains list of utterances.

    Attributes:
        _utts (list):       List of :obj:`Utt` objects.
        _n_examples (int):  Total number of examples in all utterances.

    """
    def __init__(self):
        """
        Init method.

        """
        self._utts = []
        self._n_examples = 0

    def __str__(self):
        return 'List of {} utterances'.format(len(self._utts))

    # Read utterances from utterance-based data frames + reference GCIs and original speech waveforms
    def read(self, dfs, unames, samp_dir, ref_gci_dir=None, peak_label='negPeakIdx'):
        """
        Read utterances from utterance-based data frames.

        Read also the corresponding original speech waveforms and optionally reference GCIs.

        Args:
            dfs (:obj:`list`):      Utterance-wise list of data frames (:obj:`pandas.DataFrame`).
            unames (:obj:`list`):   List of utterance names `str`).
            samp_dir (str):         Directory with waveform samples.
            ref_gci_dir (str):      Directory with reference GCIs (optional).
            peak_label (str):       (Negative) peak label (optional).

        """
        self._utts = []
        utt_beg = 0
        utt_end = 0
        fs_prev = None
        for idx, (un, df) in enumerate(zip(unames, dfs)):
            # Set the index of the last example of the given utterance in the global dataset
            utt_end = utt_beg + df.shape[0]
            # Set up the structure of the given utterance and append it
            utt = Utt(un, np.arange(utt_beg, utt_end))
            # Read features
            utt.feats_from_dataframe(df, idx, peak_label)
            # Read speech samples
            fs_curr = utt.read_samples(os.path.join(samp_dir, un+'.wav'))
            # Test whether sampling frequencies match
            if fs_prev is not None:
                if fs_curr != fs_prev:
                    raise RuntimeError('Sampling frequency {} of {} differs from previous utterance {}'.
                                       format(fs_curr, un, fs_prev))
            # Read reference GCIs from the given directory
            if ref_gci_dir:
                utt.read_gcis(os.path.join(ref_gci_dir, un+'.pm'))
            # Append utterance
            self.append(utt)
            # Set for the next iteration
            utt_beg = utt_end
            fs_prev = fs_curr
        # Set up the total number of all examples in all utterances
        self._n_examples = utt_end

    def append(self, utt):
        """
        Append utterance to list of utterances.

        Args:
            utt (:obj:`Utt`): Utterance to append.

        """
        self._utts.append(utt)

    def __getitem__(self, idx):
        return self._utts[idx]

    def __len__(self):
        return len(self._utts)

    # Given utterance indices, return indices of examples
    def example_indices(self, utt_indices):
        """
        Given utterance indices, return indices of examples.

        Args:
            utt_indices (array-like): Indices of utterances

        Returns:
            :obj:`numpy.array`: Indices of examples

        """
        return np.hstack([self._utts[idx].indices for idx in utt_indices])

    @property
    def example_data(self):
        """Return examples data and targets from all utterances.

        Returns:
            (tuple): Two-element tuple consisting of:

            - :obj:`list`:  List of examples (:obj:`InfoArray`).

            - :obj:`list`:  List of example targets (`float`).

        """
        # data = []
        # targets = []
        # for utt in self._utts:
        #     data.extend(utt.examples)
        #     targets.extend(utt.targets)

        targets = []
        data = np.empty(self._n_examples, dtype=object)
        for utt in self._utts:
            data[utt.indices] = utt.examples
            targets.extend(utt.targets)
        return list(data), targets

    @property
    def samp_freq(self):
        """int: Sampling frequency"""
        return self._utts[0].samp_freq if self._utts else None

    @property
    def utt_indices_names(self):
        """list of tuples: List of tuples (utterance index, utterance name)."""
        return [(idx, utt.name) for idx, utt in enumerate(self._utts)]

    @property
    def n_examples(self):
        """int: Total number of examples in all utterances."""
        return self._n_examples


class CVUtt(object):
    """
    Class for cross validation in which each particular split contains all examples from given utterances.

    It is not possible for any two splits to contain examples from a single utterance. If an utterance is in a split,
    all examples are also included in the split. This requirement enables to evaluate GCI detection in an utterance-wise
    manner that is necessary for detection scores used to evaluate GCI detection accuracy, e.g., identification rate
    (IDR) - see :obj:`Scorer`.

    """
    def __init__(self, cv, utts):
        """
        Init method.

        Args:
            cv (:obj:`object`): A cross-validation object compatible with scikit-learn `splitter classes
                <http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection>`_
            utts (:obj:`Utts`): Utterance object.

        Notes:
            Due to the intermediate splitting on utterances (see :meth:`split`), it makes sense to use only
            the following scikit-learn cross validation objects as the `cv` parameter:

            - :obj:`~sklearn.model_selection.KFold`
            - :obj:`~sklearn.model_selection.LeaveOneOut`
            - :obj:`~sklearn.model_selection.LeavePOut`
            - :obj:`~sklearn.model_selection.PredefinedSplit`
            - :obj:`~sklearn.model_selection.RepeatedKFold`
            - :obj:`~sklearn.model_selection.ShuffleSplit`

        See Also:
            scikit-learn `splitter classes <http://scikit-learn.org/stable/modules/classes.\
            html#module-sklearn.model_selection>`_

        """
        self._cv = cv
        self._utt_iter = []
        self._utts = utts
        self._example_iter = []
        self._example2utt_mapping = []

    # noinspection PyPep8Naming
    # noinspection PyUnusedLocal
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations over examples in the cross-validator.

        Args:
            X (array-like): Always ignored, exists for compatibility (sklearn).
            y (array-like): Always ignored, exists for compatibility (sklearn).
            groups (array-like): Always ignored, exists for compatibility (sklearn).

        Returns:
            int: The number of splitting iterations over examples in the cross-validator.

        """
        return self._cv.get_n_splits()

    # noinspection PyPep8Naming
    # noinspection PyUnusedLocal
    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data examples into training and test set using the `split()` function of the
        :obj:`self._cv` object.

        Args:
            X (array-like): Training data of shape (n_examples, n_features), where n_samples is the number of samples
                and n_features is the number of features.
            y (array-like): The target variable of shape (n_samples,) for supervised learning problems.
            groups (array-like): Group labels with shape (n_samples,) for the examples used while splitting the dataset
                into train/test set.

        Notes:
            Parameters `y` and `groups` are used only to satisfy the split function signature as required by the sckit-
            learn splitter classes - they are not used in this split function. The split function is performed only to
            split utterances as both `y` and `groups` make no sense on utterances.

        Returns:
            list: List of two-element tuples with each tuple containing:

            - **train** (:obj:`numpy.ndarray`): The training set (example-level) indices for that split.
            - **test** (:obj:`numpy.ndarray`): The testing set (example-level) indices for that split.

        """
        # Train/test split on utterance level
        self._utt_iter = self._cv.split(range(len(self._utts)))
        # Train/test split on example level
        self._example_iter = [(self._utts.example_indices(utt_train), self._utts.example_indices(utt_test))
                              for utt_train, utt_test in self._utt_iter]
        return self._example_iter

    def example2utt_mapping(self):
        """
        For each train/test split return the mapping of each example to its source utterance.

        It makes sense to call this function after the :func:`split` method had been called. Otherwise empty list is
        returned.

        Returns:
            list: List of tuples with each tuple containing:

            - :obj:`numpy.ndarray`: Array of indices of the corresponding utterance for each training example.
            - :obj:`numpy.ndarray`: Array of indices of the corresponding utterance for each testing example.

        """
        # Return mapping examples to utterance
        return [(self._utts.utt2examples_indices(utt_train), self._utts.utt2examples_indices(utt_test))
                for utt_train, utt_test in self._utt_iter]

    @property
    def utts(self):
        """:obj:`Utts`: Utterances object"""
        return self._utts

    @utts.setter
    def utts(self, seq):
        self._utts = seq

    @property
    def example_iter(self):
        """Returns the output of the :func:`split` method."""
        return self._example_iter

    @property
    def utt_iter(self):
        """
        Return indices of training/testing utterances.

        It makes sense to call this function after the :func:`split` method had been called. Otherwise empty list is
        returned.

        Returns:
            list: List of tuples with each tuple containing:

            - :obj:`numpy.ndarray`: Array of training utterance indices.
            - :obj:`numpy.ndarray`: Array of testing utterance indices.

        """
        return self._utt_iter


class Scorer(object):
    """
    Object for evaluating GCI detection accuracy.
    
    Attributes:
        logger (logging.Logger):        Logger
        _dist_threshold (float/int):    Distance threshold (sec) for a tested GCI.
            If the tested GCI is closer than :attr:`dist_threshold` to the corresponding reference GCI, no misdetection
            is applied. If float, absolute distance in seconds is taken; otherwise (int), percentage of actual T0 is
            taken, i.e., `dist_threshold*T0`. The actual T0 is computed from reference GCIs.
        _scoring (str):         A string specifying the evaluation measure ('idr', 'far', 'mr', 'ida', 'iacc', 'acc').
        _min_t0 (float):        Minimum T0 (seconds).
        _n_refs (int):          Number of reference GCIs.
        _n_tsts (int):          Number of tested GCIs.
        _n_dels (int):          Number of deleted GCIs (GCIs occurring in tested GCIs but not in reference GCIs).
        _n_inss (int):          Number of inserted GCIs (GCIs occurring in reference GCIs but not in tested GCIs).
        _n_shfs (int):          Number of shifted GCIs (GCIs occurring in tested GCIs but with the distance to the
            nearest reference GCI out of the given limit :attr:`dist_threshold`).
        _errors (list):         list of detection errors in seconds (float)
        _utts (:obj:`Utts`):    Utterances object. Needed to evaluate detection measures utterance-wise.

    """
    logger = logging.getLogger('gci_detect.Scorer')

    def __init__(self, dist_threshold=0.00025, scoring='idr', min_t0=0.020, sync_le=0.0025, sync_ri=0.001):
        """Init method.

        Args:
            dist_threshold (float/int): Distance threshold (sec) for a tested GCI.
                If the tested GCI is closer than :attr:`dist_threshold` to the corresponding reference GCI, no
                misdetection is applied. If float, absolute distance in seconds is taken; otherwise (int), percentage of
                actual T0 is taken, i.e., `dist_threshold*T0`. The actual T0 is computed from reference GCIs.
            scoring (str):  A string specifying the evaluation measure ('idr', 'far', 'mr', 'ida', 'iacc', 'acc').
            min_t0 (float): Minimum T0 (seconds).
            sync_le (float): Time in seconds to the left for syncing a predicted GCI with a sample peak
            sync_ri (float): Time in seconds to the right for syncing a predicted GCI with a sample peak

        """
        self._scoring = scoring
        self._n_refs = 0
        self._n_tsts = 0
        self._n_dels = 0
        self._n_inss = 0
        self._n_shfs = 0
        self._errors = []
        self.clear()

        self._utts = None

        self._dist_threshold = dist_threshold
        self._min_t0 = min_t0
        self._sync_le = sync_le
        self._sync_ri = sync_ri

    def __str__(self):
        return 'Scoring function: {}'.format(self.scoring)

    def clear(self):
        """Clear the scorer.

        """
        self.logger.debug('Clearing the scorer')
        self._n_refs = 0
        self._n_tsts = 0
        self._n_dels = 0
        self._n_inss = 0
        self._n_shfs = 0
        self._errors = []

    # Value of type float is meant to express absolute distance threshold
    # Value of type int is meant to express percentual distance threshold
    @property
    def dist_threshold(self):
        """float/int: Distance threshold (seconds) for a tested GCI."""
        return self._dist_threshold

    @dist_threshold.setter
    def dist_threshold(self, value):
        if not isinstance(value, (float, int)):
            raise ValueError('Difference threshold to compare corresponding GCIs must be either int or float but is {}'.
                             format(type(value)))
        self._dist_threshold = value

    @property
    def scoring(self):
        """str: String identifying the scoring."""
        return self._scoring

    @scoring.setter
    def scoring(self, spec):
        self._scoring = spec

    @property
    def min_t0(self):
        """float: Minimum T0 (seconds)"""
        return self._min_t0

    @min_t0.setter
    def min_t0(self, value):
        if value < 0.002:
            raise ValueError('MinT0={} is probably too low'.format(value))
        elif value > 0.020:
            raise ValueError('MinT0={} is probably too high'.format(value))
        else:
            self._min_t0 = value

    # TODO: also satisfy condifition for self.scoring == 'acc'?
    def need_t0(self):
        """boolean: Whether the parameter `min_t0` is needed (it is not needed when `dist_theshold` is float. """
        return True if isinstance(self._dist_threshold, int) else False

    def compare(self, gci_refr, gci_test):
        """Compare two pitch-mark object: a tested one vs. reference one.

        Fill in auxilliary measures used for the final evaluation:

        - number of deletes (`n_dels`)
        - number of inserts (`n_inss`)
        - number of shifts (`n_shfs`)
        - number of reference GCIs (`n_refs`)
        - number of tested GCIs (`n_tsts`)

        Args:
            gci_refr (:obj:`pm.Pm`): Pitch-mark object with reference (gold-true) GCIs.
            gci_test (:obj:`pm.Pm`): Pitch-mark object with tested (predicted) GCIs.

        Returns:
            :obj:`pm_compare.PM_Compare`: Pitch-mark comparison object.

        """
        warnings.warn('Method Scorer.compare() is deprecated. Use Scorer.compare_and_accumulate() instead!',
                      DeprecationWarning)
        # print('Method Scorer.compare() is deprecated. Use Scorer.compare_and_accumulate() instead!', file=sys.stderr)
        if self.need_t0():
            if not gci_refr.get_all_pmks(pm_type_incl={OnePm.type_T}):
                self.logger.debug('Adding T-marks to reference GCIs')
                gci_refr = insert_trans_gci(gci_refr, self._min_t0)
            if not gci_test.get_all_pmks(pm_type_incl={OnePm.type_T}):
                self.logger.debug('Adding T-marks to tested GCIs')
                gci_test = insert_trans_gci(gci_test, self._min_t0)
        # Init pitch-mark comparison object
        cmp = PM_Compare(diff_t0=self._dist_threshold) if self.need_t0() else PM_Compare(diff_abs=self._dist_threshold)
        # Make the comparison
        cmp.compare_pmSeq(gci_refr, gci_test)
        # Store operations
        inss = set(x[cmp.outp_refr_pm] for x in cmp.inserted({cmp.outp_refr_pm}))
        # dels = [x[cmp.outp_test_pm] for x in cmp.tested({cmp.outp_test_pm})]
        dels = cmp.deleted()
        refs = cmp.reference(({cmp.outp_refr_pm, cmp.outp_dist_pm}))
        tsts = cmp.tested(({cmp.outp_test_pm, cmp.outp_dist_pm}))
        shfs = cmp.shifted(items={cmp.outp_refr_pm, cmp.outp_test_pm})

        self._n_refs += len(refs)
        self._n_tsts += len(tsts)
        self._n_dels += len(dels)
        self._n_inss += len(inss)
        self._n_shfs += len(shfs)

        self.logger.debug('GCI comparison results: refs={}, inserts={}, deletes={}, shifts={}'.
                          format(len(refs), len(inss), len(dels), len(shfs)))
        self.logger.debug('Shifted {}:'.format(shfs))

        if self.scoring == 'ida':
            errs = [x[cmp.outp_dist_pm] for x in refs if x[cmp.outp_refr_pm] not in set(inss)]
            self.logger.debug('Shifting errors (ms): {}'.format(np.array(errs) * 1000))
            self._errors.extend(errs)
        # return the comparison object
        return cmp

    def compare_gci(self, gci_refr, gci_test):
        """Compare two pitch-mark object: a tested one vs. reference one.

        Fill in auxilliary measures used for the final evaluation:

        - number of deletes (`n_dels`)
        - number of inserts (`n_inss`)
        - number of shifts (`n_shfs`)
        - number of reference GCIs (`n_refs`)
        - number of tested GCIs (`n_tsts`)

        Args:
            gci_refr (:obj:`pm.Pm`): Pitch-mark object with reference (gold-true) GCIs.
            gci_test (:obj:`pm.Pm`): Pitch-mark object with tested (predicted) GCIs.

        Returns:
            :obj:`pm_compare.PM_Compare`: Pitch-mark comparison object.

        """
        if self.need_t0():
            if not gci_refr.get_all_pmks(pm_type_incl={OnePm.type_T}):
                self.logger.debug('Adding T-marks to reference GCIs')
                gci_refr = insert_trans_gci(gci_refr, self._min_t0)
            if not gci_test.get_all_pmks(pm_type_incl={OnePm.type_T}):
                self.logger.debug('Adding T-marks to tested GCIs')
                gci_test = insert_trans_gci(gci_test, self._min_t0)
        # Init pitch-mark comparison object
        cmp = PM_Compare(diff_t0=self._dist_threshold) if self.need_t0() else PM_Compare(diff_abs=self._dist_threshold)
        # Make the comparison
        cmp.compare_pmSeq(gci_refr, gci_test)
        # Return the comparison object
        return cmp

    def accumulate_cmps(self, cmp):
        """Accumulate comparison measures from a comparison object :obj:`pm_compare.PM_Compare`:.

        The following measures are accumulated:

        - number of deletes (`n_dels`)
        - number of inserts (`n_inss`)
        - number of shifts (`n_shfs`)
        - number of reference GCIs (`n_refs`)
        - number of tested GCIs (`n_tsts`)

        Args:
            cmp (:obj:``pm_compare.PM_Compare``): Pitch-mark comparison object.
        """
        # Store operations
        inss = set(x[cmp.outp_refr_pm] for x in cmp.inserted({cmp.outp_refr_pm}))
        # dels = [x[cmp.outp_test_pm] for x in cmp.tested({cmp.outp_test_pm})]
        dels = cmp.deleted()
        refs = cmp.reference(({cmp.outp_refr_pm, cmp.outp_dist_pm}))
        tsts = cmp.tested(({cmp.outp_test_pm, cmp.outp_dist_pm}))
        shfs = cmp.shifted(items={cmp.outp_refr_pm, cmp.outp_test_pm})

        # Accumulate comparison measures
        self._n_refs += len(refs)
        self._n_tsts += len(tsts)
        self._n_dels += len(dels)
        self._n_inss += len(inss)
        self._n_shfs += len(shfs)

        self.logger.debug('GCI comparison results: refs={}, inserts={}, deletes={}, shifts={}'.
                          format(len(refs), len(inss), len(dels), len(shfs)))
        self.logger.debug('Shifted {}:'.format(shfs))

        if self.scoring == 'ida':
            errs = np.array([x[cmp.outp_dist_pm] for x in refs if x[cmp.outp_refr_pm] not in set(inss)
                             and x[cmp.outp_dist_pm] > -1])
            self.logger.debug('Shifting errors (ms): {}'.format(errs[errs > 0] * 1000))
            self._errors.extend(errs)

    def compare_and_accumulate(self, gci_refr, gci_test):
        """Compare two pitch-mark objects and accumulate comparison measures.

        Args:
            gci_refr (:obj:`pm.Pm`): Pitch-mark object with reference (gold-true) GCIs.
            gci_test (:obj:`pm.Pm`): Pitch-mark object with tested (predicted) GCIs.

        """
        cmp = self.compare_gci(gci_refr, gci_test)
        self.accumulate_cmps(cmp)
        return cmp

    @property
    def n_reference(self):
        """int: Number of reference GCIs"""
        return self._n_refs

    @property
    def n_tested(self):
        """int: Number of tested GCIs"""
        return self._n_tsts

    @property
    def n_deletes(self):
        """int: Number of deletes - GCIs occurring in tested GCIs but not in reference GCIs"""
        return self._n_dels

    @property
    def n_inserts(self):
        """int: Number of inserts - GCIs occurring in reference GCIs but not in tested GCIs"""
        return self._n_inss

    @property
    def n_shifts(self):
        """int: Number of shifts - GCIs occurring in tested GCIs but they distance to the nearest reference GCI is out
        of the given limit :attr:`dist_threshold` set up in the :meth:`__init__` or :meth:`dist_threshold` methods."""
        return self._n_shfs

    @property
    def n_matched(self):
        """int: Number of matched GCIs - GCIs occuring both in tested and reference GCIs regardless to the distance
        between` them"""
        return self._n_refs - self._n_inss

    # False alarm rate measure
    def false_alarm_rate_error(self):
        """False alarm rate error (FAR).

        Returns:
            float: False alarm rate error (FAR).

        """
        return float(self._n_dels) / self._n_refs

    # Miss rate
    def miss_rate_error(self):
        """Miss rate error (MR).

        Returns:
            float: Miss rate error (MR).

        """
        return float(self._n_inss) / self._n_refs

    # Identification rate
    def identification_rate_score(self):
        """Identification rate score (IDR).

        Returns:
            float: Identification rate score (IDR).

        """
        return float(self._n_refs - self._n_dels - self._n_inss) / self._n_refs

    # Identification accuracy to +/- diff_threshold (defined by self._dist_threshold, eg. 0.25 ms)
    # The percentage of detections for which the identification error x <= diff_threshold (the timing error between the
    # detected and the corresponding reference GCI)
    def identification_accuracy_score(self):
        """Identification accuracy (IACC) to +/- distance threshold :attr:`dist_threshold` (e.g., 0.00025 s)

        The percentage of detections for which the identification error `x <= dist_threshold` (the timing error between
        the tested and the corresponding reference GCI).

        Returns:
            float: Identification accuracy score (IACC)

        """
        return 1 - float(self._n_shfs)/self.n_matched

    def identification_accuracy_error(self):
        """Identification accuracy error (IDA).

        Returns:
            float: Identification accuracy error (IDA) in seconds.
        """
        if not self._errors:
            raise RuntimeError('Cannot apply IDA error measure: errors do not exist')
        return np.array(self._errors).std()

    # Accuracy score
    def accuracy_score(self):
        """Accuracy score (ACC)

        Computed as `(n_refs - n_shfs - n_inss - n_dels) / n_refs`.

        Returns:
            float: Accuracy score (ACC)

        """
        return (self._n_refs - self._n_shfs - self._n_inss - self._n_dels)/float(self._n_refs)

    def _score(self):
        """Helper score function that returns the score as required by the :attr:`_scoring` attribute.

        Notes:
            In `scikit-learn scoring <http://scikit-learn.org/stable/modules/model_evaluation.html#scoring>`_ by
            convention higher numbers are better. This is OK for measures expressing scores (such as 'idr', 'iacc',
            'acc'). For "error" or "loss" functions ('mr', 'far', 'ida'), the return value should be negated.

        Returns:
            float: The score as given by the :attr:`_scoring` attribute.

        """
        if self._scoring == 'far':
            return -self.false_alarm_rate_error()
        elif self._scoring == 'mr':
            return -self.miss_rate_error()
        elif self._scoring == 'idr':
            return self.identification_rate_score()
        elif self._scoring == 'iacc':
            return self.identification_accuracy_score()
        elif self._scoring == 'ida':
            return -self.identification_accuracy_error()
        elif self._scoring == 'acc':
            return self.accuracy_score()
        else:
            raise RuntimeError('Unsupported scoring: {}'.format(self.scoring))

    @property
    def utts(self):
        """:obj:`list`: List of utterances (:obj:`Utt`)"""
        return self._utts

    @utts.setter
    def utts(self, utts):
        self._utts = utts

    # noinspection PyPep8Naming
    # noinspection PyUnusedLocal
    def score(self, estimator, X, y):
        """Scorer callable function compatible with scikit-learn scoring function.

        Make a score for testing examples `X`. The score is defined by the scorer function :attr:`_scoring`.

        Args:
            estimator (:obj:`sklearn.base.BaseEstimator`):  estimator object implementing ‘fit’
            X (array-like):                                 The data to fit. Can be for example a list, or an array.
            y (array-like):                                 The target variable to try to predict.

        Notes:
            - No targets `y` are needed since the true GCIs, which are confronted with the predcited GCIs, are stored in
                the utterances object :attr:`_utts`.

            - `X` is a fraction that corresponds to the given split (fold) of the original data examples `X` as
                given by the cross-validation object used. As the data examples are inputted from a sckikit-learn code,
                the particular fold to be used is not known, only the data is available.
                The :meth:`_indices_examples2utt` is used to find out the mapping between the data examples and
                utterances in the partiular split `X`.

            - Although the custom cross validation :obj:`KFoldUtt` returns the mapping of each example to an utterance
                (see :meth:`KFoldUtt.example2utt_mapping`), the actual split (fold) from which the testing data examples
                `X` were picked, are not known!

        Warnings:
            The scorer function must have signature ``scorer(estimator, X, y)``.

        Returns:
            float: The detection accuracy score (or error, loss).

        """
        # Reset scorer for this testing set
        self.clear()
        # Map the testing data examples X to utterances
        ex2utt = self._indices_examples2utt(X)
        # !!!
        # Convert from list of InfoArray to Numpy array to enable "multiindexing". We are loosing the information about
        # the utterance -> example mapping by this operation, so it must be called after _indices_examples2utt() had
        # been called.
        # !!!
        X = np.array(X)
        self.logger.debug('Scoring for {}'.format(estimator))
        self.logger.debug('Scoring on set with {} utts {} and {} examples'.format(len(ex2utt),
                                                                                  list(ex2utt.keys()),
                                                                                  len(X)))
        self.logger.debug('Sampling frequency used for syncing predicted GCI positions: {}'.
                          format(self._utts.samp_freq))
        # Sampling frequency should be the same for all utterances in the dataset
        sync_le = seconds2samples(self._sync_le, self._utts.samp_freq)
        sync_ri = seconds2samples(self._sync_ri, self._utts.samp_freq)
        # Go through all utterances in this testing set and accumulate comparison indicators across the utterances
        for utt_idx, ex_indices in ex2utt.items():
            utt = self._utts[utt_idx]
            self.logger.debug('Scoring on {} examples from {} ({})'.format(len(ex_indices), utt.name, utt_idx))
            # Predict and sync GCIs for the given utterance
            pred_sync = sync_gci_to_samp_peak(predict_gci(estimator, X[np.array(ex_indices)]),
                                              utt.peaks,
                                              utt.samples,
                                              sync_le,
                                              sync_ri)
            # Set the predicted and synced GCIs as a Pm object
            # gci_pred = create_gci(samples2seconds(pred_sync, utt.samp_freq))
            gci_pred = Pm(times=samples2seconds(pred_sync, utt.samp_freq))
            # Compare GCI sequences: predicted GCIs with true GCIs (represented as a Pm object from the utt object)
            self.compare_and_accumulate(utt.ref_gcis, gci_pred)
        score = self._score()
        self.logger.debug('Score ({}) = {:8.6f} on set of {} utts {} with {} examples'.
                          format(self._scoring.upper(), score, len(ex2utt), np.array(list(ex2utt.keys())), len(X)))
        return score

    # noinspection PyPep8Naming
    @staticmethod
    def _indices_examples2utt(X):
        """Make a mapping between input testing examples and the corresponding source utterances.

        The input examples have to a list of :obj:`InfoArray` since each example ha to conatain the
        :attr:`InfoArray.info` attribute that denotes from which source utterance (given by an index) the example
        comes from.

        Args:
            X (:obj:`list`): The data to map (list of :obj:`InfoArray`).

        Returns
            dict: Dictionary with list of example indices from X for each utterance index.

        """
        d = defaultdict(list)
        for ex_idx, ex in enumerate(X):
            d[ex.info].append(ex_idx)
        return d


# noinspection PyPep8Naming
def predict_gci(estimator, X):
    """
    Predict GCI placements in seconds.

    Call the :func:`sync_gci_to_samp_peak` to sync the time placements with speech signal.

    Args:
        estimator (:obj:`sklearn.base.BaseEstimator`): Sklearn-style estimator object implementing 'fit'.
        X (array-like): The data to predict on.

    Returns:
        :obj:`numpy.array`: Array of GCI time placements (int) in samples.

    """
    return estimator.predict(X)
