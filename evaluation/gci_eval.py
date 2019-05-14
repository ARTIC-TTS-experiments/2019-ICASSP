#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import logging
from argparse import ArgumentParser
from glob import glob
from itertools import zip_longest
import os.path as osp
import pandas as pd
from gci_detect_clf import Scorer
from pm import Pm


# -------------------------------------------------------------------------------
def main():
    # Command line processing
    parser = ArgumentParser(description="Evaluate two directories with GCI files")
    parser.add_argument('ref',
                        help='input directory with reference GCI files (PM format) [default: %(default)s]')
    parser.add_argument('tst',
                        help='input directory with tested GCI files (PM format) [default: %(default)s]')
    parser.add_argument('-d', '--abs-dist',
                        default=0.00025,
                        type=float,
                        help='distance threshold (msec) to detect shifted GCIs [default: %(default)s]')
    parser.add_argument('-D', '--rel-dist',
                        default=10,
                        type=int,
                        help='distance threshold (integer percentage within current T0) to detect shifted GCIs'
                             '[default: %(default)s]')
    parser.add_argument('-T', '--min-T0',
                        default=0.020,
                        type=float,
                        help='minimum T0 for relative-distance based comparison [default: %(default)s]')
    parser.add_argument('-L', '--loglevel',
                        default='INFO',
                        help='logging level [default: %(default)s]')
    args = parser.parse_args()

    # ====================== Logging ======================
    logger = logging.getLogger('gci_eval')
    loglevel = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(loglevel, int):
        raise ValueError('Invalid log level: {}'.format(loglevel))
    logging.basicConfig(format='%(levelname)-10s %(message)s', level=loglevel, stream=sys.stderr)

    df = pd.DataFrame()

    # Global scorers for all input GCI files:
    # - gscorer_rel is for relative-distance based comprison (percentage of current T0)
    # - gscorer_abs is for absolute-distance based comprison (sec)
    # IDA scoring is used for only one scorer as it is needed just once to compute individual errors
    gscorer_rel = Scorer(dist_threshold=args.rel_dist, scoring='ida', min_t0=args.min_T0)
    gscorer_abs = Scorer(dist_threshold=args.abs_dist, scoring='idr')

    # Iterate over the corresponding files in tow directories
    for fn_ref, fn_tst in zip_longest(sorted(glob(osp.join(args.ref, '*.pm'))),
                                      sorted(glob(osp.join(args.tst, '*.pm')))):
        logger.debug('Comparing: {} vs. {}'.format(fn_ref, fn_tst))
        # Check filenames
        if None in (fn_ref, fn_tst):
            raise RuntimeError('The number of files in the two directories differs!')
        bn_ref = osp.splitext(osp.basename(fn_ref))[0]
        bn_tst = osp.splitext(osp.basename(fn_tst))[0]
        if bn_ref != bn_tst:
            raise RuntimeError('The GCI files have different names: {} vs {}!'.format(bn_ref, bn_tst))

        pm_ref = Pm(fn_ref)
        pm_tst = Pm(fn_tst)

        # Local scorers (scoring for a single GCI pair comparison
        lscorer_rel = Scorer(dist_threshold=args.rel_dist, scoring='ida', min_t0=args.min_T0)
        lscorer_abs = Scorer(dist_threshold=args.abs_dist, scoring='idr')

        # Compare two corresponding GCI files and accumulate the comparison measures
        gscorer_rel.accumulate_cmps(lscorer_rel.compare_and_accumulate(pm_ref, pm_tst))
        gscorer_abs.accumulate_cmps(lscorer_abs.compare_and_accumulate(pm_ref, pm_tst))

        # Get local results for the concrete comparison and store them to dataframe
        df.loc[bn_ref, '#Reference'] = lscorer_rel.n_reference
        df.loc[bn_ref, '#Tested'] = lscorer_rel.n_tested
        df.loc[bn_ref, '#Deletes'] = lscorer_rel.n_deletes
        df.loc[bn_ref, '#Inserts'] = lscorer_rel.n_inserts
        df.loc[bn_ref, '#Shifts'] = lscorer_rel.n_shifts
        df.loc[bn_ref, '#Matched'] = lscorer_rel.n_matched
        df.loc[bn_ref, 'IDR'] = lscorer_rel.identification_rate_score()
        df.loc[bn_ref, 'MR'] = lscorer_rel.miss_rate_error()
        df.loc[bn_ref, 'FAR'] = lscorer_rel.false_alarm_rate_error()
        df.loc[bn_ref, 'IDA'] = lscorer_rel.identification_accuracy_error()
        df.loc[bn_ref, 'iACC{}'.format(args.abs_dist)] = lscorer_abs.identification_accuracy_score()

        logger.info('{:20}: IDR = {:.2%}'.format(bn_ref, lscorer_rel.identification_rate_score()))

    # Get global results for all comparisons and store them to dataframe
    df.loc['TOTAL', '#Reference'] = gscorer_rel.n_reference
    df.loc['TOTAL', '#Tested'] = gscorer_rel.n_tested
    df.loc['TOTAL', '#Deletes'] = gscorer_rel.n_deletes
    df.loc['TOTAL', '#Inserts'] = gscorer_rel.n_inserts
    df.loc['TOTAL', '#Shifts'] = gscorer_rel.n_shifts
    df.loc['TOTAL', '#Matched'] = gscorer_rel.n_matched
    df.loc['TOTAL', 'IDR'] = gscorer_rel.identification_rate_score()
    df.loc['TOTAL', 'MR'] = gscorer_rel.miss_rate_error()
    df.loc['TOTAL', 'FAR'] = gscorer_rel.false_alarm_rate_error()
    df.loc['TOTAL', 'IDA'] = gscorer_rel.identification_accuracy_error()
    df.loc['TOTAL', 'iACC{}'.format(args.abs_dist)] = gscorer_abs.identification_accuracy_score()

    logger.info('{:20}: IDR = {:.2%}'.format('TOTAL', gscorer_rel.identification_rate_score()))

    # Print resulting evaluation to stdout
    print(df.to_csv(sep=';', index=True))


# -------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
