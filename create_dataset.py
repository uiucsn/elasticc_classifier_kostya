#!/usr/bin/env python3

import glob
import os
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from functools import partial
from itertools import chain, starmap
from multiprocessing import Pool
from typing import Generator

import lcdata
import numpy as np
import sncosmo
from astropy.io import ascii
from astropy.table import Table


def get_detection_mask(lc: Table) -> np.ndarray:
    """Get mask for detections which are not saturated"""
    mask_detection = np.bitwise_and(lc['PHOTFLAG'], 4096) != 0
    mask_not_saturated = np.bitwise_and(lc['PHOTFLAG'], 1024) == 0
    return mask_detection & mask_not_saturated

    
def get_detections(lc: Table) -> Table:
    """Get detections which are not saturated"""
    return lc[get_detection_mask(lc)]


def parse_fits_snana(head, phot, *, z_keyword, min_passbands):
    i = head.find('_HEAD.FITS.gz')
    assert head[:i] == phot[:i], f'HEAD and PHOT files name mismatch: {head}, {phot}'
    
    lcs = []
    for lc in sncosmo.read_snana_fits(head, phot):
        lc.meta['redshift'] = lc.meta[z_keyword]
        if min_passbands > 1:
            # we use this variable for cuts only, while putting the full light curve into dataset
            detections = get_detections(lc)
            assert len(detections) != 0, f'No detection in the light curve: {head = } {phot = } {lc.meta = }'
            if len(set(detections['BAND'])) < min_passbands:
                continue
        lc['BAND'] = ['lsst' + band[0].lower() for band in lc['BAND']]
        lcs.append(lc)
    
    return lcs


class PreProcessing(ABC):
    @abstractmethod
    def __call__(self, lc: Table) -> Table:
        raise NotImplementedError()


class NoPreProcessing(PreProcessing):
    def __call__(self, lc: Table) -> Table:
        return lc


class DropPreTrigger(PreProcessing):
    def __init__(self, days_before):
        self.days_before = days_before

    def __call__(self, lc: Table) -> Table:
        detections = get_detections(lc)
        earliest_mjd = detections['MJD'][0] - self.days_before
        if earliest_mjd <= lc['MJD'].min():
            return lc
        return lc[lc['MJD'] >= earliest_mjd]


class BasicLightCurveSplitter(ABC):
    @abstractmethod
    def __call__(self, lc: Table) -> Generator[Table, None, None]:
        raise NotImplementedError

        
class NoSplitSplitter(BasicLightCurveSplitter):
    def __call__(self, lc: Table) -> Generator[Table, None, None]:
        yield lc
        

class LastDetectionSplitter(BasicLightCurveSplitter):
    def __call__(self, lc: Table) -> Generator[Table, None, None]:
        last_detection = np.where(get_detection_mask(lc))[0][-1]
        yield lc[:last_detection]

        
class RandomDetectionSplitter(BasicLightCurveSplitter):
    def __init__(self, prob: float):
        self.prob = prob

    @staticmethod
    def _rng_from_lc(lc: Table) -> np.random._generator.Generator:
        first_flux = lc['FLUXCAL'].data[:1]
        first_flux.dtype = np.uint32
        random_seed = first_flux.item()
        return np.random.default_rng(random_seed)

    def __call__(self, lc: Table) -> Generator[Table, None, None]:
        det_idx = np.where(get_detection_mask(lc))[0]
        n_det = det_idx.size
        rng = self._rng_from_lc(lc)
        accepted_mask = rng.random(n_det) < self.prob
        for i_det in det_idx[accepted_mask]:
            yield lc[:i_det + 1]
            

class RandomAndLastDetectionsSplitter(BasicLightCurveSplitter):
    def __init__(self, prob: float):
        self.last_detection_splitter = LastDetectionSplitter()
        self.random_detection_splitter = RandomDetectionSplitter(prob=prob)

    def __call__(self, lc: Table) -> Generator[Table, None, None]:
        last_detection_lc = next(self.last_detection_splitter(lc))
        yield last_detection_lc
        longest_nobs = len(last_detection_lc)
        for random_detection_lc in self.random_detection_splitter(lc):
            # We don't want to repeat last_detection_lc
            if len(random_detection_lc) == longest_nobs:
                continue
            yield random_detection_lc


def parse_and_split(*parser_args, parser, splitter, preprocsessing):
    def gen():
        for lc in parser(*parser_args):
            for pre_fn in preprocsessing:
                lc = pre_fn(lc)
            yield from splitter(lc)
    
    return list(gen())
            

def create_dataset_from_snana_fits(dir_path, *, z_keyword, min_passbands, parallel, splitter, preprocsessing):
    heads = sorted(glob.glob(os.path.join(dir_path, '*_HEAD.FITS.gz')))
    phots = sorted(glob.glob(os.path.join(dir_path, '*_PHOT.FITS.gz')))
    assert len(heads) != 0, 'no *_HEAD_FITS.gz are found'
    assert len(heads) == len(phots), 'there are different number of HEAD and PHOT files'

    parser = partial(parse_fits_snana, z_keyword=z_keyword, min_passbands=min_passbands)
    worker = partial(parse_and_split, parser=parser, splitter=splitter, preprocsessing=preprocsessing)
    if parallel:
        with Pool() as pool:
            lcs = pool.starmap(worker, zip(heads, phots), chunksize=1)
    else:
        lcs = starmap(worker, zip(heads, phots))
    # lcs is a list of lists
    lcs = list(chain.from_iterable(lcs))

    dataset = lcdata.from_light_curves(lcs)
    return dataset


def create_dataset(dir_path, *,
                   count, z_keyword, z_std, fixed_z, min_passbands, obj_type, parallel, splitter, preprocsessing):
    dataset = create_dataset_from_snana_fits(
        dir_path,
        z_keyword=z_keyword,
        min_passbands=min_passbands,
        parallel=parallel,
        splitter=splitter,
        preprocsessing=preprocsessing,
    )
    if 'type' not in dataset.meta.columns:
        if obj_type is None:
            # https://github.com/kboone/parsnip/pull/3
            obj_type = 'unknown'
        dataset.meta['type'] = np.full(len(dataset.meta), obj_type)
    
    if count is not None:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(dataset), count, replace=False)
        dataset = dataset[idx]
        
    if fixed_z is not None:
        dataset.meta['redshift'] = fixed_z
    
    # Truncated normal distribution
    # Replace with lognorm with z_std^2 variance?
    if z_std is not None:
        rng = np.random.default_rng(0)
        z = np.full(len(dataset), -1.0)
        while np.any(z < 0):
            idx = z < 0
            z[idx] = rng.normal(dataset.meta['redshift'][idx], z_std)
        dataset.meta['redshift'] = z
    
    return dataset


def parse_args(argv=None):
    parser = ArgumentParser("Convert light curve data from SNANA format to lcdata Datasets")
    parser.add_argument('-i', '--input', required=True, help='folder containing light curves in SNANA format')
    parser.add_argument('-n', '--count', default=None, type=int, help="select given numbe of objects randomly without replacement, default is using all valid data (note that this parameter doesn't reduce evaluation time)")
    parser.add_argument('-t', '--type', default=None, help='replace object type with a given value')
    parser.add_argument('-o', '--output', required=True, help='output HDF5 filename')
    parser.add_argument('--split-prob', default=None, type=float, help='get light curve data up to given detection with some probability, if not specified the full dataset is used')
    parser.add_argument('--include-last-detection', action='store_true', help='include last detection light curve for when --split-prob specified')
    parser.add_argument('--days-before-trigger', default=np.inf, type=float, help='range of dates to include non-detections before the first detection, use 30 for Elasticc')
    parser.add_argument('--z-keyword', default='REDSHIFT_HELIO', help='redshift keyword')
    parser.add_argument('--fixed-z', default=None, type=float, help='use the given redshift value')
    parser.add_argument('--z-std', default=None, type=float,
                        help='randomize redshift adding normally distributed value with given standard deviation')
    parser.add_argument('--min-passbands', default=1, type=int,
                        help='cut out light curves having less than given number of passbands')
    parser.add_argument('--parallel', action='store_true', help='process in parallel')
    args = parser.parse_args(argv)
    return args


def removeprefix(s, prefix):
    """"Remove string prefix
    
    Like str.removeprefix() but with assert
    """
    assert s.startswith(prefix)
    return s[len(prefix):]


def get_redshifts(file, column):
    table = ascii.read(file, format='csv')
    redshifts = {obj: z for obj, z in table[['TransientName', column]] if z}
    return redshifts


def splitter_from_args(args) -> BasicLightCurveSplitter:
    if args.split_prob is None:
        return NoSplitSplitter()
    if args.include_last_detection:
        type_ = RandomAndLastDetectionsSplitter
    else:
        type_ = RandomDetectionSplitter
    return type_(prob=args.split_prob)



def main(argv=None):
    args = parse_args(argv)
    
    dataset = create_dataset(args.input, count=args.count,
                             z_keyword=args.z_keyword, z_std=args.z_std,
                             fixed_z=args.fixed_z, min_passbands=args.min_passbands,
                             obj_type=args.type, parallel=args.parallel,
                             splitter=splitter_from_args(args),
                             preprocsessing=[DropPreTrigger(args.days_before_trigger)])
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    dataset.write_hdf5(args.output)
    print(f'Dataset ({len(dataset)} objects) has been written into {args.output}')


if __name__ == '__main__':
    main()
