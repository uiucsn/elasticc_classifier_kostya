#!/usr/bin/env python3

import dataclasses
import glob
import os
from argparse import ArgumentParser, ArgumentError
from collections import defaultdict
from functools import partial
from itertools import chain, starmap
from multiprocessing import Pool

import lcdata
import numpy as np
import sncosmo
from astropy.io import ascii
from astropy.table import Table


def parse_fits_snana(head, phot, *, z_keyword, min_passbands):
    i = head.find('_HEAD.FITS.gz')
    assert head[:i] == phot[:i], f'HEAD and PHOT files name mismatch: {head}, {phot}'
    
    lcs = []
    for lc in sncosmo.read_snana_fits(head, phot):
        lc.meta['redshift'] = lc.meta[z_keyword]
        if min_passbands > 1:
            # we use this variable for cuts only, while putting the full light curve into dataset
            detections = lc[lc['PHOTFLAG'] != 0]
            assert len(detections) != 0, f'No detection in the light curve: {head = } {phot = } {lc.meta = }'
            if len(set(detections['BAND'])) < min_passbands:
                continue
        lc['BAND'] = ['lsst' + band[0].lower() for band in lc['BAND']]
        lcs.append(lc)
    
    return lcs


def create_dataset_from_snana_fits(dir_path, *, z_keyword, min_passbands, parallel):
    heads = sorted(glob.glob(os.path.join(dir_path, '*_HEAD.FITS.gz')))
    phots = sorted(glob.glob(os.path.join(dir_path, '*_PHOT.FITS.gz')))
    assert len(heads) != 0, 'no *_HEAD_FITS.gz are found'
    assert len(heads) == len(phots), 'there are different number of HEAD and PHOT files'

    parse = partial(parse_fits_snana, z_keyword=z_keyword, min_passbands=min_passbands)
    if parallel:
        with Pool() as pool:
            lcs = pool.starmap(parse, zip(heads, phots), chunksize=1)
    else:
        lcs = starmap(parse, zip(heads, phots))
    # lcs is a list of lists
    lcs = list(chain.from_iterable(lcs))

    dataset = lcdata.from_light_curves(lcs)
    return dataset


def create_dataset(dir_path, *, count, z_keyword, z_std, fixed_z, min_passbands, obj_type, parallel):
    dataset = create_dataset_from_snana_fits(
        dir_path,
        z_keyword=z_keyword,
        min_passbands=min_passbands,
        parallel=parallel,
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


def main(argv=None):
    args = parse_args(argv)
    
    dataset = create_dataset(args.input, count=args.count,
                             z_keyword=args.z_keyword, z_std=args.z_std,
                             fixed_z=args.fixed_z, min_passbands=args.min_passbands,
                             obj_type=args.type, parallel=args.parallel)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    dataset.write_hdf5(args.output)
    print(f'Dataset ({len(dataset)} objects) has been written into {args.output}')


if __name__ == '__main__':
    main()
