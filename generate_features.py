#!/usr/bin/env python3

import light_curve as lc
import operator
import os
from argparse import ArgumentParser
from dataclasses import dataclass
from functools import reduce
from itertools import chain

import lcdata
import parsnip
from astropy.io import ascii


# From the ParSNIP paper
PARSNIP_FEATURES = [
    'color',
    'color_error',
    's1',
    's1_error',
    's2',
    's2_error',
    's3',
    's3_error',
    'luminosity',
    'luminosity_error',
    'reference_time_error',
]


BANDS = ['lsst' + band for band in 'ugrizy']
LEN_BANDS = len(BANDS)

MAGERR_COEFF = 2.5 / np.log(10.0)

MJD0 = 60000.0


@dataclass
class LcExtractor():
    s2n: float

    mag_extractor: lc.Extractor = lc.Extractor([
        lc.AndersonDarlingNormal(),
        lc.Bins(
            features=[
                lc.BeyondNStd(1.0),
                lc.BeyondNStd(2.0),
                lc.EtaE(),
                lc.Kurtosis(),
                lc.MaximumSlope(),
                lc.MinimumTimeInterval(),
                lc.ObservationCount(),
                lc.Skew(),
            ],
            window=1.0,
            offset=0.0,
        ),
        lc.Duration(),
        lc.InterPercentileRange(0.01),
        lc.LinearFit(),
        lc.MaximumTimeInterval(),
        lc.ObservationCount(),
        lc.Periodogram(
            peaks=1,
            resolution=10,
            max_freq_factor=2,
            nyquist='median',
            fast=True,
            features=[
                lc.Median(),
                lc.PercentDifferenceMagnitudePercentile(0.25)
            ],
        ),
        lc.ReducedChi2(),
        lc.StetsonK(),
        lc.WeightedMean(),
    ])

    flux_extractor: lc.Extractor = lc.Extractor([
        lc.BazinFit('mcmc-lmsder', mcmc_niter=1<<13, lmsder_niter=20),
        lc.VillarFit('mcmc-lmsder', mcmc_niter=1<<13, lmsder_niter=20),
    ])
    
    def __post_init__(self):
        self.mag_names = [f'mag_{name}_{band}' for band in BANDS for name in self.mag_extractor.names]
        self.mag_size = len(mag_names)
        self.flux_names = [f'flux_{name}_{band}' for band in BANDS for name in self.flux_extractor.names]
        self.flux_size = len(flux_names)
        self.names = self.mag_names + self.flux_names
        self.size = len(self.names)
    
    def prepare_lc(self, lc):
        lc = lc[lc['flux'] / lc['fluxerr'] > self.s2n]
        lc['t_f32'] = np.asarray(lc['t'] - MJD0, dtype=np.float32)
        lc['mag'] = -2.5 * np.log10(lc['flux'])
        lc['magerr'] = MAGERR_COEFF * lc['flux'] / lc['fluxerr']
        return [lc[lc['band'] == band] for band in BANDS]
    
    @staticmethod
    def lcs_mag_list_tuples(lcs):
        return [(lc['t_f32'], lc['mag'], lc['magerr']) for lc in lcs]
    
    @staticmethod
    def lcs_flux_list_tuples(lcs):
        return [(lc['t_f32'], lc['flux'], lc['fluxerr']) for lc in lcs]
    
    def __call__(self, light_curves, *, chunk_size=1<<14, n_jobs=-1):
        n_light_curves = len(light_curves)
        features = np.empty((n_light_curves, self.size), dtype=np.float32)
        for i in range(0, n_light_curves, chunk_size):
            lc_idx = slice(i, i + chunk_size)
            lcs = list(chain.from_iterable(prepare_lc[lc] for light_curves[lc_idx]))
            features[lc_idx,0:self.mag_size] = self.mag_extractor.many(
                self.lcs_mag_list_tuples(lcs),
                fill_value=np.nan,
                n_jobs=n_jobs,
                sorted=True,
                check=False,
            ).reshape(-1, self.mag_size)
            features[lc_idx,self.mag_size:self.size] = self.mag_extractor.many(
                self.lcs_flux_list_tuples(lcs),
                fill_value=np.nan,
                n_jobs=n_jobs,
                sorted=True,
                check=False,
            ).reshape(-1, self.flux_size)
        return features
    

def parse_args(argv=None):
    parser = ArgumentParser("Generate ParSNIP features for the dataset")
    parser.add_argument('-i', '--input', required=True, action='append',
                        help='HDF5 file containing lcdata Dataset to extract features, may be used multiple times')
    parser.add_argument('-m', '--model', required=True, help='ParSNIP model file to use')
    parser.add_argument('-o', '--output', required=True, help='output CSV filename')
    parser.add_argument('--device', default='cuda', help='PyTroch device')
    parser.add_argument('--s2n', default=5.0, type=float, help='S/N threshold for light-curve features')
    args = parser.parse_args(argv)
    return args


def main(argv=None):
    args = parse_args(argv)
    dataset = reduce(operator.add, (lcdata.read_hdf5(file) for file in args.input))
    model = parsnip.load_model(args.model, device=args.device)
    predictions = model.predict_dataset(dataset)
    avail_features = [f for f in FEATURES if f in predictions.columns]
    features = predictions[['object_id'] + avail_features]
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    ascii.write(features, args.output, format='csv', delimiter=',')


if __name__ == '__main__':
    main()
