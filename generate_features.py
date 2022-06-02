#!/usr/bin/env python3

import re
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import lcdata
import light_curve as licu
import numpy as np
import parsnip
from astropy.coordinates import SkyCoord
from astropy.table import Table


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
    # 'luminosity',
    # 'luminosity_error',
    'reference_time_error',
]


SNANA_TO_TAXONOMY = {
    'AGN': 'AGN',
    'CART': 'CART',
    'Cepheid': 'Cepheid',
    'd-Sct': 'Delta Scuti',
    'dwarf-nova': 'Dwarf Novae',
    'EB': 'EB',
    'ILOT': 'ILOT',
    'KN_B19': 'KN',
    'KN_K17': 'KN',
    'Mdwarf-flare': 'M-dwarf Flare',
    'PISN': 'PISN',
    'RRL': 'RR Lyrae',
    'SLSN-I+host': 'SLSN',
    'SLSN-I_no_host': 'SLSN',
    'SNIa-91bg': 'Ia',
    'SNIa-SALT2': 'Ia',
    'SNIax': 'Iax',
    'SNIb+HostXT_V19': 'Ib/c',
    'SNIb-Templates': 'Ib/c',
    'SNIcBL+HostXT_V19': 'Ib/c',
    'SNIc+HostXT_V19': 'Ib/c',
    'SNIc-Templates': 'Ic',
    'SNIIb+HostXT_V19': 'II',
    'SNII+HostXT_V19': 'II',
    'SNIIn+HostXT_V19': 'II',
    'SNII-NMF': 'II',
    'SNIIn-MOSFIT': 'II',
    'SNII-Templates': 'II',
    'TDE': 'TDE',
    'uLens-Binary': 'uLens',
    'uLens-Single-GenLens': 'uLens',
    'uLens-Single_PyLIMA': 'uLens',
}


BANDS = ['lsst' + band for band in 'ugrizy']
LEN_BANDS = len(BANDS)

MAGERR_COEFF = 2.5 / np.log(10.0)

MJD0 = 60000.0


class MetaExtractor():
    features = ['abs_gal_b', 'redshift']
    size = len(features)

    _redshift_threshold = 0.01

    def _prepare(self, coord: SkyCoord, redshift: np.ndarray):
        abs_gal_b = np.abs(coord.galactic.b.deg)
        return dict(abs_gal_b=abs_gal_b, redshift=redshift)

    def _extract_lcdata(self, meta: Table):
        coord = SkyCoord(ra=meta['ra'], dec=meta['dec'], unit='deg')
        redshift = meta['redshift']
        return self._prepare(coord=coord, redshift=redshift)

    def __call__(self, meta: Table, *, schema:str):
        try:
            feature_dict = getattr(self, f'_extract_{schema}')(meta)
        except AttributeError as e:
            raise ValueError(f'schema {schema!r} is not supported') from e
        table = Table({k: feature_dict[k] for k in self.features})
        table['redshift'] = np.where(table['redshift'] >= self._redshift_threshold, table['redshift'], np.nan)
        return table


@dataclass
class LcExtractor():
    s2n: float = 5.0

    mag_extractor = licu.Extractor(
        licu.AndersonDarlingNormal(),
        licu.Bins(
            features=[
                licu.BeyondNStd(1.0),
                licu.BeyondNStd(2.0),
                licu.EtaE(),
                licu.Kurtosis(),
                licu.MaximumSlope(),
                licu.MinimumTimeInterval(),
                licu.ObservationCount(),
                licu.Skew(),
            ],
            window=1.0,
            offset=0.0,
        ),
        licu.Duration(),
        licu.InterPercentileRange(0.01),
        licu.LinearFit(),
        licu.MaximumTimeInterval(),
        licu.ObservationCount(),
        licu.Periodogram(
            peaks=1,
            resolution=10,
            max_freq_factor=2,
            nyquist='median',
            fast=True,
            features=[
                licu.Median(),
                licu.PercentDifferenceMagnitudePercentile(0.25)
            ],
        ),
        licu.ReducedChi2(),
        licu.StetsonK(),
        licu.WeightedMean(),
    )

    flux_extractor = licu.Extractor(
        licu.BazinFit('mcmc-lmsder', mcmc_niter=1 << 13, lmsder_niter=20),
        licu.VillarFit('mcmc-lmsder', mcmc_niter=1 << 13, lmsder_niter=20),
    )
    
    def __post_init__(self):
        self.mag_names = [f'mag_{name}_{band}' for band in BANDS for name in self.mag_extractor.names]
        self.mag_size = len(self.mag_names)
        self.flux_names = [f'flux_{name}_{band}' for band in BANDS for name in self.flux_extractor.names]
        self.flux_size = len(self.flux_names)
        self.names = self.mag_names + self.flux_names
        self.size = len(self.names)
        assert self.mag_size + self.flux_size == self.size

    def prepare_lc_lcdata(self, lc: Table) -> Table:
        lc = lc[lc['flux'] / lc['fluxerr'] > self.s2n]
        lc['time'] = np.asarray(lc['time'] - MJD0, dtype=np.float32)
        lc['mag'] = -2.5 * np.log10(lc['flux'])
        lc['magerr'] = MAGERR_COEFF * lc['flux'] / lc['fluxerr']
        return lc

    def _prepare_lc_funcs(self, schema: str):
        try:
            return getattr(self, f'prepare_lc_{schema}')
        except AttributeError as e:
            raise ValueError(f'schema {schema!r} is not supported') from e

    @staticmethod
    def lcs_mag_list_tuples(lcs: Iterable[Table]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        return [(lc['time'], lc['mag'], lc['magerr']) for lc in lcs]
    
    @staticmethod
    def lcs_flux_list_tuples(lcs: Iterable[Table]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        return [(lc['time'], lc['flux'], lc['fluxerr']) for lc in lcs]
    
    def __call__(self, light_curves: Sequence[Table],
                 *, schema: str, chunk_size: int = 1 << 14, n_jobs: int = -1, out=None) -> np.ndarray:
        prepare_lc_func = self._prepare_lc_funcs(schema)
        n_light_curves = len(light_curves)
        if out is None:
            out = np.empty((n_light_curves, self.size), dtype=np.float32)
        else:
            assert out.shape == (n_light_curves, self.size)
            assert out.dtype is np.float32
        for i in range(0, n_light_curves, chunk_size):
            lc_idx = slice(i, i + chunk_size)
            # Split each light curve to len(BANDS) light curves
            lcs = list(lc[lc['band'] == band]
                       for lc in map(prepare_lc_func, light_curves[lc_idx])
                       for band in BANDS)
            # We are reshaping from (n_lc * len(BANDS), {mag,flux}_features.size) to (n_lc, self.{mag,flux}_size)
            out[lc_idx, :self.mag_size] = self.mag_extractor.many(
                self.lcs_mag_list_tuples(lcs),
                fill_value=np.nan,
                n_jobs=n_jobs,
                sorted=True,
                check=False,
            ).reshape(-1, self.mag_size)
            out[lc_idx, self.mag_size:] = self.flux_extractor.many(
                self.lcs_flux_list_tuples(lcs),
                fill_value=np.nan,
                n_jobs=n_jobs,
                sorted=True,
                check=False,
            ).reshape(-1, self.flux_size)
        return out
    

def parse_args(argv=None):
    parser = ArgumentParser("Generate ParSNIP features for the dataset")
    parser.add_argument('-i', '--input', required=True,
                        help='HDF5 file containing lcdata Dataset to extract features, may be used multiple times')
    parser.add_argument('-m', '--model', required=True, help='ParSNIP model file to use')
    parser.add_argument('-o', '--output', default='features', help='output directory')
    parser.add_argument('--device', default='cuda', help='PyTroch device')
    parser.add_argument('--s2n', default=LcExtractor.s2n, type=float, help='S/N threshold for light-curve features')
    args = parser.parse_args(argv)
    return args


def filename_to_snana(fname):
    s = Path(fname).stem
    s = re.sub(r'_count\d+', '', s)
    s = re.sub(r'_z[\d.]+', '', s)
    return s


def main(argv=None):
    args = parse_args(argv)

    dataset = lcdata.read_hdf5(args.input)

    lc_extractor = LcExtractor(s2n=args.s2n)
    meta_extractor = MetaExtractor()

    lc_idx = slice(0, lc_extractor.size)
    parsnip_idx = slice(lc_idx.stop, lc_idx.stop + len(PARSNIP_FEATURES))
    meta_idx = slice(parsnip_idx.stop, parsnip_idx.stop + meta_extractor.size)
    all_size = meta_idx.stop
    all_features = np.empty((len(dataset), all_size), dtype=np.float32)

    lc_features = all_features[:, lc_idx]
    lc_extractor(dataset.light_curves, schema='lcdata', out=lc_features)

    parsnip_model = parsnip.load_model(args.model, device=args.device)
    predictions = parsnip_model.predict_dataset(dataset)
    parsnip_features = all_features[:, parsnip_idx]
    parsnip_features[:] = np.stack([np.asarray(predictions[f], dtype=np.float32) for f in PARSNIP_FEATURES], axis=-1)

    meta_features = all_features[:, meta_idx]
    meta_table = meta_extractor(dataset.meta, schema='lcdata')
    meta_features[:] = np.stack([np.asarray(column, dtype=np.float32) for column in meta_table.itercols()], axis=-1)

    object_ids = predictions['object_id']
    snana_model_name = filename_to_snana(args.input)
    types = np.full(len(dataset), SNANA_TO_TAXONOMY[snana_model_name])

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    np.save(output_dir.joinpath(f'{snana_model_name}_features.npy'), all_features)
    np.save(output_dir.joinpath(f'{snana_model_name}_types.npy'), types)
    np.save(output_dir.joinpath(f'{snana_model_name}_ids.npy'), object_ids)


if __name__ == '__main__':
    main()
