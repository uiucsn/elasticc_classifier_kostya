#!/usr/bin/env python3

import logging
from argparse import ArgumentParser
from collections import Counter
from functools import lru_cache
from pathlib import Path
from pprint import pprint
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, Booster
from xgboost.callback import EarlyStopping, TrainingCallback

from util import SNANA_TO_TAXONOMY


class SaveModelCallback(TrainingCallback):
    def __init__(self, rounds: int, path: Union[Path, str]):
        super().__init__()
        self.rounds = rounds
        self.path = path

    def after_iteration(self, model: Booster, epoch: int, evals_log) -> bool:
        del evals_log
        if epoch == 0:
            return False
        if epoch % self.rounds != 0:
            return False
        model.save_model(self.path)
        print(f'xgboost model is saved at epoch {epoch} to {self.path}')
        return False



@lru_cache(maxsize=1)
def type_weights() -> Dict[str, float]:
    type_counts = Counter(SNANA_TO_TAXONOMY.values())
    return {type_: 1.0 / count for type_, count in type_counts.items()}


def get_weights(types: np.ndarray) -> np.ndarray:
    d = type_weights()
    return np.vectorize(d.get, otypes='g')(types)


def get_XyId(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(path)

    feature_suffix = '_features.npy'
    type_suffix = '_types.npy'
    id_suffix = '_ids.npy'

    feature_paths = sorted(path.glob(f'*{feature_suffix}'))
    type_paths = sorted(path.glob(f'*{type_suffix}'))
    id_paths = sorted(path.glob(f'*{id_suffix}'))

    assert len(feature_paths) > 0
    assert len(feature_paths) == len(feature_paths) == len(id_paths)

    features = []
    types = []
    ids = []
    for feature_path, type_path, id_path in zip(feature_paths, type_paths, id_paths):
        assert (feature_path.name.removesuffix(feature_suffix)
                == type_path.name.removesuffix(type_suffix)
                == id_path.name.removesuffix(id_suffix))
        f = np.load(feature_path)
        features.append(f)

        t = np.load(type_path)
        assert f.shape[0] == t.shape[0]
        types.append(t)

        id_ = np.load(id_path)
        assert f.shape[0] == id_.shape[0]
        ids.append(id_)
    X = np.concatenate(features)
    y = np.concatenate(types)
    ids = np.concatenate(ids)

    # ids are unique within a class
    ids = np.array([f'{id_}_{t_}' for t_, id_ in zip(y, ids)])

    return X, y, ids


def fix_features(X: np.ndarray) -> np.ndarray:
    X[np.isneginf(X)] = np.finfo(X.dtype.type).min
    X[np.isposinf(X)] = np.finfo(X.dtype.type).max
    return X


def get_feature_names(path: Union[str, Path]) -> List[str]:
    path = Path(path)
    with open(path.joinpath('names.txt')) as fh:
        return fh.read().split()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--features', required=True, help='path with features')
    parser.add_argument('--figures', required=True, help='output figure path')
    parser.add_argument('--output', required=True, help='output model path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    path = args.features
    figpath = Path(args.figures)
    figpath.mkdir(exist_ok=True)

    X, y, ids = get_XyId(path)

    X = fix_features(X)

    weights = get_weights(y)
    label_encoder = {label: i for i, label in enumerate(np.unique(y))}
    label_decoder = np.array(list(label_encoder))
    labels, y = y, np.vectorize(label_encoder.get, otypes='i')(y)

    feature_names = get_feature_names(path)
    assert X.shape[1] == len(feature_names)

    # Split over unique objects to have 0.6/0.2/0.2 train/val/test samples
    # Is there a way to do it using indexing without np.isin?
    unique_ids = np.unique(ids)
    ids_trainval, ids_test = train_test_split(unique_ids, test_size=0.2, shuffle=True, random_state=0)
    ids_train, ids_val = train_test_split(ids_trainval, test_size=0.25, shuffle=False)
    mask_train = np.isin(ids, ids_train)
    mask_val = np.isin(ids, ids_val)
    mask_test = np.isin(ids, ids_test)
    X_train, y_train, w_train = X[mask_train], y[mask_train], weights[mask_train]
    X_val, y_val, w_val = X[mask_val], y[mask_val], weights[mask_val]
    X_test, y_test, w_test = X[mask_test], y[mask_test], weights[mask_test]

    assert set(y_train) == set(y_test) == set(y_val), 'some types are underrepresented in one of train/val/test sample'

    model_path = f'{args.output}/xgb.ubj'

    early_stopping = EarlyStopping(
        rounds=10,
        min_delta=1e-5,
        save_best=True,
        maximize=False,
        data_name="validation_0",
        metric_name="mlogloss",
    )
    save_model = SaveModelCallback(
        rounds=10,
        path=model_path,
    )
    classifier = XGBClassifier(
        n_estimators=10000,
        learning_rate=0.1,
        use_label_encoder=False,
        booster='gbtree',
        seed=0,
        nthread=-1,
        missing=np.nan,
        # max_depth=max(6, int(np.log2(len(feature_names)) + 1)),  # 6 is default
    )
    classifier.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        sample_weight_eval_set=[w_val],
        callbacks=[early_stopping, save_model],
        verbose=True,
    )
    classifier.get_booster().feature_names = feature_names
    classifier.save_model(model_path)

    pprint(sorted(classifier.get_booster().get_fscore().items(), key=lambda x: x[1], reverse=True))
    accuracy = accuracy_score(y_test, classifier.predict(X_test), sample_weight=w_test)
    print('Accuracy', accuracy)

    plt.figure(figsize=(20, 20))
    ConfusionMatrixDisplay.from_predictions(
        label_decoder[y_test],
        label_decoder[classifier.predict(X_test)],
        normalize='true',
        ax=plt.gca(),
    )
    plt.title(f'Accuracy {accuracy:.3f}')
    plt.savefig(figpath.joinpath('conf_matrix.pdf'))
    plt.close()


if __name__ == '__main__':
    main()
