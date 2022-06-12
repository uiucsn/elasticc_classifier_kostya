#!/usr/bin/env python3

from collections import Counter
from functools import lru_cache
from glob import glob
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from util import SNANA_TO_TAXONOMY


@lru_cache(maxsize=1)
def type_weights() -> Dict[str, float]:
    type_counts = Counter(SNANA_TO_TAXONOMY.values())
    return {type_: 1.0 / count for type_, count in type_counts.items()}


def get_weights(types: np.ndarray) -> np.ndarray:
    d = type_weights()
    return np.vectorize(d.get)(types)


def get_Xy(path: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    path = Path(path)

    feature_suffix = '_features.npy'
    type_suffix = '_types.npy'

    feature_paths = sorted(glob(str(path.joinpath(f'*{feature_suffix}'))))
    type_paths = sorted(glob(str(path.joinpath(f'*{type_suffix}'))))

    assert len(feature_paths) > 0
    assert len(feature_paths) == len(feature_paths)

    features = []
    types = []
    for feature_path, type_path in zip(feature_paths, type_paths):
        assert Path(feature_path).name.removesuffix(feature_suffix) == Path(type_path).name.removesuffix(type_suffix)
        f = np.load(feature_path)
        features.append(f)
        t = np.load(type_path)
        assert f.shape[0] == t.shape[0]
        types.append(t)
    X = np.concatenate(features)
    y = np.concatenate(types)
    return X, y


def main():
    X, y = get_Xy('./features/')
    weights = get_weights(y)

    X_trainval, X_test, y_trainval, y_test, w_trainval, w_test = train_test_split(X, y, weights, test_size=0.2,
                                                                                  shuffle=True, random_state=0)
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(X_trainval, y_trainval, w_trainval,
                                                                      test_size=0.25, shuffle=False)

    classifier = XGBClassifier(
        use_label_encoder=True,
        booster='gbtree',
        seed=0,
        nthread=-1,
        missing=np.nan,
    )
    classifier.fit(X_trainval, y_trainval, sample_weight=w_trainval, eval_metric='mlogloss')
    print('Accuracy', accuracy_score(y_test, classifier.predict(X_test), sample_weight=w_test))
    classifier.save_model('model/xgb.ubj')


if __name__ == '__main__':
    main()