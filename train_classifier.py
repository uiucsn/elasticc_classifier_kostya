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
import torch
import torch.utils.data
from torch import nn
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
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


def fix_features_for_xgboost(X: np.ndarray) -> np.ndarray:
    X[np.isneginf(X)] = np.finfo(X.dtype.type).min
    X[np.isposinf(X)] = np.finfo(X.dtype.type).max
    return X


def preprocess_for_xgboost(X_train, X_val, X_test):
    X_train = fix_features_for_xgboost(X_train)
    X_val = fix_features_for_xgboost(X_val)
    X_test = fix_features_for_xgboost(X_test)

    return X_train, X_val, X_test


def xgboost_classifier(X_train, y_train, w_train, X_val, y_val, w_val, *, feature_names, tree_method, output, **_kwargs):
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
        path=output / 'xgb_intermediate.ubj',
    )
    classifier = XGBClassifier(
        n_estimators=10000,
        learning_rate=0.1,
        use_label_encoder=False,
        booster='gbtree',
        tree_method=tree_method,
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
    classifier.save_model(output / 'xgb.ubj')

    pprint(sorted(classifier.get_booster().get_fscore().items(), key=lambda x: x[1], reverse=True))

    return classifier


class MLP(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()

        self.nn = nn.Sequential(
            nn.Linear(n_features, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 400),
            nn.ReLU(),
            nn.Linear(400, n_classes),
        )

    def forward(self, X):
        X = self.nn(X)
        return nn.functional.log_softmax(X, dim=1)


class TorchClassifier:
    def __init__(self, module, device='cpu'):
        self.module = module
        self.device = torch.device(device)

    def predict(self, X):
        X = torch.tensor(X, device=self.device)
        y = self.module(X)
        y = torch.argmax(y, dim=1)
        return y.detach().numpy()


class Normalizer:
    def __init__(self):
        self.means = None
        self.scaler = QuantileTransformer(n_quantiles=1_000, subsample=100_000, output_distribution='normal',
                                          random_state=0)

    def fit(self, X):
        soft_max = np.sqrt(np.finfo(X.dtype.type).max)
        soft_X = np.clip(X, -soft_max, soft_max)
        self.means = np.nanmean(soft_X, axis=0)
        self.scaler.fit(X)
        return self

    def transform(self, X):
        if self.means is None:
            raise RuntimeError('Normalizer is not fitted')
        X = np.where(np.isfinite(X), X, self.means)
        return self.scaler.transform(X)


def preprocess_for_torch(X_train, X_val, X_test):
    normalizer = Normalizer().fit(X_train)
    X_train = normalizer.transform(X_train)
    X_val = normalizer.transform(X_val)
    X_test = normalizer.transform(X_test)

    return X_train, X_val, X_test


def mlp_classifier(X_train, y_train, X_val, y_val, class_weights, *, device, output, **_kwargs):
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    if device == 'cuda':
        torch.backends.cudnn.benchmark = False

    device = torch.device(device)

    model = MLP(X_train.shape[1], np.unique(y_train).size)
    model = model.to(device)

    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.uint8, device=device)
    X_val = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val = torch.tensor(y_val, dtype=torch.uint8, device=device)
    class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True)

    loss_fn = nn.NLLLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    early_stop_rounds = 10
    val_loss_history = []

    for epoch in range(10000):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            y_val_pred = model(X_val)
            loss = loss_fn(y_val_pred, y_val)
            loss = loss.item()
            val_loss_history.append(loss)
        print(f'epoch: {epoch}, loss: {loss:.5f}')

        if len(val_loss_history) > early_stop_rounds and all(np.diff(val_loss_history[-early_stop_rounds:]) > 0):
            print('Validation loss is not decreasing, stopping training')
            break

        if epoch % 10 == 0 and epoch != 0:
            path = output / 'mlp_intermediate.pt'
            torch.save(model.state_dict(), path)
            print(f'PyTorch MLP model is saved at epoch {epoch} tp {path}')

    torch.save(model.state_dict(), output / 'mlp.pt')

    return TorchClassifier(model)


@lru_cache(maxsize=1)
def type_weights() -> Dict[str, float]:
    type_counts = Counter(SNANA_TO_TAXONOMY.values())
    return {type_: 1.0 / count for type_, count in type_counts.items()}


def get_weights(types: np.ndarray) -> np.ndarray:
    d = type_weights()
    return np.vectorize(d.get, otypes=[np.float32])(types)


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


def get_feature_names(path: Union[str, Path]) -> List[str]:
    path = Path(path)
    with open(path / 'names.txt') as fh:
        return fh.read().split()


def parse_args():
    main_parser = ArgumentParser()

    main_parser.add_argument('--features', type=Path, required=True, help='path with features')
    main_parser.add_argument('--figures', type=Path, required=True, help='output figure path')
    main_parser.add_argument('--output', type=Path, required=True, help='output model path')

    algo_subparsers = main_parser.add_subparsers(title='algo', dest='algo', required=True, help='algorithm to use')

    xgboost_parser = algo_subparsers.add_parser('xgboost')
    xgboost_parser.add_argument('--tree-method', default='auto', help='xgboost tree method, e.g. "auto" or "gpu_hist"')

    mlp_parser = algo_subparsers.add_parser('mlp')
    mlp_parser.add_argument('--device', default='cpu', help='device to use, e.g. "cuda" or "cpu" or "mps"')

    args = main_parser.parse_args()
    return args


def main():
    args = parse_args()

    path = args.features
    figpath = args.figures
    figpath.mkdir(exist_ok=True)

    X, y, ids = get_XyId(path)

    weights = get_weights(y)
    label_encoder = {label: i for i, label in enumerate(np.unique(y))}
    label_decoder = np.array(list(label_encoder))
    class_weights = get_weights(np.array(list(label_encoder)))
    labels, y = y, np.vectorize(label_encoder.get, otypes=[np.uint8])(y)

    with open(args.output / 'label_decoder.txt', 'w') as fh:
        fh.write('\n'.join(label_decoder))

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

    if args.algo == 'xgboost':
        X_train, X_val, X_test = preprocess_for_xgboost(X_train, X_val, X_test)
        classifier = xgboost_classifier(X_train, y_train, w_train, X_val, y_val, w_val, feature_names=feature_names,
                                        **vars(args))
    elif args.algo == 'mlp':
        X_train, X_val, X_test = preprocess_for_torch(X_train, X_val, X_test)
        classifier = mlp_classifier(X_train, y_train, X_val, y_val, class_weights=class_weights, **vars(args))
    else:
        raise ValueError(f'Unknown algorithm: {args.algo}')

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
    plt.savefig(figpath / 'conf_matrix.pdf')
    plt.close()


if __name__ == '__main__':
    main()
