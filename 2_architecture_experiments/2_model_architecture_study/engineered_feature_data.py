"""Causal v4 engineered rows, windows, and trajectories for architecture Run 10."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for folder in ('2_tabular_data_adapter', '3_sequence_data_adapter', '3_trajectory_data_adapter', '4_model_adapters'):
    sys.path.insert(0, str(HERE / folder))

from tabular_data_adapter import TabularDataset
from sequence_data_adapter import SequenceDataset
from trajectory_data_adapter import TrajectoryDataset, TrajectoryReferenceLibrary


def load_script(path):
    spec = importlib.util.spec_from_file_location('engineered_v4_source', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_raw(raw):
    if raw.duplicated(['uav_id', 'flight_cycle']).any():
        raise ValueError('Duplicate UAV/cycle keys')
    for uid, group in raw.groupby('uav_id'):
        if not np.array_equal(np.sort(group.flight_cycle), np.arange(1, len(group) + 1)):
            raise ValueError(f'Non-contiguous cycle history: {uid}')


class FeatureView:
    """Feature selection and scaling use only this actual fit's training UAVs.

    Building causal features on the other histories uses neither their labels
    nor any aggregate across UAVs. Each time position contains the script's
    entire feature row, never an artificial ordering of feature columns as time.
    """

    def __init__(self, raw, training_ids, script, expected_features=266):
        self.raw = raw.sort_values(['uav_id', 'flight_cycle']).reset_index(drop=True)
        self.training_ids = set(training_ids)
        train_mask = self.raw.uav_id.isin(self.training_ids)
        if not self.training_ids or set(self.raw.loc[train_mask, 'uav_id']) != self.training_ids:
            raise ValueError('Missing training UAVs')
        self.sensors = script.get_sensor_cols(self.raw.loc[train_mask].drop(columns='RUL'))
        features = script.build_features(self.raw.drop(columns='RUL'), self.sensors)
        self.names = tuple(c for c in features if c != 'uav_id')
        if len(self.names) != expected_features:
            raise ValueError(f'Expected {expected_features} features, got {len(self.names)}')
        if not features[['uav_id', 'flight_cycle']].equals(self.raw[['uav_id', 'flight_cycle']]):
            raise ValueError('Feature/target cycle alignment changed')
        self.values = features[list(self.names)].to_numpy(np.float32)
        if not np.isfinite(self.values).all():
            raise ValueError('Nonfinite engineered features')
        train_values = self.values[train_mask]
        self.center = np.median(train_values, axis=0)
        q25, q75 = np.percentile(train_values, [25, 75], axis=0)
        scale = (q75 - q25) / 1.349
        std = np.std(train_values, axis=0, dtype=np.float64)
        self.scale = np.where(scale > 1e-8, scale, np.where(std > 1e-8, std, 1.)).astype(np.float32)
        self.scaled = ((self.values - self.center) / self.scale).astype(np.float32)
        self.indices = {uid: group.index.to_numpy() for uid, group in self.raw.groupby('uav_id', sort=True)}
        self.key_index = pd.MultiIndex.from_frame(self.raw[['uav_id', 'flight_cycle']])

    def rows(self, endpoints):
        idx = self.key_index.get_indexer(pd.MultiIndex.from_frame(endpoints[['uav_id', 'cutoff']]))
        if (idx < 0).any():
            raise ValueError('Requested endpoint cycle missing')
        return idx

    def dataset(self, endpoints, family, lookback=None, *, training=False, labels=False):
        metadata = endpoints[['uav_id', 'cutoff']].reset_index(drop=True)
        if training and not set(metadata.uav_id).issubset(self.training_ids):
            raise ValueError('Training endpoint outside active training UAVs')
        if not training and set(metadata.uav_id) & self.training_ids:
            raise ValueError('Evaluation UAV appears in fit/scaler/reference partition')
        idx = self.rows(metadata)
        target = self.raw.iloc[idx].RUL.reset_index(drop=True).astype(float) if labels else None
        weights = pd.Series(np.ones(len(metadata))) if training else None
        if family in ('xgboost', 'mlp'):
            return TabularDataset(pd.DataFrame(self.values[idx], columns=self.names), metadata, target, weights)
        # These are duplicate access to two of the 266 columns, not new features.
        side = np.column_stack([metadata.cutoff, np.log1p(metadata.cutoff)]).astype(np.float32)
        side_names = ('flight_cycle', 'flight_cycle_log')
        if family == 'trajectory_dtw_knn':
            trajectories, cycles = [], []
            for uid, cutoff in metadata.itertuples(index=False, name=None):
                positions = self.indices[uid][:int(cutoff)]
                trajectories.append(self.scaled[positions])
                cycles.append(self.raw.iloc[positions].flight_cycle.to_numpy(np.int64))
            library = None
            if training:
                uids = sorted(self.training_ids)
                positions = [self.indices[uid] for uid in uids]
                library = TrajectoryReferenceLibrary(
                    tuple(self.scaled[p] for p in positions),
                    tuple(self.raw.iloc[p].flight_cycle.to_numpy(np.int64) for p in positions),
                    tuple(np.minimum(self.raw.iloc[p].RUL.to_numpy(np.float32), 125.) for p in positions),
                    pd.DataFrame({'uav_id': uids}), self.names, True)
            return TrajectoryDataset(tuple(trajectories), tuple(cycles), side, metadata,
                target, weights, self.names, side_names, library, True)
        if lookback is None or lookback < 1:
            raise ValueError('Temporal network needs a positive lookback')
        sequences = np.zeros((len(metadata), lookback, len(self.names)), dtype=np.float32)
        padding = np.ones((len(metadata), lookback), dtype=bool)
        for i, (uid, cutoff) in enumerate(metadata.itertuples(index=False, name=None)):
            positions = self.indices[uid][max(0, int(cutoff)-lookback):int(cutoff)]
            sequences[i, -len(positions):] = self.scaled[positions]
            padding[i, -len(positions):] = False
        return SequenceDataset(sequences, padding, side, metadata, target, weights,
            self.names, side_names, lookback, True)


def training_endpoints(raw, ids, trajectory=False):
    selected = raw.loc[raw.uav_id.isin(ids), ['uav_id', 'flight_cycle']]
    if trajectory:
        selected = selected.groupby('uav_id', sort=True).tail(1)
    return selected.rename(columns={'flight_cycle': 'cutoff'}).reset_index(drop=True)
