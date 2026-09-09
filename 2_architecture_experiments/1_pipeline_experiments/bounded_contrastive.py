"""Optional three-arm TCN pilot; all selection stays inside training UAVs.

The degradation-distance weighted contrastive loss is an adaptation inspired by
FSGRI, not a reproduction of its published benchmark. Age is excluded from the
contrastive embedding. All arms see the same sampled windows and forward passes.
"""
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.preprocessing import RobustScaler

import campaign_data  # establishes the model adapter import path
from bounded_lightgbm import stopping_split
from campaign_models import training_endpoints
from confirmation_utils import select_uavs
from feature_comparison_features import SENSORS_22
from models.neural.tcn import TCNRegressor

ARMS = ('tcn_mse', 'tcn_augmentation', 'tcn_contrastive')
NOISE_CHANNELS = tuple(SENSORS_22.index(f'telemetry_{i:02d}') for i in (13, 19, 21, 22, 25, 28))


class Encoder(TCNRegressor):
    def forward(self, sequences, padding_mask, side_features):
        hidden = sequences.transpose(1, 2)
        valid = (~padding_mask).unsqueeze(1).to(hidden.dtype)
        hidden = hidden * valid
        for block in self.blocks:
            hidden = block(hidden, valid)
        embedding = hidden[:, :, -1]
        prediction = self.output(torch.cat([embedding, side_features], dim=1)).squeeze(-1)
        return prediction, embedding


class Prefixes:
    def __init__(self, raw, training, lookback):
        self.lookback = lookback
        self.groups = {uav: group.sort_values('flight_cycle') for uav, group in raw.groupby('uav_id')}
        ids = set(training.metadata.uav_id)
        # Scaling uses only rows observed in the fitting prefixes, not later life.
        maxima = training.metadata.groupby('uav_id').cutoff.max()
        observed = raw.loc[raw.uav_id.isin(ids) & raw.flight_cycle.le(raw.uav_id.map(maxima))]
        self.scaler = RobustScaler().fit(observed[list(SENSORS_22)].to_numpy(float))
        self.age_scaler = RobustScaler().fit(self.ages(training.metadata))

    @staticmethod
    def ages(metadata):
        age = metadata.cutoff.to_numpy(float)
        return np.column_stack([age, np.log1p(age)])

    def transform(self, metadata):
        x = np.zeros((len(metadata), self.lookback, len(SENSORS_22)), np.float32)
        mask = np.ones((len(metadata), self.lookback), bool)
        for i, row in enumerate(metadata.itertuples()):
            group = self.groups[row.uav_id]
            prefix = group.loc[group.flight_cycle.le(row.cutoff)].tail(self.lookback)
            if prefix.empty or int(prefix.flight_cycle.iloc[-1]) != int(row.cutoff):
                raise ValueError('Sequence endpoint missing from raw history')
            values = self.scaler.transform(prefix[list(SENSORS_22)].to_numpy(float))
            x[i, -len(values):] = values
            mask[i, -len(values):] = False
        side = self.age_scaler.transform(self.ages(metadata)).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(side)


def pairs(metadata, targets, rng, minimum_gap):
    """One same-UAV negative per anchor, selected using training labels only."""
    ids = metadata.uav_id.to_numpy()
    y = np.minimum(np.asarray(targets, float), 125.) / 125.
    negative = np.arange(len(ids))
    distance = np.zeros(len(ids), np.float32)
    for i in range(len(ids)):
        eligible = np.flatnonzero((ids == ids[i]) & (np.abs(y-y[i]) > minimum_gap))
        if len(eligible):
            negative[i] = rng.choice(eligible)
            distance[i] = abs(y[i]-y[negative[i]])
    return negative, distance


def augment(x, mask, rng, scale):
    noise = np.zeros(x.shape, np.float32)
    noise[:, :, NOISE_CHANNELS] = rng.normal(0, scale, (len(x), x.shape[1], len(NOISE_CHANNELS)))
    noise[mask.numpy()] = 0
    return x + torch.from_numpy(noise)


def contrastive_loss(anchor, positive, negative, distances, temperature):
    valid = distances > 0
    if not valid.any():
        return anchor.sum()*0.
    a, p, n = (F.normalize(v[valid], dim=1) for v in (anchor, positive, negative))
    positive_logit = (a*p).sum(1)/temperature
    negative_logit = (a*n).sum(1)/temperature + distances[valid].clamp_min(1e-8).log()
    return F.cross_entropy(torch.stack([positive_logit, negative_logit], dim=1),
                           torch.zeros(len(a), dtype=torch.long))


def build_network(workflow, seed):
    torch.manual_seed(seed)
    return Encoder(len(SENSORS_22), 2, workflow['residual_blocks'], workflow['channels'],
                   3, 2, workflow['dropout'])


def predict(model, inputs, batch_size):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(inputs[0]), batch_size):
            predictions.append(model(*(v[start:start+batch_size] for v in inputs))[0].numpy()*125.)
    return np.maximum(np.concatenate(predictions), 0.)


def train(raw, training, stopping, arm, seed, workflow, epochs):
    representation = Prefixes(raw, training, workflow['lookback'])
    inputs = representation.transform(training.metadata)
    stop_inputs = representation.transform(stopping.metadata) if stopping is not None else None
    target = torch.tensor(np.minimum(training.target.to_numpy(float), 125.)/125., dtype=torch.float32)
    model = build_network(workflow, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=workflow['learning_rate'])
    history, best, best_epoch = [], float('inf'), 1
    for epoch in range(1, epochs+1):
        rng = np.random.default_rng(seed+epoch*1009)
        negative, distance = pairs(training.metadata, training.target, rng, workflow['minimum_negative_gap'])
        order = rng.permutation(len(training))
        losses = []
        model.train()
        for start in range(0, len(order), workflow['batch_size']):
            idx = order[start:start+workflow['batch_size']]
            neg = negative[idx]
            x, mask, side = (v[idx] for v in inputs)
            # Draw noise in every arm so sampling RNG and forward shapes match.
            noisy = augment(x, mask, rng, workflow['noise_std'])
            positive = x if arm == 'tcn_mse' else noisy
            all_inputs = (torch.cat([x, positive, inputs[0][neg]]),
                          torch.cat([mask, mask, inputs[1][neg]]),
                          torch.cat([side, side, inputs[2][neg]]))
            estimate, embedding = model(*all_inputs)
            supervised = F.mse_loss(estimate, torch.cat([target[idx], target[idx], target[neg]]))
            n = len(idx)
            auxiliary = contrastive_loss(embedding[:n], embedding[n:2*n], embedding[2*n:],
                                        torch.from_numpy(distance[idx]), workflow['temperature'])
            loss = supervised + (workflow['contrastive_weight']*auxiliary if arm == 'tcn_contrastive' else 0.)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite neural loss')
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(float(loss.detach()))
        entry = {'epoch': epoch, 'loss': float(np.mean(losses))}
        if stopping is not None:
            p = predict(model, stop_inputs, workflow['batch_size'])
            w = 1./stopping.metadata.groupby('uav_id').uav_id.transform('size').to_numpy(float)
            rmse = float(np.sqrt(np.average((p-stopping.target.to_numpy(float))**2, weights=w)))
            entry['stopping_rmse'] = rmse
            if rmse < best:
                best, best_epoch = rmse, epoch
            history.append(entry)
            if epoch-best_epoch >= workflow['stopping_patience']:
                break
        else:
            history.append(entry)
    return model, representation, best_epoch, history


def fit_contrastive(raw, training, calibration, held, arm, seed, workflow):
    if arm not in ARMS:
        raise ValueError('Unknown pilot arm')
    ids = set(training.metadata.uav_id)
    if ids & set(held.metadata.uav_id):
        raise ValueError('Held UAVs leaked into TCN training')
    counts = training.metadata.groupby('uav_id').size()
    if counts.nunique() != 1 or not np.allclose(training.sample_weights, 1./counts.iloc[0]):
        raise ValueError('Pilot requires equally many prefixes and total weight one per UAV')
    torch.set_num_threads(workflow['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    fit_ids, stop_ids = stopping_split(ids, seed)
    fit_data = select_uavs(training, fit_ids)
    stopping = training_endpoints(calibration, stop_ids)
    if set(stopping.metadata.uav_id) != stop_ids:
        raise ValueError('Stopping endpoint coverage incomplete')
    start = time.perf_counter()
    _, _, rounds, stopping_history = train(raw, fit_data, stopping, arm, seed, workflow, workflow['maximum_epochs'])
    model, representation, _, refit_history = train(raw, training, None, arm, seed, workflow, rounds)
    prediction = predict(model, representation.transform(held.metadata), workflow['batch_size'])
    return prediction, {'fit_uavs': sorted(fit_ids), 'stopping_uavs': sorted(stop_ids),
        'refit_uavs': sorted(ids), 'best_epoch': rounds, 'neural_fits': 2,
        'fit_seconds': time.perf_counter()-start, 'stopping_history': stopping_history,
        'refit_history': refit_history, 'device': 'cpu', 'contrastive_embedding_excludes_age': True}
