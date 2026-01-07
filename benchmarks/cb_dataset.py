import os
import numpy as np
import torch
from torch.utils.data import Dataset

from benchmarks.colour_bomb import ColourBombGridworldV1Env, CBConfig


class CBSequenceDataset(Dataset):
    """
    Generates offline episodes once and can cache them to disk (.npz).
    """

    def __init__(
        self,
        num_episodes=1000,
        max_steps=200,
        sequence_length=200,
        discount=0.99,
        stochastic=False,
        seed=0,
        dataset_path=None,
    ):
        self.sequence_length = int(sequence_length)
        self.discount = float(discount)

        cfg = CBConfig(max_steps=max_steps, stochastic=stochastic)
        self.env = ColourBombGridworldV1Env(cfg)

        self.observation_dim = 1
        self.action_dim = 1
        self.joined_dim = 4

        self.rows_per_seg = max(1, self.sequence_length // self.joined_dim)

        if dataset_path is not None and os.path.exists(dataset_path):
            payload = np.load(dataset_path, allow_pickle=True)
            self.episodes_tokens = payload["episodes_tokens"].tolist()
        else:
            self.episodes_tokens = self._generate(num_episodes, max_steps, seed)
            if dataset_path is not None:
                os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
                np.savez_compressed(dataset_path, episodes_tokens=np.array(self.episodes_tokens, dtype=object))

        self.indices = self._build_indices()

    def _generate(self, num_episodes, max_steps, seed):
        rng = np.random.RandomState(seed)
        episodes_tokens = []

        for _ in range(num_episodes):
            s, _ = self.env.reset()
            states, actions, rewards = [], [], []

            for _t in range(max_steps):
                a = int(rng.randint(self.env.action_space.n))
                ns, r, done, _info = self.env.step(a)
                states.append(int(s))
                actions.append(int(a))
                rewards.append(float(r))
                s = ns
                if done:
                    break

            T = len(states)
            if T == 0:
                continue

            tokens = np.zeros((T, 4), dtype=np.int64)
            tokens[:, 0] = np.array(states, dtype=np.int64)
            tokens[:, 1] = np.array(actions, dtype=np.int64)
            tokens[:, 2] = 0
            tokens[:, 3] = 0

            # append stop row (same token id for all dims)
            max_bin = max(self.env.observation_space.n, self.env.action_space.n, 1)
            end_token = max_bin
            end_row = np.array([end_token] * 4, dtype=np.int64)
            tokens = np.vstack([tokens, end_row])

            episodes_tokens.append(tokens)

        return episodes_tokens

    def _build_indices(self):
        indices = []
        for ep_idx, rows in enumerate(self.episodes_tokens):
            R = rows.shape[0]
            if R < self.rows_per_seg + 1:
                continue

            # non-overlapping windows + tail
            starts = list(range(0, max(1, R - (self.rows_per_seg + 1)), self.rows_per_seg))
            tail = R - (self.rows_per_seg + 1)
            if tail not in starts:
                starts.append(tail)

            for start_row in starts:
                indices.append((ep_idx, start_row))
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep_idx, start_row = self.indices[idx]
        rows = self.episodes_tokens[ep_idx]
        seg_rows = rows[start_row:start_row + self.rows_per_seg + 1]  # +1 for next-token targets
        flat = seg_rows.reshape(-1)

        x = torch.from_numpy(flat[:-self.joined_dim].astype(np.int64))
        y = torch.from_numpy(flat[self.joined_dim:].astype(np.int64))
        mask = torch.ones_like(x, dtype=torch.float32)
        return x, y, mask
