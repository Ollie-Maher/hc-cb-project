import datetime
import io
import random
import traceback
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open("wb") as f:
            f.write(bs.read())


def load_episode(fn):
    with fn.open("rb") as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


class ReplayBufferStorage:
    def __init__(self, data_specs, meta_specs, replay_dir):
        self._data_specs = data_specs
        self._meta_specs = meta_specs
        self._replay_dir = replay_dir
        replay_dir.mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._preload()

    def __len__(self):
        return self._num_transitions

    def reset(self):
        """
        Reset the replay buffer, clearing all stored episodes and data.
        """
        self._num_episodes = 0
        self._num_transitions = 0
        self._current_episode = defaultdict(list)

        # Delete episode files from disk
        for fn in self._replay_dir.glob("*.npz"):
            fn.unlink()

    def add(self, time_step, meta):
        for key, value in meta.items():
            self._current_episode[key].append(value)
        for spec in self._data_specs:
            value = time_step[spec.name]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)

            # print("spec.shape", spec.shape)
            # print("spec.name", spec.name)
            # print("time_step[spec.name]", time_step[spec.name])
            # print("value:", value.shape)
            # print("value.shape", value.shape)
            assert spec.shape == value.shape and spec.dtype == value.dtype
            self._current_episode[spec.name].append(value)
        if time_step.last():
            episode = dict()
            for spec in self._data_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
            for spec in self._meta_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
            self._current_episode = defaultdict(list)
            self._store_episode(episode)

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob("*.npz"):
            _, _, eps_len = fn.stem.split("_")
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        eps_fn = f"{ts}_{eps_idx}_{eps_len}.npz"
        save_episode(episode, self._replay_dir / eps_fn)


class ReplayBuffer(IterableDataset):
    def __init__(
        self,
        storage,
        max_size,
        num_workers,
        nstep,
        discount,
        fetch_every,
        save_snapshot,
        sequence_len,
    ):
        self._storage = storage
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot
        self._sequence_length = sequence_len

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def reset_memory(self):
        """
        Reset the memory of the replay buffer, clearing all data in memory.
        """
        self._size = 0
        self._episode_fns = []
        self._episodes = dict()

    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn)
        except:
            return False
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            early_eps_fn.unlink(missing_ok=True)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._storage._replay_dir.glob("*.npz"), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split("_")[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break

    def _sample(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = (
            np.random.randint(
                0,
                episode_len(episode) - self._nstep + 1
                # 0, episode_len(episode) - self._sequence_length + 1 - self._nstep + 1
            )
            + 1
        )
        # idx = idx - self._sequence_length if idx > self._sequence_length else 1
        obs_seq = []
        action_seq = []
        reward_seq = []
        discount_seq = []
        next_obs_seq = []
        eps_len = episode_len(episode)
        # print("eps_len", eps_len)
        reward = None
        discount = None
        for i in range(self._sequence_length):
            # print("idx", idx)
            # print("i", i)
            # append with zeros for sequence above episode length
            if idx + i > eps_len:
                # print("eps_len", eps_len)
                # print("appending zeros")
                obs_seq.append(np.zeros_like(episode["observation"][eps_len - 1]))
                action_seq.append(np.zeros_like(episode["action"][eps_len - 1]))
                next_obs_seq.append(np.zeros_like(episode["observation"][eps_len - 1]))
                # reward_seq.append(np.zeros_like(episode["reward"][eps_len - 1]))
                # discount_seq.append(np.zeros_like(episode["discount"][eps_len - 1]))
                reward = (
                    np.zeros_like(episode["reward"][eps_len - 1])
                    if reward is None
                    else reward
                )
                discount = (
                    np.ones_like(episode["discount"][eps_len - 1])
                    if discount is None
                    else discount
                )
                reward_seq.append(reward)
                discount_seq.append(discount)
                continue
            obs_seq.append(episode["observation"][idx - 1 + i])
            action_seq.append(episode["action"][idx + i])
            # reward_seq.append()
            # discount_seq.append()
            next_obs_seq.append(episode["observation"][idx + i + self._nstep - 1])
            reward = np.zeros_like(episode["reward"][idx + i])
            discount = np.ones_like(episode["discount"][idx + i])
            for j in range(self._nstep):
                step_reward = episode["reward"][idx + i + j]
                reward += discount * step_reward
                discount *= episode["discount"][idx + i + j] * self._discount
            reward_seq.append(reward)
            discount_seq.append(discount)

        meta = []
        for spec in self._storage._meta_specs:
            meta.append(episode[spec.name][idx - 1])
        return (
            np.stack(obs_seq),
            np.stack(action_seq),
            np.stack(reward_seq),
            np.stack(discount_seq),
            np.stack(next_obs_seq),
            *meta,
        )

        # meta = []
        # for spec in self._storage._meta_specs:
        #     meta.append(episode[spec.name][idx - 1])
        # obs = episode["observation"][idx - 1]
        # action = episode["action"][idx]
        # next_obs = episode["observation"][idx + self._nstep - 1]
        # reward = np.zeros_like(episode["reward"][idx])
        # discount = np.ones_like(episode["discount"][idx])
        # for i in range(self._nstep):
        #     step_reward = episode["reward"][idx + i]
        #     reward += discount * step_reward
        #     discount *= episode["discount"][idx + i] * self._discount
        # return (obs, action, reward, discount, next_obs, *meta)

    def __iter__(self):
        while True:
            yield self._sample()


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    # print(f"Setting worker {worker_id} seed to {seed}")
    # print(type(seed))
    np.random.seed(seed)
    random.seed(int(seed))


def make_replay_loader(
    storage,
    max_size,
    batch_size,
    num_workers,
    save_snapshot,
    nstep,
    discount,
    sequence_len,
):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = ReplayBuffer(
        storage,
        max_size_per_worker,
        num_workers,
        nstep,
        discount,
        fetch_every=1000,
        save_snapshot=save_snapshot,
        sequence_len=sequence_len,
    )

    loader = torch.utils.data.DataLoader(
        iterable,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )
    return loader
