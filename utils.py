import random
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal


class Activations(object):
    """Class to record and save model activations"""

    def __init__(self, root_dir):
        if root_dir is not None:
            self.save_dir = root_dir / "activations"
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.columns = [f"neuron_{i}" for i in range(512)] + [
            "step",
            "episode",
            "reward",
            "env_id",
        ]
        self.df = pd.DataFrame(columns=self.columns)
        self.lst = []  # list to store tensors tmp because df is not efficient

    def add_activations(self, acts, step, episode, reward, env_id):
        # acts: (1, 1, 512)
        acts = acts.squeeze(0).squeeze(0).cpu().numpy()
        # tensor_row shape (516,)
        tensor_row = np.concatenate([acts, [step, episode, reward, env_id]])
        self.lst.append(tensor_row)

    def save(self):
        # print("Saving activations...")
        new_df = pd.DataFrame(self.lst, columns=self.columns)
        self.df = pd.concat([self.df, new_df], ignore_index=True)
        self.df.to_csv(self.save_dir / "ca3.csv", index=False)
        self.lst = []


class AgentPos(object):
    """Class to record and save agent position"""

    def __init__(self, root_dir):
        if root_dir is not None:
            self.save_dir = root_dir / "locations"
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.columns = [
            "position_x",
            "position_y",
            "step",
            "episode",
            "reward",
            "env_id",
        ]
        self.df = pd.DataFrame(columns=self.columns)
        self.lst = []  # list to store tensors tmp because df is not efficient

    def record_loc(self, agent_pos, step, episode, reward, env_id):
        # tensor_row shape (6,)
        tensor_row = np.array(
            [agent_pos[0], agent_pos[1], step, episode, reward, env_id]
        )
        self.lst.append(tensor_row)

    def save(self):
        # print("Saving activations...")
        new_df = pd.DataFrame(self.lst, columns=self.columns)
        self.df = pd.concat([self.df, new_df], ignore_index=True)
        self.df.to_csv(self.save_dir / "agent_pos.csv", index=False)
        self.lst = []


class eval_mode:
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def hard_update_params(net, target_net):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        # for param, target_param in zip(net.state_dict(), target_net.state_dict()):
        # print(param, "\t", net.state_dict()[param].size())
        # print(target_param, "\t", target_net.state_dict()[target_param].size())
        # target_param[1].data.copy_(param[1].data)
        # print(param)
        # print(param.data)
        # input(...)
        target_param.data.copy_(param.data)


def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step < until


class Once:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._until is None:
            return True
        until = self._until // self._action_repeat
        return step == until


class Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time
        self._last_time = time.time()
        total_time = time.time() - self._start_time
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time


def schedule(schdl, step):
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r"linear\((.+),(.+),(.+)\)", schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        match = re.match(r"step_linear\((.+),(.+),(.+),(.+),(.+)\)", schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)
