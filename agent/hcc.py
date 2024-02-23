# HCC agent

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from torchvision import transforms
from torchvision.models import resnet18


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(
            -1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype
        )[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(
            0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype
        )
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        # self.repr_dim = 32 * 35 * 35 # dm_control
        # self.repr_dim = 32 * 25 * 25 # minigrid partial obs
        self.repr_dim = 32 * 9 * 9  # minigrid partial obs
        # self.repr_dim = 32 * 1 * 1  # minigrid partial obs
        # self.repr_dim = 32 * 17 * 17  # minigrid full
        # self.repr_dim = 32 * 38 * 38  # atari
        # self.repr_dim = 32 * 29 * 29  # neuro_maze

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 2, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 2, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 2, stride=1),
            nn.ReLU(),
        )

        self.apply(utils.weight_init)

    def forward(self, obs):
        # obs = obs.permute(
        #     0, 3, 1, 2
        # )  # permute can swap all dimensions while transpose only 2
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        # h = h.reshape(-1, self.repr_dim)
        return h


class ResEncoder(nn.Module):
    def __init__(self, obs_shape):
        super(ResEncoder, self).__init__()
        self.model = resnet18(pretrained=True)
        self.transform = transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224)]
        )

        for param in self.model.parameters():
            param.requires_grad = False

        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Identity()
        self.repr_dim = 256  # 1024
        self.image_channel = 3
        # x = torch.randn([32] + [9, 84, 84])
        # x = torch.randn([32] + [obs_shape[0], 64, 64])  # neuro_maze
        x = torch.randn([32] + [obs_shape[0], 24, 24])  # minigrid
        # x = torch.randn([32] + [obs_shape[0], 65, 65])  # minigrid
        with torch.no_grad():
            out_shape = self.forward_conv(x).shape
        self.out_dim = out_shape[1]
        self.fc = nn.Linear(self.out_dim, self.repr_dim)
        self.ln = nn.LayerNorm(self.repr_dim)
        #
        # Initialization
        nn.init.orthogonal_(self.fc.weight.data)
        self.fc.bias.data.fill_(0.0)

    @torch.no_grad()
    def forward_conv(self, obs, flatten=True):
        obs = obs / 255.0 - 0.5

        for name, module in self.model._modules.items():
            obs = module(obs)
            if name == "layer2":
                break

        if flatten:
            conv = obs.view(obs.size(0), -1)

        return conv

    def forward(self, obs):
        # print(f"obs shape: {obs.shape}")
        conv = self.forward_conv(obs)
        out = self.fc(conv)
        out = self.ln(out)
        # obs = self.model(self.transform(obs.to(torch.float32)) / 255.0 - 0.5)
        return out


class RecurrentQNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, device):
        super(RecurrentQNet, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        # +1 for the action size
        # self.gru = nn.GRU(state_dim, hidden_dim, batch_first=True)
        self.gru = nn.GRU(state_dim + state_dim, hidden_dim, batch_first=True)
        # self.gru = nn.GRU(state_dim + action_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

        self.cerebellum_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # nn.Linear(hidden_dim, action_dim),
            nn.Linear(hidden_dim, state_dim),
        )

        # self.phi = nn.Linear(hidden_dim, state_dim)
        # self.rew_model = nn.Linear(hidden_dim, 1)

        self.apply(utils.weight_init)  # why

    def forward(self, x, hidden=None, cereb_pred=None):
        # x shape (batch_size, seq_len, state_dim)
        batch_size = x.shape[0]  # obs.size(0)
        sequence_length = x.shape[1]
        hidden = self.init_hidden(batch_size) if hidden is None else hidden
        pred = self.init_pred(batch_size) if cereb_pred is None else cereb_pred

        out = torch.zeros(batch_size, sequence_length, self.hidden_dim).to(self.device)
        cb_preds = torch.zeros(batch_size, sequence_length, self.state_dim).to(
            self.device
        )
        # cb_preds = torch.zeros(batch_size, sequence_length, self.action_dim).to(
        #     self.device
        # )
        # breakpoint()
        gru_input = torch.cat(
            [x[:, 0:1, :], pred], dim=2
        )  # shappe (batch_size, 1, state_dim + action_dim)
        # gru_input = x[:, 0:1, :]  # shappe (batch_size, 1, state_dim + action_dim)
        # hidden = hidden.squeeze(0)
        # breakpoint()

        for t in range(sequence_length):
            # with torch.no_grad():
            self.gru.requires_grad_(False)
            # self.gru.weight_ih_l0.requires_grad_(True)
            gru_out, hidden = self.gru(gru_input, hidden)
            # gru_out goes to fc (ca1)
            # ca1_out = F.relu(self.fc(gru_out))
            # pred = self.cerebellum_net(ca1_out)
            pred = self.cerebellum_net(hidden)
            pred = pred.transpose(0, 1)
            # add random noise to the cerebellum prediction
            # noise = torch.randn_like(pred) * 1.0 + 0
            # pred = pred + noise
            if t < sequence_length - 1:
                # gru_input = torch.cat([x[:, t + 1 : t + 2, :], hidden], dim=2)
                gru_input = torch.cat([x[:, t + 1 : t + 2, :], pred], dim=2)
            # gru_input = x[:, t + 1 : t + 2, :]
            out[:, t : t + 1, :] = gru_out
            # out[:, t : t + 1, :] = ca1_out
            cb_preds[:, t : t + 1, :] = pred

        ca1_out = F.relu(self.fc(out))
        out = self.out(ca1_out)
        # breakpoint()

        # # predict next state and action
        # pred_next_state = self.phi(gru_out)
        # pred_rew = self.rew_model(gru_out)

        return out, cb_preds, hidden, ca1_out

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_dim, device=self.device)

    def init_pred(self, batch_size):
        # return torch.zeros(batch_size, 1, self.action_dim, device=self.device)
        return torch.zeros(batch_size, 1, self.state_dim, device=self.device)


class HCCAgent:
    def __init__(
        self,
        name,
        obs_type,
        obs_shape,
        action_shape,
        hidden_dim,
        lr,
        gamma,
        batch_size,
        tau,
        update_every_steps,
        device,
        use_wandb,
        nstep,
        sequence_length,
        epsilon,
    ):
        self.name = name
        self.obs_type = obs_type
        self.obs_shape = obs_shape
        # self.action_dim = action_shape[0]
        # self.action_dim = 6
        self.action_dim = 3
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.update_every_steps = update_every_steps
        self.device = device
        self.use_wandb = use_wandb
        self.epsilon = epsilon
        self.gru_hidden = None
        self.cereb_pred = None

        if obs_type == "pixels":
            self.encoder = Encoder(obs_shape).to(self.device)
            # self.encoder = ResEncoder(obs_shape).to(self.device)
            self.obs_dim = self.encoder.repr_dim

        else:
            self.encoder = nn.Identity()
            self.obs_dim = obs_shape[0]

        self.q_net = RecurrentQNet(
            self.encoder.repr_dim, self.hidden_dim, self.action_dim, self.device
        ).to(self.device)
        self.target_net = RecurrentQNet(
            self.encoder.repr_dim, self.hidden_dim, self.action_dim, self.device
        ).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        # optimizers
        self.q_net_optim = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)

        if obs_type == "pixels":
            self.encoder_optim = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
        else:
            self.encoder_optim = None

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.target_net.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.q_net.train(training)

    def init_from(self, other):
        utils.hard_update_params(other.encoder, self.encoder)
        utils.hard_update_params(other.q_net, self.q_net)
        utils.hard_update_params(other.target_net, self.target_net)

    def reset(self):
        self.gru_hidden = None
        self.cereb_pred = None

    def act(self, obs, step, eval_mode=False, neurons_reset_idx=None):
        # self.epsilon = 0.55
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        if np.random.rand() > self.epsilon:
            with torch.no_grad():  # probably don't need this as it is done before act
                features = self.encoder(obs)  # (1, 32*29*29)
                q_values, self.cereb_pred, self.gru_hidden, ca1_out = self.q_net(
                    x=features.unsqueeze(0),
                    hidden=self.gru_hidden,
                    cereb_pred=self.cereb_pred,
                    neurons_reset_idx=neurons_reset_idx,
                    # cereb_pred=None,
                )
            action = q_values.argmax().item()
            return action, q_values, self.cereb_pred, self.gru_hidden, ca1_out, features
        else:
            action = np.random.randint(self.action_dim)
            return action, None, None, None, None, None

        # return action

    def learn(self, obs, actions, rewards, discount, next_obs, step):
        metrics = dict()
        # print(obs.shape, actions.shape, rewards.shape, next_obs.shape)
        # Update Q network
        # breakpoint()
        q_values, cb_preds, _, _ = self.q_net(obs)
        q_values = q_values.gather(2, actions.unsqueeze(-1))

        with torch.no_grad():
            next_q_values, _, _, _ = self.target_net(next_obs)
            # next_q_values shape is [batch_size, action_dim], getting max(1)[0} will
            # give shape of [batch_size], hence unsqueeze(-1) to get [batch_size, 1]
            # which will match the shape rewards and q_values
            # next_q_values = next_q_values.max(1)[0].view(self.batch_size, 1)
            next_q_values = next_q_values.max(2)[0].unsqueeze(-1)

            # discount will be zero for terminal states, so we don't need to worry to do
            # any masking
            next_q_values = rewards + self.gamma * discount * next_q_values

        loss = F.mse_loss(q_values, next_q_values)
        # cb_pred_loss = F.mse_loss(cb_preds, next_q_values)
        # cb_pred_loss = F.mse_loss(cb_preds, next_obs)
        # loss = q_loss + cb_pred_loss

        if self.use_wandb:
            # metrics["q_loss"] = q_loss.item()
            # metrics["cb_pred_loss"] = cb_pred_loss.item()
            metrics["loss"] = loss.item()
            metrics["q_val"] = q_values.mean().item()
            metrics["q_target"] = next_q_values.mean().item()

        if self.encoder_optim is not None:
            self.encoder_optim.zero_grad()
        self.q_net_optim.zero_grad()
        loss.backward()
        for param in self.q_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.q_net_optim.step()
        # if self.encoder_optim is not None:
        #     self.encoder_optim.step()

        return metrics

    def update(self, replay_iter, step, test=False):
        metrics = dict()

        if test:
            return metrics
        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, actions, rewards, discount, next_obs = utils.to_torch(batch, self.device)
        # print(obs.shape, actions.shape, rewards.shape, discount.shape, next_obs.shape)
        # actions = actions[:, 0]  # need to fix this in the replay buffer or wrapper
        actions = actions.squeeze(-1)
        actions = actions.type(torch.int64)

        # augment, put batch and seq_len together
        B, S, C, H, W = obs.shape

        obs = self.aug(obs.reshape(B * S, C, H, W).float())
        next_obs = self.aug(next_obs.reshape(B * S, C, H, W).float())
        # obs = obs.reshape(B * S, C, H, W).float()
        # next_obs = next_obs.reshape(B * S, C, H, W).float()

        # # encode
        obs = self.encoder(obs)
        obs = obs.reshape(B, S, -1)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)
            next_obs = next_obs.reshape(B, S, -1)

        if self.use_wandb:
            metrics["batch_reward"] = rewards.mean().item()

        # Update Q network
        metrics.update(self.learn(obs, actions, rewards, discount, next_obs, step))

        # Update target network
        with torch.no_grad():
            utils.soft_update_params(self.q_net, self.target_net, self.tau)

        return metrics
