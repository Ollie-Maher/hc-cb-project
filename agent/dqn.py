# DQN agent

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
        # self.repr_dim = 32 * 17 * 17  # minigrid full
        # self.repr_dim = 32 * 38 * 38  # atari
        self.repr_dim = 32 * 29 * 29  # neuro_maze

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 2, stride=2),
            # nn.Conv2d(9, 32, 2, stride=2),
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
        self.repr_dim = 1024
        self.image_channel = 3
        # x = torch.randn([32] + [9, 84, 84])
        x = torch.randn([32] + [obs_shape[0], 64, 64])
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
        # print(f"obs shape: {obs.shape}")
        obs = obs / 255.0 - 0.5
        # time_step = obs.shape[1] // self.image_channel
        # obs = obs.view(
        #     obs.shape[0], time_step, self.image_channel, obs.shape[-2], obs.shape[-1]
        # )
        # obs = obs.view(
        #     obs.shape[0] * time_step, self.image_channel, obs.shape[-2], obs.shape[-1]
        # )

        for name, module in self.model._modules.items():
            obs = module(obs)
            if name == "layer2":
                break

        # conv = obs.view(
        #     obs.size(0) // time_step, time_step, obs.size(1), obs.size(2), obs.size(3)
        # )
        # conv_current = conv[:, 1:, :, :, :]
        # conv_prev = conv_current - conv[:, : time_step - 1, :, :, :].detach()
        # conv = torch.cat([conv_current, conv_prev], axis=1)
        # conv = conv.view(
        #     conv.size(0), conv.size(1) * conv.size(2), conv.size(3), conv.size(4)
        # )
        # print(obs.shape)
        if flatten:
            conv = obs.view(obs.size(0), -1)

        # print(conv.shape)

        return conv

    def forward(self, obs):
        # print(f"obs shape: {obs.shape}")
        conv = self.forward_conv(obs)
        out = self.fc(conv)
        out = self.ln(out)
        # obs = self.model(self.transform(obs.to(torch.float32)) / 255.0 - 0.5)
        return out


class QNetEncoder(nn.Module):
    def __init__(self, obs_shape, hidden_dim, action_dim):
        super().__init__()

        assert len(obs_shape) == 3
        # self.repr_dim = 32 * 35 * 35 # dm_control
        # self.repr_dim = 32 * 25 * 25 # minigrid partial obs
        # self.repr_dim = 32 * 17 * 17  # minigrid full
        # self.repr_dim = 32 * 39 * 39  # atari
        self.repr_dim = 32 * 29 * 29  # neuro_maze

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 2, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 2, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.repr_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0
        h = self.convnet(obs)
        return h


class QNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        # self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = self.fc3(x)
        return x


class DQNAgent:
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
        self.action_dim = 6
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.update_every_steps = update_every_steps
        self.device = device
        self.use_wandb = use_wandb
        self.epsilon = epsilon

        if obs_type == "pixels":
            # self.encoder = Encoder(obs_shape).to(self.device)
            self.encoder = ResEncoder(obs_shape).to(self.device)
            self.obs_dim = self.encoder.repr_dim

        else:
            self.encoder = nn.Identity()
            self.obs_dim = obs_shape[0]

        # self.q_net = QNetEncoder(self.obs_shape, self.hidden_dim, self.action_dim).to(
        #     self.device
        # )
        # self.target_net = QNetEncoder(
        #     self.obs_shape, self.hidden_dim, self.action_dim
        # ).to(self.device)
        self.q_net = QNet(self.encoder.repr_dim, self.hidden_dim, self.action_dim).to(
            self.device
        )
        self.target_net = QNet(
            self.encoder.repr_dim, self.hidden_dim, self.action_dim
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

    def act(self, obs, step):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        if np.random.rand() > self.epsilon:
            with torch.no_grad():  # probably don't need this as it is done before act
                q_values = self.q_net(self.encoder(obs))
                # q_values = self.q_net(obs)
            action = q_values.argmax().item()
        else:
            action = np.random.randint(self.action_dim)

        return action

    def learn(self, obs, actions, rewards, discount, next_obs, step):
        metrics = dict()
        # print(obs.shape, actions.shape, rewards.shape, next_obs.shape)
        # Update Q network
        # q_values = self.q_net(self.encoder(obs))
        q_values = self.q_net(obs)
        # we unsqueeze(-1) the actions to get shape (batch_size, 1) which matchtes
        # rewards shape of (batch_size, 1). Unsqueeze is not required and alternatively
        # we can make sure rewards to be shape of (batch_size)
        q_values = q_values.gather(1, actions.unsqueeze(-1))

        with torch.no_grad():
            # next_q_values = self.target_net(self.encoder(next_obs))
            next_q_values = self.target_net(next_obs)
            # next_q_values shape is [batch_size, action_dim], getting max(1)[0} will
            # give shape of [batch_size], hence unsqueeze(-1) to get [batch_size, 1]
            # which will match the shape rewards and q_values
            # next_q_values = next_q_values.max(1)[0].view(self.batch_size, 1)
            next_q_values = next_q_values.max(1)[0].unsqueeze(-1)
            # next_q_values = next_q_values.max(1)[0]

            # discount will be zero for terminal states, so we don't need to worry to do
            # any masking
            next_q_values = rewards + self.gamma * discount * next_q_values

        # print(q_values.shape, next_q_values.shape)
        q_loss = F.mse_loss(q_values, next_q_values)

        if self.use_wandb:
            metrics["q_loss"] = q_loss.item()
            metrics["q_val"] = q_values.mean().item()
            metrics["q_target"] = next_q_values.mean().item()

        if self.encoder_optim is not None:
            self.encoder_optim.zero_grad()
        self.q_net_optim.zero_grad()
        q_loss.backward()
        self.q_net_optim.step()
        if self.encoder_optim is not None:
            self.encoder_optim.step()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, actions, rewards, discount, next_obs = utils.to_torch(batch, self.device)
        # actions = actions[:, 0]  # need to fix this in the replay buffer or wrapper
        actions = actions.type(torch.int64)

        # reshape
        B, T, C, H, W = obs.shape
        # obs = obs.reshape(B * T, C, H, W)
        # next_obs = next_obs.reshape(B * T, C, H, W)
        obs = obs.squeeze(1)
        next_obs = next_obs.squeeze(1)
        actions = actions.squeeze(1)
        # actions = actions.unsqueeze(-1)
        # print(actions.shape)
        rewards = rewards.squeeze(1)
        discount = discount.squeeze(1)
        # print(obs.shape, actions.shape, rewards.shape, discount.shape, next_obs.shape)

        # augment
        obs = self.aug(obs.float())
        next_obs = self.aug(next_obs.float())
        # encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        # Update Q network
        metrics.update(self.learn(obs, actions, rewards, discount, next_obs, step))

        # Update target network
        with torch.no_grad():
            utils.soft_update_params(self.q_net, self.target_net, self.tau)

        return metrics
