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
        # # self.repr_dim = 32 * 35 * 35 # dm_control
        # # self.repr_dim = 32 * 25 * 25  # minigrid partial obs
        # # self.repr_dim = 32 * 7 * 7  # minigrid partial obs 3
        # self.repr_dim = 32 * 9 * 9  # minigrid partial obs 3
        # # self.repr_dim = 32 * 17 * 17  # minigrid full
        # # self.repr_dim = 32 * 38 * 38  # atari
        # # self.repr_dim = 32 * 29 * 29  # neuro_maze
        self.repr_dim = 1024  # 256  # 1024
        self.out_dim = 32 * 9 * 9  # minigrid partial obs 3

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

        self.fc = nn.Linear(self.out_dim, self.repr_dim)
        self.apply(utils.weight_init)

    def forward(self, obs):
        # obs = obs.permute(
        #     0, 3, 1, 2
        # )  # permute can swap all dimensions while transpose only 2
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        # h = h.reshape(-1, self.repr_dim)
        h = self.fc(h)
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
        # self.image_channel = 9
        # x = torch.randn([32] + [9, 84, 84])
        # x = torch.randn([32] + [obs_shape[0], 64, 64])  # neuromaze
        x = torch.randn([32] + [obs_shape[0], 24, 24])  # minigrid
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
        self.repr_dim = 32 * 9 * 9  # minigrid partial obs 3
        # self.repr_dim = 32 * 17 * 17  # minigrid full
        # self.repr_dim = 32 * 39 * 39  # atari
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


# class CA3Net(nn.Module):
#     # def __init__(self, state_dim, hidden_dim, action_dim):
#     #     super(CA3Net, self).__init__()
#     #     self.fc1 = nn.Linear(state_dim, hidden_dim)
#     #     # self.rew_model = nn.Linear(hidden_dim, 1)
#     #     self.fc2 = nn.Linear(hidden_dim, action_dim)

#     # def forward(self, state=None, action=None):
#     #     # x is state and action concatenated
#     #     state = F.relu(self.fc1(state))
#     #     q_values = self.fc2(state)
#     #     return q_values, None, None

#     def __init__(self, state_dim, hidden_dim, action_dim):
#         super(CA3Net, self).__init__()
#         self.fc1 = nn.Linear(state_dim + 1, hidden_dim)
#         self.rew_model = nn.Linear(hidden_dim, 1)
#         self.phi = nn.Linear(hidden_dim, state_dim)

#     def forward(self, state=None, action=None):
#         # x is state and action concatenated
#         conc_state = torch.cat([state, action], dim=1)
#         state = F.relu(self.fc1(conc_state))
#         # x = torch.cat([state, action], dim=1)
#         pred_rew = self.rew_model(state)
#         pred_state = self.phi(state)
#         return state, pred_state, pred_rew


class CA3Net(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, device):
        super(CA3Net, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        # +1 for the action size
        self.gru = nn.GRU(state_dim, hidden_dim, batch_first=True)
        # self.fc = nn.Linear(hidden_dim, state_dim)
        self.fc = nn.Linear(hidden_dim + action_dim, hidden_dim // 2)
        # self.out = nn.Linear(hidden_dim // 2, state_dim)
        self.out = nn.Linear(hidden_dim // 2, hidden_dim)

        self.rew_model = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )

        # self.phi = nn.Linear(hidden_dim, state_dim)
        # self.rew_model = nn.Linear(hidden_dim, 1)

        self.apply(utils.weight_init)  # why

    def forward(self, x=None, hidden=None, action=None):
        # x shape (batch_size, seq_len, state_dim)
        batch_size = x.shape[0]  # obs.size(0)
        hidden = self.init_hidden(batch_size) if hidden is None else hidden

        # concatenate x and action [batch_size, seq_len, state_dim + action_dim]
        # if action is not None:
        #     x_pred = torch.cat([x, action], dim=-1)

        # with torch.no_grad():
        # breakpoint()
        # self.gru.requires_grad_(False)
        gru_out, gru_hidden = self.gru(x, hidden)
        if action is not None:
            x_conc = torch.cat([gru_out.detach(), action], dim=-1)
        # pred_next_state = F.relu(self.fc(x_conc))
        pred_next_state = F.elu(self.fc(x_conc))
        # pred_next_state = F.relu(self.fc(gru_out))
        pred_next_state = self.out(pred_next_state)

        # # predict next state and action
        # pred_next_state = self.phi(gru_out)
        pred_rew = self.rew_model(x_conc)

        # return out, gru_hidden, ca1_out
        return gru_out, pred_next_state, pred_rew, gru_hidden

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_dim, device=self.device)


class QNet(nn.Module):
    # def __init__(self, state_dim, hidden_dim, action_dim):
    #     super(QNet, self).__init__()
    #     self.fc1 = nn.Linear(action_dim, hidden_dim)
    #     # self.fc1 = nn.Linear(state_dim + state_dim, hidden_dim)
    #     # self.fc2 = nn.Linear(hidden_dim, action_dim)
    #     self.fc2 = nn.Linear(hidden_dim, state_dim)

    # def forward(self, state=None):
    #     # conc_state = torch.cat([state, action], dim=1)
    #     pred_state = F.relu(self.fc1(state))
    #     pred_state = self.fc2(pred_state)
    #     return pred_state

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QNet, self).__init__()
        # self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        # self.fc1 = nn.Linear(hidden_dim + state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class HCDYNAAgent:
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
        self.skip_update = False
        self.pred_err = 0.0

        if obs_type == "pixels":
            self.encoder = Encoder(obs_shape).to(self.device)
            # self.encoder = ResEncoder(obs_shape).to(self.device)
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

        self.ca3_net = CA3Net(
            self.encoder.repr_dim, self.hidden_dim, self.action_dim, self.device
        ).to(self.device)
        # self.target_net = CA3Net(
        #     self.encoder.repr_dim, self.hidden_dim, self.action_dim
        # ).to(self.device)
        # self.target_net.load_state_dict(self.ca3_net.state_dict())

        # optimizers
        self.q_net_optim = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.ca3_net_optim = torch.optim.Adam(self.ca3_net.parameters(), lr=self.lr)

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

    def reset(self):
        self.prev_action = torch.zeros(1).to(self.device)
        self.prev_repres = torch.zeros((1, self.encoder.repr_dim)).to(self.device)
        self.gru_hidden = None
        self.pred_err = 0.0
        # pass

    def intrinsic_reward(self, obs, next_obs, action):
        # it should be based on the outcome of ca3_net and scaled accordingly
        state, pred_next_obs, _ = self.ca3_net(x=obs, action=action.unsqueeze(-1))
        # reward = F.mse_loss(pred_next_obs, next_obs)
        reward = torch.linalg.norm(
            pred_next_obs - next_obs, ord=2, dim=-1, keepdim=True
        )
        reward = torch.log(reward + 1.0)

        return reward

    def act(self, obs, step, eval_mode=False, prev_reward=None):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        # breakpoint()
        if np.random.rand() > self.epsilon:
            with torch.no_grad():  # probably don't need this as it is done before act
                repres = self.encoder(obs)  # repres [1, 2592]
                # q_values, self.gru_hidden, ca1_out = self.ca3_net(
                #     repres.unsqueeze(0), self.gru_hidden
                # )  # (1, 1, features_dim) adding extra dim for seq_len
                # state_in = torch.cat([repres, self.prev_action.unsqueeze(0)], dim=-1)
                # q_values, _, _ = self.ca3_net(
                #     state=repres, action=self.prev_action.unsqueeze(0)
                # )

                # # need to create a batch of 2 for the previous time step and current one to pass to CA3 net
                # # of shape [2, 1, features_dim]
                self.prev_action = self.prev_action.type(torch.int64)
                one_hot_action = F.one_hot(
                    self.prev_action, num_classes=self.action_dim
                )  # shape [1, 3]
                # batch_repres = torch.cat(
                #     (self.prev_repres.unsqueeze(0), repres.unsqueeze(0)), dim=0
                # )
                # batch_gru_hidden = self.gru_hidden
                # # torch.cat(
                # #     self.prev_gru_hidden, self.gru_hidden, dim=0
                # # )
                # batch_action = torch.cat(
                #     (one_hot_action.unsqueeze(0), one_hot_action.unsqueeze(0)), dim=0
                # )  # the second action doesn't matter

                # state, pred_repres, pred_rew, self.gru_hidden = self.ca3_net(
                #     x=batch_repres,
                #     hidden=batch_gru_hidden,
                #     action=batch_action,
                # )
                state, pred_repres, pred_rew, self.gru_hidden = self.ca3_net(
                    x=repres.unsqueeze(0),
                    hidden=self.gru_hidden,
                    action=one_hot_action.unsqueeze(0),
                    # action=self.prev_action.unsqueeze(0).unsqueeze(0),
                )
                # conc_state = torch.cat([repres.unsqueeze(0), state], dim=2)
                conc_state = state  # select the current state
                q_values = self.q_net(conc_state)
                # conc_repres = torch.cat([repres, pred_repres], dim=1)
                # conc_repres = torch.cat([self.prev_repres, repres], dim=1)
                # q_values = self.q_net(repres)
                # q_values = self.q_net(repres.unsqueeze(0))
                # q_values = self.q_net(conc_repres)
                # q_values = self.q_net(pred_repres)

                # # predict next state and reward
                # state_err = torch.linalg.norm(
                #     pred_repres[0].squeeze(0) - repres, ord=2, dim=-1, keepdim=True
                # ).mean()
                # rew_err = torch.linalg.norm(
                #     pred_rew[0].squeeze(0) - prev_reward, ord=2, dim=-1, keepdim=True
                # ).mean()
                # if step > 10000 and prev_reward == 1.0:
                #     print(f"pred rew: {pred_rew[0].squeeze(0)}, rew: {prev_reward}")
                #     print(f"state err: {state_err}, rew err: {rew_err}")
                # self.pred_err += state_err + (rew_err * 1000.0)
                # # if step > 10000:
                # #     print(f"pred err: {self.pred_err}")
                # #     print(f"state err: {state_err}, rew err: {rew_err}")

                # if self.pred_err < 500.0:
                #     # if not self.skip_update:
                #     #     print(f"setting skip to true, pred err: {self.pred_err}")
                #     #     print(f"pred rew: {pred_rew[0].squeeze(0)}, rew: {prev_reward}")
                #     self.skip_update = True
                # else:
                #     # if self.skip_update:
                #     #     print(f"setting skip to false, pred err: {self.pred_err}")
                #     #     print(f"pred rew: {pred_rew[0].squeeze(0)}, rew: {prev_reward}")
                #     self.skip_update = False

            action = q_values.argmax().item()
        else:
            action = np.random.randint(self.action_dim)

        self.prev_action[0] = action
        # self.prev_repres = repres
        with torch.no_grad():
            self.prev_repres = self.encoder(obs)
        return action, None, None, None, None, None

    def learn(self, obs, actions, rewards, discount, next_obs, step):
        metrics = dict()
        # print(obs.shape, actions.shape, rewards.shape, next_obs.shape)
        # breakpoint()

        one_hot_actions = F.one_hot(actions, num_classes=self.action_dim).float()

        # Update CA3 network
        # with torch.no_grad():
        #     next_state, pred_next_obs, _ = self.ca3_net(
        #         state=next_obs, action=actions.unsqueeze(-1)
        #     )
        # ca3_in = torch.cat([obs, actions.unsqueeze(-1)], dim=-1)
        # q_values, _, _ = self.ca3_net(state=obs, action=actions.unsqueeze(-1).detach())
        # state, pred_obs, pred_rew, _ = self.ca3_net(x=obs, action=actions.unsqueeze(-1))
        state, pred_obs, pred_rew, _ = self.ca3_net(x=obs, action=one_hot_actions)

        # concatenate state and predicted state for CA1
        # conc_state = torch.cat([obs, pred_obs.detach()], dim=1)
        # conc_state = torch.cat([obs, pred_obs], dim=1)
        # conc_next_state = torch.cat([next_obs, pred_next_obs], dim=1)
        conc_state = torch.cat([obs, state.detach()], dim=2)
        # conc_state = torch.cat([obs, state], dim=2)
        # conc_state = obs

        # Update Q network
        # q_values = self.q_net(obs)
        # q_values = self.q_net(obs.detach())
        # q_values = self.q_net(pred_obs)
        # pred_obs = self.q_net(state=q_values)
        # q_values = self.q_net(conc_state)
        q_values = self.q_net(state)

        with torch.no_grad():
            next_action = q_values.argmax(dim=2)
            one_hot_next_action = F.one_hot(next_action, num_classes=self.action_dim)
            # next_action = q_values.argmax(dim=1)
            # print(next_action.shape)
            # next_action = q_values.argmax().item()
            next_state, pred_next_obs, _, _ = self.ca3_net(
                # state=next_obs, action=actions.unsqueeze(-1)
                x=next_obs,
                action=one_hot_next_action,
                # action=one_hot_actions,
                # action=next_action.unsqueeze(-1),
            )
            # conc_next_state = torch.cat([next_obs, pred_next_obs], dim=1)
            conc_next_state = torch.cat([next_obs, next_state], dim=2)
            # conc_next_state = next_state
        # we unsqueeze(-1) the actions to get shape (batch_size, 1) which matchtes
        # rewards shape of (batch_size, 1). Unsqueeze is not required and alternatively
        # we can make sure rewards to be shape of (batch_size)
        # q_values = q_values.gather(1, actions.unsqueeze(-1))
        q_values = q_values.gather(2, actions.unsqueeze(-1))

        with torch.no_grad():
            # next_q_values = self.target_net(self.encoder(next_obs))
            # next_q_values = self.target_net(next_obs)
            next_q_values = self.target_net(next_state)
            # next_q_values = self.target_net(conc_next_state)
            # next_q_values = self.target_net(pred_next_obs)
            # next_q_values, _, _ = self.target_net(state=next_obs, action=None)
            # next_q_values shape is [batch_size, action_dim], getting max(1)[0} will
            # give shape of [batch_size], hence unsqueeze(-1) to get [batch_size, 1]
            # which will match the shape rewards and q_values
            # next_q_values = next_q_values.max(1)[0].view(self.batch_size, 1)
            # next_q_values = next_q_values.max(1)[0].unsqueeze(-1)
            next_q_values = next_q_values.max(2)[0].unsqueeze(-1)

            # discount will be zero for terminal states, so we don't need to worry to do
            # any masking
            next_q_values = rewards + self.gamma * discount * next_q_values

        # print(q_values.shape, next_q_values.shape)
        q_loss = F.mse_loss(q_values, next_q_values)

        # state_loss = F.mse_loss(pred_obs, next_obs)
        state_loss = torch.linalg.norm(
            # pred_obs - next_obs, ord=2, dim=-1, keepdim=True
            pred_obs - next_state,
            ord=2,
            dim=-1,
            keepdim=True,
        ).mean()
        rew_loss = torch.linalg.norm(
            pred_rew - rewards, ord=2, dim=-1, keepdim=True
        ).mean()

        # neg_sample = torch.roll(obs, 1, dims=1)
        # # loss_neg_rnd = torhc.mean(
        # #     torch.exp(-torch.linalg.norm(obs - neg_sample, ord=2, dim=-1, keepdim=True))
        # # )
        # loss_neg_rnd = torch.mean(torch.exp(-5 * torch.norm(obs - neg_sample, dim=-1)))
        # loss_neg_neigh = torch.mean(torch.exp(-5 * torch.norm(obs - next_obs, dim=-1)))
        # # loss_neg_neigh = torch.mean(torch.exp(-torch.linalg.norm(obs - next_obs, ord=2, dim=-1, keepdim=True)))
        # loss_pos = torch.mean(F.mse_loss(pred_obs, next_obs))

        zp_loss = ((pred_obs - next_state) ** 2).sum(-1).mean()

        # rew_loss = F.mse_loss(pred_rew, rewards)
        # scale state loss
        # pred_loss = state_loss  # * 0.001  # + rew_loss
        # loss = q_loss + state_loss + rew_loss
        # loss = 1e-4 * q_loss + 1e-5 * loss_neg_rnd + 1e-6 * loss_pos
        # loss = 1e-4 * q_loss
        loss = q_loss + zp_loss
        # loss = q_loss

        if self.use_wandb:
            metrics["q_loss"] = q_loss.item()
            metrics["q_val"] = q_values.mean().item()
            metrics["q_target"] = next_q_values.mean().item()
            # metrics["pred_loss"] = pred_loss.item()
            metrics["state_loss"] = state_loss.item()
            metrics["rew_loss"] = rew_loss.item()

        if self.encoder_optim is not None:
            self.encoder_optim.zero_grad()
        self.q_net_optim.zero_grad()
        self.ca3_net_optim.zero_grad()
        # q_loss.backward()
        loss.backward()
        # pred_loss.backward()
        self.q_net_optim.step()
        self.ca3_net_optim.step()
        if self.encoder_optim is not None:
            self.encoder_optim.step()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        # if self.skip_update:
        #     # print(f"skipping update, pred err: {self.pred_err}")
        #     return metrics

        batch = next(replay_iter)
        obs, actions, rewards, discount, next_obs = utils.to_torch(batch, self.device)
        # actions = actions[:, 0]  # need to fix this in the replay buffer or wrapper
        actions = actions.squeeze(-1)
        actions = actions.type(torch.int64)

        # augment, put batch and seq_len together
        B, S, C, H, W = obs.shape

        # obs = self.aug(obs.reshape(B * S, C, H, W).float())
        # next_obs = self.aug(next_obs.reshape(B * S, C, H, W).float())
        obs = obs.reshape(B * S, C, H, W).float()
        next_obs = next_obs.reshape(B * S, C, H, W).float()

        # # encode
        obs = self.encoder(obs)
        obs = obs.reshape(B, S, -1)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)
            next_obs = next_obs.reshape(B, S, -1)

        # # uncomment these for framse stack of 3
        # obs = obs.squeeze(1)
        # next_obs = next_obs.squeeze(1)
        # actions = actions.squeeze(1)
        # rewards = rewards.squeeze(1)
        # discount = discount.squeeze(1)
        # # print(obs.shape, actions.shape, rewards.shape, discount.shape, next_obs.shape)

        # # # augment
        # # obs = self.aug(obs.float())
        # # next_obs = self.aug(next_obs.float())
        # # encode
        # obs = self.encoder(obs)
        # with torch.no_grad():
        #     next_obs = self.encoder(next_obs)

        # # intrinsic reward
        # with torch.no_grad():
        #     rewards = self.intrinsic_reward(obs, next_obs, actions)

        if self.use_wandb:
            metrics["batch_reward"] = rewards.mean().item()

        # Update Q network
        metrics.update(self.learn(obs, actions, rewards, discount, next_obs, step))

        # Update target network
        with torch.no_grad():
            # utils.soft_update_params(self.ca3_net, self.target_net, self.tau)
            utils.soft_update_params(self.q_net, self.target_net, self.tau)

        return metrics
