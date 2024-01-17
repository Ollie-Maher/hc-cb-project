# HCC agent

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from torchvision import transforms
from torchvision.models import resnet18

from agent.hcc import HCCAgent, RandomShiftsAug, ResEncoder


class RecurrentQNet(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, device):
        super(RecurrentQNet, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        # +1 for the action size
        # self.gru = nn.GRU(state_dim, hidden_dim, batch_first=True)
        # self.gru = nn.GRU(state_dim + state_dim, hidden_dim, batch_first=True)
        self.gru = nn.GRU(state_dim + action_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim)

        self.cerebellum_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            # nn.Linear(hidden_dim, state_dim),
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
        # cb_preds = torch.zeros(batch_size, sequence_length, self.state_dim).to(
        #     self.device
        # )
        cb_preds = torch.zeros(batch_size, sequence_length, self.action_dim).to(
            self.device
        )
        # breakpoint()
        gru_input = torch.cat(
            [x[:, 0:1, :], pred], dim=2
        )  # shappe (batch_size, 1, state_dim + action_dim)
        # gru_input = x[:, 0:1, :]  # shappe (batch_size, 1, state_dim + action_dim)
        # hidden = hidden.squeeze(0)
        # breakpoint()

        for t in range(sequence_length):
            self.gru.requires_grad_(False)
            gru_out, hidden = self.gru(gru_input, hidden)
            # # gru_out goes to fc (ca1)
            # # ca1_out = F.relu(self.fc(gru_out))
            # # pred = self.cerebellum_net(ca1_out)
            pred = self.cerebellum_net(hidden.detach())
            pred = pred.transpose(0, 1)
            # # add random noise to the cerebellum prediction
            # # noise = torch.randn_like(pred) * 1.0 + 0
            # # pred = pred + noise
            if t < sequence_length - 1:
                # gru_input = torch.cat([x[:, t + 1 : t + 2, :], hidden], dim=2)
                gru_input = torch.cat([x[:, t + 1 : t + 2, :], pred.detach()], dim=2)
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
        return torch.zeros(batch_size, 1, self.action_dim, device=self.device)
        # return torch.zeros(batch_size, 1, self.state_dim, device=self.device)


class HCCBioAgent(HCCAgent):
    # def __init__(
    #     self,
    #     **kwargs,
    # ):
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
        super().__init__(
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
        )
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
            # self.encoder = Encoder(obs_shape).to(self.device)
            self.encoder = ResEncoder(obs_shape).to(self.device)
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

    def learn(self, obs, actions, rewards, discount, next_obs, step):
        metrics = dict()
        # print(obs.shape, actions.shape, rewards.shape, next_obs.shape)
        # Update Q network
        # breakpoint()
        q_values, cb_preds, _, _ = self.q_net(obs)
        q_values = q_values.gather(2, actions.unsqueeze(-1))

        with torch.no_grad():
            next_q_values, _, _, _ = self.target_net(next_obs)
            target_next_q_values = next_q_values
            # next_q_values shape is [batch_size, action_dim], getting max(1)[0} will
            # give shape of [batch_size], hence unsqueeze(-1) to get [batch_size, 1]
            # which will match the shape rewards and q_values
            # next_q_values = next_q_values.max(1)[0].view(self.batch_size, 1)
            next_q_values = next_q_values.max(2)[0].unsqueeze(-1)

            # discount will be zero for terminal states, so we don't need to worry to do
            # any masking
            next_q_values = rewards + self.gamma * discount * next_q_values

        q_loss = F.mse_loss(q_values, next_q_values)
        # breakpoint()
        cb_pred_loss = F.mse_loss(cb_preds, target_next_q_values)
        # cb_pred_loss = F.mse_loss(cb_preds, next_obs)
        loss = q_loss + cb_pred_loss

        if self.use_wandb:
            metrics["q_loss"] = q_loss.item()
            metrics["cb_pred_loss"] = cb_pred_loss.item()
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
