import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import sys

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"
os.environ["HYDRA_FULL_ERROR"] = "1"

import random
from collections import OrderedDict
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from dm_env import specs

import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark = True

from wrappers import make_env


class Workspace:
    def __init__(self, cfg):
        self.cfg = cfg

        self.work_dir = Path.cwd()
        self.device = torch.device(cfg.device)

        # Weigths and biases (wandb) logger
        if cfg.use_wandb:
            exp_name = " ".join(
                [cfg.experiment, cfg.agent.name, cfg.task, cfg.obs_type, str(cfg.seed)]
            )
            wandb.init(project="rl-algo", group=cfg.agent.name, name=exp_name)

        self.logger = Logger(self.work_dir, use_tb=cfg.use_tb, use_wandb=cfg.use_wandb)

        self.train_envs = make_env(cfg.task, cfg.seed, cfg.frame_stack, test=True)
        # self.train_env = make_env(cfg.task, cfg.seed, cfg.frame_stack)
        self.train_env = self.train_envs[0]
        self.eval_envs = make_env(cfg.task, cfg.seed, cfg.frame_stack, test=True)
        # self.eval_env = make_env(cfg.task, cfg.seed, cfg.frame_stack)
        self.eval_env = self.eval_envs[0]
        # self.eval_env = self.train_env

        # create agent
        print(self.train_env.observation_spec().name)
        print(self.train_env.observation_spec())
        print(self.train_env.action_spec())
        self.agent = hydra.utils.instantiate(
            cfg.agent,
            obs_type=cfg.obs_type,
            obs_shape=self.train_env.observation_spec().shape,
            action_shape=self.train_env.action_spec().shape,
        )
        print("agent created")
        # initialize from pretrained
        if cfg.snapshot_ts > 0:
            # pretrained_agent_model = self.load_snapshot()["agent"]
            # print(pretrained_agent_model)
            # input(...)
            # self.agent.init_from(pretrained_agent_model)
            snapshot_base_dir = Path(self.cfg.snapshot_load_dir)
            domain = self.cfg.domain
            snapshot_dir = (
                snapshot_base_dir
                / self.cfg.obs_type
                / domain
                / self.cfg.agent.name
                / self.cfg.experiment
                / f"{self.cfg.seed}"
            )

            snapshot = (
                snapshot_dir
                / f"snapshot_{self.cfg.snapshot_ts}_{self.cfg.experiment}.pt"
            )
            # pretrained_agent_model_pth =
            print(snapshot)
            self.agent = torch.load(snapshot)["agent"]
            self.agent.train(False)
            print("agent loaded")

        # get meta specs
        # meta_specs = self.agent.get_meta_specs()
        meta_specs = tuple()
        # create replay buffer
        data_specs = (
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            specs.Array((1,), np.float32, "reward"),
            specs.Array((1,), np.float32, "discount"),
        )

        # # create data storage
        # self.replay_storage = ReplayBufferStorage(
        #     data_specs, meta_specs, self.work_dir / "buffer"
        # )

        # # create replay buffer
        # self.create_replay_buffer()

        # create video recorders
        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None,
            # render_size=64,
            # fps=10,
            # 0 for top down view, 3 for ego view
            camera_id=3 if "quadruped" not in self.cfg.domain else 2,
            use_wandb=self.cfg.use_wandb,
        )
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if cfg.save_train_video else None,
            camera_id=0 if "quadruped" not in self.cfg.domain else 2,
            use_wandb=self.cfg.use_wandb,
        )

        # create agent_stats
        self.agent_ca3_stats = utils.Activations(
            self.work_dir if cfg.save_stats else None
        )
        self.agent_ca1_stats = utils.Activations(
            self.work_dir if cfg.save_stats else None
        )
        self.agent_stats_pos = utils.AgentPos(self.work_dir if cfg.save_stats else None)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        self._switch_step = 0

    @property
    def global_step(self):
        return self._global_step

    @property
    def switch_step(self):
        return self._switch_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    # @property
    def create_replay_buffer(self):
        self.replay_loader = make_replay_loader(
            self.replay_storage,
            self.cfg.replay_buffer_size,
            self.cfg.batch_size,
            self.cfg.replay_buffer_num_workers,
            False,
            self.cfg.nstep,
            self.cfg.discount,
            self.cfg.sequence_length,
        )
        self._replay_iter = None

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        # meta = self.agent.init_meta()
        meta = OrderedDict()
        print("evaluating")
        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.agent.reset()  # reset agent hidden state
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action, q_val, cb_pred, ca3_out, ca1_out, feats = self.agent.act(
                        time_step.observation["image"], self.global_step, eval_mode=True
                    )
                    # print("action", action)
                time_step = self.eval_env.step(action)
                # self.eval_env.render()
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f"{self.global_frame}.mp4")

        with self.logger.log_and_dump_ctx(self.global_frame, ty="eval") as log:
            log("episode_reward", total_reward / episode)
            log("episode_length", step * self.cfg.action_repeat / episode)
            log("episode", self.global_episode)
            log("step", self.global_step)

    def train(self):
        # predicates
        train_until_step = utils.Until(
            self.cfg.num_train_frames, self.cfg.action_repeat
        )
        seed_until_step = utils.Until(self.cfg.num_seed_frames, self.cfg.action_repeat)
        eval_every_step = utils.Every(
            self.cfg.eval_every_frames, self.cfg.action_repeat
        )

        episode_step, episode_reward = 0, 0
        reward_switch = 1
        switch_env = False
        time_step = self.train_env.reset()
        # meta = self.agent.init_meta()
        meta = OrderedDict()
        # self.replay_storage.add(time_step, meta)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f"{self.global_frame}.mp4")
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(
                        self.global_frame, ty="train"
                    ) as log:
                        log("fps", episode_frame / elapsed_time)
                        log("total_time", total_time)
                        log("episode_reward", episode_reward)
                        log("episode_length", episode_frame)
                        log("episode", self.global_episode)
                        # log("buffer_size", len(self.replay_storage))
                        log("step", self.global_step)

                # reset env
                # switch north vs south env after 600 episodes
                if self.global_episode % 100 == 0:
                    # alternate each trial between north and south
                    # if self.global_episode % 1 == 0:
                    # if reward_switch >= 30:
                    # print("SWITCHING ENV")
                    # switch_env = not switch_env
                    reward_switch = 1
                    # switch_env = random.choice([True, False])
                    switch_env = False
                    # switch_env = True  # set to True to switch env

                    # self.replay_storage.reset()
                    # self.create_replay_buffer()
                    # self._switch_step = 0
                    self.train_env = (
                        self.train_envs[1] if switch_env else self.train_envs[0]
                    )
                    self.eval_env = (
                        self.eval_envs[1] if switch_env else self.eval_envs[0]
                    )

                self.agent_stats_pos.record_loc(
                    time_step.observation["agent_pos"],
                    episode_step,
                    self.global_episode,
                    episode_reward,
                    switch_env,
                )
                time_step = self.train_env.reset()
                # obs = OrderedDict({"image": time_step.observation["image"]})
                # time_step = time_step._replace(observation=obs)
                # self.replay_storage.add(time_step, meta)
                self.train_video_recorder.init(time_step.observation)
                # # try to save snapshot
                # if self.global_episode in self.cfg.snapshots:
                #     print("SAVING SNAPSHOT")
                #     self.save_snapshot()

                episode_step = 0
                episode_reward = 0
                self.agent.reset()  # reset agent, set rnn hidden to None
                self.agent_stats_pos.save()
                self.agent_ca3_stats.save("ca3")
                self.agent_ca1_stats.save("ca1")

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log(
                    "eval_total_time", self.timer.total_time(), self.global_frame
                )
                self.eval()

            # set weights corresponding to neuron_indices to zero
            full_array = np.arange(512)
            place_idx = np.array(self.cfg.neurons_indices)
            border_idx = np.array(self.cfg.bdc_neurons_indices)
            feature_idx = np.concatenate((place_idx, border_idx))
            # idx = np.setdiff1d(full_array, feature_idx)
            # ony first time select random neurons
            if self.global_step == 0:
                print("### Randomly selecting neurons ###")
                # idx = np.setdiff1d(full_array, feature_idx)  #
                # idx = np.random.choice(idx, 181, replace=False)  # max 331
                # idx = np.random.choice(feature_idx, 25, replace=False)
                # idx = feature_idx  # 181
                # idx = place_idx # 125
                # print(len(idx))
                # print("idx", idx)
                idx = None

                # with torch.no_grad():
                #     # for idx in self.cfg.neurons_indices:
                #     self.agent.q_net.fc.weight.data[:, idx] = 0
                #     self.agent.q_net.fc.bias.data[idx] = 0

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action, q_val, cb_pred, ca3_out, ca1_out, feats = self.agent.act(
                    time_step.observation["image"],
                    self.global_step,
                    eval_mode=False,
                    neurons_reset_idx=idx,
                )
                # if ca1_out is not None:
                #     print("ca1_out", ca1_out.shape)

            # try to update the agent
            if not seed_until_step(self.global_step) and (
                not seed_until_step(self.switch_step)
            ):
                metrics = self.agent.update(None, self.global_step, test=True)
                self.logger.log_metrics(metrics, self.global_frame, ty="train")

            # from PIL import Image

            # print(type(time_step.observation))
            # im = Image.fromarray(time_step.observation.T, "RGB")
            # im.save("your_file.png")
            # input("Press Enter to continue...")
            self.agent_stats_pos.record_loc(
                time_step.observation["agent_pos"],
                episode_step,
                self.global_episode,
                episode_reward,
                switch_env,
            )
            # take env step
            time_step = self.train_env.step(action)
            # self.train_env.render()
            episode_reward += time_step.reward
            # self.replay_storage.add(time_step, meta)
            self.train_video_recorder.record(time_step.observation["image"])
            if ca3_out is not None:
                self.agent_ca3_stats.add_activations(
                    ca3_out,
                    time_step.observation["agent_pos"],
                    episode_step,
                    self.global_episode,
                    episode_reward,
                    switch_env,
                )
            if ca1_out is not None:
                self.agent_ca1_stats.add_activations(
                    ca1_out,
                    time_step.observation["agent_pos"],
                    episode_step,
                    self.global_episode,
                    episode_reward,
                    switch_env,
                )

            # replace the obs with image only instead of the full obs with
            # agent_pos so that it can be stored in the replay buffer
            # obs = OrderedDict({"observation": time_step.observation["image"]})
            # obs = time_step.observation["image"]
            # tm_store = time_step
            # tm_store = tm_store._replace(observation=obs)
            # self.replay_storage.add(tm_store, meta)
            episode_step += 1
            self._global_step += 1
            self._switch_step += 1

    def save_snapshot(self):
        snapshot_dir = self.work_dir / Path(self.cfg.snapshot_dir)
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        snapshot = (
            snapshot_dir / f"snapshot_{self.global_episode}_{self.cfg.experiment}.pt"
        )
        keys_to_save = ["agent", "_global_step", "_global_episode"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open("wb") as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot_base_dir = Path(self.cfg.snapshot_load_dir)
        domain = self.cfg.domain
        snapshot_dir = (
            snapshot_base_dir
            / self.cfg.obs_type
            / domain
            / self.cfg.agent.name
            / self.cfg.experiment
            / self.cfg.seed
        )
        # print("Loading snapshot from", snapshot_dir)

        def try_load(seed):
            snapshot = (
                snapshot_dir
                / f"snapshot_{self.cfg.snapshot_ts}_{self.cfg.experiment}.pt"
            )
            print(f"Trying to load snapshot from {snapshot}")
            if not snapshot.exists():
                return None
            with snapshot.open("rb") as f:
                payload = torch.load(f)
            return payload

        # try to load current seed
        payload = try_load(self.cfg.seed)
        if payload is not None:
            return payload
        # otherwise try random seed
        while True:
            seed = np.random.randint(1, 11)
            payload = try_load(seed)
            if payload is not None:
                return payload
        return None


@hydra.main(config_path=".", config_name="config_test")
def main(cfg):
    from test import Workspace as W

    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / "snapshot.pt"
    if snapshot.exists():
        print(f"resuming: {snapshot}")
        workspace.load_snapshot()
    workspace.train()


if __name__ == "__main__":
    main()
