from collections import OrderedDict, deque
from typing import Any, NamedTuple

import dm_env
import numpy as np
from dm_control.suite.wrappers import action_scale
from dm_env import StepType, specs


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)

        def default_on_none(value, default):
            if value is None:
                return default
            return value

        return ExtendedTimeStep(
            observation=time_step.observation,
            step_type=time_step.step_type,
            action=action,
            reward=default_on_none(time_step.reward, 0.0),
            discount=default_on_none(time_step.discount, 1.0),
        )

    def specs(self):
        obs_spec = self._env.observation_spec()
        action_spec = self._env.action_spec()
        next_obs_spec = specs.Array(obs_spec.shape, obs_spec.dtype, "next_observation")
        reward_spec = specs.Array((1,), action_spec.dtype, "reward")
        discount_spec = specs.Array((1,), action_spec.dtype, "discount")
        return (obs_spec, action_spec, reward_spec, discount_spec, next_obs_spec)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(
            wrapped_action_spec.shape,
            dtype,
            wrapped_action_spec.minimum,
            wrapped_action_spec.maximum,
            "action",
        )

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FramePosWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key="image"):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        # assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # pixels_shape = wrapped_obs_spec.shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        # if len(pixels_shape) == 2:  # for atari grayscale add extra dim for channels
        #     pixels_shape = (pixels_shape[0], pixels_shape[1], 1)
        self._obs_spec = specs.BoundedArray(
            shape=np.concatenate(
                [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0
            ),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name="observation",
        )

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        # print(time_step.observation)
        observation = OrderedDict(
            {self._pixels_key: obs, "agent_pos": time_step.observation["agent_pos"]}
        )
        return time_step._replace(observation=observation)

    def _extract_pixels(self, time_step):
        # print(self._pixels_key)
        pixels = time_step.observation[self._pixels_key]
        # pixels = time_step.observation
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key="observation"):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        # assert pixels_key in wrapped_obs_spec

        # pixels_shape = wrapped_obs_spec[pixels_key].shape
        pixels_shape = wrapped_obs_spec.shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        # if len(pixels_shape) == 2:  # for atari grayscale add extra dim for channels
        #     pixels_shape = (pixels_shape[0], pixels_shape[1], 1)
        self._obs_spec = specs.BoundedArray(
            shape=np.concatenate(
                [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0
            ),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name="observation",
        )

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        # pixels = time_step.observation[self._pixels_key]
        pixels = time_step.observation
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


# DM Env Wrapper for atari
class Atari(dm_env.Environment):
    def __init__(self, env):
        self.env = env
        wrapped_obs_spec = self.env.observation_space
        self._action_spec = dm_env.specs.BoundedArray(
            shape=(env.action_space.n,),
            # shape=(1,),
            dtype=np.int64,
            minimum=0,
            maximum=env.action_space.n - 1,
            name="action",
        )
        self._observation_spec = dm_env.specs.BoundedArray(
            shape=wrapped_obs_spec.shape,
            dtype=wrapped_obs_spec.dtype,
            minimum=0,
            maximum=255,
            name="observation",
        )
        self._reward_spec = dm_env.specs.Array(
            shape=(), dtype=np.float32, name="reward"
        )

    def reset(self):
        timestep = self.env.reset()
        # filter out image from tuple timestep = (array([...]), {...})
        obs = timestep[0]
        # obs = OrderedDict()
        # obs["observation"] = obs_img

        return dm_env.TimeStep(
            dm_env.StepType.FIRST, reward=None, discount=1.0, observation=obs
        )

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        # obs = self._transform_image(obs)
        if done:
            return dm_env.TimeStep(
                dm_env.StepType.LAST, reward, discount=0.0, observation=obs
            )
        else:
            return dm_env.TimeStep(
                dm_env.StepType.MID, reward, discount=1.0, observation=obs
            )

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def close(self):
        self.env.close()

    def render(self, mode="human"):
        return self.env.render()

    def seed(self, seed=None):
        return self.env.seed(seed)


# DM Env Wrapper for minigrid
class Minigrid(dm_env.Environment):
    def __init__(self, env):
        self.env = env
        wrapped_obs_spec = self.env.observation_space["image"]
        self._action_spec = dm_env.specs.BoundedArray(
            shape=(env.action_space.n - 6,),
            # shape=(1,),
            dtype=np.int64,
            minimum=0,
            maximum=env.action_space.n - 5,  # use only 3 actions instead of 7
            name="action",
        )
        self._observation_spec = dm_env.specs.BoundedArray(
            shape=wrapped_obs_spec.shape,
            dtype=wrapped_obs_spec.dtype,
            minimum=0,
            maximum=255,
            name="observation",
        )
        self._reward_spec = dm_env.specs.Array(
            shape=(), dtype=np.float32, name="reward"
        )

    # def _transform_image(self, img):
    #     return img.transpose(2, 0, 1).copy()

    def reset(self):
        timestep = self.env.reset()
        # filter out image from tuple timestep = ({'image': [...], 'other': []}, {})
        obs = timestep[0]["image"]
        # obs = self._transform_image(obs)
        # obs = OrderedDict()
        # obs["observation"] = obs_img

        return dm_env.TimeStep(
            dm_env.StepType.FIRST, reward=None, discount=1.0, observation=obs
        )

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        obs = obs["image"]  # take only img and drop info like direction and mission
        # obs = self._transform_image(obs)
        if done or truncated:  # truncated is when max_steps is reached
            return dm_env.TimeStep(
                dm_env.StepType.LAST, reward, discount=0.0, observation=obs
            )
        else:
            return dm_env.TimeStep(
                dm_env.StepType.MID, reward, discount=1.0, observation=obs
            )

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def reward_spec(self):
        return self._reward_spec

    def close(self):
        self.env.close()

    def render(self, mode="human"):
        return self.env.render()

    def seed(self, seed=None):
        return self.env.seed(seed)


def make_env(name, seed, frame_stack, test=False):
    suite, task = name.split("-", 1)
    print(suite)
    if suite == "MiniGrid":  # name_format = "MiniGrid-Empty-8x8-v0"
        from minigrid.wrappers import (RGBImgObsWrapper,
                                       RGBImgPartialObsWrapper, gym)

        # # env = Minigrid(name)
        # # env = gym.make(name, render_mode="rgb_array", max_steps=10)
        # env = gym.make(name, render_mode="rgb_array", agent_view_size=3)
        # env = RGBImgPartialObsWrapper(env)  # , tile_size=3)  # returns (56,56,3) image
        # # env = RGBImgObsWrapper(env)  # returns (40,40,3) image
        # print(env.render_mode)
        # env = Minigrid(env)
        # # env = ExtendedTimeStepWrapper(env)
        # # env = FrameStackWrapper(env, 3)
        # print("env created")
        env_list = []
        agent_view_size = 3
        tile_size = 8

        env_north = gym.make(
            name,
            env_ns="North",
            # reward_pos=(4, 5),
            render_mode="rgb_array",
            agent_view_size=agent_view_size,
        )
        env_north = RGBImgPartialObsWrapper(
            env_north, tile_size=tile_size
        )  # , tile_size=3)  # returns (56,56,3) image
        # env = RGBImgObsWrapper(env)  # returns (40,40,3) image
        env_north = Minigrid(env_north)
        # env_north = ActionDTypeWrapper(env_north, np.float32)
        # env_north = ActionRepeatWrapper(env_north, 2)
        # env_north = action_scale.Wrapper(env_north, minimum=-1.0, maximum=+1.0)
        env_north = FrameStackWrapper(env_north, num_frames=frame_stack)
        env_north = ExtendedTimeStepWrapper(env_north)
        env_list.append(env_north)

        env_south = gym.make(
            name,
            env_ns="South",
            render_mode="rgb_array",
            agent_view_size=agent_view_size,
        )
        env_south = RGBImgPartialObsWrapper(
            env_south, tile_size=tile_size
        )  # , tile_size=3)  # returns (56,56,3) image
        # env = RGBImgObsWrapper(env)  # returns (40,40,3) image
        env_south = Minigrid(env_south)
        # env_south = ActionDTypeWrapper(env_south, np.float32)
        # env_south = ActionRepeatWrapper(env_south, 2)
        # env_south = action_scale.Wrapper(env_south, minimum=-1.0, maximum=+1.0)
        env_south = FrameStackWrapper(env_south, num_frames=frame_stack)
        env_south = ExtendedTimeStepWrapper(env_south)
        env_list.append(env_south)

        return env_list
    elif suite == "atari":
        # import gymnasium
        import gymnasium as gym
        # import gym
        from gymnasium.wrappers import AtariPreprocessing, FrameStack

        print(task)
        # make_kwargs = {"frameskip": 1}
        # env = gym.make("GymV26Environment-v0", env_id=task, make_kwargs=make_kwargs)
        # env = gym.make("PongNoFrameskip-v4")
        env = gym.make("ALE/Pong-v5", frameskip=1, render_mode="rgb_array")
        env = AtariPreprocessing(env, grayscale_newaxis=True)
        env = Atari(env)
        # env = gym.make("PongNoFrameskip-v4")
        # noopreset, max_and_skip, fire_reset, episodic_life, clip_reward,
        # resize_obs, greyscale, frame_stack
    elif suite == "NeuroMaze":
        from neuro_maze import load

        env_list = []

        time_limit = 20
        top_camera = False

        env_north = load(
            domain="base1",
            task_name="reach_target",
            # domain="linear",
            # task_name="linear_track",
            time_limit=time_limit,
            seed=seed,
            top_camera=top_camera,
            image_only_obs=(not test),
            global_observables=test,
            discrete_actions=True,
        )
        # env_north = ActionDTypeWrapper(env_north, np.float32)
        # env_north = ActionRepeatWrapper(env_north, 2)
        # env_north = action_scale.Wrapper(env_north, minimum=-1.0, maximum=+1.0)
        if test:
            env_north = FramePosWrapper(env_north, num_frames=frame_stack)
        else:
            env_north = FrameStackWrapper(env_north, num_frames=frame_stack)
        # env_north = FramePosWrapper(env_north, num_frames=frame_stack)
        env_north = ExtendedTimeStepWrapper(env_north)
        env_list.append(env_north)

        env_south = load(
            domain="base1",
            task_name="reach_target",
            time_limit=time_limit,
            seed=seed,
            top_camera=top_camera,
            image_only_obs=(not test),
            global_observables=test,
            maze_ori="South",
            reward_loc="Right",
            discrete_actions=True,
        )
        # env_south = ActionDTypeWrapper(env_south, np.float32)
        # env_south = ActionRepeatWrapper(env_south, 2)
        # env_south = action_scale.Wrapper(env_south, minimum=-1.0, maximum=+1.0)
        if test:
            env_south = FramePosWrapper(env_south, num_frames=frame_stack)
        else:
            env_south = FrameStackWrapper(env_south, num_frames=frame_stack)
        env_south = ExtendedTimeStepWrapper(env_south)
        env_list.append(env_south)

        return env_list

    else:
        env = None

    # env = ExtendedTimeStepWrapper(env)
    env = FrameStackWrapper(env, num_frames=frame_stack)
    env = ExtendedTimeStepWrapper(env)
    return env
