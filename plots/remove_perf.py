import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import unit_metric_computers as umc

# # test with 1k with 2 cube cues
# removed n random neurons
# dp_hc_state_rem_25 = "/home/rh19400/neuro-rl/exp_local/2024.02.20/165353_hccstate_agent=hccstate,experiment=test_allo_cue,save_stats=true,seed=1"
dp_hc_state_rem_25 = "/home/rh19400/neuro-rl/exp_local/2024.02.20/172934_hccstate_agent=hccstate,experiment=test_allo_cue,save_stats=true,seed=1"

# dp_hc_state_rem_25 = "/home/rh19400/neuro-rl/exp_local/2024.02.20/171030_hccstate_agent=hccstate,experiment=test_allo_cue,save_stats=true,seed=1"
# dp_hc_state_rem_50 = "/home/rh19400/neuro-rl/exp_local/2024.02.20/165156_hccstate_agent=hccstate,experiment=test_allo_cue,save_stats=true,seed=1"
dp_hc_state_rem_50 = "/home/rh19400/neuro-rl/exp_local/2024.02.20/191524_hccstate_agent=hccstate,experiment=test_allo_cue,save_stats=true,seed=1"
dp_hc_state_rem_75 = "/home/rh19400/neuro-rl/exp_local/2024.02.20/161748_hccstate_agent=hccstate,experiment=test_allo_cue,save_stats=true,seed=1"
dp_hc_state_rem_100 = "/home/rh19400/neuro-rl/exp_local/2024.02.20/161051_hccstate_agent=hccstate,experiment=test_allo_cue,save_stats=true,seed=1"
dp_hc_state_rem_all = "/home/rh19400/neuro-rl/exp_local/2024.02.20/160108_hccstate_agent=hccstate,experiment=test_allo_cue,save_stats=true,seed=1"
dp_hc_state_rem_200 = "/home/rh19400/neuro-rl/exp_local/2024.02.20/173921_hccstate_agent=hccstate,experiment=test_allo_cue,save_stats=true,seed=1"
dp_hc_state_rem_300 = "/home/rh19400/neuro-rl/exp_local/2024.02.20/174031_hccstate_agent=hccstate,experiment=test_allo_cue,save_stats=true,seed=1"
# dp_hc_state_rem_331 = ""

# # remove 100 random non-place neurons
dp_hc_state_rem_25_pc = "/home/rh19400/neuro-rl/exp_local/2024.02.20/152802_hccstate_agent=hccstate,experiment=test_allo_cue,save_stats=true,seed=1"
dp_hc_state_rem_50_pc = "/home/rh19400/neuro-rl/exp_local/2024.02.20/152147_hccstate_agent=hccstate,experiment=test_allo_cue,save_stats=true,seed=1"
dp_hc_state_rem_75_pc = "/home/rh19400/neuro-rl/exp_local/2024.02.20/151529_hccstate_agent=hccstate,experiment=test_allo_cue,save_stats=true,seed=1"
dp_hc_state_rem_100_pc = "/home/rh19400/neuro-rl/exp_local/2024.02.20/150827_hccstate_agent=hccstate,experiment=test_allo_cue,save_stats=true,seed=1"
# dp_hc_state_rem_all_pc = "/home/rh19400/neuro-rl/exp_local/2024.02.20/154843_hccstate_agent=hccstate,experiment=test_allo_cue,save_stats=true,seed=1"
dp_hc_state_rem_all_pc = "/home/rh19400/neuro-rl/exp_local/2024.02.20/193832_hccstate_agent=hccstate,experiment=test_allo_cue,save_stats=true,seed=1"
dp_hc_state_rem_border = "/home/rh19400/neuro-rl/exp_local/2024.02.21/094931_hccstate_agent=hccstate,experiment=test_allo_cue,save_stats=true,seed=1"
dp_hc_state_rem_place = "/home/rh19400/neuro-rl/exp_local/2024.02.21/095127_hccstate_agent=hccstate,experiment=test_allo_cue,save_stats=true,seed=1"

dp_hc_state_rem_all_units = "/home/rh19400/neuro-rl/exp_local/2024.02.21/100414_hccstate_agent=hccstate,experiment=test_allo_cue,save_stats=true,seed=1"


def get_df(data_path):
    df_ca1 = pd.read_csv(f"{data_path}/ca1.csv")
    df_ca3 = pd.read_csv(f"{data_path}/ca3.csv")
    return df_ca1, df_ca3


def get_df_pos(data_path):
    df = pd.read_csv(f"{data_path}/locations/agent_pos.csv")
    return df


def get_reward_and_steps(model, episode_num=50):

    df_filter = model.loc[model["episode"] < episode_num]
    df_group = df_filter.groupby("episode")
    steps_mean = df_group.describe()["step"]["max"].mean()  # mean=124.41 sem=11.298
    steps_sem = df_group.describe()["step"]["max"].sem()
    rew = df_group.describe()["reward"]["max"].mean()  # 97%
    rew_sem = df_group.describe()["reward"]["max"].sem()  # 97%
    return steps_mean, steps_sem, rew, rew_sem


def get_neurons_hm(df):

    df = df.loc[df["episode"] <= 50]
    # df = df.loc[df["episode"] == 6]
    # # filter between 100 and 120
    # df = df.loc[df["episode"] <= 130]
    # df = df.loc[df["episode"] >= 110]
    df["position_x"] = df["position_x"].round()
    df["position_y"] = df["position_y"].round()

    # df = df.groupby(['position_x', 'position_y']).mean().reset_index()
    # df = df.pivot(index='position_x', columns='position_y', values='firing_rate')

    df_pos = pd.DataFrame(columns=df.columns)
    df_pos = df_pos.drop(columns=["step", "episode", "reward", "env_id"])
    i = 0
    neurons = np.zeros((512, 9, 9))
    for x in range(1, 10):
        df_posx = df.loc[df["position_x"] == x]
        for y in range(1, 10):
            df_posxy = df_posx.loc[df_posx["position_y"] == y]
            df_posxy = df_posxy.drop(columns=["step", "episode", "reward", "env_id"])
            df_pos.loc[i] = df_posxy.mean()
            neurons[:, x - 1, y - 1] = df_posxy.drop(
                columns=["position_x", "position_y"]
            ).mean()
            i += 1

    neurons[np.isnan(neurons)] = 0.0  # + 1e-10

    norm_neurons = neurons / np.linalg.norm(neurons)
    norm_neurons[norm_neurons < 1e-3] = 0.0  # careful this is not absolute zero
    # norm_neurons = neurons
    # m_vals = norm_neurons.mean(axis=(1,2))
    # # Get the indices that would sort the mean values in descending order
    # sorted_indices = np.argsort(m_vals)#[::-1]
    # # sorted_indices = np.argsort(m_vals)[::-1]
    # # Use take_along_axis to reorder the original array based on the sorted indices
    # sorted_neurons = np.take_along_axis(norm_neurons, sorted_indices[:, None, None], axis=0)
    sorted_neurons = norm_neurons
    return sorted_neurons


def plot_neurons_hm(data, name="ca3", axs=None, idxs=None):
    # Set the number of rows and columns for the subplot grid
    num_rows = 1
    num_cols = 6
    if idxs is None:
        idxs = np.arange(num_rows * num_cols)
    # idxs = num_rows * num_cols

    # # Create a figure and a grid of subplots
    # fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 12))
    # # fig, axs = plt.subplots(num_rows, num_cols)

    # # Flatten the 2D array of Axes objects into a 1D array for easier indexing
    # axs = axs.flatten()

    cmap = "bwr"
    interp = "none"
    # interp = "nearest"
    # interp = "gaussian"
    vmin = -0.008
    vmax = 0.008
    origin = "lower"
    offset = 0  # 9

    # Loop through each subplot and create a heatmap
    # for i in range(num_rows * num_cols):
    for i, neuron_i in enumerate(idxs[: num_rows * num_cols]):
        # im = axs[i].imshow(data[i], cmap="viridis", extent=[0, 100, 0, 100], origin="lower")
        im = axs[i].imshow(
            data[neuron_i + offset],
            cmap=cmap,
            origin=origin,
            interpolation=interp,
            vmin=vmin,
            vmax=vmax,
        )
        axs[i].set_title(f"Neuron {neuron_i}")
        # axs[i].set_axis_off()

    # Add a colorbar for reference
    cbar = fig.colorbar(im, ax=axs[0], orientation="vertical", fraction=0.05, pad=0.05)
    # plt.axis("off")

    # Adjust layout to prevent clipping of titles
    plt.tight_layout()

    # Show the plot
    # plt.show()
    # fig.savefig(f"neurons_{name}_hm.pdf", bbox_inches="tight")


# Set the number of rows and columns for the subplot grid
num_rows = 1
num_cols = 2

# Create a figure and a grid of subplots
# fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 6 * 2))
fig = plt.figure(figsize=(5 * 5, 5 * num_rows))
gs = plt.GridSpec(num_rows, num_cols, wspace=0.5, hspace=0.5)
axs = []
for i in range(num_rows):
    # axs.append(fig.add_subplot(gs[i, 0]))
    for j in range(num_cols):
        axs.append(fig.add_subplot(gs[i, j]))

# Flatten the 2D array of Axes objects into a 1D array for easier indexing
# axs = axs.flatten()
# print(axs)
# df_ca1_start, df_ca3_start = get_df(eval(f"dp_{n_start}"))

# names_og = ["25_pc", "50_pc", "75_pc", "100_pc"]
names_og = [
    # "25_pc",
    # "50_pc",
    # "75_pc",
    # "100_pc",
    # "25",
    # "50",
    # "75",
    # "100",
    # "200",
    # "300",
    "all_units",
    "all_pc",
    "all",
    "border",
    "place",
]

name_plot = ["none", "spatial", "non-spatial", "border-only", "place-only"]

for name, name_model in zip(names_og, name_plot):
    df_start_pos = get_df_pos(eval(f"dp_hc_state_rem_{name}"))

    steps_m, steps_sem, rew_m, rew_sem = get_reward_and_steps(
        df_start_pos, episode_num=50
    )
    # print(f"steps: {steps_m}, rewards: {rewards_m}")
    width = 0.35

    axs[0].bar(name_model, rew_m * 100, yerr=rew_sem * 100, width=width)
    axs[1].bar(name_model, steps_m, yerr=steps_sem, width=width)

# Remove top and right spines
axs[0].spines["top"].set_visible(False)
axs[0].spines["right"].set_visible(False)
axs[1].spines["top"].set_visible(False)
axs[1].spines["right"].set_visible(False)
axs[0].set_ylabel("Performance (%)")
axs[1].set_ylabel("Steps")
plt.tight_layout()
plt.show()
fig.savefig(f"test_cells_removed.pdf", bbox_inches="tight")
