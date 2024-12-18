from dataclasses import fields

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catppuccin import Flavour
from numpy.random import rand, randint, randn

#### Bio model ####
# # plastic hc-cb ##
# data_path_bio_hc_cb_plastic = "/home/rh19400/neuro-rl/exp_local/2023.12.02/200235_hccbio_agent=hccbio,experiment=wm_plastic,save_stats=true,seed=1,use_wandb=false/locations"
# # plastic hc-cb-abl ##
# data_path_bio_hc_cb_plastic_abl = "/home/rh19400/neuro-rl/exp_local/2023.12.02/201451_hccbio_agent=hccbio,experiment=wm_plastic,save_stats=true,seed=1,use_wandb=false/locations"
# fixed hc-cb ##
# data_path_bio_hc_cb = "/home/rh19400/neuro-rl/exp_local/2024.01.12/174653_hccbio_agent=hccbio,experiment=wm,save_stats=true,seed=1/locations"
# # fixed hc-cb-abl ##
# data_path_bio_hc_cb_abl = "/home/rh19400/neuro-rl/exp_local/2023.12.11/111353_hccbio_agent=hccbio,experiment=wm,save_stats=true,seed=1/locations"

#### Fixed model ####
# hc fixed 10k trained steps
# data_path_hc = "/home/rh19400/neuro-rl/exp_local/2024.01.12/173014_drqn_agent=drqn,experiment=wm_fixed_s,save_stats=true,seed=5/locations"
# hc-cb fixed 10k trained steps
# data_path_hc_cb = "/home/rh19400/neuro-rl/exp_local/2024.01.12/171235_hcc_agent=hcc,experiment=wm_s,save_stats=true,seed=1/locations"
data_path_hc_cb = "/home/rh19400/neuro-rl/exp_local/2024.01.12/182035_hcc_agent=hcc,experiment=wm_s,save_stats=true,seed=1/locations"
# hc_cb ablation, 10k trained steps
# data_path_hc_cb_abl = ""


names = [
    "hc",
    "hc-cb",
    "plastic-hc",
    "plastic-hc-cb",
    "bio-plastic-hc-cb",
    "bio-hc-cb",
    "hc-cb abl",
    "plastic-hc-cb abl",
    "bio-plastic-hc-cb abl",
    "bio-hc-cb abl",
]
model_color = {}
width = 0.3
flavour = Flavour.frappe()
for field, name in zip(fields(flavour), names):
    colour = getattr(flavour, field.name)
    print(f"{name} - {field.name}: #{colour.hex}")
    model_color[name] = f"#{colour.hex}"

print(model_color)


def get_df(data_path):
    df = pd.read_csv(f"{data_path}/agent_pos.csv")
    return df


# columns = [f'neuron_{i}' for i in range(512)]
# df.head()


def get_reward_and_steps(model, episode_num=100):

    df_filter = model.loc[model["episode"] < episode_num]
    df_group = df_filter.groupby("episode")
    steps_mean = df_group.describe()["step"]["max"].mean()  # mean=124.41 sem=11.298
    steps_sem = df_group.describe()["step"]["max"].sem()
    rew = df_group.describe()["reward"]["max"].mean()  # 97%
    rew_sem = df_group.describe()["reward"]["max"].sem()  # 97%
    return steps_mean, steps_sem, rew, rew_sem


def search_score(model, trials=50, bins=10):
    tot_quadrants = 0
    distance_to_reward = 0
    reward = [5.5, 6.5]
    # print(f"model: {model}")

    for i in range(0, trials):

        model_f = model.loc[model["episode"] == i].reset_index(drop=True)
        x_trajectory = model_f["position_x"][1:]
        y_trajectory = model_f["position_y"][1:]

        # Define quadrant boundaries
        x_bins = np.linspace(0, 10, bins)  # Adjust the number of bins as needed
        y_bins = np.linspace(0, 10, bins)

        # Use np.digitize to find the bin indices for each coordinate
        quadrant_x = np.digitize(x_trajectory, x_bins)
        quadrant_y = np.digitize(y_trajectory, y_bins)

        # Combine x and y quadrant indices to get a unique quadrant identifier for each point
        quadrant = (quadrant_y - 1) * len(x_bins) + quadrant_x

        # Use a set to get unique quadrants
        visited_quadrants = set(quadrant)
        # get running average of unique quadrants visited
        tot_quadrants += len(visited_quadrants)
        distance_to_reward += np.sum(
            np.sqrt((x_trajectory - reward[0]) ** 2 + (y_trajectory - reward[1]) ** 2)
        ) / len(x_trajectory)

        # return tot_quadrants/trials

        # # Print the number of unique quadrants visited
        # print(f'The animal visited {len(visited_quadrants)} unique quadrants.')
        # print(f'len quadrant: {len(quadrant)}')
        # print(f'len visited_quadrants: {len(visited_quadrants)}')
        # print(f'quadrant_x: {quadrant}')
        # print(f'visited_quadrants: {visited_quadrants}')
        # print(f"mean distance to reward: {distance_to_reward/len(x_trajectory)}")

    # print(f'The animal visited {tot_quadrants/trials} unique quadrants on average.')
    # print(f'The animal visited {tot_quadrants} unique quadrants in total.')
    # print(f"mean distance to reward: {distance_to_reward/trials}")
    # # Plot the trajectory for visualization
    # plt.plot(x_trajectory, y_trajectory, label='Animal Trajectory')
    # plt.xlim(0, 10)
    # plt.ylim(0, 10)
    # plt.xlabel('X-coordinate')
    # plt.ylabel('Y-coordinate')

    # # Display the plot with quadrant boundaries
    # plt.grid(True, linestyle='--', alpha=0.7)
    # # plt.grid(True, linestyle='--', alpha=0.7, which='both', axis='both', color='gray', linewidth=0.5)
    # plt.xticks([0, 2.5, 5, 7.5, 10])
    # plt.yticks([0, 2.5, 5, 7.5, 10])

    # # plt.legend()
    # plt.show()

    return tot_quadrants / trials, distance_to_reward / trials


def get_ax_trajectory(
    ax1, model1, trials=100, title="Trajectory", s_start=10**2, s_reward=11**4
):
    reward = [5.5, 6.5]
    colors = {
        (1.5, 7.5): "red",
        (1.5, 7.4): "red",
        (1.6, 7.4): "red",
        (1.5, 11.5): "red",
        (7.5, 3.5): "green",
        (11.5, 5.5): "green",
        (3.5, 4.5): "blue",
        (4.5, 6.5): "blue",
        (2.5, 1.5): "orange",
        (2.6, 1.6): "orange",
    }
    for i in range(0, trials):
        model1_1 = model1.loc[model1["episode"] == i].reset_index(drop=True)
        ax1.scatter(
            model1_1["position_x"][1],
            model1_1["position_y"][1],
            marker="x",
            linewidth=8,
            s=s_start,
            color=colors[
                (
                    round(model1_1["position_x"][1], 1),
                    round(model1_1["position_y"][1], 1),
                )
            ],
        )
        ax1.plot(
            model1_1["position_x"][1:],
            model1_1["position_y"][1:],
            # linestyle="--",
            linewidth=1,
            alpha=0.9,
            color=colors[
                (
                    round(model1_1["position_x"][1], 1),
                    round(model1_1["position_y"][1], 1),
                )
            ],
        )
    ax1.scatter(reward[0], reward[1], marker="o", linewidth=1, s=s_reward, alpha=0.5)
    ax1.set_ylim([0.5, 9.5])
    ax1.set_xlim([0.5, 9.5])
    # ax1.hlines(5, 0, 9, linewidth=1, alpha=0.5)
    # ax1.vlines(5, 0, 9, linewidth=1, alpha=0.5)
    ax1.set(title=title, xlabel="x", ylabel="y")
    ax1.set_aspect("equal", adjustable="box")

    return ax1


def plot_all(models):

    fig = plt.figure(figsize=(10, 9))
    gs = plt.GridSpec(3, 6, wspace=0.8, hspace=0.65)
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    # ax3 = fig.add_subplot(gs[0, 2])
    # ax4 = fig.add_subplot(gs[0, 3])
    # ax5 = fig.add_subplot(gs[0, 4])
    ax3 = fig.add_subplot(gs[0, 4:6])
    ax4 = fig.add_subplot(gs[1, 2:4])
    ax7 = fig.add_subplot(gs[1, 0:2])
    # ax9 = fig.add_subplot(gs[1, 4])
    ax5 = fig.add_subplot(gs[1, 4:6])
    axes = [ax1, ax2, ax3, ax4, ax5]  # , ax6, ax7, ax8, ax9, ax10]
    ax11 = fig.add_subplot(gs[2, 0:2])
    ax12 = fig.add_subplot(gs[2, 2:4])
    ax13 = fig.add_subplot(gs[2, 4:6])
    names = [
        # "hc",
        "hc-cb",
        # "plastic-hc",
        # "plastic-hc-cb",
        # "bio-plastic-hc-cb",
        # "bio-hc-cb",
        # "hc-cb abl",
        # "plastic-hc-cb abl",
        # "bio-plastic-hc-cb abl",
        # "bio-hc-cb abl",
    ]

    for i in range(len(models)):
        ax = get_ax_trajectory(
            axes[i],
            models[i],
            trials=100,
            title=names[i],
            s_start=10**1,
            s_reward=10**3,
        )

        steps_m, steps_sem, rew_m, rew_sem = get_reward_and_steps(models[i], 50)
        ax11.bar(
            names[i], rew_m, yerr=rew_sem, width=width, color=model_color[names[i]]
        )
        ax12.bar(
            names[i], steps_m, yerr=steps_sem, width=width, color=model_color[names[i]]
        )

        s_score, distance_to_rew = search_score(models[i], 100)
        ax7.bar(names[i], s_score, width=width, color=model_color[names[i]])
        ax13.bar(names[i], distance_to_rew, width=width, color=model_color[names[i]])
    # remove top, left and right spines
    ax11.spines["top"].set_visible(False)
    ax11.spines["right"].set_visible(False)
    ax12.spines["top"].set_visible(False)
    ax12.spines["right"].set_visible(False)
    ax11.set(title="Average reward")
    ax11.tick_params(axis="x", rotation=45)
    ax12.set(title="Average steps")
    ax12.tick_params(axis="x", rotation=45)

    ax7.spines["top"].set_visible(False)
    ax7.spines["right"].set_visible(False)
    ax7.set(title="Search score")
    ax7.tick_params(axis="x", rotation=45)
    ax13.spines["top"].set_visible(False)
    ax13.spines["right"].set_visible(False)
    ax13.set(title="Distance to reward")
    ax13.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()
    # fig.savefig("figure_p4.pdf", bbox_inches="tight")


all_models_list = []
all_data_paths = [
    # data_path_hc,
    data_path_hc_cb,
    # data_path_hc_plastic,
    # data_path_hc_cb_plastic,
    # data_path_bio_hc_cb_plastic,
    # data_path_bio_hc_cb,
    # data_path_hc_cb_abl,
    # data_path_hc_cb_plastic_abl,
    # data_path_bio_hc_cb_plastic_abl,
    # data_path_bio_hc_cb_abl,
]
for data_path in all_data_paths:
    all_models_list.append(get_df(data_path))
plot_all(models=all_models_list)
