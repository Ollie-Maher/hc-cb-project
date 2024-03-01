from dataclasses import fields

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catppuccin import Flavour
from numpy.random import rand, randint, randn

#### Bio model ####
# plastic hc-cb ##
data_path_bio_hc_cb_plastic = "/home/rh19400/neuro-rl/exp_local/2023.12.02/200235_hccbio_agent=hccbio,experiment=wm_plastic,save_stats=true,seed=1,use_wandb=false/locations"
# plastic hc-cb-abl ##
data_path_bio_hc_cb_plastic_abl = "/home/rh19400/neuro-rl/exp_local/2023.12.02/201451_hccbio_agent=hccbio,experiment=wm_plastic,save_stats=true,seed=1,use_wandb=false/locations"
# fixed hc-cb ##
data_path_bio_hc_cb = "/home/rh19400/neuro-rl/exp_local/2023.12.11/110514_hccbio_agent=hccbio,experiment=wm,save_stats=true,seed=1/locations"
# fixed hc-cb-abl ##
data_path_bio_hc_cb_abl = "/home/rh19400/neuro-rl/exp_local/2023.12.11/111353_hccbio_agent=hccbio,experiment=wm,save_stats=true,seed=1/locations"

#### Fixed model ####
# hc fixed 10k trained steps
data_path_hc = "/home/rh19400/neuro-rl/exp_local/2023.11.28/091550_drqn_agent=drqn,experiment=wm_fixed_s,save_stats=true,seed=5,use_wandb=false/locations"
# hc-cb fixed 10k trained steps
data_path_hc_cb = "/home/rh19400/neuro-rl/exp_local/2023.11.28/091619_hcc_agent=hcc,experiment=wm_s,save_stats=true,seed=1,use_wandb=false/locations"
# hc_cb ablation, 10k trained steps
data_path_hc_cb_abl = "/home/rh19400/neuro-rl/exp_local/2023.11.28/104213_hcc_agent=hcc,experiment=wm_s,save_stats=true,seed=1,use_wandb=false/locations"

### Plastic model ###
# drqn plastic
data_path_hc_plastic = "/home/rh19400/neuro-rl/exp_local/2023.12.11/115628_drqn_agent=drqn,experiment=wm_plastic,save_stats=true,seed=10/locations"
# hc-cb plastic 10k trained steps
data_path_hc_cb_plastic = "/home/rh19400/neuro-rl/exp_local/2023.11.28/182923_hcc_agent=hcc,experiment=wm_plastic,save_stats=true,seed=15,use_wandb=false/locations"
# hc-cb plastic ablation, 10k trained steps
data_path_hc_cb_plastic_abl = "/home/rh19400/neuro-rl/exp_local/2023.11.28/183532_hcc_agent=hcc,experiment=wm_plastic,save_stats=true,seed=15,use_wandb=false/locations"

# # # drqn plastic
# data_path_drqn_2 = "/home/rh19400/neuro-rl/exp_local/2023.11.27/163557_drqn_agent=drqn,experiment=wm_plastic_s2,save_stats=true,seed=1,use_wandb=false/locations"
# data_path_drqn_1 = "/home/rh19400/neuro-rl/exp_local/2023.11.27/163047_drqn_agent=drqn,experiment=wm_plastic_s,save_stats=true,seed=1,use_wandb=false/locations"
# # # hcc fixed 2k trained steps
# # data_path_hcc = "/home/rh19400/neuro-rl/exp_local/2023.11.27/174214_hcc_agent=hcc,experiment=wm_s,save_stats=true,seed=1,use_wandb=false/locations"

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
# model_color = {}
width = 0.3
# flavour = Flavour.frappe()
# for field, name in zip(fields(flavour), names):
#     colour = getattr(flavour, field.name)
#     print(f"{name} - {field.name}: #{colour.hex}")
#     model_color[name] = f"#{colour.hex}"

# color palette for models
model_color = {
    "hc": "#2D435B",
    "hc-cb": "#FCE38A",
    "plastic-hc": "#82AAC4",
    "plastic-hc-cb": "#D2D3A0",
    "bio-plastic-hc-cb": "#4E637C",
    "bio-hc-cb": "#5BB111",
    "hc-cb abl": "#1F962F",
    "plastic-hc-cb abl": "#176DE8",
    "bio-plastic-hc-cb abl": "#FFF4B6",
    "bio-hc-cb abl": "#D4D3CF",
}

# # custom theme
# model_color = { "hc": "#B2ABFF", 
#                "hc-cb": "#D5CFB2", 
#                "plastic-hc": "#FFF2AB", 
#                "plastic-hc-cb": "#B4B1CC", 
#                "bio-plastic-hc-cb": "#AAA587", 
#                "bio-hc-cb": "#141133", 
#                "hc-cb abl": "#554E27", 
#                "plastic-hc-cb abl": "#3A3666", 
#                "bio-plastic-hc-cb abl": "#807850", 
#                "bio-hc-cb abl": "#737099" }

# print(model_color)


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
    tot_quadrants = []
    distance_to_reward = []
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
        tot_quadrants.append(len(visited_quadrants))
        # tot_quadrants_err = np.sqrt(len(visited_quadrants))
        distance_to_reward.append(np.sum(
            np.sqrt((x_trajectory - reward[0]) ** 2 + (y_trajectory - reward[1]) ** 2)
        ) / len(x_trajectory))

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
    
    tot_quadrants = np.array(tot_quadrants)
    distance_to_reward = np.array(distance_to_reward)
    mean_tot_quadrants = np.mean(tot_quadrants)
    sem_tot_quadrants = np.std(tot_quadrants) / np.sqrt(trials)
    mean_distance_to_reward = np.mean(distance_to_reward)
    sem_distance_to_reward = np.std(distance_to_reward) / np.sqrt(trials)

    return mean_tot_quadrants, sem_tot_quadrants, mean_distance_to_reward, sem_distance_to_reward 
    # return tot_quadrants / trials, distance_to_reward / trials


def get_ax_trajectory(
    ax1, model1, trials=100, title="Trajectory", s_start=10**2, s_reward=11**4
):
    reward = [5.5, 6.5]
    colors = {
        (1.5, 7.5): "red",
        (7.5, 3.5): "green",
        (3.5, 4.5): "blue",
        (2.5, 1.5): "orange",
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
    gs = plt.GridSpec(3, 6, wspace=0.9, hspace=0.65)
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
        "hc",
        "hc-cb",
        # "plastic-hc",
        # "plastic-hc-cb",
        # "bio-plastic-hc-cb",
        "bio-hc-cb",
        "hc-cb abl",
        # "plastic-hc-cb abl",
        # "bio-plastic-hc-cb abl",
        "bio-hc-cb abl",
    ]

    for i in range(len(models)):
        ax = get_ax_trajectory(
            axes[i],
            models[i],
            trials=100,
            title=names[i],
            s_start=10**1,
            s_reward=11**3,
        )

        steps_m, steps_sem, rew_m, rew_sem = get_reward_and_steps(models[i], 100)
        ax11.bar(
            names[i], rew_m * 100, yerr=rew_sem * 100, width=width, color=model_color[names[i]]
        )
        ax12.bar(
            names[i], steps_m, yerr=steps_sem, width=width, color=model_color[names[i]]
        )

        s_score, sem_score, distance_to_rew, sem_distance_to_rew = search_score(models[i])
        ax7.bar(names[i], s_score, yerr=sem_score, width=width, color=model_color[names[i]])
        ax13.bar(names[i], distance_to_rew, yerr=sem_distance_to_rew, width=width, color=model_color[names[i]])

    # remove top, left and right spines
    ax11.spines["top"].set_visible(False)
    ax11.spines["right"].set_visible(False)
    ax12.spines["top"].set_visible(False)
    ax12.spines["right"].set_visible(False)
    ax11.set(title="Average reward", ylabel="Performance (%)")
    ax11.tick_params(axis="x", rotation=45)
    ax12.set(title="Average steps", ylabel="Steps")
    ax12.tick_params(axis="x", rotation=45)

    ax7.spines["top"].set_visible(False)
    ax7.spines["right"].set_visible(False)
    ax7.set(title="Search score")
    ax7.set(ylabel="Score")
    ax7.tick_params(axis="x", rotation=45)
    ax13.spines["top"].set_visible(False)
    ax13.spines["right"].set_visible(False)
    ax13.set(title="Distance to reward", ylabel="Squared distance")
    ax13.tick_params(axis="x", rotation=45)

    # Add a bold 'a' or 'b' in the top-left corner
    ax1.text(-0.5, 1.1, "A", transform=ax1.transAxes, fontsize=18, weight="bold")
    ax7.text(-0.5, -0.5, "B", transform=ax1.transAxes, fontsize=18, weight="bold")
    ax4.text(1.5, -0.5, "C", transform=ax1.transAxes, fontsize=18, weight="bold")
    ax11.text(-0.5, -2.2, "D", transform=ax1.transAxes, fontsize=18, weight="bold")
    ax12.text(1.3, -2.2, "E", transform=ax1.transAxes, fontsize=18, weight="bold")
    ax13.text(3.0, -2.2, "F", transform=ax1.transAxes, fontsize=18, weight="bold")

    plt.tight_layout()
    plt.show()
    # fig.savefig("figure_p4.pdf", bbox_inches="tight")


all_models_list = []
all_data_paths = [
    data_path_hc,
    data_path_hc_cb,
    # data_path_hc_plastic,
    # data_path_hc_cb_plastic,
    # data_path_bio_hc_cb_plastic,
    data_path_bio_hc_cb,
    data_path_hc_cb_abl,
    # data_path_hc_cb_plastic_abl,
    # data_path_bio_hc_cb_plastic_abl,
    data_path_bio_hc_cb_abl,
]
for data_path in all_data_paths:
    all_models_list.append(get_df(data_path))
plot_all(models=all_models_list)
