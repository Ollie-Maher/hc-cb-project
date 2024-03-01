from dataclasses import fields

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from catppuccin import Flavour

# names = [
#     "hc",
#     "hc-cb",
#     "plastic-hc",
#     "plastic-hc-cb",
#     "bio-plastic-hc-cb",
#     "bio-hc-cb",
#     "hc-cb abl",
#     "plastic-hc-cb abl",
#     "bio-plastic-hc-cb abl",
#     "bio-hc-cb abl",
# ]
# model_color = {}
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

# animal_data, first 10 is control, next 10 is without cb, next 10 control error, next 10 without cb error
animal_data_x = [
    0.4588607594936709,
    1.4873417721518989,
    2.4841772151898738,
    3.496835443037975,
    4.462025316455697,
    5.474683544303798,
    6.455696202531646,
    7.515822784810127,
    8.496835443037975,
    9.525316455696203,
    0.5221518987341772,
    1.5348101265822787,
    2.5,
    3.5601265822784813,
    4.493670886075949,
    5.537974683544304,
    6.518987341772152,
    7.531645569620253,
    8.528481012658228,
    9.556962025316457,
    0.44303797468354433,
    1.5348101265822787,
    2.5,
    3.528481012658228,
    4.509493670886076,
    5.522151898734178,
    6.518987341772152,
    7.54746835443038,
    8.496835443037975,
    9.556962025316457,
    0.5063291139240507,
    1.518987341772152,
    2.5,
    3.528481012658228,
    4.509493670886076,
    5.522151898734178,
    6.518987341772152,
    7.531645569620253,
    8.512658227848101,
    9.556962025316457,
]
animal_data_y = [
    77.57685352622062,
    54.97287522603978,
    39.78300180831826,
    34.719710669077756,
    37.43218806509946,
    29.29475587703436,
    25.31645569620253,
    20.07233273056058,
    19.168173598553345,
    19.89150090415913,
    82.45931283905968,
    67.63110307414105,
    68.35443037974683,
    46.292947558770344,
    50.09041591320072,
    35.98553345388788,
    41.048824593128394,
    33.99638336347197,
    28.933092224231466,
    29.475587703435806,
    80.10849909584087,
    59.6745027124774,
    45.02712477396022,
    38.33634719710669,
    41.22965641952984,
    32.91139240506329,
    28.571428571428573,
    22.965641952983724,
    20.795660036166364,
    22.24231464737794,
    85.71428571428571,
    71.97106690777576,
    72.33273056057867,
    51.717902350813745,
    54.249547920434,
    39.059674502712475,
    45.56962025316456,
    37.25135623869801,
    33.273056057866185,
    34.90054249547921,
]


def update_cols_name(path, model):
    df = pd.read_csv(path)
    df.columns = [model + "-" + col for col in df.columns]
    columns = [
        "Step",
        model,
        model + "_min",
        model + "_max",
    ]
    df.columns = columns
    return df


def plot_from_df(path1, path2, path3, path4, ax1, ax2, ax3, save_filename=None):

    data_path_p = path1
    data_path_f = path2
    data_path_bio_p = path3
    data_path_bio_f = path4

    df_plastic = pd.read_csv(data_path_p)
    columns = [
        "Step",
        "plastic-hc",
        "plastic-hc_min",
        "plastic-hc_max",
        "plastic-hc-cb",
        "plastic-hc-cb_min",
        "plastic-hc-cb_max",
    ]
    df_plastic.columns = columns

    # for fixed ordered is swapped
    df_fixed = pd.read_csv(data_path_f)
    columns = [
        "Step",
        "hc-cb",
        "hc-cb_min",
        "hc-cb_max",
        "hc",
        "hc_min",
        "hc_max",
    ]
    df_fixed.columns = columns

    df_bio_plastic = update_cols_name(data_path_bio_p, "bio-plastic-hc-cb")
    df_bio_fixed = update_cols_name(data_path_bio_f, "bio-hc-cb")

    df = pd.concat(
        [df_plastic, df_fixed, df_bio_plastic, df_bio_fixed],
        axis=1,
        ignore_index=False,
        sort=False,
    )

    ind = 2000
    df_ewma = df.ewm(alpha=0.02).mean()[
        :ind
    ]  # .plot(x='Step', y=['hcc','drqn'])[:4000]
    df_ewma_var = df.ewm(alpha=0.02).var()[:ind]
    X = np.linspace(0, ind, num=ind)

    df_ewma = df_ewma * 100
    df_ewma_var = df_ewma_var * 100

    # fig = plt.figure(figsize=(16, 8))
    # # fig = plt.figure(figsize=(16, 4))
    # gs = plt.GridSpec(2, 3, wspace=0.2, hspace=0.1)
    # ax1 = fig.add_subplot(gs[0, 0])
    # ax2 = fig.add_subplot(gs[0, 1])
    # ax3 = fig.add_subplot(gs[0, 2])
    # ax4 = fig.add_subplot(gs[1, 0])

    start_pos = [[1.5, 7.5], [7.5, 3.5], [3.8, 4.5], [2.5, 1.5]]
    colors = {
        tuple(start_pos[0]): "red",
        tuple(start_pos[1]): "green",
        tuple(start_pos[2]): "blue",
        tuple(start_pos[3]): "orange",
    }
    img = plt.imread("td_wm.png")
    ax1.imshow(img, extent=[0, 10, 0, 10])
    # Add an 'X' by plotting two lines
    ax1.scatter(6, 7, s=50 * 2**5, c="deepskyblue", marker="o", alpha=0.9)
    for i in range(4):
        ax1.scatter(
            start_pos[i][0],
            start_pos[i][1],
            marker="X",
            # linewidth=8,
            s=10**2,
            # color=colors[(start_pos[i][0], start_pos[i][1])],
            facecolors=colors[(start_pos[i][0], start_pos[i][1])],
            alpha=0.8,
            # facecolors="none",
            edgecolors="white",
        )

    # ax1.spines["top"].set_visible(False)
    # ax1.spines["right"].set_visible(False)
    ax1.set(title="Gym environment", xlabel="X", ylabel="Y")

    ax2.plot(X, df_ewma["hc"], color=model_color["hc"], alpha=1, label="hc")
    ax2.fill_between(
        X,
        df_ewma["hc"] - df_ewma_var["hc"],
        df_ewma["hc"] + df_ewma_var["hc"],
        color=model_color["hc"],
        alpha=0.5,
    )

    ax2.plot(X, df_ewma["hc-cb"], color=model_color["hc-cb"], alpha=1, label="hc-cb")
    ax2.fill_between(
        X,
        df_ewma["hc-cb"] - df_ewma_var["hc-cb"],
        df_ewma["hc-cb"] + df_ewma_var["hc-cb"],
        color=model_color["hc-cb"],
        alpha=0.5,
    )

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set(title="Episode reward", xlabel="Trials", ylabel="Performance (%)")
    ax2.legend()

    width = 0.3

    ax3.bar(
        ["hc-cb"],
        df_ewma["hc-cb"].mean(),
        yerr=df_ewma_var["hc-cb"].mean(),
        width=width,
        color=model_color["hc-cb"],
    )
    ax3.bar(
        ["hc"],
        df_ewma["hc"].mean(),
        yerr=df_ewma_var["hc"].mean(),
        width=width,
        color=model_color["hc"],
    )

    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.set(title="Train", ylabel="Performance (%)")
    ax3.set_ylim([0, 100])
    ax3.tick_params(axis="x", labelrotation=30)
    ax3.hlines(y=50, xmin=-0.2, xmax=1.2, linewidth=2, color="black", linestyle="--")

    return ax1, ax2, ax3

    # plt.tight_layout()
    # plt.show()
    # fig.savefig(f"{save_filename}.pdf", bbox_inches="tight")


def plot_steps_from_df(path1, path2, path3, path4, ax1, ax2, ax3, save_filename=None):

    data_path_p = path1
    data_path_f = path2
    data_path_bio_p = path3
    data_path_bio_f = path4

    df_plastic = pd.read_csv(data_path_p)
    columns = [
        "Step",
        "plastic-hc",
        "plastic-hc_min",
        "plastic-hc_max",
        "plastic-hc-cb",
        "plastic-hc-cb_min",
        "plastic-hc-cb_max",
    ]
    df_plastic.columns = columns

    # for fixed ordered is swapped
    df_fixed = pd.read_csv(data_path_f)
    columns = [
        "Step",
        "hc-cb",
        "hc-cb_min",
        "hc-cb_max",
        "hc",
        "hc_min",
        "hc_max",
    ]
    df_fixed.columns = columns

    df_bio_plastic = update_cols_name(data_path_bio_p, "bio-plastic-hc-cb")
    df_bio_fixed = update_cols_name(data_path_bio_f, "bio-hc-cb")

    df = pd.concat(
        [df_plastic, df_fixed, df_bio_plastic, df_bio_fixed],
        axis=1,
        ignore_index=False,
        sort=False,
    )

    start = 0  # 8000
    end = 15000  # 10000
    df_ewma = df.ewm(alpha=0.02).mean()[
        start:end
    ]  # .plot(x='Step', y=['hcc','drqn'])[:4000]
    df_ewma_var = df.ewm(alpha=0.02).var()[start:end]
    X = np.linspace(0, end - start, num=end - start)

    # df_ewma = df_ewma * 100
    # df_ewma_var = df_ewma_var * 100
    animal_x = np.linspace(1, 10, num=10)

    ax1.errorbar(
        animal_x,
        animal_data_y[:10],
        marker="s",
        color="black",
        markerfacecolor="white",
        # alpha=1,
        label="control",
        yerr=(np.array(animal_data_y[20:30]) - np.array(animal_data_y[:10])) / 2,
    )
    ax1.errorbar(
        animal_x,
        animal_data_y[10:20],
        marker="s",
        # color=model_color["plastic-hc"],
        color="black",
        # alpha=1,
        label="L7-PKCI",
        yerr=(np.array(animal_data_y[30:40]) - np.array(animal_data_y[10:20])) / 2,
    )

    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_ylim([0, 100])
    ax1.set(title="Animal data", xlabel="Training session", ylabel="Escape latency (s)")
    ax1.legend()

    ax2.plot(X, df_ewma["hc"], color=model_color["hc"], alpha=1, label="hc")
    ax2.fill_between(
        X,
        df_ewma["hc"] - df_ewma_var["hc"].sem(),
        df_ewma["hc"] + df_ewma_var["hc"].sem(),
        color=model_color["hc"],
        alpha=0.5,
    )

    ax2.plot(X, df_ewma["hc-cb"], color=model_color["hc-cb"], alpha=1, label="hc-cb")
    ax2.fill_between(
        X,
        df_ewma["hc-cb"] - df_ewma_var["hc-cb"].sem(),
        df_ewma["hc-cb"] + df_ewma_var["hc-cb"].sem(),
        color=model_color["hc-cb"],
        alpha=0.5,
    )

    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_xlim([0, 2100])
    ax2.set(title="Episode steps", xlabel="Trials", ylabel="Steps")
    ax2.legend()

    width = 0.3

    ax3.bar(
        ["hc-cb"],
        df_ewma["hc-cb"].mean(),
        yerr=df_ewma_var["hc-cb"].sem(),
        # yerr=df_ewma["hc-cb"].sem(),
        width=width,
        color=model_color["hc-cb"],
    )
    ax3.bar(
        ["hc"],
        df_ewma["hc"].mean(),
        yerr=df_ewma_var["hc"].sem(),
        # yerr=df_ewma["hc"].sem(),
        width=width,
        color=model_color["hc"],
    )

    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    ax3.set(title="Train", ylabel="Steps")
    ax3.tick_params(axis="x", labelrotation=30)
    # ax3.hlines(y=50, xmin=-0.2, xmax=5.2, linewidth=2, color="black", linestyle="--")

    return ax1, ax2, ax3


data_path_p = "/home/rh19400/neuro-rl/plots/data/plastic_rew.csv"
data_path_f = "/home/rh19400/neuro-rl/plots/data/fixed_rew.csv"
data_path_bio_p = "/home/rh19400/neuro-rl/plots/data/bio_plastic_rew.csv"
data_path_bio_f = "/home/rh19400/neuro-rl/plots/data/bio_rew.csv"
# ax1, ax2, ax3 = plot_from_df(
#     data_path_p, data_path_f, data_path_bio_p, data_path_bio_f, "all_reward"
# )


fig = plt.figure(figsize=(16, 8))
# fig = plt.figure(figsize=(16, 4))
gs = plt.GridSpec(2, 6, wspace=0.5, hspace=0.5)
ax1 = fig.add_subplot(gs[0, 0:2])
ax2 = fig.add_subplot(gs[0, 2:4])
ax3 = fig.add_subplot(gs[0, 4:6])
ax4 = fig.add_subplot(gs[1, 0:2])
ax5 = fig.add_subplot(gs[1, 2:3])
ax6 = fig.add_subplot(gs[1, 3:5])
ax7 = fig.add_subplot(gs[1, 5:6])

# ax1, ax2, ax3 = plot_from_df(
_, _, _ = plot_from_df(
    data_path_p,
    data_path_f,
    data_path_bio_p,
    data_path_bio_f,
    ax1,
    ax4,
    ax5,
    "all_reward",
)
data_path_p = "/home/rh19400/neuro-rl/plots/data/plastic_steps.csv"
data_path_f = "/home/rh19400/neuro-rl/plots/data/fixed_steps.csv"
data_path_bio_p = "/home/rh19400/neuro-rl/plots/data/bio_plastic_steps.csv"
data_path_bio_f = "/home/rh19400/neuro-rl/plots/data/bio_steps.csv"
ax4, ax5, ax6 = plot_steps_from_df(
    data_path_p,
    data_path_f,
    data_path_bio_p,
    data_path_bio_f,
    ax3,
    ax6,
    ax7,
    "all_steps",
)

# Add a bold 'a' or 'b' in the top-left corner
ax1.text(-0.25, 1.00, "A", transform=ax1.transAxes, fontsize=18, weight="bold")  # 'a'
ax2.text(1.40, 1.00, "B", transform=ax1.transAxes, fontsize=18, weight="bold")  # 'a'
ax3.text(3.00, 1.00, "C", transform=ax1.transAxes, fontsize=18, weight="bold")  # 'a'
ax3.text(-0.55, -0.45, "D", transform=ax1.transAxes, fontsize=18, weight="bold")  # 'a'
ax3.text(2.15, -0.45, "E", transform=ax1.transAxes, fontsize=18, weight="bold")  # 'a'

# Add animal spatial water maze
img = plt.imread("animal_wm.png")
ax2.imshow(img)  # , extent=[0, 10, 0, 10])
ax2.set(title="Animal spatial watermaze", xlabel="X", ylabel="Y")
ax2.set_axis_off()


plt.tight_layout()
plt.show()
save_filename = "figure_p3"
# fig.savefig(f"{save_filename}.pdf", bbox_inches="tight")
