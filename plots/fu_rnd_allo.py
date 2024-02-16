import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import unit_metric_computers as umc

# dp_hc_state_s1 = "/home/rh19400/neuro-rl/exp_local/2024.02.12/112629_hccstate_agent=hccstate,experiment=test_rnd_allo,save_stats=true,seed=1/activations"
# # s2 change starting location
# dp_hc_state_s2 = "/home/rh19400/neuro-rl/exp_local/2024.02.12/113916_hccstate_agent=hccstate,experiment=test_rnd_allo,save_stats=true,seed=1/activations"

# # s1 test trained at 2k
dp_hc_state_s1 = "/home/rh19400/neuro-rl/exp_local/2024.02.12/123423_hccstate_agent=hccstate,experiment=test_rnd_allo,save_stats=true,seed=1/activations"
dp_hc_state_s2 = "/home/rh19400/neuro-rl/exp_local/2024.02.12/124027_hccstate_agent=hccstate,experiment=test_rnd_allo,save_stats=true,seed=1/activations"
# # s3 trained at 2k
dp_hc_state_s3 = "/home/rh19400/neuro-rl/exp_local/2024.02.12/133348_hccstate_agent=hccstate,experiment=test_rnd_allo,save_stats=true,seed=1/activations"
# # goal moved and loc changed
# dp_hc_state_sg2 = "/home/rh19400/neuro-rl/exp_local/2024.02.12/142150_hccstate_agent=hccstate,experiment=test_rnd_allo,save_stats=true,seed=1/activations"

# # hc trained at 2k
dp_hc_s1 = "/home/rh19400/neuro-rl/exp_local/2024.02.12/134834_drqn_agent=drqn,experiment=test_rnd_allo,save_stats=true,seed=1/activations"
dp_hc_s2 = "/home/rh19400/neuro-rl/exp_local/2024.02.12/135309_drqn_agent=drqn,experiment=test_rnd_allo,save_stats=true,seed=1/activations"
dp_hc_s3 = "/home/rh19400/neuro-rl/exp_local/2024.02.12/134229_drqn_agent=drqn,experiment=test_rnd_allo,save_stats=true,seed=1/activations"
# # test reward moved
# dp_hc_sg2 = "/home/rh19400/neuro-rl/exp_local/2024.02.12/140857_drqn_agent=drqn,experiment=test_rnd_allo,save_stats=true,seed=1/activations"

# # test trained at 1k with wall cue
# dp_hc_s1 = "/home/rh19400/neuro-rl/exp_local/2024.02.12/180947_drqn_agent=drqn,experiment=test_allo_wall,save_stats=true,seed=1/activations"
# dp_hc_s2 = "/home/rh19400/neuro-rl/exp_local/2024.02.12/181502_drqn_agent=drqn,experiment=test_allo_wall,save_stats=true,seed=1/activations"
# dp_hc_s3 = "/home/rh19400/neuro-rl/exp_local/2024.02.12/182155_drqn_agent=drqn,experiment=test_allo_wall,save_stats=true,seed=1/activations"

# dp_hc_state_s1 = "/home/rh19400/neuro-rl/exp_local/2024.02.12/181030_hccstate_agent=hccstate,experiment=test_allo_wall,save_stats=true,seed=1/activations"
# dp_hc_state_s2 = "/home/rh19400/neuro-rl/exp_local/2024.02.12/181506_hccstate_agent=hccstate,experiment=test_allo_wall,save_stats=true,seed=1/activations"
# dp_hc_state_s3 = "/home/rh19400/neuro-rl/exp_local/2024.02.12/182203_hccstate_agent=hccstate,experiment=test_allo_wall,save_stats=true,seed=1/activations"

# test with 1k with 2 cube cues
dp_hc_s1 = "/home/rh19400/neuro-rl/exp_local/2024.02.13/091628_drqn_agent=drqn,experiment=test_allo_cue,save_stats=true,seed=1/activations"
dp_hc_s2 = "/home/rh19400/neuro-rl/exp_local/2024.02.13/092537_drqn_agent=drqn,experiment=test_allo_cue,save_stats=true,seed=1/activations"
# start loc moved half way
dp_hc_s2_2 = "/home/rh19400/neuro-rl/exp_local/2024.02.13/094009_drqn_agent=drqn,experiment=test_allo_cue,save_stats=true,seed=1/activations"
dp_hc_s3 = "/home/rh19400/neuro-rl/exp_local/2024.02.13/094924_drqn_agent=drqn,experiment=test_allo_cue,save_stats=true,seed=1/activations"

dp_hc_state_s1 = "/home/rh19400/neuro-rl/exp_local/2024.02.13/091700_hccstate_agent=hccstate,experiment=test_allo_cue,save_stats=true,seed=1/activations"
dp_hc_state_s2 = "/home/rh19400/neuro-rl/exp_local/2024.02.13/092541_hccstate_agent=hccstate,experiment=test_allo_cue,save_stats=true,seed=1/activations"
# start loc moved half way
dp_hc_state_s2_2 = "/home/rh19400/neuro-rl/exp_local/2024.02.13/094010_hccstate_agent=hccstate,experiment=test_allo_cue,save_stats=true,seed=1/activations"
dp_hc_state_s3 = "/home/rh19400/neuro-rl/exp_local/2024.02.13/094926_hccstate_agent=hccstate,experiment=test_allo_cue,save_stats=true,seed=1/activations"


def get_df(data_path):
    df_ca1 = pd.read_csv(f"{data_path}/ca1.csv")
    df_ca3 = pd.read_csv(f"{data_path}/ca3.csv")
    return df_ca1, df_ca3


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


def _single_env_produce_unit_chart(
    # config_version,
    # experiment,
    # moving_trajectory,
    data_ca1,
    name="ca1",
    reference_experiment=None,
    feature_selection=None,
    decoding_model_choice=None,
    sampling_rate=None,
    random_seed=None,
    sorted_by=None,
    filterings=None,
    # charting all units, use Nones to maintain API consistency
):
    """
    Produce unit chart for each unit and save to disk, which
    will be used for plotting by `_single_env_viz_unit_chart`.

    Unit chart is intended to capture characteristics of each
    unit (no filtering; ALL units). Currently, the chart includes:
        0. if dead (if true, continue to next unit)
        .  fields info - [
                1. num_clusters,
                2. num_pixels_in_clusters,
                3. max_value_in_clusters,
                4. mean_value_in_clusters,
                5. var_value_in_clusters,
                6. entire_map_mean,
                7. entire_map_var,
            ]
        8. gridness - gridness score
        9. borderness - border score
        .  directioness - [
                10. mean_vector_length,
                11. per_rotation_vector_length,
            ]
    """
    # os.environ["TF_NUM_INTRAOP_THREADS"] = f"{TF_NUM_INTRAOP_THREADS}"
    # os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    # config = utils.load_config(config_version)
    config = "first"
    logging.info(f"[Begin run] config: {config}")
    movement_mode = "2d"

    # charted info:
    charted_info = [
        "dead",
        # --- fields info
        "num_clusters",
        "num_pixels_in_clusters",
        "max_value_in_clusters",
        "mean_value_in_clusters",
        "var_value_in_clusters",
        "entire_map_mean",
        "entire_map_var",
        # ---
        "gridness",
        # ---
        "borderness",
        # --- directioness
        "mean_vector_length",
        "per_rotation_vector_length",
        # ---
    ]

    # initialize unit chart collector
    # shape \in (total_n_units, len(charted_info))
    # init from zero so as we first check if a unit is dead
    # if dead, we continue to next unit so the rest of the info re this unit
    # will be kept as zero
    unit_chart_info = np.zeros((data_ca1.shape[0], len(charted_info)), dtype=object)
    # model_reps_summed = np.sum(model_reps, axis=1, keepdims=True)

    # for unit_index in range(model_reps_summed.shape[2]):
    for unit_index in range(data_ca1.shape[0]):
        logging.info(f"[Charting] unit_index: {unit_index}")

        if movement_mode == "2d":
            # # reshape to (n_locations, n_rotations, n_features)
            # # but rotation dimension is 1 after summing, so
            # # we just hard code 0.
            # heatmap = model_reps_summed[:, 0, unit_index].reshape(
            #     (
            #         env_x_max * multiplier - env_x_min * multiplier + 1,
            #         env_y_max * multiplier - env_y_min * multiplier + 1,
            #     )
            # )
            # # rotate heatmap to match Unity coordinate system
            # # ref: tests/testReshape_forHeatMap.py
            # heatmap = np.rot90(heatmap, k=1, axes=(0, 1))
            heatmap = data_ca1[unit_index]

            ###### Go thru each required info, maybe modularise later.
            if umc._is_dead_unit(heatmap):
                logging.info(f"Unit {unit_index} dead.")
                unit_chart_info[unit_index, 0] = np.array([0])
                continue
            else:
                logging.info(f"Unit {unit_index} active")
                unit_chart_info[unit_index, 0] = np.array([1])
                # compute, collect and save unit chart info
                # 1. fields info
                (
                    num_clusters,
                    num_pixels_in_clusters,
                    max_value_in_clusters,
                    mean_value_in_clusters,
                    var_value_in_clusters,
                    bounds_heatmap,
                ) = umc._compute_single_heatmap_fields_info(
                    heatmap=heatmap,
                    pixel_min_threshold=5,
                    pixel_max_threshold=15,  # int(heatmap.shape[0] * heatmap.shape[1] * 0.5),
                )
                unit_chart_info[unit_index, 1] = num_clusters
                unit_chart_info[unit_index, 2] = num_pixels_in_clusters
                unit_chart_info[unit_index, 3] = max_value_in_clusters
                unit_chart_info[unit_index, 4] = mean_value_in_clusters
                unit_chart_info[unit_index, 5] = var_value_in_clusters
                unit_chart_info[unit_index, 6] = np.array([np.mean(heatmap)])
                unit_chart_info[unit_index, 7] = np.array([np.var(heatmap)])

                # # 2. gridness
                # (
                #     score_60_,
                #     _,
                #     _,
                #     _,
                #     sac,
                #     scorer,
                # ) = umc._compute_single_heatmap_grid_scores(heatmap)
                # unit_chart_info[unit_index, 8] = score_60_

                # 3. borderness
                border_score = umc._compute_single_heatmap_border_scores(heatmap, db=3)
                unit_chart_info[unit_index, 9] = border_score

                # 4. directioness (use model_reps instead of model_reps_summed)
                # (
                #     directional_score,
                #     per_rotation_vector_length,
                # ) = umc._compute_single_heatmap_directional_scores(
                #     activation_maps=model_reps[:, :, unit_index]
                # )
                # unit_chart_info[unit_index, 10] = directional_score
                # unit_chart_info[unit_index, 11] = per_rotation_vector_length

    # results_path = utils.load_results_path(
    #     config=config,
    #     experiment=experiment,
    #     moving_trajectory=moving_trajectory,
    # )
    results_path = "results_xl"
    fpath = f"{results_path}/unit_chart_{name}.npy"
    np.save(fpath, unit_chart_info)
    logging.info(f"[Saved] {fpath}")
    print("saved")


def _single_env_viz_unit_chart(
    # config_version,
    # experiment,
    # moving_trajectory,
    name="ca1",
    axes=None,
    reference_experiment=None,
    feature_selection=None,
    decoding_model_choice=None,
    sampling_rate=None,
    random_seed=None,
    sorted_by=None,
    filterings=None,
):
    """
    Visualize unit chart info produced by `_single_env_produce_unit_chart`.
    """
    charted_info = [
        "dead",
        "num_clusters",
        "num_pixels_in_clusters",
        "max_value_in_clusters",
        "gridness",
        "borderness",
        "directioness",
    ]

    # config = utils.load_config(config_version)

    # # load unit chart info
    # results_path = utils.load_results_path(
    #     config=config,
    #     experiment=experiment,
    #     moving_trajectory=moving_trajectory,
    # )
    results_path = "results_xl"
    unit_chart_info = np.load(
        f"{results_path}/unit_chart_{name}.npy", allow_pickle=True
    )
    logging.info(f"unit_chart_info.shape: {unit_chart_info.shape}")

    # Given `unit_chart_info`, for now we test `% dead`, `% num_clusters`
    #   For `% dead`, we iterate through unit_chart_info and count
    # how many units are dead along the first column (i.e. unit_chart_info[i, 0] == 0)
    # we return the percentage of dead units.
    #   For `% num_clusters`, we iterate through unit_chart_info and count
    # how many units have 1, 2, .., max num_clusters.
    # and for each unique `num_clusters`, we return the percentage of corresponding units.
    #   For `% cluster size`, we iterate through unit_chart_info and count
    # the sizes of fields of qualified units.
    #   For `% peak cluster activation`, we iterate through unit_chart_info and count
    # the peak activation of fields of qualified units.
    #   For `% borderness`, we iterate through unit_chart_info and count
    # the borderness of qualified units (w borderness>0.5).
    n_dead_units = 0
    max_num_clusters = np.max(
        unit_chart_info[:, 1]
    )  # global max used for setting xaxis.
    num_clusters = np.zeros(max_num_clusters + 1)
    cluster_sizes = []
    cluster_peaks = []
    grid_cell_indices = []
    border_cell_indices = []
    place_cells_indices = []
    direction_cell_indices = []

    for unit_index in range(unit_chart_info.shape[0]):
        if unit_chart_info[unit_index, 0] == 0:
            n_dead_units += 1
        else:
            num_clusters[int(unit_chart_info[unit_index, 1])] += 1
            cluster_sizes.extend(unit_chart_info[unit_index, 2])
            cluster_peaks.extend(unit_chart_info[unit_index, 3])
            if unit_chart_info[unit_index, 1] > 0:
                place_cells_indices.append(unit_index)
            if unit_chart_info[unit_index, 6] > 0.47:
                direction_cell_indices.append(unit_index)
            if unit_chart_info[unit_index, 8] > 0.37:
                grid_cell_indices.append(unit_index)
            if unit_chart_info[unit_index, 9] > 0.5:
                border_cell_indices.append(unit_index)

    # plot
    n_dead_units = n_dead_units
    n_active_units = unit_chart_info.shape[0] - n_dead_units
    n_place_cells = len(place_cells_indices)
    n_grid_cells = len(grid_cell_indices)
    n_border_cells = len(border_cell_indices)
    n_direction_cells = len(direction_cell_indices)

    if axes is None:
        fig, axes = plt.subplots(nrows=1, ncols=8, figsize=(5 * 8, 5))
    # fig, axes = plt.subplots(nrows=8, ncols=1, figsize=(5, 5 * 8))

    # 0-each bar is % of dead/active units
    # left bar is dead, right bar is active,
    # plot in gray for dead units, plot in blue for active units
    axes[0].bar(
        np.arange(2),
        [
            n_dead_units / unit_chart_info.shape[0],
            (n_active_units) / unit_chart_info.shape[0],
        ],
        color=["gray", "blue"],
    )
    axes[0].set_xticks(np.arange(2))
    axes[0].set_xticklabels(["dead", "active"])
    axes[0].set_ylabel("% units")
    axes[0].set_title(f"% units dead/active")
    axes[0].set_ylim([-0.05, 1.05])
    # axes[0].grid()

    # 1-each bar is % of a num_clusters
    axes[1].bar(np.arange(max_num_clusters + 1), num_clusters / n_active_units)
    axes[1].set_xlabel("num_clusters")
    axes[1].set_ylabel("% units")
    axes[1].set_title(f"% units with 1, 2, .., {max_num_clusters[0]} clusters")
    axes[1].set_ylim([-0.05, 1.05])
    # axes[1].grid()

    # 2-each bar is % of a cluster size (bined)
    axes[2].hist(cluster_sizes, bins=20, density=True)
    axes[2].set_xlabel("cluster size")
    axes[2].set_ylabel("density")
    axes[2].set_title(f"cluster size distribution")
    # axes[2].grid()

    # # 3-each bar is % of a cluster peak (bined)
    # axes[3].hist(cluster_peaks, bins=20, density=True)
    # axes[3].set_xlabel("cluster peak")
    # axes[3].set_ylabel("density")
    # axes[3].set_title(f"cluster peak distribution")
    # axes[3].grid()

    # # 4-each bar is % of a gridness
    # # non-grid left, grid right
    # axes[4].bar(
    #     np.arange(2),
    #     [1 - n_grid_cells / n_active_units, n_grid_cells / n_active_units],
    #     color=["grey", "blue"],
    # )
    # axes[4].set_xticks(np.arange(2))
    # axes[4].set_xticklabels(["non-grid", "grid"])
    # axes[4].set_ylabel("% units")
    # axes[4].set_title(f"% units grid/non-grid")
    # axes[4].set_ylim([-0.05, 1.05])
    # axes[4].grid()

    # 5-each bar is % of a borderness
    # non-border left, border right
    axes[3].bar(
        np.arange(2),
        [1 - n_border_cells / n_active_units, n_border_cells / n_active_units],
        color=["grey", "blue"],
    )
    axes[3].set_xticks(np.arange(2))
    axes[3].set_xticklabels(["non-border", "border"])
    axes[3].set_ylabel("% units")
    axes[3].set_title(f"% units border/non-border")
    axes[3].set_ylim([-0.05, 1.05])
    # axes[3].grid()

    # # 6-each bar is % of a directioness
    # axes[6].hist(unit_chart_info[:, 10], bins=20, density=True)
    # axes[6].set_xlabel("directioness")
    # axes[6].set_ylabel("density")
    # axes[6].set_title(f"directioness distribution")
    # axes[6].grid()

    # 7-each bar is % of place cells
    # non-place left, place right
    axes[4].bar(
        np.arange(2),
        [1 - n_place_cells / n_active_units, n_place_cells / n_active_units],
        color=["grey", "blue"],
    )
    axes[4].set_xticks(np.arange(2))
    axes[4].set_xticklabels(["non-place", "place"])
    axes[4].set_ylabel("% units")
    axes[4].set_title(f"% units place/non-place")
    axes[4].set_ylim([-0.05, 1.05])
    # axes[4].grid()

    # figs_path = utils.load_figs_path(
    #     config=config,
    #     experiment=experiment,
    #     moving_trajectory=moving_trajectory,
    # )
    figs_path = "results_xl"

    plt.tight_layout()
    # plt.suptitle(f'{config["output_layer"]}')
    # plt.savefig(f"{figs_path}/unit_chart_{name}.png")
    # plt.close()

    # TODO: improve this plot
    # plot cell type proportions (dead, place, direction, border)
    # and for place, direction and border cells, we plot the overlap
    # between them as stacked bar chart (e.g. for the place cell
    # plot, we plot the proportion of place cells that are
    # exclusive place cells, place and direction cells, place and border cells, etc)
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    # Collect the indices of units that are all three types
    # (place + border + direction)
    place_border_direction_cells_indices = list(
        set(place_cells_indices)
        & set(border_cell_indices)
        & set(direction_cell_indices)
    )

    # Collect the indices of units that are two types (inc. three types)
    # (place + border cells)
    # (place + direction cells)
    # (border + direction cells)
    place_and_border_cells_indices = list(
        set(place_cells_indices) & set(border_cell_indices)
    )
    place_and_direction_cells_indices = list(
        set(place_cells_indices) & set(direction_cell_indices)
    )
    border_and_direction_cells_indices = list(
        set(border_cell_indices) & set(direction_cell_indices)
    )

    # Collect the indices of units that are only two types
    # (place  + border - direction),
    # (place  + direction   - border),
    # (border + direction   - place)
    place_and_border_not_direction_cells_indices = list(
        set(place_and_border_cells_indices) - set(place_border_direction_cells_indices)
    )
    place_and_direction_not_border_cells_indices = list(
        set(place_and_direction_cells_indices)
        - set(place_border_direction_cells_indices)
    )
    border_and_direction_not_place_cells_indices = list(
        set(border_and_direction_cells_indices)
        - set(place_border_direction_cells_indices)
    )

    # Collect the indices of units that are exclusive
    # place cells,
    # border cells,
    # direction cells
    exclusive_place_cells_indices = list(
        set(place_cells_indices)
        - (set(place_and_border_cells_indices) | set(place_and_direction_cells_indices))
    )
    exclusive_border_cells_indices = list(
        set(border_cell_indices)
        - (
            set(place_and_border_cells_indices)
            | set(border_and_direction_cells_indices)
        )
    )
    exclusive_direction_cells_indices = list(
        set(direction_cell_indices)
        - (
            set(place_and_direction_cells_indices)
            | set(border_and_direction_cells_indices)
        )
    )

    top = np.array(
        [
            n_dead_units / unit_chart_info.shape[0],
            len(exclusive_place_cells_indices) / n_active_units,
            len(exclusive_border_cells_indices) / n_active_units,
            # len(exclusive_direction_cells_indices) / n_active_units,
        ]
    )
    bottom = np.array([0.0, 0.0, 0.0])
    axes[5].bar(np.arange(3), top, bottom=bottom, label="exclusive")

    bottom += top
    top = np.array(
        [
            0,
            len(place_and_border_cells_indices)
            / n_active_units,  # place+border-direction
            len(place_and_border_cells_indices)
            / n_active_units,  # place+border-direction
            # 0,
        ]
    )
    axes[5].bar(np.arange(3), top, bottom=bottom, label="place+border")

    axes[5].set_xticks(np.arange(3))
    axes[5].set_xticklabels(["dead", "place", "border"])
    axes[5].set_ylabel("% units")
    # output_layer = config["output_layer"]
    # if output_layer == "predictions":
    #     output_layer = "logits"
    # ax.set_title(f"% units place/border/direction ({output_layer})")
    axes[5].set_ylim([-0.05, 1.05])
    # ax.grid()
    axes[5].legend()
    # plt.savefig(f"{figs_path}/unit_chart_overlaps_{name}.png")

    # return place_cells_indices, border_cell_indices, direction_cell_indices
    return (
        exclusive_place_cells_indices,
        exclusive_border_cells_indices,
        # place_and_border_cells_indices,
        # place_and_direction_cells_indices,
        axes,
        # ax,
    )


# unit = data_ca1[0]
# heatmap = unit

# vals = umc._compute_single_heatmap_fields_info(
#     heatmap=heatmap,
#     pixel_min_threshold=10,
#     pixel_max_threshold=int(heatmap.shape[0] * heatmap.shape[1] * 0.5),
# )
# # print(vals)
model_names_t = ["hc_state_s3"]
# model_names_t = ["hc_state_s2"]
# model_names_t = ["hc_state_s2_2"]
model_names = ["hc_state_s1"]
# # model_names_t = ["hc_state_sg2"]
# model_names_t = ["hc_s3"]
# model_names_t = ["hc_s2"]
# model_names_t = ["hc_s2_2"]
# model_names = ["hc_s1"]
# # # model_names_t = ["hc_sg2"]

# for name in model_names:
#     df_ca1, df_ca3 = get_df(eval(f"dp_{name}"))
#     data_ca1 = get_neurons_hm(df_ca1)
#     data_ca3 = get_neurons_hm(df_ca3)
#     _single_env_produce_unit_chart(data_ca1=data_ca1, name=f"{name}_ca1")
#     _single_env_produce_unit_chart(data_ca1=data_ca3, name=f"{name}_ca3")

plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
# plt.rcParams["axes.grid"] = False
fig = plt.figure(figsize=(5 * 6, 5 * 3))
gs = plt.GridSpec(3, 6, wspace=0.5, hspace=0.5)
axs_unit_hc = []
axs_unit_hc_cb = []
axs_unit_bio_hc_cb = []
for i in range(6):
    axs_unit_hc.append(fig.add_subplot(gs[0, i]))
    axs_unit_hc_cb.append(fig.add_subplot(gs[1, i]))
    axs_unit_bio_hc_cb.append(fig.add_subplot(gs[2, i]))

plc_hc, bdc_hc, axis = _single_env_viz_unit_chart(
    name=f"{model_names[0]}_ca1", axes=axs_unit_hc
)
plc_hc_gen_start, bdc_hc_gen_start, axis = _single_env_viz_unit_chart(
    f"{model_names_t[0]}_ca1", axs_unit_hc
)
# plc_hc_cb, bdc_hc_cb, axis = _single_env_viz_unit_chart(
#     name=f"{model_names[1]}_ca1", axes=axs_unit_hc_cb
# )
# plc_hc_cb_gen_start, bdc_hc_cb_gen_start, axis = _single_env_viz_unit_chart(
#     f"{model_names_t[1]}_ca1", axs_unit_hc
# )
# plc_bio_hc_cb, bdc_bio_hc_cb, axis = _single_env_viz_unit_chart(
#     name=f"{model_names[2]}_ca1", axes=axs_unit_bio_hc_cb
# )
# plc_bio_hc_cb_gen_start, bdc_bio_hc_cb_gen_start, axis = _single_env_viz_unit_chart(
#     f"{model_names_t[2]}_ca1", axs_unit_hc
# )

plc_hc_intersection = set(plc_hc) & set(plc_hc_gen_start)
# print(plc_hc_intersection)
print(plc_hc)
print(bdc_hc)
print(f"hc plc intersection: {len(plc_hc_intersection)}")
bdc_hc_intersection = set(bdc_hc) & set(bdc_hc_gen_start)
# print(f"hc bdc intersection: {len(bdc_intersection)}")
plc_hc_intersection = np.array(list(plc_hc_intersection))
bdc_hc_intersection = np.array(list(bdc_hc_intersection))

# plc_hc_cb_intersection = set(plc_hc_cb) & set(plc_hc_cb_gen_start)
# bdc_hc_cb_intersection = set(bdc_hc_cb) & set(bdc_hc_cb_gen_start)
# plc_hc_cb_intersection = np.array(list(plc_hc_cb_intersection))
# bdc_hc_cb_intersection = np.array(list(bdc_hc_cb_intersection))

# plc_bio_hc_cb_intersection = set(plc_bio_hc_cb) & set(plc_bio_hc_cb_gen_start)
# bdc_bio_hc_cb_intersection = set(bdc_bio_hc_cb) & set(bdc_bio_hc_cb_gen_start)
# plc_bio_hc_cb_intersection = np.array(list(plc_bio_hc_cb_intersection))
# bdc_bio_hc_cb_intersection = np.array(list(bdc_bio_hc_cb_intersection))


# add text to axes
axs_unit_hc[0].text(
    -0.6,
    0.5,
    "hc",
    transform=axs_unit_hc[0].transAxes,
    fontsize=14,
    fontweight="bold",
    va="top",
    ha="center",
)
axs_unit_hc[0].text(
    -0.6,
    -1.0,
    "hc-cb",
    transform=axs_unit_hc[0].transAxes,
    fontsize=14,
    fontweight="bold",
    va="top",
    ha="center",
)
axs_unit_hc[0].text(
    -0.6,
    -2.5,
    "bio-hc-cb",
    transform=axs_unit_hc[0].transAxes,
    fontsize=14,
    fontweight="bold",
    va="top",
    ha="center",
)
# fig.savefig(f"neurons_stats_ca1_gen_noisy.pdf", bbox_inches="tight")

# Set the number of rows and columns for the subplot grid
num_rows = 2
num_cols = 5

# Create a figure and a grid of subplots
# fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 6 * 2))
fig = plt.figure(figsize=(5 * 5, 5 * 2))
gs = plt.GridSpec(2, 5, wspace=0.5, hspace=0.5)
axs = []
for i in range(2):
    # axs.append(fig.add_subplot(gs[i, 0]))
    for j in range(5):
        axs.append(fig.add_subplot(gs[i, j]))

# Flatten the 2D array of Axes objects into a 1D array for easier indexing
# axs = axs.flatten()
# print(axs)

# add pie charts
pie_val_hc = [
    len(plc_hc_intersection) / len(plc_hc),
    1 - len(plc_hc_intersection) / len(plc_hc),
]
axs[0].pie(pie_val_hc, labels=["same place cells", "different pc"], autopct="%1.1f%%")
# pie_val_hc_cb = [
#     len(plc_hc_cb_intersection) / len(plc_hc_cb),
#     1 - len(plc_hc_cb_intersection) / len(plc_hc_cb),
# ]
# axs[1].pie(
#     pie_val_hc_cb, labels=["same place cells", "different pc"], autopct="%1.1f%%"
# )
# pie_val_bio_hc_cb = [
#     len(plc_bio_hc_cb_intersection) / len(plc_bio_hc_cb),
#     1 - len(plc_bio_hc_cb_intersection) / len(plc_bio_hc_cb),
# ]
# axs[2].pie(
#     pie_val_bio_hc_cb, labels=["same place cells", "different pc"], autopct="%1.1f%%"
# )


start_idx = 5
end_idx = 12
# name_idx = ["hc", "hc"]  # , "hc_cb", "bio_hc_cb"]
# model_names_og = ["hc_s1"]
# model_names_start = ["hc_s2"]
# model_names_start = ["hc_s2_2"]
# model_names_start = ["hc_sg2"]
# model_names_start = ["hc_s3"]
model_names_og = ["hc_state_s1"]
# # model_names_start = ["hc_state_sg2"]
# model_names_start = ["hc_state_s2"]
# model_names_start = ["hc_state_s2_2"]
model_names_start = ["hc_state_s3"]
# model_names_og = ["hc", "bio_hc_cb"]
# model_names_start = ["hc_gen_start", "bio_hc_cb_gen_start"]
# model_names_start = ["hc_gen_big", "hc_cb_gen_big", "bio_hc_cb_gen_big"]
# model_names_start = ["hc_gen_noisy", "hc_cb_gen_noisy", "bio_hc_cb_gen_noisy"]
max_indices = []


def get_max_index(data, indexes):
    # print(indexes.shape)
    # print(f'indexes: {indexes}')
    heatmap_data = data[indexes]
    # print(heatmap_data.shape, type(heatmap_data))
    # Reshape the 3D array into a 2D array where each row corresponds to a unit
    reshaped_data = heatmap_data.reshape(heatmap_data.shape[0], -1)

    # Find the column indices (maximum values within each 2D array)
    col_indices = np.argmax(reshaped_data, axis=1)

    # Calculate the row indices based on flattened indices
    num_cols = heatmap_data.shape[2]
    row_indices = col_indices // num_cols
    col_indices = col_indices % num_cols

    # Combine row and column indices
    max_indices_per_unit = np.column_stack((row_indices, col_indices))

    print("Maximum indices for each unit:")
    print(max_indices_per_unit.shape)
    return max_indices_per_unit


for name, n_start in zip(model_names_og, model_names_start):
    df_ca1, df_ca3 = get_df(eval(f"dp_{name}"))
    data_ca1 = get_neurons_hm(df_ca1)
    data_ca3 = get_neurons_hm(df_ca3)

    name_c = "hc"

    df_ca1_start, df_ca3_start = get_df(eval(f"dp_{n_start}"))
    data_ca1_start = get_neurons_hm(df_ca1_start)
    data_ca3_start = get_neurons_hm(df_ca3_start)

    max_idx_ca1 = get_max_index(data_ca1, eval(f"plc_{name_c}_intersection"))
    max_idx_ca1_start = get_max_index(
        data_ca1_start, eval(f"plc_{name_c}_intersection")
    )

    # max_idx_ca1_bdc = get_max_index(data_ca1, eval(f"bdc_{name_c}_intersection"))
    # max_idx_ca1_start_bdc = get_max_index(
    #     data_ca1_start, eval(f"bdc_{name_c}_intersection")
    # )

    # distance_bdc = max_idx_ca1_bdc - max_idx_ca1_start_bdc
    # distance_bdc = np.linalg.norm(distance_bdc, axis=1, ord=2)

    distance = max_idx_ca1 - max_idx_ca1_start
    print("distance.shape: ", distance.shape)
    distance = np.linalg.norm(distance, axis=1, ord=2)
    count = np.count_nonzero(distance < 2)
    print(f"{name}, count: {count}, distance: {distance.mean()}")
    # axs[start_idx].hist(distance)
    # axs[start_idx].set_title(f"{name} plc")

    # axs[start_idx + 3].hist(distance_bdc)
    # axs[start_idx + 3].set_title(f"{name} bdc")

    pie_val_zero = [count / len(distance), 1 - count / len(distance)]
    axs[start_idx].pie(pie_val_zero, labels=["stable", "unstable"], autopct="%1.1f%%")
    axs[start_idx].set_title(f"{name} plc")
    start_idx += 1

    hm_zero_idx = np.where(distance == 0)[0]
    hm_idx = np.where(distance == 0)[0]
    hm_one_idx = np.where(distance == 2)[0]

    print(hm_zero_idx)
    print(hm_one_idx)
    # print(data_ca1[hm_zero_idx[3]])
    # print(data_ca1_start[hm_zero_idx[3]])

    # max_indices.append(max_indices_per_unit)
    # print(max_indices_per_unit.shape)
    # print(max_index.shape)
    # # continue
    # # print(eval(f"plc_{name}"))
    # if name == "bio_hc_cb":
    hm_zero_idx = hm_zero_idx[0:1]
    hm_one_idx = hm_one_idx[0:1]
    hm_idx = np.concatenate((hm_zero_idx, hm_one_idx))
    # hm_idx = hm_idx[2:4]
    # else:
    #     continue
    plot_neurons_hm(
        data_ca1,
        name=f"{name}_ca1",
        axs=axs[3:5],
        # axs=axs[start_idx:end_idx],
        # idxs=hm_idx,  # ca1
        idxs=eval(f"plc_{name_c}_intersection")[hm_idx],  # ca1
        # idxs=eval(f"plc_{name}")[0:],  # ca3
    )
    # start_idx += 6
    # end_idx += 6
    plot_neurons_hm(
        data_ca1_start,
        name=f"{n_start}_ca1",
        axs=axs[8:10],
        # idxs=hm_idx,  # ca1
        idxs=eval(f"plc_{name_c}_intersection")[hm_idx],  # ca1
        # idxs=eval(f"bdc_{name}")[0:],  # ca3
    )
    # start_idx += 6
    # end_idx += 6
    # break
    axs[3].set(title="stable place cell")
    axs[4].set(title="unstable place cell")


# axs[0].text(
#     -0.2,
#     1.8,
#     "A",
#     transform=axs[0].transAxes,
#     fontsize=14,
#     va="top",
#     ha="center",
#     fontweight="bold",
# )
# axs[0].text(
#     0.5,
#     1.8,
#     "Place cells",
#     transform=axs[0].transAxes,
#     fontsize=14,
#     va="top",
#     ha="left",
#     fontweight="bold",
# )
# axs[6].text(
#     -0.2,
#     1.8,
#     "B",
#     transform=axs[6].transAxes,
#     fontsize=14,
#     va="top",
#     ha="center",
#     fontweight="bold",
# )
# axs[6].text(
#     0.5,
#     1.8,
#     "Border cells",
#     transform=axs[6].transAxes,
#     fontsize=14,
#     va="top",
#     ha="left",
#     fontweight="bold",
# )

# # find square distance between max indices
# max_indices = np.array(max_indices)
# distance = max_indices[0] - max_indices[1]
# print(distance.shape)
# distance = np.linalg.norm(distance, axis=1, ord=2)
# # # distance = np.sum([distance[:, 0] ** 2, distance[:, 1] ** 2])
# # print(distance)
# count = np.count_nonzero(distance < 1)
# print(count)
# axs[0].hist(distance)


# # sort filtered cells by firing rate
# df_ca1, df_ca3 = get_df(dp_hc)
# data_ca1 = get_neurons_hm(df_ca1)
# filtered_neurons = data_ca1[plc_hc]
# m_vals = filtered_neurons.mean(axis=(1,2))
# # Get the indices that would sort the mean values in descending order
# # sorted_indices = np.argsort(m_vals)#[::-1]
# sorted_indices = np.argsort(m_vals)[::-1]
# # Use take_along_axis to reorder the original array based on the sorted indices
# sorted_neurons = np.take_along_axis(filtered_neurons, sorted_indices[:, None, None], axis=0)
# plot_neurons_hm(sorted_neurons, name=f"hc_cb_ca1", axs=axs, idxs=None)

# plt.axis("off")
plt.tight_layout()
plt.show()
# fig.savefig(f"plc_stability_ca1.pdf", bbox_inches="tight")
