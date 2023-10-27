from turtle import st
import numpy as np
from ahrs.filters import Complementary
import orientation_estimation as oe
import matplotlib.pyplot as plt
from ahrs.filters import EKF
from ahrs.common.orientation import acc2q
import scipy.stats as stats
from scipy.stats import ttest_1samp


def plot_euler(
    fs: int,
    ground_truth: np.ndarray,
    estimate: np.ndarray,
    estimate_name: str,
    title: str,
    drop: int = 0,
):
    """
    ground_truth: ground truth euler angles (N, 3)
    estimate: estimated euler angles (N, 3)
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    labels = ["Ground Truth", estimate_name]
    ground_truth = ground_truth[:, drop:]
    estimate = estimate[drop:]

    for i in range(3):
        time = np.arange(len(ground_truth[:, i])) / fs
        axs[i].plot(time, ground_truth[:, i], label=labels[0])
        axs[i].plot(time, estimate[:, i], label=labels[1])
        axs[i].set_title(["Roll", "Pitch", "Yaw"][i])
        axs[i].set_ylabel("Angle [deg]")
        axs[i].legend()

    axs[2].set_xlabel("Time [s]")

    fig.suptitle(title)
    plt.show()


def plot_rmse(
    ground_truths: np.ndarray, filter: np.ndarray, filter_name: str, sampling_frequency
):
    n_runs, n_samples, _ = ground_truths.shape
    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    time = np.arange(n_samples) / sampling_frequency
    time = time[50:]

    # seq_names = ["Static Sequence", "Dynamic Sequence 1", "Dynamic Sequence 2"]
    # for seq in range(3):
    for axis in range(3):
        error = ((ground_truths[:, :, axis] - filter[:, :, axis]) ** 2).mean(axis=0)[
            50:
        ]

        rmse = np.sqrt(error)

        axs[axis].plot(time, rmse, label="RMSE")
        axs[axis].set_title(["Roll", "Pitch", "Yaw"][axis])
        axs[axis].set_ylabel("RMSE [deg]")
        axs[axis].legend()
        # axs[axis].set_ylim([0, 6])

    axs[2].set_xlabel("Time [s]")
    fig.suptitle("Mean RMSE of " + filter_name + " for " + str(n_runs) + " runs")
    plt.show()


def calculate_rmse(ground_truths, estimates):
    rmses = []
    for ground_truth, estimate in zip(ground_truths, estimates):
        rmse = np.sqrt(np.mean((ground_truth - estimate) ** 2, axis=1))
        rmses.append(rmse)
    return np.array(rmses)


def compute_comparison_metrics(comp_rmse, ekf_rmse, intervals):
    """
    Compute comparison metrics for given RMSE datasets and time intervals.

    Parameters:
    - comp_rmse : numpy array
        RMSE values for the comp_filter algorithm.
    - ekf_rmse : numpy array
        RMSE values for the ekf algorithm.
    - intervals : list of tuples
        Time intervals for which to compute the metrics. E.g., [(40,60), (60,70)]

    Returns:
    - table_data : list of tuples
        Results in the format:
        (start, end, comp_mean, comp_std, t_stat, ekf_mean, ekf_std)
    """

    table_data = []

    for start, end in intervals:
        start *= 100
        end *= 100
        # Ensure the interval is valid
        if (
            start >= comp_rmse.shape[1]
            or end > comp_rmse.shape[1]
            or start >= ekf_rmse.shape[1]
            or end > ekf_rmse.shape[1]
        ):
            print(f"Warning: Skipping interval ({start},{end}) as it's out of bounds.")
            continue

        # Slice the RMSE data for the current time interval
        comp_data_interval = comp_rmse[:, start:end]
        ekf_data_interval = ekf_rmse[:, start:end]

        # Check if there's data for the current interval
        if comp_data_interval.size == 0 or ekf_data_interval.size == 0:
            print(
                f"Warning: No data available for interval ({start},{end}). Skipping calculations for this interval."
            )
            continue

        # Remaining calculations as before
        comp_mean = np.mean(comp_data_interval)
        comp_std = np.std(comp_data_interval)
        ekf_mean = np.mean(ekf_data_interval)
        ekf_std = np.std(ekf_data_interval)

        # t_stat, p_value = stats.ttest_ind(comp_data_interval, ekf_data_interval, axis=0)

        mean_performance_differences = np.mean(
            comp_rmse[:, start:end] - ekf_rmse[:, start:end], axis=0
        )
        t_stat, p_value = ttest_1samp(mean_performance_differences, 0)
        if p_value < 0.05:
            print(
                "Reject null hypothesis: There is a significant difference in performance."
            )
            if np.mean(mean_performance_differences) > 0:
                print("EKF has better performance.")
            else:
                print("Complementary Filter has better performance.")
        else:
            print(
                "Fail to reject null hypothesis: No significant difference in performance."
            )

        table_data.append((start, end, comp_mean, comp_std, t_stat, ekf_mean, ekf_std))

    return table_data


def comp_filter(acc, gyr, mag, ground_truth, fs):
    # convert gyr from deg/s to rad/s
    gyr = np.deg2rad(gyr)
    # convert mag from uT to mT
    mag = -mag * 1e-3
    # convert acc from g to m/s^2
    acc = acc * 9.81
    complementary_filter = Complementary(
        gyr=gyr,
        acc=acc,
        mag=mag,
        frequency=fs,
        gain=0.1,
    )

    comp_filter_estimates = complementary_filter._compute_all()
    comp_euler = oe.quaternions_to_eulers_deg(comp_filter_estimates)
    comp_euler[:, 0], comp_euler[:, 1] = -comp_euler[:, 1], -comp_euler[:, 0].copy()

    # plot_euler(
    #     fs,
    #     ground_truth,
    #     comp_euler,
    #     "Complementary Filter",
    #     "Complementary Filter vs Ground Truth"
    # )

    return comp_euler


def ekf(acc, gyr, mag, ground_truth, fs, num_samples):
    # convert gyr from deg/s to rad/s
    gyr = -np.deg2rad(gyr)
    mag = -mag
    # convert acc from g to m/s^2
    acc = -acc * 9.81

    ekf = EKF(frequency=100, frame="ENU")
    Q = np.zeros((num_samples, 4))  # Allocate array for quaternions
    Q[0] = acc2q(np.array([0, 0, 0]))

    for t in range(1, num_samples):
        Q[t] = ekf.update(
            Q[t - 1],
            gyr[t],
            acc[t],
            mag[t],
        )

    ekf_euler = oe.quaternions_to_eulers_deg(Q)
    ekf_euler[:, 0], ekf_euler[:, 1] = ekf_euler[:, 1], ekf_euler[:, 0].copy()
    ekf_euler[:, 0] *= -1
    ekf_euler[:, 0] += 5
    ekf_euler[:, 1] -= 160
    ekf_euler[:, 2] *= -1

    # plot_euler(fs, ground_truth, ekf_euler, "EKF", "EKF vs Ground Truth", drop=50)
    return ekf_euler


print("Starting")

ground_truths = []
comp_runs = []
ekf_runs = []

num_runs = 3
for i in range(0, num_runs):
    acc = np.load("Collected_Data/A_List_" + str(i + 1) + ".npy")
    gyr = np.load("Collected_Data/G_List_" + str(i + 1) + ".npy")
    mag = np.load("Collected_Data/M_List_" + str(i + 1) + ".npy")
    roll = np.load("Collected_Data/rollList_" + str(i + 1) + ".npy")
    print("roll", roll)
    pitch = np.load("Collected_Data/pitchList_" + str(i + 1) + ".npy")
    print("pitch", pitch)
    yaw = np.load("Collected_Data/yawList_" + str(i + 1) + ".npy")
    print("yaw", yaw)

    ground_truths.append(ground_truth := np.transpose(np.array((roll, pitch, yaw))))
    comp_runs.append(comp_res := comp_filter(acc, gyr, mag, ground_truth, 100))
    ekf_runs.append(ekf_res := ekf(acc, gyr, mag, ground_truth, 100, len(acc)))

    # plot_euler(100, ground_truth, comp_res, "Comp", "Comp vs Ground Truth")
    # plot_euler(100, ground_truth, ekf_res, "EKF", "EKF vs Ground Truth")


# plot_rmse(np.array(ground_truths), np.array(comp_runs), "Complementary Filter", 100)
# plot_rmse(np.array(ground_truths), np.array(ekf_runs), "EKF", 100)


comp_rmse = calculate_rmse(ground_truths, comp_runs)
print("comp rmse: ", comp_rmse)
ekf_rmse = calculate_rmse(ground_truths, ekf_runs)
print("ekf rmse: ", len(ekf_rmse))

intervals = [(0, 10), (11, 13), (14, 20)]
results = compute_comparison_metrics(comp_rmse, ekf_rmse, intervals)
print("RESULTS:", results)


print("Done")
