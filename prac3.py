import numpy as np
from ahrs.filters import Complementary
import matplotlib.pyplot as plt
from ahrs.filters import EKF
import scipy.stats as stats
from scipy.stats import ttest_1samp
from tabulate import tabulate
from ahrs import Quaternion
import copy


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
    # ground_truth = ground_truth[:, drop:]
    # estimate = estimate[:, drop:]

    for i in range(3):
        time = np.arange(len(ground_truth[:, i])) / fs
        axs[i].plot(time, ground_truth[:, i], label=labels[0])
        axs[i].plot(time, estimate[:, i], label=labels[1])
        axs[i].set_title(["Roll", "Pitch", "Yaw"][i])
        axs[i].set_ylabel("Angle [deg]")
        axs[i].legend()

    axs[2].set_xlabel("Time [s]")

    fig.suptitle(title)


def plot_rmse(
    ground_truths: np.ndarray, filter: np.ndarray, filter_name: str, sampling_frequency
):
    n_runs, n_samples, _ = ground_truths.shape
    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    time = np.arange(n_samples) / sampling_frequency
    time = time[50:]

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

        table_data.append(
            (start / 100, end / 100, comp_mean, comp_std, t_stat, ekf_mean, ekf_std)
        )
    # Tabulate the results after all intervals have been processed
    headers = [
        "Start",
        "End",
        "Comp. Mean",
        "Comp. Std. Dev.",
        "Test Statistic",
        "EKF Mean",
        "EKF Std. Dev.",
    ]
    print(tabulate(table_data, headers=headers))

    return table_data


def comp_filter(acc, gyr, mag, ground_truth, fs):
    acc *= 9.81
    mag *= 1000
    complementary_filter = Complementary(
        gyr=gyr,
        acc=acc,
        mag=mag,
        frequency=fs,
        gain=0.1,
    )

    comp_filter_estimates = complementary_filter._compute_all()
    comp_euler = np.array(
        [Quaternion(q).to_angles() * 180 / np.pi for q in comp_filter_estimates]
    )
    comp_euler[:, 0], comp_euler[:, 1] = comp_euler[:, 1], comp_euler[:, 0].copy()
    comp_euler[:, 2] *= -1
    comp_euler[:, 2] += 90
    return comp_euler


def ekf(acc, gyr, mag, ground_truth, fs, num_samples):
    acc *= 9.81
    mag *= 1000

    ekf = EKF(frequency=100, frame="NED", magnetic_ref=mag[0])
    Q = np.zeros((num_samples, 4))  # Allocate array for quaternions
    Q[0] = Quaternion()

    for t in range(1, num_samples):
        Q[t] = ekf.update(
            Q[t - 1],
            gyr[t],
            acc[t],
            mag[t],
        )

    ekf_euler = np.array([Quaternion(q).to_angles() * 180 / np.pi for q in Q])
    ekf_euler[:, 0], ekf_euler[:, 1] = ekf_euler[:, 1], ekf_euler[:, 0].copy()
    ekf_euler[:, 2] *= -1

    return ekf_euler


# region
# def optimal_test(true_state, baseline_state, research_state):
#     actual = [QuaternionArray(q) for q in true_state]
#     baseline = [QuaternionArray(q) for q in baseline_state]
#     research = [QuaternionArray(q) for q in research_state]
#     actual = np.array([q.to_angles() * RAD2DEG for q in actual])
#     baseline = np.array([q.to_angles() * RAD2DEG for q in baseline])
#     research = np.array([q.to_angles() * RAD2DEG for q in research])

#     error_baseline = (actual - baseline) ** 2
#     error_research = (actual - research) ** 2

#     C_baseline = np.mean(error_baseline, axis=1)
#     C_research = np.mean(error_research, axis=1)

#     delta_C = C_baseline - C_research
#     mu = np.mean(delta_C, axis=0)
#     std = np.std(delta_C, axis=0)
#     T = mu / std

#     return mu, std, T
# endregion


def plot_sensor_readings(accs, gyrs, mags, num_runs):
    # global acc, gyr, mag
    Ts = 1 / 100
    t = np.arange(0, len(acc) * Ts, Ts)

    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for i in range(3):
        # axs[i].plot(t, gyr[:, i])
        for j in range(num_runs):
            axs[i].plot(t, gyrs[j, :, i])
        axs[i].set_title(["Gyro X", "Gyro Y", "Gyro Z"][i])
        axs[i].set_ylabel("Angle [rad]")
    axs[2].set_xlabel("Time [s]")
    fig.suptitle("Gyroscope Readings")

    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for i in range(3):
        # axs[i].plot(t, acc[:, i])
        for j in range(num_runs):
            axs[i].plot(t, accs[j, :, i])
        axs[i].set_title(["Acc X", "Acc Y", "Acc Z"][i])
        axs[i].set_ylabel("Acceleration [g]")
    axs[2].set_xlabel("Time [s]")
    fig.suptitle("Accelerometer Readings")

    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for i in range(3):
        # axs[i].plot(t, mag[:, i])
        for j in range(num_runs):
            axs[i].plot(t, mags[j, :, i])
        axs[i].set_title(["Mag X", "Mag Y", "Mag Z"][i])
        axs[i].set_ylabel("Magnetic Field [uT]")
    axs[2].set_xlabel("Time [s]")
    fig.suptitle("Magnetometer Readings")


print("Starting")

ground_truths = []
comp_runs = []
ekf_runs = []

accs = []
gyrs = []
mags = []

num_runs = 2
# folder_path = "Collected_Data_23Okt"
folder_path = "Collected_Data"
for i in range(0, num_runs):
    acc = np.load(folder_path + "/A_List_" + str(i + 1) + ".npy")
    gyr = np.load(folder_path + "/G_List_" + str(i + 1) + ".npy")
    mag = np.load(folder_path + "/M_List_" + str(i + 1) + ".npy") / 2
    accs.append(acc)
    gyrs.append(gyr)
    mags.append(mag)

    roll = np.load(folder_path + "/rollList_" + str(i + 1) + ".npy")
    pitch = np.load(folder_path + "/pitchList_" + str(i + 1) + ".npy")
    yaw = np.load(folder_path + "/yawList_" + str(i + 1) + ".npy")
    # print("roll", roll)
    # print("pitch", pitch)
    # print("yaw", yaw)

    ground_truths.append(ground_truth := np.transpose(np.array((roll, pitch, yaw))))
    comp_runs.append(
        comp_res := comp_filter(
            copy.deepcopy(acc),
            copy.deepcopy(gyr),
            copy.deepcopy(mag),
            copy.deepcopy(ground_truth),
            100,
        )
    )
    ekf_runs.append(
        ekf_res := ekf(
            copy.deepcopy(acc),
            copy.deepcopy(gyr),
            copy.deepcopy(mag),
            copy.deepcopy(ground_truth),
            100,
            len(acc),
        )
    )

    plot_euler(100, ground_truth, comp_res, "Comp", "Comp vs Ground Truth")
    plot_euler(100, ground_truth, ekf_res, "EKF", "EKF vs Ground Truth")

accs = np.array(accs)
gyrs = np.array(gyrs)
mags = np.array(mags)


plot_sensor_readings(accs, gyrs, mags, num_runs)
plot_rmse(np.array(ground_truths), np.array(comp_runs), "Complementary Filter", 100)
plot_rmse(np.array(ground_truths), np.array(ekf_runs), "EKF", 100)


# comp_rmse = calculate_rmse(ground_truths, comp_runs)
# print("comp rmse: ", comp_rmse)
# ekf_rmse = calculate_rmse(ground_truths, ekf_runs)
# print("ekf rmse: ", len(ekf_rmse))

# intervals = [(0, 10), (11, 13), (14, 20)]
# results = compute_comparison_metrics(comp_rmse, ekf_rmse, intervals)

plt.show()
print("Done")
