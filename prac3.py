import numpy as np
from ahrs.filters import Complementary
import matplotlib.pyplot as plt
from ahrs.filters import EKF
from ahrs import Quaternion
import copy
import pandas as pd
import matplotlib as mpl


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

    labels = ["Truth", estimate_name]
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


def plot_euler_both(
    fs: int,
    ground_truth: np.ndarray,
    ekf: np.ndarray,
    comp: np.ndarray,
):
    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    for i in range(3):
        time = np.arange(len(ground_truth[:, i])) / fs
        axs[i].plot(time, comp[:, i], label="CF")
        axs[i].plot(time, ekf[:, i], label="EKF")
        axs[i].plot(time, ground_truth[:, i], label="Truth")
        axs[i].set_title(["Roll", "Pitch", "Yaw"][i])
        axs[i].set_ylabel("Angle [deg]")
        axs[i].legend()

    axs[2].set_xlabel("Time [s]")


def plot_euler_both_all(
    fs: int,
    ground_truth: np.ndarray,
    ekf: np.ndarray,
    comp: np.ndarray,
):
    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    num_runs = ground_truth.shape[0]
    colors = mpl.rcParams["axes.prop_cycle"].by_key()["color"][
        :3
    ]  # Default first three colors

    for i in range(3):
        for run in range(num_runs):
            time = np.arange(len(ground_truth[run, 50:, i])) / fs
            axs[i].plot(
                time,
                comp[run, 50:, i],
                color=colors[0],
                label="CF" if run == 0 else "_nolegend_",
            )
            axs[i].plot(
                time,
                ekf[run, 50:, i],
                color=colors[1],
                label="EKF" if run == 0 else "_nolegend_",
            )
            axs[i].plot(
                time,
                ground_truth[run, 50:, i],
                color=colors[2],
                label="Truth" if run == 0 else "_nolegend_",
            )
        axs[i].set_title(["Roll", "Pitch", "Yaw"][i])
        axs[i].set_ylabel("Angle [deg]")
        axs[i].legend()

    axs[2].set_xlabel("Time [s]")


def plot_rmse(ground_truths: np.ndarray, filter: np.ndarray, sampling_frequency):
    n_runs, n_samples, _ = ground_truths.shape
    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    time = np.arange(n_samples) / sampling_frequency
    time = time[50:]

    names = ["CF", "EKF"]
    for axis in range(3):
        for j in range(2):
            error = ((ground_truths[:, :, axis] - filter[j, :, :, axis]) ** 2).mean(
                axis=0
            )[50:]
            rmse = np.sqrt(error)
            axs[axis].plot(time, rmse, label=names[j])
        axs[axis].set_title(["Roll", "Pitch", "Yaw"][axis])
        axs[axis].set_ylabel("RMSE [deg]")
        axs[axis].legend()

    axs[2].set_xlabel("Time [s]")
    fig.suptitle("Mean RMSE of CF vs EKF for " + str(n_runs) + " runs")


def calculate_rmse(ground_truths, estimates):
    rmses = []
    for ground_truth, estimate in zip(ground_truths, estimates):
        rmse = np.sqrt(np.mean((ground_truth - estimate) ** 2, axis=1))
        rmses.append(rmse)
    return np.array(rmses)


def comp_filter(acc, gyr, mag, fs, data_type: str):
    if data_type == "exam":
        acc *= 9.81
        gyr = np.deg2rad(gyr)
        mag[:, :] = 0.1
    elif data_type == "coll":
        acc *= 9.81
        mag *= 1000
    elif data_type == "sim":
        pass

    complementary_filter = Complementary(
        gyr=gyr,
        acc=acc,
        mag=mag,
        frequency=fs,
        gain=0.05,
    )

    comp_filter_estimates = complementary_filter._compute_all()
    comp_euler = np.array(
        [Quaternion(q).to_angles() * 180 / np.pi for q in comp_filter_estimates]
    )
    if data_type == "exam":
        comp_euler[:, 0], comp_euler[:, 1] = -comp_euler[:, 1], -comp_euler[:, 0].copy()
        comp_euler[:, 2] *= -1
        comp_euler[:, 2] -= 138
    elif data_type == "sim":
        comp_euler[:, 2] -= 90
    elif data_type == "coll":
        comp_euler[:, 0], comp_euler[:, 1] = comp_euler[:, 1], comp_euler[:, 0].copy()
        comp_euler[:, 2] *= -1
        comp_euler[:, 2] += 90

    return comp_euler


def ekf(acc, gyr, mag, num_samples, data_type: str):
    noises = [0.3**2, 0.5**2, 0.8**2]  # default
    if data_type == "exam":
        acc *= 9.81
        gyr = np.deg2rad(gyr)
        mag *= 100_000
        noises = [0.65**2, 0.9**2, 30**2]
    elif data_type == "coll":
        acc *= 9.81
        mag *= 1000
    elif data_type == "sim":
        mag /= 1000_000
        noises = [5**2, 0.5**2, 0.5**2]

    ekf = EKF(
        frequency=100,
        frame="NED",
        magnetic_ref=mag[0],
        noises=noises,
    )
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
    if data_type == "exam":
        ekf_euler[:, 0], ekf_euler[:, 1] = -ekf_euler[:, 1], -ekf_euler[:, 0].copy()
        ekf_euler[:, 2] *= -1
    elif data_type == "sim":
        pass
    elif data_type == "coll":
        ekf_euler[:, 0], ekf_euler[:, 1] = ekf_euler[:, 1], ekf_euler[:, 0].copy()
        ekf_euler[:, 2] *= -1

    return ekf_euler


def optimal_test_fixed(actual, baseline, research):
    square_error_baseline = (actual - baseline) ** 2
    squared_error_research = (actual - research) ** 2
    squared_error_delta = square_error_baseline - squared_error_research

    mse_delta = np.mean(squared_error_delta, axis=1)

    mu = np.mean(mse_delta, axis=0)
    std = np.std(mse_delta, axis=0)
    test = mu / std

    return mu, std, test


def display_optimal_test_fixed(intervals, ground_truths, comp_runs, ekf_runs):
    results = []
    Fs = 100
    axes = ["Roll", "Pitch", "Yaw"]
    for interval in intervals:
        true_state = ground_truths[:, interval[0] * Fs : interval[1] * Fs]
        baseline_state = comp_runs[:, interval[0] * Fs : interval[1] * Fs]
        research_state = ekf_runs[:, interval[0] * Fs : interval[1] * Fs]
        result = optimal_test_fixed(true_state, baseline_state, research_state)
        results.append(result)
        print("\n--------------------------------------------")
        print("Time Interval:", interval)
        print("--------------------------------------------")
        data = {
            "Axis": axes,
            "Mean": result[0],
            "Std": result[1],
            "Test Stat": result[2],
        }
        df = pd.DataFrame(data)
        print(df.to_string(index=False))
        print("--------------------------------------------")


def plot_sensor_readings(accs, gyrs, mags, num_runs):
    Ts = 1 / 100
    t = np.arange(0, len(acc) * Ts, Ts)

    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    # gyrs[:, :, 0], gyrs[:, :, 1] = gyrs[:, :, 1], gyrs[:, :, 0].copy()
    # gyrs[:, :, 2] *= -1
    for i in range(3):
        for j in range(num_runs):
            axs[i].plot(t, gyrs[j, :, i])
        axs[i].set_title(["Gyro X", "Gyro Y", "Gyro Z"][i])
        axs[i].set_ylabel("Angle [rad]")
    axs[2].set_xlabel("Time [s]")
    fig.suptitle("Gyroscope Readings")

    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    # accs /= 9.81
    # accs[:, :, 0], accs[:, :, 1] = -accs[:, :, 1], -accs[:, :, 0].copy()
    for i in range(3):
        for j in range(num_runs):
            axs[i].plot(t, accs[j, :, i])
        axs[i].set_title(["Acc X", "Acc Y", "Acc Z"][i])
        axs[i].set_ylabel("Acc [g]")
    axs[2].set_xlabel("Time [s]")
    fig.suptitle("Accelerometer Readings")

    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for i in range(3):
        for j in range(num_runs):
            axs[i].plot(t, mags[j, :, i])
        axs[i].set_title(["Mag X", "Mag Y", "Mag Z"][i])
        axs[i].set_ylabel("Magnetic Field [Gauss]")
    axs[2].set_xlabel("Time [s]")
    fig.suptitle("Magnetometer Readings")


print("Starting")

ground_truths = []
comp_runs = []
ekf_runs = []

accs = []
gyrs = []
mags = []

num_runs = 104
# folder_path = "Collected_Data"
# data_type = "coll"

folder_path = "Exam_Given_Data"
data_type = "exam"

# folder_path = "imu_sim_data"
# data_type = "sim"
for i in range(0, num_runs):
    acc = np.load(folder_path + "/A_List_" + str(i + 1) + ".npy")
    gyr = np.load(folder_path + "/G_List_" + str(i + 1) + ".npy")
    mag = np.load(folder_path + "/M_List_" + str(i + 1) + ".npy")
    accs.append(acc)
    gyrs.append(gyr)
    mags.append(mag)

    roll = np.load(folder_path + "/rollList_" + str(i + 1) + ".npy")
    pitch = np.load(folder_path + "/pitchList_" + str(i + 1) + ".npy")
    yaw = np.load(folder_path + "/yawList_" + str(i + 1) + ".npy")

    ground_truths.append(ground_truth := np.transpose(np.array((roll, pitch, yaw))))
    comp_runs.append(
        comp_res := comp_filter(
            copy.deepcopy(acc),
            copy.deepcopy(gyr),
            copy.deepcopy(mag),
            100,
            data_type,
        )
    )
    ekf_runs.append(
        ekf_res := ekf(
            copy.deepcopy(acc),
            copy.deepcopy(gyr),
            copy.deepcopy(mag),
            len(acc),
            data_type,
        )
    )

for i in range(0, num_runs, 30):
    # plot_euler(100, ground_truths[i], comp_runs[i], "Comp", "Comp vs Ground Truth")
    # plot_euler(100, ground_truths[i], ekf_runs[i], "EKF", "EKF vs Ground Truth")
    plot_euler_both(100, ground_truths[i], ekf_runs[i], comp_runs[i])

plot_euler_both_all(
    100, np.array(ground_truths), np.array(ekf_runs), np.array(comp_runs)
)

accs = np.array(accs)
gyrs = np.array(gyrs)
mags = np.array(mags)

plot_sensor_readings(accs, gyrs, mags, num_runs)
filters = np.array([comp_runs, ekf_runs])
plot_rmse(np.array(ground_truths), filters, 100)

ground_truths = np.array(ground_truths)
comp_runs = np.array(comp_runs)
ekf_runs = np.array(ekf_runs)

intervals = [(0, 8), (8, 15), (15, 20)]
display_optimal_test_fixed(intervals, ground_truths, comp_runs, ekf_runs)

plt.show()
print("Done")
