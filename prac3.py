import re
import numpy as np
from ahrs.filters import Complementary
import orientation_estimation as oe
import matplotlib.pyplot as plt
from ahrs.filters import EKF
from ahrs.common.orientation import acc2q


def plot_euler(
    fs: int,
    ground_truth: np.ndarray,
    estimate: np.ndarray,
    estimate_name: str,
    title: str,
    drop: int = 0,
):
    """
    ground_truth: ground truth euler angles (3, N)
    estimate: estimated euler angles (N, 3)
    """
    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    labels = ["Ground Truth", estimate_name]
    ground_truth = ground_truth[:, drop:]
    estimate = estimate[drop:]

    for i in range(3):
        time = np.arange(len(ground_truth[i, :])) / fs
        axs[i].plot(time, ground_truth[i, :], label=labels[0])
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
    n_runs, _, n_samples = ground_truths.shape
    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    time = np.arange(n_samples) / sampling_frequency
    time = time[50:]

    # seq_names = ["Static Sequence", "Dynamic Sequence 1", "Dynamic Sequence 2"]
    # for seq in range(3):
    for axis in range(3):
        error = ((ground_truths[:, axis, :] - filter[:, :, axis]) ** 2).mean(axis=0)[
            50:
        ]

        rmse = np.sqrt(error)

        # axs[axis].plot(time, rmse, label=seq_names[seq])
        axs[axis].set_title(["Roll", "Pitch", "Yaw"][axis])
        axs[axis].set_ylabel("RMSE [deg]")
        axs[axis].legend()
        # axs[axis].set_ylim([0, 6])

    axs[2].set_xlabel("Time [s]")
    fig.suptitle("Mean RMSE of " + filter_name + " for " + str(n_runs) + " runs")
    plt.show()


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
    #     "Complementary Filter vs Ground Truth",
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

num_runs = 2
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
    comp_runs.append(comp_filter(acc, gyr, mag, ground_truth, 100))
    ekf_runs.append(ekf(acc, gyr, mag, ground_truth, 100, len(acc)))

plot_rmse(np.array(ground_truths), np.array(comp_runs), "Complementary Filter", 100)

print("Done")
