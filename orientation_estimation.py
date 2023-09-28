import numpy as np
import matplotlib.pyplot as plt
from imu_simulator import imuSimulator


def quaternions_to_eulers_deg(quaternions_rad):
    """Convert quaternions to euler angles (roll, pitch, yaw)"""

    def deg(rad):
        return rad * 180 / np.pi

    def quaternion_to_euler(q):
        roll = np.arctan2(
            2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)
        )
        pitch = np.arcsin(2 * (q[0] * q[2] - q[3] * q[1]))
        yaw = np.arctan2(
            2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)
        )
        return np.array([deg(roll), deg(pitch), deg(yaw)])

    euler_angles = np.zeros((quaternions_rad.shape[0], 3))
    for i in range(quaternions_rad.shape[0]):
        euler_angles[i, :] = quaternion_to_euler(quaternions_rad[i, :])
    return euler_angles


def plot_euler_angles(
    sampling_frequency: int,
    ground_truth: np.ndarray,
    estimate: np.ndarray,
    estimate_name: str,
    title: str,
):
    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    labels = ["Ground Truth", estimate_name]
    colors = ["C0", "C1"]

    for i in range(3):
        time = np.arange(len(ground_truth[:, i])) / sampling_frequency
        axs[i].plot(time, ground_truth[:, i], label=labels[0], color=colors[0])
        axs[i].plot(time, estimate[:, i], label=labels[1], color=colors[1])
        axs[i].set_title(["Roll", "Pitch", "Yaw"][i])
        axs[i].set_ylabel("Angle [deg]")

    axs[2].set_xlabel("Time [s]")

    fig.suptitle(title)
    plt.legend()
    plt.show()


def plot_rmse(ground_truth, comp_filter, ekf, sampling_frequency):
    """Plot RMSE between ground truth and two estimates"""

    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    for i in range(3):
        rmse1 = []
        rmse2 = []
        time = np.arange(len(ground_truth[:, i])) / sampling_frequency

        for j in range(len(ground_truth)):
            rmse1.append(
                np.sqrt(np.mean((ground_truth[j][i] - comp_filter[j][i]) ** 2))
            )
            rmse2.append(np.sqrt(np.mean((ground_truth[j][i] - ekf[j][i]) ** 2)))

        axs[i].plot(time, rmse1, label="Complementary Filter")
        axs[i].plot(time, rmse2, label="EKF")
        axs[i].set_title(["Roll", "Pitch", "Yaw"][i])
        axs[i].set_ylabel("RMSE [deg]")
        axs[i].legend(loc="upper right")

    axs[2].set_xlabel("Time [s]")
    fig.suptitle("RMSE")
    plt.show()


def plot_rmse_new0(ground_truths, comp_filters, ekfs, sampling_frequency):
    """Plot RMSE between ground truth and two estimates"""

    n_runs = len(ground_truths)
    n_samples = len(ground_truths[0])

    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    for axis in range(3):
        rmse1 = np.zeros(n_samples)
        rmse2 = np.zeros(n_samples)

        for t in range(n_samples):
            error1 = 0
            error2 = 0
            for run in range(n_runs):
                error1 = (ground_truths[run][t, axis] - comp_filters[run][t, axis]) ** 2
                error2 = (ground_truths[run][t, axis] - ekfs[run][t, axis]) ** 2

            rmse1[t] = np.sqrt(error1 / n_runs)
            rmse2[t] = np.sqrt(error2 / n_runs)

        time = np.arange(n_samples) / sampling_frequency
        axs[axis].plot(time, rmse1, label="Complementary Filter")
        axs[axis].plot(time, rmse2, label="EKF")
        axs[axis].set_title(["Roll", "Pitch", "Yaw"][axis])
        axs[axis].set_ylabel("RMSE [deg]")
        axs[axis].legend(loc="upper right")

    axs[2].set_xlabel("Time [s]")
    fig.suptitle("RMSE")
    plt.show()


def plot_rmse_new(ground_truths, comp_filters, ekfs, sampling_frequency):
    """Plot RMSE between ground truth and two estimates"""

    print(np.array(ground_truths))
    n_runs, n_samples, _ = np.array(ground_truths).shape

    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    time = np.arange(n_samples) / sampling_frequency

    for axis in range(3):
        error1 = (
            (np.array(ground_truths)[:, :, axis] - np.array(comp_filters)[:, :, axis])
            ** 2
        ).mean(axis=0)
        error2 = (
            (np.array(ground_truths)[:, :, axis] - np.array(ekfs)[:, :, axis]) ** 2
        ).mean(axis=0)

        rmse1 = np.sqrt(error1)
        rmse2 = np.sqrt(error2)

        axs[axis].plot(time, rmse1, label="Complementary Filter")
        axs[axis].plot(time, rmse2, label="EKF")
        axs[axis].set_title(["Roll", "Pitch", "Yaw"][axis])
        axs[axis].set_ylabel("RMSE [deg]")
        axs[axis].legend(loc="upper right")

    axs[2].set_xlabel("Time [s]")
    fig.suptitle("Mean RMSE for " + str(n_runs) + " runs")
    plt.show()


def plot_average_rmse(ground_truth_runs, comp_filter_runs, ekf_runs):
    """Plot average RMSE between ground truth and two estimates over multiple runs"""

    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    for i in range(3):
        for j in range(len(ground_truth_runs)):
            print("Ground Truth Shape:", ground_truth_runs[j].shape)
            print("EKF Runs Shape:", ekf_runs[j].shape)

        avg_rmse1 = np.mean(
            [
                np.sqrt(
                    np.mean(
                        (ground_truth_runs[j][:, i] - comp_filter_runs[j][:, i]) ** 2
                    )
                )
                for j in range(len(ground_truth_runs))
            ]
        )
        avg_rmse2 = np.mean(
            [
                np.sqrt(np.mean((ground_truth_runs[j][:, i] - ekf_runs[j][:, i]) ** 2))
                for j in range(len(ground_truth_runs))
            ]
        )

        axs[i].bar(["Complementary Filter", "EKF"], [avg_rmse1, avg_rmse2])
        axs[i].set_title(["Roll", "Pitch", "Yaw"][i])
        axs[i].set_ylabel("Average RMSE [deg]")

    axs[2].set_xlabel("Filter Type")
    fig.suptitle("Average RMSE over Multiple Runs")
    plt.show()


def plot_individual_rmse(ground_truth_runs, comp_filter_runs, ekf_runs):
    """Plot individual RMSEs between ground truth and two estimates over multiple runs"""

    num_runs = len(ground_truth_runs)
    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    for i in range(3):
        for j in range(num_runs):
            rmse1 = np.sqrt(
                np.mean((ground_truth_runs[j][:, i] - comp_filter_runs[j][:, i]) ** 2)
            )
            rmse2 = np.sqrt(
                np.mean((ground_truth_runs[j][:, i] - ekf_runs[j][:, i]) ** 2)
            )

            axs[i].plot(
                j,
                rmse1,
                "o",
                label=f"Run {j+1} - Complementary Filter" if i == 0 else "",
            )
            axs[i].plot(j, rmse2, "s", label=f"Run {j+1} - EKF" if i == 0 else "")

        axs[i].set_title(["Roll", "Pitch", "Yaw"][i])
        axs[i].set_ylabel("RMSE [deg]")

    axs[2].set_xlabel("Run Number")
    if num_runs > 1:  # Show legend only if there is more than one run
        axs[0].legend(loc="upper left", bbox_to_anchor=(1, 1))
    fig.suptitle("Individual RMSEs over Multiple Runs")
    plt.tight_layout(
        rect=[0, 0, 0.75, 1]
    )  # To ensure the suptitle and legend fit into the figure
    plt.show()


def run_comp_filter(imu: imuSimulator):
    from ahrs.filters import Complementary

    complementary_filter = Complementary(
        gyr=imu.get_gyr(),
        acc=imu.get_acc(),
        mag=imu.get_mag(),
        frequency=imu.sampling_frequency,
        gain=0.1,
    )

    comp_filter_estimates = complementary_filter._compute_all()

    # comp_filter_estimates = []
    # comp_filter_estimates.append(complementary_filter.am_estimation(imu.get_acc()[0]))
    # for i in range(1, len(imu.get_gyr())):
    #     comp_filter_estimates.append(
    #         complementary_filter.update(
    #             comp_filter_estimates[i - 1],
    #             imu.get_gyr()[i],
    #             imu.get_acc()[i],
    #             # imu.get_mag()[i],
    #         )
    #     )
    # comp_filter_estimates = np.array(comp_filter_estimates)

    # plot_euler_angles(
    #     imu.sampling_frequency,
    #     np.array(imu.ground_truth_deg),
    #     quaternions_to_eulers_deg(comp_filter_estimates),
    #     "Complementary Filter",
    #     "Complementary Filter vs Ground Truth",
    # )

    temp = quaternions_to_eulers_deg(comp_filter_estimates)
    temp[:, 2] -= 90
    return temp


def run_ekf(imu: imuSimulator):
    from ahrs.filters import EKF
    from ahrs.common.orientation import acc2q

    ekf = EKF(frequency=50, frame="ENU")
    num_samples = imu.get_num_samples()

    Q = np.zeros((num_samples, 4))  # Allocate array for quaternions
    # Q[0] = acc2q(imu.accelorometer_data[0])  # First sample of tri-axial accelerometer
    Q[0] = acc2q(np.array([0, 0, 0]))

    for t in range(1, num_samples):
        Q[t] = ekf.update(
            Q[t - 1],
            imu.gyroscope_data[t],
            imu.accelorometer_data[t],
            imu.magnetometer_data[t],
        )

    # plot_euler_angles(
    #     imu.sampling_frequency,
    #     np.array(imu.ground_truth_deg),
    #     quaternions_to_eulers_deg(Q),
    #     "EKF",
    #     "EKF vs Ground Truth",
    # )

    temp = quaternions_to_eulers_deg(Q)
    temp[:, 2] += 90
    return temp


def plot_models(imu: imuSimulator):
    comp_euler = run_comp_filter(imu)
    # comp_euler[:, 2] -= 90
    # comp_euler[:, 2] = np.maximum(comp_euler[:, 2], -70)

    plot_euler_angles(
        imu.sampling_frequency,
        np.array(imu.ground_truth_deg),
        comp_euler,
        "Complementary Filter",
        "Complementary Filter vs Ground Truth",
    )

    ekf_euler = run_ekf(imu)
    # ekf_euler[:, 2] += 90
    # ekf_euler[:, 2] = np.minimum(ekf_euler[:, 2], 100)

    plot_euler_angles(
        imu.sampling_frequency,
        np.array(imu.ground_truth_deg),
        ekf_euler,
        "EKF",
        "EKF vs Ground Truth",
    )

    # plot_rmse(
    #     np.array(imu.ground_truth_deg), comp_euler, ekf_euler, imu.sampling_frequency
    # )


def monte_carlo(imu: imuSimulator):
    # Number of Monte Carlo runs
    num_runs = 20

    comp_runs = []
    ekf_runs = []
    ground_truth_runs = []

    # Perform 20 Monte Carlo runs
    for run in range(num_runs):
        # imu.dynamic_seq_1()
        imu.dynamic_seq_2()
        # imu.static_seq()

        comp_runs.append(run_comp_filter(imu))
        ekf_runs.append(run_ekf(imu))
        ground_truth_runs.append(np.array(imu.ground_truth_deg))

    plot_rmse_new(ground_truth_runs, comp_runs, ekf_runs, imu.sampling_frequency)


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
# REMEBER TO CHANGE SEQUENCE IN MONTE CARLO #
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

# print("Dynamic Sequence 1")
# dyn_seq_1_imu = imuSimulator(sampling_frequency=50)
# dyn_seq_1_imu.dynamic_seq_1()
# plot_models(dyn_seq_1_imu)
# monte_carlo(dyn_seq_1_imu)

print("Dynamic Sequence 2")
dyn_seq_2_imu = imuSimulator(sampling_frequency=50)
dyn_seq_2_imu.dynamic_seq_2()
plot_models(dyn_seq_2_imu)
monte_carlo(dyn_seq_2_imu)

# print("Static Sequence")
# static_seq_imu = imuSimulator(sampling_frequency=50)
# static_seq_imu.static_seq()
# plot_models(static_seq_imu)
# monte_carlo(static_seq_imu)
