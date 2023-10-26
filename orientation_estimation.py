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


# def euler_deg_to_quaternion(euler_angles_deg):
#     """Convert euler angles (roll, pitch, yaw) to quaternions"""

#     def quaternion(roll, pitch, yaw):
#         q = np.zeros(4)
#         q[0] = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(
#             roll / 2
#         ) * np.sin(pitch / 2) * np.sin(yaw / 2)
#         q[1] = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(
#             roll / 2
#         ) * np.sin(pitch / 2) * np.sin(yaw / 2)
#         q[2] = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(
#             roll / 2
#         ) * np.cos(pitch / 2) * np.sin(yaw / 2)
#         q[3] = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(
#             roll / 2
#         ) * np.sin(pitch / 2) * np.cos(yaw / 2)
#         return q

#     euler_angles_deg = np.array(euler_angles_deg)
#     quaternions = np.zeros((euler_angles_deg.shape[0], 4))
#     for i in range(euler_angles_deg.shape[0]):
#         quaternions[i, :] = quaternion(
#             euler_angles_deg[i, 0], euler_angles_deg[i, 1], euler_angles_deg[i, 2]
#         )
#     return quaternions


def eulers_deg_to_quaternions(euler_angles_deg):
    """Convert Euler angles in degrees to quaternions.

    Args:
        euler_angles_deg (numpy.ndarray): An Nx3 array where N is the number of samples and the columns are Roll, Pitch, and Yaw in degrees.

    Returns:
        numpy.ndarray: An Nx4 array of quaternions [q_w, q_x, q_y, q_z].
    """

    # Convert degrees to radians
    euler_angles_rad = np.radians(euler_angles_deg)

    # Preallocate the array for quaternions
    quaternions = np.zeros((euler_angles_rad.shape[0], 4))

    roll = euler_angles_rad[:, 0] / 2.0
    pitch = euler_angles_rad[:, 1] / 2.0
    yaw = euler_angles_rad[:, 2] / 2.0

    cy = np.cos(yaw)
    sy = np.sin(yaw)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cr = np.cos(roll)
    sr = np.sin(roll)

    quaternions[:, 0] = cy * cp * cr + sy * sp * sr  # q_w
    quaternions[:, 1] = cy * cp * sr - sy * sp * cr  # q_x
    quaternions[:, 2] = sy * cp * sr + cy * sp * cr  # q_y
    quaternions[:, 3] = sy * cp * cr - cy * sp * sr  # q_z

    return quaternions


def plot_quaternions(
    sampling_frequency: int,
    ground_truth_quat: np.ndarray,
    estimate_quat: np.ndarray,
    title: str,
):
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    labels = ["Ground Truth", "Estimate"]
    components = ["q_w", "q_x", "q_y", "q_z"]

    time = np.arange(len(ground_truth_quat)) / sampling_frequency

    for i in range(4):
        axs[i].plot(time, ground_truth_quat[:, i], label=labels[0])
        axs[i].plot(time, estimate_quat[:, i], label=labels[1])
        axs[i].set_title(components[i])
        axs[i].set_ylabel("Value")
        axs[i].legend()

    axs[3].set_xlabel("Time [s]")
    fig.suptitle(title)
    plt.show()


def plot_euler_angles(
    sampling_frequency: int,
    ground_truth: np.ndarray,
    estimate: np.ndarray,
    estimate_name: str,
    title: str,
):
    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    labels = ["Ground Truth", estimate_name]
    ground_truth = ground_truth[50:]
    estimate = estimate[50:]

    for i in range(3):
        time = np.arange(len(ground_truth[:, i])) / sampling_frequency
        axs[i].plot(time, ground_truth[:, i], label=labels[0])
        axs[i].plot(time, estimate[:, i], label=labels[1])
        axs[i].set_title(["Roll", "Pitch", "Yaw"][i])
        axs[i].set_ylabel("Angle [deg]")
        axs[i].legend()

    axs[2].set_xlabel("Time [s]")

    # fig.suptitle(title)
    plt.show()


def ekf_plot_euler_angles(
    sampling_frequency: int,
    ground_truth: np.ndarray,
    estimate: np.ndarray,
    estimate_name: str,
    title: str,
):
    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    labels = ["Ground Truth", estimate_name]
    ground_truth = ground_truth[50:]
    estimate = estimate[50:]

    for i in range(3):
        time = np.arange(len(ground_truth[:, i])) / sampling_frequency
        axs[i].plot(time, ground_truth[:, i], label=labels[0])
        axs[i].plot(time, estimate[:, i], label=labels[1])
        axs[i].set_title(["Roll", "Pitch", "Yaw"][i])
        axs[i].set_ylabel("Angle [deg]")
        axs[i].legend()

    axs[2].set_xlabel("Time [s]")

    fig.suptitle(title)
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

    # return comp_filter_estimates
    temp = quaternions_to_eulers_deg(comp_filter_estimates)
    temp[:, 2] -= 90
    return temp


def run_ekf(imu: imuSimulator):
    from ahrs.filters import EKF
    from ahrs.common.orientation import acc2q

    ekf = EKF(frequency=50, frame="ENU")
    num_samples = imu.get_num_samples()

    Q = np.zeros((num_samples, 4))  # Allocate array for quaternions
    Q[0] = acc2q(np.array([0, 0, 0]))

    # print(imu.accelorometer_data)
    # imu.accelorometer_data[:][0] = -imu.accelorometer_data[:][0]
    # imu.accelorometer_data[:][1] = -imu.accelorometer_data[:][1]
    # imu.accelorometer_data[:][2] = -imu.accelorometer_data[:][2]

    for t in range(1, num_samples):
        Q[t] = ekf.update(
            Q[t - 1],
            imu.gyroscope_data[t],
            imu.accelorometer_data[t],
            imu.magnetometer_data[t],
        )

    # return np.array(Q)
    # return Q

    temp = quaternions_to_eulers_deg(Q)
    temp[:, 2] += 90
    return temp


def plot_comp(imu: imuSimulator):
    comp_euler = run_comp_filter(imu)
    plot_euler_angles(
        imu.sampling_frequency,
        np.array(imu.ground_truth_deg),
        comp_euler,
        "Complementary Filter",
        "Complementary Filter vs Ground Truth",
    )

    # plot_quaternions(
    #     imu.sampling_frequency,
    #     eulers_deg_to_quaternions(imu.ground_truth_deg),
    #     comp_euler,
    #     "EKF vs Ground Truth",
    # )


def plot_ekf(imu: imuSimulator):
    ekf_euler = run_ekf(imu)
    ekf_plot_euler_angles(
        imu.sampling_frequency,
        np.array(imu.ground_truth_deg),
        ekf_euler,
        "EKF",
        "EKF vs Ground Truth",
    )

    # plot_quaternions(
    #     imu.sampling_frequency,
    #     eulers_deg_to_quaternions(imu.ground_truth_deg),
    #     ekf_euler,
    #     "EKF vs Ground Truth",
    # )


def monte_carlo(imu: imuSimulator):
    num_runs = 20

    comp_runs = []
    ekf_runs = []
    ground_truth_runs = []

    for run in range(num_runs):
        imu.dynamic_seq_1()
        # imu.dynamic_seq_2()
        # imu.static_seq()

        comp_runs.append(run_comp_filter(imu))
        ekf_runs.append(run_ekf(imu))
        ground_truth_runs.append(np.array(imu.ground_truth_deg))

    plot_rmse_new(ground_truth_runs, comp_runs, ekf_runs, imu.sampling_frequency)


def plot_rmse_report(ground_truths, filter, filter_name: str, sampling_frequency):
    n_runs, n_samples, _ = np.array(ground_truths[0]).shape
    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    time = np.arange(n_samples) / sampling_frequency
    time = time[50:]

    seq_names = ["Static Sequence", "Dynamic Sequence 1", "Dynamic Sequence 2"]
    for seq in range(3):
        for axis in range(3):
            error = (
                (
                    np.array(ground_truths[seq])[:, :, axis]
                    - np.array(filter[seq])[:, :, axis]
                )
                ** 2
            ).mean(axis=0)[50:]

            rmse = np.sqrt(error)

            axs[axis].plot(time, rmse, label=seq_names[seq])
            axs[axis].set_title(["Roll", "Pitch", "Yaw"][axis])
            axs[axis].set_ylabel("RMSE [deg]")
            axs[axis].legend()
            # axs[axis].set_ylim([0, 6])

    axs[2].set_xlabel("Time [s]")
    # fig.suptitle("Mean RMSE of " + filter_name + " for " + str(n_runs) + " runs")
    plt.show()


def monte_carlo_report_comp():
    num_runs = 5

    comp_runs = [[], [], []]
    ground_truth_runs = [[], [], []]

    imu1 = imuSimulator(sampling_frequency=50)
    imu2 = imuSimulator(sampling_frequency=50)
    imu3 = imuSimulator(sampling_frequency=50)

    for run in range(num_runs):
        imu1.static_seq()
        imu2.dynamic_seq_1()
        imu3.dynamic_seq_2()

        comp_runs[0].append(run_comp_filter(imu1))
        ground_truth_runs[0].append(np.array(imu1.ground_truth_deg))
        comp_runs[1].append(run_comp_filter(imu2))
        ground_truth_runs[1].append(np.array(imu2.ground_truth_deg))
        comp_runs[2].append(run_comp_filter(imu3))
        ground_truth_runs[2].append(np.array(imu3.ground_truth_deg))

    plot_rmse_report(
        ground_truth_runs,
        comp_runs,
        "Complementary Filter",
        imu1.sampling_frequency,
    )


def monte_carlo_report_ekf():
    num_runs = 20

    ekf_runs = [[], [], []]
    ground_truth_runs = [[], [], []]

    imu1 = imuSimulator(sampling_frequency=50)
    imu2 = imuSimulator(sampling_frequency=50)
    imu3 = imuSimulator(sampling_frequency=50)

    for run in range(num_runs):
        imu1.static_seq()
        imu2.dynamic_seq_1()
        imu3.dynamic_seq_2()

        ekf_runs[0].append(run_ekf(imu1))
        ground_truth_runs[0].append(np.array(imu1.ground_truth_deg))
        ekf_runs[1].append(run_ekf(imu2))
        ground_truth_runs[1].append(np.array(imu2.ground_truth_deg))
        ekf_runs[2].append(run_ekf(imu3))
        ground_truth_runs[2].append(np.array(imu3.ground_truth_deg))

    plot_rmse_report(
        ground_truth_runs,
        ekf_runs,
        "EKF",
        imu1.sampling_frequency,
    )


# monte_carlo_report_comp()
# monte_carlo_report_ekf()

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
# REMEBER TO CHANGE SEQUENCE IN MONTE CARLO #
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #

# print("Dynamic Sequence 1")
# dyn_seq_1_imu = imuSimulator(sampling_frequency=50)
# dyn_seq_1_imu.dynamic_seq_1()
# plot_comp(dyn_seq_1_imu)
# plot_ekf(dyn_seq_1_imu)
# plot_models(dyn_seq_1_imu)
# monte_carlo(dyn_seq_1_imu)

# print("Dynamic Sequence 2")
# dyn_seq_2_imu = imuSimulator(sampling_frequency=50)
# dyn_seq_2_imu.dynamic_seq_2()
# plot_ekf(dyn_seq_2_imu)
# plot_models(dyn_seq_2_imu)
# monte_carlo(dyn_seq_2_imu)

# print("Static Sequence")
# static_seq_imu = imuSimulator(sampling_frequency=50)
# static_seq_imu.static_seq()
# plot_models(static_seq_imu)
# monte_carlo(static_seq_imu)
