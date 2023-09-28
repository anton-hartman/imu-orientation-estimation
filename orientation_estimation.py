import numpy as np
import matplotlib.pyplot as plt
import imu_simulator


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


def plot_rmse(ground_truth_list, estimate_list, estimate_name_list):
    """Plot RMSE between ground truth and estimate"""
    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    for i in range(3):
        rmse = []
        for j in range(len(ground_truth_list)):
            rmse.append(
                np.sqrt(np.mean((ground_truth_list[j][i] - estimate_list[j][i]) ** 2))
            )
        axs[i].plot(rmse, label=estimate_name_list)
        axs[i].set_title(["Roll", "Pitch", "Yaw"][i])
        axs[i].set_ylabel("RMSE [deg]")

    axs[2].set_xlabel("Time [s]")

    fig.suptitle("RMSE between Estimates and Ground Truth")
    plt.legend(estimate_name_list)
    plt.show()


def ekf_oud():
    from ahrs.filters import EKF

    # Initialize EKF
    # ekf = EKF()
    ekf = EKF(Q=np.array([1.0, 0.0, 0.0, 0.0]))  # Initializing with a unit quaternion

    # Assume acc, gyro, and mag are your raw IMU data
    # acc, gyro, and mag should be numpy arrays with shape (n_samples, 3)
    # For example:
    # acc = np.array([[0.1, 0.2, 9.8], [0.2, 0.1, 9.7], ...])
    # gyro = np.array([[0.01, -0.02, 0.03], [-0.01, 0.02, -0.03], ...])
    # mag = np.array([[20.0, 30.0, 40.0], [21.0, 29.0, 41.0], ...])

    # n_samples = imu.get_acc.shape[0]
    n_samples = imu.total_time * imu.sampling_frequency
    ekf_quaternions = np.zeros((n_samples, 4))

    for i in range(n_samples):
        ekf.update(imu.get_acc()[i], imu.get_gyr()[i], imu.get_mag()[i])
        norm = np.linalg.norm(ekf.Q)
        if norm != 0:
            ekf.Q /= norm
        ekf_quaternions[i] = ekf.Q

    # Now, quaternions contains the estimated orientation in quaternion form for each sample.

    plot_euler_angles(
        imu.sampling_frequency,
        np.array(imu.ground_truth_deg),
        quaternions_to_eulers_deg(ekf_quaternions),
        "EKF",
        "EKF vs Ground Truth",
    )


def run_comp_filter():
    from ahrs.filters import Complementary

    complementary_filter = Complementary(
        gyr=imu.get_gyr(),
        acc=imu.get_acc(),
        mag=imu.get_mag(),
        frequency=imu.sampling_frequency,
        gain=0.1,
    )

    print("Complementary Filter")
    comp_filter_estimates = complementary_filter._compute_all()

    plot_euler_angles(
        imu.sampling_frequency,
        np.array(imu.ground_truth_deg),
        quaternions_to_eulers_deg(comp_filter_estimates),
        "Complementary Filter",
        "Complementary Filter vs Ground Truth",
    )

    plot_rmse(
        np.array(imu.ground_truth_deg),
        quaternions_to_eulers_deg(comp_filter_estimates),
        "Complementary Filter",
    )


def run_ekf():
    from ahrs.filters import EKF
    from ahrs.common.orientation import acc2q

    ekf = EKF()
    num_samples = imu.get_num_samples()

    Q = np.zeros((num_samples, 4))  # Allocate array for quaternions
    Q[0] = acc2q(imu.accelorometer_data[0])  # First sample of tri-axial accelerometer

    for t in range(1, num_samples):
        Q[t] = ekf.update(
            Q[t - 1],
            imu.gyroscope_data[t],
            imu.accelorometer_data[t],
            imu.magnetometer_data[t],
        )

    plot_euler_angles(
        imu.sampling_frequency,
        np.array(imu.ground_truth_deg),
        quaternions_to_eulers_deg(Q),
        "EKF",
        "EKF vs Ground Truth",
    )


imu = imu_simulator.imuSimulator(sampling_rate=50)
imu.run_sequence()

run_comp_filter()
run_ekf()
