import numpy as np
import matplotlib.pyplot as plt
from ahrs.filters import Complementary
import imu_simulator

imu = imu_simulator.imuSimulator(sampling_rate=50)
imu.run_sequence()

complementary_filter = Complementary(
    gyr=imu.get_gyr(),
    acc=imu.get_acc(),
    mag=imu.get_mag(),
    frequency=imu.sampling_frequency,
    gain=0.1,
)

print("Complementary Filter")
comp_filter_estimates = complementary_filter._compute_all()


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


plot_euler_angles(
    imu.sampling_frequency,
    np.array(imu.ground_truth_deg),
    quaternions_to_eulers_deg(comp_filter_estimates),
    "Complementary Filter",
    "Complementary Filter vs Ground Truth",
)

# imu.plot_imu_data()
