import numpy as np
import matplotlib.pyplot as plt


class imuSimulator:
    sampling_frequency: int  # hz
    total_time: int = 0  # sec

    # B: float  # magnetic field strength
    # inclination: float  # inclination angle

    _roll_rad: float = 0
    _pitch_rad: float = 0
    _yaw_rad: float = 0

    ground_truth_deg = []
    accelorometer_data = []  # m/s^2
    gyroscope_data = []  # rad/s
    magnetometer_data = []  # mT

    def __init__(self, sampling_frequency=50):
        self.sampling_frequency = sampling_frequency

    def reset(self):
        self.total_time = 0
        self._roll_rad = 0
        self._pitch_rad = 0
        self._yaw_rad = 0
        self.ground_truth_deg = []
        self.accelorometer_data = []
        self.gyroscope_data = []
        self.magnetometer_data = []

    def _rad(self, deg):
        return deg * np.pi / 180

    def _deg(self, rad):
        return rad * 180 / np.pi

    def _move_to_orientation(
        self, roll_deg: int, pitch_deg: int, yaw_deg: int, period_sec: int
    ):
        def _calc_acc_delta():
            R_x = [
                [1, 0, 0],
                [0, np.cos(self._roll_rad), np.sin(self._roll_rad)],
                [0, -1 * np.sin(self._roll_rad), np.cos(self._roll_rad)],
            ]
            R_y = [
                [np.cos(self._pitch_rad), 0, -1 * np.sin(self._pitch_rad)],
                [0, 1, 0],
                [np.sin(self._pitch_rad), 0, np.cos(self._pitch_rad)],
            ]
            R_z = [
                [np.cos(self._yaw_rad), np.sin(self._yaw_rad), 0],
                [-1 * np.sin(self._yaw_rad), np.cos(self._yaw_rad), 0],
                [0, 0, 1],
            ]

            # R_x = [
            #     [1, 0, 0],
            #     [0, np.cos(self._roll_rad), -np.sin(self._roll_rad)],
            #     [0, np.sin(self._roll_rad), np.cos(self._roll_rad)],
            # ]

            # R_y = [
            #     [np.cos(self._pitch_rad), 0, np.sin(self._pitch_rad)],
            #     [0, 1, 0],
            #     [-np.sin(self._pitch_rad), 0, np.cos(self._pitch_rad)],
            # ]

            # R_z = [
            #     [np.cos(self._yaw_rad), -np.sin(self._yaw_rad), 0],
            #     [np.sin(self._yaw_rad), np.cos(self._yaw_rad), 0],
            #     [0, 0, 1],
            # ]

            R_x_mat = np.matrix(R_x)
            R_y_mat = np.matrix(R_y)
            R_z_mat = np.matrix(R_z)

            R_xy = np.matmul(R_x_mat, R_y_mat)
            R_xyz = np.matmul(R_xy, R_z_mat)

            acc_ref_point = np.matrix([0, 0, -1]).transpose()
            acc_orientation = -9.8 * np.matmul(R_xyz, acc_ref_point)

            acc_bias = [0, 0, 0]
            new_data = [
                acc_orientation.tolist()[0][0] + acc_bias[0],
                acc_orientation.tolist()[1][0] + acc_bias[1],
                acc_orientation.tolist()[2][0] + acc_bias[2],
            ]
            A_noise = 0.7 * np.random.normal(0, 1, 3)
            new_data = new_data + A_noise

            self.accelorometer_data.append(new_data)

        def _calc_gyro_delta(roll_step_rad, pitch_step_rad, yaw_step_rad):
            roll_rate = roll_step_rad * self.sampling_frequency
            pitch_rate = pitch_step_rad * self.sampling_frequency
            yaw_rate = yaw_step_rad * self.sampling_frequency

            gyro_bias = [0, 0, 0]
            new_data = [
                self._deg(roll_rate) + gyro_bias[0],
                self._deg(pitch_rate) + gyro_bias[1],
                self._deg(yaw_rate) + gyro_bias[2],
            ]
            G_noise = np.random.normal(0, 1, 3)
            new_data = new_data + G_noise
            self.gyroscope_data.append(
                np.array([self._rad(deg_data) for deg_data in new_data])
            )

        def _calc_mag_delta_old():
            R_x = [
                [1, 0, 0],
                [0, np.cos(self._roll_rad), np.sin(self._roll_rad)],
                [0, -1 * np.sin(self._roll_rad), np.cos(self._roll_rad)],
            ]
            R_y = [
                [np.cos(self._pitch_rad), 0, -1 * np.sin(self._pitch_rad)],
                [0, 1, 0],
                [np.sin(self._pitch_rad), 0, np.cos(self._pitch_rad)],
            ]
            R_z = [
                [np.cos(self._yaw_rad), np.sin(self._yaw_rad), 0],
                [-1 * np.sin(self._yaw_rad), np.cos(self._yaw_rad), 0],
                [0, 0, 1],
            ]

            # R_x = [
            #     [1, 0, 0],
            #     [0, np.cos(self._roll_rad), -np.sin(self._roll_rad)],
            #     [0, np.sin(self._roll_rad), np.cos(self._roll_rad)],
            # ]

            # R_y = [
            #     [np.cos(self._pitch_rad), 0, np.sin(self._pitch_rad)],
            #     [0, 1, 0],
            #     [-np.sin(self._pitch_rad), 0, np.cos(self._pitch_rad)],
            # ]

            # R_z = [
            #     [np.cos(self._yaw_rad), -np.sin(self._yaw_rad), 0],
            #     [np.sin(self._yaw_rad), np.cos(self._yaw_rad), 0],
            #     [0, 0, 1],
            # ]

            R_x_mat = np.matrix(R_x)
            R_y_mat = np.matrix(R_y)
            R_z_mat = np.matrix(R_z)

            R_xy = np.matmul(R_x_mat, R_y_mat)
            R_xyz = np.matmul(R_xy, R_z_mat)

            B = 28310.9e-9
            # B = 28.3
            inclination = self._rad(61.49)
            # inclination = np.pi / 2
            # inclination = self._rad(80)

            mag_ref_point = (
                B * np.matrix([np.cos(inclination), 0, np.sin(inclination)]).transpose()
            )
            mag_orientation = -1e-6 * np.matmul(R_xyz, mag_ref_point)

            mag_bias = [0, 0, 0]
            new_data = [
                mag_orientation.tolist()[0][0] + mag_bias[0],
                mag_orientation.tolist()[1][0] + mag_bias[1],
                mag_orientation.tolist()[2][0] + mag_bias[2],
            ]
            # M_noise = 1e-6 * np.random.normal(0, 0.01, 3)
            M_noise = 1e-6 * np.random.normal(0, 0.01e-6, 3)
            new_data = new_data + M_noise
            # new_data = np.multiply(1 / 1000, new_data)

            self.magnetometer_data.append(new_data)
            # self.magnetometer_data.append(np.array(new_data) * 1_000_000)

        # def _calc_mag_delta_new():
        #     # Construct Rotation Matrices
        #     R_x = np.array(
        #         [
        #             [1, 0, 0],
        #             [0, np.cos(self._roll_rad), -np.sin(self._roll_rad)],
        #             [0, np.sin(self._roll_rad), np.cos(self._roll_rad)],
        #         ]
        #     )

        #     R_y = np.array(
        #         [
        #             [np.cos(self._pitch_rad), 0, np.sin(self._pitch_rad)],
        #             [0, 1, 0],
        #             [-np.sin(self._pitch_rad), 0, np.cos(self._pitch_rad)],
        #         ]
        #     )

        #     R_z = np.array(
        #         [
        #             [np.cos(self._yaw_rad), -np.sin(self._yaw_rad), 0],
        #             [np.sin(self._yaw_rad), np.cos(self._yaw_rad), 0],
        #             [0, 0, 1],
        #         ]
        #     )

        #     # Combine Rotation Matrices
        #     R = np.dot(np.dot(R_z, R_y), R_x)

        #     # Define the reference magnetic field in the global coordinate system
        #     mag_ref_point = self.B * np.array(
        #         [np.cos(self.inclination), 0, np.sin(self.inclination)]
        #     )

        #     # Transform this reference field to the local coordinate system using the rotation matrix
        #     mag_orientation = np.dot(R, mag_ref_point)

        #     # Add noise and bias
        #     mag_bias = [0, 0, 0]
        #     M_noise = 1e-6 * np.random.normal(0, 1, 3)
        #     new_data = mag_orientation + mag_bias + M_noise

        #     self.magnetometer_data.append(new_data)

        def _update_ground_truth():
            self.ground_truth_deg.append(
                [
                    self._deg(self._roll_rad),
                    self._deg(self._pitch_rad),
                    self._deg(self._yaw_rad),
                    # self._deg(-self._roll_rad),
                    # self._deg(-self._pitch_rad),
                    # self._deg(-self._yaw_rad),
                ]
            )

        self.total_time += period_sec
        number_of_samples: int = self.sampling_frequency * period_sec
        roll_step_rad = self._rad(roll_deg) / number_of_samples
        pitch_step_rad = self._rad(pitch_deg) / number_of_samples
        yaw_step_rad = self._rad(yaw_deg) / number_of_samples
        for _ in range(0, number_of_samples, 1):
            self._roll_rad += roll_step_rad
            self._pitch_rad += pitch_step_rad
            self._yaw_rad += yaw_step_rad

            _calc_acc_delta()
            _calc_gyro_delta(roll_step_rad, pitch_step_rad, yaw_step_rad)
            _calc_mag_delta_old()
            _update_ground_truth()

    def _convert_list_format(self, input_array):
        if not isinstance(input_array, list):
            raise ValueError("Input must be a list of arrays")

        temp_list = []
        for arr in input_array:
            if not isinstance(arr, np.ndarray):
                raise ValueError("Each element in the input list must be a NumPy array")
            if arr.shape != (3,):
                raise ValueError(
                    "Each array in the input list must have a shape of (3,)"
                )
            temp_list.append(arr)

        return np.array(temp_list)

    def get_acc(self):
        return self._convert_list_format(self.accelorometer_data)

    def get_gyr(self):
        return self._convert_list_format(self.gyroscope_data)

    def get_mag(self):
        return self._convert_list_format(self.magnetometer_data)

    def get_num_samples(self):
        return self.total_time * self.sampling_frequency

    def run_sequence(self):
        self.reset()
        self._move_to_orientation(roll_deg=0, pitch_deg=0, yaw_deg=0, period_sec=5)
        self._move_to_orientation(0, 70, 0, 5)
        self._move_to_orientation(0, -70, 0, 5)
        self._move_to_orientation(33, 0, 0, 6)
        self._move_to_orientation(-33, 0, 0, 6)
        self._move_to_orientation(0, 0, 85, 5)
        self._move_to_orientation(0, 0, -85, 5)
        self._move_to_orientation(45, 32, 0, 5)
        self._move_to_orientation(0, 0, 0, 5)

    def static_seq(self):
        self.reset()
        # self._move_to_orientation(roll_deg=0, pitch_deg=0, yaw_deg=0, period_sec=30)
        self._move_to_orientation(roll_deg=0, pitch_deg=0, yaw_deg=0, period_sec=50)

    def dynamic_seq_1(self):
        self.reset()
        self._move_to_orientation(roll_deg=0, pitch_deg=0, yaw_deg=0, period_sec=5)
        self._move_to_orientation(0, 45, 0, 5)
        self._move_to_orientation(0, -45, 0, 5)
        self._move_to_orientation(0, 0, 0, 5)
        self._move_to_orientation(50, 0, 0, 5)
        self._move_to_orientation(-50, 0, 0, 5)
        self._move_to_orientation(0, 0, 0, 5)
        self._move_to_orientation(0, 0, 80, 5)
        self._move_to_orientation(0, 0, -80, 5)
        self._move_to_orientation(0, 0, 0, 5)

    def dynamic_seq_2(self):
        self.reset()
        self._move_to_orientation(roll_deg=0, pitch_deg=0, yaw_deg=0, period_sec=5)
        self._move_to_orientation(25, 45, 0, 5)
        self._move_to_orientation(-25, -45, 0, 5)
        self._move_to_orientation(0, 0, 0, 5)
        self._move_to_orientation(-50, 0, 15, 5)
        self._move_to_orientation(50, 0, -15, 5)
        self._move_to_orientation(0, 0, 0, 5)
        self._move_to_orientation(-15, -75, 25, 5)
        self._move_to_orientation(15, 75, -25, 5)
        self._move_to_orientation(0, 0, 0, 5)

    def exam_seq(self):
        self.reset()
        self._move_to_orientation(roll_deg=0, pitch_deg=0, yaw_deg=0, period_sec=2)
        self._move_to_orientation(-35, 0, 0, 2)
        self._move_to_orientation(35, 0, 0, 2)
        self._move_to_orientation(0, 0, 0, 3)
        self._move_to_orientation(0, -35, 0, 2)
        self._move_to_orientation(0, 35, 0, 2)
        self._move_to_orientation(0, 0, 0, 3)
        self._move_to_orientation(0, 0, 30, 1)
        self._move_to_orientation(0, 0, -30, 1)
        self._move_to_orientation(0, 0, 0, 2)

    def self_seq(self):
        self.reset()
        self._move_to_orientation(roll_deg=0, pitch_deg=0, yaw_deg=0, period_sec=2)
        self._move_to_orientation(0, 0, 0, 2)
        self._move_to_orientation(0, 35, 0, 2)
        self._move_to_orientation(0, -35, 0, 2)
        self._move_to_orientation(0, 0, 0, 3)
        self._move_to_orientation(-35, 0, 0, 1)
        self._move_to_orientation(35, 0, 0, 1)
        self._move_to_orientation(0, 0, 0, 3)
        self._move_to_orientation(0, 0, 30, 1)
        self._move_to_orientation(0, 0, -30, 1)
        self._move_to_orientation(0, 0, 0, 2)

    def store_data(self):
        folder_path = "imu_sim_data/"
        np.save(folder_path + "rollList_1", np.array(self.ground_truth_deg)[:, 0])
        np.save(folder_path + "pitchList_1", np.array(self.ground_truth_deg)[:, 1])
        np.save(folder_path + "yawList_1", np.array(self.ground_truth_deg)[:, 2])
        np.save(folder_path + "A_List_1", np.array(self.accelorometer_data))
        np.save(folder_path + "G_List_1", np.array(self.gyroscope_data))
        np.save(folder_path + "M_List_1", np.array(self.magnetometer_data))

    def plot_imu_data(self):
        """
        Plots the acceleration, angular velocity and magnetic field strength data against the total time.

        Returns:
        None
        """
        t = np.linspace(
            start=0, stop=self.total_time, num=self.total_time * self.sampling_frequency
        )

        plt.figure(figsize=(10, 8))

        # Accelerometer subplot
        plt.subplot(3, 1, 1)
        plt.plot(t, np.array(self.accelorometer_data)[:, 0], "-r", label="roll")
        plt.plot(t, np.array(self.accelorometer_data)[:, 1], "-b", label="pitch")
        plt.plot(t, np.array(self.accelorometer_data)[:, 2], "-g", label="yaw")
        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration (m/s^2)")
        plt.legend()

        # Gyroscope subplot
        plt.subplot(3, 1, 2)
        plt.plot(t, np.array(self.gyroscope_data)[:, 0], "-r", label="roll")
        plt.plot(t, np.array(self.gyroscope_data)[:, 1], "-b", label="pitch")
        plt.plot(t, np.array(self.gyroscope_data)[:, 2], "-g", label="yaw")
        plt.xlabel("Time (s)")
        plt.ylabel("Angular velocity (degrees/s)")
        plt.legend()

        # Magnetometer subplot
        plt.subplot(3, 1, 3)
        plt.plot(t, np.array(self.magnetometer_data)[:, 0], "-r", label="roll")
        plt.plot(t, np.array(self.magnetometer_data)[:, 1], "-b", label="pitch")
        plt.plot(t, np.array(self.magnetometer_data)[:, 2], "-g", label="yaw")
        plt.xlabel("Time (s)")
        plt.ylabel("Magnetic field strength (mT)")
        plt.legend()

        plt.show()
