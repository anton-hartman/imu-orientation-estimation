import matplotlib.pyplot as plt
import numpy as np
import statistics


sampling_rate = 1  # hz
g_roll = 0
g_pitch = 0
g_yaw = 0

Acc_rot_data = []
Gyro_rot_data = []
Mag_rot_data = []


def rotate(roll, pitch, yaw, time):
    global sampling_rate
    roll_rad = roll * np.pi / 180
    pitch_rad = pitch * np.pi / 180
    yaw_rad = yaw * np.pi / 180
    number_of_samples = sampling_rate * time
    new_roll_rad = roll_rad / number_of_samples
    new_pitch_rad = pitch_rad / number_of_samples
    new_yaw_rad = yaw_rad / number_of_samples
    for t in range(0, int(number_of_samples), 1):
        print("Sample : %d" % t)
        calc_data_time(new_roll_rad, new_pitch_rad, new_yaw_rad)


def calc_data_time(roll, pitch, yaw):
    global g_roll
    global g_pitch
    global g_yaw

    g_roll = g_roll + roll
    g_pitch = g_pitch + pitch
    g_yaw = g_yaw + yaw

    calc_acc_delta(g_roll, g_pitch, g_yaw)
    calc_gyro_delta(roll, pitch, yaw)
    calc_mag_delta(g_roll, g_pitch, g_yaw)


def calc_acc_delta(roll, pitch, yaw):
    global Acc_rot_data
    R_x = [
        [1, 0, 0],
        [0, np.cos(roll), np.sin(roll)],
        [0, -1 * np.sin(roll), np.cos(roll)],
    ]
    R_y = [
        [np.cos(pitch), 0, -1 * np.sin(pitch)],
        [0, 1, 0],
        [np.sin(pitch), 0, np.cos(pitch)],
    ]
    R_z = [[np.cos(yaw), np.sin(yaw), 0], [-1 * np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]

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
    Acc_rot_data.append(new_data)


def calc_gyro_delta(roll, pitch, yaw):
    global Gyro_rot_data
    global sampling_rate
    roll_rate = 180 / np.pi * roll * sampling_rate
    pitch_rate = 180 / np.pi * pitch * sampling_rate
    yaw_rate = 180 / np.pi * yaw * sampling_rate

    gyro_bias = [0, 0, 0]
    new_data = [
        roll_rate + gyro_bias[0],
        pitch_rate + gyro_bias[1],
        yaw_rate + gyro_bias[2],
    ]
    G_noise = 0.7 * np.random.normal(0, 1, 3)
    new_data = new_data + G_noise
    Gyro_rot_data.append(new_data)


def calc_mag_delta(roll, pitch, yaw):
    global Mag_rot_data
    R_x = [
        [1, 0, 0],
        [0, np.cos(roll), np.sin(roll)],
        [0, -1 * np.sin(roll), np.cos(roll)],
    ]
    R_y = [
        [np.cos(pitch), 0, -1 * np.sin(pitch)],
        [0, 1, 0],
        [np.sin(pitch), 0, np.cos(pitch)],
    ]
    R_z = [[np.cos(yaw), np.sin(yaw), 0], [-1 * np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]

    R_x_mat = np.matrix(R_x)
    R_y_mat = np.matrix(R_y)
    R_z_mat = np.matrix(R_z)

    R_xy = np.matmul(R_x_mat, R_y_mat)

    R_xyz = np.matmul(R_xy, R_z_mat)
    B = 28310.9e-9
    # B = 28.3
    inclination = 61.49

    mag_ref_point = (
        B * np.matrix([np.cos(inclination), 0, np.sin(inclination)]).transpose()
    )
    mag_orientation = -1 * np.matmul(R_xyz, mag_ref_point)

    mag_bias = [0, 0, 0]
    new_data = [
        mag_orientation.tolist()[0][0] + mag_bias[0],
        mag_orientation.tolist()[1][0] + mag_bias[1],
        mag_orientation.tolist()[2][0] + mag_bias[2],
    ]
    M_noise = 1e-6 * np.random.normal(0, 1, 3)
    new_data = new_data + M_noise
    Mag_rot_data.append(new_data)


def plot_rot_data(time):
    global sampling_rate
    t = np.linspace(0, time, time * sampling_rate)
    # print("Time : ")
    # print(t)

    # print("Acc_rot_data : ")
    # print(Acc_rot_data[:][1])

    plt.figure(1)
    plt.plot(t, np.array(Acc_rot_data)[:, 0], "-r", label="Ax")
    plt.plot(t, np.array(Acc_rot_data)[:, 1], "-b", label="Ay")
    plt.plot(t, np.array(Acc_rot_data)[:, 2], "-g", label="Az")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s^2)")
    plt.legend()
    # plt.show()

    # print("Gyro data : ")
    # print(Gyro_rot_data)
    # print(Gyro_rot_data[:][:][3])
    # print(np.shape(Gyro_rot_data))

    plt.figure(2)
    plt.plot(t, np.array(Gyro_rot_data)[:, 0], "-r", label="Gx")
    plt.plot(t, np.array(Gyro_rot_data)[:, 1], "-b", label="Gy")
    plt.plot(t, np.array(Gyro_rot_data)[:, 2], "-g", label="Gz")
    plt.xlabel("Time (s)")
    plt.ylabel("Angular velocity (dps)")
    plt.legend()
    # plt.show()

    print(Mag_rot_data)
    plt.figure(3)
    plt.plot(t, np.array(Mag_rot_data)[:, 0], "-r", label="Mx")
    plt.plot(t, np.array(Mag_rot_data)[:, 1], "-b", label="My")
    plt.plot(t, np.array(Mag_rot_data)[:, 2], "-g", label="Mz")
    plt.xlabel("Time (s)")
    plt.ylabel("Magnetic field strength")
    plt.legend()
    plt.show()


def stat_analysis(AD, GD, MD):
    plt.figure(4)
    plt.hist(np.array(AD)[:, 0], bins=60, color="b", edgecolor="k")
    Ax_mean = statistics.mean(np.array(AD)[:, 0])
    Ax_std = statistics.stdev(np.array(AD)[:, 0])
    print("AX mean ", Ax_mean)
    print("Ax std ", Ax_std)
    print("Ax_ variance ", pow(Ax_std, 2))

    plt.axvline(Ax_mean, color="r")
    plt.axvline(Ax_mean + Ax_std, color="g")
    plt.axvline(Ax_mean - Ax_std, color="g")

    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title("Accelerometer X")
    # //////////////////////////
    plt.figure(5)
    plt.hist(np.array(AD)[:, 1], bins=60, color="b", edgecolor="k")
    Ay_mean = statistics.mean(np.array(AD)[:, 1])
    Ay_std = statistics.stdev(np.array(AD)[:, 1])
    print("Ay mean ", Ay_mean)
    print("Ay std ", Ay_std)
    print("Ay_ variance ", pow(Ay_std, 2))

    plt.axvline(Ay_mean, color="r")
    plt.axvline(Ay_mean + Ay_std, color="g")
    plt.axvline(Ay_mean - Ay_std, color="g")

    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title("Accelerometer Y")
    # //////////////////////////
    plt.figure(6)
    plt.hist(np.array(AD)[:, 2], bins=60, color="b", edgecolor="k")
    Az_mean = statistics.mean(np.array(AD)[:, 2])
    Az_std = statistics.stdev(np.array(AD)[:, 2])
    print("Az mean ", Az_mean)
    print("Az std ", Az_std)
    print("Az_ variance ", pow(Az_std, 2))

    plt.axvline(Az_mean, color="r")
    plt.axvline(Az_mean + Az_std, color="g")
    plt.axvline(Az_mean - Az_std, color="g")

    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title("Accelerometer Z")

    #################################################################################
    plt.figure(7)
    plt.hist(np.array(GD)[:, 0], bins=60, color="b", edgecolor="k")
    Gx_mean = statistics.mean(np.array(GD)[:, 0])
    Gx_std = statistics.stdev(np.array(GD)[:, 0])
    print("gX mean ", Gx_mean)
    print("gx std ", Gx_std)
    print("gx_ variance ", pow(Gx_std, 2))

    plt.axvline(Gx_mean, color="r")
    plt.axvline(Gx_mean + Gx_std, color="g")
    plt.axvline(Gx_mean - Gx_std, color="g")

    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title("Gyroscope X")
    # //////////////////////////
    plt.figure(8)
    plt.hist(np.array(GD)[:, 1], bins=60, color="b", edgecolor="k")
    Gy_mean = statistics.mean(np.array(GD)[:, 1])
    Gy_std = statistics.stdev(np.array(GD)[:, 1])
    print("gy mean ", Gy_mean)
    print("gy std ", Gy_std)
    print("gy_ variance ", pow(Gy_std, 2))

    plt.axvline(Gy_mean, color="r")
    plt.axvline(Gy_mean + Gy_std, color="g")
    plt.axvline(Gy_mean - Gy_std, color="g")

    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title("Gyroscope Y")
    # //////////////////////////
    plt.figure(9)
    plt.hist(np.array(GD)[:, 2], bins=60, color="b", edgecolor="k")
    Gz_mean = statistics.mean(np.array(GD)[:, 2])
    Gz_std = statistics.stdev(np.array(GD)[:, 2])
    print("gz mean ", Gz_mean)
    print("gz std ", Gz_std)
    print("gz_ variance ", pow(Gz_std, 2))

    plt.axvline(Gz_mean, color="r")
    plt.axvline(Gz_mean + Gz_std, color="g")
    plt.axvline(Gz_mean - Gz_std, color="g")

    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title("Gyroscope Z")

    #################################################################################
    plt.figure(10)
    plt.hist(np.array(MD)[:, 0], bins=60, color="b", edgecolor="k")
    Mx_mean = statistics.mean(np.array(MD)[:, 0])
    Mx_std = statistics.stdev(np.array(MD)[:, 0])
    print("MX mean ", Mx_mean)
    print("Mx std ", Mx_std)
    print("Mx_ variance ", pow(Mx_std, 2))

    plt.axvline(Mx_mean, color="r")
    plt.axvline(Mx_mean + Mx_std, color="g")
    plt.axvline(Mx_mean - Mx_std, color="g")

    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title("Magnometer X")
    # //////////////////////////
    plt.figure(11)
    plt.hist(np.array(MD)[:, 1], bins=60, color="b", edgecolor="k")
    My_mean = statistics.mean(np.array(MD)[:, 1])
    My_std = statistics.stdev(np.array(MD)[:, 1])
    print("My mean ", My_mean)
    print("My std ", My_std)
    print("My_ variance ", pow(My_std, 2))

    plt.axvline(My_mean, color="r")
    plt.axvline(My_mean + My_std, color="g")
    plt.axvline(My_mean - My_std, color="g")

    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title("Magnometer Y")
    # //////////////////////////
    plt.figure(12)
    plt.hist(np.array(MD)[:, 2], bins=60, color="b", edgecolor="k")
    Mz_mean = statistics.mean(np.array(MD)[:, 2])
    Mz_std = statistics.stdev(np.array(MD)[:, 2])
    print("Mz mean ", Mz_mean)
    print("Mz std ", Mz_std)
    print("Mz_ variance ", pow(Mz_std, 2))

    plt.axvline(Mz_mean, color="r")
    plt.axvline(Mz_mean + Mz_std, color="g")
    plt.axvline(Mz_mean - Mz_std, color="g")

    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title("Magnometer Z")

    plt.show()


# plot_rot_data(43)
# stat_analysis(Acc_rot_data,Gyro_rot_data,Mag_rot_data)


def get_array(input_array):
    if not isinstance(input_array, list):
        raise ValueError("Input must be a list of arrays")

    # Initialize an empty list to store the acceleration arrays
    temp_list = []

    # Iterate through each array in the input list
    for arr in input_array:
        if not isinstance(arr, np.ndarray):
            raise ValueError("Each element in the input list must be a NumPy array")

        # Check if the shape of the array is (3,)
        if arr.shape != (3,):
            raise ValueError("Each array in the input list must have a shape of (3,)")

        # Append the array to the acceleration list
        temp_list.append(arr)

    # Convert the list of arrays to a NumPy array
    out_array = np.array(temp_list)

    return out_array


def run_sequence():
    rotate(0, 0, 0, 6)
    rotate(0, 70, 0, 5)
    # rotate(0, -70, 0, 5)
    # rotate(33, 0, 0, 6)
    # rotate(-33, 0, 0, 6)
    # rotate(0, 0, 95, 5)
    # rotate(0, 0, -95, 5)
    # rotate(0, 0, 0, 5)


def get_gyr():
    return get_array(Gyro_rot_data)


def get_mag():
    return get_array(Mag_rot_data)


def get_acc():
    return get_array(Acc_rot_data)


def get_frequency():
    return sampling_rate
