import numpy as np

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


# plot_rot_data(43)
# stat_analysis(Acc_rot_data,Gyro_rot_data,Mag_rot_data)
