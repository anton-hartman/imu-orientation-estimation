import matplotlib.pyplot as plt
import numpy as np
import time
from ahrs import Quaternion, DEG2RAD, RAD2DEG, QuaternionArray
from ahrs.common.quaternion import slerp
from ahrs import QuaternionArray
from ahrs.filters import AngularRate, Complementary, EKF, Madgwick
import ahrs
import os

# Natural values
G = 9.81  # m/s^2
MAG_FIELD_STRENGTH = 40  # μT (Earth's magnetic field strength)
# DEG_TO_RAD = np.pi / 180


# Parameters
Fs = 100  # Sampling freq (Hz)
Ts = 1 / Fs  # Sampling period (s)
# magnetic_field_world = np.array([48.825, 23.25, 43.5])
# magnetic_field_world = np.array([0, -40, 0])
# magnetic_field_world = np.array([-50, 10, 10])
# magnetic_field_world = np.array([-2.477, 9.207, 13.812])
magnetic_field_world = np.array([9.207, -2.477, -13.812])


SUB_BIAS = False

# CHANGE THIS TO CHANGE FILTER
FILTER_BASELINE = "complementary"
FILTER_RESEARCH = "EKF"
FILTER = "EKF"
N = 16  # number of monte carlo simulations
# num_timesteps = 1822
num_timesteps = 2112
# end of parameters


# Sensor noise
acc_x_noise = [-0.0013000000000000025, np.sqrt(8.870532328788443e-07)]  # bias, std
acc_y_noise = [-0.0157, np.sqrt(8.120552212894452e-07)]  # bias, std
acc_z_noise = [0.022850000000000037, np.sqrt(9.953609964086445e-07)]  # bias, std

gyro_x_noise = [0.017750000000000002, np.sqrt(1.808794646050863e-07)]  # bias, std
gyro_y_noise = [-0.01125, np.sqrt(1.4218865070558618e-07)]  # bias, std
gyro_z_noise = [-0.0062, np.sqrt(9.462255166959345e-08)]  # bias, std

mag_x_noise = [0.1, np.sqrt(0.2307042450614686)]  # bias, std
mag_y_noise = [0.1, np.sqrt(0.2248067835949665)]  # bias, std
mag_z_noise = [0.08, np.sqrt(0.2749442606006313)]  # bias, std

orientation_state = Quaternion()  # [roll, pitch, yaw] ACTUAL

estimated_state = Quaternion()  # [roll, pitch, yaw]

gyro_std = np.array(
    [
        1.808794646050863e-07,
        1.4218865070558618e-07,
        9.462255166959345e-08,
    ]
)
gyro_bias = np.array([0.01775, -0.01125, -0.0062])
acc_bias = np.array([-0.0013, -0.0157, 0.02285])
mag_bias = np.array([mag_x_noise[0], mag_y_noise[0], mag_z_noise[0]])


A = AngularRate(None, Quaternion(), Fs)
COMP = Complementary(
    frequency=Fs,
    gain=0.9,
)

E = EKF(
    frequency=Fs,
    magnetic_ref=magnetic_field_world,
)

TRUE_STATE = np.ndarray((N, num_timesteps, 3))
ESTIMATED_STATE = np.ndarray((N, num_timesteps, 3))
SENSOR_READINGS = np.ndarray((N, num_timesteps, 9))

TRUE_STATE_q = np.ndarray((N, num_timesteps, 4))
ESTIMATED_STATE_baseline = np.ndarray((N, num_timesteps, 4))
ESTIMATED_STATE_research = np.ndarray((N, num_timesteps, 4))

def estimate_orientation(estimated, gyro_reading, acc_reading, mag_reading):
    if FILTER == "gyro":
        # integrate gyro readings
        q = A.update(estimated, gyro_reading)

    if FILTER == "complementary":
        # complementary filter
        q = COMP.update(estimated, gyro_reading, acc_reading * G, mag_reading / 1000)


    if FILTER == "EKF":
        # extended kalman filter
        q = E.update(estimated, gyro_reading, acc_reading * G, mag_reading * 1000)

    return Quaternion(q)


def get_estimated_state(gyro_readings, acc_readings, mag_readings):
    global E
    mag_ref = mag_readings[0]
    E = EKF(
        frequency=Fs,
        magnetic_ref=mag_ref,
    )
    n = len(gyro_readings)
    estimated_state = np.empty((0, 4))
    for i in range(n):
        if i == 0:
            q = Quaternion()
        else:
            if SUB_BIAS:
                new_gyro_reading = gyro_readings[i] - gyro_bias
                new_acc_reading = acc_readings[i] - acc_bias
                new_mag_reading = mag_readings[i] - mag_bias
            else:
                new_gyro_reading = gyro_readings[i]
                new_acc_reading = acc_readings[i]
                new_mag_reading = mag_readings[i]

            q = estimate_orientation(
                estimated_state[i - 1],
                new_gyro_reading,
                new_acc_reading,
                new_mag_reading,
            )

        estimated_state = np.vstack((estimated_state, q.to_array()))

    estimated_state = [Quaternion(q) for q in estimated_state]

    if FILTER == FILTER_RESEARCH:
        ang = np.array([q.to_angles() * RAD2DEG for q in estimated_state])
        ang[:, 0] = ang[:, 0]
        ang[:, 1] = ang[:, 1]
        ang[:, 2] = (ang[:, 2]) * -1
        q = np.array([Quaternion().from_angles(a * DEG2RAD) for a in ang])
        q = [Quaternion(a) for a in q]
        estimated_state = q

    if FILTER == FILTER_BASELINE:
        ang = np.array([q.to_angles() * RAD2DEG for q in estimated_state])
        ang[:, 0] = ang[:, 0]
        ang[:, 1] = ang[:, 1]
        ang[:, 2] = (ang[:, 2] - 90) * -1
        q = np.array([Quaternion().from_angles(a * DEG2RAD) for a in ang])
        q = [Quaternion(a) for a in q]
        estimated_state = q

    return estimated_state



def read_sensor_data():
    global TRUE_STATE, ESTIMATED_STATE, SENSOR_READINGS

    folderName =  os.getcwd()
    folderName = os.path.join(folderName, "Collected_Data_clickup")
    numFiles = int(len(os.listdir(folderName))/6)
    for monteNumber in range(1, numFiles+1):
        AList = np.load(folderName + "/A_List_" + str(monteNumber) + ".npy")
        GList = np.load(folderName + "/G_List_" + str(monteNumber) + ".npy")
        MList = np.load(folderName + "/M_List_" + str(monteNumber) + ".npy")

        rollList = np.load(folderName + "/rollList_" + str(monteNumber) + ".npy")
        pitchList = np.load(folderName + "/pitchList_" + str(monteNumber) + ".npy")
        yawList = np.load(folderName + "/yawList_" + str(monteNumber) + ".npy")

        Ax = [x[0] for x in AList]
        Ay = [x[1] for x in AList]
        Az = [x[2] for x in AList]
        Gx = [x[0] for x in GList]
        Gy = [x[1] for x in GList]
        Gz = [x[2] for x in GList]
        Mx = [x[0] for x in MList]
        My = [x[1] for x in MList]
        Mz = [x[2] for x in MList]

        # convert gyro readings to rad/s
        Gx = np.array(Gx) * DEG2RAD
        Gy = np.array(Gy) * DEG2RAD
        Gz = np.array(Gz) * DEG2RAD

        # orientation = np.array([pitchList, rollList, yawList]).T
        orientation = np.array([-1*pitchList, -1*rollList, yawList]).T

        # add to global arrays
        TRUE_STATE[monteNumber-1] = orientation
        SENSOR_READINGS[monteNumber-1] = np.array([Gx, Gy, Gz, Ax, Ay, Az, Mx, My, Mz]).T

def RMSE(actual, estimated):
    actual = [QuaternionArray(q) for q in actual]
    estimated = [QuaternionArray(q) for q in estimated]
    actual = np.array([q.to_angles() * RAD2DEG for q in actual])
    estimated = np.array([q.to_angles() * RAD2DEG for q in estimated])
    error = np.zeros((actual.shape[1], actual.shape[2]))
    for i in range(estimated.shape[0]):
        e = actual[i] - estimated[i]
        e = np.square(e)
        error += e

    error = error / estimated.shape[0]
    error = np.sqrt(error)
    return error


# def optimal_test(rmse_baseline, rmse_research):
#     T_IV = rmse_baseline - rmse_research
#     T_mu = np.mean(T_IV, axis=0)
#     T_std = np.std(T_IV, axis=0)

#     T = T_mu / T_std

#     return T_mu, T_std, T

def optimal_test(true_state, baseline_state, research_state):
    actual = [QuaternionArray(q) for q in true_state]
    baseline = [QuaternionArray(q) for q in baseline_state]
    research = [QuaternionArray(q) for q in research_state]
    actual = np.array([q.to_angles() * RAD2DEG for q in actual])
    baseline = np.array([q.to_angles() * RAD2DEG for q in baseline])
    research = np.array([q.to_angles() * RAD2DEG for q in research])

    error_baseline = (actual - baseline) ** 2
    error_research = (actual - research) ** 2

    C_baseline = np.mean(error_baseline, axis=1)
    C_research = np.mean(error_research, axis=1)

    delta_C = C_baseline - C_research
    mu = np.mean(delta_C, axis=0)
    std = np.std(delta_C, axis=0)
    T = mu / std

    return mu, std, T
    

def plot_true_estimated(orientation, baseline, research):
    time = np.arange(0, len(orientation) * Ts, Ts)
    orientation_ang = np.array([q.to_angles() * RAD2DEG for q in orientation])
    baseline_ang = np.array([q.to_angles() * RAD2DEG for q in baseline])
    research_ang = np.array([q.to_angles() * RAD2DEG for q in research])

    # plot 3 subfigures for axis
    plt.figure(1)
    plt.suptitle("Orientation estimates for 1 monte carlo simulation")
    plt.subplot(311)
    plt.plot(time, baseline_ang[:, 0])
    plt.plot(time, research_ang[:, 0], "--")
    plt.plot(time, orientation_ang[:, 0], "-.")
    plt.title("Roll")
    plt.legend(["Baseline", "Research", "True"])

    plt.subplot(312)
    plt.plot(time, baseline_ang[:, 1])
    plt.plot(time, research_ang[:, 1], "--")
    plt.plot(time, orientation_ang[:, 1], "-.")
    plt.title("Pitch")

    plt.subplot(313)
    plt.plot(time, baseline_ang[:, 2])
    plt.plot(time, research_ang[:, 2], "--")
    plt.plot(time, orientation_ang[:, 2], "-.")
    plt.title("Yaw")

    # add sup x and y labels
    plt.text(0.5, 0.04, "Time (s)", ha="center", va="center", transform=plt.gcf().transFigure)
    plt.text(0.06, 0.5, "Angle (deg)", ha="center", va="center", rotation="vertical", transform=plt.gcf().transFigure)

    plt.show()


def plot_RMSE(rmse_err_baseline, rmse_err_research):
    t = np.arange(0, len(rmse_err_baseline) * Ts, Ts)
    plt.figure(3)
    plt.suptitle(
        f"RMSE for {N} Monte Carlo Simulations. Baseline ({FILTER_BASELINE}) vs Research ({FILTER_RESEARCH})"
    )
    plt.subplot(311)
    plt.plot(t,rmse_err_baseline[:, 0])
    plt.plot(t,rmse_err_research[:, 0])
    plt.title("Roll RMSE")
    plt.legend(["Baseline", "Research"])

    plt.subplot(312)
    plt.plot(t,rmse_err_baseline[:, 1])
    plt.plot(t,rmse_err_research[:, 1])
    plt.title("Pitch RMSE")

    plt.subplot(313)
    plt.plot(t,rmse_err_baseline[:, 2])
    plt.plot(t,rmse_err_research[:, 2])
    plt.title("Yaw RMSE")

    # add sup x and y labels
    plt.text(0.5, 0.04, "Time (s)", ha="center", va="center", transform=plt.gcf().transFigure)
    plt.text(0.06, 0.5, "RMSE (deg)", ha="center", va="center", rotation="vertical", transform=plt.gcf().transFigure)

    plt.show()

def plot_true_orientation():
    # plot true orientation
    roll = TRUE_STATE[:, :, 0]
    pitch = TRUE_STATE[:, :, 1]
    yaw = TRUE_STATE[:, :, 2]

    # subfigures for axis
    plt.figure(1)
    plt.suptitle("True Orientation")
    plt.subplot(311)
    plt.plot(roll.T)
    plt.title("Roll")
    plt.subplot(312)
    plt.plot(pitch.T)
    plt.title("Pitch")
    plt.subplot(313)
    plt.plot(yaw.T)
    plt.title("Yaw")
    plt.show()

def plot_sensor_readings():
    t = np.arange(0, len(SENSOR_READINGS[0]) * Ts, Ts)
    plt.figure(1)
    plt.suptitle("Gyro Readings")
    plt.subplot(311)
    plt.plot(t,SENSOR_READINGS[:, :, 0].T)
    plt.title("Gyro X")
    plt.subplot(312)
    plt.plot(t,SENSOR_READINGS[:, :, 1].T)
    plt.title("Gyro Y")
    plt.subplot(313)
    plt.plot(t,SENSOR_READINGS[:, :, 2].T)
    plt.title("Gyro Z")

    plt.figure(2)
    plt.suptitle("Accelerometer Readings")
    plt.subplot(311)
    plt.plot(t,SENSOR_READINGS[:, :, 3].T)
    plt.title("Acc X")
    plt.subplot(312)
    plt.plot(t,SENSOR_READINGS[:, :, 4].T)
    plt.title("Acc Y")
    plt.subplot(313)
    plt.plot(t,SENSOR_READINGS[:, :, 5].T)
    plt.title("Acc Z")

    plt.figure(3)
    plt.suptitle("Magnetometer Readings")
    plt.subplot(311)
    plt.plot(t,SENSOR_READINGS[:, :, 6].T)
    plt.title("Mag X")
    plt.subplot(312)
    plt.plot(t,SENSOR_READINGS[:, :, 7].T)
    plt.title("Mag Y")
    plt.subplot(313)
    plt.plot(t,SENSOR_READINGS[:, :, 8].T)
    plt.title("Mag Z")

    # add sup x and y labels
    plt.figure(1)
    plt.text(0.5, 0.04, "Time (s)", ha="center", va="center", transform=plt.gcf().transFigure)
    plt.text(0.06, 0.5, "Angular Velocity (rad/s)", ha="center", va="center", rotation="vertical", transform=plt.gcf().transFigure)

    plt.figure(2)
    plt.text(0.5, 0.04, "Time (s)", ha="center", va="center", transform=plt.gcf().transFigure)
    plt.text(0.06, 0.5, "Acceleration (m/s^2)", ha="center", va="center", rotation="vertical", transform=plt.gcf().transFigure)

    plt.figure(3)
    plt.text(0.5, 0.04, "Time (s)", ha="center", va="center", transform=plt.gcf().transFigure)
    plt.text(0.06, 0.5, "Magnetic Field (μT)", ha="center", va="center", rotation="vertical", transform=plt.gcf().transFigure)

    plt.show()


def show_optimal_test_table(intervals):
    results = []
    for interval in intervals:
        true_state = TRUE_STATE_q[:, interval[0] * Fs : interval[1] * Fs]
        baseline_state = ESTIMATED_STATE_baseline[:, interval[0] * Fs : interval[1] * Fs]
        research_state = ESTIMATED_STATE_research[:, interval[0] * Fs : interval[1] * Fs]
        result = optimal_test(true_state, baseline_state, research_state)
        results.append(result)

        

    # print header
    print ("{:>15}       {:^30}      {:^30}      {:^30}".format('','Pitch','Roll','Yaw'))
    head_str = "{:<10}{:<10}{:<10}".format('mean','std','T')
    print ("{:>15}       {:<30}      {:<30}      {:<30}".format('Time interval',head_str,head_str,head_str))

    for result, interval in zip(results, intervals):
        p_str = "{:<+10.2f}{:<+10.2f}{:<+10.2f}".format(result[0][0], result[1][0], result[2][0])
        r_str = "{:<+10.2f}{:<+10.2f}{:<+10.2f}".format(result[0][1], result[1][1], result[2][1])
        y_str = "{:<+10.2f}{:<+10.2f}{:<+10.2f}".format(result[0][2], result[1][2], result[2][2])
        print ("{:>15}       {:<30}      {:<30}      {:<30}".format(str(interval),p_str,r_str,y_str))



    # print("time interval 1", end="\t\t")
    # for i in range(3):
    #     print(f"({result_t1[0][i]:.2f}, {result_t1[1][i]:.2f}, {result_t1[2][i]:.2f})", end="\t\t")
    # print()

    # print("time interval 2", end="\t\t")
    # for i in range(3):
    #     print(f"({result_t2[0][i]:.2f}, {result_t2[1][i]:.2f}, {result_t2[2][i]:.2f})", end="\t\t")
    # print()

    # print("time interval 3", end="\t\t")
    # for i in range(3):
    #     print(f"({result_t3[0][i]:.2f}, {result_t3[1][i]:.2f}, {result_t3[2][i]:.2f})", end="\t\t")
    # print()

read_sensor_data()

# get the sensor readings of the first monte carlo simulation and estimate the orientation
MONTE_RUN = 9
gyro_readings = SENSOR_READINGS[MONTE_RUN, :, :3]
acc_readings = SENSOR_READINGS[MONTE_RUN, :, 3:6]
mag_readings = SENSOR_READINGS[MONTE_RUN, :, 6:]

true_state = TRUE_STATE[MONTE_RUN]
true_state = [Quaternion(Quaternion().from_angles(ang * DEG2RAD)) for ang in true_state]

FILTER = FILTER_BASELINE

estimated_state_baseline = get_estimated_state(gyro_readings, acc_readings, mag_readings)

FILTER = FILTER_RESEARCH

estimated_state_research = get_estimated_state(gyro_readings, acc_readings, mag_readings)


plot_sensor_readings()

plot_true_estimated(true_state, estimated_state_baseline, estimated_state_research)

for n in range(N):
    gyro_readings = SENSOR_READINGS[n, :, :3]
    acc_readings = SENSOR_READINGS[n, :, 3:6]
    mag_readings = SENSOR_READINGS[n, :, 6:]

    true_state = TRUE_STATE[n]
    true_state = [Quaternion(Quaternion().from_angles(ang * DEG2RAD)) for ang in true_state]

    FILTER = FILTER_BASELINE
    estimated_state = get_estimated_state(gyro_readings, acc_readings, mag_readings)
    ESTIMATED_STATE_baseline[n] = estimated_state

    FILTER = FILTER_RESEARCH
    estimated_state = get_estimated_state(gyro_readings, acc_readings, mag_readings)
    ESTIMATED_STATE_research[n] = estimated_state

    TRUE_STATE_q[n] = true_state

rmse_err_baseline = RMSE(TRUE_STATE_q, ESTIMATED_STATE_baseline)
rmse_err_research = RMSE(TRUE_STATE_q, ESTIMATED_STATE_research)



# time 1: 0 - 7
# time 2: 7 - 13
# time 3: 13 - 18

t1 = [0, 6]
t2 = [6, 12]
t3 = [12, 18]



show_optimal_test_table([t1, t2, t3])
plot_RMSE(rmse_err_baseline, rmse_err_research)
# print table with results
# horisontal is pitch, roll and yaw
# vertical is time 1, 2 and 3

