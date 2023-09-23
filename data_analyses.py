import matplotlib.pyplot as plt
import numpy as np


import statistics


def stat_analysis(AD, GD, MD):
    """
    This function performs statistical analysis on the accelerometer, gyroscope, and magnetometer data.
    It plots histograms of the data and calculates the mean, standard deviation, and variance of each axis.

    Parameters:
    AD (list): List of accelerometer data.
    GD (list): List of gyroscope data.
    MD (list): List of magnetometer data.

    Returns:
    None
    """
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
