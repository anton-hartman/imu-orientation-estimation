from ahrs.filters import Complementary
import imu_simulator as imu

imu.run_sequence()

complementary_filter = Complementary(
    gyr=imu.get_gyr(),
    acc=imu.get_acc(),
    mag=imu.get_mag(),
    frequency=imu.get_frequency(),
    gain=0.1,
)

print("Complementary Filter")
print(complementary_filter._compute_all())


from ahrs.filters import EKF

# ekf_filter = EKF()
