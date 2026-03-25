# File Name:   fusion_ahrs.py
# Author:      OSCP
#
# Description: This file provides stable orientation estimation from raw
#              MK2M2 sensor data using the xioTechnologies Fusion library.
#              It takes raw gyroscope, accelerometer, and magnetometer readings
#              every sample and outputs Roll, Pitch, Yaw in degrees and a
#              quaternion representing the current orientation.
#
#              To achieve this we use the imufusion Python package which
#              implements a Mahony-based AHRS algorithm. This is the same
#              algorithm family as the MK2M2 internal AHRS described in
#              Section 10.5.13 of the datasheet. We configure it with the
#              exact default values from Table 25 so simulated orientation
#              behavior matches what the real device would produce.
#
#              Keep in mind: simple gyro integration accumulates error over
#              time because gyroscopes drift. The Fusion algorithm corrects
#              this by fusing gyro data with accelerometer as a gravity
#              reference and magnetometer as a heading reference. This is
#              why orientation from this file is more stable than the simple
#              roll, pitch, yaw integration in simulator.py.
#              Suggested by Thibaut Gervais, OSCP during the interview process.
#              Library: https://github.com/xioTechnologies/Fusion

import imufusion    # xioTechnologies Fusion library for AHRS orientation
import numpy as np  # numerical arrays required by imufusion for sensor inputs
import sys          # used to modify Python path for imports
import os           # used to build file paths

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))  # add current folder to path
from simulator import MK2M2Simulator  # import simulator for the standalone demo at bottom


# FusionAHRS: wraps the imufusion AHRS algorithm for use with MK2M2 sensor data
# Call update() every sample with gyro, accel, and magnetometer readings
# Then call get_euler() for roll, pitch, yaw or get_quaternion() for raw quaternion
class FusionAHRS:

    # __init__: sets up the AHRS algorithm with settings matching MK2M2 defaults from Table 25
    # Input:  sample_rate (int) - IMU output rate in Hz, 100 for Low mode, 500 for Medium mode
    # Return: None
    def __init__(self, sample_rate=100):
        self.sample_rate = sample_rate  # store sample rate for time step calculation

        self._ahrs = imufusion.Ahrs()   # create the AHRS algorithm object from imufusion

        # configure algorithm settings to match MK2M2 AHRS defaults from Table 25
        # CONVENTION_NWU = North West Up axes convention, FCO default value from Table 25
        # 0.5 = gyroscope gain FGA, default value from Table 25
        # 2000 = gyroscope range in deg/sec, matches our simulator gyro range
        # 10.0 = acceleration rejection threshold FAR in degrees, default from Table 25
        # 10.0 = magnetic rejection threshold FMR in degrees, default from Table 25
        # 5 * sample_rate = recovery trigger period FRT of 5 seconds, default from Table 25
        settings = imufusion.Settings(
            imufusion.CONVENTION_NWU,   # NWU axes convention, FCO from Table 25
            0.5,                         # gyroscope gain FGA from Table 25
            2000,                        # gyroscope range in deg/sec
            10.0,                        # acceleration rejection FAR from Table 25
            10.0,                        # magnetic rejection FMR from Table 25
            5 * sample_rate              # recovery trigger period FRT from Table 25
        )
        self._ahrs.settings = settings  # apply the settings to the AHRS object

        self._offset = imufusion.Offset(sample_rate)  # gyro offset correction to reduce drift at runtime

        self._quaternion = (1.0, 0.0, 0.0, 0.0)  # current orientation as quaternion w,x,y,z, identity = no rotation
        self._euler      = (0.0, 0.0, 0.0)        # current orientation as roll, pitch, yaw in degrees
        self._linear_acc = (0.0, 0.0, 0.0)        # linear acceleration with gravity removed

    # update: feeds one sample of sensor data into the AHRS algorithm
    # Must be called at the IMU output rate, 100Hz for Low mode, 500Hz for Medium mode
    # After calling this, get_euler() and get_quaternion() return updated orientation
    # Input:  gyro_xyz  (tuple) - gyroscope reading (x, y, z) in degrees per second
    #         accel_xyz (tuple) - accelerometer reading (x, y, z) in g
    #         mag_xyz   (tuple) - magnetometer reading (x, y, z) in uT, pass None for gyro and accel only
    # Return: None
    def update(self, gyro_xyz, accel_xyz, mag_xyz=None):
        gyro_corrected = self._offset.update(np.array(gyro_xyz, dtype=float))  # apply gyro offset correction to reduce drift

        if mag_xyz is not None:  # if magnetometer data provided, use full 9-axis fusion
            self._ahrs.update(
                gyro_corrected,                     # corrected gyro in deg/sec
                np.array(accel_xyz, dtype=float),   # accel in g
                np.array(mag_xyz,   dtype=float),   # mag in uT
                1.0 / self.sample_rate              # time step in seconds
            )
        else:  # no magnetometer, use 6-axis gyro and accel fusion only
            self._ahrs.update_no_magnetometer(
                gyro_corrected,                     # corrected gyro in deg/sec
                np.array(accel_xyz, dtype=float),   # accel in g
                1.0 / self.sample_rate              # time step in seconds
            )

        q = self._ahrs.quaternion                               # get quaternion from AHRS
        self._quaternion = (q.w, q.x, q.y, q.z)                # store as plain tuple w,x,y,z

        euler = self._ahrs.quaternion.to_euler()                # convert quaternion to Euler angles
        self._euler = (euler[0], euler[1], euler[2])            # store roll, pitch, yaw in degrees

        la = self._ahrs.linear_acceleration                     # get linear accel with gravity removed
        self._linear_acc = (float(la[0]), float(la[1]), float(la[2]))  # store as plain tuple

    # get_quaternion: returns current orientation as a quaternion
    # Quaternions avoid gimbal lock and are used for 3D visualization
    # Reference: MK2M2 datasheet Section 10.5.16
    # Input:  None
    # Return: tuple (w, x, y, z) - quaternion components, (1,0,0,0) = no rotation
    def get_quaternion(self):
        return self._quaternion  # return the stored quaternion tuple

    # get_euler: returns current orientation as roll, pitch, yaw in degrees
    # Euler angles are human readable but susceptible to gimbal lock
    # Reference: MK2M2 datasheet Section 10.5.15
    # Input:  None
    # Return: tuple (roll, pitch, yaw) in degrees
    def get_euler(self):
        return self._euler  # return the stored Euler angles tuple

    # get_linear_acceleration: returns acceleration with gravity component removed
    # Useful for detecting motion independent of the gravitational 1g on Z axis
    # Input:  None
    # Return: tuple (x, y, z) in g, with gravity removed
    def get_linear_acceleration(self):
        return self._linear_acc  # return the stored linear acceleration tuple

    # reset: resets the AHRS algorithm back to its initial state
    # Call this when reconnecting or after a soft reset of the IMU
    # Input:  None
    # Return: None
    def reset(self):
        self._ahrs = imufusion.Ahrs()           # create fresh AHRS object
        self._quaternion = (1.0, 0.0, 0.0, 0.0)  # reset to identity quaternion
        self._euler      = (0.0, 0.0, 0.0)        # reset angles to zero


# standalone demo runs when this file is executed directly
# shows the AHRS working with the simulator for 3 seconds stationary
# then 2 seconds with 90 deg/sec rotation on Z axis to show yaw increasing
if __name__ == "__main__":
    import time  # import time here only for the demo

    print("=" * 55)
    print("  Fusion AHRS Demo - MK2M2 Digital Twin")
    print("  Using imufusion (xioTechnologies Fusion library)")
    print("=" * 55)

    sim = MK2M2Simulator()  # create the simulator instance
    sim.start()              # start generating frames

    ahrs = FusionAHRS(sample_rate=100)  # create AHRS at 100Hz matching Low speed mode

    print("\n  Stationary IMU for 3 seconds...")
    print(f"  {'Time(s)':<10} {'Roll':<10} {'Pitch':<10} {'Yaw':<10} {'Quaternion (w,x,y,z)'}")
    print(f"  {'-'*65}")

    start = time.time()  # record start time
    while time.time() - start < 3.0:  # run for 3 seconds
        data = sim.get_latest_sensor_values()  # get latest sensor readings

        ahrs.update(  # feed sensor data into AHRS
            gyro_xyz  = (data['gyro_x'],  data['gyro_y'],  data['gyro_z']),   # gyro in deg/sec
            accel_xyz = (data['accel_x'], data['accel_y'], data['accel_z']),  # accel in g
            mag_xyz   = (0.3, 0.0, 0.4)   # simulated magnetometer in uT
        )

        roll, pitch, yaw = ahrs.get_euler()       # get current Euler angles
        w, x, y, z       = ahrs.get_quaternion()  # get current quaternion
        elapsed           = time.time() - start   # calculate elapsed time

        print(f"  {elapsed:<10.2f} {roll:<10.3f} {pitch:<10.3f} {yaw:<10.3f} ({w:.3f}, {x:.3f}, {y:.3f}, {z:.3f})")
        time.sleep(0.3)  # print every 0.3 seconds for readability

    sim.stop()  # stop the first simulator instance

    print("\n  Rotating at 90 deg/sec on Z axis (yaw) for 2 seconds...")

    sim = MK2M2Simulator()   # create fresh simulator instance
    sim.start()               # start generating frames
    sim.apply_rotation('z', 90.0)  # command 90 deg/sec rotation on Z axis

    start = time.time()  # record start time for rotation phase
    while time.time() - start < 2.0:  # run for 2 seconds
        data = sim.get_latest_sensor_values()  # get latest sensor readings

        ahrs.update(  # feed sensor data into AHRS
            gyro_xyz  = (data['gyro_x'],  data['gyro_y'],  data['gyro_z']),
            accel_xyz = (data['accel_x'], data['accel_y'], data['accel_z']),
            mag_xyz   = (0.3, 0.0, 0.4)
        )

        roll, pitch, yaw = ahrs.get_euler()      # get current Euler angles
        elapsed = time.time() - start            # calculate elapsed time
        print(f"  {elapsed:<10.2f} {roll:<10.3f} {pitch:<10.3f} {yaw:<10.3f}")  # yaw should increase
        time.sleep(0.3)  # print every 0.3 seconds

    sim.stop()  # stop the simulator
    print("\n  Done. Yaw should show increasing rotation on Z axis.")
    print("=" * 55)