# File Name:   misalignment.py
# Author:      OSCP
#
# Description: This file automates the gyroscope misalignment test for
#              the MK2M2 digital twin. Misalignment means that when you
#              rotate the IMU on only one axis, the other two axes should
#              read zero. If they do not, some of the rotation is leaking
#              into the wrong axes because the sensor axes inside the MEMS
#              chip are not perfectly perpendicular to each other.
#              This leakage as a percentage of the input is the misalignment.
#
#              To achieve this we command a known rotation rate on one axis,
#              collect samples from all three axes simultaneously, then check
#              how much the two off-axes report relative to the input rate.
#              We repeat this with each axis as the active one in turn.
#
#              Keep in mind: the pass/fail threshold of 1.0% used here is a
#              practical engineering threshold. The datasheet Table 1 lists
#              Cross-Axis Sensitivity for the accelerometer at 0.5%. For the
#              gyroscope misalignment we use 1.0% as a reasonable threshold.
#              In a real calibration workflow the misalignment correction from
#              Section 10.5.5 of the datasheet would be applied to compensate
#              for any measured misalignment before the unit ships.
#
# Run:         python test_automation/misalignment.py

import sys          # used to modify Python path for imports
import os           # used to build file paths
import time         # used for settle time between commands and sample collection
import statistics   # provides mean function for sample analysis

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # add parent folder to path
from mk2m2_sdk.imu import MK2M2  # import the SDK public API


# rotation rate used for misalignment test in deg/sec
TEST_RATE = 250.0

# axes to drive one at a time
TEST_AXES = ['x', 'y', 'z']

# number of samples to collect per test
NUM_SAMPLES = 200

# time to wait after applying rotation before collecting samples, in seconds
SETTLE_TIME = 0.3

# maximum allowed misalignment as percentage of input rotation rate
MAX_MISALIGNMENT_PERCENT = 1.0


# collect_all_axes: collects N samples from all three gyro axes simultaneously
# Polls the IMU at 10ms intervals to match 100Hz Low speed mode output rate
# Input:  imu (MK2M2) - connected SDK instance to read from
#         n   (int)   - number of samples to collect per axis
# Return: dict - keys 'x', 'y', 'z' each mapping to a list of float samples
def collect_all_axes(imu, n):
    samples = {'x': [], 'y': [], 'z': []}  # empty lists for each axis
    count = 0                               # track how many samples collected
    while count < n:                        # keep collecting until we have enough
        data = imu.get_latest_sensor_values()  # get latest readings from SDK
        if data:                               # only store if data is available
            samples['x'].append(data['gyro_x'])  # store gyro X sample
            samples['y'].append(data['gyro_y'])  # store gyro Y sample
            samples['z'].append(data['gyro_z'])  # store gyro Z sample
            count += 1                           # increment sample count
        time.sleep(0.01)    # wait 10ms between samples to match 100Hz output rate
    return samples           # return dictionary of sample lists


# run_misalignment_test_axis: tests misalignment with one axis as the active driven axis
# Commands rotation on the active axis and measures leakage on the two off axes
# Input:  imu         (MK2M2) - connected SDK instance
#         active_axis (str)   - 'x', 'y', or 'z', the axis being driven
# Return: dict - results with active axis mean and off axis misalignment percentages
def run_misalignment_test_axis(imu, active_axis):
    imu.apply_rotation(active_axis, TEST_RATE)  # command rotation on the active axis only
    time.sleep(SETTLE_TIME)                     # wait for noise model to stabilize

    samples = collect_all_axes(imu, NUM_SAMPLES)  # collect from all three axes at once
    imu.stop_rotation()                           # stop rotation after collecting

    active_mean = statistics.mean(samples[active_axis])  # average reported on the driven axis

    off_axes = [a for a in ['x', 'y', 'z'] if a != active_axis]  # the two axes that should read zero
    off_axis_results = []  # list to store results for each off axis

    for off_axis in off_axes:  # check each off axis for leakage
        off_mean = statistics.mean(samples[off_axis])           # average reported on this off axis
        misalignment_pct = abs(off_mean) / TEST_RATE * 100      # leakage as percentage of input
        passed = misalignment_pct < MAX_MISALIGNMENT_PERCENT    # pass if within 1.0% threshold

        off_axis_results.append({
            'axis':             off_axis,         # which off axis was measured
            'mean':             off_mean,          # mean reading on this off axis in deg/sec
            'misalignment_pct': misalignment_pct,  # leakage percentage
            'passed':           passed,            # True if within threshold
        })

    return {
        'active_axis':      active_axis,      # the axis that was driven
        'true_rate':        TEST_RATE,        # the commanded rotation rate
        'active_mean':      active_mean,      # mean reported on the active axis
        'off_axis_results': off_axis_results, # list of off axis leakage results
    }


# run_misalignment_test: runs the full misalignment test driving each axis in turn
# Prints a formatted report showing leakage on each off axis and overall outcome
# Input:  None
# Return: bool - True if all cross axis checks passed, False if any failed
def run_misalignment_test():
    print("=" * 65)
    print("  OSCP MK2M2 - Gyroscope Misalignment Test")
    print(f"  Input rate: {TEST_RATE} deg/sec per axis")
    print(f"  Spec: Cross-axis leakage < {MAX_MISALIGNMENT_PERCENT}%")
    print("=" * 65)

    imu = MK2M2()       # create SDK instance
    imu.connect()        # connect to digital twin
    imu.set_mode('L')   # set Low speed mode for 100Hz output rate

    time.sleep(0.5)  # wait 500ms for simulator to start generating frames

    all_passed  = True   # track overall pass/fail, assume pass until a failure occurs
    all_results = []     # list to store all axis result dictionaries

    for active_axis in TEST_AXES:  # loop through each axis as the active driven axis
        result = run_misalignment_test_axis(imu, active_axis)  # run test for this active axis
        all_results.append(result)                              # store the result

        print(f"\n  Active Axis: {active_axis.upper()}"
              f"  Commanded: {result['true_rate']:.1f} deg/s"
              f"  Reported: {result['active_mean']:.4f} deg/s")
        print(f"  {'Off Axis':<12} {'Mean Reading (deg/s)':<24} {'Misalignment %':<18} {'Result'}")
        print(f"  {'-'*60}")

        for off in result['off_axis_results']:  # loop through each off axis result
            status = "PASS" if off['passed'] else "FAIL"  # readable pass/fail label
            if not off['passed']:   # if any off axis fails
                all_passed = False  # mark overall result as failed

            print(
                f"  {off['axis'].upper():<12}"           # off axis label
                f"{off['mean']:<24.6f}"                  # mean reading on off axis
                f"{off['misalignment_pct']:<18.4f}"      # misalignment percentage
                f"{status}"                              # pass or fail
            )

    imu.disconnect()  # disconnect from digital twin after all tests complete

    total_off_axis = sum(len(r['off_axis_results']) for r in all_results)  # total cross axis checks
    passed_count   = sum(                                                    # count passed checks
        1 for r in all_results
        for off in r['off_axis_results'] if off['passed']
    )
    failed_count = total_off_axis - passed_count  # count failed checks

    print("\n" + "=" * 65)
    print(f"  Total cross-axis checks : {total_off_axis}")
    print(f"  Passed                  : {passed_count}")
    print(f"  Failed                  : {failed_count}")
    print()

    if all_passed:
        print("  OVERALL RESULT: PASS - Misalignment within spec on all axes")
    else:
        print("  OVERALL RESULT: FAIL - Misalignment out of spec")

    print("=" * 65)

    return all_passed  # return overall result for use by other scripts


# entry point when file is run directly
if __name__ == "__main__":
    run_misalignment_test()  # run the full test suite and print results