# File Name:   scale_factor.py
# Author:      OSCP
#
# Description: This file automates the gyroscope scale factor test for
#              the MK2M2 digital twin. Scale factor error measures how
#              accurately the IMU reports a known rotation rate. If you
#              command the simulator to rotate at exactly 250 deg/sec,
#              does the sensor report 250 deg/sec or something slightly off?
#              The difference expressed as a percentage is the scale factor error.
#
#              To achieve this we command a known rotation rate on one axis,
#              collect 200 samples of what the sensor reports, compute the
#              mean reported value, and compare it to the true commanded value.
#              We repeat this for all three axes and multiple rotation rates.
#
#              Keep in mind: the pass/fail threshold of 1.0% comes directly
#              from Table 1 of the MK2M2 datasheet REV08, Scale Factor Error
#              maximum column. Every test result is checked against this spec.
#              The settle time after applying rotation exists because the noise
#              model needs a moment to stabilize around the new true value.
#
# Run:         python test_automation/scale_factor.py

import sys      # used to modify Python path for imports
import os       # used to build file paths
import time     # used for settle time between rotation commands and sample collection
import statistics  # provides mean and stdev functions for sample analysis

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # add parent folder to path
from mk2m2_sdk.imu import MK2M2  # import the SDK public API


# rotation rates to test in deg/sec, chosen to cover low, mid, and high range
TEST_RATES = [50.0, 100.0, 250.0, 500.0]

# axes to test one at a time
TEST_AXES = ['x', 'y', 'z']

# number of samples to collect per test point
NUM_SAMPLES = 200

# time to wait after applying rotation before collecting samples, in seconds
SETTLE_TIME = 0.3

# maximum allowed scale factor error from Table 1 of MK2M2 datasheet REV08
MAX_ERROR_PERCENT = 1.0

# maps axis letter to the dictionary key returned by get_latest_sensor_values
AXIS_KEYS = {'x': 'gyro_x', 'y': 'gyro_y', 'z': 'gyro_z'}


# collect_samples: collects N gyroscope samples from one axis
# Polls the IMU at 10ms intervals which matches 100Hz Low speed mode output rate
# Input:  imu  (MK2M2) - the connected SDK instance to read from
#         axis (str)   - 'x', 'y', or 'z' axis to collect from
#         n    (int)   - number of samples to collect
# Return: list - list of float values in deg/sec
def collect_samples(imu, axis, n):
    key = AXIS_KEYS[axis]   # get the dictionary key for this axis
    samples = []             # empty list to store collected samples
    while len(samples) < n:  # keep collecting until we have enough
        data = imu.get_latest_sensor_values()  # get latest sensor readings from SDK
        if data:                               # only store if data is available
            samples.append(data[key])          # append the value for our axis
        time.sleep(0.01)    # wait 10ms between samples to match 100Hz output rate
    return samples           # return the collected samples list


# run_single_test: runs scale factor test for one axis at one rotation rate
# Commands the rotation, collects samples, computes error, returns result dict
# Input:  imu       (MK2M2) - connected SDK instance
#         axis      (str)   - 'x', 'y', or 'z'
#         true_rate (float) - commanded rotation rate in deg/sec
# Return: dict - test results with keys axis, true_rate, mean_reported,
#                error_percent, std_dev, passed
def run_single_test(imu, axis, true_rate):
    imu.apply_rotation(axis, true_rate)  # command the known rotation rate on this axis
    time.sleep(SETTLE_TIME)              # wait for noise model to stabilize

    samples = collect_samples(imu, axis, NUM_SAMPLES)  # collect samples from this axis
    imu.stop_rotation()                                # stop rotation after collecting

    mean_reported = statistics.mean(samples)    # average of all collected samples
    std_dev       = statistics.stdev(samples)   # standard deviation shows noise spread

    error_percent = abs(mean_reported - true_rate) / true_rate * 100  # scale factor error formula
    passed = error_percent < MAX_ERROR_PERCENT   # pass if error is within 1.0% spec from Table 1

    return {
        'axis':          axis,           # which axis was tested
        'true_rate':     true_rate,      # commanded rotation rate in deg/sec
        'mean_reported': mean_reported,  # average reported value in deg/sec
        'std_dev':       std_dev,        # standard deviation of samples
        'error_percent': error_percent,  # scale factor error as percentage
        'passed':        passed,         # True if within 1.0% spec from Table 1
    }


# run_scale_factor_test: runs the full scale factor test across all axes and rates
# Prints a formatted report to the terminal showing each result and overall outcome
# Input:  None
# Return: bool - True if all tests passed, False if any failed
def run_scale_factor_test():
    print("=" * 65)
    print("  OSCP MK2M2 - Gyroscope Scale Factor Test")
    print("  Spec: Scale Factor Error < 1.0% (Table 1, REV08)")
    print("=" * 65)

    imu = MK2M2()       # create SDK instance
    imu.connect()        # connect to digital twin
    imu.set_mode('L')   # set Low speed mode for 100Hz output rate

    time.sleep(0.5)  # wait 500ms for simulator to start generating frames

    results    = []     # list to store all test result dictionaries
    all_passed = True   # track overall pass/fail, assume pass until a failure occurs

    for axis in TEST_AXES:  # loop through each axis
        print(f"\n  Testing Axis: {axis.upper()}")
        print(f"  {'Rate (deg/s)':<14} {'Reported (deg/s)':<18} {'Error %':<12} {'Std Dev':<12} {'Result'}")
        print(f"  {'-'*60}")

        for rate in TEST_RATES:  # loop through each rotation rate
            result = run_single_test(imu, axis, rate)  # run test for this axis and rate
            results.append(result)                      # store the result

            status = "PASS" if result['passed'] else "FAIL"  # readable pass/fail label
            if not result['passed']:    # if any test fails
                all_passed = False      # mark overall result as failed

            print(
                f"  {result['true_rate']:<14.1f}"       # commanded rate
                f"{result['mean_reported']:<18.4f}"     # average reported value
                f"{result['error_percent']:<12.4f}"     # error percentage
                f"{result['std_dev']:<12.6f}"           # standard deviation
                f"{status}"                             # pass or fail
            )

    imu.disconnect()  # disconnect from digital twin after all tests complete

    total  = len(results)                           # total number of test points
    passed = sum(1 for r in results if r['passed']) # count passed tests
    failed = total - passed                         # count failed tests

    print("\n" + "=" * 65)
    print(f"  Total tests : {total}")
    print(f"  Passed      : {passed}")
    print(f"  Failed      : {failed}")
    print()

    if all_passed:
        print("  OVERALL RESULT: PASS - All axes within spec")
    else:
        print("  OVERALL RESULT: FAIL - One or more axes out of spec")

    print("=" * 65)

    return all_passed  # return overall result for use by other scripts


# entry point when file is run directly
if __name__ == "__main__":
    run_scale_factor_test()  # run the full test suite and print results