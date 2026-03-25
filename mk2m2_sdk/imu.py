# File Name:   imu.py
# Author:      OSCP
#
# Description: This file is the public API of the MK2M2 SDK.
#              It is the only file a customer or developer needs to interact with.
#              All COBS decoding, CRC verification, and binary frame parsing
#              is handled internally. The user just calls simple methods like
#              connect(), read(), set_mode(), and set_gyro_range().
#
#              To achieve this we wrap the simulator and parser behind a clean
#              class interface. In real hardware this class would open the RS422
#              serial port at 921600 baud 8N1 instead of starting the simulator.
#              The public methods would be identical either way.
#
#              Keep in mind: every configuration method here maps to a real
#              ASCII command from Section 11 of the MK2M2 datasheet REV08.
#              The comments above each method show the exact command sequence
#              that would be sent to real hardware following the state machine
#              from Section 10.2. This makes the SDK ready for real hardware
#              with minimal changes.

import sys      # used to modify the Python path so imports work correctly
import os       # used to build file paths that work on any operating system

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # add parent folder to path

from simulator import MK2M2Simulator                            # import the digital twin
from mk2m2_sdk.parser import decode_frame, parse_startup_frame, cobs_decode_frame  # import parsers


# MK2M2: the clean public SDK interface for the OSCP MK2M2 IMU
# Wraps the simulator in simulation mode, would wrap RS422 serial in real hardware mode
# The public API is identical in both cases, that is the purpose of this class
class MK2M2:

    # __init__: creates the SDK object and sets up internal state
    # Does not connect to the IMU yet, call connect() to start
    # Input:  None
    # Return: None
    def __init__(self):
        self._simulator = MK2M2Simulator()  # create the digital twin instance
        self._connected = False             # track whether we are currently connected
        self._last_data = None              # store the most recent parsed frame data

    # connect: starts the IMU connection and begins receiving frames
    # In simulation mode this starts the simulator background thread
    # In real hardware this would open the RS422 serial port at 921600 baud 8N1
    # Reference: RS422 settings from MK2M2 datasheet Section 10.1.1
    # Input:  None
    # Return: None
    def connect(self):
        self._simulator.start()     # start the simulator background thread
        self._connected = True      # mark as connected
        print("[MK2M2 SDK] Connected to MK2M2 (simulation mode)")  # confirm connection

    # disconnect: stops the IMU and closes the connection
    # Input:  None
    # Return: None
    def disconnect(self):
        self._simulator.stop()      # stop the simulator background thread
        self._connected = False     # mark as disconnected
        print("[MK2M2 SDK] Disconnected")  # confirm disconnection

    # is_connected: returns whether we currently have an active connection
    # Input:  None
    # Return: bool - True if connected, False if not
    def is_connected(self):
        return self._connected  # return the connection state flag

    # read: reads and returns the latest sensor data as a clean dictionary
    # Runs the full decode pipeline: get frame, strip delimiter, COBS decode, CRC verify, unpack
    # Returns None if not connected or no frame is available yet
    # Input:  None
    # Return: dict or None - sensor values with keys gyro_x/y/z, accel_x/y/z, temperature etc
    def read(self):
        if not self._connected:     # cannot read if not connected
            return None

        encoded = self._simulator.get_latest_frame()  # get latest COBS-encoded frame from simulator
        if encoded is None:         # no frame generated yet
            return None

        frame_data = encoded.rstrip(b'\x00')  # strip the 0x00 frame delimiter before decoding
        parsed = decode_frame(frame_data)     # run full decode pipeline: COBS, CRC, unpack

        if parsed and parsed.get('crc_ok'):   # only store data if CRC passed
            self._last_data = parsed          # save for reference

        return parsed  # return the parsed dictionary to the caller

    # get_startup_info: reads and returns the Startup Frame configuration
    # The Startup Frame contains unit number, firmware version, and current settings
    # In real hardware this sends the SUF command from Section 11.4 and waits for response
    # Reference: MK2M2 datasheet Section 10.3, Table 5
    # Input:  None
    # Return: dict or None - unit configuration as dictionary, None if not connected
    def get_startup_info(self):
        if not self._connected:     # cannot read if not connected
            return None

        encoded = self._simulator.build_startup_frame()  # build startup frame from simulator
        frame_data = encoded.rstrip(b'\x00')             # strip the 0x00 delimiter
        raw = cobs_decode_frame(frame_data)              # COBS decode to get raw bytes
        if raw is None:             # decoding failed
            return None
        return parse_startup_frame(raw)  # parse and return the startup frame fields

    # get_euler_angles: returns current orientation as roll, pitch, yaw in degrees
    # Simple integration of gyro data from the simulator
    # For drift-corrected orientation use fusion_ahrs.py instead
    # Input:  None
    # Return: tuple (roll, pitch, yaw) in degrees
    def get_euler_angles(self):
        return self._simulator.get_euler_angles()  # delegate to simulator

    # get_latest_sensor_values: returns most recent sensor values as a dictionary
    # Faster than parsing a full frame, used by the GUI for live display updates
    # Input:  None
    # Return: dict with keys gyro_x/y/z, accel_x/y/z, roll, pitch, yaw, temp
    def get_latest_sensor_values(self):
        return self._simulator.get_latest_sensor_values()  # delegate to simulator

    # set_mode: changes the IMU operating mode and output rate
    # Valid modes: 'I' = Idle 0Hz, 'L' = Low 100Hz, 'M' = Medium 500Hz
    # In real hardware sends: CONFIG\r\n then OM<mode>\r\n then EXIT\r\n
    # Reference: MK2M2 datasheet Section 11.6, Table 34
    # Input:  mode (str) - 'I', 'L', or 'M'
    # Return: None
    def set_mode(self, mode):
        if not self._connected:     # cannot configure if not connected
            print("[MK2M2 SDK] Not connected")
            return
        self._simulator.set_mode(mode)  # apply mode change to simulator
        mode_names = {'I': 'Idle', 'L': 'Low (100Hz)', 'M': 'Medium (500Hz)'}  # readable names
        print(f"[MK2M2 SDK] Mode set to: {mode_names.get(mode, mode)}")  # confirm change

    # set_gyro_range: changes the gyroscope dynamic range
    # Valid values: 125, 250, 500, 1000, 2000, 4000 deg/sec, from Table 2
    # In real hardware sends: CONFIG\r\n then DRG<range>\r\n then EXIT\r\n
    # Reference: MK2M2 datasheet Section 11.8, Table 36
    # Input:  range_dps (int) - dynamic range in degrees per second
    # Return: None
    def set_gyro_range(self, range_dps):
        if not self._connected:     # cannot configure if not connected
            return
        self._simulator.set_gyro_range(range_dps)  # apply range change to simulator
        print(f"[MK2M2 SDK] Gyro range set to: +/-{range_dps} deg/sec")  # confirm change

    # set_accel_range: changes the accelerometer dynamic range
    # Valid values: 2, 4, 8, 16 g, from Table 2
    # In real hardware sends: CONFIG\r\n then DRA<range>\r\n then EXIT\r\n
    # Reference: MK2M2 datasheet Section 11.9, Table 37
    # Input:  range_g (int) - dynamic range in g
    # Return: None
    def set_accel_range(self, range_g):
        if not self._connected:     # cannot configure if not connected
            return
        self._simulator.set_accel_range(range_g)  # apply range change to simulator
        print(f"[MK2M2 SDK] Accel range set to: +/-{range_g} g")  # confirm change

    # set_misalignment_correction: enables or disables misalignment correction
    # In real hardware sends: CONFIG\r\n then EMCORR\r\n or DMCORR\r\n then EXIT\r\n
    # Reference: MK2M2 datasheet Section 11.11, Table 39
    # Input:  enabled (bool) - True to enable correction, False to disable
    # Return: None
    def set_misalignment_correction(self, enabled):
        if not self._connected:     # cannot configure if not connected
            return
        self._simulator.misalignment_correction = enabled  # apply setting to simulator
        state = "enabled" if enabled else "disabled"        # readable state string
        print(f"[MK2M2 SDK] Misalignment correction {state}")  # confirm change

    # reset: performs a soft reset of the IMU
    # In real hardware sends: RESET\r\n which works from any state, Section 11.3
    # Reference: MK2M2 datasheet Section 11.3, Table 31
    # Input:  None
    # Return: None
    def reset(self):
        if not self._connected:     # cannot reset if not connected
            return
        self._simulator.stop()      # stop the simulator
        self._simulator.start()     # restart it, equivalent to power cycle
        print("[MK2M2 SDK] IMU reset")  # confirm reset

    # apply_rotation: simulates the IMU rotating at a known rate on one axis
    # Used by test scripts for scale factor and misalignment testing
    # Input:  axis (str)       - 'x', 'y', or 'z'
    #         rate_dps (float) - rotation rate in degrees per second
    # Return: None
    def apply_rotation(self, axis, rate_dps):
        self._simulator.apply_rotation(axis, rate_dps)  # delegate to simulator

    # stop_rotation: stops all simulated rotation and returns to stationary
    # Input:  None
    # Return: None
    def stop_rotation(self):
        self._simulator.stop_rotation()  # delegate to simulator