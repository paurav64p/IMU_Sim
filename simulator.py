# File Name:   simulator.py
# Author:      OSCP
#
# Description: This file is the digital twin of the OSCP MK2M2 IMU.
#              It generates binary frames in the exact format the real
#              hardware produces over RS422, including COBS encoding and
#              CRC-16 integrity check. Sensor values are made realistic
#              using a noise model tuned to the spec values in Table 1
#              of the MK2M2 datasheet REV08.
#
#              To achieve this we build each frame byte by byte following
#              Table 7 of the datasheet, apply COBS encoding from Section
#              10.5.1, and append a CRC-16 checksum from Section 10.5.2.
#
#              Keep in mind: every number in this file traces back to a
#              specific table or section in the MK2M2 datasheet REV08.
#              The CRC polynomial 0xD175 in the datasheet is Koopman
#              notation. The actual computation uses 0xA2EB standard
#              notation. crcmod requires 0x1A2EB with implicit bit added.
#              Clarified by Thibaut Gervais, OSCP.

import struct       # packs Python values into binary bytes for frame building
import time         # provides timestamps and sleep between frame outputs
import math         # provides sqrt for noise density scaling calculation
import random       # generates realistic random noise and drift values
import threading    # runs frame generator in background so GUI stays responsive
import crcmod       # computes CRC-16 checksum with correct polynomial
from cobs import cobs  # encodes and decodes frames using COBS algorithm


# CRC-16 function setup using crcmod
# 0x1A2EB = polynomial 0xA2EB with implicit leading bit added, required by crcmod
# initCrc = 0xFFFF means we start with all 16 bits set to 1
# rev = False means we process bits MSB first, not reversed
# Reference: MK2M2 datasheet Section 10.5.2
_crc16_fn = crcmod.mkCrcFun(0x1A2EB, initCrc=0xFFFF, rev=False)


# compute_crc16: computes a 16-bit CRC checksum over a block of bytes
# Input:  data (bytes) - the frame payload to checksum, excluding CRC bytes
# Return: int - 16-bit CRC value to append to the frame
def compute_crc16(data):
    return _crc16_fn(data)  # run the crcmod function on the data and return result


# Operating modes dictionary mapping mode letter to output rate and encoding values
# Matches Table 4 in the MK2M2 datasheet REV08
# output_rate is frames per second, ascii is the ASCII code, binary is the 3-bit field
OPERATING_MODES = {
    'I': {'output_rate': 0,   'ascii': 0x49, 'binary': 0b000},  # Idle - no frames output
    'L': {'output_rate': 100, 'ascii': 0x4C, 'binary': 0b001},  # Low speed - 100 Hz
    'M': {'output_rate': 500, 'ascii': 0x4D, 'binary': 0b010},  # Medium speed - 500 Hz
}

# Frame type binary codes packed into bits 2:0 of byte 0 of every frame
# Matches Table 6 in the MK2M2 datasheet REV08
FRAME_TYPES = {
    'R': 0b000,  # Raw Operating Frame - gyro, accel, mag, inclinometer, temperature
    'E': 0b001,  # Euler Angles Frame - roll, pitch, yaw
    'Q': 0b010,  # Quaternion Frame - orientation as quaternion
    'S': 0b111,  # Startup Frame - unit configuration sent on power up
}


# NoiseModel: adds realistic noise and slow drift to a single sensor axis
# This makes simulated sensor readings behave like a real MEMS sensor
# Two noise components are modelled:
#   white noise  - random jitter on every sample, from noise density in Table 1
#   bias drift   - slow random walk accumulating over time, from bias instability in Table 1
class NoiseModel:

    # __init__: sets up the noise model with spec values from Table 1
    # Input:  noise_density (float)    - random jitter per sample in sensor units per sqrt Hz
    #         bias_instability (float) - drift rate in sensor units per second
    # Return: None
    def __init__(self, noise_density, bias_instability):
        self.noise_density = noise_density      # white noise amplitude, from Table 1
        self.bias_instability = bias_instability  # drift rate, from Table 1
        self._bias = 0.0            # current accumulated bias, starts at zero on power up
        self._bias_velocity = 0.0   # rate of change of bias, drives the random walk

    # sample: returns one noisy sensor reading for the given true value
    # Input:  true_value (float) - the actual physical quantity being measured
    #         dt (float)         - time step in seconds between samples
    # Return: float - noisy measurement = true value + drift + random jitter
    def sample(self, true_value, dt):
        white_noise = random.gauss(0, self.noise_density * math.sqrt(1.0 / dt))  # random jitter scaled by sample rate
        self._bias_velocity += random.gauss(0, self.bias_instability * dt * 0.01)  # nudge the drift velocity randomly
        self._bias_velocity *= 0.999    # dampen velocity slightly to prevent unbounded growth
        self._bias += self._bias_velocity * dt  # accumulate bias from velocity over this time step
        return true_value + self._bias + white_noise  # return true value plus all noise components


# MK2M2Simulator: the digital twin of the OSCP MK2M2 IMU
# Generates binary frames at the correct output rate for the selected operating mode
# Frames follow the exact byte layout from Table 7 of the MK2M2 datasheet REV08
# Runs a background thread so the GUI can poll the latest frame without blocking
class MK2M2Simulator:

    # __init__: sets up all simulator state with default configuration values
    # Default values match the MK2M2 factory defaults from Table 2 of the datasheet
    # Input:  None
    # Return: None
    def __init__(self):
        self.operating_mode = 'L'           # default mode is Low speed at 100 Hz
        self.gyro_range = 250               # default gyro range is 250 deg/sec, from Table 2
        self.accel_range = 4                # default accel range is 4g, from Table 2
        self.misalignment_correction = False  # misalignment correction off by default
        self.unit_number = 1                # unit serial number for the startup frame
        self.sw_version = (1, 0, 0)         # simulated firmware version major, minor, patch

        self.true_gyro  = [0.0, 0.0, 0.0]  # true rotation rate in deg/sec for each axis
        self.true_accel = [0.0, 0.0, 1.0]  # true acceleration in g, Z is 1.0 because gravity
        self.true_mag   = [0.3, 0.0, 0.4]  # true magnetic field in uT for each axis
        self.true_incl  = [0.0, 0.0]       # true inclinometer reading in mg for X and Y

        # gyro noise model tuned to Table 1 spec values
        # noise_density 0.001 deg/sec/sqrt(Hz) from Table 1
        # bias_instability 0.000139 deg/sec = 0.5 deg/hr divided by 3600 from Table 1
        self._gyro_noise = [NoiseModel(0.001, 0.000139) for _ in range(3)]

        # accel noise model tuned to Table 1 spec values
        # noise_density 0.000035 g/sqrt(Hz) = 35 ug from Table 1
        # bias_instability 0.00003 g = 30 ug from Table 1
        self._accel_noise = [NoiseModel(0.000035, 0.00003) for _ in range(3)]

        self._frame_counter = 0     # counts transmitted frames, wraps at 255, from Section 10.5.6
        self._start_time = time.time()  # records power-up time for timestamp calculation
        self._running = False       # controls whether the background thread is active
        self._thread = None         # holds the background thread object
        self._latest_frame = None   # stores the most recently generated frame bytes
        self._lock = threading.Lock()  # prevents simultaneous access from GUI and background thread

        self._roll  = 0.0  # integrated roll angle in degrees from gyro X
        self._pitch = 0.0  # integrated pitch angle in degrees from gyro Y
        self._yaw   = 0.0  # integrated yaw angle in degrees from gyro Z

    # start: starts the background thread that generates frames continuously
    # Input:  None
    # Return: None
    def start(self):
        self._running = True                        # set flag so background thread keeps running
        self._start_time = time.time()              # record start time for frame timestamps
        self._thread = threading.Thread(target=self._run, daemon=True)  # create background thread
        self._thread.start()                        # start the background thread

    # stop: stops the background thread
    # Input:  None
    # Return: None
    def stop(self):
        self._running = False  # clear flag so background thread exits its loop

    # get_latest_frame: returns the most recently generated COBS-encoded frame
    # Uses a lock to safely read the frame from another thread
    # Input:  None
    # Return: bytes or None - latest frame bytes, None if no frame generated yet
    def get_latest_frame(self):
        with self._lock:            # acquire lock to prevent reading while frame is being written
            return self._latest_frame  # return the latest frame bytes

    # set_mode: changes the operating mode and output rate of the simulator
    # Matches the OM command from Section 11.6 of the datasheet
    # Input:  mode (str) - 'I' for Idle, 'L' for Low 100Hz, 'M' for Medium 500Hz
    # Return: None
    def set_mode(self, mode):
        if mode in OPERATING_MODES:     # check mode is valid before applying
            self.operating_mode = mode  # update the operating mode

    # set_gyro_range: changes the gyroscope dynamic range
    # Matches the DRG command from Section 11.8 of the datasheet
    # Valid ranges from Table 2: 125, 250, 500, 1000, 2000, 4000 deg/sec
    # Input:  range_dps (int) - the new dynamic range in degrees per second
    # Return: None
    def set_gyro_range(self, range_dps):
        valid = [125, 250, 500, 1000, 2000, 4000]  # valid ranges from Table 2
        if range_dps in valid:          # only apply if value is in the valid list
            self.gyro_range = range_dps  # update the gyro range

    # set_accel_range: changes the accelerometer dynamic range
    # Matches the DRA command from Section 11.9 of the datasheet
    # Valid ranges from Table 2: 2, 4, 8, 16 g
    # Input:  range_g (int) - the new dynamic range in g
    # Return: None
    def set_accel_range(self, range_g):
        valid = [2, 4, 8, 16]           # valid ranges from Table 2
        if range_g in valid:            # only apply if value is in the valid list
            self.accel_range = range_g  # update the accel range

    # apply_rotation: simulates the IMU rotating at a known rate on one axis
    # Used by test scripts to command a specific input for scale factor testing
    # Input:  axis (str)      - 'x', 'y', or 'z'
    #         rate_dps (float) - rotation rate in degrees per second
    # Return: None
    def apply_rotation(self, axis, rate_dps):
        axis_map = {'x': 0, 'y': 1, 'z': 2}    # map axis letter to list index
        if axis in axis_map:                      # check axis is valid
            self.true_gyro[axis_map[axis]] = rate_dps  # set the true rotation rate on that axis

    # stop_rotation: sets all rotation rates back to zero
    # Input:  None
    # Return: None
    def stop_rotation(self):
        self.true_gyro = [0.0, 0.0, 0.0]  # reset all three axes to zero rotation

    # _run: background thread loop that generates frames at the correct output rate
    # Runs continuously until stop() is called
    # Input:  None
    # Return: None
    def _run(self):
        while self._running:                            # keep running until stop() is called
            mode = OPERATING_MODES[self.operating_mode]  # get current mode settings
            rate = mode['output_rate']                  # get output rate in Hz

            if rate == 0:           # if Idle mode, no frames to generate
                time.sleep(0.1)     # sleep briefly and check again
                continue            # skip frame generation

            dt = 1.0 / rate                     # calculate time step from output rate
            frame = self._build_raw_frame(dt)   # build a new frame

            with self._lock:                    # acquire lock before writing the frame
                self._latest_frame = frame      # store the new frame for the GUI to read

            time.sleep(dt)  # wait one time step before generating the next frame

    # _build_raw_frame: builds one complete Raw Operating Frame ready to send
    # Follows the byte layout from Table 7 of the MK2M2 datasheet REV08
    # Applies COBS encoding from Section 10.5.1 and CRC-16 from Section 10.5.2
    # Input:  dt (float) - time step in seconds, used for noise model and timestamp
    # Return: bytes - complete COBS-encoded frame with 0x00 delimiter appended
    def _build_raw_frame(self, dt):
        gx = self._gyro_noise[0].sample(self.true_gyro[0],  dt)  # sample noisy gyro X
        gy = self._gyro_noise[1].sample(self.true_gyro[1],  dt)  # sample noisy gyro Y
        gz = self._gyro_noise[2].sample(self.true_gyro[2],  dt)  # sample noisy gyro Z
        ax = self._accel_noise[0].sample(self.true_accel[0], dt)  # sample noisy accel X
        ay = self._accel_noise[1].sample(self.true_accel[1], dt)  # sample noisy accel Y
        az = self._accel_noise[2].sample(self.true_accel[2], dt)  # sample noisy accel Z

        self._roll  += gx * dt  # integrate gyro X to get roll angle in degrees
        self._pitch += gy * dt  # integrate gyro Y to get pitch angle in degrees
        self._yaw   += gz * dt  # integrate gyro Z to get yaw angle in degrees

        timestamp_ms = int((time.time() - self._start_time) * 1000)  # ms since power-up, Section 10.5.7
        counter = self._frame_counter % 256  # wrap frame counter at 255, Section 10.5.6
        self._frame_counter += 1             # increment frame counter for next frame

        mode_bits  = OPERATING_MODES[self.operating_mode]['binary']  # get 3-bit mode value
        frame_type = FRAME_TYPES['R']                                 # Raw frame type = 0b000
        mc_bits    = 0b01 if self.misalignment_correction else 0b00   # misalignment correction flag
        byte0 = (mc_bits << 6) | (mode_bits << 3) | frame_type  # pack three fields into byte 0

        temperature = 38.0 + random.gauss(0, 0.1)  # internal temp ~13C above ambient, Section 10.5.14
        status = 0x00  # 0x00 means all systems healthy, Section 10.5.21

        # pack the payload into bytes following Table 7 byte layout exactly
        # < means little endian, B = uint8, Q = uint64, f = float32
        # format: byte0, counter, timestamp, gyro xyz, accel xyz, incl xy, mag xyz, temp, status
        payload = struct.pack(
            '<BBQ6f2f3ffB',
            byte0,              # byte 0   - packed status, mode, frame type
            counter,            # byte 1   - frame counter 0 to 255
            timestamp_ms,       # bytes 2-9  - timestamp in ms as uint64
            gx, gy, gz,         # bytes 10-21 - gyro X Y Z as float32
            ax, ay, az,         # bytes 22-33 - accel X Y Z as float32
            self.true_incl[0],  # bytes 34-37 - inclinometer X as float32
            self.true_incl[1],  # bytes 38-41 - inclinometer Y as float32
            self.true_mag[0],   # bytes 42-45 - magnetometer X as float32
            self.true_mag[1],   # bytes 46-49 - magnetometer Y as float32
            self.true_mag[2],   # bytes 50-53 - magnetometer Z as float32
            temperature,        # bytes 54-57 - temperature as float32
            status              # byte 58   - status byte
        )

        crc = compute_crc16(payload)            # compute CRC over payload, Section 10.5.2
        crc_bytes = struct.pack('<H', crc)       # pack CRC as little-endian uint16
        raw_frame = payload + crc_bytes          # append CRC to payload, total 61 bytes

        return cobs.encode(raw_frame) + b'\x00'  # COBS encode and append 0x00 delimiter

    # build_startup_frame: builds the Startup Frame sent once on power-up
    # Contains the current unit configuration for the receiver to read
    # Follows Table 5 of the MK2M2 datasheet REV08
    # Input:  None
    # Return: bytes - complete COBS-encoded startup frame with 0x00 delimiter
    def build_startup_frame(self):
        mode_bits  = OPERATING_MODES[self.operating_mode]['binary']  # get 3-bit mode value
        mc_bits    = 0b01 if self.misalignment_correction else 0b00   # misalignment flag
        byte0 = (mc_bits << 6) | (mode_bits << 3) | FRAME_TYPES['S']  # pack byte 0

        mark     = b'OSCP-MK2M2'                    # bytes 1-10, fixed ASCII string, Table 5
        unit_num = struct.pack('<H', self.unit_number)  # bytes 11-12, unit serial number as uint16
        sw       = struct.pack('BBB', *self.sw_version)  # bytes 13-15, firmware version as 3 x uint8

        enabled_frames = 0b00000001  # byte 16, bit 0 set means Raw frame enabled, Table 15

        # dynamic range values encoded as 4-bit fields packed into one byte
        # accel range in bits 7:4, gyro range in bits 3:0, from Table 17 and Table 21
        gyro_dr_map  = {250: 0b0000, 4000: 0b0001, 125: 0b0010,
                        500: 0b0100, 1000: 0b1000, 2000: 0b1100}
        accel_dr_map = {2: 0b0000, 16: 0b0001, 4: 0b0010, 8: 0b0011}
        dr_byte = (accel_dr_map.get(self.accel_range, 0) << 4) | \
                   gyro_dr_map.get(self.gyro_range, 0)  # pack accel and gyro DR into one byte

        config_bytes = bytes(20)  # bytes 17-36, filter and AHRS config, default zeros
        status = 0x00             # byte 37, 0x00 means all systems healthy

        payload = (
            bytes([byte0]) +      # byte 0 - packed misalignment, mode, frame type
            mark +                # bytes 1-10 - mark number
            unit_num +            # bytes 11-12 - unit number
            sw +                  # bytes 13-15 - software version
            bytes([enabled_frames, dr_byte]) +  # bytes 16-17 - frame types and dynamic ranges
            config_bytes +        # bytes 17-36 - filter and AHRS configuration
            bytes([status])       # byte 37 - status
        )

        payload = payload[:38].ljust(38, b'\x00')  # trim or pad to exactly 38 bytes

        crc = compute_crc16(payload)            # compute CRC over the 38-byte payload
        crc_bytes = struct.pack('<H', crc)       # pack CRC as little-endian uint16
        raw_frame = payload + crc_bytes          # append CRC, total 40 bytes

        return cobs.encode(raw_frame) + b'\x00'  # COBS encode and append 0x00 delimiter

    # get_euler_angles: returns the current integrated orientation angles
    # Simple integration of gyro data, not drift corrected
    # Use fusion_ahrs.py for drift-corrected orientation
    # Input:  None
    # Return: tuple (roll, pitch, yaw) in degrees
    def get_euler_angles(self):
        return (self._roll, self._pitch, self._yaw)  # return integrated angles as tuple

    # get_latest_sensor_values: returns the most recent sensor readings as a dictionary
    # Faster than parsing a full frame, used by the GUI for live display
    # Input:  None
    # Return: dict with keys gyro_x/y/z, accel_x/y/z, roll, pitch, yaw, temp
    def get_latest_sensor_values(self):
        dt = 1.0 / (OPERATING_MODES[self.operating_mode].get('output_rate') or 100)  # get time step
        return {
            'gyro_x':  self._gyro_noise[0].sample(self.true_gyro[0],  dt),  # noisy gyro X
            'gyro_y':  self._gyro_noise[1].sample(self.true_gyro[1],  dt),  # noisy gyro Y
            'gyro_z':  self._gyro_noise[2].sample(self.true_gyro[2],  dt),  # noisy gyro Z
            'accel_x': self._accel_noise[0].sample(self.true_accel[0], dt),  # noisy accel X
            'accel_y': self._accel_noise[1].sample(self.true_accel[1], dt),  # noisy accel Y
            'accel_z': self._accel_noise[2].sample(self.true_accel[2], dt),  # noisy accel Z
            'roll':    self._roll,   # integrated roll angle in degrees
            'pitch':   self._pitch,  # integrated pitch angle in degrees
            'yaw':     self._yaw,    # integrated yaw angle in degrees
            'temp':    38.0 + random.gauss(0, 0.1),  # simulated internal temperature in C
        }