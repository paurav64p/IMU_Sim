# File Name:   parser.py
# Author:      OSCP
#
# Description: This file handles decoding and parsing of MK2M2 binary frames.
#              It is the receiving side of the digital twin pipeline.
#              Every frame that comes in goes through three steps in order:
#              COBS decode to remove the encoding, CRC verify to confirm
#              the data arrived intact, and struct unpack to split the bytes
#              into labeled sensor values that the rest of the code can use.
#
#              To achieve this we reverse exactly what simulator.py did.
#              The struct format string here mirrors Table 7 of the datasheet
#              the same way the simulator used it to build the frame.
#
#              Keep in mind: this file does not know or care whether the bytes
#              came from the simulator or from real hardware over RS422.
#              It just processes bytes. That is the point of separating it.
#              If CRC verification fails the frame is discarded immediately
#              and None is returned. Corrupted data is never passed further.

import struct           # unpacks binary bytes back into Python values
from cobs import cobs   # decodes COBS-encoded frames
from simulator import compute_crc16  # reuse the same CRC function from simulator


# Frame type identifiers extracted from bits 2:0 of byte 0
# Matches Table 6 in the MK2M2 datasheet REV08
FRAME_TYPE_RAW     = 0b000  # Raw Operating Frame with all sensor data
FRAME_TYPE_EULER   = 0b001  # Euler Angles Frame with roll pitch yaw
FRAME_TYPE_QUAT    = 0b010  # Quaternion Frame with orientation quaternion
FRAME_TYPE_STARTUP = 0b111  # Startup Frame with unit configuration


# cobs_decode_frame: removes COBS encoding from a received frame
# Input should be everything received up to but not including the 0x00 delimiter
# If decoding fails the frame was corrupted so we return None instead of crashing
# Reference: MK2M2 datasheet Section 10.5.1
# Input:  encoded_bytes (bytes) - the raw received bytes without the 0x00 delimiter
# Return: bytes or None - decoded raw frame bytes, None if decoding failed
def cobs_decode_frame(encoded_bytes):
    try:
        return cobs.decode(encoded_bytes)  # decode and return the raw bytes
    except Exception:
        return None  # frame was corrupted, return None so caller can discard it


# verify_crc: checks the CRC-16 of a decoded frame to confirm data integrity
# The last 2 bytes of every frame are the CRC value attached by the sender
# We compute the CRC fresh over all bytes except the last two and compare
# Reference: MK2M2 datasheet Section 10.5.2
# Input:  raw_frame (bytes) - the fully decoded frame bytes including CRC at end
# Return: bool - True if CRC matches and data is intact, False if corrupted
def verify_crc(raw_frame):
    if len(raw_frame) < 3:              # frame too short to contain any data plus CRC
        return False                    # reject it immediately
    payload  = raw_frame[:-2]           # everything except the last 2 bytes is the payload
    received = struct.unpack('<H', raw_frame[-2:])[0]  # unpack last 2 bytes as little-endian uint16
    computed = compute_crc16(payload)   # compute CRC over the payload using same function as simulator
    return computed == received         # True if they match, False if corrupted


# parse_raw_frame: unpacks a Raw Operating Frame into a readable dictionary
# Follows the byte layout from Table 7 of the MK2M2 datasheet REV08 exactly
# Returns None if frame is too short, returns dict with crc_ok False if CRC fails
# Input:  raw_frame (bytes) - decoded frame bytes, should be 61 bytes total
# Return: dict or None - sensor values as labeled dictionary, None if frame too short
def parse_raw_frame(raw_frame):
    if len(raw_frame) < 61:     # Raw Operating Frame must be exactly 61 bytes, Table 4
        return None             # too short, discard

    if not verify_crc(raw_frame):  # check CRC before doing anything with the data
        return {
            'crc_ok': False,                        # flag that CRC failed
            'error': 'CRC mismatch, frame discarded'  # description for caller
        }

    byte0 = raw_frame[0]                            # byte 0 contains three packed fields
    frame_type     = byte0 & 0b00000111             # bits 2:0 are the frame type
    operating_mode = (byte0 >> 3) & 0b00000111      # bits 5:3 are the operating mode
    misalignment   = (byte0 >> 6) & 0b00000011      # bits 7:6 are misalignment correction status

    # unpack remaining fields from byte 1 onwards using Table 7 layout
    # offset=1 skips byte 0 which we already unpacked manually above
    # < = little endian, B = uint8, Q = uint64, f = float32
    (
        frame_counter,   # byte 1       - frame counter 0 to 255
        timestamp_ms,    # bytes 2-9    - timestamp in ms as uint64
        gyro_x,          # bytes 10-13  - gyro X in deg/sec as float32
        gyro_y,          # bytes 14-17  - gyro Y in deg/sec as float32
        gyro_z,          # bytes 18-21  - gyro Z in deg/sec as float32
        accel_x,         # bytes 22-25  - accel X in g as float32
        accel_y,         # bytes 26-29  - accel Y in g as float32
        accel_z,         # bytes 30-33  - accel Z in g as float32
        incl_x,          # bytes 34-37  - inclinometer X in mg as float32
        incl_y,          # bytes 38-41  - inclinometer Y in mg as float32
        mag_x,           # bytes 42-45  - magnetometer X in uT as float32
        mag_y,           # bytes 46-49  - magnetometer Y in uT as float32
        mag_z,           # bytes 50-53  - magnetometer Z in uT as float32
        temperature,     # bytes 54-57  - internal temperature in C as float32
        status           # byte 58      - status byte, 0x00 means healthy
    ) = struct.unpack_from('<BQ6f2f3ffB', raw_frame, offset=1)

    return {
        'frame_type':     frame_type,      # 0 for Raw frame
        'operating_mode': operating_mode,  # 0=Idle, 1=Low, 2=Medium
        'misalignment':   misalignment,    # 0=off, 1=on
        'frame_counter':  frame_counter,   # 0 to 255 wrapping
        'timestamp_ms':   timestamp_ms,    # ms since power-up
        'gyro_x':         gyro_x,          # deg/sec
        'gyro_y':         gyro_y,          # deg/sec
        'gyro_z':         gyro_z,          # deg/sec
        'accel_x':        accel_x,         # g
        'accel_y':        accel_y,         # g
        'accel_z':        accel_z,         # g
        'incl_x':         incl_x,          # mg
        'incl_y':         incl_y,          # mg
        'mag_x':          mag_x,           # uT
        'mag_y':          mag_y,           # uT
        'mag_z':          mag_z,           # uT
        'temperature':    temperature,     # degrees C
        'status':         status,          # 0x00 = all good
        'crc_ok':         True,            # CRC passed
    }


# parse_startup_frame: unpacks a Startup Frame into a readable dictionary
# The Startup Frame contains the current unit configuration
# Follows Table 5 of the MK2M2 datasheet REV08
# Input:  raw_frame (bytes) - decoded frame bytes, should be 40 bytes total
# Return: dict or None - configuration fields as dictionary, None if too short
def parse_startup_frame(raw_frame):
    if len(raw_frame) < 40:         # Startup Frame must be exactly 40 bytes, Table 4
        return None                 # too short, discard

    if not verify_crc(raw_frame):   # verify CRC before reading any configuration
        return {'crc_ok': False, 'error': 'CRC mismatch'}

    byte0 = raw_frame[0]                            # byte 0 contains three packed fields
    frame_type     = byte0 & 0b00000111             # bits 2:0 are the frame type
    operating_mode = (byte0 >> 3) & 0b00000111      # bits 5:3 are the operating mode
    misalignment   = (byte0 >> 6) & 0b00000011      # bits 7:6 are misalignment correction status

    mark_number = raw_frame[1:11].decode('ascii', errors='replace').rstrip('\x00')  # bytes 1-10, ASCII mark number
    unit_number = struct.unpack_from('<H', raw_frame, offset=11)[0]  # bytes 11-12, unit serial number as uint16

    sw_major = raw_frame[13]    # byte 13, firmware major version
    sw_minor = raw_frame[14]    # byte 14, firmware minor version
    sw_patch = raw_frame[15]    # byte 15, firmware patch version

    enabled_frames   = raw_frame[16]        # byte 16, bitmask of enabled frame types
    dr_byte          = raw_frame[17]        # byte 17, dynamic ranges packed into one byte
    gyro_range_bits  = dr_byte & 0x0F       # bits 3:0 are the gyro dynamic range code
    accel_range_bits = (dr_byte >> 4) & 0x0F  # bits 7:4 are the accel dynamic range code

    status = raw_frame[37] if len(raw_frame) > 37 else 0x00  # byte 37, unit health status

    return {
        'frame_type':       frame_type,       # should be 7 for startup frame
        'operating_mode':   operating_mode,   # current mode
        'misalignment':     misalignment,     # correction on or off
        'mark_number':      mark_number,      # fixed string OSCP-MK2M2
        'unit_number':      unit_number,      # serial number
        'sw_major':         sw_major,         # firmware version major
        'sw_minor':         sw_minor,         # firmware version minor
        'sw_patch':         sw_patch,         # firmware version patch
        'enabled_frames':   enabled_frames,   # bitmask of enabled frame types
        'gyro_range_bits':  gyro_range_bits,  # encoded gyro DR from Table 17
        'accel_range_bits': accel_range_bits, # encoded accel DR from Table 21
        'status':           status,           # 0x00 = healthy
        'crc_ok':           True,             # CRC passed
    }


# decode_frame: top level function that handles any incoming encoded frame
# Runs the full pipeline: COBS decode, CRC verify, then route to correct parser
# Automatically detects the frame type from byte 0 and calls the right parser
# Input:  encoded_bytes (bytes) - raw bytes from transport, everything before 0x00
# Return: dict or None - parsed frame as dictionary, None if decoding fails entirely
def decode_frame(encoded_bytes):
    raw = cobs_decode_frame(encoded_bytes)  # step 1: remove COBS encoding
    if raw is None or len(raw) == 0:        # if decoding failed the frame was corrupted
        return None                         # discard it entirely

    frame_type = raw[0] & 0b00000111  # read frame type from bottom 3 bits of byte 0

    if frame_type == FRAME_TYPE_RAW:        # route to raw frame parser
        return parse_raw_frame(raw)
    elif frame_type == FRAME_TYPE_STARTUP:  # route to startup frame parser
        return parse_startup_frame(raw)
    else:
        return {                    # unsupported frame type, return basic info
            'frame_type': frame_type,   # the frame type code we received
            'raw':        raw,          # the raw bytes in case caller wants them
            'crc_ok':     verify_crc(raw)  # still verify CRC even if we cannot parse
        }