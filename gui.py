# File Name:   gui.py
# Author:      Paurav Pathak
#
# Description: This file is the full configuration and monitoring GUI for
#              the MK2M2 digital twin. It is the customer-facing evaluation
#              tool that shows everything the sensor produces in real time.
#
#              It shows unit information read from the Startup Frame,
#              configuration controls to change operating mode and dynamic
#              ranges, live plots of gyroscope and accelerometer data on
#              all three axes, a CRC and frame health status bar, and
#              live Roll Pitch Yaw orientation derived from the Fusion
#              AHRS algorithm suggested by Thibaut Gervais at OSCP.
#
#              To achieve this we use tkinter for the window and controls,
#              matplotlib embedded in tkinter for the live plots, and the
#              FuncAnimation class to update plots every 100ms without
#              blocking the GUI thread. The orientation panel uses the
#              imufusion library configured with MK2M2 AHRS defaults from
#              Table 25 of the datasheet.
#
#              Keep in mind: all communication with the sensor goes through
#              the MK2M2 SDK class in mk2m2_sdk/imu.py. This GUI never
#              touches the simulator directly. Every configuration button
#              maps to a real ASCII command from Section 11 of the datasheet.
#              The Startup Frame panel is populated by parsing the actual
#              Startup Frame bytes from the simulator, not by hardcoding values.
#
# Run:         python gui.py

import sys                  # used to modify Python path for imports
import os                   # used to build file paths
import tkinter as tk        # main GUI framework, built into Python
from tkinter import ttk     # themed widgets for dropdowns
import matplotlib           # plotting library
matplotlib.use('TkAgg')     # use TkAgg backend so matplotlib works inside tkinter
import matplotlib.pyplot as plt                             # pyplot for creating figures
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # embeds matplotlib in tkinter
from matplotlib.animation import FuncAnimation             # animates live plot updates
import collections          # provides deque for rolling data buffers

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))  # add current folder to path
from mk2m2_sdk.imu import MK2M2     # import the SDK public API
from fusion_ahrs import FusionAHRS  # import the Fusion AHRS orientation module


# rolling buffer size, number of samples shown on the plot at once
BUFFER_SIZE = 200

# rolling buffers for live plot data, one deque per axis
# deque automatically discards oldest values when full
gyro_x_buf  = collections.deque([0.0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)  # gyro X history
gyro_y_buf  = collections.deque([0.0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)  # gyro Y history
gyro_z_buf  = collections.deque([0.0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)  # gyro Z history
accel_x_buf = collections.deque([0.0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)  # accel X history
accel_y_buf = collections.deque([0.0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)  # accel Y history
accel_z_buf = collections.deque([0.0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)  # accel Z history


# MK2M2GUI: the full configuration and monitoring GUI for the MK2M2 digital twin
# Builds all panels, runs live plot animation, and handles configuration commands
class MK2M2GUI:

    # __init__: sets up the GUI window and all internal objects
    # Input:  root (tk.Tk) - the main tkinter window object
    # Return: None
    def __init__(self, root):
        self.root = root                                                # store root window reference
        self.root.title("OSCP MK2M2 - IMU Configuration and Monitor") # set window title
        self.root.geometry("900x820")                                  # set initial window size
        self.root.configure(bg="#1e1e2e")                              # dark background colour

        self.imu  = MK2M2()                 # create SDK instance
        self.ahrs = FusionAHRS(sample_rate=100)  # create AHRS at 100Hz for Low speed mode
        self._connected = False             # track connection state
        self._anim = None                   # holds the matplotlib animation object

        self._build_ui()  # build all GUI panels and widgets

    # _build_ui: calls each panel builder in order to construct the full GUI
    # Input:  None
    # Return: None
    def _build_ui(self):
        self._build_header()        # title bar and connect button
        self._build_unit_info()     # startup frame unit information panel
        self._build_config_panel()  # configuration controls panel
        self._build_status_bar()    # CRC status and frame health bar
        self._build_plots()         # live gyro and accel plots
        self._build_orientation_panel()  # Fusion AHRS roll pitch yaw panel

    # _build_header: creates the top title bar with connect/disconnect button
    # Input:  None
    # Return: None
    def _build_header(self):
        header = tk.Frame(self.root, bg="#313244", pady=8)  # header frame with panel background
        header.pack(fill=tk.X)                               # stretch across full width

        tk.Label(
            header,
            text="OSCP MK2M2 - IMU Evaluation Tool",  # title text
            font=("Helvetica", 16, "bold"),
            bg="#313244", fg="#cdd6f4"
        ).pack(side=tk.LEFT, padx=16)  # align left with padding

        self.connect_btn = tk.Button(
            header,
            text="Connect",                 # initial button label
            font=("Helvetica", 11),
            bg="#a6e3a1", fg="#1e1e2e",     # green button initially
            width=12,
            command=self._toggle_connection  # call toggle on click
        )
        self.connect_btn.pack(side=tk.RIGHT, padx=16)  # align right with padding

    # _build_unit_info: creates the unit information panel populated from the Startup Frame
    # Shows unit number, firmware version, operating mode, dynamic ranges, misalignment status
    # Reference: MK2M2 datasheet Section 10.3, Table 5
    # Input:  None
    # Return: None
    def _build_unit_info(self):
        frame = tk.LabelFrame(
            self.root,
            text="Unit Information  (from Startup Frame)",  # panel title
            font=("Helvetica", 10, "bold"),
            bg="#1e1e2e", fg="#89b4fa",
            padx=10, pady=6
        )
        frame.pack(fill=tk.X, padx=12, pady=(8, 0))  # stretch across with padding

        labels = [
            ("Unit Number",    "unit_number"),   # unit serial number from startup frame
            ("Firmware",       "firmware"),      # software version from startup frame
            ("Operating Mode", "op_mode"),       # current mode from startup frame
            ("Gyro Range",     "gyro_range"),    # gyro dynamic range from startup frame
            ("Accel Range",    "accel_range"),   # accel dynamic range from startup frame
            ("Misalignment",   "misalignment"),  # misalignment correction status
        ]

        self._info_vars = {}  # dictionary to hold StringVar for each info field
        for i, (label_text, key) in enumerate(labels):  # loop through each field
            tk.Label(
                frame, text=label_text + ":",
                font=("Helvetica", 9),
                bg="#1e1e2e", fg="#a6adc8"
            ).grid(row=0, column=i*2, padx=(10, 2), sticky=tk.W)  # place label in grid

            var = tk.StringVar(value="--")  # create StringVar with placeholder
            tk.Label(
                frame, textvariable=var,
                font=("Helvetica", 9, "bold"),
                bg="#1e1e2e", fg="#cdd6f4"
            ).grid(row=0, column=i*2+1, padx=(0, 16), sticky=tk.W)  # place value next to label

            self._info_vars[key] = var  # store StringVar mapped to this field key

    # _build_config_panel: creates the configuration controls panel
    # Each control maps to a real command from Section 11 of the MK2M2 datasheet
    # Input:  None
    # Return: None
    def _build_config_panel(self):
        frame = tk.LabelFrame(
            self.root,
            text="Configuration",
            font=("Helvetica", 10, "bold"),
            bg="#1e1e2e", fg="#89b4fa",
            padx=10, pady=8
        )
        frame.pack(fill=tk.X, padx=12, pady=(8, 0))

        tk.Label(
            frame, text="Operating Mode:",
            font=("Helvetica", 9),
            bg="#1e1e2e", fg="#a6adc8"
        ).grid(row=0, column=0, padx=8, sticky=tk.W)

        self.mode_var = tk.StringVar(value="L")  # default mode is Low speed
        for i, (label, val) in enumerate([("Idle", "I"), ("Low 100Hz", "L"), ("Medium 500Hz", "M")]):
            tk.Radiobutton(
                frame, text=label, variable=self.mode_var, value=val,
                font=("Helvetica", 9),
                bg="#1e1e2e", fg="#cdd6f4",
                selectcolor="#313244",
                command=self._on_mode_change  # call handler when selection changes
            ).grid(row=0, column=i+1, padx=6)

        tk.Label(
            frame, text="Gyro Range (deg/s):",
            font=("Helvetica", 9),
            bg="#1e1e2e", fg="#a6adc8"
        ).grid(row=1, column=0, padx=8, pady=4, sticky=tk.W)

        self.gyro_range_var = tk.StringVar(value="250")  # default gyro range from Table 2
        gyro_menu = ttk.Combobox(
            frame, textvariable=self.gyro_range_var,
            values=["125", "250", "500", "1000", "2000", "4000"],  # valid ranges from Table 2
            width=8, state="readonly"
        )
        gyro_menu.grid(row=1, column=1, padx=6, sticky=tk.W)
        gyro_menu.bind("<<ComboboxSelected>>", self._on_gyro_range_change)  # bind change event

        tk.Label(
            frame, text="Accel Range (g):",
            font=("Helvetica", 9),
            bg="#1e1e2e", fg="#a6adc8"
        ).grid(row=1, column=2, padx=8, sticky=tk.W)

        self.accel_range_var = tk.StringVar(value="4")  # default accel range from Table 2
        accel_menu = ttk.Combobox(
            frame, textvariable=self.accel_range_var,
            values=["2", "4", "8", "16"],  # valid ranges from Table 2
            width=8, state="readonly"
        )
        accel_menu.grid(row=1, column=3, padx=6, sticky=tk.W)
        accel_menu.bind("<<ComboboxSelected>>", self._on_accel_range_change)  # bind change event

        self.misalign_var = tk.BooleanVar(value=False)  # misalignment correction off by default
        tk.Checkbutton(
            frame,
            text="Misalignment Correction",
            variable=self.misalign_var,
            font=("Helvetica", 9),
            bg="#1e1e2e", fg="#cdd6f4",
            selectcolor="#313244",
            command=self._on_misalignment_change  # call handler when toggled
        ).grid(row=1, column=4, padx=12)

        tk.Button(
            frame,
            text="Soft Reset",
            font=("Helvetica", 9),
            bg="#f38ba8", fg="#1e1e2e",
            width=10,
            command=self._on_reset  # call reset handler on click
        ).grid(row=1, column=5, padx=12)

    # _build_status_bar: creates the status bar showing connection, CRC, frame counter, temperature
    # Input:  None
    # Return: None
    def _build_status_bar(self):
        bar = tk.Frame(self.root, bg="#313244", pady=4)  # status bar frame
        bar.pack(fill=tk.X, padx=12, pady=(8, 0))

        self.status_label = tk.Label(
            bar, text="Status: Disconnected",
            font=("Helvetica", 9),
            bg="#313244", fg="#a6adc8"
        )
        self.status_label.pack(side=tk.LEFT, padx=10)  # connection status leftmost

        self.crc_label = tk.Label(
            bar, text="CRC: --",
            font=("Helvetica", 9, "bold"),
            bg="#313244", fg="#a6adc8"
        )
        self.crc_label.pack(side=tk.LEFT, padx=20)  # CRC status next to connection

        self.frame_counter_label = tk.Label(
            bar, text="Frame: --",
            font=("Helvetica", 9),
            bg="#313244", fg="#a6adc8"
        )
        self.frame_counter_label.pack(side=tk.LEFT, padx=20)  # frame counter

        self.temp_label = tk.Label(
            bar, text="Temp: --",
            font=("Helvetica", 9),
            bg="#313244", fg="#a6adc8"
        )
        self.temp_label.pack(side=tk.LEFT, padx=20)  # temperature reading

    # _build_plots: creates the live gyroscope and accelerometer plots
    # Uses matplotlib FuncAnimation to update lines every 100ms
    # Input:  None
    # Return: None
    def _build_plots(self):
        plot_frame = tk.Frame(self.root, bg="#1e1e2e")  # container frame for plots
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(8, 0))

        self.fig, (self.ax_gyro, self.ax_accel) = plt.subplots(
            2, 1, figsize=(9, 4), facecolor="#1e1e2e"  # two subplots stacked vertically
        )
        self.fig.tight_layout(pad=2.5)  # add spacing between subplots

        self.ax_gyro.set_facecolor("#181825")                               # dark plot background
        self.ax_gyro.set_title("Gyroscope (deg/sec)", color="#cdd6f4", fontsize=10)  # plot title
        self.ax_gyro.set_ylabel("deg/sec", color="#a6adc8", fontsize=8)     # Y axis label
        self.ax_gyro.tick_params(colors="#a6adc8", labelsize=7)             # tick colour and size
        for spine in self.ax_gyro.spines.values():
            spine.set_edgecolor("#45475a")  # set border colour for each spine
        self.ax_gyro.set_xlim(0, BUFFER_SIZE)   # X axis spans the full buffer
        self.ax_gyro.set_ylim(-5, 5)            # initial Y axis range in deg/sec

        self.line_gx, = self.ax_gyro.plot([], [], color="#f38ba8", label="X", linewidth=1)  # gyro X line red
        self.line_gy, = self.ax_gyro.plot([], [], color="#a6e3a1", label="Y", linewidth=1)  # gyro Y line green
        self.line_gz, = self.ax_gyro.plot([], [], color="#89b4fa", label="Z", linewidth=1)  # gyro Z line blue
        self.ax_gyro.legend(loc="upper right", fontsize=7, facecolor="#313244", labelcolor="#cdd6f4")

        self.ax_accel.set_facecolor("#181825")                              # dark plot background
        self.ax_accel.set_title("Accelerometer (g)", color="#cdd6f4", fontsize=10)  # plot title
        self.ax_accel.set_ylabel("g", color="#a6adc8", fontsize=8)          # Y axis label
        self.ax_accel.tick_params(colors="#a6adc8", labelsize=7)            # tick colour and size
        for spine in self.ax_accel.spines.values():
            spine.set_edgecolor("#45475a")  # set border colour for each spine
        self.ax_accel.set_xlim(0, BUFFER_SIZE)  # X axis spans the full buffer
        self.ax_accel.set_ylim(-2, 2)           # initial Y axis range in g

        self.line_ax, = self.ax_accel.plot([], [], color="#f38ba8", label="X", linewidth=1)  # accel X line red
        self.line_ay, = self.ax_accel.plot([], [], color="#a6e3a1", label="Y", linewidth=1)  # accel Y line green
        self.line_az, = self.ax_accel.plot([], [], color="#89b4fa", label="Z", linewidth=1)  # accel Z line blue
        self.ax_accel.legend(loc="upper right", fontsize=7, facecolor="#313244", labelcolor="#cdd6f4")

        canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)  # embed matplotlib figure in tkinter
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)   # stretch to fill available space
        self.canvas = canvas  # store canvas reference for animation

    # _build_orientation_panel: creates the Roll Pitch Yaw display from Fusion AHRS
    # Uses imufusion library configured with MK2M2 AHRS defaults from Table 25
    # Reference: MK2M2 datasheet Section 10.5.13, Table 25
    # Input:  None
    # Return: None
    def _build_orientation_panel(self):
        frame = tk.LabelFrame(
            self.root,
            text="Orientation  (Fusion AHRS - Roll / Pitch / Yaw)",  # panel title
            font=("Helvetica", 10, "bold"),
            bg="#1e1e2e", fg="#89b4fa",
            padx=10, pady=8
        )
        frame.pack(fill=tk.X, padx=12, pady=(6, 8))

        self._orientation_vars = {}  # dictionary to hold StringVar for each angle
        fields = [
            ("Roll",  "roll",  "#f38ba8"),   # roll in red
            ("Pitch", "pitch", "#a6e3a1"),   # pitch in green
            ("Yaw",   "yaw",   "#89b4fa"),   # yaw in blue
        ]

        for i, (label, key, color) in enumerate(fields):  # loop through each angle
            tk.Label(
                frame, text=label,
                font=("Helvetica", 11, "bold"),
                bg="#1e1e2e", fg=color
            ).grid(row=0, column=i*2, padx=20, pady=4)  # place label in grid

            var = tk.StringVar(value="--")  # create StringVar with placeholder
            tk.Label(
                frame, textvariable=var,
                font=("Helvetica", 14, "bold"),
                bg="#1e1e2e", fg=color,
                width=10
            ).grid(row=1, column=i*2, padx=20)  # place value below label

            self._orientation_vars[key] = var  # store StringVar mapped to this angle

        tk.Label(
            frame, text="Quaternion (w, x, y, z)",
            font=("Helvetica", 9),
            bg="#1e1e2e", fg="#a6adc8"
        ).grid(row=0, column=6, padx=20)  # quaternion label

        self._quat_var = tk.StringVar(value="--")  # StringVar for quaternion display
        tk.Label(
            frame, textvariable=self._quat_var,
            font=("Helvetica", 9),
            bg="#1e1e2e", fg="#cdd6f4"
        ).grid(row=1, column=6, padx=20)  # quaternion value below label

    # _toggle_connection: switches between connected and disconnected on button click
    # Input:  None
    # Return: None
    def _toggle_connection(self):
        if not self._connected:  # if not connected, connect
            self._connect()
        else:                    # if connected, disconnect
            self._disconnect()

    # _connect: starts the IMU, loads startup info, and starts live animation
    # Input:  None
    # Return: None
    def _connect(self):
        self.imu.connect()                                              # start simulator via SDK
        self.ahrs.reset()                                               # reset AHRS to fresh state
        self._connected = True                                          # mark as connected
        self.connect_btn.config(text="Disconnect", bg="#f38ba8")       # change to red disconnect button
        self.status_label.config(text="Status: Connected (simulation)", fg="#a6e3a1")  # green status
        self._load_startup_info()                                       # populate unit info panel
        self._start_animation()                                         # start live plot animation

    # _disconnect: stops animation and disconnects from IMU
    # Input:  None
    # Return: None
    def _disconnect(self):
        if self._anim:                      # if animation is running
            self._anim.event_source.stop()  # stop the animation
        self.imu.disconnect()               # stop simulator via SDK
        self._connected = False             # mark as disconnected
        self.connect_btn.config(text="Connect", bg="#a6e3a1")          # restore green connect button
        self.status_label.config(text="Status: Disconnected", fg="#a6adc8")  # restore muted status

    # _load_startup_info: reads Startup Frame and populates the unit info panel
    # Parses real Startup Frame bytes from the simulator, not hardcoded values
    # Reference: MK2M2 datasheet Section 10.3, Table 5
    # Input:  None
    # Return: None
    def _load_startup_info(self):
        info = self.imu.get_startup_info()  # get parsed startup frame from SDK
        if info and info.get('crc_ok'):     # only populate if CRC passed
            mode_names      = {0: 'Idle', 1: 'Low', 2: 'Medium'}  # mode code to name mapping
            gyro_range_map  = {0: '+/-250', 1: '+/-4000', 2: '+/-125',
                               4: '+/-500', 8: '+/-1000', 12: '+/-2000'}  # gyro DR code to range
            accel_range_map = {0: '+/-2g', 1: '+/-16g', 2: '+/-4g', 3: '+/-8g'}  # accel DR code to range

            self._info_vars['unit_number'].set(str(info.get('unit_number', '--')))  # unit serial number
            self._info_vars['firmware'].set(
                f"{info.get('sw_major',0)}.{info.get('sw_minor',0)}.{info.get('sw_patch',0)}"  # version string
            )
            self._info_vars['op_mode'].set(mode_names.get(info.get('operating_mode', 0), '--'))  # mode name
            self._info_vars['gyro_range'].set(
                gyro_range_map.get(info.get('gyro_range_bits', 0), '--') + ' deg/s'  # gyro range string
            )
            self._info_vars['accel_range'].set(
                accel_range_map.get(info.get('accel_range_bits', 0), '--')  # accel range string
            )
            self._info_vars['misalignment'].set('On' if info.get('misalignment') else 'Off')  # correction state

    # _start_animation: starts the matplotlib FuncAnimation for live plot updates
    # Animation calls the update function every 100ms to refresh plot data
    # Input:  None
    # Return: None
    def _start_animation(self):
        x_data = list(range(BUFFER_SIZE))  # X axis data is just the buffer indices

        def update(frame):
            if not self._connected:  # stop updating if disconnected
                return

            data = self.imu.get_latest_sensor_values()  # get latest sensor values from SDK
            if not data:             # skip if no data available yet
                return

            gyro_x_buf.append(data['gyro_x'])   # push new gyro X into rolling buffer
            gyro_y_buf.append(data['gyro_y'])   # push new gyro Y into rolling buffer
            gyro_z_buf.append(data['gyro_z'])   # push new gyro Z into rolling buffer
            accel_x_buf.append(data['accel_x'])  # push new accel X into rolling buffer
            accel_y_buf.append(data['accel_y'])  # push new accel Y into rolling buffer
            accel_z_buf.append(data['accel_z'])  # push new accel Z into rolling buffer

            self.line_gx.set_data(x_data, list(gyro_x_buf))   # update gyro X plot line
            self.line_gy.set_data(x_data, list(gyro_y_buf))   # update gyro Y plot line
            self.line_gz.set_data(x_data, list(gyro_z_buf))   # update gyro Z plot line
            self.line_ax.set_data(x_data, list(accel_x_buf))  # update accel X plot line
            self.line_ay.set_data(x_data, list(accel_y_buf))  # update accel Y plot line
            self.line_az.set_data(x_data, list(accel_z_buf))  # update accel Z plot line

            all_gyro = list(gyro_x_buf) + list(gyro_y_buf) + list(gyro_z_buf)  # combine all gyro data
            g_min, g_max = min(all_gyro), max(all_gyro)                         # find min and max
            margin = max(0.5, (g_max - g_min) * 0.2)                           # add 20% margin
            self.ax_gyro.set_ylim(g_min - margin, g_max + margin)              # auto scale gyro Y axis

            self.ahrs.update(  # feed latest sensor data into Fusion AHRS
                gyro_xyz  = (data['gyro_x'],  data['gyro_y'],  data['gyro_z']),   # gyro in deg/sec
                accel_xyz = (data['accel_x'], data['accel_y'], data['accel_z']),  # accel in g
                mag_xyz   = (0.3, 0.0, 0.4)  # simulated magnetometer in uT
            )

            roll, pitch, yaw = self.ahrs.get_euler()   # get Fusion AHRS Euler angles
            self._orientation_vars['roll'].set(f"{roll:.2f} deg")    # update roll display
            self._orientation_vars['pitch'].set(f"{pitch:.2f} deg")  # update pitch display
            self._orientation_vars['yaw'].set(f"{yaw:.2f} deg")      # update yaw display

            w, x, y, z = self.ahrs.get_quaternion()  # get quaternion from Fusion AHRS
            self._quat_var.set(f"({w:.3f}, {x:.3f}, {y:.3f}, {z:.3f})")  # update quaternion display

            self.temp_label.config(text=f"Temp: {data['temp']:.1f} C")  # update temperature in status bar

            parsed = self.imu.read()   # read and decode a full frame for CRC status
            if parsed:                 # if a frame was returned
                if parsed.get('crc_ok'):   # CRC passed
                    self.crc_label.config(text="CRC: OK", fg="#a6e3a1")  # green CRC OK
                    self.frame_counter_label.config(
                        text=f"Frame: {parsed.get('frame_counter', '--')}"  # update frame counter
                    )
                else:                      # CRC failed
                    self.crc_label.config(text="CRC: FAIL", fg="#f38ba8")  # red CRC FAIL

        self._anim = FuncAnimation(
            self.fig, update,
            interval=100,           # call update every 100ms
            blit=False,             # full redraw each frame, simpler than blitting
            cache_frame_data=False  # do not cache frames, always use fresh data
        )
        self.canvas.draw()  # initial draw to display the empty plots

    # _on_mode_change: handles operating mode radio button change
    # Sends mode change to SDK and refreshes startup info panel
    # In real hardware sends: CONFIG then OM<mode> then EXIT, Section 11.6
    # Input:  None
    # Return: None
    def _on_mode_change(self):
        if self._connected:                         # only act if connected
            self.imu.set_mode(self.mode_var.get())  # send mode change via SDK
            self._load_startup_info()               # refresh unit info panel

    # _on_gyro_range_change: handles gyro range dropdown change
    # Sends range change to SDK and refreshes startup info panel
    # In real hardware sends: CONFIG then DRG<range> then EXIT, Section 11.8
    # Input:  event - tkinter combobox selection event, not used directly
    # Return: None
    def _on_gyro_range_change(self, event=None):
        if self._connected:                                          # only act if connected
            self.imu.set_gyro_range(int(self.gyro_range_var.get())) # send range change via SDK
            self._load_startup_info()                                # refresh unit info panel

    # _on_accel_range_change: handles accel range dropdown change
    # Sends range change to SDK and refreshes startup info panel
    # In real hardware sends: CONFIG then DRA<range> then EXIT, Section 11.9
    # Input:  event - tkinter combobox selection event, not used directly
    # Return: None
    def _on_accel_range_change(self, event=None):
        if self._connected:                                           # only act if connected
            self.imu.set_accel_range(int(self.accel_range_var.get())) # send range change via SDK
            self._load_startup_info()                                 # refresh unit info panel

    # _on_misalignment_change: handles misalignment correction checkbox toggle
    # In real hardware sends: CONFIG then EMCORR or DMCORR then EXIT, Section 11.11
    # Input:  None
    # Return: None
    def _on_misalignment_change(self):
        if self._connected:                                                   # only act if connected
            self.imu.set_misalignment_correction(self.misalign_var.get())    # send setting via SDK
            self._load_startup_info()                                         # refresh unit info panel

    # _on_reset: handles soft reset button click
    # Resets the IMU and AHRS and refreshes the unit info panel
    # In real hardware sends: RESET, Section 11.3
    # Input:  None
    # Return: None
    def _on_reset(self):
        if self._connected:             # only act if connected
            self.imu.reset()            # send reset via SDK
            self.ahrs.reset()           # reset AHRS to fresh state after IMU reset
            self._load_startup_info()   # refresh unit info panel
            self.status_label.config(text="Status: Reset complete", fg="#fab387")  # orange reset status


# entry point when file is run directly
if __name__ == "__main__":
    root = tk.Tk()          # create the main tkinter window
    app = MK2M2GUI(root)    # create the full GUI instance
    root.mainloop()         # start the tkinter event loop, blocks until window is closed