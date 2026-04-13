import sys
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QFormLayout, QDoubleSpinBox, 
                             QPushButton, QGroupBox, QTabWidget, QTableWidget, 
                             QTableWidgetItem, QHeaderView)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import random

# Ensure your local files ping.py and utils.py are in the same folder
from ping import Ping, Transducer, Reflector, World

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100, projection=None):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111, projection=projection)
        super().__init__(self.fig)
        
        if projection == '3d':
            # This captures mouse movement to lock the Z-axis vertical
            self.mpl_connect('motion_notify_event', self.force_z_vertical)
            # This enables the "hover" coordinate display in the toolbar/statusbar
            self.axes.format_coord = lambda x, y: f"Pos: ({x:.2f}, {y:.2f})"

    def force_z_vertical(self, event):
        # button 1 is left-click (drag to rotate)
        if event.button == 1 and self.axes.name == '3d':
            # View_init(elevation, azimuth, roll)
            # We preserve current elev/azim but force roll to 0
            self.axes.view_init(elev=self.axes.elev, azim=self.axes.azim, roll=0)
            self.draw_idle()

class SimulationGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sonar Physics Simulator")
        self.resize(1300, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 1. INITIALIZE ALL VISUAL COMPONENTS FIRST
        self.tabs = QTabWidget()
        self.canvas_3d = MplCanvas(self, projection='3d')
        self.canvas_tx = MplCanvas(self)
        self.canvas_rx = MplCanvas(self)
        
        # 2. SETUP THE RIGHT PANEL (Layout)
        right_panel = QVBoxLayout()
        self.tabs.addTab(self.canvas_3d, "3D World (Initial)")
        self.tabs.addTab(self.canvas_tx, "Transmit Preview")
        self.tabs.addTab(self.canvas_rx, "Simulation Results (Echoes)")
        right_panel.addWidget(self.tabs)

        # 3. SETUP THE LEFT PANEL (Controls)
        left_panel = QVBoxLayout()
        
        # Ping Config
        ping_group = QGroupBox("1. Ping Parameters")
        ping_form = QFormLayout()
        self.freq_in = self.create_sb(200e3, 1e3, 1e6, 1e3, ping_form, "Freq (Hz)")
        self.cycl_in = self.create_sb(100, 1, 1000, 1, ping_form, "Cycles")
        self.amp_in = self.create_sb(200, 1, 1000, 10, ping_form, "Amplitude")
        ping_group.setLayout(ping_form)
        left_panel.addWidget(ping_group)

        # Transducer Config
        trans_group = QGroupBox("2. Transducer Velocity")
        trans_form = QFormLayout()
        self.vx_in = self.create_sb(1.0, -50, 50, 0.5, trans_form, "Vx (m/s)")
        trans_group.setLayout(trans_form)
        left_panel.addWidget(trans_group)

        # Reflector Table (This will now safely find canvas_3d)
        self.setup_ref_table(left_panel)

        self.run_btn = QPushButton("RUN SIMULATION")
        self.run_btn.clicked.connect(self.run_simulation)
        left_panel.addWidget(self.run_btn)
        left_panel.addStretch()

        # 4. ADD BOTH PANELS TO MAIN LAYOUT
        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(right_panel, 3)

        # 5. FINAL CONNECTIONS & INITIAL SYNC
        self.freq_in.valueChanged.connect(self.update_tx_preview)
        self.cycl_in.valueChanged.connect(self.update_tx_preview)
        self.update_tx_preview()
        self.sync_3d_from_table()

    def create_sb(self, val, min_v, max_v, step, layout, lbl):
        sb = QDoubleSpinBox()
        sb.setRange(min_v, max_v); sb.setSingleStep(step); sb.setValue(val)
        layout.addRow(lbl, sb)
        return sb

    def setup_ref_table(self, layout):
        group = QGroupBox("3. Reflectors")
        vbox = QVBoxLayout()
        self.ref_table = QTableWidget(0, 4)
        self.ref_table.setHorizontalHeaderLabels(["X", "Y", "Z", "Rad"])
        self.ref_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Initial data
        self.add_ref_row(40, 0, -100, 0.5)
        
        # Connect changes for real-time 3D updates
        self.ref_table.itemChanged.connect(self.sync_3d_from_table)
        
        vbox.addWidget(self.ref_table)
        btns = QHBoxLayout()
        add_b = QPushButton("+ Add"); rem_b = QPushButton("- Remove")
        add_b.clicked.connect(lambda: self.add_ref_row(20, 0, -50, 1.0))
        rem_b.clicked.connect(lambda: self.ref_table.removeRow(self.ref_table.currentRow()))
        btns.addWidget(add_b); btns.addWidget(rem_b)
        vbox.addLayout(btns)
        group.setLayout(vbox)
        layout.addWidget(group)

    def add_ref_row(self, x=0, y=0, z=-100, r=0.5):
        row_count = self.ref_table.rowCount()
        
        # Default starting values
        new_x, new_y, new_z, new_r = x, y, z, r
        
        # Logic: If a reflector exists, base the new one on the previous one
        if row_count > 0:
            try:
                prev_x = float(self.ref_table.item(row_count - 1, 0).text())
                prev_y = float(self.ref_table.item(row_count - 1, 1).text())
                prev_z = float(self.ref_table.item(row_count - 1, 2).text())
                
                # Maintain Z level
                new_z = prev_z
                
                # 5m offset in either X or Y at random
                if random.choice(['x', 'y']) == 'x':
                    new_x = prev_x + 5.0
                    new_y = prev_y
                else:
                    new_x = prev_x
                    new_y = prev_y + 5.0
            except (AttributeError, ValueError):
                pass # Fall back to defaults if previous row is malformed

        # Insert the row
        self.ref_table.insertRow(row_count)
        self.ref_table.blockSignals(True) # Prevent flickering
        for i, val in enumerate([new_x, new_y, new_z, new_r]):
            self.ref_table.setItem(row_count, i, QTableWidgetItem(str(val)))
        self.ref_table.blockSignals(False)
        
        # Trigger the 3D update
        self.sync_3d_from_table()

    def sync_3d_from_table(self):
        """Extracts table data and refreshes 3D plot."""
        reflectors = []
        for i in range(self.ref_table.rowCount()):
            try:
                vals = [float(self.ref_table.item(i, j).text()) for j in range(4)]
                reflectors.append(Reflector(np.array(vals[:3]), vals[3]))
            except (AttributeError, ValueError): continue
        
        t_temp = Transducer(position=np.array([0,0,0]))
        self.update_3d_view(reflectors, t_temp)

    def update_tx_preview(self):
        p = Ping(self.freq_in.value(), int(self.cycl_in.value()), self.amp_in.value())
        self.canvas_tx.axes.cla()
        self.canvas_tx.axes.plot(p.ping_time, p.signal, color='blue')
        self.canvas_tx.axes.set_title("Transmit Signal Preview")
        self.canvas_tx.draw()

    def update_3d_view(self, reflectors, transducer):
        ax = self.canvas_3d.axes
        ax.cla()
        
        # 1. Hide the axes and grid for a clean look
        ax.set_facecolor('lightblue') 
        self.canvas_3d.figure.set_facecolor('lightblue')
        ax.set_axis_off()

        if reflectors:
            min_z = min([r.position[2] for r in reflectors]) - 10
        else:
            min_z = -110

        # Create a simple 10x10 grid on the floor
        floor_size = 50 
        gx, gy = np.meshgrid(np.linspace(-floor_size, floor_size, 5), 
                            np.linspace(-floor_size, floor_size, 5))
        gz = np.full_like(gx, min_z)
        ax.plot_wireframe(gx, gy, gz, color='blue', alpha=0.1, linewidth=0.5)
        
        # 2. Plot Transducer
        p = transducer.initial_position
        ax.scatter(p[0], p[1], p[2], color='red', s=100, label='Transducer', depthshade=False)
        
        # 3. Plot Reflectors
        all_points = [p] # We'll use this to calculate equal bounds
        for ref in reflectors:
            u, v = np.mgrid[0:2*np.pi:15j, 0:np.pi:10j]
            x = ref.position[0] + ref.radius * np.cos(u) * np.sin(v)
            y = ref.position[1] + ref.radius * np.sin(u) * np.sin(v)
            z = ref.position[2] + ref.radius * np.cos(v)
            ax.plot_surface(x, y, z, color='gray', alpha=0.6, linewidth=0)
            all_points.append(ref.position)

        # 4. FORCE EQUAL AXES (The Manual Way)
        # Matplotlib's 'equal' aspect is buggy in 3D; this manual bound-setting is safer.
        pts = np.array(all_points)
        max_range = np.array([pts[:,0].max()-pts[:,0].min(), 
                            pts[:,1].max()-pts[:,1].min(), 
                            pts[:,2].max()-pts[:,2].min()]).max() / 2.0

        mid_x, mid_y, mid_z = pts.mean(axis=0)
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # 5. Lock orientation
        ax.view_init(elev=20, azim=-45, roll=0)
        self.canvas_3d.draw()

    def run_simulation(self):
        # 1. Setup Objects
        p = Ping(self.freq_in.value(), int(self.cycl_in.value()), self.amp_in.value())
        t = Transducer(np.array([0,0,0]), np.array([self.vx_in.value(), 0, 0]))
        
        reflectors = []
        for i in range(self.ref_table.rowCount()):
            try:
                vals = [float(self.ref_table.item(i, j).text()) for j in range(4)]
                reflectors.append(Reflector(np.array(vals[:3]), vals[3]))
            except: continue

        # 2. Compute
        world = World(t, reflectors)
        world.run_simulation(p)

        # 3. Update Plots
        self.canvas_rx.axes.cla()
        self.canvas_rx.axes.plot(t.receive_signal, color='orange')
        self.canvas_rx.axes.set_title("Received Signal (Simulation Output)")
        self.canvas_rx.draw()
        
        self.tabs.setCurrentIndex(2)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Style the app a bit
    app.setStyle("Fusion")
    win = SimulationGUI()
    win.show()
    sys.exit(app.exec())