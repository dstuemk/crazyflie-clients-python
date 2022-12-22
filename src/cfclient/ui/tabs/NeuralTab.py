import pathlib
from signal import signal
import json
import time
import os
import datetime
import math
import csv

import logging
import numpy as np

from PyQt5 import uic
from PyQt5.QtCore import * 
from PyQt5 import QtWidgets

import cfclient, cflib
from cfclient.ui.tab import Tab

from cflib.crtp.crtpstack import CRTPPacket
from cflib.crtp.crtpstack import CRTPPort

__author__ = 'Team Phoenix'
__all__ = ['NeuralTab']

logger = logging.getLogger(__name__)

neural_tab_class = uic.loadUiType(cfclient.module_path +
                                   "/ui/tabs/neuralTab.ui")[0]


class Threaded(QObject):
    progress=pyqtSignal(int)
    data_arr=pyqtSignal(str)
    finished=pyqtSignal(str)

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)

    @pyqtSlot()
    def start(self): print("Thread started")
    
    @pyqtSlot(str)
    def transferParameters(self, data_in):
        data_in_obj = json.loads(data_in)
        policy_net = data_in_obj["policy_net"]
        line_buf = []
        for ln in policy_net.split("\n"):
            if len(ln.strip()) > 0:
                line_buf.append([
                    *np.fromstring(ln, sep=' ')  # Data in bytes
                ])
        for ln_idx in range(len(line_buf)):
            self.progress.emit(ln_idx*100 // len(line_buf))
            ln = line_buf[ln_idx]
            ln_chunks = 20
            for ln_start in range(0,len(ln)-1,ln_chunks):
                ln_sub = ln[ln_start:ln_start+ln_chunks]
                self.data_arr.emit(json.dumps([
                    4,       # Command transfer
                    *ln_sub  # Data
                ]))
                time.sleep(0.01)
        self.finished.emit("done")

class DroneStateBuilder:
    def __init__(self) -> None:
        self.data_keys = [
            "x", "y", "z",
            "x_dot", "y_dot", "z_dot",
            "roll", "pitch", "yaw",
            "roll_dot", "pitch_dot", "yaw_dot",
            "quaternion_a", "quaternion_b", "quaternion_c", "quaternion_d",
            "gyro_x", "gyro_y", "gyro_z",
            "acc_x", "acc_y", "acc_z",
            #"R1", "R2", "R3",
            #"R4", "R5", "R6",
            #"R7", "R8", "R9",
            "M1", "M2", "M3", "M4",
            "time"
        ]
        self.data_format = [*([2]*27)] # Size of each data point
        assert len(self.data_format) == len(self.data_keys), \
            f"data_format len. {len(self.data_format)} != data_keys len. {len(self.data_keys)}"
        self.bytes_per_packet = 29
        self.last_idx = -1
        self.buffer = [0]*(np.sum(self.data_format).astype("int32"))
        self.finished = False
    
    def update(self, data):
        data_idx = int(data[0])
        total_packets = math.ceil(np.sum(self.data_format) / self.bytes_per_packet)
        assert data_idx == (self.last_idx + 1) % total_packets, \
            (f"Invalid data index: {self.last_idx} -> {data_idx}")
        start_idx = data_idx*self.bytes_per_packet
        end_idx = (data_idx + 1)*self.bytes_per_packet
        self.buffer[start_idx:end_idx] = data[1:1+self.bytes_per_packet]
        self.last_idx = data_idx
        self.finished = data_idx == (total_packets - 1)
    
    def getDroneState(self):
        assert self.finished == True
        drone_state = {}
        data_idx = 0
        for data_key,data_size in zip(self.data_keys,self.data_format):
            data_raw = bytes(self.buffer[data_idx:data_idx+data_size])
            drone_state[data_key] = float(np.fromstring(data_raw, dtype=f"<f{data_size}"))
            data_idx = data_idx + data_size
        return drone_state

class NeuralTab(Tab, neural_tab_class):
    """Neural tab for controlling Crazyflie with neural networks"""
    _connected_signal = pyqtSignal(str)
    _disconnected_signal = pyqtSignal(str)
    _update = pyqtSignal(str)

    transferData=pyqtSignal(str)

    def __init__(self, tabWidget, helper, *args):
        super(NeuralTab, self).__init__(*args)
        self.setupUi(self)

        self.tabWidget = tabWidget
        self._helper = helper

        self.tabName = "Neural"
        self.menuName = "Neural"

        self._helper.cf.add_port_callback(0x0C, self._cf_drone_recv)

        self.control_state_names = [
            "ON_GROUND",
            "TAKE_OFF",
            "HOVERING",
            "EXECUTING",
            "TRANSFERING",
            "LANDING",
            "INIT",
            "ADJUSTING",
            "CIRCLING"
        ]
        self.current_control_state = -1
        self.current_drone_state = None

        self.droneStateBuilder = DroneStateBuilder()
        self.droneStateRecvCtr = 0

        self.button_on.clicked.connect(self.button_on_action)
        self.button_hover.clicked.connect(self.button_hover_action)
        self.button_land.clicked.connect(self.button_land_action)
        self.button_circle.clicked.connect(self.button_circle_action)
        self.button_neural_load.clicked.connect(self.button_load_action)
        self.button_neural_transfer.clicked.connect(self.button_transfer_action)
        self.button_neural_start.clicked.connect(self.button_start_net_action)
        self.button_adjust.clicked.connect(self.button_adjust_action)
        
        self.button_on.setEnabled(False)
        self.button_hover.setEnabled(True)
        self.button_land.setEnabled(False)
        self.button_circle.setEnabled(False)

        self.recordings = []
        self.button_save_record.clicked.connect(self.button_save_action)

        self.recordings_pid = []
        self.button_save_pid.clicked.connect(self.button_save_pid_action)

        timer = QTimer(self)
        timer.setSingleShot(False)
        timer.timeout.connect(self.updateControlStateInfo)
        timer.start(500)

        self._thread=QThread()
        self._threaded=Threaded()
        self._threaded.progress.connect(self.transfer_progress)
        self._threaded.finished.connect(self.finished_transfer)
        self._threaded.data_arr.connect(self.send_transfer_packet)
        self.transferData.connect(self._threaded.transferParameters)
        self._threaded.moveToThread(self._thread)
        self._thread.start()

    def button_circle_action(self):
        self.recordings_pid = []
        packet = CRTPPacket()
        packet.set_header(0x0C, 0)
        packet._set_data([7])
        self._helper.cf.send_packet(packet)
        # Automatically land after 20s
        timer = QTimer(self)
        timer.setSingleShot(True)
        timer.timeout.connect(lambda: self.button_land_action())
        timer.start(20000)

    def button_save_pid_action(self):
        if len(self.recordings_pid) == 0:
            print("Nothing recorded!")
            return
        timestamp = datetime.datetime.today()
        csv_filename,_ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save CSV File',
            f"pid__{timestamp.strftime('%Y_%m_%d__%H_%M_%S')}.csv")
        print(csv_filename) 
        if "" == csv_filename:
            print("No file selected")
            return    
        with open(csv_filename, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, self.recordings_pid[0].keys())
            dict_writer.writeheader()
            dict_writer.writerows(self.recordings_pid)        

    def button_save_action(self):
        if len(self.recordings) == 0:
            print("Nothing recorded!")
            return
        timestamp = datetime.datetime.today()
        csv_filename,_ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save CSV File',
            os.path.join(
                os.path.dirname(self.text_path.text()),
                f"flight__{timestamp.strftime('%Y_%m_%d__%H_%M_%S')}.csv"))
        print(csv_filename) 
        if "" == csv_filename:
            print("No file selected")
            return    
        with open(csv_filename, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, self.recordings[0].keys())
            dict_writer.writeheader()
            dict_writer.writerows(self.recordings)

    def button_adjust_action(self):
        adjust_ms = self.doubleSpinBox_adjust.value()
        adjust_sec = adjust_ms / 1000.0
        print(f"Set inference time: {adjust_sec} sec.")
        packet = CRTPPacket()
        packet.set_header(0x0C, 0)
        packet._set_data([5, *list(np.array(adjust_ms,dtype="<f2").tostring())])
        self._helper.cf.send_packet(packet)

    def button_transfer_action(self):
        sel_file = self.text_path.text()
        if "" == sel_file:
            QtWidgets.QMessageBox.about(self, "No File selected", "Please select model file")
            return
        with open(sel_file) as hndl:
            file_data = json.load(hndl)
            packet = CRTPPacket()
            packet.set_header(0x0C, 0)
            obs_model = 1 if file_data["observation_model"] == 'state' else 0
            packet._set_data([
                8,                                     # config command
                file_data["observation_history_size"], # History size
                obs_model ])                           # 0->sensor | 1->state
            self._helper.cf.send_packet(packet)
        self.transferData.emit(json.dumps({
            "policy_net": file_data['policy_net']
        }))
    
    @pyqtSlot(int)
    def transfer_progress(self, prog):
        print(f"Transfer progress {prog} %")

    @pyqtSlot(str)
    def send_transfer_packet(self, data_str):
        try:
            data_arr = json.loads(data_str)
            packet = CRTPPacket()
            packet.set_header(0x0C, 0)
            packet._set_data([int(v) for v in data_arr])
            self._helper.cf.send_packet(packet)
        except:
            print("exception:")
            print(data_arr)
                
    @pyqtSlot(str)
    def finished_transfer(self, data_str):
        packet = CRTPPacket()
        packet.set_header(0x0C, 0)
        packet._set_data([3])
        self._helper.cf.send_packet(packet)
        # Set inference time after transmitting
        self.button_adjust_action()

    def button_load_action(self):
        sel_file = QtWidgets.QFileDialog.getOpenFileName(self, "Select Model File")
        self.text_path.setText(sel_file[0])
            
    def button_on_action(self):
        packet = CRTPPacket()
        packet.set_header(0x0C, 0)
        packet._set_data([3])
        self._helper.cf.send_packet(packet)

    def button_hover_action(self):
        packet = CRTPPacket()
        packet.set_header(0x0C, 0)
        packet._set_data([0])
        self._helper.cf.send_packet(packet)

    def button_land_action(self):
        packet = CRTPPacket()
        packet.set_header(0x0C, 0)
        packet._set_data([1])
        self._helper.cf.send_packet(packet)
    
    def button_start_net_action(self):
        timer_ = QTimer(self)
        def timer_fn():
            self.recordings = []
            packet = CRTPPacket()
            packet.set_header(0x0C, 0)
            packet._set_data([2])
            self._helper.cf.send_packet(packet)
            # Automatically land if selected
            if self.checkBox_auto_land.isChecked():
                timer = QTimer(self)
                timer.setSingleShot(True)
                timer.timeout.connect(lambda: self.button_land_action())
                timer.start(self.doubleSpinBox_land.value()*1000)
        timer_.setSingleShot(True)
        timer_.timeout.connect(timer_fn)
        print("Start in 1 seconds...")
        timer_.start(1000)
    
    def updateControlStateInfo(self):
        self.text_drone_state.setText(
            self.control_state_names[self.current_control_state])
        
        if self.current_control_state == 0:
            # ON GROUND
            self.button_on.setEnabled(False)
            self.button_hover.setEnabled(True)
            self.button_land.setEnabled(False)
            self.button_circle.setEnabled(False)
            
        if self.current_control_state == 1:
            # TAKE OFF
            self.button_on.setEnabled(False)
            self.button_hover.setEnabled(False)
            self.button_land.setEnabled(False)
            self.button_circle.setEnabled(False)
            
        if self.current_control_state == 2:
            # HOVERING
            self.button_on.setEnabled(False)
            self.button_hover.setEnabled(False)
            self.button_land.setEnabled(True)
            self.button_circle.setEnabled(True)
            
        if self.current_control_state == 3:
            # EXECUTING
            self.button_on.setEnabled(False)
            self.button_hover.setEnabled(True)
            self.button_land.setEnabled(True)
            self.button_circle.setEnabled(False)

        if self.current_control_state == 4:
            # Transfering
            self.button_on.setEnabled(False)
            self.button_hover.setEnabled(False)
            self.button_land.setEnabled(False)
            self.button_circle.setEnabled(False)

        if self.current_control_state == 5:
            # LANDING
            self.button_on.setEnabled(False)
            self.button_hover.setEnabled(False)
            self.button_land.setEnabled(False)
            self.button_circle.setEnabled(False)
            
        if self.current_control_state == 6:
            # INIT
            self.button_on.setEnabled(True)
            self.button_hover.setEnabled(False)
            self.button_land.setEnabled(False)
            self.button_circle.setEnabled(False)

        if self.current_control_state == 7:
            # Adjusting
            self.button_on.setEnabled(False)
            self.button_hover.setEnabled(False)
            self.button_land.setEnabled(False)
            self.button_circle.setEnabled(False)

        if self.current_control_state == 8:
            # Circling
            self.button_on.setEnabled(False)
            self.button_hover.setEnabled(False)
            self.button_land.setEnabled(True)
            self.button_circle.setEnabled(False)
            
    def updateControlState(self, new_ctrl_state):
        self.current_control_state = new_ctrl_state
        
    def updateDroneState(self, new_drone_state):
        self.current_drone_state = new_drone_state
        self.droneStateRecvCtr = self.droneStateRecvCtr + 1
        # If flying circles with PID, then store state
        if self.current_control_state == 8:
            self.recordings_pid.append(new_drone_state)
        # If recording is enabled and drone is executing NN, then store state
        if self.checkBox_record.isChecked() and self.current_control_state == 3:
            self.recordings.append(new_drone_state)
        if self.droneStateRecvCtr % 100 == 0:
            print(new_drone_state)      
            pass  

    def _cf_drone_recv(self, packet):
        data = packet._data
        if data[0] == 0xFF:
            self.updateControlState(int(data[1]))            
        else:
            self.droneStateBuilder.update(data)
            if self.droneStateBuilder.finished:
                self.updateDroneState(self.droneStateBuilder.getDroneState())

        