#!./bin/python3.9
#%%
#all library imports
import platform
import joblib
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import load_model
from sklearn.pipeline import Pipeline
import sys
import time
import h5py
import scipy.interpolate as interpolation
import serial.tools.list_ports


from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QSizePolicy
)
from PyQt5 import QtCore
from PyQt5.QtCore import (
    QThreadPool, 
    QRunnable,  
    pyqtSlot,
    Qt
)
import pyqtgraph as pg
from pyqtgraph import PlotWidget
from pyqtgraph.functions import interpolateArray
from serial import Serial
import numpy as np
import serial
import logging
import datetime as dt
import os
from PyQt5.QtGui import QFont
#%%
N_SENSORS = 11
TRANSMIT_BUFFER_SIZE = 2+2*N_SENSORS #initial A0 byte, ending C0 byte and each sensor has 2 bytes
CONN_STATUS = False #no serial communication with microcontoller initially
window_size = 1 #previous attempts were made with bigger windows: 10 -> 5 -> 4 -> 3 -> 1 with increasing performance for smaller windows
#the rationale behind using more frames in time was to make the model more robust (feed more adjacient heatmaps in time) and also sensitive to 
#dynamics (movements between postures). But sitting is mainly a static task so making a prediction per frame is more accurate, guarantees online 
#prediciton (for each time point) and maintains performance with respect to the best window size different from 1 (3) 
logging.basicConfig(format="%(message)s", level=logging.INFO)
class SerialWorker(QRunnable):
    """!
    @brief Main class for serial communication: handles connection with device.
    """
    def __init__(self, serial_port_name):
        """!
        @brief Init worker.
        """
        self.is_killed = True #initialize 
        super().__init__() 

        self.port = serial.Serial()
        
        #contemplate 2 cases for interoperability:
        #working on Windows operative systems 
        if (platform.system() == 'Windows'):
            self.port_name = str(serial_port_name)
        #or others (Mac or Linux)
        else:               
            self.port_name = "/dev/"+ str(serial_port_name)
        self.baudrate = 115200 # hard coded since fixed at PSoC level (top design)
        self.serial_ports = [
            p.name
            for p in serial.tools.list_ports.comports() #scan com ports
        ]

    @pyqtSlot()
    def run(self,is_killed=True):
        """!
        @brief Estabilish connection with desired serial port.
        """
        global CONN_STATUS
        self.is_killed=is_killed
        if not CONN_STATUS: #do if not connected
            try:
                self.port = serial.Serial(port=self.port_name, baudrate=self.baudrate,
                                        write_timeout=0, timeout=2) #timeout is to not get stuck in port for a longer time than specified when scanning them           
                if self.port.is_open:
                    CONN_STATUS = True
                    time.sleep(0.01)     #allow time for responce by suspending execution of current thread
            except serial.SerialException: #for robustness (notify of error in connection)
                print("Error with port {}.".format(self.port_name))
                time.sleep(0.01)


    @pyqtSlot()
    def send(self, char):
        """!
        @brief Basic function to send a single char on serial port.
        """
        try:
            self.port.write(char.encode('utf-8')) #used for device recognition (mutual)
            logging.info("Written {} on port {}.".format(char, self.port_name))
        except: #notify of error in writing char to connected port
            logging.info("Could not write {} on port {}.".format(char, self.port_name))

    @pyqtSlot()
    def killed(self):
        """!
        @brief Close the serial port before closing the app.
        """
        global CONN_STATUS
        self.is_killed=True
        if self.is_killed and CONN_STATUS:
            self.port.close() #close connection if it was running and if the port was opened
            time.sleep(0.01)
            CONN_STATUS = False #terminate connection with port
        logging.info("Killing the process")
    @pyqtSlot()
  

    def search_port(self):
        for i in range(len(self.serial_ports)):
            try: #maintain interoperability between OSs
                if (platform.system() == 'Windows'):
                    port=serial.Serial(str(self.serial_ports[i]),115200)
                else: 
                    port=serial.Serial("/dev/" + str(self.serial_ports[i]),115200)
                
                logging.info("Try to connect {}".format(self.serial_ports[i])) 
                port.write("v".encode('utf-8')) #send char by serial communication with the set baud rate (try doing so to each port until...)
                port.reset_output_buffer() #ready to send

                port.reset_input_buffer() #ready to receive
                
                                                        #...until one replies with this string
                if str(port.read(10))== "b'Posture$$$'": #at this point the PSoC has recognized the this device (v) and this device has recognized the PSoC (Posture$$$)
                    port.write("e".encode('utf-8'))  #send "e" to advise the PSoC that connection has been initialized
                    porta = self.serial_ports[i] 
                    port.close()
                    logging.info("Successfully connected")

                    break
                else:       #actually ctrl+C is the command to skip the port if it were to get stuck in it (rare issue for mac devices with devices connected to bluetooth)
                    enter=int(dt.datetime.now().strftime("%S"))        
                    if (int(dt.datetime.now().strftime("%S")) > enter+2): #deals with rare case in which it gets stuck in a port that does not repond as expected, forces to pass on (as timeout should already do automatically)
                        pass
            except: 
                logging.info("passed by{}".format(self.serial_ports[i])) #skip port that do not respond as expected
                pass
        return porta

###############
# MAIN WINDOW #
###############
class MainWindow(QMainWindow):
    def __init__(self):
        """!
        @brief Init MainWindow.
        """
        super(MainWindow, self).__init__()

        # title and geometry (window sizes)
        self.setWindowTitle("PSOCuscino")
        width = 200
        height = 160
        self.setMinimumSize(width, height)

        self.initUI()
   
    #####################
    # GRAPHIC INTERFACE #
    #####################
    def initUI(self):
        """!
        @brief Set up the graphical interface structure.
        """
        global CONN_STATUS
        #self.port = Serial("/dev/ttyACM0",115200)
        self.serialwork = SerialWorker(None)
        self.threadpool = QThreadPool() #manage threads
        self.connected = CONN_STATUS
        
        self.serialwork = SerialWorker(SerialWorker(None).search_port()) #call method of SerialWorker to connect to desired port
        self.threadpool.start(self.serialwork)
        
        # Create intuitive interface to control device functioning and visualize heatmap
        self.imageWidget = pg.GraphicsLayoutWidget()
        self.imageWidget.setBackground('#112233') #gray image widget background
    
        #define label for posture prediction
        self.posture_outcomes = QLabel() #predicted label for the posture (instantiation)
        self.posture_outcomes.setStyleSheet('QLabel {background-color: #112233; font-size: 20pt;font-weight: bold; color: red;}')
        self.posture_outcomes.setAlignment(Qt.AlignCenter)

        # Define buttons (for user interagtion with GUI: start data streaming, stop it, export file for specified user)
        self.start_btn = QPushButton(
            text="Start",
            clicked=lambda: [self.h5_btn.hide(),self.serialwork.run(is_killed=False),self.serialwork.send("b")] #click Start -> send b -> PSoC starts sampling, sending and calibrating
        )
        self.set_font(self.start_btn)

        self.stop_btn = QPushButton(
            text="Stop", 
            clicked=lambda: [self.h5_btn.show(),self.serialwork.send("s"),self.serialwork.killed()] #click Stop -> send s -> PSoC stops sampling and sending; and connection is killed
        )
        self.set_font(self.stop_btn)

        self.h5_btn = QPushButton(text="Export file",clicked=lambda: self.image_to_h5()) #export h5 file of all frames of heatmap collected from start to stop, save to predefined folder
        self.set_font(self.h5_btn)
        self.h5_btn.hide() #hide the "export file" button until stop is pressed (file ready to be exported)
        
        # layout
        button_hlay = QHBoxLayout() #order of buttons from left to right
        button_hlay.addWidget(self.start_btn) 
        button_hlay.addWidget(self.stop_btn)
        button_hlay.addWidget(self.h5_btn)
        textbox_hlay = QHBoxLayout()
        textbox_hlay.addWidget(self.posture_outcomes)
        vlay = QVBoxLayout() #buttons above, heatmap below
        vlay.addLayout(button_hlay)
        vlay.addWidget(self.imageWidget)
        vlay.addLayout(textbox_hlay)
        widget = QWidget()
        widget.setLayout(vlay)
        self.setCentralWidget(widget)
        
        #Dark widget color 
        widget.setAutoFillBackground(True)
        palette = widget.palette()
        palette.setColor(widget.backgroundRole(),QtCore.Qt.black)
        widget.setPalette(palette)

        #Load pipeline: the pipeling already accounts for data transformations to be applyed to new data that were fit on the 
        # training data: MinMax scaler [-1;1]; and also the load keras model
        folder_name = "../pipeline/posture_modelTue_18-01-2022_12_17_20" #path to folder with the trained model, scaler, classes
        self.pipe= self.load_pipeline("scaler.pkl", "classes.pkl", 'model.h5', folder_name) 
        
        #Receive sampled data from Nsensors=11
        #create grid and set the position of sensors and padding (1 to 1 representation of real sensor distribution on chair)
        self.x=np.linspace(0,44,45)  #chair width is 44cm : with 45 pixels, resolution is of 1cm
        self.y=np.linspace(0,40,41)  #chair length (from back to front) is 40cm : with 41 pixels, resolution is of 1cm 
        self.X,self.Y=np.meshgrid(self.x,self.y)
        
        #position in cm of sensors first and then padding; each from bottom (behind) left  to  top (front) right
        #position on x axis (horizontal)
        self.px= np.array([10,18.5,25.5,34,9.5,16,28,34.5,10.5,22.5,33.5,0,10,18.5,21.5,25.5,34,44,0,21.5,44,0,22,44,0,18.4,26,44,0,10,18.5,21.5,25.5,34,44])
        #position on y axis (vertical)
        self.py= np.array([12.5,10.5,10.5,12.5,19,18.5,18.5,19,27.5,31.5,27.5,0,0,0,0,0,0,0,11.5,11.5,11.5,19,19,19,29,29,29,29,40,40,40,40,40,40,40])

            
        self.memory=np.empty((0,41,45),int) #initialize collection of frames in time for offline model training (using acquired files from different users)
        self.Fsr=np.ndarray(shape=(N_SENSORS)) #initialize array for sensors' 2 bytes values
        self.timer = QtCore.QTimer(self) 
        self.timer.setInterval(300) # corresponds to time between 2 successive transmissions of Nsensors at a time
        self.timer.start()
        self.timer.timeout.connect(self.onNewData) #call every 300ms (new data received)
        self.image = pg.ImageItem()
        self.image.setOpts(axisOrder='row-major') #to fill in the image: inner loop(from left to right), outer loop(from bottom to top)
        self.plot=pg.PlotItem()
        self.plot.addItem(self.image)
        self.imageWidget.addItem(self.plot)
        self.bar=pg.ColorBarItem(values=(0,65_535),colorMap=pg.colormap.get('magma')) #plot item with pixel values from 0 to 65535 to which "magma" color map is made correspond
        
    def onNewData(self):
        
        if not  self.serialwork.is_killed: #if connections is not killed (closed)
            self.serialwork.port.reset_input_buffer() #prepare for reception of new data: synchronizes reading with received signal (has fixed header and tail)
            self.data=self.serialwork.port.read(TRANSMIT_BUFFER_SIZE).hex()[2:-2] #serial communication of 2 byte per sensor (remove header and tail from decoding: [2:-2])
            for i in range (N_SENSORS): #decode data
                self.Fsr[i]=int(self.data[i*4:i*4+4],16) #4 hex numbers corrispond to 2 bytes, so for each sensor i take 4 hex numbers and convert in int from base 16
            
            #create padding: to improve upon 0 padding a uniform distribution of the sensors is modelled here (5x7 matrix like)
            #this is then fed to griddata which will interpolate the sensors and these values rather than zeros; this is a semplification
            #since only adjacient cells are considered and they are simply linearly combined; but heuristically griddata proved to work
            #better on these values (introducing non linear combinations of sensors based on the real vicinity) than even simpler zeroes
            #notice: border values were given greater dropping rates than central ones
            self.pad=np.array([ int(self.Fsr[0]/3),
            int(0.7*self.Fsr[0]+0.3*self.Fsr[1]),
            int(0.3*self.Fsr[0]+0.7*self.Fsr[1]),
            int((self.Fsr[1]+self.Fsr[2])/3), 
            int(0.7*self.Fsr[2]+0.3*self.Fsr[3]),
            int(0.3*self.Fsr[2]+0.7*self.Fsr[3]),
            int(self.Fsr[3]/3),
            int(0.7*self.Fsr[0]+0.3*self.Fsr[4]),
            int((self.Fsr[2]+self.Fsr[1])/2),
            int(0.7*self.Fsr[3]+0.3*self.Fsr[7]),
            int(0.25*self.Fsr[0]+0.5*self.Fsr[4]+0.25*self.Fsr[8]),
            int((self.Fsr[5]+self.Fsr[6]+self.Fsr[9])/3),
            int(0.25*self.Fsr[3]+0.5*self.Fsr[7]+0.25*self.Fsr[10]),
            int(0.7*self.Fsr[8]+0.3*self.Fsr[4]),
            int((self.Fsr[8]+self.Fsr[9]+self.Fsr[5])/3),
            int((self.Fsr[9]+self.Fsr[10]+self.Fsr[8])/3),
            int(0.7*self.Fsr[10]+0.3*self.Fsr[7]),
            int(self.Fsr[8]/3),
            int(self.Fsr[8]/3),
            int((self.Fsr[8]+self.Fsr[9])/3),
            int(self.Fsr[9]/3),
            int((self.Fsr[9]+self.Fsr[10])/3),
            int(self.Fsr[10]/3),
            int(self.Fsr[10]/3)])
            
            self.im = np.concatenate((self.Fsr,self.pad/4)) #feed both sensors and custom padding to griddata
            
            #cubic (4th input) interpolation of sensors (and paddings) values: 1st input; based on real distance among them: 2nd input; with specified resolution, given by grid: 3rd input
            self.T=interpolation.griddata((self.px,self.py),self.im,(self.X,self.Y),method="cubic")  
            
            self.memory=np.append(self.memory,self.T.reshape(1,self.T.shape[0],self.T.shape[1]).astype(int),axis=0) #append each heatmap frame 
            self.tmp =self.memory
        
            self.padded_array = np. zeros((45, 49))
            self.padded_array[1:self.T.shape[0]+1,1:self.T.shape[1]+1] = self.T 
            self.image.setImage(self.padded_array)
            self.bar.setImageItem(self.image,insert_in=self.plot) #plot interpolated result
            
            #Posture net predict and show in a text box
            self.predict = self.pipe.predict(self.T.reshape(1,self.T.shape[0],self.T.shape[1]))  #before was 45,41
            if self.predict == 0:
                self.posture_outcomes.setText("Posture: CORRECT")
            elif self.predict == 1:
                self.posture_outcomes.setText("Posture: LEANING FORWARDS")
            elif self.predict == 2:
                self.posture_outcomes.setText("Posture: LEANING BACKWARDS")
            elif self.predict == 3:
                self.posture_outcomes.setText("Posture: LEANING RIGHT")
            else:
                self.posture_outcomes.setText("Posture: LEANING LEFT")
            
        
        else: #in case serialworker.is_killed == True
            
            self.memory = np.empty((0,41,45),int) #reset the memory for next aquisition 
    
    def ExitHandler(self): #otherwise threads stay open untill execution is terminated, uselessly using up CPU
        """!
        @brief Kill every possible running thread upon exiting application.
        """
        self.serialwork.killed()

    def image_to_h5(self): #when "export file" is clicked
        self.h5_btn.hide()

        #OS interoperability
        if (platform.system() == 'Windows'):
            path= "..\Data\RAW\\" + str(sys.argv[1])  #the user is specified when running the program from terminal; a userfriendly way to do so
                                                      #has not been implemented since this proces is only necessary to train the network; once this is done 
                                                      #we are only interested in online predictions (no need to save the file) 
        else: 
            path= "../Data/RAW/"+str(sys.argv[1])
        if not os.path.exists(path):  #creates folder in which data must be saved unless it already exists (1 folder per user); notice: more files per user are possible
            os.makedirs(path) 
        now = dt.datetime.now()
        hf = h5py.File(path +"/Aquisition_" + now.strftime("%a_%d-%m-%Y_%H_%M_%S")+".h5", 'w') #saves file in Data/RAW/<selected user> if you run from GUI_Python folder
        hf.create_dataset('image', data=self.tmp)
        hf.close()
        logging.info("save file to disk")
           
    def load_pipeline(self,scaler, classes, model, folder_name): #files provided to this function are provided from pipeline (saved from posture_net.py)
        if platform.system()=='Windows':
            scaler = joblib.load(folder_name + '\\' + scaler, 'r')
            build_model = lambda: load_model(folder_name + '\\' + model)
            classifier = KerasClassifier(build_fn = build_model) #for sklearn - keras interoperability
            classifier.classes_ = joblib.load(open(folder_name + '\\' + classes, 'rb')) #define the classes to predict
            classifier.model = build_model()
        else: #other OS
            scaler = joblib.load(folder_name + '/' + scaler, 'r')
            build_model = lambda: load_model(folder_name + '/' + model)
            classifier = KerasClassifier(build_fn = build_model, epochs = 10, batch_size = 100, verbose = 1)
            classifier.classes_ = joblib.load(open(folder_name + '/' + classes, 'rb'))
            classifier.model = build_model()
        return Pipeline([ ('scaler', scaler), ('clf', classifier)])
    
    def set_font(self,btn):
        btn.setAutoFillBackground(True)
        btn.setStyleSheet('QPushButton {background-color: #112233; font-size: 20pt; font-weight: bold; color: red;}')

#############
#  RUN APP  #
#############
if __name__ == '__main__':
    app = QApplication(sys.argv)
  
    w = MainWindow()
    app.aboutToQuit.connect(w.ExitHandler) #close all when closing the app
    w.show()
    
    sys.exit(app.exec_())


# %%
