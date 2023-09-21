#!./bin/python3.9
#%%
import serial
import numpy as np 
import matplotlib.pyplot as plt
import numpy

import pyqtgraph as pg
from pyqtgraph import PlotWidget




def update_image(new_data):
    
   plt.imshow(new_data, cmap = 'coolwarm', aspect = 'auto')


def update_line(hl, new_data):
   hl.set_xdata(numpy.append(hl.get_xdata(), new_data))
   plt.draw()
   plt.show()

TRANSMIT_BUFFER_SIZE = 4
N_SENSORS= 1
def main():
   serial_port = serial.Serial("/dev/ttyACM0",9600)
   Fsr = np.ndarray(shape=(N_SENSORS))
   serial_port.reset_input_buffer()
    # .read() returns bytes right?no str
   hl, = plt.plot([], [])
   
   while True:
      serial_port.reset_input_buffer()
      data= serial_port.read(TRANSMIT_BUFFER_SIZE).hex()[2:-2]
      #data = int(data[0:4],16)
      
      for i in range(N_SENSORS):
         Fsr[i] = int(data[i*4:i*4+4],16)
      '''put reshape when we  have 12 sensor'''
      #update_image(Fsr)
      #update_line(hl,Fsr)#da rivedere la salva ma non la fa ancora vedere
      print(Fsr.astype(int))
'''Run App'''
if __name__ == "__main__":
    
   main()


# %%
