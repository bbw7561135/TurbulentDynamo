##################################################################
## MODULES
##################################################################
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from IPython import get_ipython

##################################################################
## PREPARE TERMINAL/WORKSPACE/CODE
#################################################################
os.system('clear')                  # clear terminal window
get_ipython().magic('reset -sf')    # clear workspace
plt.close('all')                    # close all pre-existing plots
mpl.style.use('classic')            # plot in classic style

##################################################################
## FUNCTIONS
##################################################################
def createFilePath(names):
    return ('/'.join(names) + '/')

def loadData(directory):
    ## load data
    data_split = [x.split() for x in open(directory).readlines()]
    ## save maximum number of columns in a row. less indicated rows stores a message.
    len_thresh = len(data_split[0]) # ignore extra lines (len < len_thresh) resulting from restarting the simulation
    ## save x-data
    data_x = []
    for row in data_split[1:]:
        if len(row) == len_thresh:
            data_x.append(float(row[0]) / t_eddy) # normalise time-domain
    ## save kinetic data
    data_kin = []
    for row in data_split[1:]:
        if len(row) == len_thresh:
            data_kin.append(float(row[6]))
    ## save magnetic data
    data_mag = []
    for row in data_split[1:]:
        if len(row) == len_thresh:
            data_mag.append(float(row[29]))
    data_mag = [i / data_mag[1] for i in data_mag]
    return [data_x, data_kin, data_mag]

##################################################################
## USER DEFINED VARIABLES
##################################################################
t_eddy  = 5 # L/(2*Mach)
## specify where files are located and needs to be saved
folder_main = os.path.dirname(os.path.realpath(__file__)) # get directory where file is stored
folder_k10  = 'dyna128_Bk10' # folder where data is located
folder_k100 = 'dyna256_Bk100' # folder where data is located

##################################################################
## PRINT OUT/PLOTTING DATA
##################################################################
directory_k10 =  createFilePath([folder_main, folder_k10]) + 'Turb.dat'
directory_k100 = createFilePath([folder_main, folder_k100]) + 'Turb.dat'
## open figure
fig = plt.figure(figsize=(10, 7), dpi=100)
ax  = fig.add_subplot()
## load data
data_x_k10, data_kin_k10, data_mag_k10 = loadData(directory_k10)
data_x_k100, data_kin_k100, data_mag_k100 = loadData(directory_k100)
## plot data
plt.plot(data_x_k10, data_kin_k10, 
        color='black', linewidth=2, label=r'$k=10$ $E_\mathregular{kin}/E_{\mathregular{kin}0}$')
plt.plot(data_x_k10, data_mag_k10, 
        color='black', linewidth=2, label=r'$k=10$ $E_\mathregular{mag}/E_{\mathregular{mag}0}$')
plt.plot(data_x_k100, data_kin_k100, 
        color='gray', linewidth=2, label=r'$k=100$ $E_\mathregular{kin}/E_{\mathregular{kin}0}$', linestyle='--')
plt.plot(data_x_k100, data_mag_k100, 
        color='gray', linewidth=2, label=r'$k=100$ $E_\mathregular{mag}/E_{\mathregular{mag}0}$')
## include legend
plt.legend(loc="lower right", frameon=False)
## adjust plot axis
plt.ylim(1e-9, 1e4)
## scale y-axis
ax.set_yscale('log')
## label plot
plt.xlabel(r'$t/t_{\mathregular{eddy}}$', fontsize=20)
plt.ylabel(r'$E/E_{0}$', fontsize=20)
## major grid
ax.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.35)
## minor grid
ax.grid(which='minor', linestyle='--', linewidth='0.5', color='black', alpha=0.2)
## display the plot
plt.show()
