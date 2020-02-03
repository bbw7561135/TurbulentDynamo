##################################################################
## MODULES
##################################################################
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from IPython import get_ipython

##################################################################
## PREPARE TERMINAL/WORKSPACE/CODE
#################################################################
os.system('clear')                  # clear terminal window
get_ipython().magic('reset -sf')    # clear workspace
plt.close('all')                    # close all pre-existing plots
mpl.style.use('classic')            # plot in classic style

##################################################################
## USER DEFINED VARIABLES
##################################################################
global t_eddy, bool_disp_header, var_y
t_eddy           = 5 # L/(2*Mach)
## specify where files are located and needs to be saved
folder_main      = os.path.dirname(os.path.realpath(__file__)) # get directory where file is stored
folder_name_1    = 'dyna288_Bk10' # first plot's data
folder_name_2    = 'dyna288_Bk100' # second plot's data
label_1          = r'$k_{B} = 10$'
label_2          = r'$k_{B} = 100$'
name_fig         = folder_main + '/turb_dyna288_'
bool_disp_header = bool(1) # display all the header names stored in Turb.dat
var_y            = 29 # y-axis variable: 6 (E_kin), 8 (rms_Mach), 29 (E_mag)
var_scale        = ''
## should the plot be saved?
bool_save_fig    = bool(0)

##################################################################
## AUTOMATIC ADJUSTMENTS
##################################################################
if var_y == 6:
    ## mach number
    label_y       = r'$E_{\nu}/E_{\nu 0}$'
    bool_norm_dat = bool(1)
    var_scale     = 'log'
elif var_y == 8:
    ## mach number
    label_y       = r'$\mathcal{M}$'
    bool_norm_dat = bool(0)
    var_scale     = 'linear'
elif var_y == 29:
    ## magnetic field
    label_y       = r'$E_{B}/E_{B 0}$'
    bool_norm_dat = bool(1)
    var_scale     = 'log'

##################################################################
## FUNCTIONS
##################################################################
def createFilePath(names):
    return ('/'.join(names) + '/')

def loadData(directory):
    global t_eddy, bool_disp_header, bool_norm_dat, var_x, var_y
    ## load data
    data_split = [x.split() for x in open(directory).readlines()]
    ## show the header file
    if bool_disp_header: print('\nHeader names: for ' + directory.split('/')[-2] + '/Turb.dat\n--------------\n' + '\n'.join(data_split[0]))
    ## save maximum number of columns in a row. less indicated rows stores a message.
    len_thresh = len(data_split[0]) # ignore extra lines (len < len_thresh) resulting from restarting the simulation
    ## save data
    data_x = []
    data_y = []
    for row in data_split[1:]:
        if len(row) == len_thresh:
            if ((row[var_x][0] == '#') or (row[var_y][0] == '#')):
                break
            data_x.append(float(row[var_x]) / t_eddy) # normalise time-domain
            data_y.append(float(row[var_y]))
    if bool_norm_dat:
        # y_data = var_y / var_y[1]
        data_y = [i / data_y[1] for i in data_y]
    ## return variables
    return [data_x, data_y, data_split[0][var_y][4:]]

##################################################################
## INITIALISING VARIABLES
##################################################################
directory_1 = createFilePath([folder_main, folder_name_1]) + 'Turb.dat'
directory_2 = createFilePath([folder_main, folder_name_2]) + 'Turb.dat'
## open figure
fig = plt.figure(figsize=(10, 7), dpi=100)
ax  = fig.add_subplot()
## define x-variable
global var_x
var_x   = 0
label_x = r'$t/t_{\mathregular{eddy}}$'

##################################################################
## LOADING DATA
##################################################################
data_x_1, data_y_1, var_name = loadData(directory_1)
data_x_2, data_y_2, var_name = loadData(directory_2)

##################################################################
## PLOTTING DATA
##################################################################
plt.plot(data_x_1, data_y_1, 'k', label=label_1)
plt.plot(data_x_2, data_y_2, 'b', label=label_2)

##################################################################
## LABEL and ADJUST PLOT
##################################################################
# add legend
ax.legend(loc='lower right', fontsize=17, frameon=False)
## major grid
ax.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.35)
## minor grid
ax.grid(which='minor', linestyle='--', linewidth='0.5', color='black', alpha=0.2)
## label plot
plt.xlabel(label_x, fontsize=20)
plt.ylabel(label_y, fontsize=20)
## scale y-axis
ax.set_yscale(var_scale)

##################################################################
## SAVE IMAGE
##################################################################
## save figure
if bool_save_fig:
    name_fig += var_name + '.png'
    plt.savefig(name_fig)
    print('\nFigure saved: ' + name_fig)
## display the plot
plt.show()

## END OF PROGRAM