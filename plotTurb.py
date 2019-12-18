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
## USER VARIABLES
##################################################################
t_eddy = 5 # L/(2*Mach)
## Specify where files are located and needs to be saved
folder_main      = os.path.dirname(os.path.realpath(__file__)) # get directory where file is stored
folder_sub_files = '/simDyna256/' # folder where data is located
folder_sub_vis   = '/simDyna256/visFiles/' # folder where visualisation is saved
bool_disp_folder = bool(0) # display all files that end with: name_file_*
## Specify which variables you want to plot
bool_disp_header = bool(1)
var_x            = 0
var_y            = 29 # 6 (E_kin), 8 (rms_Mach), 29 (E_mag)
var_label_x      = r'$t/t_{\mathregular{eddy}}$'
var_label_y      = r''
## Set the figure's axis limits
bool_set_lim = bool(0)
xlim_min     = 3
xlim_max     = 10
ylim_min     = 1.0e-16
ylim_max     = 4.2e-03
var_scale    = ''
## Extra plotting features
bool_auto_adj   = bool(1) # automatically adjust axis labels/scales and following variables
bool_norm_dat   = bool(0) # normalise y-axis data
bool_ave        = bool(0) # plot average of data over specified x-range
bool_regression = bool(0) # plot regression line for data over specified x-range
## Set x-range for regression and averaging
x_min = 0 # TODO: implement regression and averaging
x_max = 0
## Should the plot be saved?
bool_save_fig   = bool(0)

##################################################################
## AUTOMATIC ADJUSTMENTS
##################################################################
if bool_auto_adj:
    if var_y == 6:
        ## mach number
        var_label_y     = r'$E_{\mathregular{k}}/E_{\mathregular{k}0}$'
        bool_norm_dat   = bool(1)
        bool_ave        = bool(0)
        bool_regression = bool(1)
        var_scale       = 'log'
    elif var_y == 8:
        ## mach number
        var_label_y     = r'$\mathcal{M}$'
        bool_norm_dat   = bool(0)
        bool_ave        = bool(1)
        bool_regression = bool(0)
        var_scale       = 'linear'
    elif var_y == 29:
        ## magnetic field
        var_label_y     = r'$E_{\mathregular{m}}/E_{\mathregular{m}0}$'
        bool_norm_dat   = bool(1)
        bool_ave        = bool(0)
        bool_regression = bool(1)
        var_scale       = 'log'

##################################################################
## PRINT OUT/PLOTTING DATA
##################################################################
directory = folder_main + folder_sub_files + 'Turb.dat'
fig       = plt.figure(figsize=(10, 7), dpi=100)
ax        = fig.add_subplot()
## load data
data_split = [x.split() for x in open(directory).readlines()]
## show the header file
if bool_disp_header: print('\nHeader names: for Turb.dat\n--------------\n' + '\n'.join(data_split[0]))
## save maximum number of columns in a row. less indicated rows stores a message.
num_cols_thresh = len(data_split[0])
## save x-data
data_x = []
for row in data_split[1:]:
    if len(row) == num_cols_thresh:
        data_x.append(float(row[var_x]) / t_eddy)
## save y-data
data_y = []
for row in data_split[1:]:
    if len(row) == num_cols_thresh:
        data_y.append(float(row[var_y]))
## plot data
plt.plot(data_x, data_y, 'k')
## major grid
ax.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.35)
## minor grid
ax.grid(which='minor', linestyle='--', linewidth='0.5', color='black', alpha=0.2)
## set axis limits
if bool_set_lim:
    plt.xlim(xlim_min, xlim_max)
    plt.ylim(ylim_min, ylim_max)
## label plot
plt.xlabel(var_label_x, fontsize=20)
plt.ylabel(var_label_y, fontsize=20)
## scale y-axis
ax.set_yscale(var_scale)
## save figure
if bool_save_fig:
    name_fig = (folder_main + folder_sub_vis + 'turb_plot.png')
    plt.savefig(name_fig)
    print('\nFigure saved: ' + name_fig)
## display the plot
plt.show()
