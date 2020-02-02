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
# Specify where files are located and needs to be saved
folder_main      = os.path.dirname(os.path.realpath(__file__)) # get directory where file is stored
folder_name_1    = 'dyna288_Bk10' # first plot's data
folder_name_2    = 'dyna288_Bk100' # second plot's data
folder_data      = 'spectFiles' # folder where visualisation is saved
bool_disp_header = bool(1)
## Specify which variables you want to plot
var_iter = 76 # time point (simulation time)
var_time = var_iter/t_eddy # time point (t_eddy: normalised by eddy-turnover time)
## Specify which variables you want to plot
var_x            = 1  # variable: wave number (k)
var_y            = 15 # variable: power spectrum
## Set the figure's axis limits
bool_set_limits  = bool(1)
xlim_min         = 1
xlim_max         = 1.3e+02
ylim_min         = 1.0e-16
ylim_max         = 4.2e-03
## Should the plot be saved?
bool_save_fig    = bool(0)

##################################################################
## FUNCTIONS
##################################################################
def meetsCondition(element):
    return bool(element.endswith('mags.dat') or element.endswith('vels.dat'))

def loadData(directory, name_file, var_x, var_y, bool_disp_header):
    '''
    (1) bool_disp_folder(?): 
        - display all files in the firectory that meets condition set by 'meetsCondition'
    (2) read the file that satisfies 'meetsCondition'
        - save the header names
        - save the data of the relevant variables (columns)
    (3) bool_disp_header(?): 
        - display all the header names in the file
    (4) return the header names and (x, y) data
    '''

    filedata     = open(directory + name_file).readlines() # load in data
    header       = filedata[5].split() # save the header
    data         = np.array([x.strip().split() for x in filedata[6:]]) # store all data. index: data[row, col]
    header_names = [name[4:] for name in header] # store only the header names
    if bool_disp_header:
        print('\nHeader names: for ' + name_file + '\n--------------')
        print('\n'.join(header)) # print all header names (with index)
    data_x = list(map(float, data[:, var_x]))
    data_y = list(map(float, data[:, var_y]))
    return header_names, data_x, data_y

##################################################################
## PLOTTING CODE
##################################################################
name_file_Re100 = 'Turb_hdf5_plt_cnt_' + '{0:04}'.format(var_iter) + '_spect_vels.dat' # magnetic file
name_file_Re50 = 'Turb_hdf5_plt_cnt_' + '{0:04}'.format(var_iter) + '_spect_vels.dat' # kinetic file
fig           = plt.figure(figsize=(10, 7), dpi=100)
ax            = fig.add_subplot()
## load and plot data sets
# Magnetic power spectrum
header_names_Re100, data_x_Re100, data_y_Re100 = loadData((folder_main + folder_name_1), 
        name_file_Re100, var_x, var_y, bool_disp_header)
line_Re100, = plt.plot(data_x_Re100, data_y_Re100, 'k', label=r'$\mathcal{P}_{Re=100}$')
plt.xlim(data_x_Re100[0], data_x_Re100[-1])
# Kinetic power spectrum
header_names_Re50, data_x_Re50, data_y_Re50 = loadData((folder_main + folder_name_2), 
        name_file_Re50, var_x, var_y, bool_disp_header)
line_Re50, = plt.plot(data_x_Re50, data_y_Re50, 'b', label=r'$\mathcal{P}_{Re=50}$')
plt.xlim(data_x_Re50[0], data_x_Re50[-1])

##################################################################
## LABEL and ADJUST PLOT
##################################################################
# add legend
ax.legend(loc='upper right', fontsize=17, frameon=False)
## major grid
ax.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.35)
## minor grid
ax.grid(which='minor', linestyle='--', linewidth='0.5', color='black', alpha=0.2)
# label plots
plt.xlabel(r'$k$',           fontsize=20)
plt.ylabel(r'$\mathcal{P}$', fontsize=20)
## scale axies
ax.set_xscale('log')
ax.set_yscale('log')
## set axis limits
if bool_set_limits:
    line_Re100.axes.set_xlim(xlim_min, xlim_max) # set x-axis limits
    line_Re50.axes.set_xlim(xlim_min, xlim_max)
    line_Re100.axes.set_ylim(ylim_min, ylim_max) # set y-axis limits
    line_Re50.axes.set_ylim(ylim_min, ylim_max)
## annote time (eddy tunrover-time)
ax.text(0.5, 0.975,
        r"$t/t_{\mathregular{eddy}} = $" + u"%0.1f"%var_time,
        fontsize=17,
        ha="center", va='top', transform=ax.transAxes)

##################################################################
## SAVE IMAGE
##################################################################
## save figure
if bool_save_fig:
        name_fig = (folder_main + '/spectra_compare=%0.1f'%var_time)
        name_fig = name_fig.replace(".", "p")
        plt.savefig(name_fig + '.png')
        print('\nFigure saved: ' + name_fig)
## display the plot
plt.show()

## END OF PROGRAM