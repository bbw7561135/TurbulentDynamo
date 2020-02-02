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
t_eddy = 10
# Specify where files are located and needs to be saved
folder_main      = os.path.dirname(os.path.realpath(__file__)) # get directory where file is stored
folder_name      = 'dyna288_Bk100' # folder where data is located
folder_data      = 'spectFiles' # folder where visualisation is saved
folder_vis       = 'visFiles' # folder where visualisation is saved
bool_disp_folder = bool(0) # display all files that end with: name_file_*
bool_disp_header = bool(1) # display all the header names stored in spectra data
## Specify which variables you want to plot
var_iter         = 39 # time point (simulation time)
var_time         = var_iter/t_eddy # time point (t_eddy: normalised by eddy-turnover time)
fig_name         = folder_name + '_plot_spectra_ted=%0.1f'%var_time
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
def createFilePath(names):
    return ('/'.join([x for x in names if x != '']) + '/')

def meetsCondition(element):
    return bool(element.endswith('mags.dat') or element.endswith('vels.dat'))

def loadData(directory, name_file):
    '''
    bool_disp_folder(?): 
        - display all files in the directory that meets the condition set by 'meetsCondition'
    read the files that satisfies 'meetsCondition'
        - save the header names
        - save the data of the relevant variables (columns)
    bool_disp_header(?): 
        - display all the header names in the file
    return the header names and (x, y) data
    '''
    global var_x, var_y
    global bool_disp_folder, bool_disp_header

    if bool_disp_folder:
        print('\nFiles in directory:\n-------------------')
        print('\n'.join(sorted(filter(meetsCondition, os.listdir(directory)))))
    filedata     = open(directory + name_file).readlines() # load in data
    header       = filedata[5].split() # save the header
    data         = np.array([x.strip().split() for x in filedata[6:]]) # index: [row, col]
    header_names = [name[4:] for name in header] # store the header names only
    if bool_disp_header:
        print('\nHeader names: for ' + name_file + '\n--------------')
        print('\n'.join(header)) # print all header names (with index)
    data_x = list(map(float, data[:, var_x]))
    data_y = list(map(float, data[:, var_y]))
    return header_names, data_x, data_y

##################################################################
## PLOTTING CODE
##################################################################
name_file_mag = 'Turb_hdf5_plt_cnt_' + '{0:04}'.format(var_iter) + '_spect_mags.dat' # magnetic file
name_file_vel = 'Turb_hdf5_plt_cnt_' + '{0:04}'.format(var_iter) + '_spect_vels.dat' # kinetic file
directory     = createFilePath([folder_main, folder_name, folder_data])
fig           = plt.figure(figsize=(10, 7), dpi=100)
ax            = fig.add_subplot()
## load and plot data sets
# Magnetic power spectrum
header_names_mag, data_x_mag, data_y_mag = loadData(directory, name_file_mag)
line_mag, = plt.plot(data_x_mag, data_y_mag, 'k', label=r'$\mathcal{P}_{\mathregular{mag}}$')
plt.xlim(data_x_mag[0], data_x_mag[-1])
# Kinetic power spectrum
header_names_vel, data_x_vel, data_y_vel = loadData(directory, name_file_vel)
line_vel, = plt.plot(data_x_vel, data_y_vel, 'b', label=r'$\mathcal{P}_{\mathregular{kin}}$')
plt.xlim(data_x_vel[0], data_x_vel[-1])
# add legend
ax.legend(loc='upper right', fontsize=20, frameon=False)
## major grid
ax.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.35)
## minor grid
ax.grid(which='minor', linestyle='-', linewidth='0.5', color='black', alpha=0.2)
## label plot
plt.xlabel(r'$k$',           fontsize=20)
plt.ylabel(r'$\mathcal{P}$', fontsize=20)
## scale axies
ax.set_xscale('log')
ax.set_yscale('log')
## set axis limits
if bool_set_limits:
    line_mag.axes.set_xlim(xlim_min, xlim_max) # set x-axis limits
    line_vel.axes.set_xlim(xlim_min, xlim_max)
    line_mag.axes.set_ylim(ylim_min, ylim_max) # set y-axis limits
    line_vel.axes.set_ylim(ylim_min, ylim_max)
## annote time (eddy tunrover-time)
ax.text(0.5, 0.95,
        r"$t/t_{\mathregular{eddy}} = %0.1f$"%var_time,
        fontsize=20, color='black', 
        ha="center", va='top', transform=ax.transAxes)
## save figure
if bool_save_fig:
        fig_name = (createFilePath([folder_main, folder_name, folder_vis]) + fig_name)
        fig_name = (fig_name.replace(".", "p") + '.png')
        plt.savefig(fig_name)
        print('\nFigure saved: ' + fig_name)
## display the plot
plt.show()
