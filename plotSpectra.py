# TODO: fix the time being plotted

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
folder_main      = os.path.dirname(os.path.realpath(__file__)) # get directory where file is stored
folder_sub_files = '/simDyna256/specFiles/' # folder where data is located
folder_sub_vis   = '/simDyna256/visFiles/' # folder where visualisation is saved
bool_disp_folder = bool(0) # display all files that end with: name_file_*
var_iter         = 20 # which file iter should be plotted?
var_time         = 20/5 # what is the time (t_eddy) that corresponds with the iter being plotted?
# Specify which variables you want to plot
bool_disp_header = bool(0)
var_x_mag        = 2
var_y_mag        = 15
var_x_vel        = 2
var_y_vel        = 15
# Axis domain limits
bool_def_limits  = bool(1)
xlim_min         = 1.5
xlim_max         = 1.3e+02
ylim_min         = 1.0e-13
ylim_max         = 4.2e-03
bool_save_fig    = bool(0)

##################################################################
## FUNCTIONS
##################################################################
def meetsCondition(element):
    return bool(element.endswith('mags.dat') or element.endswith('vels.dat'))

def loadData(directory, name_file, var_x, var_y, bool_disp_folder, bool_disp_header):
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
    if bool_disp_folder:
        print('\nFiles in directory:\n-------------------')
        print('\n'.join(sorted(filter(meetsCondition, os.listdir(directory)))))
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
name_file_mag = 'Turb_hdf5_plt_cnt_' + '{0:04}'.format(var_iter) + '_spect_mags.dat' # magnetic file
name_file_vel = 'Turb_hdf5_plt_cnt_' + '{0:04}'.format(var_iter) + '_spect_vels.dat' # kinetic file
directory     = folder_main + folder_sub_files
fig           = plt.figure(figsize=(10, 7), dpi=100)
ax            = fig.add_subplot()
## load and plot data sets
# Magnetic power spectrum
header_names_mag, data_x_mag, data_y_mag = loadData(directory, name_file_mag, var_x_mag, var_y_mag, bool_disp_folder, bool_disp_header)
line_mag, = plt.plot(data_x_mag, data_y_mag, 'k--', label=r'$\mathcal{P}_{\mathregular{mag}}$')
plt.xlim(data_x_mag[0], data_x_mag[-1])
# Kinetic power spectrum
header_names_vel, data_x_vel, data_y_vel = loadData(directory, name_file_vel, var_x_vel, var_y_vel, bool_disp_folder, bool_disp_header)
line_vel, = plt.plot(data_x_vel, data_y_vel, 'b--', label=r'$\mathcal{P}_{\mathregular{kin}}$')
plt.xlim(data_x_vel[0], data_x_vel[-1])
# add legend
ax.legend()
# label plots
plt.xlabel(r'$k$', fontsize=20)
plt.ylabel(r'$\mathcal{P}$', fontsize=20)
ax.set_xscale('log')
ax.set_yscale('log')
if bool_def_limits:
    line_mag.axes.set_xlim(xlim_min, xlim_max) # set x-axis limits
    line_vel.axes.set_xlim(xlim_min, xlim_max)
    line_mag.axes.set_ylim(ylim_min, ylim_max) # set y-axis limits
    line_vel.axes.set_ylim(ylim_min, ylim_max)
# add eddy tunrover-time
ax.text(2e-2, 3e-2, 
        r"$t/t_{\mathregular{eddy}}$ = " + u"%0.1f"%var_time,
        {'color': 'k', 'fontsize': 20, 'ha': 'left', 'va': 'bottom',
        'bbox': dict(boxstyle="round", fc="w", ec="k", pad=0.2)},
        transform=ax.transAxes)
# save figure
if bool_save_fig:
        name_fig = (folder_main + folder_sub_vis + 'spectra_plot_ted=%0.1f'%var_time)
        name_fig = name_fig.replace(".", "p")
        plt.savefig(name_fig + '.png')
        print('\nFigure saved: ' + name_fig)
# show plot
plt.show()
