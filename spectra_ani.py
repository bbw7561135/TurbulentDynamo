##################################################################
## MODULES
##################################################################
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

from matplotlib import animation
from IPython import get_ipython

##################################################################
## PREPARE TERMINAL/WORKSPACE/CODE
##################################################################
os.system('clear')                  # clear terminal window
get_ipython().magic('reset -sf')    # clear workspace
plt.close('all')                    # close all pr-existing plots
mpl.style.use('classic')            # plot in classic style
plt.rc('font', family='serif')      # specify font choice

##################################################################
## USER VARIABLES
##################################################################
t_eddy           = 10 # number of spectra files per eddy turnover
# Specify where files are located and needs to be saved
folder_main      = os.path.dirname(os.path.realpath(__file__)) # get directory_files where file is stored
folder_name      = 'dyna288_Bk10'
folder_files     = 'spectFiles' # folder where data is located
folder_vis       = 'visFiles' # folder where visualisation is saved
name_vid         = folder_name + '_ani_spectra'
bool_disp_folder = bool(0) # display all files in the directory_files
# Specify which variables you want to plot
var_x            = 1
var_y            = 15
## Choose whether you want to set the figure's axis limits
bool_disp_limits = bool(0) # if true then axies are set automatically
xlim_min         = 1.0
xlim_max         = 1.3e+02
ylim_min         = 1e-25
ylim_max         = 4.2e-03
## Should the animation be saved?
bool_save_ani    = bool(0)

##################################################################
## FUNCTIONS
##################################################################
def createFilePath(names):
    return ('/'.join(names) + '/')
    
def endsWithMags(element):
    return bool(element.endswith('mags.dat'))

def endsWithVels(element):
    return bool(element.endswith('vels.dat'))

def loadData(directory_files, name_file):
    global var_x, var_y
    filedata = open(directory_files + name_file).readlines() # read in data
    data     = np.array([x.strip().split() for x in filedata[6:]]) # store all data. index: data[row, col]
    data_x   = list(map(float, data[:, var_x]))
    data_y   = list(map(float, data[:, var_y]))
    return data_x, data_y

def updateIter():
    global file_min_num, file_max_num
    var_iter = -1
    while (var_iter < file_max_num - file_min_num):
        # update iter
        var_iter += 1
        yield var_iter

# http://www.roboticslab.ca/wp-content/uploads/2012/11/robotics_lab_animation_example.txt
def updateData(data):
    global directory_files, t_eddy
    global file_min_num, file_max_num
    global bool_print_progress, bool_disp_limits
    global file_names_mags, data_x_mag, data_y_mag, var_x, var_y
    global file_names_vels, data_x_vel, data_y_vel, var_x, var_y
    global var_xlim_min, var_xlim_max, var_ylim_min, var_ylim_max
    var_iter = data
    # load new data in
    data_x_mag, data_y_mag = loadData(directory_files, file_names_mags[var_iter])
    data_x_vel, data_y_vel = loadData(directory_files, file_names_vels[var_iter])
    # print to terminal (to show progress of save)
    if bool_print_progress:
        print("saving: %0.5f"%(100*var_iter/(file_max_num - file_min_num)) + '%% complete')
    # update data fields
    line_mag.set_data(data_x_mag, data_y_mag)
    line_vel.set_data(data_x_vel, data_y_vel)
    if (bool_disp_limits and (var_iter > 1)):
        line_mag.axes.set_xlim(min(data_x_mag), max(data_x_mag)) # set magnetic field x, y limits
        line_mag.axes.set_ylim(min(data_y_mag), max(data_y_mag))
        line_vel.axes.set_xlim(min(data_x_vel), max(data_x_vel)) # set velocity field x, y limits
        line_vel.axes.set_ylim(min(data_y_vel), max(data_y_vel))
        var_xlim_min = np.nanmin([var_xlim_min, min(data_x_mag), min(data_x_vel)])
        var_xlim_max = np.nanmax([var_xlim_max, max(data_x_mag), max(data_x_vel)])
        var_ylim_min = np.nanmin([var_ylim_min, min(data_y_mag), min(data_y_vel)])
        var_ylim_max = np.nanmax([var_ylim_max, max(data_y_mag), max(data_y_vel)])
    title.set_text(r"$t/t_{\mathregular{eddy}} = %0.1f$"%(var_iter/t_eddy))
    return line_mag, line_vel, title,

##################################################################
## PLOTTING CODE
##################################################################
## Set up animation writer
Writer = animation.writers['ffmpeg']
writer = Writer(fps=10)
## setup information for loading data
directory_files = createFilePath([folder_main, folder_name, folder_files])
file_names = sorted(os.listdir(directory_files))
file_min_mags = file_min_vels = 0
file_max_mags = file_max_vels = 0
## load magnetic field data
file_names_mags = list(filter(endsWithMags, file_names))
file_min_mags   = int(min(file_names_mags)[18:22]) 
file_max_mags   = int(max(file_names_mags)[18:22]) 
## load velocity field data
file_names_vels = list(filter(endsWithVels, file_names))
file_min_vels   = int(min(file_names_vels)[18:22]) 
file_max_vels   = int(max(file_names_vels)[18:22]) 
## print all relevant files to terminal
if bool_disp_folder:
    print('\nMagnetic files in directory_files:\n----------------------------')
    print('\n'.join(file_names_mags))
    print('\nVelocity files in directory_files:\n----------------------------')
    print('\n'.join(file_names_vels))
## store the minimum and maximum number of files
file_min_num = min(file_min_mags, file_min_vels) # TODO: update to use len()
file_max_num = max(file_max_mags, file_max_vels)
bool_print_progress = False
## initialise the plot and those features that will be updated by the animation
fig = plt.figure(figsize=(10, 7), dpi=100)
ax  = fig.add_axes([0.1, 0.1, 0.8, 0.8])
data_x_mag = data_y_mag = 1 # initialise the line coordinates
data_x_vel = data_y_vel = 1
line_mag, = plt.plot(data_x_mag, data_y_mag, 'k-', label=r'$\mathcal{P}_{\mathregular{mag}}$')
line_vel, = plt.plot(data_x_vel, data_y_vel, 'b-', label=r'$\mathcal{P}_{\mathregular{kin}}$')
## add legend
ax.legend([line_mag, line_vel], [line_mag.get_label(), line_vel.get_label()], 
            fontsize=20, frameon=False)
## major grid
ax.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.35)
## minor grid
ax.grid(which='minor', linestyle='-', linewidth='0.5', color='black', alpha=0.2)
## annotate time (eddy turnover-time)
title = ax.text(0.5, 0.95,
            r"$t/t_{\mathregular{eddy}} = %0.1f$"%0, 
            fontsize=20, color='black', 
            # path_effects=[path_effects.withSimplePatchShadow(offset=(1, -1))],
            ha="center", va='top', transform=ax.transAxes)
## label the plot
plt.xlabel(r'$k$',           fontsize=20)
plt.ylabel(r'$\mathcal{P}$', fontsize=20)
## scale axies
ax.set_xscale('log')
ax.set_yscale('log')
line_mag.axes.set_xlim(xlim_min, xlim_max) # set x-axis limits
line_vel.axes.set_xlim(xlim_min, xlim_max)
line_mag.axes.set_ylim(ylim_min, ylim_max) # set y-axis limits
line_vel.axes.set_ylim(ylim_min, ylim_max)
## animate the spectra evolution
# blit: only update portion of frame that has changed
# interval: draw new frame every 'interval' ms
# save_count: number of frames to draw
ani = animation.FuncAnimation(fig, updateData, updateIter, 
                blit=False, interval=200, save_count=file_max_num, repeat=bool(0))
## display animation
plt.show()
## display the domain limits
if bool_disp_limits:
    print('\nxlim: [%0.3e'%var_xlim_min +', %0.3e]'%var_xlim_max)
    print('ylim: [%0.3e'%var_ylim_min +', %0.3e]'%var_ylim_max)
## save animation
if bool_save_ani:
    print('\nsaving animation')
    ani_name = (createFilePath([folder_main, folder_name, folder_vis]) + name_vid + '.mp4')
    bool_print_progress = True # indicate the progress of saving
    ani.save(ani_name, writer=writer, dpi=512)
    bool_print_progress = False
    print('saved animation: ' + ani_name)

## END OF PROGRAM