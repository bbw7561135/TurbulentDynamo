#!/usr/bin/env python2

##################################################################
## MODULES
##################################################################
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
import matplotlib.patheffects as path_effects

from matplotlib.colors import LogNorm
from matplotlib import animation
from IPython import get_ipython

##################################################################
## RUNNING CURRENT SETUP
##################################################################
# ./setup StirFromFileDynamo -3d -auto -objdir=objStirFromFileDynamo/ -nxb=64 -nyb=64 -nzb=64 +ug --with-unit=physics/Hydro/HydroMain/split/Bouchut/IsothermalSoundSpeedOne --without-unit=PhysicalConstants +parallelIO

##################################################################
## COPYING FILES ACROSS (gadi -> home)
##################################################################
# scp -r gadi:/scratch/ek9/nk7952/simData/ ~/Documents/University/Year4Sem2/Summer-19/ANU-Turbulence-Dynamo/simDyn256/

##################################################################
## PREPARE TERMINAL/CODE
##################################################################
os.system('clear')                  # clear terminal window
get_ipython().magic('reset -sf')    # clear workspace
plt.close('all')                    # close all pr-existing plots
mpl.style.use('classic')            # plot in classic style
plt.rc('font', family='serif')      # specify font choice

##################################################################
## PLOT DETAILS
##################################################################
t_eddy = 5 # L/(2*Mach)
## specify information about simulation setup
var_dim             = '3D' # number of dimensions of the simulation
nx_g                = 64   # number of cells per direction
num_proc            = 2    # number of processors per axis (assumes same for each axis)
## Specify where files are located and needs to be saved
folder_main         = os.path.dirname(os.path.realpath(__file__)) # get directory where file is saved
folder_name         = 'dyna128_Bk10/Mach0p098'
folder_files        = '' # folder where slice data is stored
folder_vis          = '' # folder where visualisation is saved
## Specify which file you want to save
bool_print_dir      = bool(0) # display all the files stored in the directory
name_file           = 'Turb_slice_xy_' # string that the data files start with
## Specify which variable you want to save
bool_print_keys     = bool(0) # display all the variables stored in the data
var_norm            = 1e-10
name_var            = 'mag' # data-field name
var_label           = r'$B/B_{0}$'
col_map_min         = 1.10e-05
col_map_max         = 2.57e+06
bool_repeat_ani     = bool(0) # loop the animation?
## Should the animation be saved?
bool_save_ani       = bool(0) # save the animation?

##################################################################
## FUNCTIONS
##################################################################
def reformatField(field, nx=None, procs=None, dim="3D"):
    """
    Author: James Beattie (26 November 2019)
    Edit:   Neco Kriel
    This code reformats the FLASH block / xyz format into xyz format for processing

    INPUTS:
    field   — the FLASH field
    nx      - number of blocks (assuming it is the same for all dim)
    procs   - number of cores (assuming it is the same for all dim)
    dim     - "2D" or "3D" (default). For the "2D" case: nzb = kprocs = 1

    OUTPUTs:
    field_sorted - the organised 2D field
    """
    # Interpreting function arguments
    if dim == "3D":
        # for the 3D simulation
        iprocs = procs; nxb = nx_g
        jprocs = procs; nyb = nx_g
        kprocs = procs; nzb = nx_g
    elif dim == "2D":
        # for the 2D simulation
        iprocs = procs; nxb = nx_g
        jprocs = procs; nyb = nx_g
        kprocs = 1;     nzb = 1
    # Create a dummy field
    field_sorted = np.zeros([nzb*kprocs, nxb*iprocs, nyb*jprocs])
    # Sort the unsorted field
    for k in range(kprocs):
        for j in range(jprocs):
            for i in range(iprocs):
                field_sorted[k*nzb:(k+1)*nzb, i*nxb:(i+1)*nxb, j*nyb:(j+1)*nyb] = field[k + j*iprocs + i*(jprocs*iprocs)]
    return field_sorted

def fileName(names):
    return ('/'.join(names) + '/')

def loadData(var_directory, var_names, var_iter, var_dim):
    global var_norm, t_eddy
    f = h5py.File(var_directory + '/' + var_names[var_iter], 'r') # load the file
    name_vars = [s for s in list(f.keys()) if name_var in s] # store all the strings that contain the chars: name_var
    var_cons  = sum(np.array(f[name])**2 for name in name_vars)/var_norm # calculate the magnitude (consentration) of vector comps (name_var)
    var_time  = np.array(f['time'])/t_eddy # load time points
    f.close() # close the file stream
    if var_dim == "2D":
        var_cons = reformatField(var_cons, nx=nx_g, procs=num_proc, dim="2D")
    return var_cons, var_time

def updateIter():
    global file_max_num
    var_iter = -1
    while var_iter < file_max_num:
        # update iter
        var_iter += 1
        yield var_iter

def updateFig(data):
    global bool_disp_progress, var_abs_min, var_abs_max, file_max_num
    var_iter = data
    # calculate new data
    var_cons, var_time = loadData(var_directory=directory, var_names=file_names, var_iter=var_iter, var_dim='3D')
    var_abs_min = min(var_abs_min, min(map(min, var_cons)))
    var_abs_max = max(var_abs_max, max(map(max, var_cons)))
    if bool_disp_progress:
        print(u"saving: %0.5f"%(100 * var_iter/file_max_num) + '%')
    # update data
    im.set_data(var_cons)
    title.set_text(r"$t/t_{\mathregular{eddy}} = $" + u"%0.1f"%(var_time))
    return im, title,

def meetCondition(element):
    global name_file
    return bool(element.startswith(name_file))

##################################################################
## PLOTTING CODE
##################################################################
## Global Variable
bool_disp_progress = False
## setup animation writer
Writer = animation.writers['ffmpeg']
writer = Writer(fps=10)
## setup information for loading data
directory       = fileName([folder_main, folder_name, folder_files])
directory_files = sorted(os.listdir(directory))
file_names      = list(filter(meetCondition, directory_files))
file_max_num    = int(max(file_names)[-6:]) # number of frames in the animation
## display the files in the directory
if bool_print_dir:
    print('\nFiles in directory:')
    print('\n'.join(directory_files))
## display the variables stored in the file
if bool_print_keys:
    print('\nStored keys:')
    print('\n'.join(list(h5py.File(directory + '/' + file_names[0], 'r').keys())))
## load data
var_cons, _ = loadData(var_directory=directory, var_names=file_names, var_iter=0, var_dim='3D')
var_abs_min = min(map(min, var_cons)) # initialise the absolute minimum value (updated in the animation)
var_abs_max = 0 # initialise the absolute maximum value (updated in the animation)
## initialise the plot and those features that will be updated by the animation
fig = plt.figure(figsize=(10, 7), dpi=100)
ax  = fig.add_axes([0.1, 0.1, 0.8, 0.8])
im  = plt.imshow(var_cons, 
            extent=(0.0, 1.0, 0.0, 1.0),
            interpolation='none',
            cmap='plasma',
            norm=LogNorm(),
            animated=True)
title = ax.text(0.05, 0.95,
            r"$t/t_{\mathregular{eddy}} = $" + u"{}".format(0), 
            fontsize=17, color='black', path_effects=[path_effects.withSimplePatchShadow(offset=(1, -1))],
            ha="left", va='top', transform=ax.transAxes)
## label the plot
cbar = plt.colorbar(label=var_label) # colour bar
plt.clim(col_map_min, col_map_max) # set the colour bar limits
plt.xlim([0.0,1.0]); plt.ylim([0.0,1.0]) # set the x,y-limits
plt.xticks([0.0,0.5,1.0]); plt.yticks([0.0,0.5,1.0]) # specify marker points
plt.xticks([0.0,0.5,1.0], [r'$0$', r'$L/2$', r'$L$'], fontsize=17) # label the marker points
plt.yticks([0.0,0.5,1.0], [r'$0$', r'$L/2$', r'$L$'], fontsize=17)
plt.minorticks_on()
## animate the slice evolution
# interval:     draw new frame every 'interval' ms
# save_count:   number of frames to draw
# https://stackoverflow.com/questions/44594887/how-to-update-plot-title-with-matplotlib-using-animation
ani = animation.FuncAnimation(fig, updateFig, updateIter, interval=1, save_count=file_max_num, repeat=bool_repeat_ani)
## display the animation
plt.show()
## display the y-domain limits
print('set colour-bar: \tmin=%0.2e'%var_abs_min + '\tmax=%0.2e'%var_abs_max)
## save plot
if bool_save_ani:
    print('\nsaving animation')
    bool_disp_progress = True
    ani_name = (fileName([folder_main, folder_name, folder_vis]) + name_var + '.mp4')
    ani.save(ani_name, writer=writer, dpi=512)
    bool_disp_progress = False
    print('saved animation: \n' + ani_name)



