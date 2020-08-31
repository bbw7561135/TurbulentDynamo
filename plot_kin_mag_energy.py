#!/usr/bin/env python3

''' AUTHOR: Neco Kriel
    
    EXAMPLE: 
    plot_kin_mag_energy.py
        (required)
            -base_path      $scratch
            -dat_folder1    dyna288_Bk10/Re10/hdf5Files
            -dat_folder2    dyna288_Bk10/Pm1/hdf5Files
            -pre_name       dyna288_Bk
        (optional)
            -debug          False
            -vis_folder     visFiles
            -plasma_beta    1e-10
            -nxb            36
            -nyb            36
            -nzb            48
            -iProc          8
            -jProc          8
            -kProc          6
    
    OTHER: 
    Compiling flash4 command:
        ./setup StirFromFileDynamo -3d -auto -objdir=objStirFromFileDynamo/ -nxb=36 -nyb=36 -nzb=48 +ug --with-unit=physics/Hydro/HydroMain/split/Bouchut/IsothermalSoundSpeedOne --without-unit=PhysicalConstants +parallelIO
'''

##################################################################
## MODULES
##################################################################
import os
import h5py
import math
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as path_effects

from mpl_toolkits import mplot3d
from matplotlib import ticker, cm
from matplotlib.colors import LogNorm

##################################################################
## PREPARE TERMINAL/CODE
##################################################################
os.system('clear')                  # clear terminal window
plt.close('all')                    # close all pre-existing plots
mpl.style.use('classic')            # plot in classic style

##################################################################
## FUNCTION
##################################################################
def str2bool(v):
    '''
    FROM:
        https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    '''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def stringChop(var_string, var_remove):
    ''' stringChop
    PURPOSE / OUTPUT:
        Remove the occurance of the string 'var_remove' at both the start and end of the string 'var_string'.
    '''
    if var_string.endswith(var_remove):
        var_string = var_string[:-len(var_remove)]
    if var_string.startswith(var_remove):
        var_string = var_string[len(var_remove):]
    return var_string

def createFilePath(paths):
    ''' creatFilePath
    PURPOSE / OUTPUT:
        Turn an ordered list of names and concatinate them into a filepath.
    '''
    temp_path = ('/'.join(paths))
    return temp_path.replace('//', '/')

def createFolder(folder_name):
    ''' createFolder
    PURPOSE:
        Create the folder passed as a filepath to inside the folder.
    OUTPUT:
        Commandline output of the success/failure status of creating the folder.
    '''
    if not(os.path.exists(folder_name)):
        os.makedirs(folder_name)
        print('SUCCESS: \n\tFolder created. \n\t' + folder_name)
        print(' ')
    else:
        print('WARNING: \n\tFolder already exists (folder not created). \n\t' + folder_name)
        print(' ')

def reformatField(field, nx=None, procs=None):
    """ reformatField
    AUTHORS: 
        James Beattie (version 26 November 2019)
        Neco Kriel (Edited)
    PURPOSE:
        This code reformats the FLASH block / zxy format into zxy format for processing
    INPUTS:
        field   — the FLASH field
        nx      - number of blocks per spatial direction (stored in an array [x, y, z])
        procs   - number of cores per spatial direction (stored in an array [i, j, k])
    OUTPUT:
        field_sorted - the organised 2D field
    """
    ## interpret the function arguments
    iprocs = procs[0]
    jprocs = procs[1]
    kprocs = procs[2]
    nxb = nx[0]
    nyb = nx[1]
    nzb = nx[2]
    ## initialise the output field
    field_sorted = np.zeros([nzb*kprocs, nxb*iprocs, nyb*jprocs])
    ## sort and store the unsorted field into the output field
    for k in range(kprocs):
        for j in range(jprocs):
            for i in range(iprocs):
                field_sorted[i*nxb:(i+1)*nxb, j*nyb:(j+1)*nyb, k*nzb:(k+1)*nzb] = field[k + j*jprocs + i*jprocs*kprocs]
    return field_sorted

def loadData(filepath, file_type):
    global bool_debug_mode
    global num_blocks, num_procs
    f     = h5py.File(filepath, 'r')  # open hdf5 file stream: [iProc*jProc*kProc, nzb, nyb, nxb]
    names = [s for s in list(f.keys()) if s.startswith(file_type)] # save all keys containing the string file_type
    data  = sum(np.array(f[i])**2 for i in names)                  # determine the variable's magnitude
    if file_type == 'mags': data /= plasma_beta                    # normalise the magnitude
    if bool_debug_mode: 
        print('--------- All the keys stored in the file:\n\t' + '\n\t'.join(list(f.keys()))) # print keys
        print('--------- All keys that are used: ' + str(names))
    f.close() # close the file stream
    ## reformat data
    data_sorted = reformatField(data, num_blocks, num_procs)
    return data_sorted

def nameFile(num):
    return 'Turb_hdf5_plt_cnt_' + '{0:04}'.format(num) # data file

##################################################################
## INPUT COMMAND LINE ARGUMENTS
##################################################################
global bool_debug_mode
global pre_name, plasma_beta
ap = argparse.ArgumentParser(description='A bunch of input arguments')
## ------------------- DEFINE OPTIONAL ARGUMENTS
ap.add_argument('-debug',      type=str2bool,   default=False,        required=False, help='Debug mode', nargs='?', const=True)
ap.add_argument('-vis_folder', type=str,        default='visFiles',   required=False, help='Name of the plot folder')
ap.add_argument('-plasma_beta',type=int,        default=1e-10,        required=False, help='Plasma-Beta')
ap.add_argument('-nxb',        type=int,        default=36,           required=False, help='Number of blocks in the x-direction')
ap.add_argument('-nyb',        type=int,        default=36,           required=False, help='Number of blocks in the y-direction')
ap.add_argument('-nzb',        type=int,        default=48,           required=False, help='Number of blocks in the z-direction')
ap.add_argument('-iProc', type=int, default=8, required=False, help='Number of processors in the x-direction')
ap.add_argument('-jProc', type=int, default=8, required=False, help='Number of processors in the y-direction')
ap.add_argument('-kProc', type=int, default=6, required=False, help='Number of processors in the z-direction')
## ------------------- DEFINE REQUIRED ARGUMENTS
ap.add_argument('-base_path',   type=str, required=True, help='Base path')
ap.add_argument('-dat_folder1', type=str, required=True, help='File path to folder with data set 1')
ap.add_argument('-dat_folder2', type=str, required=True, help='File path to folder with data set 2')
ap.add_argument('-pre_name',    type=str, required=True, help='Name of figure')
# ap.add_argument('-file_time_vals', type=int, required=True, help='File numbers to plot', nargs='+')
## ------------------- OPEN ARGUMENTS
args = vars(ap.parse_args())
## ---------------------------- SAVE BOOLEAN PARAMETERS
bool_debug_mode = args['debug']
## ---------------------------- SAVE SIMULATION PARAMETERS
# file_time_vals      = args['file_time_vals'] # starting processing frame
plasma_beta     = args['plasma_beta']
nxb             = args['nxb']
nyb             = args['nyb']
nzb             = args['nzb']
iProc           = args['iProc']
jProc           = args['jProc']
kProc           = args['kProc']
## ---------------------------- SAVE FILEPATH PARAMETERS
base_filepath   = args['base_path']
data_filepath_1 = args['dat_folder1']
data_filepath_2 = args['dat_folder2']
folder_vis      = args['vis_folder'] # subfolder where animation and plots will be saved
pre_name        = args['pre_name']   # name attached to the front of figures and animation

##################################################################
## SETUP VARIABLES
##################################################################
global num_blocks, num_procs
num_blocks = [nzb,     nxb,    nyb]
num_procs  = [kProc,   iProc,  jProc]
label_energy      = [r'$10^{-1}$', r'$10^{1}$', r'$10^{3}$', r'$10^{5}$', r'sat.']
label_sim_name    = [r'Pm100', r'Pm1']
file_time_vals    = [[70, 146, 227, 310, 500], [36, 137, 243, 335, 600]]
filepath_datasets = [   createFilePath([base_filepath, data_filepath_1]), 
                        createFilePath([base_filepath, data_filepath_2])]
filepath_plot     = createFilePath([base_filepath, folder_vis])
## print information to screen
print('\nDirectory to the first hdf5 dataset: \t'  + filepath_datasets[0])
print('Directory to the second hdf5 dataset: \t'   + filepath_datasets[1])
print('Time points to check:')
print('\tIn folder 1: {}'.format(file_time_vals[0]))
print('\tIn folder 2: {}'.format(file_time_vals[1]))
print('Directory to the plotting folder: \t'       + filepath_plot)
print('Figure name: '                              + pre_name)
print(' ')
## create folder where plots are saved
createFolder(filepath_plot)
## initialise the figure
fig   = plt.figure(figsize=(12, 14.5))
ax    = fig.add_subplot(111)
gspec = gridspec.GridSpec(5, 4)

##################################################################
## CALCULATE COLOURBAR LIMITS
##################################################################
## initialise colourbar limits
col_map_min_vel = np.nan
col_map_max_vel = np.nan
col_map_min_mag = np.nan
col_map_max_mag = np.nan
print('Calculating colourbar limits...')
## loop over each dataset folder
for filepath_index in range(len(filepath_datasets)):
    ## CALCULATE MIN and MAX VALS (for colour-map)
    ##############################################
    print('\tLooking at folder: ' + filepath_datasets[filepath_index])
    ## calculate velocity field colour-map bounds
    for file_num in file_time_vals[filepath_index]:
        temp_data       = loadData(createFilePath([filepath_datasets[filepath_index], nameFile(file_num)]), 'vel')
        col_map_min_vel = np.nanmin([col_map_min_vel, np.nanmin(temp_data[0, :, :])])
        col_map_max_vel = np.nanmax([col_map_max_vel, np.nanmax(temp_data[0, :, :])])
    ## calculate magnetic field colour-map bounds
    for file_num in file_time_vals[filepath_index]:
        temp_data       = loadData(createFilePath([filepath_datasets[filepath_index], nameFile(file_num)]), 'mag')
        col_map_min_mag = np.nanmin([col_map_min_mag, np.nanmin(temp_data[0, :, :])])
        col_map_max_mag = np.nanmax([col_map_max_mag, np.nanmax(temp_data[0, :, :])])
## print colourbar limits
print('\tVelocity field colour map domain: \n\t\tmin=' + str(col_map_min_vel) + '\n\t\tmax='+str(col_map_max_vel))
print('\tMagnetic field colour map domain: \n\t\tmin=' + str(col_map_min_mag) + '\n\t\tmax='+str(col_map_max_mag))
print(' ')

##################################################################
## LOAD + PLOT VELOCITY FIELD DATA
##################################################################
print('Loading and plotting data...')
## loop over each dataset folder
bool_energy_label = False
for filepath_index in range(len(filepath_datasets)):
    ## loop over each file in the dataset folder
    bool_label_sim_name = False
    print('\tLooking at folder: ' + filepath_datasets[filepath_index])
    for file_index in range(len(file_time_vals[filepath_index])):
        ## load data
        temp_data_vel = loadData(createFilePath([filepath_datasets[filepath_index], nameFile(file_time_vals[filepath_index][-(file_index+1)])]), 'vel')
        temp_data_mag = loadData(createFilePath([filepath_datasets[filepath_index], nameFile(file_time_vals[filepath_index][-(file_index+1)])]), 'mag')
        ## prepare subplot for velocity field data
        ax1 = plt.subplot(gspec[(file_index * 4) + filepath_index])
        plt.axis('on')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')
        ## plot velocity field data
        vel = plt.imshow(temp_data_vel[int(len(temp_data_vel[:,0,0])/2), :, :], extent=(0.0, 1.0, 0.0, 1.0), interpolation='none', 
                        cmap='bone', norm=LogNorm(), vmin=col_map_min_vel, vmax=col_map_max_vel)
        vel.axes.get_xaxis().set_visible(False)
        vel.axes.get_yaxis().set_visible(False)
        ## prepare subplot for magnetic field data
        ax2 = plt.subplot(gspec[(file_index * 4) + 2 + filepath_index])
        plt.axis('on')
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        ax2.set_aspect('equal')
        ## plot velocity field data
        mag = plt.imshow(temp_data_mag[int(len(temp_data_mag[:,0,0])/2), :, :], extent=(0.0, 1.0, 0.0, 1.0), interpolation='none', 
                        cmap='plasma', norm=LogNorm(), vmin=col_map_min_mag, vmax=col_map_max_mag)
        mag.axes.get_xaxis().set_visible(False)
        mag.axes.get_yaxis().set_visible(False)
        ## add simulation parameter labels
        if not(bool_label_sim_name):
            plt.text(0.5, 1.025,
                label_sim_name[filepath_index],
                fontsize=15, color='black',
                ha='center', va='bottom', transform=ax1.transAxes)
            plt.text(0.5, 1.025,
                label_sim_name[filepath_index],
                fontsize=15, color='black',
                ha='center', va='bottom', transform=ax2.transAxes)
            bool_label_sim_name = True
        ## labeling energy levels
        if not(bool_energy_label):
            plt.text(-0.05, 0.5,
                (r'$E_B/E_{B0} =$ ' + label_energy[-(file_index+1)]),
                fontsize=18, color='black',
                rotation='vertical', ha='right', va='center', transform=ax1.transAxes)
    bool_energy_label = True

##################################################################
## ADJUST + ADD COLOURBAR to PLOT
##################################################################
print('Adjusting and labelling plot...')
## add velocity field colourbar
cbar_1 = fig.colorbar(vel, orientation="horizontal", cax=fig.add_axes([0.125, 0.08, 0.38, 0.015]))
cbar_1.ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=7))
plt.text(0.25, -0.045,
    r'$u^2/u_0^2$',
    fontsize=20, color='black',
    ha='center', va='top', transform=ax.transAxes)
## add magnetic field colourbar
cbar_2 = fig.colorbar(mag, orientation="horizontal", cax=fig.add_axes([0.52, 0.08, 0.38, 0.015]))
cbar_2.ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=7))
plt.text(0.75, -0.045,
    r'$B^2/B_0^2$',
    fontsize=20, color='black',
    ha='center', va='top', transform=ax.transAxes)
# remove spacing between subplots
gspec.update(wspace=0, hspace=0)
print(' ')

##################################################################
## SAVE FIGURE
##################################################################
print('Saving figure...')
fig_name = createFilePath([filepath_plot, pre_name]) + '_energy.png'
plt.savefig(fig_name, bbox_inches='tight', dpi=300)
plt.close()
print('Figure saved: ' + fig_name)
print(' ')
