#!/usr/bin/env python3

''' AUTHOR: Neco Kriel
    
    EXAMPLE: 
    hdf5_plot_inline.py
        -base_path /Users/dukekriel/Documents/University/Year4Sem2/Summer-19/ANU-Turbulence-Dynamo/dyna288_Bk10 
        -pre_name dyna288_Bk10
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
def stringChop(var_string, var_remove):
    if var_string.endswith(var_remove):
        var_string = var_string[:-len(var_remove)]
    if var_string.startswith(var_remove):
        var_string = var_string[len(var_remove):]
    return var_string

def meetCondition(element):
    global bool_debug_mode, file_max
    element_start_right = bool(element.startswith('Turb_hdf5_plt_cnt_'))
    if element_start_right:
        print(element)
        print(element.split('_')[-1])
        iter_number = int(element.split('_')[-1])
        if bool_debug_mode:
            return bool(element_start_right and (iter_number > 0) and (iter_number <= 5))
        elif file_max != np.Inf:
            return bool(element_start_right and (iter_number > 0) and (iter_number <= file_max))
        else:
            return bool(element_start_right and (iter_number > 0))
    return False

def createFilePath(paths):
    return ('/'.join(paths) + '/')

def createFolder(folder_name):
    if not(os.path.exists(folder_name)):
        os.makedirs(folder_name)
        print('SUCCESS: \n\tFolder created. \n\t' + folder_name)
        print(' ')
    else:
        print('WARNING: \n\tFolder already exists (folder not created). \n\t' + folder_name)
        print(' ')

def reformatField(field, nx=None, procs=None):
    """
    AUTHORS: 
        James Beattie (version 26 November 2019)
        Neco Kriel (Editing)
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

def loadData(filepath):
    global bool_debug_mode
    global num_blocks, num_procs
    global type_var
    f     = h5py.File(filepath, 'r')  # open hdf5 file stream: [iProc*jProc*kProc, nzb, nyb, nxb]
    names = [s for s in list(f.keys()) if s.startswith(type_var)] # save all keys containing the string type_var
    data  = sum(np.array(f[i])**2 for i in names)                 # determine the variable's magnitude
    if type_var == 'mags': data /= plasma_beta                    # normalise the magnitude
    if bool_debug_mode: 
        print('--------- All the keys stored in the file:\n\t' + '\n\t'.join(list(f.keys()))) # print keys
        print('--------- All keys that are used: ' + str(names))
    f.close() # close the file stream
    ## reformat data
    data_sorted = reformatField(data, num_blocks, num_procs)
    return data_sorted

def plotData_3D(data, temp_iter):
    global col_map_min, col_map_max
    x = np.linspace(0, 1, data.shape[0])
    X, Y = np.meshgrid(x, x)
    ## initialise the figure
    fig = plt.figure(figsize=(10, 7), dpi=100)
    ax = fig.gca(projection='3d')
    ## plot data as contours
    print('\tPlotting contours...')
    lev_exp = np.arange(np.floor(np.log10(col_map_min)-1),
                        np.ceil(np.log10(col_map_max)+1), 0.5)
    levs = np.power(10, lev_exp)
    cbar_plot = ax.contourf(data[0, :, :], X, Y, levs, zdir='x', cmap='plasma', offset=0, alpha=1, 
                norm=mpl.colors.LogNorm())
    ax.contourf(X, data[:, 0, :], Y, levs, zdir='y', cmap='plasma', offset=1, alpha=1,
                norm=mpl.colors.LogNorm())
    ax.contourf(X, Y, data[:, :, 0], levs, zdir='z', cmap='plasma', offset=1, alpha=1,
                norm=mpl.colors.LogNorm())
    print('\tAnotating plot...')
    plt.tight_layout()
    ## set figure range
    ax.set_xlim((-0, 1)); ax.set_ylim((-0, 1)); ax.set_zlim((-0, 1))
    ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1]); ax.set_zticks([0, 0.5, 1])
    ## set 3D viewing angle
    ax.view_init(elev=30, azim=140)
    ## add time anotation
    ax.text2D(0.5, 1.0,
        r'$t/t_{\mathregular{eddy}} = $' + u'%0.1f'%(int(temp_iter)/10),
        fontsize=20, color='black',
        ha='center', va='top', transform=ax.transAxes)
    ## add a colour-bar
    cbar = plt.colorbar(cbar_plot, shrink=0.8, pad=0.01, label=r'$B/B_{0}$')
    ax = cbar.ax
    cbar.ax.tick_params(labelsize=13)
    text = ax.yaxis.label
    font = mpl.font_manager.FontProperties(family='times new roman', style='italic', size=18)
    text.set_font_properties(font)
    ## save plot
    print('\tSaving 3D figure...')
    plt.savefig(filepath_plot + pre_name +'_'+ type_var +'_'+ temp_iter + '_3D.png')
    print('\tSaved 3D figure: %i.'%int(temp_iter) + '  \t%0.3f%% complete\n'%(100 * int(temp_iter)/len(file_names)))
    plt.close()

def plotData_2D(data, temp_iter):
    global col_map_min, col_map_max
    ## initialise the figure
    fig = plt.figure(figsize=(10, 7), dpi=100)
    ax  = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ## plot data
    plt.imshow(data[0, :, :], 
            extent=(0.0, 1.0, 0.0, 1.0),
            interpolation='none',
            cmap='plasma',
            norm=LogNorm())
    ## add a colour-bar
    print('\tAnotating plot...')
    ## annotate the plot
    ax.text(0.5, 0.95,
        r'$t/t_{\mathregular{eddy}} = $' + u'%0.1f'%(int(temp_iter)/10),
        fontsize=20, color='black',
        ha='center', va='top', transform=ax.transAxes)
    ## add a colour-bar
    cbar = plt.colorbar(label=r'$B/B_{0}$')
    ax   = cbar.ax
    cbar.ax.tick_params(labelsize=13)
    plt.clim(col_map_min, col_map_max) # set the colour bar limits
    ## label and tune the plot
    text = ax.yaxis.label
    font = mpl.font_manager.FontProperties(family='times new roman', style='italic', size=18)
    text.set_font_properties(font)
    plt.xlim([0.0,1.0]); plt.ylim([0.0,1.0]) # set the x,y-limits
    plt.xticks([0.0,0.5,1.0]); plt.yticks([0.0,0.5,1.0]) # specify marker points
    plt.xticks([0.0,0.5,1.0], [r'$0$', r'$L/2$', r'$L$'], fontsize=20) # label the marker points
    plt.yticks([0.0,0.5,1.0], [r'$0$', r'$L/2$', r'$L$'], fontsize=20)
    plt.minorticks_on()
    ## save plot
    print('\tSaving 2D figure...')
    plt.savefig(filepath_plot + pre_name +'_'+ type_var +'_'+ temp_iter + '_2D.png')
    print('\tSaved 2D figure: %i.'%int(temp_iter) + '  \t%0.3f%% complete\n'%(100 * int(temp_iter)/len(file_names)))
    plt.close()

##################################################################
## INPUT COMMAND LINE ARGUMENTS
##################################################################
global file_max, bool_debug_mode, num_proc
ap = argparse.ArgumentParser(description='A bunch of input arguments')
## ------------------- DEFINE OPTIONAL ARGUMENTS
ap.add_argument('-debug',      type=bool, default=False,        required=False, help='Debug mode')
ap.add_argument('-ani_only',   type=bool, default=False,        required=False, help='Only animate currently existing plots')
ap.add_argument('-sub_folder', type=str,  default='hdf5Files',  required=False, help='Name of the data folder')
ap.add_argument('-vis_folder', type=str,  default='visFiles',   required=False, help='Name of the plot folder')
ap.add_argument('-type_var',   type=str,  default='mag',        required=False, help='Variable to be plotted')
ap.add_argument('-num_dims',   type=str,  default='3D',         required=False, help='Number of dimensions to plot')
ap.add_argument('-start',      type=str,  default='0',          required=False, help='Start frame number')
ap.add_argument('-fps',        type=str,  default='40',         required=False, help='Animation frame rate')
ap.add_argument('-num_files',  type=int,  default=np.Inf,       required=False, help='Number of files to process')
ap.add_argument('-num_proc',   type=int,  default=8,            required=False, help='Number of processors')
## ------------------- DEFINE REQUIRED ARGUMENTS
ap.add_argument('-base_path',  type=str,  required=True, help='File path to data')
ap.add_argument('-pre_name',   type=str,  required=True, help='Name of figures')
## ------------------- OPEN ARGUMENTS
args = vars(ap.parse_args())
## ------------------- SAVE BOOLEANS
bool_debug_mode = args['debug']
bool_ani_only = args['ani_only']
## ------------------- SAVE ANIMATION PARAMETERS
type_var  = args['type_var']
num_dims  = args['num_dims']
ani_start = args['start'] # starting animation frame
ani_fps   = args['fps']   # animation's fps
## the number of plots to process
file_max  = args['num_files']
## ------------------- SAVE FILEPATH PARAMETERS
filepath_base = args['base_path']  # home directory of data
folder_vis    = args['vis_folder'] # subfolder where animation and plots will be saved
folder_sub    = args['sub_folder'] # sub-subfolder where data is stored's name
pre_name      = args['pre_name']   # name attached to the front of figures and animation
num_proc      = args['num_proc']   # number of processors
## ------------------- ADJUST ARGUMENTS
## remove the trailing '/' from the input filepath
if filepath_base.endswith('/'):
    filepath_base = filepath_base[:-1]
## replace '//' with '/'
filepath_base = filepath_base.replace('//', '/')
## remove '/' from variables
folder_vis = stringChop(folder_vis, '/')
folder_sub = stringChop(folder_sub, '/')
pre_name   = stringChop(pre_name,   '/')

##################################################################
## SETUP VARIABLES
##################################################################
global num_blocks, num_procs
t_eddy      = 5
plasma_beta = 1e-10
num_blocks  = [48, 36, 36]
num_procs   = [6, 8, 8]

filepath_data = createFilePath([filepath_base, folder_sub]) # where data is stored
filepath_plot = createFilePath([filepath_base, folder_vis, 'plotHDF5_' + num_dims]) # where plots will be saved
print('The base directory is:\n\t' + filepath_base)
print('The directory to the slice data is:\n\t' + filepath_data)
print('The directory to where the figure will be saved is:\n\t' + filepath_plot)
print('The chosen variable to plot:\n\t' + type_var)
print(' ')
## create folder where plots are saved
createFolder(filepath_plot)
## save all file names that will to be processed
file_names = list(filter(meetCondition, sorted(os.listdir(filepath_data))))
num_figs = len(file_names)
print('The files in filepath_data that satisfied meetCondition:')
print('\t' + '\n\t'.join(file_names))
print(' ')

##################################################################
## CALCULATE MIN and MAX VALS (for colour-map)
##################################################################
global col_map_min, col_map_max
col_map_min = np.nan
col_map_max = np.nan
if not(bool_ani_only):
    temp_data = loadData(filepath_data + file_names[-1])
    col_map_min = np.nanmin([col_map_min, np.nanmin(temp_data[0, :, :])])
    col_map_max = np.nanmax([col_map_max, np.nanmax(temp_data[0, :, :])])
    print('Colour map domain: \n\tmin=' + str(col_map_min) + '\n\tmax='+str(col_map_max))
    print(' ')

##################################################################
## PROCESS and PLOT DATA
##################################################################
if not(bool_ani_only):
    print('Starting to plot...')
    print(' ')
    figs_complete = 0
    for file_name in file_names:
        ## process data
        print('Loading data...')
        temp_data = loadData(filepath_data + file_name)
        ## plot and save data
        if num_dims == '3D':
            print('Plotting 3D data...')
            plotData_3D(temp_data, file_name.split('_')[-1])
        else:
            print('Plotting 2D data...')
            plotData_2D(temp_data, file_name.split('_')[-1])
    print('Finished plotting...')

##################################################################
## ANIMATING CODE
##################################################################
filepath_input  = (filepath_plot + pre_name +'_'+ type_var + '_%04d_' + num_dims + '.png')
filepath_output = (filepath_plot + '../' + pre_name +'_ani_'+ type_var + '_' + num_dims + '.mp4')
ffmpeg_input    = ('ffmpeg -start_number '          + ani_start + 
                ' -i '                              + filepath_input + 
                ' -vb 40M -framerate '              + ani_fps + 
                ' -vf scale=1440:-1 -vcodec mpeg4 ' + filepath_output)
if bool_debug_mode:
    print('--------- Debug: Check FFMPEG input -----------------------------------')
    print('Input: ' + filepath_input)
    print(' ')
    print('Output: ' + filepath_output)
    print(' ')
    print('FFMPEG command: ' + ffmpeg_input)
    print(' ')
else:
    print('Animating plots...')
    os.system(ffmpeg_input) 
    print('Animation finished: ' + filepath_output)
    # eg. ffmpeg -start_number 0 -i ./plotSlices/name_plot_slice_mag_%6d.png -vb 40M -framerate 25 -vf scale=1440:-1 -vcodec mpeg4 ./name_ani_mag.mp4

