#!/usr/bin/env python3

''' AUTHOR: Neco Kriel
    
    EXAMPLE: 
    slice_plot_inline.py 
        -base_path /Users/dukekriel/Documents/University/Year4Sem2/Summer-19/ANU-Turbulence-Dynamo/dyna288_Bk10 
        -pre_name dyna288_Bk10 
        -num_proc 4

    OTHER: 
    Compiling flash4 command:
        ./setup StirFromFileDynamo -3d -auto -objdir=objStirFromFileDynamo/ -nxb=36 -nyb=36 -nzb=48 +ug --with-unit=physics/Hydro/HydroMain/split/Bouchut/IsothermalSoundSpeedOne --without-unit=PhysicalConstants +parallelIO
'''

##################################################################
## MODULES
##################################################################
import os
import h5py
import argparse
import numpy as np
import multiprocessing as mp
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm

##################################################################
## PREPARE TERMINAL/CODE
##################################################################
os.system('clear')                  # clear terminal window
plt.close('all')                    # close all pre-existing plots
mpl.style.use('classic')            # plot in classic style
data_queue = mp.Queue()

##################################################################
## FUNCTIONS
##################################################################
def stringChop(var_string, var_remove):
    if var_string.endswith(var_remove):
        var_string = var_string[:-len(var_remove)]
    if var_string.startswith(var_remove):
        var_string = var_string[len(var_remove):]
    return var_string

def createFolder(folder_name):
    if not(os.path.exists(folder_name)):
        os.makedirs(folder_name)
        print('SUCCESS: \n\tFolder created. \n\t' + folder_name)
        print(' ')
    else:
        print('WARNING: \n\tFolder already exists (folder not created). \n\t' + folder_name)
        print(' ')

def createCommand(commands):
    return (' '.join(commands))

def createFilePath(paths):
    return ('/'.join(paths) + '/')

def meetCondition(element):
    global bool_debug_mode, file_max
    if bool_debug_mode:
        return bool(element.startswith('Turb_slice_xy_') and (int(element[-6:]) <= 5))
    elif file_max != np.Inf:
        return bool(element.startswith('Turb_slice_xy_') and (int(element[-6:]) <= file_max))
    else:
        return bool(element.startswith('Turb_slice_xy_'))

def calcMinMax():
    global filepath_data, file_names, num_figs
    global t_eddy, plasma_beta
    var_min = np.nan
    var_max = np.nan
    for var_iter in range(num_figs):
        ## load data 
        file_name = filepath_data + '/' + file_names[var_iter] # create the filepath to the file
        f         = h5py.File(file_name, 'r')                  # load slice file
        names     = [s for s in list(f.keys()) if 'mag' in s]  # save all keys that contain 'mag'
        data      = sum(np.array(f[i])**2 for i in names)      # determine the magnetic-field magnitude
        data     /= plasma_beta                                # normalise the magnitude
        f.close()                                              # close the file stream
        ## calculate min/max
        var_min = np.nanmin([var_min, np.nanmin(data)])
        var_max = np.nanmax([var_max, np.nanmax(data)])
    ## return min and max values
    return [var_min, var_max]

## based on: https://stackoverflow.com/questions/16254191/python-rpy2-and-matplotlib-conflict-when-using-multiprocessing
def parPlot():
    global num_figs, filepath_plot, name_fig, num_proc
    ## open processor
    pool = mp.Pool(num_proc)
    pool.map(worker, range(num_figs))
    figs_complete = 0
    ## loop over data (in parallel)
    print('Started plotting.')
    while figs_complete < num_figs:
        ## initialise the figure
        fig = plt.figure(figsize=(10, 7), dpi=100)
        ax  = fig.add_axes([0.1, 0.1, 0.8, 0.8])

        #############################################
        ## RECEIVE THE PROCESSED DATA FROM WORKER
        #############################################
        data, time_point, var_iter = data_queue.get()
        
        #############################################
        ## PLOT DATA
        #############################################
        plt.imshow(data, 
            extent=(0.0, 1.0, 0.0, 1.0),
            interpolation='none',
            cmap='plasma',
            norm=LogNorm())
        
        #############################################
        ## LABEL and TUNE PLOT
        #############################################
        ## annotate the plot
        ax.text(0.5, 0.95,
            r'$t/t_{\mathregular{eddy}} = $' + u'%0.2f'%(time_point),
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

        #############################################
        ## SAVE FIGURE
        #############################################
        temp_name = '/' + name_fig + '{0:06}'.format(var_iter)
        plt.savefig(filepath_plot + temp_name)
        print('Saved figure: %i'%int(temp_name[-6:]) + '  \t%0.3f%% complete'%(100 * figs_complete/num_figs))
        plt.close()
        figs_complete += 1
    ## close pool of processors
    pool.close()
    print('Finished plotting.')
    print(' ')

def worker(var_iter):
    global bool_debug_mode
    global filepath_data, file_names
    global t_eddy, plasma_beta
    file_name  = filepath_data + '/' + file_names[var_iter] # create the filepath to the file
    f          = h5py.File(file_name, 'r')                  # load slice file
    time_point = np.array(f['time']) / t_eddy               # save time point
    if bool_debug_mode:
        print('Looking at:' + file_name)
    names      = [s for s in list(f.keys()) if 'mag' in s]  # save all keys that contain 'mag'
    data       = sum(np.array(f[i])**2 for i in names)      # determine the magnetic-field magnitude
    data      /= plasma_beta                                # normalise the magnitude
    f.close()                                               # close the file stream
    data_queue.put([data, time_point, var_iter])            # send data to be plot

def reformatField(field, nx=None, procs=None):
    """
    AUTHORS: 
        James Beattie (version 26 November 2019)
        Neco Kriel (Editing)
    PURPOSE:
        This code reformats the FLASH block / xyz format into xyz format for processing
    INPUTS:
        field   — the FLASH field
        nx      - number of blocks per spatial direction (stored in an array [x, y, z])
        procs   - number of cores per spatial direction (stored in an array [i, j, k])
    OUTPUT:
        field_sorted - the organised 2D field
    """
    # interpret the function arguments
    iprocs = procs[0]
    jprocs = procs[1]
    kprocs = procs[2]
    nxb = nx[0]
    nyb = nx[1]
    nzb = nx[2]
    # initialise the output field
    field_sorted = np.zeros([nzb*kprocs, nxb*iprocs, nyb*jprocs])
    # sort and store the unsorted field into the output field
    for k in range(kprocs):
        for j in range(jprocs):
            for i in range(iprocs):
                field_sorted[k*nzb:(k+1)*nzb, i*nxb:(i+1)*nxb, j*nyb:(j+1)*nyb] = field[k + j*iprocs + i*(jprocs*iprocs)]
    return field_sorted

##################################################################
## INPUT COMMAND LINE ARGUMENTS
##################################################################
global file_max, bool_debug_mode, num_proc
ap = argparse.ArgumentParser(description='A bunch of input arguments')
## ------------------- DEFINE OPTIONAL ARGUMENTS
ap.add_argument('-ani_only',   required=False, help='Debug mode',                                   type=bool, default=False)
ap.add_argument('-debug',      required=False, help='Debug mode',                                   type=bool, default=False)
ap.add_argument('-vis_folder', required=False, help='Name of the plot folder',                      type=str,  default='visFiles')
ap.add_argument('-sub_folder', required=False, help='Name of the folder where the data is stored',  type=str,  default='sliceFiles')
ap.add_argument('-start',      required=False, help='Start frame number',                           type=str,  default='0')
ap.add_argument('-fps',        required=False, help='Animation frame rate',                         type=str,  default='40')
ap.add_argument('-num_files',  required=False, help='Number of files to process',                   type=int,  default=-1)
ap.add_argument('-num_proc',   required=False,  help='Number of processors',                         type=int,  default=8)
## ------------------- DEFINE REQUIRED ARGUMENTS
ap.add_argument('-base_path',  required=True, help='File path to data',    type=str)
ap.add_argument('-pre_name',   required=True, help='Name of figures',      type=str)
## ------------------- OPEN ARGUMENTS
args = vars(ap.parse_args())
## ------------------- SAVE BOOLEANS
## enable/disable debug mode
if (args['debug'] == True):
    bool_debug_mode = True
else:
    bool_debug_mode = False
## enable/disable animating only mode
if (args['ani_only'] == True):
    bool_ani_only_mode = True
else:
    bool_ani_only_mode = False
## ------------------- SAVE ANIMATION PARAMETERS
ani_start     = args['start'] # starting animation frame
ani_fps       = args['fps']   # animation's fps
## the number of plots to process
if (args['num_files'] < 0):
    file_max = np.Inf
else:
    file_max = args['num_files']
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
folder_vis    = stringChop(folder_vis, '/')
folder_sub    = stringChop(folder_sub, '/')
pre_name      = stringChop(pre_name, '/')
## ------------------- START CODE
print("Began running the slice code in folder: \n\t" + filepath_base)
print('Visualising folder: ' + folder_vis)
print('Figure name: ' + pre_name)
print(' ')

##################################################################
## USER DEFINED VARIABLES
##################################################################
## define parameters
global t_eddy, plasma_beta
global col_map_min, col_map_max
t_eddy      = 5 # L/(2*Mach)
plasma_beta = 1e-10
## define the directory
global name_fig, name_vid
name_fig    = pre_name + '_plot_slice_mag_'
name_vid    = pre_name + '_ani_mag'

##################################################################
## SETUP VARIABLES
##################################################################
## announce debug status
if bool_debug_mode:
    print('--------- Debug mode is on. -----------------------------------')
    print(' ')
# initialise filepaths
global filepath_data, filepath_plot
global file_names, num_figs
filepath_data = createFilePath([filepath_base, folder_sub]) # where data is stored
filepath_plot = createFilePath([filepath_base, folder_vis, 'plotSlices']) # where plots will be saved
if bool_debug_mode:
    print('--------- Debug: Check directories -----------------------------------')
    print('The base directory is:\n\t' + filepath_base)
    print('The directory to the slice data is:\n\t' + filepath_data)
    print('The directory to where the figure will be saved is:\n\t' + filepath_plot)
    print(' ')
## create folder where plots are saved
createFolder(filepath_plot)
## save the the names of the files that will be processed
file_names = list(filter(meetCondition, sorted(os.listdir(filepath_data))))
num_figs = len(file_names)
print('The files in filepath_data that satisfied meetCondition:')
print('\t' + '\n\t'.join(file_names))
print(' ')

##################################################################
## CALCULATE MIN and MAX VALS (for colour-map)
##################################################################
if not(bool_ani_only_mode):
    print('Calculating the colour-map domain...')
    col_map_min, col_map_max = calcMinMax()
    print('The colour-map domain:')
    print('\tMin: ' + str(col_map_min))
    print('\tMax: ' + str(col_map_max))
    print(' ')

##################################################################
## PROCESS DATA and SAVE IMAGES
##################################################################
if not(bool_ani_only_mode):
    print('Processing and plotting data...')
    print(' ')
    parPlot()
    print('Finished processing data.')
    print(' ')

##################################################################
## ANIMATE IMAGES
##################################################################
filepath_input  = (filepath_plot + name_fig + '%06d.png')
filepath_output = (filepath_plot + '../' + name_vid + '.mp4')
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

## END OF PROGRAM