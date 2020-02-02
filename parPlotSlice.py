#!/usr/bin/env python3

## NECO KRIEL
## https://stackoverflow.com/questions/16254191/python-rpy2-and-matplotlib-conflict-when-using-multiprocessing

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
os.system('clear')
data_queue = mp.Queue()

##################################################################
## COMMAND LINE ARGUMENT INPUT
##################################################################
global file_max, debug_mode, num_proc
ap = argparse.ArgumentParser(description = 'A bunch of input arguments')
ap.add_argument('-debug',     required=False, help='Debug mode',                 type=bool, default=False)
ap.add_argument('-num_files', required=False, help='Number of files to process', type=int,  default=-1)
ap.add_argument('-num_proc',  required=True, help='Number of processors',        type=int)
ap.add_argument('-dir',       required=True, help='Directory of files',          type=str)
ap.add_argument('-name',      required=True, help='Name of files',               type=str)
args          = vars(ap.parse_args())
name          = args['name']     # name of file
num_proc      = args['num_proc'] # number of processors
filepath_base = args['dir']      # home directory of data
## the number of plots to process
if (args['num_files'] < 0):
    file_max = np.Inf
else:
    file_max = args['num_files']
## enable/disable debug mode
if (args['debug'] == True):
    debug_mode = True
else:
    debug_mode = False
## start code
print("Began running the slice code in folder: \n\t" + filepath_base + '\n')

##################################################################
## USER DEFINED VARIABLES
##################################################################
## define parameters
global t_eddy, plasma_beta
global col_map_min, col_map_max
t_eddy      = 5 # L/(2*Mach)
plasma_beta = 1e-10
col_map_min = 5.78e-11  # TODO: automate
col_map_max = 7.17e+08
## define the directory
global name_fig, name_vid
folder_data = 'sliceFiles'      # subfolder where the slice data is stored 
folder_vis  = 'visFiles'        # subfolder where the animation will be saved
folder_save = 'plotSlices'      # subsubfolder where the intermediate figures will be saved
name_fig    = name + '_plot_slice_mag_'
name_vid    = name + '_ani_mag'

##################################################################
## FUNCTIONS
##################################################################
def createCommand(commands):
    return (' '.join(commands))

def createFilePath(paths):
    return ('/'.join(paths) + '/')

def createFolder(folder_name):
    if not(os.path.exists(folder_name)):
        os.makedirs(folder_name)
        print('SUCCESS: \n\tFolder created. \n\t' + folder_name + '\n')
    else:
        print('WARNING: \n\tFolder already exists (folder not created). \n\t' + folder_name + '\n')

def meetCondition(element):
    global debug_mode, file_max
    if debug_mode:
        return bool(element.startswith('Turb_slice_xy_') and (int(element[-6:]) <= 5))
    elif file_max != np.Inf:
        return bool(element.startswith('Turb_slice_xy_') and (int(element[-6:]) <= file_max))
    else:
        return bool(element.startswith('Turb_slice_xy_'))

def setupInfo():
    global filepath_data, debug_mode
    ## save the the filenames to process
    file_names = list(filter(meetCondition, sorted(os.listdir(filepath_data))))
    ## check files
    if debug_mode:
        print('The files in filepath_data that satisfied meetCondition:\n\t' + '\n\t'.join(file_names) + '\n')
    ## return data
    return [file_names, len(file_names)]

def parPlot():
    global num_figs, filepath_plot, name_fig, num_proc
    ## open processor
    pool = mp.Pool(num_proc)
    pool.map(worker, range(num_figs))
    figs_complete = 0
    ## loop over data (in parallel)
    print('Started plotting.')
    while figs_complete < num_figs:
        ## PLOTTING CODE #########################
        ## receive the processed data from worker
        data, time_point, var_iter = data_queue.get()
        ## open figure
        fig = plt.figure(figsize=(10, 7), dpi=100)
        ax  = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ## plot data
        plt.imshow(data, 
            extent=(0.0, 1.0, 0.0, 1.0),
            interpolation='none',
            cmap='plasma',
            norm=LogNorm())
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
        fig_name = '/' + name_fig + '{0:06}'.format(var_iter)
        plt.savefig(filepath_plot + fig_name)
        print('Saved figure: %i'%int(fig_name[-6:]) + '  \t%0.3f%% complete'%(100 * figs_complete/num_figs))
        plt.close()
        figs_complete += 1
    pool.close()
    print('Finished plotting.')

def worker(var_iter):
    global debug_mode
    global filepath_data, file_names
    global t_eddy, plasma_beta
    file_name = filepath_data + '/' + file_names[var_iter]  # create the filepath to the file
    f = h5py.File(file_name, 'r')                           # load slice file
    time_point = np.array(f['time']) / t_eddy               # save time point
    if debug_mode:
        print('Looking at:\t' + file_name + '\ttime = %0.3f'%time_point)
    names    = [s for s in list(f.keys()) if 'mag' in s]    # save all keys that contain 'mag'
    data     = sum(np.array(f[i])**2 for i in names)        # determine the magnetic-field magnitude
    data    /= plasma_beta                                  # normalise the magnitude
    f.close()                                               # close the file stream
    data_queue.put([data, time_point, var_iter])            # send data to be plot

##################################################################
## SETUP VARIABLES
##################################################################
## announce debug status
if debug_mode:
    print('Debug mode is on.\n')
# initialise filepaths
global filepath_data, filepath_plot
global file_names, num_figs
filepath_data = createFilePath([filepath_base, folder_data]) # where data is stored
filepath_plot = createFilePath([filepath_base, folder_vis, folder_save]) # where plots will be saved
if debug_mode:
    print('Debug check:')
    print('The base directory is:\n\t' + filepath_base)
    print('The directory to the slice data is:\n\t' + filepath_data)
    print('The directory to where the figure will be saved is:\n\t' + filepath_plot + '\n')
## create folder where plots are saved
createFolder(filepath_plot)
## collect information about data
file_names, num_figs = setupInfo()

##################################################################
## PROCESS DATA and SAVE IMAGES
##################################################################
## process slice data and save figures
parPlot()

##################################################################
## ANIMATE IMAGES
##################################################################
filepath_input  = (filepath_plot + name_fig + '%06d.png')
filepath_output = (filepath_plot + '../' + name_vid + '.mp4')
ffmpeg_input    = ['ffmpeg -framerate 60 -i ' + filepath_input +                 # where the images are located
    ' -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p ' + filepath_output] # the video codec + filename
if not(debug_mode):
    os.system(createCommand(ffmpeg_input)) # ffmpeg -framerate 60 -i ./plotSlices/name_plot_slice_mag_%6d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p ./name_ani_mag.mp4
else:
    print('Input: \n\t' + filepath_input)
    print('Output: \n\t' + filepath_output)
    print('FFMPEG input: \n\t' + ffmpeg_input)

## END OF PROGRAM