#!/usr/bin/env python3

''' AUTHOR: Neco Kriel

    EXAMPLE: 
    spectra_dPlot_inline.py -base_path /Users/dukekriel/Documents/University/Year4Sem2/Summer-19/ANU-Turbulence-Dynamo -dat_folder1 dyna288_Bk10 -dat_folder2 dyna288_Bk100 -vis_folder testPlots -fig_name dyna288
'''

##################################################################
## MODULES
##################################################################
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

##################################################################
## PREPARE TERMINAL/WORKSPACE/CODE
#################################################################
os.system('clear')                  # clear terminal window
plt.close('all')                    # close all pre-existing plots
mpl.style.use('classic')            # plot in classic style

##################################################################
## INPUT COMMAND LINE ARGUMENTS
##################################################################
global file_max, bool_debug_mode, filepath_base
ap = argparse.ArgumentParser(description='A bunch of input arguments')
## ------------------- OPTIONAL ARGUMENTS
ap.add_argument('-debug', required=False, help='Debug mode', type=bool, default=False)
ap.add_argument('-num_files', required=False, help='Number of files to process', type=int, default=-1)
ap.add_argument('-start', required=False, help='Start frame number', type=str, default='0')
ap.add_argument('-fps', required=False, help='Animation frame rate', type=str, default='40')
## ------------------- REQUIRED ARGUMENTS
ap.add_argument('-base_path', required=True, help='Filepath to the base folder', type=str)
ap.add_argument('-dat_folder1', required=True, help='Name of the first folder', type=str)
ap.add_argument('-dat_folder2', required=True, help='Name of the second folder', type=str)
ap.add_argument('-vis_folder', required=True, help='Name of the folder where the figures will be saved', type=str)
ap.add_argument('-fig_name', required=True, help='Name of figures', type=str)
## save arguments
args = vars(ap.parse_args())
## ------------------- BOOLEANS
## enable/disable debug mode
if (args['debug'] == True):
    bool_debug_mode = True
else:
    bool_debug_mode = False
## ------------------- ANIMATION PARAMETERS
## save required arguments
ani_start     = args['start']       # starting animation frame
ani_fps       = args['fps']         # animation's fps
## the number of plots to process
if (args['num_files'] < 0):
    file_max = np.Inf
else:
    file_max = args['num_files']
## ------------------- FILEPATH PARAMETERS
filepath_base = args['base_path']   # home directory
folder_data_1 = args['dat_folder1'] # first subfolder's name
folder_data_2 = args['dat_folder2'] # second subfolder's name
folder_vis    = args['vis_folder']  # subfolder where animation and plots will be saved
fig_name      = args['fig_name']    # name of figures
## remove the trailing '/' from the input filepath and folders
if filepath_base.endswith('/'):
    filepath_base = filepath_base[:-1]
if folder_data_1.endswith('/'):
    folder_data_1 = folder_data_1[:-1]
if folder_data_2.endswith('/'):
    folder_data_2 = folder_data_2[:-1]
if folder_vis.endswith('/'):
    folder_vis = folder_vis[:-1]
## start code
print('Began running the spectra plotting code in base filepath: \n\t' + filepath_base)
print('Data folder 1: ' + folder_data_1)
print('Data folder 2: ' + folder_data_2)
print('Visualising folder: ' + folder_vis)
print(' ')

##################################################################
## USER VARIABLES
##################################################################
t_eddy = 10 # number of spectra files per eddy turnover # TODO: input?
## specify where files are located and needs to be saved
## first plot's information
label_1_kin = r'$\mathcal{P}_{k_{B}=10, \mathregular{kin}}$'
label_1_mag = r'$\mathcal{P}_{k_{B}=10, \mathregular{mag}}$'
## second plot's information
label_2_kin = r'$\mathcal{P}_{k_{B}=100, \mathregular{kin}}$'
label_2_mag = r'$\mathcal{P}_{k_{B}=100, \mathregular{mag}}$'
## specify which variables you want to plot
global var_x, var_y
var_x = 1  # variable: wave number (k)
var_y = 15 # variable: power spectrum
## set the figure's axis limits
xlim_min = 1.0
xlim_max = 1.3e+02
ylim_min = 1.0e-25
ylim_max = 4.2e-03

##################################################################
## FUNCTIONS
##################################################################
def createFolder(folder_name):
    if not(os.path.exists(folder_name)):
        os.makedirs(folder_name)
        print('SUCCESS: \n\tFolder created. \n\t' + folder_name)
        print(' ')
    else:
        print('WARNING: \n\tFolder already exists (folder not created). \n\t' + folder_name)
        print(' ')

def setupInfo(filepath):
    global bool_debug_mode
    ## save the the filenames to process
    file_names = list(filter(meetsCondition, sorted(os.listdir(filepath))))
    ## check files
    if bool_debug_mode:
        print('The files in the filepath:')
        print('\t' + filepath)
        print('\tthat satisfied meetCondition are the files:')
        print('\t\t' + '\n\t\t'.join(file_names))
        print(' ')
    ## return data
    return [file_names, int(len(file_names)/2)]

def createFilePath(names):
    return ('/'.join([x for x in names if x != '']) + '/')

def meetsCondition(element):
    global bool_debug_mode, file_max
    ends_correct = (element.endswith('mags.dat') or element.endswith('vels.dat'))
    if bool_debug_mode:
        return bool(ends_correct and (int(element.split('_')[4]) <= 5))
    elif file_max != np.Inf:
        return bool(ends_correct and (int(element.split('_')[4]) <= file_max))
    else:
        return bool(ends_correct)

def loadData(directory):
    global bool_debug_mode, var_x, var_y
    filedata     = open(directory).readlines() # load in data
    header       = filedata[5].split() # save the header
    data         = np.array([x.strip().split() for x in filedata[6:]]) # store all data. index: data[row, col]
    if bool_debug_mode:
        print('\nHeader names: for ' + directory.split('/')[-1])
        print('\n'.join(header)) # print all header names (with index)
    data_x = list(map(float, data[:, var_x]))
    data_y = list(map(float, data[:, var_y]))
    return data_x, data_y

##################################################################
## INITIALISING VARIABLES
##################################################################
filepath_data_1 = createFilePath([filepath_base, folder_data_1, 'spectFiles']) # first folder with data
filepath_data_2 = createFilePath([filepath_base, folder_data_2, 'spectFiles']) # second folder with data
filepath_plot   = createFilePath([filepath_base, folder_vis, 'plotSpectra']) # folder where plots will be saved
file_names_1, num_figs_1 = setupInfo(filepath_data_1)
file_names_2, num_figs_2 = setupInfo(filepath_data_2)
num_figs = min(num_figs_1, num_figs_2)
createFolder(filepath_plot) # create folder where plots are saved

for var_iter in range(num_figs):
    #############################################
    ## INITIALISE LOOP
    #############################################
    fig = plt.figure(figsize=(10, 7), dpi=100)
    var_time = var_iter/t_eddy # normalise time point by eddy-turnover time
    print('Processing: %0.3f%% complete'%(100 * var_iter/num_figs))

    #############################################
    ## LOAD DATA
    #############################################
    print('Loading data...')
    name_file_kin = 'Turb_hdf5_plt_cnt_' + '{0:04}'.format(var_iter) + '_spect_vels.dat' # kinetic file
    name_file_mag = 'Turb_hdf5_plt_cnt_' + '{0:04}'.format(var_iter) + '_spect_mags.dat' # magnetic file
    ## load dataset 1
    data_x_1_kin, data_y_1_kin = loadData(filepath_data_1 + '/' + name_file_kin) # kinetic power spectrum
    data_x_1_mag, data_y_1_mag = loadData(filepath_data_1 + '/' + name_file_mag) # magnetic power spectrum
    ## load dataset 2
    data_x_2_kin, data_y_2_kin = loadData(filepath_data_2 + '/' + name_file_kin) # kinetic power spectrum
    data_x_2_mag, data_y_2_mag = loadData(filepath_data_2 + '/' + name_file_mag) # magnetic power spectrum

    #############################################
    ## PLOT DATA
    #############################################
    print('Plotting data...')
    ## plot dataset 1
    line_1_kin, = plt.plot(data_x_1_kin, data_y_1_kin, 'k', label=label_1_kin) # kinetic power spectrum
    line_1_mag, = plt.plot(data_x_1_mag, data_y_1_mag, 'k--', label=label_1_mag) # magnetic power spectrum
    ## plot dataset 2
    line_1_kin, = plt.plot(data_x_2_kin, data_y_2_kin, 'b', label=label_2_kin) # kinetic power spectrum
    line_1_mag, = plt.plot(data_x_2_mag, data_y_2_mag, 'b--', label=label_2_mag) # magnetic power spectrum

    #############################################
    ## LABEL and ADJUST PLOT
    #############################################
    print('Labelling plot...')
    ## scale axies
    plt.xscale('log')
    plt.yscale('log')
    ## set axis limits
    plt.xlim(xlim_min, xlim_max)
    plt.ylim(ylim_min, ylim_max)
    ## annote time (eddy tunrover-time)
    title = plt.annotate(r'$t/t_{\mathregular{eddy}} = $' + u'%0.2f'%(var_time),
                    xy=(0.5, 0.95),
                    fontsize=20, color='black', 
                    ha='center', va='top', xycoords='axes fraction')
    # label plots
    plt.xlabel(r'$k$',           fontsize=20)
    plt.ylabel(r'$\mathcal{P}$', fontsize=20)
    # add legend
    plt.legend(loc='upper right', fontsize=17, frameon=False)
    ## major grid
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.35)
    ## minor grid
    plt.grid(which='minor', linestyle='--', linewidth='0.5', color='black', alpha=0.2)

    #############################################
    ## SAVE IMAGE
    #############################################
    print('Saving figure...')
    temp_name = filepath_plot + fig_name + '_spectra={0:06}'.format(int(var_time*10)) + '.png'
    plt.savefig(temp_name)
    plt.close()
    print('Figure saved: ' + temp_name)
    print(' ')

## create animation
filepath_input  = (filepath_plot + fig_name + '_spectra=%06d.png')
filepath_output = (filepath_plot + '../' + fig_name + '_ani_spectra.mp4')
ffmpeg_input    = ('ffmpeg -start_number '          + ani_start + 
                ' -i '                              + filepath_input + 
                ' -vb 40M -framerate '              + ani_fps + 
                ' -vf scale=1440:-1 -vcodec mpeg4 ' + filepath_output)
if bool_debug_mode:
    print('--------- Debug: Check FFMPEG input -----------------------------------')
    print('Input: \n\t' + filepath_input)
    print('Output: \n\t' + filepath_output)
    print('FFMPEG input: \n\t' + ffmpeg_input)
    print(' ')
else:
    print('Animating plots...')
    os.system(ffmpeg_input) 
    print('Animation finished: ' + filepath_output)
    # eg. From within visualising subfolder
    # ffmpeg -start_number 0 -i ./plotSlices/dyna288_spectra=%06d.png -vb 40M -framerate 40 -vf scale=1440:-1 -vcodec mpeg4 ./dyna288_ani_spectra.mp4

## END OF PROGRAM