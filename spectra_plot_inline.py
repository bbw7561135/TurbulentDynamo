#!/usr/bin/env python3

''' AUTHOR: Neco Kriel
    
    EXAMPLE: 
    spectra_plot_inline.py 
        -base_path /Users/dukekriel/Documents/University/Year4Sem2/Summer-19/ANU-Turbulence-Dynamo/dyna288_Bk10 
        -pre_name dyna288_Bk10
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
    global bool_debug_mode, file_max, file_start
    ends_correct = (element.endswith('mags.dat') or element.endswith('vels.dat')) # check file is magnetic or velocity field
    if (ends_correct and (len(element.split('_')) >= 4)):
        number_correct = (int(element.split('_')[4]) >= file_start) # check file number is larger than file_start
        if bool_debug_mode:
            return bool(number_correct and ends_correct and (int(element.split('_')[4]) <= 5))
        elif file_max != np.Inf:
            return bool(number_correct and ends_correct and (int(element.split('_')[4]) <= file_max))
        else:
            return bool(number_correct and ends_correct)
    return False

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
## INPUT COMMAND LINE ARGUMENTS
##################################################################
global file_max, bool_debug_mode, filepath_base, file_start
ap = argparse.ArgumentParser(description='A bunch of input arguments')
## ------------------- DEFINE OPTIONAL ARGUMENTS
ap.add_argument('-debug',      required=False, help='Debug mode',                                   type=bool, default=False)
ap.add_argument('-sub_folder', required=False, help='Name of the folder where the data is stored',  type=str,  default='spectFiles')
ap.add_argument('-vis_folder', required=False, help='Name of the plot folder',                      type=str,  default='visFiles')
ap.add_argument('-fps',        required=False, help='Animation frame rate',                         type=str,  default='40')
ap.add_argument('-start_ani',  required=False, help='Start animation number',                       type=str,  default='0')
ap.add_argument('-start_file', required=False, help='File number to start plotting from',           type=int,  default=0)
ap.add_argument('-num_files',  required=False, help='Number of files to process',                   type=int,  default=-1)
## ------------------- DEFINE REQUIRED ARGUMENTS
ap.add_argument('-base_path', required=True, help='Filepath to the base folder', type=str)
ap.add_argument('-pre_name',  required=True, help='Name of figures', type=str)
## ------------------- OPEN ARGUMENTS
args = vars(ap.parse_args())
## ------------------- SAVE BOOLEANS
## enable/disable debug mode
if (args['debug'] == True):
    bool_debug_mode = True
else:
    bool_debug_mode = False
## ------------------- SAVE PROCESSING PARAMETERS
file_start    = args['start_file']  # starting processing frame
## ------------------- SAVE ANIMATION PARAMETERS
ani_start     = args['start_ani']   # starting animation frame
ani_fps       = args['fps']         # animation's fps
## the number of plots to process
if (args['num_files'] < 0):
    file_max = np.Inf
else:
    file_max = args['num_files']
## ------------------- SAVE FILEPATH PARAMETERS
filepath_base = args['base_path']   # home directory
folder_vis    = args['vis_folder']  # subfolder where animation and plots will be saved
folder_sub    = args['sub_folder']  # sub-subfolder where data is stored's name
pre_name      = args['pre_name']    # name of figures
## ------------------- ADJUST ARGUMENTS
## remove the trailing '/' from the input filepath and plot folder
if filepath_base.endswith('/'):
    filepath_base = filepath_base[:-1]
## replace '//' with '/'
filepath_base = filepath_base.replace('//', '/')
## remove '/' from variables
folder_vis    = stringChop(folder_vis, '/')
folder_sub    = stringChop(folder_sub, '/')
pre_name      = stringChop(pre_name, '/')
## ------------------- START CODE
print('Began running the spectra plotting code in base filepath: \n\t' + filepath_base)
print('Visualising folder: ' + folder_vis)
print('Figure name: ' + pre_name)
print(' ')

##################################################################
## USER VARIABLES
##################################################################
t_eddy = 10 # number of spectra files per eddy turnover # TODO: input?
## specify which variables you want to plot
global var_x, var_y
var_x = 1  # variable: wave number (k)
var_y = 15 # variable: power spectrum
label_kin = r'$\mathcal{P}_{k_{B}=10, \mathregular{kin}}$'
label_mag = r'$\mathcal{P}_{k_{B}=10, \mathregular{mag}}$'
## set the figure's axis limits
xlim_min = 1.0
xlim_max = 1.3e+02
ylim_min = 1.0e-25
ylim_max = 4.2e-03

##################################################################
## INITIALISING VARIABLES
##################################################################
filepath_data = createFilePath([filepath_base, folder_sub])
filepath_plot = createFilePath([filepath_base, folder_vis, 'plotSpectra']) # folder where plots will be saved
file_names, num_figs = setupInfo(filepath_data)
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
    data_x_kin, data_y_kin = loadData(filepath_data + '/' + name_file_kin) # kinetic power spectrum
    data_x_mag, data_y_mag = loadData(filepath_data + '/' + name_file_mag) # magnetic power spectrum

    #############################################
    ## PLOT DATA
    #############################################
    print('Plotting data...')
    line_kin, = plt.plot(data_x_kin, data_y_kin, 'k', label=label_kin) # kinetic power spectrum
    line_mag, = plt.plot(data_x_mag, data_y_mag, 'k--', label=label_mag) # magnetic power spectrum

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
    ## major grid
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.35)
    ## minor grid
    plt.grid(which='minor', linestyle='--', linewidth='0.5', color='black', alpha=0.2)

    #############################################
    ## SAVE IMAGE
    #############################################
    print('Saving figure...')
    temp_name = filepath_plot + pre_name + '_spectra={0:06}'.format(int(var_time*10)) + '.png'
    plt.savefig(temp_name)
    plt.close()
    print('Figure saved: ' + temp_name)
    print(' ')

## create animation
filepath_input  = (filepath_plot + pre_name + '_spectra=%06d.png')
filepath_output = (filepath_plot + '../' + pre_name + '_ani_spectra.mp4')
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