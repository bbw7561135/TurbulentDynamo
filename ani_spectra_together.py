#!/usr/bin/env python3

##################################################################
## MODULES
##################################################################
import os
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlibrc import *

##################################################################
## PREPARE TERMINAL/WORKSPACE/CODE
#################################################################
os.system('clear') # clear terminal window
plt.close('all')   # close all pre-existing plots

##################################################################
## FUNCTIONS
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
        Remove the occurance of the string 'var_remove' from both the start and end of the string 'var_string'.
    '''
    if var_string.endswith(var_remove):
        var_string = var_string[:-len(var_remove)]
    if var_string.startswith(var_remove):
        var_string = var_string[len(var_remove):]
    return var_string

def createFolder(folder_name):
    ''' createFolder
    PURPOSE:
        Create the folder passed as a filepath to inside the folder.
    OUTPUT:
        Commandline output of the success/failure status of creating the folder.
    '''
    if not(os.path.exists(folder_name)):
        os.makedirs(folder_name)
        print('SUCCESS: Folder created. \n\t' + folder_name)
        print(' ')
    else:
        print('WARNING: Folder already exists (folder not created). \n\t' + folder_name)
        print(' ')

def setupInfo(filepath):
    ''' setupInfo
    PURPOSE:
        Collect filenames that will be processedm and the number of these files
    '''
    global bool_debug_mode
    ## save the the filenames to process
    file_names = list(filter(meetsCondition, sorted(os.listdir(filepath))))
    print('Filepath: ' + filepath)
    print('Number of sepctra files: ' + str(len(file_names)/2))
    print(' ')
    ## check files
    if bool_debug_mode:
        print('The files in the filepath:')
        print('\t' + filepath)
        print('\tthat satisfied meetCondition are the files:')
        print('\t\t' + '\n\t\t'.join(file_names))
        print(' ')
    ## return data
    return int(file_names[-1].split('_')[-3])

def createFilePath(names):
    ''' creatFilePath
    PURPOSE / OUTPUT:
        Turn an ordered list of names and concatinate them into a filepath.
    '''
    return ('/'.join([x for x in names if x != '']))

def meetsCondition(element):
    global bool_debug_mode, file_end, file_start
    ## accept files that look like: Turb_hdf5_plt_cnt_*(mags.dat or vels.dat)
    if (element.__contains__('Turb_hdf5_plt_cnt_') and (element.endswith('mags.dat') or element.endswith('vels.dat'))):
        ## check that the file meets the minimum file number requirement
        bool_domain_upper = (int(element.split('_')[-3]) >= file_start)
        if bool_debug_mode:
            ## return the first 5 files
            return bool(bool_domain_upper and (int(element.split('_')[-3]) <= 5))
        elif file_end != np.Inf:
            ## return the files in the domain [file_start, file_end]
            return bool(bool_domain_upper and (int(element.split('_')[-3]) <= file_end))
        else:
            ## return every file with a number greater than file_start
            return bool(bool_domain_upper)
    return False

def loadData(directory):
    global bool_debug_mode, var_y
    filedata     = open(directory).readlines() # load in data
    header       = filedata[5].split() # save the header
    data         = np.array([x.strip().split() for x in filedata[6:]]) # store all data. index: data[row, col]
    if bool_debug_mode:
        print('\nHeader names: for ' + directory.split('/')[-1])
        print('\n'.join(header)) # print all header names (with index)
    data_y = list(map(float, data[:, var_y]))
    return data_y

def plot_k_0(k_0):
    global ylim_min, ylim_max
    plt.plot([k_0, k_0], [ylim_min, ylim_max], 'k--')
    plt.text(k_0*0.95, ylim_max*0.8, r'$k_0$', ha='right', va='bottom', fontsize=28)

def plot_k_nu(k_0, Re, var_str):
    global ylim_min, ylim_max
    k_nu = Re**(3/4) * k_0
    plt.plot([k_nu, k_nu], [ylim_min, ylim_max], var_str)
    plt.text(k_nu*1.05, ylim_max*0.8, r'$k_\nu$', ha='left', va='bottom', fontsize=28)

def plot_k_eta(k_0, Re, Pm, var_str):
    global ylim_min, ylim_max
    k_eta = Re**(1/4) * (Re*Pm)**(1/2) * k_0
    plt.plot([k_eta, k_eta], [ylim_min, ylim_max], var_str)
    plt.text(k_eta*0.95, ylim_max*0.8, r'$k_\eta$', ha='right', va='bottom', fontsize=28)

def calc_moving_ave(data_vals, window_size):
    dy_moving_ave = []
    for tmp_index in range(len(data_vals)):
        if tmp_index < window_size:
            dy_moving_ave.append(data_vals[tmp_index])
        else:
            ## calculate moving average of the magnetic spectra peak
            dy_moving_ave.append(sum(data_vals[tmp_index : tmp_index + window_size]) / window_size)
    return dy_moving_ave

def plotData(data_x_mag, data_y_mag, label_mag, col_str):
    if ~all(v == 0 for v in data_y_mag):
        plt.plot(data_x_mag, data_y_mag, color=col_str, linestyle='--', label=label_mag)
        max_mag = max(data_y_mag)
        plt.plot(data_x_mag[data_y_mag.index(max_mag)], max_mag, color=col_str, marker='.', markersize=10)

##################################################################
## INPUT COMMAND LINE ARGUMENTS
##################################################################
global bool_debug_mode
global filepath_base, file_start, file_end
ap = argparse.ArgumentParser(description='A bunch of input arguments')
## ------------------- DEFINE (OPTIONAL) ARGUMENTS
ap.add_argument('-debug',    type=str2bool, default=False, required=False, help='Debug mode',             nargs='?', const=True)
ap.add_argument('-plot_kin', type=str2bool, default=True,  required=False, help='Plot kinetic spectra?',  nargs='?', const=False)
ap.add_argument('-plot_mag', type=str2bool, default=True,  required=False, help='Plot magnetic spectra?', nargs='?', const=False)
ap.add_argument('-vis_folder', type=str, default='visFiles',   required=False, help='Name of the plot folder')
ap.add_argument('-sub_folder', type=str, default='spectFiles', required=False, help='Name of the folder where the data is stored')
ap.add_argument('-file_start', type=int, default=0,            required=False, help='File number to start plotting from')
ap.add_argument('-file_end',   type=int, default=np.Inf,       required=False, help='Number of files to process')
## ------------------- DEFINE (REQUIRED) ARGUMENTS
ap.add_argument('-base_path',   type=str, required=True, help='Base filepath to all the folders')
ap.add_argument('-dat_folders', type=str, required=True, help='List of folders with data', nargs='+')
ap.add_argument('-dat_labels',  type=str, required=True, help='Data labels', nargs='+')
ap.add_argument('-pre_name',    type=str, required=True, help='Figure name')
## ---------------------------- OPEN ARGUMENTS
args = vars(ap.parse_args())
## ---------------------------- SAVE PARAMETERS
bool_debug_mode = args['debug']       # enable/disable debug mode
bool_plot_kin   = args['plot_kin']    # plot the kinetic spectra?
bool_plot_mag   = args['plot_mag']    # plot the magnetic spectra?
file_start      = args['file_start']  # starting processing frame
file_end        = args['file_end']    # the last file to process
## ---------------------------- SAVE FILEPATH PARAMETERS
filepath_base   = args['base_path']   # home directory
folders_data    = args['dat_folders'] # list of subfolders where each simulation's data is stored
labels_data     = args['dat_labels']  # list of labels for plots
folder_sub      = args['sub_folder']  # sub-subfolder where actual data is stored
folder_vis      = args['vis_folder']  # subfolder where animation and plots will be saved
pre_name        = args['pre_name']    # name of figures
## ---------------------------- ADJUST ARGUMENTS
# ## remove the trailing '/' from the input filepath and folders
if filepath_base.endswith('/'):
    filepath_base = filepath_base[:-1]
## replace any '//' with '/'
filepath_base   = filepath_base.replace('//', '/')
## remove '/' from variable names
folder_sub    = stringChop(folder_sub, '/')
folder_vis    = stringChop(folder_vis, '/')
pre_name      = stringChop(pre_name, '/')
for i in range(len(folders_data)): 
    folders_data[i] = stringChop(folders_data[i], '/')

##################################################################
## USER VARIABLES
##################################################################
## number of spectra files per eddy turnover
t_eddy = 10
## set the figure's axis limits
global ylim_min, ylim_max
xlim_min = 1
xlim_max = 1.3e+02
ylim_min = 1.0e-25
ylim_max = 1
## specify which variables you want to plot
global var_x, var_y
var_x = 1  # variable: wave number (k)
var_y = 15 # variable: power spectrum

##################################################################
## INITIALISING VARIABLES
##################################################################
filepaths_data = []
## create the filepaths to the data
for i in range(len(folders_data)):
    filepaths_data.append(createFilePath([filepath_base, folders_data[i], folder_sub]).replace('//', '/'))
## create the folder where plots will be saved
filepath_plot = createFilePath([filepath_base, folder_vis, 'plotSpectra']).replace('//', '/')
createFolder(filepath_plot)
## ---------------------------- START CODE
print('Base filepath: \t\t'                  + filepath_base)
for i in range(len(filepaths_data)): 
    print('Data folder ' + str(i) + ': \t\t' + filepaths_data[i])
print('Figure folder: \t\t'                  + filepath_plot)
print('Figure name: \t\t'                    + pre_name)
print(' ')

##################################################################
## CREATE FIGURES
##################################################################
## count the number of frames to animate
num_figs = np.nan
for i in range(len(folders_data)):
    num_figs = np.nanmax([num_figs, setupInfo(filepaths_data[i])])
## create and save each frame of the animation
for fig_index in range(1, int(num_figs)):
    #################### INITIALISE LOOP
    ####################################
    fig, ax = plt.subplots(constrained_layout=True)
    ## normalise time point by eddy-turnover time
    var_time = fig_index/t_eddy
    ## let the user know how the animation is progressing
    print('Processing: %0.3f%% complete'%(100 * fig_index/num_figs))
    #################### LOAD & PLOT DATA
    ##############################
    name_file_kin = 'Turb_hdf5_plt_cnt_' + '{0:04}'.format(fig_index) + '_spect_vels.dat' # kinetic file
    name_file_mag = 'Turb_hdf5_plt_cnt_' + '{0:04}'.format(fig_index) + '_spect_mags.dat' # magnetic file
    for i in range(len(filepaths_data)):
        ## load velocity spectra
        if bool_plot_kin:
            file_path = createFilePath([filepaths_data[i], name_file_kin])
            ## check if file exists
            if os.path.isfile(file_path):
                ## load data
                data_y_kin = loadData(file_path)
                ## plot data
                print('\tPlotting: ' + file_path)
                plt.plot(range(1, len(data_y_kin)+1), data_y_kin,
                    color=sns.color_palette("PuBu", n_colors=len(filepaths_data))[i], 
                    linestyle='-', linewidth=2)
        ## load magnetic spectra
        if bool_plot_mag:
            file_path = createFilePath([filepaths_data[i], name_file_mag])
            ## check if file exists
            if os.path.isfile(file_path):
                ## load data
                data_y_mag = loadData(file_path)
                ## plot data
                print('\tPlotting: ' + file_path)
                plt.plot(range(1, len(data_y_mag)+1), data_y_mag, 
                    label=labels_data[i], 
                    color=sns.color_palette("OrRd", n_colors=len(filepaths_data))[i], 
                    linestyle='--', linewidth=2)
    # add legend
    plt.legend(loc='lower center', ncol=3, fontsize=17, frameon=False)
    #################### LABEL and ADJUST PLOT
    ##########################################
    ## scale axies
    plt.xscale('log')
    plt.yscale('log')
    ## set axis limits
    plt.xlim(xlim_min, xlim_max)
    plt.ylim(ylim_min, ylim_max)
    ## annote time (eddy tunrover-time)
    title = plt.annotate(r'$t/t_{\mathrm{eddy}} = $' + u'%0.2f'%(var_time),
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
    #################### SAVE IMAGE
    ###############################
    print('Saving figure...')
    temp_name = createFilePath([filepath_plot, (pre_name + '_spectra={0:06}'.format(int(var_time*10)) + '.png')])
    print(temp_name)
    plt.savefig(temp_name)
    plt.close()
    print('Figure saved: ' + temp_name)
    print(' ')

## create animation
filepath_input  = createFilePath([filepath_plot, (pre_name + '_spectra=%06d.png')])
filepath_output = createFilePath([filepath_plot, ('../' + pre_name + '_ani_spectras_combined.mp4')])
ffmpeg_input    = ('ffmpeg -start_number 0 -i '                           + filepath_input + 
                ' -vb 40M -framerate 40 -vf scale=1440:-1 -vcodec mpeg4 ' + filepath_output)
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
    # eg. To check: execute the following within the visualising folder
    # ffmpeg -start_number 0 -i ./plotSlices/dyna288_spectra=%06d.png -vb 40M -framerate 40 -vf scale=1440:-1 -vcodec mpeg4 ./dyna288_ani_spectra.mp4

## END OF PROGRAM