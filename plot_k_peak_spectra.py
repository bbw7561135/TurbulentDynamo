#!/usr/bin/env python3

''' AUTHOR: Neco Kriel
    
    EXAMPLE: 
    plot_k_peak_spectra.py 
        (required)
            -base_path      $scratch/dyna288_Bk10/Re10/
            -pre_name       dyna288_Re10
        (optional)
            -debug          False
            -sub_folder     spectFiles
            -vis_folder     visFiles
            -file_start     0
            -file_end       np.Inf
'''

##################################################################
## MODULES
##################################################################
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib import ticker, cm
from matplotlib.colors import LogNorm
from statistics import stdev, mean

##################################################################
## PREPARE TERMINAL/WORKSPACE/CODE
#################################################################
os.system('clear')                  # clear terminal window
plt.close('all')                    # close all pre-existing plots
mpl.style.use('classic')            # plot in classic style

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
        Remove the occurance of the string 'var_remove' at both the start and end of the string 'var_string'.
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
        print('SUCCESS: \n\tFolder created. \n\t' + folder_name)
        print(' ')
    else:
        print('WARNING: \n\tFolder already exists (folder not created). \n\t' + folder_name)
        print(' ')

def setupInfo(filepath):
    ''' setupInfo
    PURPOSE:
        Collect filenames that will be processed and the number of these files
    '''
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
    ''' creatFilePath
    PURPOSE / OUTPUT:
        Turn an ordered list of names and concatinate them into a filepath.
    '''
    return ('/'.join([x for x in names if x != '']))

def meetsCondition(element):
    global bool_debug_mode, file_end, file_start
    ## accept files that look like: Turb_hdf5_plt_cnt_*(mags.dat or vels.dat)
    if (element.startswith('Turb_hdf5_plt_cnt_') and (element.endswith('mags.dat') or element.endswith('vels.dat'))):
        ## check that the file meets the minimum file number requirement
        bool_domain_upper = (int(element.split('_')[4]) >= file_start)
        if bool_debug_mode:
            ## return the first 5 files
            return bool(bool_domain_upper and (int(element.split('_')[4]) <= 5))
        elif file_end != np.Inf:
            ## return the files in the domain [file_start, file_end]
            return bool(bool_domain_upper and (int(element.split('_')[4]) <= file_end))
        else:
            ## return every file with a number greater than file_start
            return bool(bool_domain_upper)
    return False

def loadData(directory):
    filedata = open(directory).readlines() # load in data
    data     = np.array([x.strip().split() for x in filedata[6:]]) # store all data. index: data[row, col]
    data_x   = list(map(float, data[:, 1]))  # variable: wave number (k)
    data_y   = list(map(float, data[:, 15])) # variable: power spectrum
    return data_x, data_y

##################################################################
## INPUT COMMAND LINE ARGUMENTS
##################################################################
global file_end, bool_debug_mode, filepath_base, file_start
ap = argparse.ArgumentParser(description='A bunch of input arguments')
## ------------------- DEFINE OPTIONAL ARGUMENTS
ap.add_argument('-debug',      type=str2bool,   default=False,        required=False, help='Debug mode', nargs='?', const=True)
ap.add_argument('-sub_folder', type=str,        default='spectFiles', required=False, help='Name of the folder where the data is stored')
ap.add_argument('-vis_folder', type=str,        default='visFiles',   required=False, help='Name of the plot folder')
ap.add_argument('-file_start', type=int,        default=0,            required=False, help='First file to process')
ap.add_argument('-file_end',   type=int,        default=np.Inf,       required=False, help='Last file to process')
## ------------------- DEFINE REQUIRED ARGUMENTS
ap.add_argument('-base_path',  type=str,        required=True, help='Filepath to the base folder')
ap.add_argument('-pre_name',   type=str,        required=True, help='Name of figures')
## ---------------------------- OPEN ARGUMENTS
args = vars(ap.parse_args())
## ---------------------------- SAVE PARAMETERS
bool_debug_mode = args['debug']       # enable/disable debug mode
file_start      = args['file_start']  # starting processing frame
file_end        = args['file_end']    # the last file to process
## ---------------------------- SAVE FILEPATH PARAMETERS
filepath_base   = args['base_path']   # home directory
folder_vis      = args['vis_folder']  # subfolder where animation and plots will be saved
folder_sub      = args['sub_folder']  # sub-subfolder where data is stored's name
pre_name        = args['pre_name']    # name of figures
## ---------------------------- ADJUST ARGUMENTS
## remove the trailing '/' from the input filepath and plot folder
if filepath_base.endswith('/'):
    filepath_base = filepath_base[:-1]
## replace any '//' with '/'
filepath_base = filepath_base.replace('//', '/')
## remove '/' from variable names
folder_vis    = stringChop(folder_vis, '/')
folder_sub    = stringChop(folder_sub, '/')
pre_name      = stringChop(pre_name, '/')
## ---------------------------- START CODE
print('Began running the spectra plotting code in base filepath: \n\t' + filepath_base)
print('Visualising folder: '                                           + folder_vis)
print('Figure name: '                                                  + pre_name)
print(' ')

##################################################################
## USER VARIABLES
##################################################################
t_eddy = 10 # number of spectra files per eddy turnover # TODO: input?

##################################################################
## INITIALISING VARIABLES
##################################################################
filepath_data = createFilePath([filepath_base, folder_sub])
filepath_plot = createFilePath([filepath_base, folder_vis, 'plotSpectra']) # folder where plots will be saved
file_names, num_time_points = setupInfo(filepath_data)
createFolder(filepath_plot) # create folder where plots are saved

fig = plt.figure(figsize=(10, 7), dpi=100)

for cur_iter in range(num_time_points):
    #################### START OF LOOP
    ####################################
    cur_time = cur_iter/t_eddy # normalise time point by eddy-turnover time
    if ((100 * cur_iter/num_time_points) % 5 < 0.1):
        print('Loading data... %0.3f%% complete'%(100 * cur_iter/num_time_points))
    #################### LOAD DATA
    ##############################
    name_file_mag = 'Turb_hdf5_plt_cnt_' + '{0:04}'.format(cur_iter) + '_spect_mags.dat' # magnetic file
    data_x, data_y = loadData(filepath_data + '/' + name_file_mag) # magnetic power spectrum
    if (cur_iter == 0):
        #################### INITIALISE DATA
        ##############################
        ## initialise the list of time points
        time_points = []
        ## initialise the list of peak power spectrum values
        peak_power = []
    #################### ANALYSE DATA
    ##############################
    ## append the current time point
    time_points.append(cur_time)
    ## calculate the peak of the power spectra
    peak_power.append(max(data_y))

##################################################################
## PLOT DATA
##################################################################
# plot peak of magnetic power spectra
plt.plot(time_points, peak_power, 'k-')

##################################################################
## LABEL and ADJUST PLOT
##################################################################
print('\nLabelling plot...\n')
## scale axis
plt.yscale('log')
## label axies
plt.xlabel(r'$t/T$', fontsize=28, rotation=0)
plt.ylabel(r'$max(\mathcal{P}_{B}(k))$', fontsize=28, labelpad=30)
## major grid
plt.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.35)

##################################################################
## SAVE FIGURE
##################################################################
print('Saving figure...')
fig_name = createFilePath([filepath_plot, pre_name]) + '_k_peak_spectra.png'
plt.savefig(fig_name, bbox_inches='tight', dpi=300)
plt.close()
print('Figure saved: ' + fig_name)
print(' ')

## END OF PROGRAM