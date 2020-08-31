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

#################################################################
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

def createFilePath(names):
    ''' creatFilePath
    PURPOSE / OUTPUT:
        Turn an ordered list of names and concatinate them into a filepath.
    '''
    return ('/'.join([x for x in names if x != '']))

def loadTurbDat(filepath):
    ''' loadTurbDat
    PURPOSE:
        Load and process the Turb.dat data located in 'filepath'. 
    OUTPUT:
        x (time), y data and the name of the y-axis data
    '''
    global t_eddy, var_x, var_y
    ## load data
    print('Loading data...')
    filepath_turb = createFilePath([filepath, 'Turb.dat'])
    first_line = open(filepath_turb).readline().split()
    len_thresh = len(first_line)
    ## save x and y data
    data_x = []
    data_y = []
    prev_time = -1
    with open(filepath_turb) as file_lines:
        for line in file_lines:
            data_split = line.split()
            if len(data_split)  == len_thresh:
                if (not(data_split[var_x][0] == '#') and not(data_split[var_y][0] == '#')):
                    cur_time = float(data_split[var_x]) / t_eddy
                    ## if the simulation has been restarted, make sure that only the progressed data is used
                    if cur_time > prev_time:
                        data_x.append(cur_time) # normalise time-domain
                        data_y.append(float(data_split[var_y]))
                        prev_time = cur_time
    ## return variables
    return data_x, data_y

##################################################################
## INPUT COMMAND LINE ARGUMENTS
##################################################################
global bool_debug_mode, bool_norm_wrt_self, filepath_base
ap = argparse.ArgumentParser(description='A bunch of input arguments')
## ------------------- DEFINE OPTIONAL ARGUMENTS
ap.add_argument('-debug',     type=str2bool, default=False, required=False, help='Debug mode', nargs='?', const=True)
ap.add_argument('-norm_wrt_self', type=str2bool, default=False, required=False, help='Normalise each dataset wrt itself', nargs='?', const=True)
ap.add_argument('-vis_folder',  type=str,       default='visFiles', required=False, help='Name of the plot folder')
ap.add_argument('-max_time',    type=int,       default=-1,         required=False, help='Maximum plotted time')
## ------------------- DEFINE REQUIRED ARGUMENTS
ap.add_argument('-base_path',   type=str, required=True, help='Filepath to the base folder')
ap.add_argument('-dat_folders', type=str, required=True, help='List of folders with data', nargs='+')
ap.add_argument('-dat_labels',  type=str, required=True, help='Data labels', nargs='+')
ap.add_argument('-pre_name',    type=str, required=True, help='Name of figures')
## ---------------------------- OPEN ARGUMENTS
args = vars(ap.parse_args())
## ---------------------------- SAVE PARAMETERS
bool_debug_mode = args['debug']     # enable/disable debug mode
bool_norm_wrt_self = args['norm_wrt_self'] # should each dataset be normalised wrt its first value
## ---------------------------- SAVE FILEPATH PARAMETERS
filepath_base = args['base_path']   # home directory
folders_data  = args['dat_folders'] # list of subfolders where each simulation's data is stored
labels_data   = args['dat_labels']  # list of labels for plots
folder_plot   = args['vis_folder']  # subfolder where animation and plots will be saved
pre_name      = args['pre_name']    # pre_name of figures
max_time      = args['max_time']    # maximum plotted time
## ---------------------------- ADJUST ARGUMENTS
## remove the trailing '/' from the input filepath/folders
if filepath_base.endswith('/'):
    filepath_base = filepath_base[:-1]
## replace any '//' with '/'
filepath_base = filepath_base.replace('//', '/')
## remove '/' from variable names
folder_plot   = stringChop(folder_plot, '/')
pre_name      = stringChop(pre_name, '/')
for i in range(len(folders_data)): 
    folders_data[i] = stringChop(folders_data[i], '/')

##################################################################
## DEFINE PLOTTING VARIABLES
##################################################################
global t_eddy, var_x, var_y
## constants
t_eddy           = 5 # L/(2*Mach)
var_x            = 0 # time
## ------------------- GET USER INPUT
## accept input for the y-axis variable
print('Which variable do you want to plot on the y-axis?')
print('\tOptions: 6 (E_kin), 8 (rms_Mach), 29 (E_mag)')
var_y = int(input('\tInput: '))
while ((var_y != 6) and (var_y != 8) and (var_y != 29)):
    print('\tInvalid input. Choose an option from: 6 (E_kin), 8 (rms_Mach), 29 (E_mag)')
    var_y = int(input('\tInput: '))
print(' ')
## initialise variables
var_scale        = ''
label_y          = r''
if var_y == 6:
    ## mach number
    label_y       = r'$E_{\nu}/E_{\nu 0}$'
    bool_norm_dat = bool(1)
    var_scale     = 'log'
    var_name      = 'E_kin'
elif var_y == 8:
    ## mach number
    label_y       = r'$\mathcal{M}$'
    bool_norm_dat = bool(0)
    var_scale     = 'linear'
    var_name      = 'rms_Mach'
else:
    ## magnetic field
    label_y       = r'$E_{B}/E_{B 0}$'
    bool_norm_dat = bool(1)
    var_scale     = 'log'
    var_name      = 'E_mag'

##################################################################
## INITIALISING VARIABLES
##################################################################
## create the filepaths to data
filepaths_data = []
for i in range(len(folders_data)):
    filepaths_data.append(createFilePath([filepath_base, folders_data[i]]))
## create folder where the figure will be saved
filepath_plot = createFilePath([filepath_base, folder_plot])
createFolder(filepath_plot)
## print information to screen
print('Base filepath: \t\t'                  + filepath_base)
for i in range(len(filepaths_data)): 
    print('Data folder ' + str(i) + ': \t\t' + filepaths_data[i])
print('Figure folder: \t\t'                  + filepath_plot)
print('Figure name: \t\t'                    + pre_name)
print(' ')
## create figure
fig, ax = plt.subplots(constrained_layout=True)

##################################################################
## LOAD & PLOT DATA
##################################################################
for i in range(len(filepaths_data)):
    #################### LOADING DATA
    ##############################
    print('Loading data from: ' + filepaths_data[i])
    data_x, data_y = loadTurbDat(filepaths_data[i])
    if bool_norm_dat:
        if i == 0:
            first_data_y = data_y[1]
        if bool_norm_wrt_self:
            data_y = [i / data_y[1] for i in data_y]
        else: 
            data_y = [i / first_data_y for i in data_y]
    #################### PLOTTING DATA
    ##############################
    print('Plotting data...')
    if (max_time < 0):
        plt.plot(data_x, data_y, 
            color=sns.color_palette("PuBu", n_colors=len(filepaths_data))[i], 
            linewidth=2, label=labels_data[i])
    else:
        max_time_1 = np.abs(np.asarray(data_x) - max_time).argmin()
        plt.plot(data_x[:max_time_1], data_y[:max_time_1], 
            color=sns.color_palette("PuBu", n_colors=len(filepaths_data))[i], 
            linewidth=2, label=labels_data[i])
    print(' ')

##################################################################
## LABEL and ADJUST PLOT
##################################################################
print('Labelling plot...')
## major grid
ax.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.35)
## add legend
ax.legend(loc='lower right', facecolor='white', framealpha=1, fontsize=20)
# ## label plot
ax.set_xlabel(r'$t / t_\mathrm{eddy}$', fontsize=22)
ax.set_ylabel(label_y, fontsize=22)
## scale y-axis
ax.set_yscale(var_scale)

##################################################################
## SAVE IMAGE
##################################################################
print('Saving the figure...')
name_fig = filepath_plot + '/' + pre_name + '_' + var_name + '.pdf'
plt.savefig(name_fig)
plt.close()
print('Figure saved: ' + name_fig)
print(' ')

## END OF PROGRAM