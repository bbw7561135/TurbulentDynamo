#!/usr/bin/env python3

''' AUTHOR: Neco Kriel
    
    EXAMPLE: 
    turb_dPlot_inline.py 
        (required)
            -base_path      /Users/dukekriel/Documents/University/Year4Sem2/Summer-19/ANU-Turbulence-Dynamo 
            -dat_folder1    dyna288_Bk10 
            -dat_folder2    dyna288_Bk100 
            -pre_name       dyna288
        (optional)
            -debug          False
            -vis_folder     visFiles
'''

##################################################################
## MODULES
##################################################################
import os
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#################################################################
## PREPARE TERMINAL/WORKSPACE/CODE
#################################################################
os.system('clear')       # clear terminal window
plt.close('all')         # close all pre-existing plots
mpl.style.use('classic') # plot in classic style

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
    global t_eddy, bool_norm_dat, var_x, var_y
    ## load data
    print('Loading data...')
    filepath_turb = createFilePath([filepath, 'Turb.dat'])
    first_line = open(filepath_turb).readline().split()
    len_thresh = len(first_line)
    ## save x and y data
    data_x = []
    data_y = []
    with open(filepath_turb) as file_lines:
        for line in file_lines:
            data_split = line.split()
            if len(data_split)  == len_thresh:
                if (not(data_split[var_x][0] == '#') and not(data_split[var_y][0] == '#')):
                    data_x.append(float(data_split[var_x]) / t_eddy) # normalise time-domain
                    data_y.append(float(data_split[var_y]))
    if bool_norm_dat:
        # y_data = var_y / var_y[1]
        data_y = [i / data_y[1] for i in data_y]
    ## return variables
    return [data_x, data_y, first_line[var_y][4:]]

##################################################################
## INPUT COMMAND LINE ARGUMENTS
##################################################################
global bool_debug_mode, filepath_base
ap = argparse.ArgumentParser(description='A bunch of input arguments')
## ------------------- DEFINE OPTIONAL ARGUMENTS
ap.add_argument('-debug',       type=str2bool,  default=False,      required=False, help='Debug mode', nargs='?', const=True)
ap.add_argument('-vis_folder',  type=str,       default='visFiles', required=False, help='Name of the plot folder')
## ------------------- DEFINE REQUIRED ARGUMENTS
ap.add_argument('-base_path',   type=str, required=True, help='Filepath to the base folder')
ap.add_argument('-dat_folder1', type=str, required=True, help='Name of the first data folder')
ap.add_argument('-dat_folder2', type=str, required=True, help='Name of the second data folder')
ap.add_argument('-pre_name',    type=str, required=True, help='Name of figures')
## ---------------------------- OPEN ARGUMENTS
args = vars(ap.parse_args())
## ---------------------------- SAVE PARAMETERS
bool_debug_mode = args['debug']     # enable/disable debug mode
## ---------------------------- SAVE FILEPATH PARAMETERS
filepath_base = args['base_path']   # home directory
folder_data_1 = args['dat_folder1'] # first subfolder's pre_name
folder_data_2 = args['dat_folder2'] # second subfolder's pre_name
folder_plot   = args['vis_folder']  # subfolder where animation and plots will be saved
pre_name      = args['pre_name']    # pre_name of figures
## ---------------------------- ADJUST ARGUMENTS
## remove the trailing '/' from the input filepath/folders
if filepath_base.endswith('/'):
    filepath_base = filepath_base[:-1]
## replace any '//' with '/'
filepath_base = filepath_base.replace('//', '/')
## remove '/' from variable names
folder_data_1 = stringChop(folder_data_1, '/')
folder_data_2 = stringChop(folder_data_2, '/')
folder_plot   = stringChop(folder_plot, '/')
pre_name      = stringChop(pre_name, '/')
## ---------------------------- START CODE
print('Began running the spectra plotting code in the filepath: \n\t' + filepath_base)
print('Data folder 1: '                                               + folder_data_1)
print('Data folder 2: '                                               + folder_data_2)
print('Visualising folder: '                                          + folder_plot)
print('Figure name: '                                                 + pre_name)
print(' ')

##################################################################
## DEFINE PLOTTING VARIABLES
##################################################################
global t_eddy, var_x, var_y, bool_norm_dat
## constants
t_eddy           = 5 # L/(2*Mach)
var_x            = 0
label_x          = r'$t/t_{\mathregular{eddy}}$'
label_data_1     = r'$k_{B} = 10$'
label_data_2     = r'$k_{B} = 100$'
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
elif var_y == 8:
    ## mach number
    label_y       = r'$\mathcal{M}$'
    bool_norm_dat = bool(0)
    var_scale     = 'linear'
else:
    ## magnetic field
    label_y       = r'$E_{B}/E_{B 0}$'
    bool_norm_dat = bool(1)
    var_scale     = 'log'

##################################################################
## INITIALISING VARIABLES
##################################################################
filepath_data_1 = createFilePath([filepath_base, folder_data_1])
filepath_data_2 = createFilePath([filepath_base, folder_data_2])
filepath_plot   = createFilePath([filepath_base, folder_plot])
## create folder where the figure will be saved
createFolder(filepath_plot)
## open figure
fig = plt.figure(figsize=(10, 7), dpi=100)

##################################################################
## LOADING DATA
##################################################################
data_x_1, data_y_1, var_name = loadTurbDat(filepath_data_1)
data_x_2, data_y_2, var_name = loadTurbDat(filepath_data_2)

##################################################################
## PLOTTING DATA
##################################################################
print('Plotting data...')
plt.plot(data_x_1, data_y_1, 'k', label=label_data_1)
plt.plot(data_x_2, data_y_2, 'b', label=label_data_2)

##################################################################
## LABEL and ADJUST PLOT
##################################################################
print('Labelling plot...')
# add legend
plt.legend(loc='lower right', fontsize=17, frameon=False)
## major grid
plt.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.35)
## minor grid
plt.grid(which='minor', linestyle='--', linewidth='0.5', color='black', alpha=0.2)
## label plot
plt.xlabel(label_x, fontsize=20)
plt.ylabel(label_y, fontsize=20)
## scale y-axis
plt.yscale(var_scale)

##################################################################
## SAVE IMAGE
##################################################################
print('Saving the figure...')
name_fig = createFilePath([filepath_plot, (pre_name + '_turb_' + var_name + '_combined.png')])
plt.savefig(name_fig)
print('Figure saved: ' + name_fig)

## END OF PROGRAM