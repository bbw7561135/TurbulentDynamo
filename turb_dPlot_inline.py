#!/usr/bin/env python3

''' AUTHOR: Neco Kriel
    
    EXAMPLE: 
    turb_dPlot_inline.py -base_path /Users/dukekriel/Documents/University/Year4Sem2/Summer-19/ANU-Turbulence-Dynamo -dat_folder1 dyna288_Bk10 -dat_folder2 dyna288_Bk100 -vis_folder testPlots -fig_name dyna288
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
os.system('clear')                  # clear terminal window
plt.close('all')                    # close all pre-existing plots
mpl.style.use('classic')            # plot in classic style

##################################################################
## INPUT COMMAND LINE ARGUMENTS
##################################################################
global bool_debug_mode, filepath_base
ap = argparse.ArgumentParser(description='A bunch of input arguments')
## ------------------- OPTIONAL ARGUMENTS
ap.add_argument('-debug', required=False, help='Debug mode', type=bool, default=False)
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
## ------------------- FILEPATH PARAMETERS
filepath_base = args['base_path']   # home directory
folder_data_1 = args['dat_folder1'] # first subfolder's fig_name
folder_data_2 = args['dat_folder2'] # second subfolder's fig_name
folder_plot   = args['vis_folder']  # subfolder where animation and plots will be saved
fig_name      = args['fig_name']    # fig_name of figures
## remove the trailing '/' from the input filepath/folders
if filepath_base.endswith('/'):
    filepath_base = filepath_base[:-1]
if folder_data_1.endswith('/'):
    folder_data_1 = folder_data_1[:-1]
if folder_data_2.endswith('/'):
    folder_data_2 = folder_data_2[:-1]
if folder_plot.endswith('/'):
    folder_plot = folder_plot[:-1]
## start code
print('Began running the spectra plotting code in the filepath: \n\t' + filepath_base)
print('Data folder 1: ' + folder_data_1)
print('Data folder 2: ' + folder_data_2)
print('Visualising folder: ' + folder_plot)
print(' ')

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

def createFilePath(names):
    return ('/'.join(names) + '/')

def loadData(directory):
    global t_eddy, bool_norm_dat, var_x, var_y
    ## load data
    data_split = [x.split() for x in open(directory).readlines()]
    ## save maximum number of columns in a row. less indicated rows stores a message.
    len_thresh = len(data_split[0]) # ignore extra lines (len < len_thresh) resulting from restarting the simulation
    ## save data
    data_x = []
    data_y = []
    for row in data_split[1:]:
        if len(row) == len_thresh:
            if ((row[var_x][0] == '#') or (row[var_y][0] == '#')):
                break
            data_x.append(float(row[var_x]) / t_eddy) # normalise time-domain
            data_y.append(float(row[var_y]))
    if bool_norm_dat:
        # y_data = var_y / var_y[1]
        data_y = [i / data_y[1] for i in data_y]
    ## return variables
    return [data_x, data_y, data_split[0][var_y][4:]]

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
filepath_data_1 = createFilePath([filepath_base, folder_data_1]) + 'Turb.dat'
filepath_data_2 = createFilePath([filepath_base, folder_data_2]) + 'Turb.dat'
filepath_plot   = createFilePath([filepath_base, folder_plot])
## create folder where the figure will be saved
createFolder(filepath_plot)
## open figure
fig = plt.figure(figsize=(10, 7), dpi=100)

##################################################################
## LOADING DATA
##################################################################
print('Loading data...')
data_x_1, data_y_1, var_name = loadData(filepath_data_1)
data_x_2, data_y_2, var_name = loadData(filepath_data_2)

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
name_fig = filepath_plot + 'turb_dyna288_' + var_name + '.png'
plt.savefig(name_fig)
print('Figure saved: ' + name_fig)

# ## END OF PROGRAM