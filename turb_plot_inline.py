#!/usr/bin/env python3

''' AUTHOR: Neco Kriel
    
    EXAMPLE: 
    turb_plot_inline.py -base_path /Users/dukekriel/Documents/University/Year4Sem2/Summer-19/ANU-Turbulence-Dynamo/dyna288_Bk10 -vis_folder visFiles -fig_name dyna288_Bk10
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
ap.add_argument('-xmin', required=False, help='Minimum x value which analysis is performed', type=float, default=3.2)
ap.add_argument('-xmax', required=False, help='Maximum x value which analysis is performed', type=float, default=6)
## ------------------- REQUIRED ARGUMENTS
ap.add_argument('-base_path', required=True, help='Filepath to the base folder', type=str)
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
folder_plot   = args['vis_folder']  # subfolder where animation and plots will be saved
fig_name      = args['fig_name']    # fig_name of figures
## remove the trailing '/' from the input filepath
if filepath_base.endswith('/'):
    filepath_base = filepath_base[:-1]
## ------------------- ANALYSIS DOMAIN
x_min = args['xmin']
x_max = args['xmax']
## start code
print('Began running the spectra plotting code in the filepath: \n\t' + filepath_base)
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
        if len(row)  == len_thresh:
            if ((row[var_x][0]  == '#') or (row[var_y][0]  == '#')):
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
global bool_norm_dat
global t_eddy, var_x, var_y
## constants
t_eddy           = 5 # L/(2*Mach)
var_x            = 0
label_x          = r'$t/t_{\mathregular{eddy}}$'
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
bool_ave         = bool(0) # plot average of data over specified x-range
bool_regression  = bool(0) # plot regression line for data over specified x-range
if var_y  == 6:
    ## kinetic field
    label_y       = r'$E_{\nu}/E_{\nu 0}$'
    bool_norm_dat = bool(1)
    var_scale     = 'log'
elif var_y  == 8:
    ## mach number
    label_y       = r'$\mathcal{M}$'
    bool_norm_dat = bool(0)
    bool_ave      = bool(1)
    var_scale     = 'linear'
else:
    ## magnetic field
    label_y       = r'$E_{B}/E_{B 0}$'
    bool_norm_dat = bool(1)
    bool_regression = bool(1)
    var_scale     = 'log'

##################################################################
## INITIALISING VARIABLES
##################################################################
filepath_data = filepath_base + '/Turb.dat'
filepath_plot = createFilePath([filepath_base, folder_plot])
## create folder where the figure will be saved
createFolder(filepath_plot)
## open figure
fig = plt.figure(figsize = (10, 7), dpi = 100)
ax  = fig.add_subplot()

##################################################################
## LOADING DATA
##################################################################
print('Loading data...')
data_x, data_y, var_name = loadData(filepath_data)
## save analysis data
index_min = min(enumerate(data_x), key = lambda x: abs(x_min - x[1]))[0]
index_max = min(enumerate(data_x), key = lambda x: abs(x_max - x[1]))[0]
fit_x     = list(map(float, data_x[index_min:index_max]))
fit_y     = list(map(float, data_y[index_min:index_max]))

##################################################################
## PLOTTING DATA
##################################################################
print('Plotting data...')
plt.plot(data_x, data_y, 'k')

##################################################################
## ADD REGRESSION / AVERAGING
##################################################################
## plot regression analysis
if bool_regression:
    log_y = np.log(fit_y)
    m, c  = np.polyfit(fit_x, log_y, 1)    # fit log(y) = m*log(x) + c
    fit_y = np.exp([m*x + c for x in fit_x]) # calculate the fitted values of y 
    plt.plot(fit_x, fit_y, 'k--', linewidth = 1)
    ax.text(0.75, 0.23,
        r"$m = %0.1f$"%m,
        fontsize = 20, color = 'black', 
        ha = "left", va = 'top', transform = ax.transAxes)
    ax.text(0.75, 0.15,
        r"$c = %0.1f$"%c,
        fontsize = 20, color = 'black', 
        ha = "left", va = 'top', transform = ax.transAxes)
## plot average analysis
if bool_ave:
    var_dt    = np.diff(fit_x)
    var_ave_y = [(prev+cur)/2 for prev, cur in zip(fit_y[:-1], fit_y[1:])]
    ave_y     = sum(var_ave_y * var_dt) / (x_max - x_min)
    fit_y     = np.repeat(ave_y, len(fit_y))
    ax.text(0.5, 0.25,
        r"$\langle %s \rangle \pm 1\sigma = $"%label_y.replace('$', '') +
        r"$%0.2f$"%ave_y + 
        r" $\pm$ " +
        r"$%0.1g$"%np.sqrt(np.var(data_y)),
        fontsize = 20, color = 'black', 
        ha = "left", va = 'top', transform = ax.transAxes)

##################################################################
## LABEL and ADJUST PLOT
##################################################################
print('Labelling plot...')
## major grid
ax.grid(which = 'major', linestyle = '-', linewidth = '0.5', color = 'black', alpha = 0.35)
## minor grid
ax.grid(which = 'minor', linestyle = '--', linewidth = '0.5', color = 'black', alpha = 0.2)
## label plot
plt.xlabel(label_x, fontsize = 20)
plt.ylabel(label_y, fontsize = 20)
## scale y-axis
ax.set_yscale(var_scale)

##################################################################
## SAVE IMAGE
##################################################################
print('Saving the figure...')
name_fig = filepath_plot + 'turb_' + fig_name + '_' + var_name + '.png'
plt.savefig(name_fig)
print('Figure saved: ' + name_fig)

# ## END OF PROGRAM