#!/usr/bin/env python3

''' AUTHOR: Neco Kriel
    
    EXAMPLE: 
    plot_spectra_together.py 
        (required)
            -base_path      $scratch
            -dat_folder1    dyna288_Bk10/Re10
            -dat_folder2    dyna288_Bk100/Re10
            -pre_name       dyna288_Bk
        (optional)
            -debug          False
            -vis_folder     visFiles
            -sub_folder     spectFiles
            -xlim_min       1.0
            -xlim_max       1.3e+02
            -ylim_min       1.0e-21
            -ylim_max       4.2e-03
'''

##################################################################
## MODULES
##################################################################
import os
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib import patches
from statistics import stdev, mean
from matplotlib.cm import get_cmap

##################################################################
## PREPARE TERMINAL/WORKSPACE/CODE
#################################################################
os.system('clear')                  # clear terminal window
plt.close('all')                    # close all pre-existing plots
mpl.style.use('classic')            # plot in classic style
plt.rcParams.update({'font.size': 18})
plt.rc('axes', axisbelow=True)

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

def plotScales(k0, Re, Rm):
    ## add viscous and magnetic dissipation scale
    k_nu  = k0 * Re**(3/4)
    # k_eta = k0 * Rm**(1/2) * Re**(1/4)
    plt.plot([k_nu, k_nu], [(ylim_min+10**(-24)+10**(-16)), ylim_max], 'k--', alpha=0.5, linewidth=1.5)
    plt.annotate(r'$k_\nu$', xy=(k_nu*1.05, ylim_max-2*10**(-2)), fontsize=24, color='black', ha='left', va='top')

def calcSatVel(sim_time, filepath_data, line_cols):
    data_y_sat = list() # initialise y-data list
    time_vals = range(sim_time[-1], (sim_time[-1]+9)) # create time values in saturated regime
    ## loop over regime and load y-data
    for index in range(len(time_vals)):
        name_file_vel = 'Turb_hdf5_plt_cnt_' + '{0:04}'.format(time_vals[index]) + '_spect_vels.dat' # velocity file
        data_x, data_y = loadData(createFilePath([filepath_data, name_file_vel])) # velocity power spectrum data
        data_y_sat.append(data_y)
    ## initialise mean and std.dev. lists
    data_mean = []
    data_std  = []
    ## calculate mean and std.dev for each k over saturated regime
    for i in range(len(data_y_sat[0])):
        temp_data = [data[i] for data in data_y_sat]
        data_mean.append(mean(temp_data))
        data_std.append(stdev(temp_data))
    data_mean = np.array(data_mean)
    data_std = np.array(data_std)
    plt.plot(data_x, data_mean, '-', color=line_cols[-1], linewidth=2)
    # plt.fill_between(data_x, (data_mean - data_std), (data_mean + data_std), facecolor=line_cols[-1], edgecolor=line_cols[-1], linewidth=2, alpha=0.5)

def calcSatMag(sim_time, filepath_data, line_cols):
    data_y_sat = list() # initialise y-data list
    time_vals = range(sim_time[-1], (sim_time[-1]+9)) # create time values in saturated regime
    ## loop over regime and load y-data
    for index in range(len(time_vals)):
        name_file_mag = 'Turb_hdf5_plt_cnt_' + '{0:04}'.format(time_vals[index]) + '_spect_mags.dat' # magnetic file
        data_x, data_y = loadData(createFilePath([filepath_data, name_file_mag])) # magnetic power spectrum data
        data_y_sat.append(data_y)
    ## initialise mean and std.dev. lists
    data_mean = []
    data_std  = []
    ## calculate mean and std.dev for each k over saturated regime
    for i in range(len(data_y_sat[0])):
        temp_data = [data[i] for data in data_y_sat]
        data_mean.append(mean(temp_data))
        data_std.append(stdev(temp_data))
    data_mean = np.array(data_mean)
    data_std = np.array(data_std)
    plt.plot(data_x, data_mean, '--', dashes=(20, 5), color=line_cols[-1], linewidth=2)
    # plt.fill_between(data_x, (data_mean - data_std), (data_mean + data_std), facecolor=line_cols[-1], edgecolor=line_cols[-1], linewidth=2, alpha=0.5)

##################################################################
## INPUT COMMAND LINE ARGUMENTS
##################################################################
global bool_debug_mode
global filepath_base, sim_time_1, sim_time_2
ap = argparse.ArgumentParser(description='A bunch of input arguments')
## ------------------- DEFINE OPTIONAL ARGUMENTS
ap.add_argument('-debug',    type=str2bool, default=False, required=False, help='Debug mode',                 nargs='?', const=True)
ap.add_argument('-sub_folder', type=str,   default='spectFiles', required=False, help='Name of the data folder')
ap.add_argument('-vis_folder', type=str,   default='visFiles',   required=False, help='Name of the plot folder')
ap.add_argument('-xlim_min',   type=float, default=1.0,          required=False, help='Figure xlim minimum')
ap.add_argument('-xlim_max',   type=float, default=1.3e+02,      required=False, help='Figure xlim maximum')
ap.add_argument('-ylim_min',   type=float, default=1.0e-25,      required=False, help='Figure ylim minimum')
ap.add_argument('-ylim_max',   type=float, default=1.0e-01,      required=False, help='Figure ylim maximum')
## ------------------- DEFINE REQUIRED ARGUMENTS
ap.add_argument('-base_path',   type=str, required=True, help='Filepath to the base folder')
ap.add_argument('-dat_folder1', type=str, required=True, help='Name of the first folder')
ap.add_argument('-dat_folder2', type=str, required=True, help='Name of the second folder')
# ap.add_argument('-file_start1', type=int, required=True, help='File number to plot from folder 1', nargs='+')
# ap.add_argument('-file_start2', type=int, required=True, help='File number to plot from folder 2', nargs='+')
ap.add_argument('-pre_name',    type=str, required=True, help='Name of figures')
## ---------------------------- OPEN ARGUMENTS
args = vars(ap.parse_args())
## ---------------------------- SAVE PARAMETERS
bool_debug_mode = args['debug']       # enable/disable debug mode
## set the figure's axis limits
xlim_min        = args['xlim_min']
xlim_max        = args['xlim_max']
ylim_min        = args['ylim_min']
ylim_max        = args['ylim_max']
## ---------------------------- SAVE FILEPATH PARAMETERS
filepath_base   = args['base_path']   # home directory
folder_data_1   = args['dat_folder1'] # first subfolder's name
folder_data_2   = args['dat_folder2'] # second subfolder's name
folder_sub      = args['sub_folder']  # sub-subfolder where data is stored's name
folder_vis      = args['vis_folder']  # subfolder where animation and plots will be saved
pre_name        = args['pre_name']    # name of figures
# sim_time_1    = args['file_start1'] # starting processing frame
# sim_time_2    = args['file_start2'] # starting processing frame
## ---------------------------- ADJUST ARGUMENTS
## remove the trailing '/' from the input filepath and folders
if filepath_base.endswith('/'):
    filepath_base = filepath_base[:-1]
## replace any '//' with '/'
filepath_base = filepath_base.replace('//', '/')
## remove '/' from variable names
folder_data_1 = stringChop(folder_data_1, '/')
folder_data_2 = stringChop(folder_data_2, '/')
folder_sub    = stringChop(folder_sub, '/')
folder_vis    = stringChop(folder_vis, '/')
pre_name      = stringChop(pre_name, '/')

##################################################################
## USER VARIABLES
##################################################################
t_eddy = 10 # number of spectra files per eddy turnover # TODO: input?
## specify which variables you want to plot
global var_x, var_y
var_x = 1  # variable: wave number (k)
var_y = 15 # variable: power spectrum

## open figure
fig = plt.figure(figsize=(10, 7))
## set parameters for plot
if ('Pm'.lower() in pre_name.lower()):
    ## compare Pm100 and Pm1 data
    sim_label_1 = r'Pm100'
    sim_label_2 = r'Pm1'
    sim_time_1   = [70, 146, 227, 310, 500]
    sim_time_2   = [36, 137, 243, 335, 600]
    line_cols_1  = ['#FCCF86', '#FAAC58', '#FF8000', '#FE2E2E', '#FF0000']
    line_cols_2  = ['#C7C7C7', '#9B9B9B', '#848484', '#565656', '#000000']
    # plotScales(2, 10**1, 10**3) # Re10, Rm1000, Pm100
    # plotScales(2, 10**3, 10**3) # Re1000, Rm1000, Pm1
elif ('Bk'.lower() in pre_name.lower()):
    ## compare Bk10 and Bk100 data
    sim_label_1 = r'Pm100Bk10'
    sim_label_2 = r'Pm100Bk100'
    sim_time_1   = [70, 146, 227, 310, 500]
    sim_time_2   = [410, 486, 580, 695, 900]
    line_cols_1  = ['#FCCF86', '#FAAC58', '#FF8000', '#FE2E2E', '#FF0000']
    line_cols_2  = ['#A9E2F3', '#81BEF7', '#2E9AFE', '#5858FA', '#0000FF']
    plotScales(2, 10**1, 10**3) # Re10, Rm1000, Pm100
else:
    raise Exception('pre_name is: {}'.format(pre_name))
## print information to screen
print('Began running the spectra plotting code in base filepath: \n\t' + filepath_base)
print('Data folder 1: '                                                + folder_data_1)
print('Data folder 2: '                                                + folder_data_2)
print('Time points to check:')
print('\tIn folder 1: {}'.format(sim_time_1))
print('\tIn folder 2: {}'.format(sim_time_2))
print('Visualising folder: '                                           + folder_vis)
print('Figure name: '                                                  + pre_name)
print(' ')

if (len(sim_time_1) != len(sim_time_2)):
    raise Exception('The number of time-points you watch to check do not match for the two simulations. Number of time points were {0} and {1}'.format(len(sim_time_1), len(sim_time_2)))

##################################################################
## INITIALISING VARIABLES
##################################################################
filepath_data_1 = createFilePath([filepath_base, folder_data_1, folder_sub]) # first folder with data
filepath_data_2 = createFilePath([filepath_base, folder_data_2, folder_sub]) # second folder with data
filepath_plot   = createFilePath([filepath_base, folder_vis, 'compSpectra']) # folder where plots will be saved
createFolder(filepath_plot) # create folder where plots are saved

## add Kazantsev spectrum
fit_x = np.linspace(1, 3, 2, endpoint=True)
fit_y = [10**( (3/2)*math.log10(x) + math.log10(10**(-14)) ) for x in fit_x]
plt.plot(fit_x, fit_y, 'k--', alpha=0.5, linewidth=1.5)
plt.annotate(r'$k^{3/2}$', xy=((fit_x[0]+0.1), (fit_y[0]+5*10**(-15))),
            fontsize=24, color='black', ha='left', va='bottom')

##################################################################
## PLOT KINEMATIC REGIME DATA
##################################################################
print('Loading and plotting velocity data...')
num_kinematic = len(sim_time_1)-1
for index in range(num_kinematic):
    #################### LOAD DATA
    ##############################
    print('\tProgress: ' + str(100*(index+1)/num_kinematic) + '%')
    ## load dataset 1
    name_file_kin = 'Turb_hdf5_plt_cnt_' + '{0:04}'.format(sim_time_1[index]) + '_spect_vels.dat' # kinetic file
    data_x_1_kin, data_y_1_kin = loadData(createFilePath([filepath_data_1, name_file_kin])) # kinetic power spectrum data
    ## load dataset 2
    name_file_kin = 'Turb_hdf5_plt_cnt_' + '{0:04}'.format(sim_time_2[index]) + '_spect_vels.dat' # kinetic file
    data_x_2_kin, data_y_2_kin = loadData(createFilePath([filepath_data_2, name_file_kin])) # kinetic power spectrum data
    #################### PLOT DATA
    ##############################
    ## plot dataset 1 and 2
    plt.plot(data_x_1_kin, data_y_1_kin, '-', color=line_cols_1[index], linewidth=1.5) # kinetic power spectrum
    plt.plot(data_x_2_kin, data_y_2_kin, '-', color=line_cols_2[index], linewidth=1.5) # kinetic power spectrum
print('Loading and plotting magnetic data...')
for index in range(num_kinematic):
    #################### LOAD DATA
    ##############################
    print('\tProgress: ' + str(100*(index+1)/num_kinematic) + '%')
    ## load dataset 1
    name_file_mag = 'Turb_hdf5_plt_cnt_' + '{0:04}'.format(sim_time_1[index]) + '_spect_mags.dat' # magnetic file
    data_x_1_mag, data_y_1_mag = loadData(createFilePath([filepath_data_1, name_file_mag])) # magnetic power spectrum data
    ## load dataset 2
    name_file_mag = 'Turb_hdf5_plt_cnt_' + '{0:04}'.format(sim_time_2[index]) + '_spect_mags.dat' # magnetic file
    data_x_2_mag, data_y_2_mag = loadData(createFilePath([filepath_data_2, name_file_mag])) # magnetic power spectrum data
    #################### PLOT DATA
    ##############################
    ## plot dataset 1 and 2
    plt.plot(data_x_1_mag, data_y_1_mag, '--', color=line_cols_1[index], dashes=(20, 5), linewidth=1.5) # magnetic power spectrum
    plt.plot(data_x_2_mag, data_y_2_mag, '--', color=line_cols_2[index], dashes=(20, 5), linewidth=1.5) # magnetic power spectrum

##################################################################
## PLOT SATURATED REGIME DATA
##################################################################
## calculate and plot magnetic spectra in saturated regime
## data set 1
calcSatVel(sim_time_1, filepath_data_1, line_cols_1)
calcSatMag(sim_time_1, filepath_data_1, line_cols_1)
## data set 2
calcSatVel(sim_time_2, filepath_data_2, line_cols_2)
calcSatMag(sim_time_2, filepath_data_2, line_cols_2)

##################################################################
## LABEL and ADJUST PLOT
##################################################################
print('Labelling plot...')
## scale axies
plt.xscale('log')
plt.yscale('log')
## set axis limits
plt.xlim(xlim_min, xlim_max)
plt.ylim(ylim_min, ylim_max)
# label plots
plt.xlabel(r'$k$',           fontsize=28)
plt.ylabel(r'$\mathcal{P}$', fontsize=28)
## major grid
plt.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.35)
## minor grid
plt.grid(which='minor', linestyle='--', dashes=(20, 5), linewidth='0.5', color='black', alpha=0.2)
## add legend
label_legend = [r'$10^{-1}$', r'$10^{1}$', r'$10^{3}$', r'$10^{5}$', (r'$sat.$')]
rectangle = plt.Rectangle((xlim_min+0.05, ylim_min+10**(-25)), 48, 10**(-16), edgecolor='k', facecolor='white')
plt.gca().add_patch(rectangle)
for legend_row in range(len(label_legend)):
    plot_height = (ylim_min + 10**(-24 + 1.2*legend_row))
    if (legend_row < ( len(label_legend)-1 )):
        plot_linewidth = 1
    else:
        plot_linewidth = 2
    plt.text(xlim_min+0.75, (ylim_min + 3*10**(-25 + 1.25*legend_row)), label_legend[legend_row], ha='right', fontsize=15)
    plt.plot([10**0.28, 10**0.58], [plot_height, plot_height], '-',  color=line_cols_1[legend_row], linewidth=plot_linewidth)
    plt.plot([10**0.63, 10**0.94], [plot_height, plot_height], '--', dashes=(20, 5), color=line_cols_1[legend_row], linewidth=plot_linewidth)
    plt.plot([10**1.005, 10**1.305], [plot_height, plot_height], '-',  color=line_cols_2[legend_row], linewidth=plot_linewidth)
    plt.plot([10**1.355, 10**1.665], [plot_height, plot_height], '--', dashes=(20, 5), color=line_cols_2[legend_row], linewidth=plot_linewidth)
legend_label_height_1 = (ylim_min + 5*10**(-25.2 + 1.25*(legend_row + 1)))
legend_label_height_2 = (ylim_min + 5*10**(-25.2 + 1.25*(legend_row + 2)))
plt.text(xlim_min+0.75, legend_label_height_1, r'$E_B/E_{B0}$', ha='right', fontsize=15)
plt.text(10**0.28, legend_label_height_1, r'$kin. spectra$', ha='left', fontsize=15)
plt.text(10**0.63, legend_label_height_1, r'$mag. spectra$', ha='left', fontsize=15)
plt.text(10**1.005, legend_label_height_1, r'$kin. spectra$', ha='left', fontsize=15)
plt.text(10**1.355, legend_label_height_1, r'$mag. spectra$', ha='left', fontsize=15)
plt.text(4, legend_label_height_2, sim_label_1, ha='center', fontsize=15)
plt.text(21, legend_label_height_2, sim_label_2, ha='center', fontsize=15)
print(' ')

##################################################################
## SAVE IMAGE
##################################################################
print('Saving figure...')
temp_name = createFilePath([filepath_plot, (pre_name + '_spectra.png')])
plt.savefig(temp_name, bbox_inches='tight', dpi=300)
plt.close()
print('Figure saved: ' + temp_name)
print(' ')

## END OF PROGRAM