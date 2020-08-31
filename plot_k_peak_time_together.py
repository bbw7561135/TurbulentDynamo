#!/usr/bin/env python3

''' AUTHOR: Neco Kriel
    
    EXAMPLE: 
    plot_scale_dependance.py
        (required)
            -base_path      $scratch
            -dat_folder1    dyna288_Bk10/Re10/hdf5Files
            -dat_folder2    dyna288_Bk10/Pm1/hdf5Files
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
plt.rcParams.update({'font.size': 18})

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

def saveData(filepath_data):
    global t_eddy
    #################### SAVE DATA
    ##############################
    num_time_points = int(len(list(filter(meetsCondition, sorted(os.listdir(filepath_data)))))/2)
    for cur_iter in range(num_time_points):
        cur_time = cur_iter/t_eddy # normalise time point by eddy-turnover time
        if ((100 * cur_iter/num_time_points) % 20 < 0.099):
            print('\t Loading data... %0.3f%% complete'%(100 * cur_iter/num_time_points))
        #################### LOAD DATA
        ##############################
        name_file_mag = 'Turb_hdf5_plt_cnt_' + '{0:04}'.format(cur_iter) + '_spect_mags.dat' # magnetic file
        data_x, data_y = loadData(filepath_data + '/' + name_file_mag) # magnetic power spectrum
        if (cur_iter == 0):
            #################### INITIALISE DATA
            ##############################
            ## initialise the list of time points
            time_points = []
            ## initialise the list of peaks values and their standard deviation
            peak_k = []
            peak_k_std = []
            ## initialise the results matrix
            spectra_matrix = np.zeros([num_time_points, len(data_x)])
        #################### SAVE DATA
        ##############################
        ## append the current time point
        time_points.append(cur_time)
        ## calculate the standard deviation of the spectra
        if (cur_iter == 0): peak_k_std.append(0)
        else: peak_k_std.append(stdev(np.log10(data_y)))
        ## calculate the peak of the spectra
        peak_k.append(data_x[data_y.index(max(data_y))])
        ## save the magnetic power spectrum
        for k in range(len(data_x)):
            spectra_matrix[cur_iter][k] = data_y[k]
    #################### CALCULATE MOVING AVERAGE
    ##############################
    window_size = 10 # moving average window size
    peak_k_moving_ave = []
    for tmp_index in range(len(peak_k)):
        if tmp_index < window_size:
            peak_k_moving_ave.append(peak_k[tmp_index])
        else:
            ## calculate moving average of the magnetic spectra peak
            peak_k_moving_ave.append(sum(peak_k[tmp_index : tmp_index + window_size]) / window_size)
    #################### CALCULATE STANDARD DEVIATION BAND
    ##############################
    peak_k_std_low = []
    peak_k_std_high = []
    for val, std_val in zip(peak_k_moving_ave, peak_k_std):
        peak_k_std_low.append(val - std_val)
        peak_k_std_high.append(val + std_val)
    ## return data
    print(' ')
    return time_points, data_x, spectra_matrix, peak_k_moving_ave, peak_k_std_low, peak_k_std_high

def plotScaleLabels(ax, peak_k_moving_ave, time_points, k_0, Re, Pm):
    ## add markers
    ax.text(max(peak_k_moving_ave[1:100])*1.05, time_points[-1]*0.05, r'$k_{max}$', ha='left', va='bottom', fontsize=28)
    ## label scales
    k_nu = Re**(3/4) * k_0
    k_eta = Re**(1/4) * (Re*Pm)**(1/2) * k_0
    ax.plot([k_0, k_0], [1, time_points[-1]], 'k--')
    ax.plot([k_nu, k_nu], [1, time_points[-1]], 'k--')
    ax.plot([k_eta, k_eta], [1, time_points[-1]], 'k--')
    ax.text(k_0*0.95, time_points[-1]*0.8, r'$k_0$', ha='right', va='bottom', fontsize=28)
    ax.text(k_nu*1.05, time_points[-1]*0.8, r'$k_\nu$', ha='left', va='bottom', fontsize=28)
    ax.text(k_eta*0.95, time_points[-1]*0.8, r'$k_\eta$', ha='right', va='bottom', fontsize=28)
    ax.text(xlim_max*0.9, time_points[-1]*0.05, r'$Pm{}$'.format(Pm), ha='right', va='bottom', fontsize=28)

def plotData(data_x, time_points, spectra_matrix, peak_k_moving_ave, peak_k_std_low, peak_k_std_high,
                bool_x_label, bool_y_label,
                ax, k_0, Re, Pm):
    global xlim_min, xlim_max
    ## plot heatmap
    mag_plot = ax.pcolormesh(data_x, time_points, spectra_matrix, cmap='plasma', norm=LogNorm())
    # plot peak of magnetic spectra
    plotScaleLabels(ax, peak_k_moving_ave, time_points, k_0, Re, Pm)
    ax.fill_betweenx(time_points, peak_k_std_low, peak_k_std_high, facecolor='black', alpha=0.2)
    ax.plot(peak_k_moving_ave, time_points, 'k-', linewidth='1.5')
    ## scale axis
    ax.minorticks_on()
    ax.set_xscale('log')
    ## major grid
    ax.grid(b=True, which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.35)
    ## minor grid
    ax.xaxis.grid(b=True, which='minor', linestyle='--', dashes=(5, 2.5), linewidth='0.5', color='black', alpha=0.2)
    ## increase space between axis and ticks
    ax.tick_params(axis='x', which='major', pad=7)
    ## set axis limits
    ax.set_xlim(xlim_min, xlim_max)
    ax.set_ylim(1, time_points[-1])
    ## label axis
    if (bool_x_label): ax.set_xlabel(r'$k$', fontsize=28, rotation=0)
    else: ax.set_xticklabels([])
    if (bool_y_label): ax.set_ylabel(r'$t/T$', fontsize=28, rotation=0, labelpad=30)
    else: ax.set_yticklabels([])
    ## return colour map
    return mag_plot

##################################################################
## INPUT COMMAND LINE ARGUMENTS
##################################################################
global file_end, bool_debug_mode, filepath_base, file_start
ap = argparse.ArgumentParser(description='A bunch of input arguments')
## ------------------- DEFINE OPTIONAL ARGUMENTS
ap.add_argument('-debug',      type=str2bool,   default=False, required=False, help='Debug mode', nargs='?', const=True)
ap.add_argument('-sub_folder', type=str, default='spectFiles', required=False, help='Name of the folder where the data is stored')
ap.add_argument('-vis_folder', type=str, default='visFiles',   required=False, help='Name of the plot folder')
ap.add_argument('-file_start', type=int, default=0,            required=False, help='First file to process')
ap.add_argument('-file_end',   type=int, default=np.Inf,       required=False, help='Last file to process')
## ------------------- DEFINE REQUIRED ARGUMENTS
ap.add_argument('-base_path',   type=str, required=True, help='Filepath to the base folder')
ap.add_argument('-dat_folder1', type=str, required=True, help='File path to folder with data set 1')
ap.add_argument('-dat_folder2', type=str, required=True, help='File path to folder with data set 2')
ap.add_argument('-dat_folder3', type=str, required=True, help='File path to folder with data set 3')
ap.add_argument('-dat_folder4', type=str, required=True, help='File path to folder with data set 4')
ap.add_argument('-pre_name',    type=str, required=True, help='Name of figures')
## ---------------------------- OPEN ARGUMENTS
args = vars(ap.parse_args())
## ---------------------------- SAVE PARAMETERS
bool_debug_mode = args['debug']       # enable/disable debug mode
file_start      = args['file_start']  # starting processing frame
file_end        = args['file_end']    # the last file to process
## ---------------------------- SAVE FILEPATH PARAMETERS
filepath_base = args['base_path']   # home directory
folder_data_1 = args['dat_folder1']
folder_data_2 = args['dat_folder2']
folder_data_3 = args['dat_folder3']
folder_data_4 = args['dat_folder4']
folder_sub    = args['sub_folder']  # sub-subfolder where data is stored's name
folder_vis    = args['vis_folder']  # subfolder where animation and plots will be saved
pre_name      = args['pre_name']    # name of figures
## ---------------------------- ADJUST ARGUMENTS
## remove the trailing '/' from the input filepath and plot folder
if filepath_base.endswith('/'):
    filepath_base = filepath_base[:-1]
## replace any '//' with '/'
filepath_base = filepath_base.replace('//', '/')
## remove '/' from variable names
folder_data_1 = stringChop(folder_data_1, '/')
folder_data_2 = stringChop(folder_data_2, '/')
folder_data_3 = stringChop(folder_data_3, '/')
folder_data_4 = stringChop(folder_data_4, '/')
folder_vis    = stringChop(folder_vis, '/')
folder_sub    = stringChop(folder_sub, '/')
pre_name      = stringChop(pre_name, '/')
## ---------------------------- START CODE
filepath_data_1 = createFilePath([filepath_base, folder_data_1, folder_sub]) # first folder with data
filepath_data_2 = createFilePath([filepath_base, folder_data_2, folder_sub]) # second folder with data
filepath_data_3 = createFilePath([filepath_base, folder_data_3, folder_sub]) # first folder with data
filepath_data_4 = createFilePath([filepath_base, folder_data_4, folder_sub]) # second folder with data
filepath_plot   = createFilePath([filepath_base, folder_vis, 'plotSpectra']) # folder where plots will be saved
createFolder(filepath_plot) # create folder where plots are saved
print('Began running the spectra plotting code in base filepath: \n\t' + filepath_base)
print('Data folder 1: '                                                + filepath_data_1)
print('Data folder 2: '                                                + filepath_data_2)
print('Data folder 3: '                                                + filepath_data_3)
print('Data folder 4: '                                                + filepath_data_4)
print('Visualising folder: '                                           + folder_vis)
print('Figure name: '                                                  + pre_name)
print(' ')

##################################################################
## USER VARIABLES
##################################################################
global t_eddy, xlim_min, xlim_max
t_eddy = 10 # number of spectra files per eddy turnover # TODO: input?
## set the figure's axis limits
xlim_min = 1.0
xlim_max = 1.3e+02

##################################################################
## LOAD DATA
##################################################################
print('Loading Data Set 1...')
time_points_1, data_x_1, spectra_matrix_1, peak_k_moving_ave_1, peak_k_std_low_1, peak_k_std_high_1 = saveData(filepath_data_1)
print('Loading Data Set 2...')
time_points_2, data_x_2, spectra_matrix_2, peak_k_moving_ave_2, peak_k_std_low_2, peak_k_std_high_2 = saveData(filepath_data_2)
print('Loading Data Set 3...')
time_points_3, data_x_3, spectra_matrix_3, peak_k_moving_ave_3, peak_k_std_low_3, peak_k_std_high_3 = saveData(filepath_data_3)
print('Loading Data Set 4...')
time_points_4, data_x_4, spectra_matrix_4, peak_k_moving_ave_4, peak_k_std_low_4, peak_k_std_high_4 = saveData(filepath_data_4)

##################################################################
## PLOT DATA
##################################################################
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(15, 13), dpi=300)
#################### DATA SET 1
##############################
plotData(data_x_1, time_points_1, spectra_matrix_1, peak_k_moving_ave_1, peak_k_std_low_1, peak_k_std_high_1,
            0, 1, ax1, k_0=2, Re=24, Pm=25)
#################### DATA SET 2
##############################
mag_plot = plotData(data_x_2, time_points_2, spectra_matrix_2, peak_k_moving_ave_2, peak_k_std_low_2, peak_k_std_high_2,
            0, 0, ax2, k_0=2, Re=15, Pm=50)
#################### DATA SET 3
##############################
mag_plot = plotData(data_x_3, time_points_3, spectra_matrix_3, peak_k_moving_ave_3, peak_k_std_low_3, peak_k_std_high_3,
            1, 1, ax3, k_0=2, Re=8, Pm=125)
#################### DATA SET 4
##############################
mag_plot = plotData(data_x_4, time_points_4, spectra_matrix_4, peak_k_moving_ave_4, peak_k_std_low_4, peak_k_std_high_4,
            1, 0, ax4, k_0=2, Re=3, Pm=500)
## add colour bar
cbar_comb = fig.colorbar(mag_plot, ax=[ax1, ax2, ax3, ax4], cax=fig.add_axes([0.91, 0.1, 0.015, 0.8]))
cbar_comb.ax.set_ylabel(r'$\mathcal{P}_{B}(k)$', fontsize=28, rotation=0, va="bottom", labelpad=45)
# remove spacing between subplots
plt.subplots_adjust(wspace=0.05, hspace=0.05)

##################################################################
## SAVE FIGURE
##################################################################
print('Saving figure...')
fig_name = createFilePath([filepath_plot, pre_name]) + '_k_peak_time_together.png'
plt.savefig(fig_name, bbox_inches='tight', dpi=300)
plt.close()
print('Figure saved: ' + fig_name)
print(' ')

## END OF PROGRAM
