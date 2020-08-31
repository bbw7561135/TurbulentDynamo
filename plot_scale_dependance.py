#!/usr/bin/env python3

##################################################################
## MODULES
##################################################################
import os
import argparse
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
from statistics import stdev

from matplotlibrc import *
rcParams['xtick.top'] = False
rcParams['xtick.minor.pad'] = '8'
rcParams['xtick.major.pad'] = '8'
rcParams['ytick.minor.pad'] = '8'
rcParams['ytick.major.pad'] = '8'

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

def meetsMagCondition(element):
    global bool_debug_mode, file_end, file_start
    ## accept files that look like: *Turb_hdf5_plt_cnt_*mags.dat
    if (element.__contains__('Turb_hdf5_plt_cnt_') and element.endswith('mags.dat')):
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
    filedata = open(directory).readlines() # load in data
    data     = np.array([x.strip().split() for x in filedata[6:]]) # store all data. index: data[row, col]
    data_x   = list(map(float, data[:, 1]))  # variable: wave number (k)
    data_y   = list(map(float, data[:, 15])) # variable: power spectrum
    return data_x, data_y

def saveData(filepath_data):
    mag_filenames = list(filter(meetsMagCondition, sorted(os.listdir(filepath_data))))
    num_time_points = len(mag_filenames)
    print('\t There are ' + str(num_time_points) + ' files')
    k_peak = []
    ## analyse all time points
    for file_index in range(num_time_points):
        if ((100 * file_index/num_time_points) % 20 < 0.25):
            print('\t Loading data... %0.3f%% complete'%(100 * file_index/num_time_points))
        ## load data
        filename = createFilePath([filepath_data, mag_filenames[file_index]])
        ## magnetic power spectrum
        data_x, data_y = loadData(filename)
        ## calculate the peak of the spectra
        k_peak.append(data_x[data_y.index(max(data_y))])
    ## return data
    return k_peak

def calcListAve(lst): 
    return sum(lst) / len(lst)

def power_law(x, a, b):
    return a * x**b

def Pm_axis(Pm):
    return [str(int(round(x, 0))) for x in Pm]


## from: https://stackoverflow.com/questions/18704353/correcting-matplotlib-colorbar-ticks
def cmap_discretize(cmap, N):
    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki,key in enumerate(('red','green','blue')):
        cdict[key] = [ (indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki]) for i in range(N+1) ]
    # Return colormap object.
    return mcolors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)

def colorbar_index(ncolors, cmap, nlabels, label_title):
    cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors+0.5)
    colorbar = plt.colorbar(mappable)
    colorbar.set_label(label_title, rotation=0, labelpad=5, fontsize=22)
    colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
    colorbar.set_ticklabels(nlabels)
    colorbar.ax.minorticks_off()

##################################################################
## INPUT COMMAND LINE ARGUMENTS
##################################################################
global file_end, bool_debug_mode, filepath_base, file_start
ap = argparse.ArgumentParser(description='A bunch of input arguments')
## ------------------- DEFINE OPTIONAL ARGUMENTS
ap.add_argument('-debug', type=str2bool, default=False,        required=False, help='Debug mode', nargs='?', const=True)
ap.add_argument('-vis_folder', type=str, default='visFiles',   required=False, help='Name of the plot folder')
ap.add_argument('-file_start', type=int, default=150,          required=False, help='First file to process')
ap.add_argument('-file_end',   type=int, default=np.Inf,       required=False, help='Last file to process')
## ------------------- DEFINE REQUIRED ARGUMENTS
ap.add_argument('-base_path',   type=str, required=True, help='Filepath to the base folder')
ap.add_argument('-dat_folders', type=str, required=True, help='List of folders with data', nargs='+')
ap.add_argument('-pre_name',    type=str, required=True, help='Name of figures')
## ---------------------------- OPEN ARGUMENTS
args = vars(ap.parse_args())
## ---------------------------- SAVE PARAMETERS
bool_debug_mode = args['debug']       # enable/disable debug mode
file_start      = args['file_start']  # starting processing frame
file_end        = args['file_end']    # the last file to process
## ---------------------------- SAVE FILEPATH PARAMETERS
filepath_base = args['base_path']   # home directory
folders_data  = args['dat_folders'] # list of subfolders where each simulation's data is stored
folder_vis    = args['vis_folder']  # subfolder where animation and plots will be saved
pre_name      = args['pre_name']    # name of figures
## ---------------------------- ADJUST ARGUMENTS
## remove the trailing '/' from the input filepath and plot folder
if filepath_base.endswith('/'):
    filepath_base = filepath_base[:-1]
## replace any '//' with '/'
filepath_base = filepath_base.replace('//', '/')
## remove '/' from variable names
folder_vis    = stringChop(folder_vis, '/')
pre_name      = stringChop(pre_name, '/')
for i in range(len(folders_data)): 
    folders_data[i] = stringChop(folders_data[i], '/')
## ---------------------------- START CODE
## folder where plots will be saved
filepath_plot = createFilePath([filepath_base, folder_vis, 'plotSpectra']).replace('//', '/')
## create folder where plots are saved
createFolder(filepath_plot)
## create the filepaths to data
filepaths_data = []
for i in range(len(folders_data)):
    filepaths_data.append(createFilePath([filepath_base, folders_data[i]]).replace('//', '/'))
## print filepath information to the console
print('Base filepath:  \t'                  + filepath_base)
for i in range(len(filepaths_data)): 
    print('Data folder  ' + str(i) + ': \t' + filepaths_data[i])
print('Figure folder:  \t'                  + filepath_plot)
print('Figure name:  \t\t'                  + pre_name)
print(' ')

##################################################################
## USER VARIABLES
##################################################################
global t_eddy
t_eddy = 10 # number of spectra files per eddy turnover # TODO: input?
## set the figure's axis limits
global k_0
k_0 = 2
## create figure
fig, ax1 = plt.subplots(constrained_layout=True)
ax2 = ax1.twiny()

##################################################################
## LOAD & PLOT DATA
##################################################################
list_k_peak_norm_ave = []
list_k_domain_norm = []
list_k_peak_norm = []
list_Pm = []
list_k_eta = []
list_k_nu = []
list_res = []
## load data for each filepath
for i in range(len(filepaths_data)):
    print('Loading data from: ' + filepaths_data[i])
    ## load the average and standard deviation data
    k_peak = saveData(filepaths_data[i])
    ## read the simulation parameter values from the flash.par file
    nu = 0
    eta = 0
    iproc = 0
    with open(createFilePath([filepaths_data[i], 'flash.par'])) as file_content:
        for line in file_content:
            if len(line.split()) > 1:
                if line.split()[0] == 'diff_visc_nu':
                    nu = float(line.split()[2])
                if line.split()[0] == 'resistivity':
                    eta = float(line.split()[2])
                if line.split()[0] == 'iProcs':
                    iproc = float(line.split()[2])
    if (nu == 0) or (eta == 0):
        raise Exception('nu and eta weren''t found in the FLASH.par file\n.')
    ## calculate simulation characteristics
    global Re
    Re = round(0.1 / (2 * nu), 2)
    Pm = round(nu / eta, 2)
    Rm = round(Re * Pm, 2)
    list_Pm.append(Pm) # append for second axis
    ## calculate scales in the simulation
    list_k_eta.append(Re**(1/4) * (Re*Pm)**(1/2) * k_0)
    list_k_nu.append(Re**(3/4) * k_0)
    list_res.append(36 * iproc)
    ## append data for fitting power law
    tmp_k_peak_norm = [x/list_k_nu[i] for x in k_peak]
    list_k_peak_norm.append(tmp_k_peak_norm)
    list_k_peak_norm_ave.append(calcListAve(tmp_k_peak_norm))
    list_k_domain_norm.append(list_k_eta[i] / list_k_nu[i])
    print(' ')

## create unique set of colours
list_k_col = list_k_eta
list_round_k_col  = [5 * round(x/5) for x in list_k_col] # find closest multiple of 5
list_unique_k_col = sorted(list(set(list_round_k_col)))
len_unique_k_col  = len(list_unique_k_col)
## define dictionary of markers
list_k_marker = list_k_nu
list_round_k_marker  = [round(x, 0) for x in list_k_marker]
list_unique_k_marker = sorted(list(set(list_round_k_marker)))
list_marker_cycle    = dict([(4, 'o'), (11, 's'), (40, '^')])

## plot data for each filepath
print('Plotting data...')
for i in range(len(filepaths_data)):
    print('\t Plotting: ' + filepaths_data[i])
    ## find which color the sim corresponds with
    tmp_col_index = list_unique_k_col.index(list_round_k_col[i])
    tmp_col = sns.color_palette('Blues', n_colors=len_unique_k_col)[tmp_col_index]
    ## choose marker
    tmp_marker = list_marker_cycle[round(list_k_nu[i], 0)]
    ## plot mean +/- std
    ax1.errorbar(list_k_domain_norm[i], list_k_peak_norm_ave[i], yerr=stdev(list_k_peak_norm[i]), 
        color=tmp_col, fmt=tmp_marker,
        markersize=10, linestyle='None', zorder=1)
print(' ')

##################################################################
## FIT POWER LAW
##################################################################
# m, b = np.polyfit(  np.array([np.array(tmp_x) for tmp_x in list_k_domain_norm]), 
#                     np.array([np.array(tmp_y) for tmp_y in list_k_peak_norm_ave]), 1)
# data = np.linspace(min(list_k_domain_norm), max(list_k_domain_norm), 100)
# ax1.plot(data, m*data + b, 'r--')
# ax1.text(0.95, 0.05, r"$%0.2f(k_\eta / k_\nu) %0.2f$"%(m, b), color='r', 
#     transform=ax1.transAxes, horizontalalignment='right', verticalalignment='bottom', fontsize=20)

##################################################################
## LABEL PLOT
##################################################################
## add colorbar
cmap = plt.get_cmap('Blues')
colorbar_index(ncolors=len_unique_k_col, cmap=cmap, nlabels=list_unique_k_col, label_title=r'$k_\eta$')
## add legend
list_handles = []
for i in range(len(list_unique_k_marker)):
    list_handles.append(Line2D([0], [0], color='k', 
        marker=list_marker_cycle[round(list_unique_k_marker[i], 0)],
        linestyle='None',
        label=r"$k_\nu =$ " + str(int(round(list_unique_k_marker[i], 0)))) )
plt.legend(handles=list_handles, handletextpad=0.1, fontsize=18)
## label bottom axis (k_eta / k_nu)
ax1.set_xlabel(r'$k_\eta / k_\nu$', fontsize=22)
ax1.set_ylabel(r'$k_\mathrm{p} / k_\nu$', fontsize=22)
## label top axis (Pm)
list_unique_Pm = sorted(list(set(list_Pm)))
axis_Pm_tick_locs = [round(x**(1/2), 0) for x in list_unique_Pm]
ax2.set_xlim(ax1.get_xlim())
ax2.set_xticks(axis_Pm_tick_locs)
ax2.set_xticklabels(Pm_axis(list_unique_Pm))
ax2.set_xlabel(r"Pm", fontsize=22)
ax2.set_xticks([], minor=True)

##################################################################
## SAVE FIGURE
##################################################################
print('Saving figure...')
fig_name = createFilePath([filepath_plot, pre_name]) + '_scale_dependance.pdf'
plt.savefig(fig_name)
plt.close()
print('Figure saved: ' + fig_name)
print(' ')

## END OF PROGRAM
