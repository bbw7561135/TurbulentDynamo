#!/usr/bin/env python3

''' AUTHOR: Neco Kriel
    
    EXAMPLE: 
    spectra_dPlot_inline.py 
        (required)
            -base_path /Users/dukekriel/Documents/University/Year4Sem2/Summer-19/ANU-Turbulence-Dynamo 
            -dat_folder1    dyna288_Bk10 
            -dat_folder2    dyna288_Bk100
            -pre_name       dyna288
        (optional)
            -debug          False
            -plot_kin       True
            -plot_mag       True
            -vis_folder     visFiles
            -sub_folder     spectFiles
            -ani_start      0
            -ani_fps        40
            -file_start     0
            -file_end       np.Inf
            -xlim_min       1.0
            -xlim_max       1.3e+02
            -ylim_min       1.0e-25
            -ylim_max       4.2e-03
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
global bool_debug_mode
global filepath_base, file_start, file_end
ap = argparse.ArgumentParser(description='A bunch of input arguments')
## ------------------- DEFINE OPTIONAL ARGUMENTS
ap.add_argument('-debug',       type=str2bool,  default=False,          required=False, help='Debug mode',                 nargs='?', const=True)
ap.add_argument('-plot_kin',    type=str2bool,  default=True,           required=False, help='Plot the kinetic spectra?',  nargs='?', const=False)
ap.add_argument('-plot_mag',    type=str2bool,  default=True,           required=False, help='Plot the magnetic spectra?', nargs='?', const=False)
ap.add_argument('-vis_folder',  type=str,       default='visFiles',     required=False, help='Name of the plot folder')
ap.add_argument('-sub_folder',  type=str,       default='spectFiles',   required=False, help='Name of the folder where the data is stored')
ap.add_argument('-ani_start',   type=str,       default='0',            required=False, help='Animation start number')
ap.add_argument('-ani_fps',     type=str,       default='40',           required=False, help='Animation frame rate')
ap.add_argument('-file_start',  type=int,       default=0,              required=False, help='File number to start plotting from')
ap.add_argument('-file_end',    type=int,       default=np.Inf,         required=False, help='Number of files to process')
ap.add_argument('-xlim_min',    type=float,     default=1.0,            required=False, help='Figure xlim minimum')
ap.add_argument('-xlim_max',    type=float,     default=1.3e+02,        required=False, help='Figure xlim maximum')
ap.add_argument('-ylim_min',    type=float,     default=1.0e-25,        required=False, help='Figure ylim minimum')
ap.add_argument('-ylim_max',    type=float,     default=4.2e-03,        required=False, help='Figure ylim maximum')
## ------------------- DEFINE REQUIRED ARGUMENTS
ap.add_argument('-base_path',   type=str, required=True, help='Filepath to the base folder')
ap.add_argument('-dat_folder1', type=str, required=True, help='Name of the first folder')
ap.add_argument('-dat_folder2', type=str, required=True, help='Name of the second folder')
ap.add_argument('-pre_name',    type=str, required=True, help='Name of figures')
## ---------------------------- OPEN ARGUMENTS
args = vars(ap.parse_args())
## ---------------------------- SAVE PARAMETERS
bool_debug_mode = args['debug']       # enable/disable debug mode
bool_plot_kin   = args['plot_kin']    # plot the kinetic spectra?
bool_plot_mag   = args['plot_mag']    # plot the magnetic spectra?
file_start      = args['file_start']  # starting processing frame
file_end        = args['file_end']    # the last file to process
ani_start       = args['ani_start']   # starting animation frame
ani_fps         = args['ani_fps']     # animation's fps
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
## ---------------------------- ADJUST ARGUMENTS
## remove the trailing '/' from the input filepath and folders
if filepath_base.endswith('/'):
    filepath_base = filepath_base[:-1]
## replace any '//' with '/'
filepath_base   = filepath_base.replace('//', '/')
## remove '/' from variable names
folder_data_1   = stringChop(folder_data_1, '/')
folder_data_2   = stringChop(folder_data_2, '/')
folder_sub      = stringChop(folder_sub, '/')
folder_vis      = stringChop(folder_vis, '/')
pre_name        = stringChop(pre_name, '/')
## ---------------------------- START CODE
print('Began running the spectra plotting code in base filepath: \n\t' + filepath_base)
print('Data folder 1: '                                                + folder_data_1)
print('Data folder 2: '                                                + folder_data_2)
print('Visualising folder: '                                           + folder_vis)
print('Figure name: '                                                  + pre_name)
print(' ')

##################################################################
## USER VARIABLES
##################################################################
t_eddy = 10 # number of spectra files per eddy turnover # TODO: input?
## specify where files are located and needs to be saved
## first plot's information
label_1_kin = r'$\mathcal{P}_{k_{B}=10, \mathregular{kin}}$'
label_1_mag = r'$\mathcal{P}_{k_{B}=10, \mathregular{mag}}$'
## second plot's information
label_2_kin = r'$\mathcal{P}_{k_{B}=100, \mathregular{kin}}$'
label_2_mag = r'$\mathcal{P}_{k_{B}=100, \mathregular{mag}}$'
## specify which variables you want to plot
global var_x, var_y
var_x = 1  # variable: wave number (k)
var_y = 15 # variable: power spectrum

##################################################################
## INITIALISING VARIABLES
##################################################################
filepath_data_1 = createFilePath([filepath_base, folder_data_1, folder_sub]) # first folder with data
filepath_data_2 = createFilePath([filepath_base, folder_data_2, folder_sub]) # second folder with data
filepath_plot   = createFilePath([filepath_base, folder_vis, 'plotSpectra']) # folder where plots will be saved
file_names_1, num_figs_1 = setupInfo(filepath_data_1)
file_names_2, num_figs_2 = setupInfo(filepath_data_2)
num_figs = min(num_figs_1, num_figs_2)
createFolder(filepath_plot) # create folder where plots are saved

for var_iter in range(num_figs):
    #################### INITIALISE LOOP
    ####################################
    fig = plt.figure(figsize=(10, 7), dpi=100)
    var_time = var_iter/t_eddy # normalise time point by eddy-turnover time
    print('Processing: %0.3f%% complete'%(100 * var_iter/num_figs))
    #################### LOAD DATA
    ##############################
    print('Loading data...')
    name_file_kin = 'Turb_hdf5_plt_cnt_' + '{0:04}'.format(var_iter) + '_spect_vels.dat' # kinetic file
    name_file_mag = 'Turb_hdf5_plt_cnt_' + '{0:04}'.format(var_iter) + '_spect_mags.dat' # magnetic file
    ## load dataset 1
    if bool_plot_kin: data_x_1_kin, data_y_1_kin = loadData(createFilePath([filepath_data_1, name_file_kin])) # kinetic power spectrum
    if bool_plot_mag: data_x_1_mag, data_y_1_mag = loadData(createFilePath([filepath_data_1, name_file_mag])) # magnetic power spectrum
    ## load dataset 2
    if bool_plot_kin: data_x_2_kin, data_y_2_kin = loadData(createFilePath([filepath_data_2, name_file_kin])) # kinetic power spectrum
    if bool_plot_mag: data_x_2_mag, data_y_2_mag = loadData(createFilePath([filepath_data_2, name_file_mag])) # magnetic power spectrum
    #################### PLOT DATA
    ##############################
    print('Plotting data...')
    ## plot dataset 1
    if bool_plot_kin: line_1_kin, = plt.plot(data_x_1_kin, data_y_1_kin, 'k', label=label_1_kin) # kinetic power spectrum
    if bool_plot_mag: line_1_mag, = plt.plot(data_x_1_mag, data_y_1_mag, 'k--', label=label_1_mag) # magnetic power spectrum
    ## plot dataset 2
    if bool_plot_kin: line_1_kin, = plt.plot(data_x_2_kin, data_y_2_kin, 'b', label=label_2_kin) # kinetic power spectrum
    if bool_plot_mag: line_1_mag, = plt.plot(data_x_2_mag, data_y_2_mag, 'b--', label=label_2_mag) # magnetic power spectrum
    #################### LABEL and ADJUST PLOT
    ##########################################
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
    # add legend
    plt.legend(loc='upper right', fontsize=17, frameon=False)
    ## major grid
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.35)
    ## minor grid
    plt.grid(which='minor', linestyle='--', linewidth='0.5', color='black', alpha=0.2)
    #################### SAVE IMAGE
    ###############################
    print('Saving figure...')
    temp_name = createFilePath([filepath_plot, (pre_name + '_spectra={0:06}'.format(int(var_time*10)) + '.png')])
    plt.savefig(temp_name)
    plt.close()
    print('Figure saved: ' + temp_name)
    print(' ')

## create animation
filepath_input  = createFilePath([filepath_plot, (pre_name + '_spectra=%06d.png')])
filepath_output = createFilePath([filepath_plot, ('../' + pre_name + '_ani_spectras_combined.mp4')])
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
    # eg. To check: execute the following within the visualising folder
    # ffmpeg -start_number 0 -i ./plotSlices/dyna288_spectra=%06d.png -vb 40M -framerate 40 -vf scale=1440:-1 -vcodec mpeg4 ./dyna288_ani_spectra.mp4

## END OF PROGRAM