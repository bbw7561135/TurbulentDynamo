#!/usr/bin/env python3

''' AUTHOR: Neco Kriel
    
    EXAMPLE: 
    runSpectra
        (required)
            -base_path      /Users/dukekriel/Documents/University/Year4Sem2/Summer-19/ANU-Turbulence-Dynamo/dyna288_Bk10/hdf5Files
            -num_proc       4
        (optional)
            -check_only     False
            -file_start     0
            -file_end       np.Inf
            -num_proc       8
'''

##################################################################
## MODULES
##################################################################
import os
import argparse
import numpy as np
os.system('clear') # clear terminal window

##################################################################
## FUNCTIONS
##################################################################
def meetsFirstCondition(element):
    global file_start, file_end
    ## accept files that look like: Turb_hdf5_plt_cnt_
    if (element.startswith('Turb_hdf5_plt_cnt_') and not(element.endswith('.dat'))):
        ## check that the file is within the domain range
        bool_domain_right = bool(int(element.split('_')[4]) >= file_start) and (int(element.split('_')[4]) <= file_end)
        ## check the file is not magnetic or velocity data
        bool_spectra = not(element.__contains__('_spect_'))
        return (bool_spectra and bool_domain_right)
    else:
        return False

def meetsSecondCondition(element):
    global file_start, file_end
    ## accept files that look like: Turb_hdf5_plt_cnt_*(mags.dat or vels.dat)
    if (element.startswith('Turb_hdf5_plt_cnt_') and (element.endswith('mags.dat') or element.endswith('vels.dat'))):
        ## check that the file is within the domain range
        bool_domain_right = bool(int(element.split('_')[4]) >= file_start) and (int(element.split('_')[4]) <= file_end)
        return bool_domain_right
    else:
        return False

def processHDF5(file_names):
    for file_name in file_names:
        print('--------- Looking at: ' + file_name + ' -----------------------------------')
        os.system('mpirun -np ' + num_proc + ' spectra_mpi_sp ' + file_name)
        print(' ')

##################################################################
## COMMAND LINE ARGUMENT INPUT
##################################################################
global file_start, file_end
ap = argparse.ArgumentParser(description='A bunch of input arguments')
## ------------------- DEFINE OPTIONAL ARGUMENTS
ap.add_argument('-check_only', type=bool, default=False,  required=False, help='Only check which files dont exist')
ap.add_argument('-file_start', type=int,  default=0,      required=False, help='File number to start processing from')
ap.add_argument('-file_end',   type=int,  default=np.Inf, required=False, help='File number to end processing from')
ap.add_argument('-num_proc',   type=str,  default='8',    required=False, help='Number of processors')
## ------------------- DEFINE REQUIRED ARGUMENTS
ap.add_argument('-base_path',  type=str, required=True, help='Filepath to where files are located')
## ---------------------------- OPEN ARGUMENTS
args = vars(ap.parse_args())
## ---------------------------- SAVE PARAMETERS
bool_check_only = args['check_only'] # only check for missing hdf5 spectra files
filepath        = args['base_path']  # base filepath to data
file_start      = args['file_start'] # first file to process
file_end        = args['file_end']   # last file to process
num_proc        = args['num_proc']   # number of processors
## ---------------------------- ADJUST ARGUMENTS
## remove the trailing '/' from the input filepath
if filepath.endswith('/'):
    filepath = filepath[:-1]
## replace any '//' with '/'
filepath = filepath.replace('//', '/')
## ---------------------------- START CODE
print("Began running the spectra code in folder: " + filepath)
print("First file number: " + str(file_start))
print("Last file number: " + str(file_end))
print("Number processors: " + str(num_proc))
print(' ')

##################################################################
## RUNNING CODE
##################################################################
# loop over filepath and execute spectra for each 'plot count' file
file_names = sorted(filter(meetsFirstCondition, os.listdir(filepath)))
print('There are ' + str(len(file_names)) + ' files to process.')
print('These files are:')
print('\t' + '\n\t'.join(file_names))
print(' ')
## loop over and process file names
if not(bool_check_only):
    processHDF5(file_names)

## save the names of all output files
spect_names = sorted(filter(meetsSecondCondition, os.listdir(filepath)))
## check if there are any files that weren't processed properly
redo_names = [] # initialise list of files to reprocess
for file_name in file_names:
    ## for each hdf5 file, check if there exists magnetic and velocity output files
    bool_mags_exists = False
    bool_vels_exists = False
    if (file_name + '_spect_mags.dat') in spect_names: bool_mags_exists = True
    if (file_name + '_spect_vels.dat') in spect_names: bool_vels_exists = True
    ## if both the magnetic and velocity files dont exists, then reprocess the hdf5 file
    if (not(bool_mags_exists) or not(bool_vels_exists)): redo_names.append(file_name)
print('There were ' + str(len(redo_names)) + ' files processed incorrectly.')
print('These files were:')
print('\t' + '\n\t'.join(redo_names)) # print file names
print(' ')
## loop over file names and processes them
print('Processing these files again...')
processHDF5(redo_names)

print("Finished running the spectra code.")

## END OF PROGRAM