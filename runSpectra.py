#!/usr/bin/env python3

''' AUTHOR: Neco Kriel
    
    EXAMPLE: 
    runSpectra
        -base_path /Users/dukekriel/Documents/University/Year4Sem2/Summer-19/ANU-Turbulence-Dynamo/dyna288_Bk10/hdf5Files
        -num_proc 4
'''

##################################################################
## MODULES
##################################################################
import os
import argparse
import numpy as np

##################################################################
## FUNCTIONS
##################################################################
def meetsCondition(element):
    global file_start, file_end
    if (len(element.split('_')) >= 4):
        return bool(element.startswith('Turb_hdf5_plt_cnt_') and not(element.__contains__('_spect_')) and (int(element.split('_')[4]) >= file_start) and (int(element.split('_')[4]) <= file_end))
    else:
        return False

##################################################################
## COMMAND LINE ARGUMENT INPUT
##################################################################
global file_start, file_end
ap = argparse.ArgumentParser(description='A bunch of input arguments')
## ------------------- DEFINE OPTIONAL ARGUMENTS
ap.add_argument('-file_start', required=False, help='File number to start processing from', type=int, default=0)
ap.add_argument('-file_end',   required=False, help='File number to end processing from',   type=int, default=-1)
ap.add_argument('-num_proc',   required=False, help='Number of processors',                 type=str, default='8')
## ------------------- DEFINE REQUIRED ARGUMENTS
ap.add_argument('-base_path', required=True, help='Filepath to where files are located', type=str)
## ------------------- OPEN ARGUMENTS
args        = vars(ap.parse_args())
## ------------------- SAVE ARGUMENTS
num_proc    = args['num_proc']   # number of processors
filepath    = args['base_path']  # base filepath to data
file_start  = args['file_start'] # file number to start processing from
## file number to end processing on
if (args['file_end'] < 0):
    file_end = np.Inf
else:
    file_end = args['file_end']
## ------------------- ADJUST ARGUMENTS
## remove the trailing '/' from the input filepath
if filepath.endswith('/'):
    filepath = filepath[:-1]
## replace '//' with '/'
filepath = filepath.replace('//', '/')
## ------------------- START CODE
print("Began running the spectra code in folder: " + filepath)
print(' ')

##################################################################
## RUNNING CODE
##################################################################
# loop over filepath and execute spectra for each 'plot count' file
file_names = sorted(filter(meetsCondition, os.listdir(filepath))) # save file names
print('The files in the filepath that satisfied meetCondition:')
print('\t' + '\n\t'.join(file_names)) # print file names
print(' ')

for file_name in file_names: # loop over file names and processes them
    print('--------- Looking at: ' + file_name + ' -----------------------------------')
    os.system('mpirun -np ' + num_proc + ' spectra_mpi_sp ' + file_name)
    print(' ')

print("Finished running the spectra code.")

## END OF PROGRAM