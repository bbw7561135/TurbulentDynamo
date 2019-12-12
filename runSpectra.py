#!/usr/bin/env python2

# PYTHON - Neco Kriel (Summer 2019)
# running spectra code

##################################################################
## MODULES
##################################################################
import os
import argparse

##################################################################
## FUNCTIONS
##################################################################
def meetsCondition(element):
    return bool(element.startswith('Turb_hdf5_plt_cnt_') and not(element.__contains__('_spect_')))

##################################################################
## COMMAND LINE ARGUMENT INPUT
##################################################################
ap = argparse.ArgumentParser(description = 'A bunch of input arguments')
ap.add_argument('-np',  required=True, help='Number of processors', type=str)
ap.add_argument('-dir', required=True, help='Directory of files',   type=str)
args      = vars(ap.parse_args())
np        = args['np']
directory = args['dir']
print("Began running the spectra code in folder: " + directory)

##################################################################
## RUNNING CODE
##################################################################
# loop over directory and execute spectra for each 'plot count' file
file_names = sorted(filter(meetsCondition, os.listdir(directory)))
print('\n'.join(file_names))
for file_name in file_names:
    print('looking at: ' + file_name)
    os.system('mpirun -np ' + np + ' spectra_mpi_sp ' + file_name)

print("Finished running the spectra code.")
