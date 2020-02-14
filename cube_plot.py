#!/usr/bin/env python3

''' AUTHOR: Neco Kriel
    
    EXAMPLE: 
'''

##################################################################
## MODULES
##################################################################
import os
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d
from matplotlib.colors import LogNorm

##################################################################
## PREPARE TERMINAL/CODE
##################################################################
os.system('clear')                  # clear terminal window
plt.close('all')                    # close all pre-existing plots
mpl.style.use('classic')            # plot in classic style

##################################################################
## FUNCTION
##################################################################
def reformatField(field, nx=None, procs=None):
    """
    AUTHORS: 
        James Beattie (version 26 November 2019)
        Neco Kriel (Editing)
    PURPOSE:
        This code reformats the FLASH block / zxy format into zxy format for processing
    INPUTS:
        field   — the FLASH field
        nx      - number of blocks per spatial direction (stored in an array [x, y, z])
        procs   - number of cores per spatial direction (stored in an array [i, j, k])
    OUTPUT:
        field_sorted - the organised 2D field
    """
    ## interpret the function arguments
    iprocs = procs[0]
    jprocs = procs[1]
    kprocs = procs[2]
    nxb = nx[0]
    nyb = nx[1]
    nzb = nx[2]
    ## initialise the output field
    field_sorted = np.zeros([nzb*kprocs, nxb*iprocs, nyb*jprocs])
    ## sort and store the unsorted field into the output field
    for k in range(kprocs):
        for j in range(jprocs):
            for i in range(iprocs):
                field_sorted[i*nxb:(i+1)*nxb, j*nyb:(j+1)*nyb, k*nzb:(k+1)*nzb] = field[k + j*jprocs + i*jprocs*kprocs]
    return field_sorted

##################################################################
## USER VARIABLES
##################################################################
t_eddy = 5
plasma_beta = 1e-10

## TODO: inline command inputs
base_path = '/Users/dukekriel/Documents/University/Year4Sem2/Summer-19/ANU-Turbulence-Dynamo/dyna288_Bk10/Re20_Mach0p1/hdf5Files'
plot_type = '3D'

file_name = 'Turb_hdf5_plt_cnt_0003'

file_path = base_path + '/' + file_name

## 2D plot parameters
var_slice_step = 100 # size of step increments [pixels]

##################################################################
## LOAD DATA
##################################################################
f = h5py.File(file_path, 'r') # load hdf5 file. dimensions: [iProc*jProc*kProc, nzb, nyb, nxb]
print('File keys:\n' + '\n'.join(list(f.keys())) + '\n') # print keys
data = np.array(f['dens'])
f.close() # close the file stream
## reformat data
print('Shape of data before formating: ' + str(data.shape))
data = reformatField(data, [48, 36, 36], [6, 8, 8])
print('Shape of data after formating: ' + str(data.shape))

##################################################################
## PLOT DATA
##################################################################
if plot_type == '2D':
    ## 2D plot - save cross section slices
    ## TODO: time or space plots
    for i in range(0, data.shape[0], var_slice_step):
        plt.imshow(data[i, :, :],
                interpolation='none',
                cmap='plasma',
                norm=LogNorm())
        plt.show()
if plot_type == '3D':
    ## 3D plot - save 3d perspective plot
    data = data[1, :, :]
    x = np.linspace(0, 1, 288)
    X, Y = np.meshgrid(x, x)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.contourf(data, X, Y, 25, zdir='x', cmap='plasma', offset=0, alpha=1, antialiased=True)
    ax.contourf(X, data, Y, 25, zdir='y', cmap='plasma', offset=1, alpha=1, antialiased=True)
    ax.contourf(X, Y, data, 25, zdir='z', cmap='plasma', offset=1, alpha=1, antialiased=True)
    ax.set_xlim((-0.1, 1.1))
    ax.set_ylim((-0.1, 1.1))
    ax.set_zlim((-0.1, 1.1))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=30, azim=140)
    plt.show()
