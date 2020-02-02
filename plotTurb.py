##################################################################
## MODULES
##################################################################
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from IPython import get_ipython

##################################################################
## PREPARE TERMINAL/WORKSPACE/CODE
#################################################################
os.system('clear')                  # clear terminal window
get_ipython().magic('reset -sf')    # clear workspace
plt.close('all')                    # close all pre-existing plots
mpl.style.use('classic')            # plot in classic style

##################################################################
## USER DEFINED VARIABLES
##################################################################
t_eddy  = 5 # L/(2*Mach)
## specify where files are located and needs to be saved
folder_main      = os.path.dirname(os.path.realpath(__file__)) # get directory where file is stored
folder_name      = 'dyna288_Bk10' # folder where data Turb.dat is located
folder_vis       = 'visFiles' # folder where visualisation should be saved
bool_disp_folder = bool(0) # display all files that end with: name_file_*
bool_disp_header = bool(1) # display all the header names stored in Turb.dat
## 6 (E_kin), 8 (rms_Mach), 29 (E_mag)
var_y            = 8 # y-axis variable
label_y          = r''
anotate_y        = r''
## set the figure's axis-limits
bool_set_lim     = bool(0)
xlim_min         = 3
xlim_max         = 9
ylim_min         = 1.0e-16
ylim_max         = 4.2e-03
var_scale        = ''
## set x-range for regression and averaging
x_min            = 3
x_max            = 5.7
## extra plotting features
bool_norm_dat    = bool(0) # normalise y-axis data
bool_ave         = bool(1) # plot average of data over specified x-range
bool_regression  = bool(0) # plot regression line for data over specified x-range
## should the plot be saved?
bool_save_fig    = bool(0)

##################################################################
## AUTOMATIC ADJUSTMENTS
##################################################################
if var_y == 6:
    ## mach number
    label_y       = r'$E_{\mathregular{k}}/E_{\mathregular{k}0}$'
    bool_norm_dat = bool(1)
    var_scale     = 'log'
elif var_y == 8:
    ## mach number
    label_y       = r'$\mathcal{M}$'
    bool_norm_dat = bool(0)
    var_scale     = 'linear'
elif var_y == 29:
    ## magnetic field
    label_y       = r'$E_{\mathregular{m}}/E_{\mathregular{m}0}$'
    bool_norm_dat = bool(1)
    var_scale     = 'log'

##################################################################
## FUNCTIONS
##################################################################
def createFilePath(names):
    return ('/'.join(names) + '/')

##################################################################
## INITIALISING VARIABLES
##################################################################
directory = createFilePath([folder_main, folder_name]) + 'Turb.dat'
## open figure
fig = plt.figure(figsize=(10, 7), dpi=100)
ax  = fig.add_subplot()
## define x-variable
var_x   = 0
label_x = r'$t/t_{\mathregular{eddy}}$'
## load data
data_split = [x.split() for x in open(directory).readlines()]
## show the header file
if bool_disp_header: print('\nHeader names: for Turb.dat\n--------------\n' + '\n'.join(data_split[0]))
## save maximum number of columns in a row. less indicated rows stores a message.
len_thresh = len(data_split[0]) # ignore extra lines (len < len_thresh) resulting from restarting the simulation

##################################################################
## LOAD DATA
##################################################################
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
## save analysis data
index_min = min(enumerate(data_x), key=lambda x: abs(x_min - x[1]))[0]
index_max = min(enumerate(data_x), key=lambda x: abs(x_max - x[1]))[0]
fit_x     = list(map(float, data_x[index_min:index_max]))
fit_y     = list(map(float, data_y[index_min:index_max]))

##################################################################
## PLOTTING DATA
##################################################################
plt.plot(data_x, data_y, 'k')

##################################################################
## ADD REGRESSION / AVERAGING
##################################################################
## plot regression analysis
if bool_regression:
    log_y = np.log(fit_y)
    m, c  = np.polyfit(fit_x, log_y, 1)    # fit log(y) = m*log(x) + c
    fit_y = np.exp([m*x + c + 1 for x in fit_x]) # calculate the fitted values of y 
    plt.plot(fit_x, fit_y, 'k--', linewidth=1)
    ax.text(0.75, 0.23,
        r"$m = %0.1f$"%m,
        fontsize=20, color='black', 
        ha="left", va='top', transform=ax.transAxes)
    ax.text(0.75, 0.15,
        r"$c = %0.1f$"%c,
        fontsize=20, color='black', 
        ha="left", va='top', transform=ax.transAxes)
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
        fontsize=20, color='black', 
        ha="left", va='top', transform=ax.transAxes)

##################################################################
## LABEL and ADJUST PLOT
##################################################################
## major grid
ax.grid(which='major', linestyle='-', linewidth='0.5', color='black', alpha=0.35)
## minor grid
ax.grid(which='minor', linestyle='--', linewidth='0.5', color='black', alpha=0.2)
## set axis limits
if bool_set_lim:
    plt.xlim(xlim_min, xlim_max)
    plt.ylim(ylim_min, ylim_max)
## label plot
plt.xlabel(label_x, fontsize=20)
plt.ylabel(label_y, fontsize=20)
## scale y-axis
ax.set_yscale(var_scale)

##################################################################
## SAVE IMAGE
##################################################################
## save figure
if bool_save_fig:
    name_fig = (createFilePath([folder_main, folder_name, folder_vis]) + 
                folder_name + '_turb_' + data_split[0][var_y][4:] + '.png')
    plt.savefig(name_fig)
    print('\nFigure saved: ' + name_fig)
## display the plot
plt.show()

## END OF PROGRAM