# PYTHON - Neco Kriel (Summer 2019)
# plotting Turb.dat file's data

##################################################################
## MODULES
##################################################################
import os # for: clearing terminal + looping over files in directory
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from IPython import get_ipython
from sklearn.linear_model import LinearRegression


##################################################################
## PREPARE TERMINAL/WORKSPACE/CODE
##################################################################
os.system('clear')                  # clear terminal window
get_ipython().magic('reset -sf')    # clear workspace
plt.close('all')                    # close all pre-existing plots
mpl.style.use('classic')            # plot in classic style
plt.rc('font', family='serif')      # specify font choice


##################################################################
## USER VARIABLES
##################################################################
## params: simulation specs
var_L    = 1
var_mach = 0.1
t_eddy   = var_L/(2*var_mach) # calculate the (large scale) eddy turnover-time

## params: locating file
folder_main = os.path.dirname(os.path.realpath(__file__))   # get directory where file is saved
folder_sub  = '/simDyn256/'                                   # folder where data is located

## params: file name
bool_folder_contents = bool(0)
name_file = 'Turb.dat'

## params: data-fields
bool_data_names = bool(1)
var_x           = 0                             # time
var_label_x     = r'$t/t_{\mathregular{eddy}}$' # x-axis label
var_y           = 29                            # common options: 6 (E_kin), 8 (rms_Mach), 29 (E_mag)
var_label_y     = r''                   # y-axis label
bool_norm_dat   = bool(0)                       # normalise by initial value
var_scale       = 'log'                      # options: linear, log

# params: target eddy turnover-time to fit exponential curve to
bool_ave        = bool(0)
bool_regression = bool(0)
t_eddy_min      = 3
t_eddy_max      = 10
text_x          = 8 # 3.5
text_y          = 0.08 # 1e5
var_fontsize    = 18
bool_save_fig   = bool(0)


##################################################################
## ADJUSTMENTS
##################################################################
if var_y == 8:
    var_label_y     = r'$\mathcal{M}$'
    bool_norm_dat   = bool(0)
    bool_ave        = bool(1)
    bool_regression = bool(0)
    var_scale       = 'linear'
elif var_y == 29:
    var_label_y     = r'$E_{\mathregular{m}}/E_{\mathregular{m}0}$'
    bool_norm_dat   = bool(1)
    bool_ave        = bool(0)
    bool_regression = bool(1)
    var_scale       = 'log'

##################################################################
## PRINT OUT/LOAD DATA
##################################################################
## print names of all files in directory
directory = folder_main + folder_sub
if bool_folder_contents:
    print('Files in Directory:')
    print('---------------------')
    print('\n'.join(sorted(os.listdir(directory))))
    print('\n')
else:
    # read each row from flash.dat to a list
    # https://stackoverflow.com/questions/37956344/reading-and-doing-calculation-from-dat-file-in-python
    data = np.array([i.strip().split() for i in open(directory + name_file).readlines()])

    # print the contents (headers) of file
    if bool_data_names:
        print('Header:')
        print('---------------------')
        print('\n'.join(data[0,:]))
        print('\n')


##################################################################
## PLOTTING CODE
##################################################################
if not(bool_folder_contents):
    # define x-axis data (time)
    data_x = list([t/t_eddy for t in map(float, data[1:, var_x])]) # time [eddy turnover-time]

    # define y-axis data
    if bool_norm_dat:
        # y_data = var_y / var_y[1]: ignore first point (division by '0')
        data_y = [float(data[1, var_y])]
        temp_int = float(data[2, var_y])
        data_y.extend(list([x/temp_int for x in map(float, data[2:, var_y])]))
    else:
        # y_data = var_y
        data_y = list(map(float, data[1:, var_y]))

    # regression analysis
    if (bool_regression or bool_ave):
        index_min   = min(enumerate(data_x), key=lambda x: abs(t_eddy_min - x[1]))[0]
        index_max   = min(enumerate(data_x), key=lambda x: abs(t_eddy_max - x[1]))[0]
        fit_x       = list(map(float, data_x[index_min:index_max]))
        fit_y       = list(map(float, data_y[index_min:index_max]))
        if bool_regression:
            log_y       = np.log(fit_y)
            m, c        = np.polyfit(fit_x, log_y, 1)    # fit log(y) = m*log(x) + c
            fit_y       = np.exp([m*x+c for x in fit_x]) # calculate the fitted values of y 
        elif bool_ave:
            var_dt    = np.diff(fit_x)
            var_ave_y = [(prev+cur)/2 for prev, cur in zip(fit_y[:-1], fit_y[1:])]
            ave_y     = sum(var_ave_y * var_dt) / (t_eddy_max - t_eddy_min)
            fit_y     = np.repeat(ave_y, len(fit_y))

    # plot
    print('Plotting: Started')
    plt.figure(figsize=(10, 7), dpi=80)
    plt.plot(data_x, data_y, 'k')
    if (bool_regression or bool_ave):
        if bool_regression:
            plt.text(text_x, text_y,
                'm=%.4f' % m, color='green', fontsize=var_fontsize,
                horizontalalignment='left',
                verticalalignment='top')
        else:
            plt.text(text_x, text_y,
            'ave=%.4f' % ave_y, color='green', fontsize=var_fontsize,
            horizontalalignment='left',
            verticalalignment='top')
        plt.plot(fit_x, fit_y, 'g--', linewidth=2)
    plt.xlabel(var_label_x, size=var_fontsize)
    plt.ylabel(var_label_y, size=var_fontsize)
    plt.yscale(var_scale)
    print('Plotting: Finished')

    # show plot / save
    if bool_save_fig:
        # add figure name flag that var_3 was defined
        name_fig = (directory + 'plot_' + data[0, var_y][4:] + '.png')
        plt.savefig(name_fig)
        print('\nFigure saved: ' + name_fig)
    else:
        plt.show()
