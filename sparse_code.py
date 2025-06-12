import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
#print(pd.__version__)
pd.set_option('display.max_columns', None)
pd.options.mode.copy_on_write = True 
import dask.dataframe as dd
from numba import njit, jit
import numba as nb
from numba import float64, int64, deferred_type, optional
from numba.core.runtime import rtsys
from numba.experimental import jitclass
from numba import njit, prange
import scipy




def gauss(x, A, t0, sigma):
    """
    Plots a gauss functions using the parameters
    
    Paramters
    ---------
    x : Input values for the gauss function (i.e. the x axis). 
    A : Amplitude of the gauss function 
    t0 : Position of the gaussian peak along the x axis 
    sigma : Width and standard deviation of the peak 
    d : Constant to raise or lower the function to the data
    
    """
    
    y = abs(A)*np.exp(-(x-t0)**2/(2*abs(sigma)**2))
    return y


def Simpsons(func, a, b, N = int(1e6)): 
    
    """ 
    
    Calculates the integral of an input function over a given range using simpsons rule
    
    Parameters 
    ---------
    
    func: Input function 
    a: left bound of the range you want to integrate over 
    b: right bound of the rage you want to integrate over 
    N: Total number of intervals in the integration (has a default value but can be changed)
    dx: The size of each interval 
    x_vals: The values of the range to be integrated over 
    x_pos: The even indices of the x_vals array corresponding to positive values
    x_neg: The odd indices of the x_vals array corresponding to negative values
    pos_func_vals: The function values evaluated at positive values 
    neg_func_vals: The function values evaluated at negative values
    Sum: The calculated sum of the integral to be returned
    
    """
    
    
    
    if N % 2 != 0:
        raise Exception("N must be an even number for Simpsons rule!")
    
    dx = (b - a)/N 
    
    x_vals = np.linspace(a,b,N+1)
    
    x_pos = x_vals[2:-1:2]           # Selects even indices starting from i = 2 element 
    x_neg = x_vals[1:-1:2]           # Selects odd indices starting from i = 1 element
    
    pos_func_vals = func(x_pos)
    neg_func_vals = func(x_neg) 
    
    Sum = dx/3*(func(a) + 4*np.sum(neg_func_vals) + 2*np.sum(pos_func_vals) + func(b))
    
    return Sum

@njit
def calc_phase(timestamps, t0):
    return np.mod(timestamps, t0)

@njit(parallel=True)
def _find_best_t0(timestamps, t0s, binmod = 1):
    num_t0s = len(t0s)
    ncounts = np.zeros(num_t0s, dtype=np.uint16)
    
    for ii in prange(num_t0s):
        t0 = t0s[ii]
        phases = calc_phase(timestamps, t0)
        maxbins = int((t0/1.5625)/binmod)
        #print(t0)
        #print(maxbins)
        counts, _ = np.histogram(phases, bins=maxbins, range=(0, t0))
        ncounts[ii] = np.sum(counts == 0)
        
    return ncounts

def find_best_t0(df, starting_t0 = 2641.6625, binmod = 1, prints = True):
    #print(starting_t0)
    timestamps = df['t_ns'].to_numpy()
    
    t0s = np.linspace(starting_t0 - .01, starting_t0 + 0.01, 2001) # 1000 over 5 ps
    ncounts = _find_best_t0(timestamps, t0s, binmod)
    t0 = t0s[np.argmax(ncounts)] 
    
    if prints:
        print(f'Broad search: found optimal t_0 as {t0:,.7f} ns')

        plt.figure()
        plt.plot(t0s, ncounts)
        plt.title('Searching for best $t_0$: Broad search')
        plt.xlabel('$t_0$ (ns)')
        plt.ylabel('Gap size')
        plt.show() 
    
    t0s = np.linspace(t0 - .0005, t0 + 0.0005, 2001) # 10000 over 1 ps
    ncounts = _find_best_t0(timestamps, t0s, binmod)
    t0 = t0s[np.argmax(ncounts)] 
    
    if prints:
        print(f'Tight search: found optimal t_0 as {t0:,.7f} ns')

        plt.figure()
        plt.plot(t0s, ncounts)
        plt.title('Searching for best $t_0$: Tight search')
        plt.xlabel('$t_0$ (ns)')
        plt.ylabel('Gap size')
        plt.show()   
    
    return t0

def add_phase_col(df, t0, binsize = 1, prints = True):
    df['phase'] = calc_phase(df['t_ns'].to_numpy(), t0)
    maxbins = int((t0/1.5625)/binsize)
    
    if prints:
        plt.figure()
        plt.hist(df['phase'], bins = maxbins, range=(0, t0))
        plt.title('Recovered Storage Ring Filling Pattern')
        plt.xlabel('Phase (ns)')
        plt.ylabel('Counts')
        plt.show()
        
    return df


def assign_photon_value(value, thresholds):
    for i, threshold in enumerate(thresholds):
        if value < threshold:
            return i + 1
    return len(thresholds) + 1


def chi_square(y_obs, y_fit, dof, uncertainty="default"):
    """
    Calculate the reduced chi-square for a fit.

    Parameters:
    ------------
    y_obs: array-like
        Observed data values (e.g., histogram bin counts).
    y_fit: array-like
        Model or fit values.
    dof: int
        Degrees of freedom (number of data points minus number of fit parameters).
    uncertainty: str, optional
        The type of uncertainty to use. Options are:
        - "poisson": Uses Poisson uncertainty (sqrt(y_obs)).
        - "custom": Allows passing a custom uncertainty array (requires modifying the function).
        - "default": Uses the fit values (y_fit) as the uncertainty.

    """
    y_fit = np.maximum(y_fit, 1e-8)
    y_obs = np.maximum(y_obs, 1e-8)
    
    if uncertainty == "poisson":
        sigma = np.sqrt(y_obs)
    elif uncertainty == "default":
        sigma = np.sqrt(y_fit)
    else:
        raise ValueError("Invalid uncertainty type. Use 'poisson' or 'default'.")
    
    sigma = np.maximum(sigma, 1e-8)
    
    chi2 = np.sum((y_obs - y_fit)**2 / sigma**2)
    
    return chi2 / dof


def find_peaks(bin_centers, counts, N_loops = 20): 

    """

    This function fits all of the ToT peaks for increasing number of photon hits, auotomatically calculates the maximum ToT value to be considered and returns 
    lists of parameters and functions to be used in other functions.

    Parameters: 
    -------------

    bin_centers: The centers of the input ToT bins 
    counts: The counts of the input ToT bins 
    N_loops: The maximum number of peaks to be considered, default is set to 20 but is a parameter that can be set higher/lower
    counts_original: Copy of the original counts array before subtraction
    peak0_x: The "x" value of the first (n_photon = 1) peak
    peak0_counts: The number of counts corresponding to the max value in the (n_photon = 1) peak
    peak_guesses: A list housing all of the peaks which may attempt to be fitted, elements are an integer multiple of the ToT value corresponding to the first peak
    i: Integer element used in the construction of the peak_guesses list 
    peaks_lists, params_list, fit_funcs, chi2_vals, x0_vals, sigma_vals: Lists holding corresponding parameters/functions 
    n: Integer element used in the main loop, corresponds to the associated peak number/n_photon value
    param_search: Boolean value used to determine if the more complex failsafe fit method needs to be used for a peak should a fit fail (found in the nested loops)
    A_guess: Guess of the amplitude value for a given peak based on the maximum value of the counts array. Note that the counts array changes as subtraction occurs
    x0_guess: The guess of the x0 value based on the current peak value found from peak_guesses
    n1_mask: Boolean mask used in the case where n = 1 to properly calculate parameters of the first peak
    sigma_guess: Guess of the sigma value, initially found using np.std for peak 1, and then calculated using prevoius value with the assumption the sigma value
                 should rise using a 1.05x multipler (This is a very arbitrary method and is potentially worth discussing better/more accurate methods). 
    fit_mask: Boolean mask used to pull out bin center and counts values for each peak by taking the data within a +/2 sigma range of the peak bin center value
    popt, pcov: Arrays holding fit parameters found using scipy.optimize.curve_fit
    x0_left, x0_right: Left/Right bound of the new x0 search range if param_search is True
    sig_left, sig_right: Left/Right bound of the new sigma search range if param_search is True
    new_x0_guesses, new_sig_guesses: Arrays housing the new guesses of x0, sig in the param_search if True
    fit_results: Results of the additional fit attempts if param_search is True 
    mask: Range of bin_centers values in the param_search using the left and right x0 bounds 
    A, x0, sig: Parameter values found from curve_fit
    dof: Degrees of freedom used to calculate chi2 
    fit_vals_loop: Fit values of the additional searches if param_search is True
    Chi2: Calculated chi square values
    best_fit: Best fit value from additional parameter search found from selecting values corresponding to minimum chi2 value of additional search params
    best_A, best_x0, best_sig: Best vals for A, x0, and sigma found from best_fit
    fit_func: Fit function for an individual peak fit
    ToT_max: Maximum ToT value, any peaks found beyond this value will be dropped
    subtract_mask: Mask used to decide which values will be subtracted 

    """

    counts_original = counts.copy()
    
    peak0_x = bin_centers[np.argmax(counts)]

    peak0_counts = np.array(counts)[np.argmax(counts)]

    peak_guesses = [i*peak0_x for i in range(1,N_loops+1)]

    
    
    peaks_list = []
    params_list = []
    fit_funcs = []
    chi2_vals = []
    x0_vals = []
    sigma_vals = []

    n = 1

    param_search = False

    ## Main Loop
    for peak in peak_guesses: 
        A_guess = counts.max()
        x0_guess = peak
        
        if n == 1: 

            n1_mask = (bin_centers > peak0_x - peak0_x/2) & (bin_centers < peak0_x + peak0_x/2)
            
            bin_cents_peak0 = bin_centers[n1_mask]
            counts_peak0 = counts[n1_mask]

            sigma_guess = np.std(bin_cents_peak0)

        else:

            sigma_guess = 1.05*params_list[n-2][2]


        fit_mask = (bin_centers > (peak - 2*sigma_guess)) & (bin_centers < (peak + 2*sigma_guess))

        try:
            popt, pcov = curve_fit(gauss, bin_centers[fit_mask], counts[fit_mask], p0=[A_guess, x0_guess, abs(sigma_guess)],
                                  bounds = ((-np.inf, x0_guess - 50, -np.inf),(np.inf, x0_guess + 50, np.inf)))
        except RuntimeError as e:
    
            current_peak_val = peak_guesses[n]
            x_guess_new = np.arange(current_peak_val - 50, current_peak_val + 50, 5)
            sig_guesses_new = np.arange(sigma_guess - 150, sigma_guess + 150, 10)
            for x_guess in x_guess_new:
                for sig_guess in sig_guesses_new:
                    try:
                        popt, pcov = curve_fit(
                            gauss, 
                            bin_centers[fit_mask], 
                            counts[fit_mask], 
                            p0=[A_guess, x_guess, abs(sig_guess)]
                        )
                        break
                    except RuntimeError:
                        pass
                else:
                    continue
                break
            else:
                print("All fallback guesses failed.")
                popt = None
                pcov = None

        A = popt[0]
        x0 = popt[1]
        sig = popt[2]
        
        ## Checking if additional parameter search is needed (if n is not = 1)
        if n != 1:
            # If there is a significant difference between the current sigma value and previous, it's considered a bad fit and additional parameter search is 
            # activated
            if abs(sig - params_list[n-2][2]) > 100:
                param_search = True

                if param_search == True: 
                    #print("Param search is true for n = {}".format(n))
                    #print("Initial x0 guess: {:.2f}".format(peak))
                    x0_left = x0_guess - 250
                    x0_right = x0_guess + 250
                    
                    sig_left = params_list[n-2][2] - 150
                    sig_right = params_list[n-2][2] + 150
                    
                    new_x0_guesses = np.arange(x0_left, x0_right + 25, 25)
                    new_sig_guesses = np.arange(sig_left, sig_right + 10, 10)
                    
                    fit_results = []
                    
                    for new_x0 in new_x0_guesses:
                        for new_sig in new_sig_guesses: 
                            mask = (bin_centers >= x0_left) & (bin_centers <= x0_right)
                            
                            try:
                                popt, pcov = curve_fit(gauss, bin_centers[mask], counts[mask],
                                                       p0 = [A_guess, new_x0, new_sig])
    
                                A = popt[0]
                                x0 = popt[1]
                                sig = popt[2]
    
                                dof = len(bin_centers[mask]) - len(popt)
    
                                fit_vals_loop = gauss(bin_centers[mask], A, x0, sig)
    
                                chi2 = chi_square(counts[mask], fit_vals_loop, dof)
    
                                fit_results.append((chi2, A, x0, sig))
                            # If a bad fit is found in the additional param search, these values are appended and will not be considered 
                            except RuntimeError as e: 
                                fit_results.append((float('inf'), np.nan, np.nan, np.nan))

        if param_search == True: 
            best_fit = min(fit_results, key = lambda x: x[0])
            best_chi2, best_A, best_x0, best_sig = best_fit
            
            A = best_A; x0 = best_x0; sig = best_sig

        x0_vals.append(x0)

        param_search == False
                        
        params = [A,x0, sig]

        params_list.append(params)

        fit_func = lambda x, A = popt[0], x0 = popt[1], sig = popt[2]: gauss(x, A, x0, sig)
        fit_funcs.append(fit_func)


        if n >= 3: 
            if A < 20:
                ToT_max = x0 + 2*sig
                break
            else:
                ToT_max = x0 + 2*sig

        subtract_mask = (bin_centers > x0 - 2.5 * abs(sig)) & (bin_centers < x0 + 2.5 * abs(sig))

        counts[subtract_mask] = counts[subtract_mask] - fit_func(bin_centers[subtract_mask])
        
        n += 1


    for param in params_list: 
        peaks_list.append(param[1])
        sigma_vals.append(param[2])


    peaks_mask = peaks_list <= ToT_max

    peaks_list = np.array(peaks_list)[peaks_mask]
    sigma_vals = np.array(sigma_vals)[peaks_mask]
    fit_funcs = np.array(fit_funcs)[peaks_mask]

    return peaks_list, sigma_vals, fit_funcs, ToT_max

        


def ToT_intercept_vals(tot, prints = False, dcounts_sigma = 1.75): 
    
    """
    
    Parameters
    -----------
    
    counts_init, bins_init: Initial bins and counts from the input ToT array 
    bin_centers_init: Initial bin centers fround from bins_init
    ToT_range_max: Maximum ToT value in the search range found from find_tot_range_max 
    counts, bins: Counts and bins found from taking a histogram of the input tot array using a maximum range defined by ToT_range_max
    pos_switch_vals, neg_switch_vals: Peaks and valleys found using find_deriv_0
    peak_locs, sig_vals, const_vals, fit_funcs, fit_funcs_base0: Peak locations and other important parameters as explained found from using find_peaks_fits 
    integral_vals: List to house the calculated integral values of each peak 
    x_begin, x_end: Left and right bounds of regions for each peak to calculate integral sum value 
    prob_vals: Calculated probabilites found by dividing each individual peak integral value by the sum of all peak integral values
    fit_func_vals_master: List to house the fitted values for each of the peaks but removing the baseline setting the constants to zero
    fit_func_vals_array: numpy array of fit_func_vals_master
    fit_func_tots: The total sum value of all fit functions 
    probs: Probabilities calculated through taking the ratio of each fit function to the total sum (May not be properly bayesian, could be worth discussing and 
           seeing if there are better methods.
    intersect_vals: Values of the intersection points (The points when one when n_photon peaks becomes more likely than the previous)
    
    
    
    
    """

    
    counts_init, bins_init = np.histogram(tot, bins = np.arange(tot.min(),tot.max() + 25, 25))

    bin_centers_init = (bins_init[:-1] + bins_init[1:]) / 2

    peaks, sig_vals, fit_funcs, ToT_range_max = find_peaks(bin_centers_init, counts_init)
    
    counts, bins = np.histogram(tot, bins = np.arange(tot.min(),ToT_range_max + 25, 25))
    
    counts_original = counts.copy()

    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    if prints:
        plt.bar(bin_centers, counts, width=np.diff(bins)[0], align='center', color='blue', edgecolor='royalblue')
        plt.xlabel("ToT [ns]", fontsize = 14)
        plt.ylabel("Counts", fontsize = 14)
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        plt.yscale('log')
        plt.show()
        
    
    
    integral_vals = []
    i = 0 
    ## Calculating the sums of each peak used to calculate the probability of each photon peak. Only values within +/- 2 sigma of each peak is used in this 
    ## calculation
    for func in fit_funcs:
        x_begin = peaks[i] - 2*sig_vals[i]
        x_end = peaks[i] + 2*sig_vals[i]
        integral_vals.append(Simpsons(func, x_begin, x_end))

        i += 1

    prob_vals = integral_vals/np.sum(integral_vals)*100
    
    
    if prints:
        plt.figure()
        
        plt.plot(bin_centers, counts_original, '.', color = 'black', markersize = 4)

        #colors = ['b', 'g', 'y', 'r','c','m', 'mediumseagreen', 'teal', 'peru', 'blueviolet' , 'deeppink']

        x_vals = np.linspace(0,ToT_range_max,int(1e6))

        i = 0 
        epsilon = 1e-10  
        for fit_func in fit_funcs:
            fill_vals = fit_func(x_vals) + epsilon
            plt.fill_between(x_vals, fill_vals, alpha = 0.3, label = "{} Photon Probability: {:.2f}%".format(i + 1,prob_vals[i]))
            
            i+=1
        plt.ylim(0 + 2e-1,counts_original.max()*1.50)
        plt.yscale('log')
        plt.xlabel("ToT (ns)")
        plt.ylabel("counts")
        #plt.xticks(fontsize = 12)
        #plt.yticks(fontsize = 12)
        plt.legend(loc="upper right", bbox_to_anchor=(1, 1)) 
        plt.tight_layout() 
        plt.show()
        
    fit_func_vals_master = []
    x_vals = np.linspace(0,ToT_range_max,int(1e6))
    for fit_func in fit_funcs: 
            fit_func_vals_master.append(fit_func(x_vals))

    fit_func_vals_array = np.array(fit_func_vals_master)
    
    fit_func_tots = np.sum(fit_func_vals_array, axis = 0)

    ## Calculating the probability of each n photon peak at every ToT value (different from prob_vals)
    probs = fit_func_vals_array/fit_func_tots

    if prints:
        plt.figure()
        i = 0
        for prob_list in probs:
            plt.plot(x_vals, prob_list, label="{} Photon probability".format(i + 1))
            i += 1
        plt.xlabel("ToT (ns)")
        plt.ylabel("probability")
        plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1)) 
        plt.tight_layout() 
        plt.show()
        
    intersect_vals = []

    ## Calculating the intersection points by finding the point when the probability curves are within a small enough difference from each other and both at a 
    ## reasonable value
    for i in range(len(probs)-1): 
        prob_i = probs[i]
        prob_diff_i = np.abs(probs[i+1] - probs[i])

        tol = 1e-3

        mask = (prob_diff_i <= tol) & (probs[i] > 0.1) & (probs[i+1] > 0.1)

        intersect_i = x_vals[mask]

        intersect_vals.append(np.mean(intersect_i))
        
    return intersect_vals


from scipy.ndimage import gaussian_filter1d
def find_phi0(phase, nbins = 500, prints = False): 
    
    """
    Calculates the phi0 value in the middle of the ion clearing gap to allow for a new set zero value for time and to drop values be
    
    
    Parameters
    -----------
    
    phase: input array of phase values
    nbins: number of bins in the histogram of phase 
    counts, bins: counts and bins of the histogram 
    bin_centers: center values of each bin
    counts_diff: derivative (difference) of the counts 
    counts_diff_smoohted: derivative of the counts smoothed with a 1D gaussian filter
    d_min, d_max: bin_center values corresponding to the maximum and minimum derivative values
    counts_left, counts_right: the counts values corresponding to the left and right edges of the histogram in the event where the ion clearing gap is on the edges
    phi_0: calculated position to set the 0 point of the phase and time values
    
    
    """
    
    if len(phase) < 100000: 
        nbins = 150
        
    phase_range = max(phase) - min(phase)
    
    counts, bins = np.histogram(phase, bins = nbins)
    
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    counts_diff = np.diff(counts)
    
    counts_diff_smoothed = gaussian_filter1d(counts_diff, sigma=2)

    d_min = bin_centers[np.argmin(counts_diff_smoothed)]
    d_max = bin_centers[np.argmax(counts_diff_smoothed)]
    
    ## If d_max - d_min > half the phase range, this mean the ion clearing gap must be on the edges
    if abs(d_max - d_min) > phase_range/2:
        counts_left = counts[0:int(len(counts)*0.1)]
        counts_right = counts[int(len(counts)*0.9):-1]
        
        if np.mean(counts_left) < np.mean(counts_right): 
            phi_0 = bin_centers[np.argmin(counts_left)]
        else:
            phi_0 = bin_centers[int(len(counts)*0.9) + np.argmin(counts_right)]
    ## If d_max - d_min < half the phase range, it the ion clearing gap shouldn't be on the edges and this can be calculated normally
    else:
        phi_0 = (d_min + d_max)/2
        
    
    if prints == True: 
        plt.plot(bin_centers[:-1], counts_diff_smoothed, color='mediumseagreen', label='Smoothed Derivative')
        plt.xlabel('Bin Centers')
        plt.ylabel('Smoothed Derivative of Counts')
        plt.legend()
        plt.show()
        
        plt.hist(phase, bins = nbins)
        plt.axvline(phi_0, color = "red")
        plt.show()
    
    
    return phi_0 

def phase_correct(df, prints = False):
    
    """
    
    Uses the returned value from find_phi0 to drop all data before the first new cycle and set time values as the difference from the 
    reference phi0 val 
    
    Parameters
    -----------
    
    phase_0: phi0 value found from find_phi0
    df['phase_diff']: the difference in phase between all phase values and phi0 
    first_pos_index: The first positive phase difference (i.e. start of a new cycle)
    first_neg_index: The first negative index in the event that the first phase diff val is positive 
    df_filtered: filtered dataframe after dropping data before first new cycle
    phase_offset: reference point to subtract time values from 
    zero_point: The new reference zero time value
    
    
    
    """

    if prints == True: 
        phase_0 = find_phi0(df['phase'], prints = True)
    else:
        phase_0 = find_phi0(df['phase'])
    
    df['phase_diff'] = df['phase'] - phase_0
    
    if phase_0 < 0: 
        first_pos_index = df[df['phase_diff'] >= 0].index[0]

        df_filtered = df.loc[(df.index >= first_pos_index)]
    else: 
        first_neg_index = df[df['phase_diff'] < 0].index[0]
        
        df_filtered = df.loc[(df.index >= first_neg_index)]
        
        first_pos_index = df_filtered[df_filtered['phase_diff'] >= 0].index[0]
        
        df_filtered = df_filtered.loc[(df_filtered.index >= first_pos_index)]
    
    phase_offset = df_filtered['phase_diff'].iloc[0]
    
    zero_point = df_filtered['t_ns'].iloc[0] - phase_offset 
    
    df_filtered['t_ns'] = df_filtered['t_ns'] - zero_point
    
    df_filtered['t'] = df_filtered['t'] - zero_point/1.5625

    df_filtered = df_filtered.reset_index(drop=True)
    
    return df_filtered

def custom_groupby_partition(partition):
    return partition.groupby(['x', 'y', 'time_bin']).agg({'n_photons': 'sum'}).reset_index()

def sf_convert(df, t0, ToT_thresholds, N = 5, manual_bin = False, tbin = None,frame_num=None, prints = False):  # time bin in microseconds
    
    """

    Function which handles the difference cases of binning and returns a dataframe with number of photons summed
    
    Parameters
    -----------
    
    df: input dataframe
    t0: calculated period of the cycle 
    ToT_thresholds: Calculated ToT threshold values between increasing n_photons vals 
    N: Number of cycles per bin 
    manual_bin: Boolean parameter that decides whether to use a manual bin value or a numerical integer multipler of the cycle period 
    tbin: manual time bin value 
    frame_num: parameter for if you want a set number of frames as opposed to setting bin width
    df_combined: Dataframe which sums the number of photons at a pixel and within a given time bin
    
    
    """
    
    if manual_bin: 
        time_bin = tbin*1000
    elif frame_num is not None:
        t_range = df['t_ns'].max() - df['t_ns'].min()
        bin_size_ns = t_range / frame_num
        time_bin = bin_size_ns
    else:
        time_bin = N*t0
    
    t_convert = time_bin / 1.5625  # Timestamp

    if prints:
        if manual_bin:
            print("Time Binning: {} microseconds".format(time_bin/1000))
        elif frame_num is not None: 
            print("Time Binning: {} microseconds".format(time_bin/1000))
        else:
            print("Time Binning: {} microseconds".format(time_bin/1000))
    
    if isinstance(df, dd.DataFrame): 
        df = df.set_index('t', shuffle_method='tasks').reset_index()
    #else: 
    #    df = df.sort_values(by='t')
    #    df = df.reset_index(drop=True)

    #df['time_bin'] = (df['t'] // t_convert).astype(int)
    df['time_bin'] = np.floor_divide(df['t'], t_convert).astype(int)
    #df['n_photons'] = df['ToT_sum'].apply(lambda x: assign_photon_value(x, ToT_thresholds))
    df['n_photons'] = np.searchsorted(ToT_thresholds, df['ToT_sum']) + 1
    
    if isinstance(df, dd.DataFrame):
        df_combined = df.map_partitions(custom_groupby_partition)
        df_combined = df_combined.compute()
        df_combined = df_combined.sort_values(by='time_bin')
    else: 
        if 'x' not in df.columns: 
            print("No centroid cols!")
            df_combined = df.groupby(['xc','yc','time_bin'], as_index = False).agg({'n_photons': 'sum'})
            
        else:
            df_combined = df.groupby(['x','y','time_bin'], as_index = False).agg({'n_photons': 'sum'}) 
        df_combined = df_combined.sort_values(by = 'time_bin') 
        
    
    df_combined.reset_index(drop=True)

    return df_combined

def sparse_format(df, N=5, tbin = None, frame_num = None,prints = False):

    """
    Main function through which all other functions are either directly or indirectly called and produces the final returned result.

    """

    if tbin is not None: 
        manual_bin = True 
    else:
        manual_bin = False 
    
    ## Using add_centroid_cols if needeed
    if 't_ns' not in df.columns: 
        if isinstance(df, dd.DataFrame): 
            df = tpx.add_centroid_cols(df)
        else:
            df = tpx.add_centroid_cols_pd(df)
    if prints:         
        ToT_thresholds = ToT_intercept_vals(df['ToT_sum'].to_numpy(dtype = np.uint32), prints = True)
        
        N_peaks = len(ToT_thresholds)
        
        print("There were {} located ToT peaks giving the following ToT threshold values for each n_photon val\n"
             .format(N_peaks + 1))
        
        print(ToT_thresholds)
        
        print()
        
    else:
        ToT_thresholds = ToT_intercept_vals(df['ToT_sum'].to_numpy(dtype = np.uint32))

    ## Phase Correcting 
    if manual_bin == False:
        t0 = find_best_t0(df, prints = prints)
        df = add_phase_col(df,t0, prints)
        df = phase_correct(df, prints)
        print(f"The final storage ring period was found to be {t0} nanoseconds!")
    else:
        t0 = tbin


    
    ## Sparsifying 
    if manual_bin == True: 
        df_sparse = sf_convert(df, t0 = t0, ToT_thresholds = ToT_thresholds, tbin = tbin, manual_bin = True)
    elif frame_num is not None:
        df_sparse = sf_convert(df, t0 = t0, ToT_thresholds = ToT_thresholds, frame_num = frame_num)
    else:
        df_sparse = sf_convert(df, N = N, t0 = t0, ToT_thresholds = ToT_thresholds)
    
    return df_sparse, t0
