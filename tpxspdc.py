"""

Stuff to work on:

- Error catching / making sure data is correct format for each function.
- Further optimize Pandas masking for speed.
- Convert to being usable with Dask.
- Better management of function parameters.
- Better and more efficient checking of duplicates from merge_asof and ts.combine.
- More helper/wrapper functions to do all of the main analysis in one line.
- Print reports to .pdf files.
- Clean up/vectorize detector acceptance functions.
- Add simulation code in.
- Detuning angle studies.

- probably a lot more on top of all that.

"""
import time
import pandas as pd
import dask.dataframe as dd
import dask.delayed
import numpy as np
import math
from fast_histogram import histogram1d, histogram2d
from scipy.optimize import curve_fit, minimize

import dask_histogram as dh
import matplotlib.pyplot as plt
import scipy.stats as stats
import gc as gc
from numba import njit, jit

NPIXELS = 514
PIXEL_LENGTH = 0.0055 # in cm

def add_centroid_cols(
    df: dd.DataFrame, gap: bool = True
) -> dd.DataFrame:
    """
    Calculates centroid positions to the nearest pixel and the timestamp in nanoseconds.

    Parameters
    ----------
    df : dd.DataFrame
        Input centroided dataframe
    gap : bool = True
        Determines whether to implement large gap correction by adding 2 empty pixels offsets

    Returns
    -------
    dd.DataFrame
        Original dataframe with new columns x, y, and t_ns added.
    """
    if gap:
        #df.loc[df['xc'] >= 255.5, 'xc'] += 2
        #df.loc[df['yc'] >= 255.5, 'yc'] += 2
        df['xc'] = df['xc'].mask(cond=df['xc'] >= 255.5, other= df['xc'] + 2)
        df['yc'] = df['yc'].mask(cond=df['yc'] >= 255.5, other= df['yc'] + 2)
    df["x"] = dd.DataFrame.round(df["xc"]).astype(np.uint16)
    df["y"] = dd.DataFrame.round(df["yc"]).astype(np.uint16)
    df["t_ns"] = df["t"] * 1.5625
    #df["t_ns"] = df["t"] / 4096 * 25
    return df



def channel(df, x_cen = 255, y_cen = 255, x_max = NPIXELS, y_max = NPIXELS, gap = True, prints = True):
    """
    Channels df into quadrants.
    Could be more efficient (instead of masking and .loc 4 times?)
    
    Look into:
    
    Might move this into tpx3awkward during .tpx3 processing...   
    """
    if gap:
        x_cen = x_cen + 1
        y_cen = y_cen + 1
    
    quad0_xmin = x_cen+1; quad0_xmax = x_max; quad0_ymin = 0; quad0_ymax = y_cen
    quad1_xmin = x_cen+1; quad1_xmax = y_max; quad1_ymin = y_cen+1; quad1_ymax = y_max
    quad2_xmin = 0; quad2_xmax = x_cen; quad2_ymin = y_cen+1; quad2_ymax = y_max
    quad3_xmin = 0; quad3_xmax = x_cen; quad3_ymin = 0; quad3_ymax = y_cen
    
    df['chan'] = pd.Series(0, index=df.index, dtype=np.uint8)    
    df['index'] = df.index # preserve a copy of the original index so it is not lost when later using pd.merge_asof
    
    
    """
    mask1 = (quad0_xmin <= df['x']) & (df['x'] <= quad0_xmax) & (quad0_ymin <= df['y']) & (df['y'] <= quad0_ymax)
    mask2 = (quad1_xmin <= df['x']) & (df['x'] <= quad1_xmax) & (quad1_ymin <= df['y']) & (df['y'] <= quad1_ymax)
    mask3 = (quad2_xmin <= df['x']) & (df['x'] <= quad2_xmax) & (quad2_ymin <= df['y']) & (df['y'] <= quad2_ymax)
    mask4 = (quad3_xmin <= df['x']) & (df['x'] <= quad3_xmax) & (quad3_ymin <= df['y']) & (df['y'] <= quad3_ymax)
    
    df.loc[mask1, 'chan'] = 1
    df.loc[mask2, 'chan'] = 2
    df.loc[mask3, 'chan'] = 3
    df.loc[mask4, 'chan'] = 4
    """

    conditions = [
        (quad0_xmin <= df['x']) & (df['x'] <= quad0_xmax) & (quad0_ymin <= df['y']) & (df['y'] <= quad0_ymax),
        (quad1_xmin <= df['x']) & (df['x'] <= quad1_xmax) & (quad1_ymin <= df['y']) & (df['y'] <= quad1_ymax),
        (quad2_xmin <= df['x']) & (df['x'] <= quad2_xmax) & (quad2_ymin <= df['y']) & (df['y'] <= quad2_ymax),
        (quad3_xmin <= df['x']) & (df['x'] <= quad3_xmax) & (quad3_ymin <= df['y']) & (df['y'] <= quad3_ymax),
    ]

    # Define the values to assign for each condition
    values = [1, 2, 3, 4]

    # Use numpy.select to perform the assignment
    df['chan'] = np.select(conditions, values, default=0)
    
    if prints: len_before = len(df)
    df = df.loc[df['chan'] > 0]
    if prints: print(f"Dropped rows: {len_before - len(df)}")
    
    return df

def channel_optimized(df, x_cen=256, y_cen=256, gap = True, prints = True):
    if gap:
        x_cen = x_cen + 1
        y_cen = y_cen + 1
        
    df['index'] = df.index # preserve a copy of the original index so it is not lost when later using pd.merge_asof
    
    # Apply integer division
    x_bit = (df['x'] // x_cen)
    y_bit = (df['y'] // y_cen)

    # Combine bits to get a two-bit number and map to quadrants
    quadrant_mapping = {0: 4, 1: 3, 2: 1, 3: 2}
    df['chan'] = (x_bit * 2 + y_bit).map(quadrant_mapping).fillna(0)
   
    df = df.loc[df['chan'] > 0]
    
    return df


def tot_filter_quad(df, optimal_tots=[350, 375, 425, 425], prints=True):
    """
    ToT filtering by chip cuts.
    """
    conditions = np.zeros(len(df), dtype=bool)
    
    for i, tot in enumerate(optimal_tots):
        conditions |= (df['ToT_sum'].values >= tot) & (df['chan'].values == i+1)
    
    if prints:
        len_before = len(df)
    
    df = df.loc[~conditions]
    
    if prints:
        print(f"Dropped rows after ToT filtering: {len_before - len(df)}")
    
    return df

def threshold(df, o):
    ret_val = o[df.x,df.y]
    return ret_val

def tot_filter_pixel(df, optimal_tots, prints = True):
    """
    ToT filtering with pixel cuts.
    """
    t1 = time.time()
    optimal_tots = optimal_tots[:,:,0]
    t2 =time.time()
    
    t3 = time.time()
    df['thresholds']=df.map_partitions(threshold,optimal_tots, align_dataframes=False, meta=('thresholds', 'int'))
    t4 = time.time()
    mask = df['ToT_sum'] <= df['thresholds']
    t5 = time.time()
    
    df = df.loc[mask] 
    t6 = time.time()
    #if prints: print(f"Dropped rows after ToT filtering: {len_before - len(df)}")
    
    #print(f"optimal tots setup: {(t2-t1):.1f}")
    #print(f"tresholds: {(t3-t2):.1f}")
    #print(f"map: {(t4-t3):.1f}")
    #print(f"mask: {(t5-t4):.1f}")
    #print(f"loc: {(t6-t5):.1f}")
    return df


def tot_filter_pixel_pandas(df, optimal_tots, prints = True):
    """
    ToT filtering with pixel cuts.
    """
    optimal_tots = optimal_tots[:,:,0]
    thresholds = optimal_tots[df['x'].values, df['y'].values]
    mask = df['ToT_sum'] <= thresholds

    if prints: len_before = len(df)
    df = df.loc[mask] 
    if prints: print(f"Dropped rows after ToT filtering: {len_before - len(df)}")
  
    return df

def merge_without_comms(df, df1, df2):
    df1 = df1.sort_values(by='t_1')
    df2 = df2.sort_values(by='t_2')
    result = pd.merge_asof(df1, df2, left_on='t_1', right_on='t_2', tolerance=500, direction='nearest', suffixes=('_1', '_2'))
    return result

def get_pairs(df, chans1 = [1,4], chans2 = [2,3], prints = True):
    """
    Gets pairs by minimal time difference.
    """
    #data is already sorted
    #df = df.sort_values(by='t')  # sorting by time, necessary for merge_asof
    
    t1 = time.time()
    df1 = df.loc[df['chan'].isin(chans1)]
    df2 = df.loc[df['chan'].isin(chans2)] 
    t2 = time.time()
    
    df1 = df1.rename(columns={'t': 't_1'})
    df2 = df2.rename(columns={'t': 't_2'})
    t3 = time.time()
   
    t4 = time.time()
    df_matched = df.map_partitions(merge_without_comms, df1, df2) # merge the dataframes by closest time in ns
    t5 = time.time()
   
    df_matched = df_matched.dropna()
    
    """
    if prints: len_before = len(df_matched)
     # drop any rows that have the same index more than once, keeping the one with earlier delta t
    if prints: print(f"Dropped rows after drop_duplicates on 'index_1': {len_before - len(df_matched)}")     
    """
   # df_matched = df_matched.drop_duplicates(subset='index_1', keep='first')
    
    #df_matched = df_matched.drop_duplicates(subset='index_2', keep='first')
    
    t6 = time.time()
    
    # print(f"loc: {(t2-t1):.1f}")
    # print(f"rename: {(t3-t2):.1f}")
    # print(f"compute: {(t4-t3):.1f}")
    # print(f"merge: {(t5-t4):.1f}")
    # print(f"drop: {(t6-t5):.1f}")
    # calculate interesting data
    return df_matched

def get_pairs_pandas(df, chans1 = [1,4], chans2 = [2,3], prints = True):
    """
    Gets pairs by minimal time difference.
    """
    df = df.sort_values(by='t_ns')  # sorting by time, necessary for merge_asof

    df1 = df.loc[df['chan'].isin(chans1)]
    df2 = df.loc[df['chan'].isin(chans2)] 
    df1 = df1.rename(columns={'t_ns': 't_ns_1'})
    df2 = df2.rename(columns={'t_ns': 't_ns_2'})

    df_matched = pd.merge_asof(df1, df2, left_on='t_ns_1', right_on='t_ns_2', direction='nearest', tolerance = 1000, suffixes=('_1', '_2')) # merge the dataframes by closest time in ns

    if prints: len_before = len(df_matched)
    df_matched = df_matched.dropna()
    if prints: print(f"Dropped rows after dropna: {len_before - len(df_matched)}")
    
    """
    if prints: len_before = len(df_matched)
    df_matched = df_matched.drop_duplicates(subset='index_1', keep='first') # drop any rows that have the same index more than once, keeping the one with earlier delta t
    if prints: print(f"Dropped rows after drop_duplicates on 'index_1': {len_before - len(df_matched)}")     
    """

    if prints: len_before = len(df_matched)
    df_matched = df_matched.drop_duplicates(subset='index_2', keep='first')
    if prints: print(f"Dropped rows after drop_duplicates on 'index_2': {len_before - len(df_matched)}")     

    # calculate interesting data
    return df_matched


def combine(df1, df2, prints = True):
    """
    Combines two dataframes into one, dropping duplicates. I think this approach needs to be re-thought and more robust.
    """
    df_combined = dd.concat([df1, df2]).compute()
    df_combined = df_combined.iloc[df_combined['delta_t'].abs().argsort()]
    if prints: len_before = len(df_combined)
    df_combined.drop_duplicates(subset='pair_id', keep='first', inplace=True)
    if prints: print(f"Dropped rows after drop_duplicates on 'pair_id': {len_before - len(df_combined)}")
    if prints: len_before = len(df_combined)
    df_combined.drop_duplicates(subset='index_1_t', keep='first', inplace=True)
    if prints: print(f"Dropped rows after drop_duplicates on 'index_1': {len_before - len(df_combined)}")
    if prints: len_before = len(df_combined)
    df_combined.drop_duplicates(subset='index_2_t', keep='first', inplace=True)
    if prints: print(f"Dropped rows after drop_duplicates on 'index_2': {len_before - len(df_combined)}")
    return df_combined#.compute()


def get_combine_pairs(df, scan_info, channels = [[[1,2], [3,4]], [[1,4], [2,3]]], prints = False):
    # Get 1 combo of pairs
    time_start = time.time()
    if prints: print("-> Getting 1/2 vs 3/4 pairs")
    
    df1 = get_pairs(df, channels[0][0], channels[0][1], prints=prints)#.compute()
    
    time_1 = time.time()
    df1 = calculate_pair_info_dask(df1, scan_info['x_cen'], scan_info['y_cen'], scan_info['Epump'], scan_info['dd'], scan_info['del_theta'], scan_info['theta'])
    time_2 = time.time()
    
    # Get other combo of pairs
    if prints: print("-> Getting 1/4 vs 2/3 pairs")
    df2 = get_pairs(df, channels[1][0], channels[1][1], prints=prints)#.compute()
    
    time_3 = time.time()
   
    df2 = calculate_pair_info_dask(df2, scan_info['x_cen'], scan_info['y_cen'], scan_info['Epump'], scan_info['dd'], scan_info['del_theta'], scan_info['theta'])
 
    time_4 = time.time()
    
    # Combine pairs and drop duplicates
    if prints: print("-> Combining pairs:")
    
    df_pairs = combine(df1, df2, prints=prints)
    time_5 = time.time()
    # print(f"get_pairs 1: {time_1 - time_start}")
    # print(f"pair info 1: {time_2 - time_1}")
    # print(f"get_pairs 2: {time_3 - time_2}")
    # print(f"pair info 2: {time_4 - time_3}")
    # print(f"combine : {time_5 - time_4}")
    if prints: print("Complete.")
    return df_pairs

@jit(nopython=True, cache=True)
def radius_to_energy(r, Epump = 15, dd = 69.749, del_theta = 0.022, theta = 11.576): 
    """
    Converts a distance of SPDC photon to an energy.
    """
    
    dd = dd * 1e4 / 55
    del_theta = math.radians(del_theta)
    theta = math.radians(theta)
             
    E = Epump/((np.arctan(r/dd))**2/(2*del_theta*np.sin(2*theta)) + 1)
    return E

def radius_to_energy_dask(r, Epump = 15, dd = 69.749, del_theta = 0.022, theta = 11.576): 
    """
    Converts a distance of SPDC photon to an energy.
    """
    
    dd = dd * 1e4 / 55
    del_theta = math.radians(del_theta)
    theta = math.radians(theta)
             
    E = Epump/((np.arctan(r/dd))**2/(2*del_theta*np.sin(2*theta)) + 1)
    return E

def closest_point_and_distance(x1, y1, x2, y2, x_c, y_c):
    """
    Calculates the closest point and the distance from (x_c, y_c) to line segment defined by (x1, y2) and (x2, y2)
    """
    # Vertical line segment
    if x1 == x2:
        y_closest = max(min(y_c, max(y1, y2)), min(y1, y2))
        x_closest = x1
    # Horizontal line segment
    elif y1 == y2:
        x_closest = max(min(x_c, max(x1, x2)), min(x1, x2))
        y_closest = y1
    else:
        # Slope and y-intercept of the line segment
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        m_perp = -1 / m
        b_perp = y_c - m_perp * x_c

        x_closest = (b_perp - b) / (m - m_perp)
        y_closest = m * x_closest + b

        # Check if intersection is within the segment bounds
        if not (min(x1, x2) <= x_closest <= max(x1, x2)):
            # Otherwise, find the closest endpoint
            dist_p1 = np.sqrt((x1 - x_c)**2 + (y1 - y_c)**2)
            dist_p2 = np.sqrt((x2 - x_c)**2 + (y2 - y_c)**2)
            x_closest, y_closest = (x1, y1) if dist_p1 < dist_p2 else (x2, y2)

    # Calculate the distance to the closest point
    dist_closest = np.sqrt((x_closest - x_c)**2 + (y_closest - y_c)**2)

    return dist_closest, x_closest, y_closest

def closest_point_and_distance_dask(df, x_c, y_c):
    dist_closest, x_closest, y_closest = np.vectorize(closest_point_and_distance)(df.xc_1, df.yc_1, df.xc_2, df.yc_2, x_c, y_c)
    return {'dp': dist_closest, 'xp': x_closest, 'yp': y_closest}

def energy_to_alpha(E, Epump = 15, del_theta = 0.022, theta = 11.576):
    """
    Converts an energy in keV to an SPDC angle (alpha) in radians.
    """
    del_theta = math.radians(del_theta)
    theta = math.radians(theta)
    return np.sqrt( np.sin(2*theta)*2*del_theta )* np.sqrt( Epump - E ) / np.sqrt( E )

def create_tuple(df):
    return list(zip(df.index_1.astype(int), df.index_2.astype(int), df.t_ns_1.astype(float), df.t_ns_2.astype(float)))

def create_index_1_t(df):
    return list(zip(df.index_1.astype(int), df.t_ns_1.astype(float)))

def create_index_2_t(df):
    return list(zip(df.index_2.astype(int), df.t_ns_2.astype(float)))

def calculate_pair_info_dask(df, x_cen = 255, y_cen = 255, Epump = 15, dd = 69.749, del_theta = 0.022, theta = 11.576):
    """
    Adds a bunch of information about the pairs to the dataframe.
    """
    df['delta_t'] = df['t_ns_1'] - df['t_ns_2']
    df['x_cen'] = (df['xc_1']+df['xc_2'])/2
    df['y_cen'] = (df['yc_1']+df['yc_2'])/2
    df['r_1'] = np.sqrt((df['xc_1']-x_cen)**2+(df['yc_1']-y_cen)**2)
    df['r_2'] = np.sqrt((df['xc_2']-x_cen)**2+(df['yc_2']-y_cen)**2)
    df['E_1'] = radius_to_energy_dask(df['r_1'].values, Epump, dd, del_theta, theta)
    df['E_2'] = radius_to_energy_dask(df['r_2'].values, Epump, dd, del_theta, theta)
    df['E_tot'] = df['E_1'] + df['E_2']
    df['dr'] = np.sqrt((df['x_cen']-x_cen)**2+(df['y_cen']-y_cen)**2)
    df['dist'] = np.sqrt((df['xc_1']-df['xc_2'])**2 + (df['yc_1']-df['yc_2'])**2)
    
    pair_ids = df.map_partitions(create_tuple)
    
    df['pair_id'] = pair_ids
    index_1_timestamped = df.map_partitions(create_index_1_t)
    index_2_timestamped = df.map_partitions(create_index_2_t)
    
    df['index_1_t'] = index_1_timestamped
    df['index_2_t'] = index_2_timestamped
    
    result = df.map_partitions(closest_point_and_distance_dask, x_c=x_cen, y_c=y_cen, meta={'dp': 'float64', 'xp': 'float64', 'yp': 'float64'})

    df['dp'] = result['dp']
    df['xp'] = result['xp']
    df['yp'] = result['yp']
    
    
    return df

def calculate_pair_info_pandas(df, x_cen = 255, y_cen = 255, Epump = 15, dd = 69.749, del_theta = 0.022, theta = 11.576):
    """
    Adds a bunch of information about the pairs to the dataframe.
    """
    df['delta_t'] = df['t_ns_1'] - df['t_ns_2']
    df['x_cen'] = (df['xc_1']+df['xc_2'])/2
    df['y_cen'] = (df['yc_1']+df['yc_2'])/2
    df['r_1'] = np.sqrt((df['xc_1']-x_cen)**2+(df['yc_1']-y_cen)**2)
    df['r_2'] = np.sqrt((df['xc_2']-x_cen)**2+(df['yc_2']-y_cen)**2)
    df['E_1'] = radius_to_energy(df['r_1'].values, Epump, dd, del_theta, theta)
    df['E_2'] = radius_to_energy(df['r_2'].values, Epump, dd, del_theta, theta)
    df['E_tot'] = df['E_1'] + df['E_2']
    df['dr'] = np.sqrt((df['x_cen']-x_cen)**2+(df['y_cen']-y_cen)**2)
    df['dist'] = np.sqrt((df['xc_1']-df['xc_2'])**2 + (df['yc_1']-df['yc_2'])**2)
    
    df['pair_id'] = df.apply(lambda row: tuple(sorted((row['index_1'], row['index_2']))), axis=1)
    #df['index_1_t'] = list(zip(df.index_1.astype(int), df.t_ns_1.astype(float)))
    #df['index_2_t'] = list(zip(df.index_2.astype(int), df.t_ns_2.astype(float)))
    
    dp,xp,yp = np.vectorize(closest_point_and_distance)(df['xc_1'], df['yc_1'], df['xc_2'], df['yc_2'], x_cen, y_cen)
    
    df['dp'] = pd.Series(dp, index = df.index)
    df['xp'] = pd.Series(xp, index = df.index)
    df['yp'] = pd.Series(yp, index = df.index)
    
    return df

def calculate_reduced_pair_info(df, x_cen = 255, y_cen = 255, Epump = 15, dd = 69.749, del_theta = 0.022, theta = 11.576):
    """
    Add only info needed for dp_search to dataframe.
    """ 
    df['delta_t'] = df['t_ns_1'] - df['t_ns_2'] 
    df['r_1'] = np.sqrt((df['xc_1']-x_cen)**2+(df['yc_1']-y_cen)**2)
    df['r_2'] = np.sqrt((df['xc_2']-x_cen)**2+(df['yc_2']-y_cen)**2)
    df['E_1'] = radius_to_energy(df['r_1'].values, Epump, dd, del_theta, theta)
    df['E_2'] = radius_to_energy(df['r_2'].values, Epump, dd, del_theta, theta)
    df['E_tot'] = df['E_1'] + df['E_2']
    dp,xp,yp = np.vectorize(closest_point_and_distance)(df['xc_1'], df['yc_1'], df['xc_2'], df['yc_2'], x_cen, y_cen)
    df['dp'] = pd.Series(dp, index = df.index)
    
    return df

def calculate_detuning_info(df, x_cen = 255, y_cen = 255, Epump = 15, dd = 69.749, del_theta = 0.022, theta = 11.576):
    df['alpha_1'] = np.arctan((df['r_1'] * PIXEL_LENGTH) / dd)
    df['alpha_2'] = np.arctan((df['r_2'] * PIXEL_LENGTH) / dd)
    df['del_theta'] = df['alpha_1']*df['alpha_2']/(2*np.sin(2*np.radians(theta)))
    df['k'] = np.radians(del_theta) / df['del_theta']
    df['theta_1'] = np.arctan2((df['yc_1'] - y_cen), (df['xc_1'] - x_cen))
    df['theta_2'] = np.arctan2((df['yc_2'] - y_cen), (df['xc_2'] - x_cen))
    df['r_1_p'] = (10000*dd/55) * np.tan(df['k'] * df['alpha_1'])
    df['r_2_p'] = (10000*dd/55) * np.tan(df['k'] * df['alpha_2'])
    df['xc_1_p'] = (df['r_1_p'] * np.cos(df['theta_1'])) + x_cen
    df['yc_1_p'] = (df['r_1_p'] * np.sin(df['theta_1'])) + y_cen
    df['xc_2_p'] = (df['r_2_p'] * np.cos(df['theta_2'])) + x_cen
    df['yc_2_p'] = (df['r_2_p'] * np.sin(df['theta_2'])) + y_cen
    
    return df


@njit
def gauss(x, A, t0, sigma, d):
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
    
    y = abs(A)*np.exp(-(x-t0)**2/(2*abs(sigma)**2)) + d
    return y


@njit
def laplace(x, A, x0, b, d):
    y = abs(A)*np.exp(-np.abs(x-x0)/abs(b)) + d
    return y

@njit
def gauss_modified(x, A, x0, sigma, const, A2, x1, sigma2):
    """
    Modified Gaussian function with two Gaussian components
    """
    
    # First Gaussian component
    Gauss1 = A * np.exp(-(x - x0)**2 / (2 * sigma**2))
    
    # Second Gaussian component
    Gauss2 = A2 * np.exp(-(x - x1)**2 / (2 * sigma2**2))
    
    # Constant offset
    C = const * np.ones_like(x)
    
    # Combine the components and offset
    return Gauss1 + Gauss2 + C


def laplace_fit_series(series, p0_guess = None, bin_size = None, model_bin_size = None, srange = None, print_p0_guess = False):
    if srange == None:
        series_max = np.max(series)
        series_min = np.min(series)
    else:
        series_max = srange[1]
        series_min = srange[0]
        
    if bin_size == None:
        if srange == None:
            slen = len(series)
        else: 
            slen = len(np.concatenate((series[series >= series_min], series[series <= series_max])))
        num_bins = int(2 * (len(series) ** (1/3)))
        bin_size = (series_max-series_min)/num_bins
        
    if model_bin_size == None:
        model_bin_size = bin_size
        
    # h_counts, h_edges = np.histogram(series, bins = np.arange(series_min, series_max + bin_size, bin_size), range = [series_min, series_max])
    h_counts, h_edges = histogram1d_fast(series, bins = np.arange(series_min, series_max + bin_size, bin_size), range = [series_min, series_max])
    
    h_centers = (h_edges[1:] + h_edges[:-1])/2
    
    max_count = np.max(h_counts)
    min_count = np.min(h_counts)
    mean_count = np.mean(h_counts)
    peak_loc = h_centers[np.argmax(h_counts)]
    stdev = np.std(series)
    
    if p0_guess == None:
        p0_guess = [max_count - min_count, peak_loc, stdev, mean_count/2]
    else:
        for i, p0_guess_val in enumerate(p0_guess):
            if p0_guess_val == None:
                if i == 0:
                    p0_guess[i] = max_count - min_count
                elif i == 1: 
                    p0_guess[i] = peak_loc
                elif i == 2:
                    p0_guess[i] = stdev
                elif i == 3:
                    p0_guess[i] = mean_count/2
                
    if print_p0_guess:
        print(p0_guess)
        
    try:
        popt, pcov = curve_fit(gauss, h_centers, h_counts, p0 = p0_guess, maxfev = 5000)
        A = abs(popt[0])
        x0 = popt[1]
        sigma = np.sqrt(2)*abs(popt[2])
        const = popt[3]       
        error_params = np.sqrt(np.diag(pcov))     
    except:

        print("--> NOTICE!")
        print("--> Optimal parameters couldn't be found!")
        print("--> Make sure to scrutinize the results carefully.")
        popt = [0.01, 0, 1, 1]
        error_params = [0, 0, 0, 0]

    A = abs(popt[0])
    x0 = popt[1]
    sigma = np.sqrt(2)*abs(popt[2])
    const = popt[3]        
        
    bin_width = (max(h_centers) - min(h_centers))/len(h_centers)
    
    A_err = error_params[0]; x0_err = error_params[1]; sigma_err = error_params[2]; const_err = error_params[3]
    N = A*sigma*np.sqrt(2*np.pi)/bin_width
    N_err = N*np.sqrt((A_err/A)**2 + (sigma_err/sigma)**2)
    SN = A/const
    SN_err = SN*np.sqrt((A_err/A)**2 + (const_err/const)**2)

    model_x_vals = np.arange(min(h_centers), max(h_centers), model_bin_size)
    model_y_vals = laplace(model_x_vals, *popt)
    
    return {
        'h_edges': h_edges,
        'h_centers': h_centers,
        'h_counts': h_counts,
        'model_x_vals': model_x_vals,
        'model_y_vals': model_y_vals,
        'p0_guess': p0_guess,
        'A': A,
        'N': N,
        'x0': x0,
        'sigma': sigma,
        'const': const,
        'SN': SN,
        'A_err': A_err,
        'N_err': N_err,
        'x0_err': x0_err,
        'sigma_err': sigma_err,
        'const_err': const_err,
        'SN_err': SN_err,
        'bin_size': bin_size
    }

def gauss_fit_series(series, p0_guess = None, bin_size = None, model_bin_size = None, srange = None, print_p0_guess = False):
    if srange == None:
        series_max = np.max(series)
        series_min = np.min(series)
    else:
        series_max = srange[1]
        series_min = srange[0]
        
    if bin_size == None:
        if srange == None:
            slen = len(series)
        else: 
            slen = len(np.concatenate((series[series >= series_min], series[series <= series_max])))
        num_bins = int(2 * (len(series) ** (1/3)))
        bin_size = (series_max-series_min)/num_bins
        
    if model_bin_size == None:
        model_bin_size = bin_size
        
    # h_counts, h_edges = np.histogram(series, bins = np.arange(series_min, series_max + bin_size, bin_size), range = [series_min, series_max])
    h_counts, h_edges = histogram1d_fast(series, bins = np.arange(series_min, series_max + bin_size, bin_size), range = [series_min, series_max])
    
    h_centers = (h_edges[1:] + h_edges[:-1])/2
    
    max_count = np.max(h_counts)
    min_count = np.min(h_counts)
    mean_count = np.mean(h_counts)
    peak_loc = h_centers[np.argmax(h_counts)]
    stdev = np.std(series)
    
    if p0_guess == None:
        p0_guess = [max_count - min_count, peak_loc, stdev, mean_count/2]
    else:
        for i, p0_guess_val in enumerate(p0_guess):
            if p0_guess_val == None:
                if i == 0:
                    p0_guess[i] = max_count - min_count
                elif i == 1: 
                    p0_guess[i] = peak_loc
                elif i == 2:
                    p0_guess[i] = stdev
                elif i == 3:
                    p0_guess[i] = mean_count/2
                
    if print_p0_guess:
        print(p0_guess)
        
    try:
        popt, pcov = curve_fit(gauss, h_centers, h_counts, p0 = p0_guess, maxfev = 5000)
        A = abs(popt[0])
        x0 = popt[1]
        sigma = abs(popt[2])
        const = popt[3]       
        error_params = np.sqrt(np.diag(pcov))     
    except:

        print("--> NOTICE!")
        print("--> Optimal parameters couldn't be found!")
        print("--> Make sure to scrutinize the results carefully.")
        popt = [0.01, 0, 1, 1]
        error_params = [0, 0, 0, 0]

    A = abs(popt[0])
    x0 = popt[1]
    sigma = abs(popt[2])
    const = popt[3]        
        
    bin_width = (max(h_centers) - min(h_centers))/len(h_centers)
    
    A_err = error_params[0]; x0_err = error_params[1]; sigma_err = error_params[2]; const_err = error_params[3]
    N = A*sigma*np.sqrt(2*np.pi)/bin_width
    N_err = N*np.sqrt((A_err/A)**2 + (sigma_err/sigma)**2)
    SN = A/const
    SN_err = SN*np.sqrt((A_err/A)**2 + (const_err/const)**2)

    model_x_vals = np.arange(min(h_centers), max(h_centers), model_bin_size)
    model_y_vals = gauss(model_x_vals, *popt)
    
    return {
        'h_edges': h_edges,
        'h_centers': h_centers,
        'h_counts': h_counts,
        'model_x_vals': model_x_vals,
        'model_y_vals': model_y_vals,
        'p0_guess': p0_guess,
        'A': A,
        'N': N,
        'x0': x0,
        'sigma': sigma,
        'const': const,
        'SN': SN,
        'A_err': A_err,
        'N_err': N_err,
        'x0_err': x0_err,
        'sigma_err': sigma_err,
        'const_err': const_err,
        'SN_err': SN_err,
        'bin_size': bin_size
    }


def dt_filter(df, selection_range = 2, p0_guess = None, bin_size = 1.5625*4, model_bin_size = None, srange = [-250, 250], prints = True, keys_to_print = ['p0_guess', 'A', 'N', 'x0', 'sigma', 'const', 'SN', 'bin_size'], plot = True, print_p0_guess = False):
    
    delta_t = df["delta_t"]
    series_range = delta_t[(delta_t >= srange[0]) & (delta_t <= srange[1])]
    stdev = np.sqrt(np.std(series_range))
    if p0_guess == None:
        p0_guess = [None, 0, stdev, None]
    
    dt_stats = gauss_fit_series(df['delta_t'], p0_guess = p0_guess, bin_size = bin_size, model_bin_size = model_bin_size, srange = srange, print_p0_guess = print_p0_guess)
    df_dt = df[df['delta_t'].abs() < dt_stats['sigma']*selection_range]
    
    t0 = dt_stats['x0']
    tsigma = dt_stats['sigma']
    
    df_dE = df[(df['delta_t'] >= t0 - tsigma*selection_range) & (df['delta_t'] <= t0 + tsigma*selection_range)]
    
    if prints:
        for key in keys_to_print:
            if key in dt_stats:  # Check if key exists in dictionary to avoid KeyError
                print(f'{key}: {dt_stats[key]}')
            else:
                print(f'{key}: Key not found in dictionary')
                
    if plot:
        plt.figure()
        plt.plot(dt_stats['h_centers'], dt_stats['h_counts'], '.', color='blue', markersize=3.0)
        plt.plot(dt_stats['model_x_vals'], dt_stats['model_y_vals'], '--', color='orange', linewidth=2.0, label='N = {:.0f}, SN = {:.2f}'.format(dt_stats['N'], dt_stats['SN']))
        plt.xlabel("$\Delta t$ (ns)", fontsize=10)
        plt.ylabel("Counts", fontsize=10)
        plt.tick_params(axis='both', which='major', labelsize=8)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.show()
        plt.close()       
    
    return df_dt, dt_stats


def modified_gauss_fit_series(series, p0_guess = None, bin_size = None, model_bin_size = None, srange = None, print_p0_guess = False):
    """
    Performs a fit two a double Gauss curve and returns fit parameters.
    
    This function has unstable behavior, in sometimes the first Guass will the top peak, but other times the second Gauss will fit the top peak, with only a small change in variables.
    Need to look into way to force the top peak to be consistent.
    """
    if srange == None:
        series_max = np.max(series)
        series_min = np.min(series)
    else:
        series_max = srange[1]
        series_min = srange[0]
    
    if bin_size == None:
        num_bins = int(2 * (len(series) ** (1/3)))
        bin_size = (series_max-series_min)/num_bins
        
    if model_bin_size == None:
        model_bin_size = bin_size
        
    # h_counts, h_edges = np.histogram(series, bins = np.arange(series_min, series_max + bin_size, bin_size))
    h_counts, h_edges = histogram1d_fast(series, bins = np.arange(series_min, series_max + bin_size, bin_size), range = [series_min, series_max])
    #h_counts, h_edges = dh.histogram(series, bins = np.arange(series_min, series_max + bin_size, bin_size), range = [series_min, series_max]).compute()
    
    h_centers = (h_edges[1:] + h_edges[:-1])/2
    
    max_count = np.max(h_counts)
    min_count = np.min(h_counts)
    peak_loc = h_centers[np.argmax(h_counts)]
    stdev = np.std(series)
    
    if p0_guess == None:
        p0_guess = [max_count - min_count, peak_loc, stdev, min_count*2, (max_count - min_count)/10, peak_loc, 5*stdev]
        
    if print_p0_guess:
        print(p0_guess)

    try:    
        popt, pcov = curve_fit(gauss_modified, h_centers, h_counts, p0 = p0_guess, maxfev = 5000)   
        error_params = np.sqrt(np.diag(pcov))      
    except:
        print("--> NOTICE!")
        print("--> Optimal parameters couldn't be found!")
        print("--> Make sure to scrutinize the results carefully.")
        popt = [0.01, 0, 1, 1, 0.001, 0, 10]
        error_params = [0, 0, 0, 0, 0, 0, 0]
        
    if popt[0] < popt[4]:
        A = abs(popt[4])
        t0 = popt[5]
        sigma = abs(popt[6])
        A2 = abs(popt[0])
        t1 = popt[1]
        sigma2 = abs(popt[2])
        const = popt[3]
    else:     
        A = abs(popt[0])
        t0 = popt[1]
        sigma = abs(popt[2])
        A2 = abs(popt[4])
        t1 = popt[5]
        sigma2 = abs(popt[6])  
        const = popt[3]      

        
    A_err = error_params[0]; t0_err = error_params[1]; sigma_err = error_params[2]; const_err = error_params[3]
    A2_err = error_params[4]; t1_err = error_params[5]; sigma2_err = error_params[6]
    N = A2*sigma2*np.sqrt(2*np.pi)/bin_size
    N_err = N*np.sqrt((A2_err/A2)**2 + (sigma2_err/sigma2)**2)

    model_x_vals = np.arange(min(h_centers), max(h_centers), model_bin_size)
    model_y_vals = gauss_modified(model_x_vals, *popt)
    
    return {
        'h_edges': h_edges,
        'h_centers': h_centers,
        'h_counts': h_counts,
        'model_x_vals': model_x_vals,
        'model_y_vals': model_y_vals,
        'p0_guess': p0_guess,
        'A1': A,
        'x0_1': t0,
        'sigma1': sigma,
        'A1_err': A_err,
        'x0_1_err': t0_err,
        'sigma1_err': sigma_err,       
        'A2': A2,
        'x0_2': t1,
        'sigma2': sigma2,
        'A2_err': A2_err,
        'x0_2_err': t1_err,
        'sigma2_err': sigma2_err,        
        'const': const,
        'const_err': const_err,      
        'N': N,
        'N_err': N_err,
        'bin_size': bin_size
    }


def dE_filter(df, selection_range = 2, p0_guess = [1200, 15, .5, 0, 200, 15, 3], bin_size = 0.25,  model_bin_size = 0.025, srange = None, prints = True, keys_to_print = ['p0_guess', 'A1', 'N', 'x0_1', 'sigma1', 'A2', 'x0_2', 'sigma2', 'const', 'bin_size'], plot = True, print_p0_guess = False):
    dE_stats = modified_gauss_fit_series(df['E_tot'], p0_guess = p0_guess, bin_size = bin_size, model_bin_size = model_bin_size, srange = None, print_p0_guess = print_p0_guess)
    
    E0 = dE_stats['x0_1']
    Esigma = dE_stats['sigma1']
    
    df_dE = df[(df['E_tot'] >= E0 - Esigma*selection_range) & (df['E_tot'] <= E0 + Esigma*selection_range)]

    if prints:
        for key in keys_to_print:
            if key in dE_stats:  # Check if key exists in dictionary to avoid KeyError
                print(f'{key}: {dE_stats[key]}')
            else:
                print(f'{key}: Key not found in dictionary')
                
    if plot:
        plt.figure()
        plt.plot(dE_stats['h_centers'], dE_stats['h_counts'], '.', color='blue', markersize=3.0)
        plt.plot(dE_stats['model_x_vals'], dE_stats['model_y_vals'], '--', color='orange', linewidth=2.0)
        plt.xlabel("E$_1$ + E$_2$ (keV)", fontsize=10)
        plt.ylabel("counts", fontsize=10)
        plt.tick_params(axis='both', which='major', labelsize=8)
        plt.tight_layout()
        plt.show()
        plt.close()       
    
    return df_dE, dE_stats


def dp_filter(df, selection_range = 4):
    return df[df['dp'] <= selection_range]

def dp_search_coord(df, scan_info, coord, dt_selection_range, Etot_selection_range):
    #make this return N_list and SN_list to make plots in next, serial step
    print("---> Working on ({},{}) [{},{}]".format(coord[0], coord[1], coord[0]-scan_info['x_cen'], coord[1]-scan_info['y_cen']))
    df_pairs = calculate_reduced_pair_info(df, x_cen = coord[0], y_cen = coord[1], Epump = scan_info['Epump'], dd = scan_info['dd'], del_theta = scan_info['del_theta'], theta = scan_info['theta'])
    #df_pairs = calculate_pair_info_pandas(df, x_cen = coord[0], y_cen = coord[1], Epump = scan_info['Epump'], dd = scan_info['dd'], del_theta = scan_info['del_theta'], theta = scan_info['theta'])
    #df_pairs = df
    
    df_dt, df_dt_stats = dt_filter(df_pairs, dt_selection_range, prints = False, plot = False)
    df_dt_Etot, df_dt_Etot_stats = dE_filter(df_dt, Etot_selection_range, p0_guess = [1200, 7.5, .5, 0, 200, 7.5, 3], prints = False, plot = False)
    E0 = df_dt_Etot_stats['x0_1']
    Esigma = df_dt_Etot_stats['sigma1']
    df_Etot = df_pairs[(df_pairs['E_tot'] >= E0 - Esigma*Etot_selection_range) & (df_pairs['E_tot'] <= E0 + Esigma*Etot_selection_range)] # select all pairs

    dp_range = np.array(range(1, 100))/4
    A_list = []
    N_list = []
    SN_list = []
    const_list = []


    for dp_selection_range in dp_range:

        df_Etot_dp = dp_filter(df_Etot, dp_selection_range)

        df_Etot_dp_dt, df_Etot_dp_dt_stats = dt_filter(df_Etot_dp, dt_selection_range, prints = False, plot = False)
        A_list.append(df_Etot_dp_dt_stats['A'])
        N_list.append(df_Etot_dp_dt_stats['N'])
        SN_list.append(df_Etot_dp_dt_stats['SN'])
        const_list.append(df_Etot_dp_dt_stats['const'])

    
    metric = np.multiply(A_list, np.sqrt(SN_list))
    metric_max = np.argmax(metric)
    
  
    return { "best_dp": dp_range[metric_max], "best_metric": metric[metric_max], "N_list": N_list, "SN_list": SN_list, "dp_range": dp_range, "metric_max": metric_max, "metric" : metric, "this_coord": coord}



def dp_search_dask(client, df, scan_info, coords, dt_selection_range = 2, Etot_selection_range=2):
    best_dp = []
    best_metric = []
    
    #reduce dataframe that we send to workers to include only necessary data 
    df_reduced = df[['t_ns_1', 't_ns_2','xc_1', 'xc_2', 'yc_1', 'yc_2', 'E_1', 'E_2']].copy(deep=True)
    print(df_reduced)
    #main loop only scales with number of processes - not threads - so we pack search coords into batches of size #number-of-workers
    batchsize=len(client.scheduler_info()['workers'])
    
    print(f"Scanning {len(coords)}-coord pairs")
    
    #broadcast pandas dataframe to workers to avoid large taskgraph warning
    df_future = client.scatter(df_reduced, broadcast=True)
    
    for batch in range(0,len(coords),batchsize):
        
        batch_start = batch
        batch_stop = min((batch+batchsize),len(coords))
        coords_batch = coords[batch_start:batch_stop]
        futures_batch = []
        
        print(f"Batch: {batch_start} to {batch_stop}")
        print(f"coords: {coords_batch}")
        for coord in coords_batch:
            print(f"submitting {coord}")
            future = client.submit(dp_search_coord, df_future, scan_info, coord, dt_selection_range, Etot_selection_range, pure=False)
            futures_batch.append(future)
        
        print(f"futures: {futures_batch}")
        print(f"gathering results")
        results_batch = client.gather(futures_batch)
        print(f"futures after gather: {futures_batch}")
#        print(f"check results: {results_batch}:")
        for result in results_batch:
            best_dp.append(result["best_dp"])
            best_metric.append(result["best_metric"])
            metric_max = result["metric_max"]
            metric = result["metric"]
            this_coord = result["this_coord"]
            N_list = result["N_list"]
            SN_list = result["SN_list"]
            dp_range = result["dp_range"]

            fig, (ax1, ax3) = plt.subplots(nrows=1, ncols=2, figsize=(10,4))  # Create a 1x2 subplot grid


            fig.suptitle('({},{}) [{},{}]: dp={}, M={:.1f}, N={:.1f}, SN={:.1f}\ndp = {}, M = {:.1f}, N = {:.1f}, SN = {:.1f}'.format(this_coord[0], this_coord[1], this_coord[0]-scan_info['x_cen'], this_coord[1]-scan_info['y_cen'], result["best_dp"], result["best_metric"], N_list[metric_max], SN_list[metric_max], dp_range[15], metric[15], N_list[15], SN_list[15]), fontsize=14)
            ax1.plot(dp_range, N_list, 'b-') 
            ax1.set_xlabel('dp selection value') 
            ax1.set_ylabel('N', color='b')  
            ax1.tick_params('y', colors='b')

            # Create a twin axis for SN_list
            ax2 = ax1.twinx()
            ax2.tick_params('y', colors='r')  # Set the color of the tick labels to match the line color
            # Plot SN_list against right y-axis
            ax2.plot(dp_range, SN_list, 'r-')  # 'r-' means red color, solid line
            ax2.set_ylabel('SNR', color='r')  # Set the y-axis label and color to match the line color

            ax3.plot(dp_range, metric, 'b')
            ax3.set_xlabel('dp selection value')
            ax3.set_ylabel('metric')

            plt.tight_layout()

            plt.show()


    return {"coords": coords, "best_dp": best_dp, "best_metric": best_metric}

def dp_search(df, scan_info, coords, dt_selection_range = 2, Etot_selection_range= 2):
    best_dp = []
    best_metric = []

    N_lists = []
    SN_lists = []
    const_lists = []

    for coord in coords:

        print("---> Working on ({},{}) [{},{}]".format(coord[0], coord[1], coord[0]-scan_info['x_cen'], coord[1]-scan_info['y_cen']))
        df_pairs = calculate_pair_info_pandas(df, x_cen = coord[0], y_cen = coord[1], Epump = scan_info['Epump'], dd = scan_info['dd'], del_theta = scan_info['del_theta'], theta = scan_info['theta'])

        df_dt, df_dt_stats = dt_filter(df_pairs, dt_selection_range, prints = False, plot = False)
        df_dt_Etot, df_dt_Etot_stats = dE_filter(df_dt, Etot_selection_range, prints = False, plot = False)
        E0 = df_dt_Etot_stats['x0_1']
        Esigma = df_dt_Etot_stats['sigma1']
        df_Etot = df_pairs[(df_pairs['E_tot'] >= E0 - Esigma*Etot_selection_range) & (df_pairs['E_tot'] <= E0 + Esigma*Etot_selection_range)] # select all pairs

        dp_range = np.array(range(1, 100))/4
        A_list = []
        N_list = []
        SN_list = []
        const_list = []
        
        
        for dp_selection_range in dp_range:

            df_Etot_dp = dp_filter(df_Etot, dp_selection_range)

            df_Etot_dp_dt, df_Etot_dp_dt_stats = dt_filter(df_Etot_dp, dt_selection_range, prints = False, plot = False)
            A_list.append(df_Etot_dp_dt_stats['A'])
            N_list.append(df_Etot_dp_dt_stats['N'])
            SN_list.append(df_Etot_dp_dt_stats['SN'])
            const_list.append(df_Etot_dp_dt_stats['const'])

        N_lists.append(N_list)
        SN_lists.append(SN_list)
        const_lists.append(const_list)

        fig, (ax1, ax3) = plt.subplots(nrows=1, ncols=2, figsize=(10,4))  # Create a 1x2 subplot grid

        metric = np.multiply(A_list, np.sqrt(SN_list))
        metric_max = np.argmax(metric)

        best_dp.append(dp_range[metric_max])
        best_metric.append(metric[metric_max])

        fig.suptitle('({},{}) [{},{}]: dp={}, M={:.1f}, N={:.1f}, SN={:.1f}\ndp = {}, M = {:.1f}, N = {:.1f}, SN = {:.1f}'.format(coord[0], coord[1], coord[0]-scan_info['x_cen'], coord[1]-scan_info['y_cen'], dp_range[metric_max], metric[metric_max], N_list[metric_max], SN_list[metric_max], dp_range[15], metric[15], N_list[15], SN_list[15]), fontsize=14)
        ax1.plot(dp_range, N_list, 'b-') 
        ax1.set_xlabel('dp selection value') 
        ax1.set_ylabel('N', color='b')  
        ax1.tick_params('y', colors='b')

        # Create a twin axis for SN_list
        ax2 = ax1.twinx()
        ax2.tick_params('y', colors='r')  # Set the color of the tick labels to match the line color
        # Plot SN_list against right y-axis
        ax2.plot(dp_range, SN_list, 'r-')  # 'r-' means red color, solid line
        ax2.set_ylabel('SNR', color='r')  # Set the y-axis label and color to match the line color

        ax3.plot(dp_range, metric, 'b')
        ax3.set_xlabel('dp selection value')
        ax3.set_ylabel('metric')

        plt.tight_layout()

        plt.show()
        
    df_pairs = calculate_pair_info_pandas(df, x_cen = scan_info['x_cen'], y_cen = scan_info['y_cen'], Epump = scan_info['Epump'], dd = scan_info['dd'], del_theta = scan_info['del_theta'], theta = scan_info['theta'])

    return {"coords" : coords, "dp_selection range" : dp_selection_range, "N" : N_lists, "SN" : SN_lists, "const" : const_lists, "best_dp" : best_dp, "best_metric" : best_metric}


def alpha_to_alpha(alpha, del_theta):
    alpha = np.radians(alpha)
    del_theta = np.radians(del_theta)
    theta = np.radians(11.576)
    ponep = (2*del_theta*np.sin(2*theta))/(alpha**2)
    alpha_prime = np.sqrt(2*del_theta*np.sin(2*theta)*ponep)
    return np.degrees(alpha_prime)


def mse(a, x, y):
    y_pred = alpha_to_alpha(x, a)
    return ((y_pred - y) ** 2).mean()


def get_optimal_del_theta(df, scan_info):
                       
    initial_guess = scan_info['del_theta']
    bounds = [(0, None)]

    result = minimize(mse, initial_guess, args=(np.degrees(df['alpha_1'].values), np.degrees(df['alpha_2'].values)), bounds=bounds)

    # Optimal value of a
    return result.x


def get_tot_vs_tot(df_pairs, dtmax = 35, dmin = 420, dmax = 460, chan1 = 2, chan2 = 4, tot_range = [[0,600], [0,600]], logscale_fix = True):
    """
    Returns ToT vs ToT for two particular chips.
    """
    df_tot_dt = df_pairs.loc[(df_pairs['delta_t'].abs() < dtmax) & (df_pairs["dist"] > dmin) & (df_pairs["dist"] < dmax)]

    df_chan = df_tot_dt[((df_tot_dt['chan_1'] == chan1) & (df_tot_dt['chan_2'] == chan2)) | ((df_tot_dt['chan_1'] == chan2) & (df_tot_dt['chan_2'] == chan1))]

    ToT_sum_1_1values = df_chan.loc[df_chan['chan_1'] == chan1]['ToT_sum_1'].values
    ToT_sum_2_1values = df_chan.loc[df_chan['chan_2'] == chan1]['ToT_sum_2'].values
    ToT_sum_chan_1 = np.concatenate((ToT_sum_1_1values, ToT_sum_2_1values))

    ToT_sum_2_2values = df_chan.loc[df_chan['chan_2'] == chan2]['ToT_sum_2'].values
    ToT_sum_1_2values = df_chan.loc[df_chan['chan_1'] == chan2]['ToT_sum_1'].values

    ToT_sum_chan_2 = np.concatenate((ToT_sum_2_2values, ToT_sum_1_2values))

    # ToT_filter_plot_vals, ToT_filter_plot_xbin, ToT_filter_plot_ybin = np.histogram2d(ToT_sum_chan_1, ToT_sum_chan_2, bins=int((tot_range[0][1] - tot_range[0][0])/25), range=tot_range, density=None, weights=None)
    ToT_filter_plot_vals, ToT_filter_plot_xbin, ToT_filter_plot_ybin = histogram2d_fast(histogram2d(ToT_sum_chan_1, ToT_sum_chan_2, bins=int((tot_range[0][1] - tot_range[0][0])/25), range=tot_range, density=None, weights=None), int((tot_range[0][1] - tot_range[0][0])/25), tot_range[0], tot_range[1])

    if logscale_fix:
        ToT_filter_plot_vals[ToT_filter_plot_vals < 1] = 1 # for log scale
    
    return ToT_filter_plot_vals, ToT_filter_plot_xbin, ToT_filter_plot_ybin   

"""
Some plotting code. Currently not used.
In general, I think it's OK to keep the plotting functions as something implemented in a notebook, with the workhorse of the calculations done in the library. This keeps the analysis functions standardized but allows the user to decide what and how they want to plot.
Nevertheless, some of the plot functions could probably go in here for plots that very routine and don't have much variability. 
"""

def plot_hline(y,xmin,xmax, set_color, width = 3.0, alpha = 1.0, fmt = "-"):   
    """
    Plots a horizontal line at a given y position over a selected range of x coordinates. This allows more control and ease of plotting a line compared to the standard ax.hline. 
    
    
    Parameters
    ----------
    x_vals : x values of the line
    y_vals : y values of the line 
    fmt : line parameters to determine the style of the line 
    set_color : selected color of the line 
    set_width : selected width of the line 
    set_alpha : selected opacity of the line
    
    """
    
    x_vals = np.arange(xmin,xmax, 0.001)
    y_vals = np.zeros(len(x_vals)) + y
    
    plt.plot(x_vals,y_vals, fmt, color = str(set_color), linewidth = width, alpha = alpha)


def plot_vline(x, ymin,ymax,set_color, width = 3.0, alpha = 1.0, fmt = "-"):  
    """
    Plots a vertical line at a given y position over a selected range of x coordinates. This allows more control and ease of plotting a line compared to the standard ax.vline. 
    
    Parameters
    ----------    
    x_vals : x values of the line
    y_vals : y values of the line 
    fmt : line parameters to determine the style of the line 
    set_color : selected color of the line 
    set_width : selected width of the line 
    set_alpha : selected opacity of the line
    
    """
    
    y_vals = np.arange(ymin,ymax, 0.001)
    x_vals = np.zeros(len(y_vals)) + x
    
    plt.plot(x_vals,y_vals, fmt, color = str(set_color), linewidth = width, alpha = alpha)


"""
Code for various simulations.
"""
def a1_vs_a2(del_theta, theta):
    """
    Returns x, alpha1, and alpha2.
    """
    x = np.arange(0.01, 1.00, 0.01)
    y = 1-x
    a1 = np.sqrt(2*np.radians(del_theta)*np.sin(2*np.radians(theta))*y/x)
    a2 = np.sqrt(2*np.radians(del_theta)*np.sin(2*np.radians(theta))*x/y)
    return x, np.degrees(a1), np.degrees(a2)


"""
Code for calculating spectral efficiency.
Definitely could be further optimized; not well-vectorized. Justin may come back to this at some point, or Joe can take a crack. As is, it's pretty fast, so not a high priority.
"""

def spdc_ce(x, E_pump, norm = True): 
    """
    Theoretical conversion efficiency.
    """
    ce = x*(E_pump-x)
    if norm:
        ce = ce/np.max(ce)
    return ce 


def conversion_efficiency(E1, Epump = 15, dd = 69.71155/100, theta = 0.202039314 , del_theta = 0.000383972):
    """
    More explicit way to calculate conversion efficiency using equations from Eisenberg.
    """
    E2 = Epump - E1
    alpha_1 = energy_to_alpha(E1)
    alpha_2 = energy_to_alpha(E2)
    
    D_pixels = dd/55e-6
    print(D_pixels)

    d1 = D_pixels*np.tan(alpha_1)
    d2 = D_pixels*np.tan(alpha_2)
    
    d_tot = d1 + d2

    Rx = np.sqrt(2*del_theta*((Epump-E1)/E1)*np.sin(2*theta))
    Ry = np.sqrt(2*del_theta*(E1/(Epump-E1))*np.sin(2*theta))
    cce = (Rx+Ry)**(-2)
    
    return cce/np.max(cce), d1


def points_within_square(points, x1, y1, x2, y2):
    """
    Checks whether a point is within a square.
    """
    min_x = min(x1, x2)
    max_x = max(x1, x2)
    min_y = min(y1, y2)
    max_y = max(y1, y2)

    x = points[:, 0]  # Extract the x-coordinates from the points array
    y = points[:, 1]  # Extract the y-coordinates from the points array
    
    within_x_range = np.logical_and(x >= min_x, x <= max_x)
    within_y_range = np.logical_and(y >= min_y, y <= max_y)
    
    within_square = np.logical_and(within_x_range, within_y_range)
    
    return points[within_square]


def points_outside_square(points, x1, y1, x2, y2):
    """ 
    Checks whether a point is outside a square.
    """
    min_x = min(x1, x2)
    max_x = max(x1, x2)
    min_y = min(y1, y2)
    max_y = max(y1, y2)

    x = points[:, 0]  
    y = points[:, 1]  
    
    outside_x_range = np.logical_or(x < min_x, x > max_x)
    outside_y_range = np.logical_or(y < min_y, y > max_y)
    
    outside_square = np.logical_or(outside_x_range, outside_y_range)
    
    return points[outside_square]


def generate_circle_points(xc, yc, r, num_points=1000):
    """
    Generates points along a circle with defined center and radius.
    """
    angles = np.linspace(0, 2 * np.pi, num_points)
    x = xc + r * np.cos(angles)
    y = yc + r * np.sin(angles)
    points = np.column_stack((x, y))
    return points


def within_region(radius, x_cen=257, y_cen=257, inner_x_cen=255.5, inner_y_cen=243.5, side_length=116, num_points=1000):
    """
    Checks what percentage of points along a circle are within an outsider region but outside an inner region.
    """
    circle_points = generate_circle_points(x_cen, y_cen, radius, num_points)
    circle_points_len = len(circle_points)
    #print(circle_points_len)
    
    inner_square_points = points_outside_square(circle_points, inner_x_cen - side_length/2, inner_y_cen - side_length/2, inner_x_cen + side_length/2, inner_y_cen + side_length/2)
    #print(len(inner_square_points))
    outer_square_points = points_within_square(inner_square_points, 0, 0, 513, 513)
    #print(len(outer_square_points))
    selected_points = len(outer_square_points)
    
    return (selected_points / circle_points_len)


def detector_acceptance(radii, x_cen, y_cen, inner_x_cen = 255.5, inner_y_cen = 243.5, side_length = 116, num_points = 100):
    """
    Looks at multiple radii to calculate acceptance as a function of energy (radius).
    """
    da_eff = []
    for rad in radii:
        da_eff.append(within_region(rad, x_cen, y_cen, inner_x_cen, inner_y_cen, side_length, num_points))
    return np.array(da_eff)


def gaussian_integral(x_c, sigma, A, x_0):
    """
    Gaussian integral.
    """
    lower_limit = 0
    upper_limit = x_c
    z = (upper_limit - x_0) / sigma
    cdf_value = stats.norm.cdf(z)
    integral_value = A * cdf_value
    
    return integral_value


def linear_fit(x, y):
    """
    Return a linear fit of x and y.
    """
    slope, intercept = np.polyfit(x, y, 1)
    
    return slope, intercept


def create_linear_function(slope, intercept):
    """
    Defines a linear function.
    """
    def linear_func(x):
        return slope * x + intercept
    
    return linear_func


def E_to_tot_peak_model(x, Eval_x = [7.5, 15], ToT_0 = [336, 545]):
    """
    Assume a linear fit between peak ToT and and energy.
    """
    slope, intercept = linear_fit(Eval_x, ToT_0)
    linear_func = create_linear_function(slope, intercept)
    return linear_func(x)


def E_to_tot_sig_model(x, Eval_x = [7.5, 15], sigma_0 = [55, 78]):
    """
    Assume a linear fit between ToT sigma and energy.
    """
    slope, intercept = linear_fit(Eval_x, sigma_0)
    linear_func = create_linear_function(slope, intercept)
    return linear_func(x)


def tot_acceptance(E, low_cutoff = 2):
    """
    Calculate the impact of ToT on experiment spectral acceptance.
    """
    y1_values = E_to_tot_peak_model(E)  # Calculate peak ToT vs E
    y2_values = E_to_tot_sig_model(E)  # Calculate sigma vs E

    tot_eff = []
    for i, x in enumerate(E):
        if x <= low_cutoff:
            tot_eff.append(0)
        else:
            x0 = y1_values[i]
            sigma = y2_values[i]
            frac_part = gaussian_integral(393.75, sigma, 1, x0)
            tot_part = gaussian_integral(5000, sigma, 1, x0)
            tot_eff.append(frac_part/tot_part)
            
    tot_eff = np.array(tot_eff)
    tot_eff_sym = np.copy(tot_eff)

    for i in range(len(tot_eff)//2): # make symmetrical
        tot_eff_sym[i] = tot_eff[i]*tot_eff[-i-1]
        tot_eff_sym[-i-1] = tot_eff[i]*tot_eff[-i-1]   
    
    tot_eff_fixed = tot_eff/tot_eff[np.argmax(tot_eff_sym)]
    tot_eff_fixed = np.where(tot_eff_fixed > 1, 1, tot_eff_fixed)
    
    return tot_eff_fixed


def total_acceptance(ce, da, tot):
    """
    Combine the different acceptances to get final curve.
    """
    model = da*tot
    
    ce = ce/np.max(ce)

    for i in range(len(model)//2):
        model[i] = model[i]*model[-i-1]
        model[-i-1] = model[i]
        
    model = model/np.max(model)    
    model2 = model*ce
    model2 = model2/np.max(model2)
    
    return model, model2

"""
Fast histogram wrapper functions.
"""

def histogram2d_fast(x, y, bins, range=None, density=None, weights=None):
    """
    Wrapper function for fast_histogram2d that also returns bin edges.
    """
    vals = histogram2d(x, y, bins=bins, range=range, weights=weights)
    if (type(bins) == list):
        xbins = np.linspace(range[0][0], range[0][1], bins[0]+1)
        ybins = np.linspace(range[1][0], range[1][1], bins[1]+1)
    else:
        xbins = np.linspace(range[0][0], range[0][1], bins+1)
        ybins = np.linspace(range[1][0], range[1][1], bins+1)
    
    return np.transpose(vals), xbins, ybins


def histogram1d_fast(data, bins, range=None, density=None, weights=None):
    """
    Wrapper function for fast_histogram1d that also returns bin edges
    """
    if (range == None):
        range = [min(data), max(data)]
    if (type(bins) == list or type(bins) == np.ndarray):
        vals = histogram1d(data, len(bins)-1, range=range, weights=weights)
        edges = bins
    else:
        vals = histogram1d(data, bins, range=range, weights=weights)
        edges = np.linspace(range[0], range[1], bins+1)
        
    return vals, edges

"""
Optimal ToT funding function.
"""

def find_optimal_tots(df, filename=None):
    # use statistics instead of constant "5" as cutoff for counts
    """
    Finds the optimal tot for each pixel given a dataframe of scattering data.
    """
    tots = df['ToT_sum']
    x = df['x']
    y = df['y']
    
    opt_tot = [[[] for y in range(0, 512)] for x in range(0, 512)]
    pixel_tots = [[[0 for z in range(0, 13)] for y in range(0, 512)] for x in range(0, 512)]
    
    for i in range(0, len(x)):
        if (tots[i] >= 300 and tots[i] <= 600):
            pixel_tots[x][y][(tots[i]/25)-12] += 1
            
    for i in range(0, 512):
        for j in range(0, 512):
            tot_counts = pixel_tots[x][y]
            for z in range(0, 13):
                if (tot_counts[z] <= 5):
                    opt_tot[i][j].append(300 + (z * 25))
                    break
                
            if (len(opt_tot[i][j]) == 0):
                opt_tot[i][j].append(400)
        

    if (filename != None):
        np.save(filename, np.asarray(opt_tot, dtype=np.int16))
    
    return np.asarray(opt_tot, dtype=np.int16)


"""
Entanglement verifications.

"""

def Pearson(x,y):
    """
    Calculates pearson coefficient and its corresponding uncertainty.
    
    Based on math done by Jonathon Schiff (former Andrei intern)
    Code written by myself. 
    
    """
    
    
    sum_1 = 0; sum_2 = 0; sum_3 = 0
    x_avg = np.mean(x)
    y_avg = np.mean(y)
    for i in range(len(x)):
        sum_1 += (x[i] - x_avg)*(y[i] - y_avg)
        sum_2 += (x[i] - x_avg)**2 
        sum_3 += (y[i] - y_avg)**2
        
    sum_2_root = np.sqrt(sum_2)
    sum_3_root = np.sqrt(sum_3)
    
    r_val = sum_1/(sum_2_root*sum_3_root)
    
    ##Step 1
    N = len(x)
    x_err = np.sqrt(x)
    y_err = np.sqrt(y)
    ##Step 2
    x_err_squared = [i**2 for i in x_err]
    y_err_squared = [i**2 for i in y_err]
    
    step_2_xsum = np.sum(x_err_squared)
    step_2_ysum = np.sum(y_err_squared)
    
    x_avg_err = np.sqrt(step_2_xsum)/N
    y_avg_err = np.sqrt(step_2_ysum)/N
    
    ##Step 3
    xi_avg_diff = [i**2 + x_avg_err**2 for i in x_err]
    yi_avg_diff = [i**2 + y_avg_err**2 for i in y_err]
    
    ##Step 4 
    x_avg = np.mean(x)
    y_avg = np.mean(y)
    
    diff_prod_err = []
    for i in range(len(x_err)):
        inside_i = (xi_avg_diff[i]/(x[i] - x_avg))**2 + (yi_avg_diff[i]/(y[i] - y_avg))**2
        inside_squared = np.sqrt(inside_i)
        diff_prod_err.append(inside_squared*(x[i] - x_avg)*(y[i] - y_avg))
    
    ##Step 5
    
    diff_prod_sum_err = np.sqrt(np.sum([i**2 for i in diff_prod_err]))
    
    ##Step 6 
    
    xi_avg_diff_squared = []
    yi_avg_diff_squared = []
    
    for i in range(len(xi_avg_diff)):
        xi_avg_diff_squared.append(2*(x[i] - x_avg)*xi_avg_diff[i])
        yi_avg_diff_squared.append(2*(y[i] - y_avg)*yi_avg_diff[i])
        
    ##Step 7 
    
    xi_avg_diff_squared_err_sum = np.sqrt(np.sum([i**2 for i in xi_avg_diff_squared]))
    yi_avg_diff_squared_err_sum = np.sqrt(np.sum([i**2 for i in yi_avg_diff_squared]))
    
    
    ##Step 8 
    
    x_avg_diff_squared_sum = np.sum([(i - x_avg)**2 for i in x])
    y_avg_diff_squared_sum = np.sum([(i - y_avg)**2 for i in y])
    
    inside_x = (xi_avg_diff_squared_err_sum/x_avg_diff_squared_sum)**2 
    inside_y = (yi_avg_diff_squared_err_sum/y_avg_diff_squared_sum)**2 
    
    inside_total = np.sqrt(inside_x + inside_y)
    
    step_8_total = inside_total*inside_x*inside_y
    
    ##Step 9
    
    inv_term = (2*np.sqrt(x_avg_diff_squared_sum*y_avg_diff_squared_sum))**-1 
    
    step_9_total = inv_term*step_8_total
    
    #Step 10
    
    diff_prod_vals = []
    
    for i in range(len(x)):
        x_prod_i = x - x_avg
        y_prod_i = y - y_avg
        diff_prod_vals.append(x_prod_i*y_prod_i)
    
    diff_prod_sum = np.sum(diff_prod_vals)
    
    inside_left_term = (diff_prod_sum_err/diff_prod_sum)**2 
    
    inside_right_term = (step_9_total/(np.sqrt(x_avg_diff_squared_sum*y_avg_diff_squared_sum)))**2
    
    r_err = r_val*np.sqrt(inside_left_term + inside_right_term)
    
    
    return r_val,abs(r_err)


def contour_2d_energy(pix_range = 1025, xcen = 258, ycen = 257, norm_max = None, norm_sum = None, c_map = "ocean_r", ticksize = 14,
                     labelsize = 18, bs_opacity = 1.0, edge_opacity = 1.0, bs_width = 1.0, edge_width = 1.0, epump = 15, 
                     contour_levels = [1,2,4,6,8,10,12,14,16], bs_xmin = 150, bs_xmax = 360, bs_ymin = 139, bs_ymax = 373, cwidth = 1.0, 
                     figsize = (7.60,7.60), cbar_tick_size = 12, clabel_size = 13, norm_density = True, detector_dist = 69.749, del_theta = 0.022,
                     theta = 11.576): 
    
    
    """
    Parameters
    ----------
    
    pix_range: max x and y values of pixels
    xcen, ycen = x and y values of bragg center 
    norm_max = boolean value to determine if normalization using max value of density is used
    norm_sum = boolean value to determine if normalization using sum value of density is used
    c_map = color map for the contour lines
    ticksize = font size for x and y ticks 
    labelsize = font size for x and y labels
    bs_opacity = opacity of beamstop lines 
    edge_opacity = opacity of detector edge lines
    bs_width = width of beamstop lines
    edge_width = width of detector edge lines
    Epump = Energy value in keV of the pump beam
    contour_levels = locations in energy of contour lines
    bs_xmin, bs_xmax = lowest and highest x value corresponding to the beamstop
    bs_ymin, bs_ymax = lowest and highest y value corresponding to the beamstop
    cwidth = widths of the contour lines
    figsize = size of the figure
    cbar_tick_size = size of the colorbar ticks 
    clabel_size = size of the contour ticks
    x_vals, y_vals = x and y arrays holding x values
    X,Y = X and Y components of the Radius matrix
    Radius = Matrix housing radius values
    density = Matrix housing density values
    detector_dist = distance to the detector
    del_theta = detuning angle
    theta = bragg angle 
    
    
    """
    
    
    x_vals = np.arange(0,pix_range,1)
    y_vals = np.arange(0,pix_range,1)

    x,y = np.meshgrid(x_vals,y_vals)
    
    radius = np.sqrt((xcen-x)**2 + (ycen-y)**2)  ## Units of pixels
    
    radius[xcen,ycen] = 1                     ## Changing radius value at center to prevent divide by zero flag
    radius[ycen,xcen] = 1
    
    energy = radius_to_energy(radius, Epump = epump, dd = detector_dist, del_theta = del_theta, theta = theta)         ## Units of keV
    
    radius = radius*0.000055                     ## Converting to units of meters
    
    if norm_density:                             ## Checking whether to divide by radius
    
        density = energy*(epump - energy)/(2*np.pi*radius)
        
    else:
        density = energy*(epump - energy)
        
    
    if norm_max:                                    
        density_new = density/np.max(density)        
    elif norm_sum:
        density_new = density/np.sum(density)
    else:
        density_new = density
        
    color = "black"
    
    fig, ax = plt.subplots(figsize = figsize)

    h = plt.contour(x,y, energy, levels = contour_levels, cmap = c_map, linewidths = cwidth)
    ax.set_xlabel("x [pix]", fontsize = labelsize)
    ax.set_ylabel("y [pix]", fontsize = labelsize)
    ax.clabel(h, fontsize = clabel_size)
    plt.xticks(fontsize = ticksize)
    plt.yticks(fontsize = ticksize)
    plot_hline(bs_ymin,bs_xmin,bs_xmax, color, width = bs_width, alpha = bs_opacity)
    plot_hline(bs_ymax,bs_xmin,bs_xmax, color, width = bs_width, alpha = bs_opacity)
    plot_vline(bs_xmin,bs_ymin,bs_ymax, color, width = bs_width, alpha = bs_opacity)
    plot_vline(bs_xmax,bs_ymin,bs_ymax, color, width = bs_width, alpha = bs_opacity)
    plot_hline(512,0,512, color, width = edge_width, alpha = edge_opacity)
    plot_vline(512,0,512, color, width = edge_width, alpha = edge_opacity)
    im = plt.imshow(density_new)
    cbar = plt.colorbar(im, shrink = 0.805)
    cbar.ax.tick_params(labelsize = cbar_tick_size)
    plt.ylim(0,pix_range)
    plt.show()

def entanglement_test(df, plot = True, prints = True):
    E1 = df["E_1"]; E2 = df["E_2"]
    hbar = 6.582e-19   ## in units of keV*s
    w_s = np.array(E1/hbar); w_i = np.array(E2/hbar)
    
    if plot:
        
        fig, ax0 = plt.subplots(ncols = 1)
        h = ax0.hist2d(w_s,w_i, bins = (np.arange(min(w_s), max(w_s) + 1e17, 1e17),np.arange(min(w_i), max(w_i) + 1e17, 1e17)))
        cbar = fig.colorbar(h[3], ax = ax0)
        plt.xlabel("$ω_s$", fontsize = 16)
        plt.ylabel("$ω_i$", fontsize = 16)
        plt.show()
    
    
    sigma_w_s = np.std(w_s/1e10)*1e10; sigma_w_i = np.std(w_i/1e10)*1e10
    
    rho_w = Pearson(w_s/1e10,w_i/1e10)[0]
    rho_w_err = Pearson(w_s/1e10,w_i/1e10)[1]
    
    if prints: print("ρ_w = {:.5f} +/- {:.5f}".format(rho_w,rho_w_err))
    
    numerator = (sigma_w_s**2 + 2*rho_w*sigma_w_i*sigma_w_s + sigma_w_i**2)**2
    denom = 4*(1-rho_w**2)*(sigma_w_s**2)*(sigma_w_i**2)

    entanglement_coeff = np.sqrt(numerator/denom)
    
    if prints: print("Δ(ω_s + ω_i)Δ(t_s - t_i) = {:.3f} ".format(entanglement_coeff))
    
    return entanglement_coeff
