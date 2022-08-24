from itertools import cycle
from lib2to3.pytree import convert
from logging import root
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
from scipy.signal import find_peaks
from scipy.signal import argrelextrema

def peak_dist_set(max_spd_rpm: float, signal_gap_deg: float, time_signal_raw: pd.Series, flood_ratio: float=0.9):
    from math import ceil
    max_spd_deg__ms = max_spd_rpm * 6 / 1000.0  # rpm to dp_ms conversion
    dt_ms = 1.0e3 * (time_signal_raw[1] - time_signal_raw[0])  # milli-second
    gap_dist = flood_ratio * signal_gap_deg / max_spd_deg__ms / dt_ms
    return int(ceil(gap_dist))

def cranking_data_load(file_path: Path):
    if (not file_path.exists()) or (not file_path.is_file()):
        raise FileExistsError
    motor_raw_file = pd.read_csv(file_path,index_col=0, sheet_name=None)#pd.read_excel
    motor_p3 = np.zeros(1440)
    motor_p4 = np.zeros(1440)
    motor_p2 = np.zeros(1440)
    motor_p1 = np.zeros(1440)
    cnt = 0
    for motor_df in motor_raw_file.values():
        cnt += 1
        motor_p3 += motor_df['cyl3'].to_numpy()
        motor_p4 += motor_df['cyl4'].to_numpy()
        motor_p2 += motor_df['cyl2'].to_numpy()
        motor_p1 += motor_df['cyl1'].to_numpy()
    motor_p3 = motor_p3 / cnt
    motor_p4 = motor_p4 / cnt
    motor_p2 = motor_p2 / cnt
    motor_p1 = motor_p1 / cnt
    return motor_p3, motor_p4, motor_p2, motor_p1

def generic_scope_read(filename: Path, channel_dict: Dict, main_scope: bool = True):

    #channel_dict = {'ch3': '2', 'ch4': '1', 'z_pulse': '3', 'a_pulse': '4'}
    # key: signal name
    # value: signal channel

    name_col = ['time']
    for i in range(1,5):
        str_key = channel_dict[str(i)]
        name_col.append(str_key)
    rowread = pd.read_csv(
        filename,
        sep=',',
        header=36,
        names=name_col,
        usecols=(0, 1, 3, 5, 7))  # read in raw csv file, skip the repeating time columns

    return rowread

def scope_data_read(main_filename: Path, aux_filename: Path, 
                    channel_dict_main: dict = {'1':'cylinder4','2':'cylinder3', '3': 'z_pulse', '4': 'a_pulse'},
                    channel_dict_aux: dict = {'1':'cylinder1','2':'cylinder2','3':'cylinder3','4':'frp'},
                   ):
    main_df = generic_scope_read(main_filename, channel_dict=channel_dict_main, main_scope=True)
    main_df.reset_index(inplace=True)
    aux_df = generic_scope_read(aux_filename, channel_dict=channel_dict_aux, main_scope=False)
    aux_df.reset_index(inplace=True)

    main_key_set = set(channel_dict_main.values())
    aux_key_set = set(channel_dict_aux.values())
    result = list(main_key_set.intersection(aux_key_set))
    assert len(result) == 1
    result = result[0]

    dt = main_df.loc[1, 'time'] - main_df.loc[0, 'time']

    peak_pressure_loc = main_df[result].argmax()
    peak_pressure_loc_aux = aux_df[result].argmax()

    time_datum = main_df.loc[peak_pressure_loc,'time']

    main_df['index'] = main_df['index'] - main_df.loc[peak_pressure_loc,'index']
    aux_df['index'] = aux_df['index'] - aux_df.loc[peak_pressure_loc_aux,'index']

    res = main_df.merge(aux_df, on='index')

    res.drop(['index','time_y', result + '_y'],inplace=True, axis=1)

    res.reset_index(inplace=True)

    res.rename(columns = {result + '_x': result,'time_x': 'time'}, inplace=True)

    res['time'] = res['time'] - time_datum

    return res


def z_pulse_processing(max_rpm: float, time_signal: pd.Series, z_signal: pd.Series, flood_ratio: float=0.9):
    z_pulse_gap_deg = 360.0 # deg; Z pulse gap
    z_peak_loc, _ = find_peaks(z_signal, 
                               height=(4.5,7),
                               threshold=None,
                               distance=peak_dist_set(max_rpm,z_pulse_gap_deg,time_signal,flood_ratio))
    z_binary_ttl = np.zeros(z_signal.shape[0])
    z_binary_ttl[z_peak_loc] = 1.0
    return z_peak_loc, z_binary_ttl

def a_pulse_processing_classic(a_signal: pd.Series):
    high_lv = 3.0
    low_lv = 2.0
    data_num = a_signal.shape[0]
    a_binary_ttl = np.zeros((data_num,))
    a_peak_loc = []
    for i in range(1, data_num-1):
        if a_signal[i] > high_lv and a_signal[i-1] < low_lv:
            a_binary_ttl[i] = 1.0
            a_peak_loc.append(i)
    a_peak_loc = np.array(a_peak_loc)
    return a_peak_loc, a_binary_ttl

def a_pulse_processing_scipy_extrema(max_rpm: float, time_signal: pd.Series, a_signal: pd.Series):
    a_pulse_gap_deg = 0.5
    a_pulse_distance=peak_dist_set(max_rpm, a_pulse_gap_deg, time_signal, flood_ratio=0.9)

    a_peak_loc = argrelextrema(np.array(a_signal), 
                                comparator=np.greater_equal,
                                order=a_pulse_distance,
                               )[0]
    a_binary_ttl = np.zeros(a_signal.shape[0])
    a_binary_ttl[a_peak_loc] = 1.0
    return a_peak_loc, a_binary_ttl


def signal_validation(z_peak_loc: np.array, a_binary_ttl: np.array, error_lim: float = 8.0, verbose: bool = False):
    result = []
    valid_flag = True
    z_pulses_number = z_peak_loc.shape[0]
    for i in range(z_pulses_number-1):
        cur_a_binary_seg = a_binary_ttl[z_peak_loc[i]:z_peak_loc[i+1]]
        elapsed_deg = np.sum(cur_a_binary_seg)*0.5
        result.append((i,elapsed_deg))
        if abs(elapsed_deg-359.5) >= error_lim:
            valid_flag = False
            if verbose:
                print(f'Elapsed degrees between Z peak {i} and {i+1} is + {elapsed_deg} + degrees')
    if verbose:
        print(f'Current data validation results: {result}')
        print(f'Current maximum error tolerance: {error_lim}')
        print(f'Passed the validation?: {valid_flag}' )
    return valid_flag, result

def res_pulses_process(res_df: pd.DataFrame, max_rpm: float = 1200.0, 
                       flood_ratio: float = 0.9, a_process_classic: bool = True,
                       error_threshold: float = 5.0, verbose: bool = False,
                       ignore_error: bool = False):
    
    
    z_peak_loc, z_binary_ttl = z_pulse_processing(max_rpm=max_rpm, 
                                                  time_signal=res_df['time'],
                                                  z_signal=res_df['z_pulse'],
                                                  flood_ratio=flood_ratio)
    if a_process_classic:
        a_peak_loc, a_binary_ttl = a_pulse_processing_classic(res_df['a_pulse'])
    else: 
        a_peak_loc, a_binary_ttl = a_pulse_processing_scipy_extrema(max_rpm, res_df['time'], res_df['a_pulse'])
    valid_bool, result = signal_validation(z_peak_loc = z_peak_loc, 
                                           a_binary_ttl = a_binary_ttl, 
                                           error_lim = error_threshold, 
                                           verbose=verbose)
    if (not valid_bool) and (not ignore_error):
        raise ValueError('Current scope data validation failed and was not ignored.')
    
    res_df['cumu_cad'] = np.cumsum(a_binary_ttl * 0.5)  # cumulative CAD deg
    res_df['z_binary'] = z_binary_ttl
    res_df['a_binary'] = a_binary_ttl

    return res_df, z_peak_loc, a_peak_loc 



def cad_pegging(res_df: pd.DataFrame, peg_cylinder_num: int, z_peak_loc: np.array):
    
    '''
    Return the data frame with cumu_cad pegged.
    0 cumulative CAD located at the pegged cylinder TDC
    '''
    
    # as firing order is fixed, only leading_cylinder_str is required
    cad_offset_lead = [-162.0, 18.0 ,18.0, -162.0]

    offset = cad_offset_lead[peg_cylinder_num - 1]

    idx = res_df.loc[res_df['time']==0].index
    
    target_cumu_cad = res_df.loc[idx,'cumu_cad'].values[0]
    if offset>0 and offset<40:
        target_cumu_cad += 40.0
    right_cushion_idx = res_df[res_df['cumu_cad']==target_cumu_cad].index.tolist()[0]


    # TODO: make cranking data selection possible;
    # reference: previous version :
    # # location of the location of Z pulse of interest
    # sync_index = sync_pt_location(p_lead, p_follow, z_peaks, motorflag9=motor_flag, forced_move_flag=forced_move)
    #   if motorflag9 and (x.shape[0] == 0 or forced_move_flag):
    #        rest_zpeaks = Zpeaks[Zpeaks > F_peaks]
    #        rest_zpeaks = rest_zpeaks[1:] # updated Jan 27 2022, potentially skipping z-pulse not neighboring cyl l
    #        i = 0 
    #        while i < (rest_zpeaks.shape[0] - 1) and p_lead[rest_zpeaks[i]] < 0.5:
    #            i += 2  # updated Jan 27 2022, skipping the middle Z pulse not close to current 2 cylinders
    #        return rest_zpeaks[i]

    x = z_peak_loc[z_peak_loc < right_cushion_idx][-1]

    res_df['cumu_cad'] = res_df['cumu_cad'] - (res_df.loc[x, 'cumu_cad'] - cad_offset_lead[peg_cylinder_num - 1])
    
    res_df['pegged_cylinder'] = np.ones(res_df.shape[0],dtype=int) * peg_cylinder_num 

    return res_df

def roi_select(res_df: pd.DataFrame, cycle_num: float = 1, cycle_start_offset: float = 0):
    # both cycle_num and cycle_start_offset can only be pure integer or float numbers ending with 0.25, 0.5, 0.75 (quartiles) 
    # cycle index 0 is the cycle starting with the cylinder TDC where 0 cumulative CAD is located
    # cylinder start specifies the starting cylinder (for saving purpose)
    #
    peg_cylinder = res_df.loc[0, 'pegged_cylinder']
    next2fire = {3:4, 4:2, 2:1, 1:3}
    previous2fire = {1:2, 3:1, 4:3, 2:4}

    def data_validation(argument: float) -> bool:
        cur_res = argument - np.floor(argument)
        if abs(cur_res) < 1e-9:
            return True
        i = 3
        while i > 0:
            if np.abs(cur_res * 4 - i) < 1e-9:
                return True
            i -= 1
        return False

    def find_starting_cylinder(start_offset: float, peg_cylinder: int) -> int:
        refer_dict = next2fire if start_offset > 0 else previous2fire
        cylinder_start = peg_cylinder
        res = start_offset - np.floor(start_offset) if start_offset >= 0 else start_offset - np.ceil(start_offset)
        res = np.abs(res)
        while res > 1e-9:
            cylinder_start = refer_dict[cylinder_start]
            res -= 0.25
        return cylinder_start
        
    def conv_cumu_cad_2_cycle_num(cumu_cad, cutoff_start):
        if cumu_cad < cutoff_start:
            return 0
        else:
            return np.ceil((cumu_cad - cutoff_start)/720.0)
    def conv_cumu_cad_2_cad(cumu_cad, cutoff_start):
        if cumu_cad < cutoff_start:
            res = (cutoff_start - cumu_cad)
            res -= np.floor((cutoff_start - cumu_cad)/720.0) * 720.0
            return 360.0 - res
        else:
            res = (cumu_cad - cutoff_start)
            quo = np.floor((cumu_cad - cutoff_start)/720.0)
            res -= quo * 720.0
            return res - 360.0

    assert cycle_num > 0
    assert data_validation(cycle_num)
    assert data_validation(cycle_start_offset)
    
    cumu_cad_start = -360 + cycle_start_offset * 720.0 
    cumu_cad_end = cumu_cad_start + 720.0 * cycle_num + 540.0

    res_df = res_df[((res_df['cumu_cad']>=cumu_cad_start)&(res_df['cumu_cad']<cumu_cad_end))]

    start_cylinder = find_starting_cylinder(cycle_start_offset, peg_cylinder)
    cur_cylinder = start_cylinder
    for i in range(4):
        print(f'current cylinder is cyl {cur_cylinder}')
        start_offset_cumu_cad = cumu_cad_start + i * 180.0
        print(f'current offset is cyl {start_offset_cumu_cad}')
        cur_cycle_num_series = res_df['cumu_cad'].apply(conv_cumu_cad_2_cycle_num,args=(start_offset_cumu_cad,))
        cur_cad_series = res_df['cumu_cad'].apply(conv_cumu_cad_2_cad,args=(start_offset_cumu_cad,))
        res_df['cylinder' + str(cur_cylinder) + '_cad'] = cur_cad_series.loc[:]
        res_df['cylinder' + str(cur_cylinder) + '_cycle_num'] = cur_cycle_num_series.loc[:]
        cur_cylinder = next2fire[cur_cylinder]

    return res_df

def pressure_volt2bar(res_df, scale=10.0):

    window_start = -349.0 
    window_end = -300.0    

    for cyl_index in range(4,0,-1):
        cur_cad_key = 'cylinder' + str(cyl_index) + '_cad'
        cur_cycle_num_key = 'cylinder' + str(cyl_index) + '_cycle_num'
        cur_set = set(res_df[cur_cycle_num_key])
        cur_set.remove(0.0)
        cycle_num_list = list(cur_set)
        cycle_num_list.sort()
        print(cyl_index, cur_cad_key, cur_cycle_num_key,cycle_num_list)
        for cur_cycle_index in cycle_num_list:
            try:
                cur_peg_val = res_df[(res_df[cur_cycle_num_key]==cur_cycle_index) &
                                            (res_df[cur_cad_key]>=window_start) & 
                                            (res_df[cur_cad_key]<window_end)
                                           ]['cylinder'+str(cyl_index)].mean()
                idx = res_df[res_df[cur_cycle_num_key]==cur_cycle_index].index
                res_df.loc[idx, 'cylinder'+str(cyl_index)] -= cur_peg_val
            except:
                print(f'not pegging cycle number {cur_cycle_index} for cylinder {cyl_index}')
        res_df.loc[:,'cylinder'+str(cyl_index)] *= scale
    res_df['frp'] = res_df['frp'] * 65 - 32.5
    return res_df

def time2cad_conversion(res_df:pd.DataFrame, cad_key: str='cumu_cad',conversion_rule='first'):
    res_df_groupby = res_df.groupby(cad_key)
    assert conversion_rule in ['first','mean','last','median']
    if conversion_rule == 'first':
        return res_df_groupby.first()
    elif conversion_rule == 'mean':
        return res_df_groupby.mean()
    elif conversion_rule == 'last':
        return res_df_groupby.last()
    elif conversion_rule == 'median':
        return res_df.median()
    else:
        raise ValueError('conversion rule input invalid')



