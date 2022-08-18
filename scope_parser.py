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

    pd_dict = {}

    rowread = pd.read_csv(
        filename,
        sep=',',
        header=36,
        names=['time', '1', '2', '3', '4'],
        usecols=(0, 1, 3, 5, 7))  # read in raw csv file, skip the repeating time columns
    
    if main_scope:
        key_list = ['cylinder3', 'cylinder4', 'z_pulse','a_pulse']
    else:
        key_list = ['cylinder3', 'cylinder2', 'cylinder1', 'frp']
    
    pd_dict['time'] = rowread['time']

    for key_ in key_list:
        assert key_ in list(channel_dict.keys())
        pd_dict[key_] = rowread[channel_dict[key_]]

    return pd.DataFrame(pd_dict)

def scope_data_read(main_filename: Path, aux_filename: Path, 
                    channel_dict_main: dict = {'cylinder3': '2', 'cylinder4': '1', 'z_pulse': '3', 'a_pulse': '4'},
                    channel_dict_aux: dict = {'cylinder3': '3', 'cylinder2': '2', 'cylinder1': '1', 'frp': '4'},
                   ):
    main_df = generic_scope_read(main_filename, channel_dict=channel_dict_main, main_scope=True)
    main_df.reset_index(inplace=True)
    aux_df = generic_scope_read(aux_filename, channel_dict=channel_dict_aux, main_scope=False)
    aux_df.reset_index(inplace=True)

    dt = main_df.loc[1, 'time'] - main_df.loc[0, 'time']
    main_df['index'] = main_df['index'] - main_df.loc[main_df['cylinder3'].argmax(),'index']
    aux_df['index'] = aux_df['index'] - aux_df.loc[aux_df['cylinder3'].argmax(),'index']

    res = main_df.merge(aux_df, on='index')

    res.drop(['index','time_x','time_y','cylinder3_y'],inplace=True, axis=1)

    res.reset_index(inplace=True)

    res.rename(columns = {'cylinder3_x':'cylinder3','index':'time'}, inplace=True)

    res['time'] = res['time'] * dt

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