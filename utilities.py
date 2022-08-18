
GOOGLE_PARSER_KEY_LIST = [
    'overall_name',
    'exp_index',
    'month',
    'day',
    'background_per_cycle_hc',
    'total_cycle_fp',
    'total_mole_fp',
    'c_mass_from_hc_fp',
    'c_mass_from_co_fp',
    'c_mass_from_co2_fp',
    'c_mass_from_hc_pc1',
    'c_mass_from_co_pc1',
    'c_mass_from_co2_pc1',
    'c_mass_from_hc_pc2',
    'c_mass_from_co_pc2',
    'c_mass_from_co2_pc2',
    'c_mass_from_hc_pc3',
]

GOOGLE_PARSER_REL_ROW_LIST = [0, 0, 1, 1, 5, 6,
                              6, 15, 15, 15, 16, 16,
                             16, 17, 17, 17, 18]

GOOGLE_PARSER_COL_LIST = ['A', 'C', 'A', 'B', 'AB', 'M',
                          'O', 'AC', 'AD', 'AE', 'AC', 'AD',
                          'AE', 'AC', 'AD', 'AE', 'AC'
                         ]
'''
0A - Overall
0C - Exp Index
1A - Month
1B - Day
5AB - Background per-cycle HC
6M - Firing cycle num
6O - Cold start mole
15AC-AE cold start HC-CO-CO2 C
16AC-AE 1P  HC-CO-CO2 C
17AC-AE 2P  HC-CO-CO2 C
18AC    3p  HC
'''

def gdi_20t_engine_vol_calc(theta_off_tdc_deg, offset: float=0.0):
    import numpy as np
    theta_off_tdc_deg = theta_off_tdc_deg - offset 
    bore_meter = 87.5e-3 # bore
    stroke_meter = 83.1e-3 # stroke
    L_meter = 155.869e-3 # length
    Rc = 10.0 # compression ratio
    theta_rad = theta_off_tdc_deg * np.pi / 180.0 
    coff = ( 
           Rc/(Rc-1) - 0.5*(1+np.cos(theta_rad)) + L_meter/stroke_meter \
           - 0.5*np.power( (2*L_meter/stroke_meter)**2 - np.power(np.sin(theta_rad), 2), 0.5)
           )
    volume_cubic_meter = coff * 500e-6
    return volume_cubic_meter    