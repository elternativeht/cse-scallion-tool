
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