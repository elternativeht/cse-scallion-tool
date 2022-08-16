import numpy as np
from scipy.optimize import fsolve
from typing import Tuple, List

def reaction_coeff_calc(rh_percent: float, p_atm_psi: float, temp_celsius: int = 22) -> tuple[float, float, float]:
    '''
    Use relative humidity (RH), current atmspheric pressure (in psi) and current temperature to calculate the current
    chemical coefficients of the atmospheric components in the combustion reaction formula.

    Params:

    - rh_percent: float; relative humidity (in percentage)

    - p_atm_psi:  float; the current atmospheric pressure in psi

    - temp_celsius: int; the current chamber temperature in Celsius degrees

    '''
    # global atmosphere_composition
    atmosphere_comp = {'N2': 78.084e-2,  
                       'O2': 20.946e-2,  
                       'CO2': 0.038e-2,
                       'Ar': 0.934e-2,
                       'Ne': 18.18e-6,
                       'He': 5.24e-6
                       }
    # treat all other gas as equivalent n2, obtain n2 coefficient in the reaction formula
    n2_coeff = (atmosphere_comp['N2']
                + atmosphere_comp['Ar']
                + atmosphere_comp['Ne']
                + atmosphere_comp['He']) / atmosphere_comp['O2']
    # obtain co2 coefficient in the reaction formula
    co2_coeff = atmosphere_comp['CO2'] / atmosphere_comp['O2']

    p_sat_vapor_kpa = {20: 2.3393, 
                       21: 2.4923,
                       22: 2.6453, 
                       23: 2.8156,
                       24: 2.9858, 
                       25: 3.1699}
    # calculate the current vapor molar fraction in the air and water vapor coefficient in the reaction formula
    p_vapor_pascal = p_sat_vapor_kpa[temp_celsius] * rh_percent * 10.0
    p_atm_pascal = p_atm_psi * 6894.76
    vapor_ratio = p_vapor_pascal / p_atm_pascal
    
    h2o_coeff = vapor_ratio * (1 + n2_coeff + co2_coeff) / (1 - vapor_ratio)

    return n2_coeff, co2_coeff, h2o_coeff


def general_reaction_func(X: List, *parameters_group: Tuple[float, float, float, float]):
    '''

    Reaction formula:

    react_fuel_n * fuel [CxHyOz] + react_air_n * air -> n1 * uHC + n2 * CO + n3 * CO2 + n4 * H2 + n5 * N2 + n6 * O2 + n7 * H2O

    params:

    - X: List, placeholder variables for the `scipy.optimize` solver
    - parameter_group: Tuple
    '''
    hc_component_list, measured_gas_tuple, react_coeff, fuel_tuple = parameters_group

    # X: the reaction component coefficient 
    react_air_n, react_fuel_n, product_h2_n, product_h2o_n, product_n2_n, product_o2_n = X

    # hc, co, co2 in mg; nproduct in mole
    product_hc_mass, product_co_mass, product_co2_mass, product_n = measured_gas_tuple

    # specify the fuel and exhaust HC atom numbers
    hc_carbon_num, hc_hydrogen_num = hc_component_list
    carbon_fuel_num, hydrogen_fuel_num, oxygen_fuel_num = fuel_tuple

    # print('HC mass: %.2f; CO mass: %.2f; CO2 mass %.2f' %(nHC,nCO,nCO2))
    product_hc_n = product_hc_mass / (hc_carbon_num * 12 + hc_hydrogen_num) / 1000.0
    product_co2_n = product_co2_mass / 44.0 / 1000.0
    product_co_n = product_co_mass / 28 / 1000.0
    react_n2_coeff, react_co2_coeff, react_h2o_coeff = react_coeff

    # carbon balance 
    eq1 = carbon_fuel_num * react_fuel_n + react_air_n * react_co2_coeff  \
          - (product_co2_n + product_co_n + hc_carbon_num * product_hc_n) 
    # hydrogen balance 
    eq2 = hydrogen_fuel_num * react_fuel_n + react_air_n * product_h2o_n * 2 \
          - (product_h2_n * 2 + product_h2o_n * 2 + hydrogen_fuel_num * product_hc_n)  
    # nitrogen mass balance
    eq3 = react_air_n * react_n2_coeff - product_n2_n  
    # oxygen mass balance
    eq4 = react_air_n * (1 * 2 + react_co2_coeff * 2 + react_h2o_coeff) + oxygen_fuel_num * react_fuel_n - (
            product_co2_n * 2 + product_co_n * 1 + product_h2o_n + product_o2_n * 2)  
    # water-gas phase equation
    eq5 = product_co_n * product_h2o_n / (product_co2_n * product_h2_n * 3.5) - 1.0  
    # mass overall balance
    eq6 = product_co2_n + product_co_n + product_h2_n + product_h2o_n \
          + product_n2_n + product_hc_n + product_o2_n - product_n  
    return [eq1, eq2, eq3, eq4, eq5, eq6]


def lambda_solve(
                emitted_hc_mass: float,
                emitted_co_mass: float,
                emitted_co2_mass: float,
                coldstart_total_n: float,
                firing_cycle: int,
                coldstart_total_cycle: float,
                fuel_tuple: tuple[float, float, float] = (8, 1.87 * 8, 0),
                hc_tuple: tuple[float, float] = (6, 14),
                rh: float = 40.0,
                temp_celsius: float = 22.0,
                p_atm_psi: float = 14.7
) -> float:
    """
    Main function to apply to solve for combustion lamabda (real AFR / stoichiometric AFR)

    - Params:

    `emitted_hc_mass`: float; collected HC mass during cold start firing process [mg]

    `emitted_co_mass`: float; collected CO mass during cold start firing process [mg]

    `emitted_co2_mass`: float; collected CO2 mass during cold start firing process [mg]

    `coldstart_total_n`: float; collected total emitted gas mole during cold start firing process [mol]

    `firing_cycle`: int; specified firing cycle number by user

    `coldstart_total_cycle`: float; total elapsed engine cycle number during the whole cold start firing process

    `fuel_tuple`: tuple; a three-element tuple containing the combusted fuel molecular C/H/O component; 
    default to be (8, 14.96, 0.0)

    `hc_tuple`: tuple; a two-element tuple containing the HC molecular C/H componenet; default to be hexane (6, 14)
    
    `rh`: float; relative humidity in percentage, default to be 40.0

    `temp_celsius`: float; room temperature in Celsius degrees, default to be 22.0

    `p_atm_psi`: float; room atmospheric pressure in pounds forces per square inch (psi), default to be 14.7 
    
    """
    X0 = [1.17569722e-02,  # reacted air mole initial guess
          8.93672076e-04,  # reacted fuel mole
          7.13116796e-05,  # reacted hydrogen mole
          5.93255496e-03,  # reacted water mole
          4.43540516e-02,  # reacted nitrogen mole
          3.25164877e-03,  # reacted oxygen mole
          ]
    cycle_mole = coldstart_total_n * firing_cycle / coldstart_total_cycle
    meas_result = (emitted_hc_mass, emitted_co_mass, emitted_co2_mass, cycle_mole)
    react_coeff = reaction_coeff_calc(rh, p_atm_psi, temp_celsius)
    root = fsolve(general_reaction_func, X0, (hc_tuple, meas_result, react_coeff, fuel_tuple))
    return root[0] / (fuel_tuple[0] + fuel_tuple[1] / 4 - fuel_tuple[2] / 2) / root[1]
