import joblib
import numpy as np

# def estimate_maf(rpm, temp, pressure, cc):
#     """Estimate the mass aif flow"""

#     # volumetric efficiency constant
#     VE = 0.75
#     # ideal gas constant
#     R = 8.3144621
#     # molar mass of dry air
#     M_air = 28.9647 #g/mol
    
#     # intake pressure from KPa to Pa
#     pressure = pressure/1000
#     # from C to K
#     temp = temp + 273.15 
    
#     # multiply by 1000 to convert from kg/s to g/s
#     return (pressure * cc * M_air * VE * rpm)/(R * temp * 120)

def _get_first(d, *names, default=None):
    for n in names:
        if n in d and d[n] is not None:
            return d[n]
    return default

def estimate_maf(
    rpm: float,
    intake_temp_c: float,
    intake_pressure_kpa: float,
    displacement_l: float = 1.6,
    ve: float = 0.85
) -> float | None:
    """
    Estimate Mass Air Flow (MAF) in grams per second (g/s) using the Speed-Density method.

    This method is used as a fallback when a MAF sensor is not available.
    It relies on the Ideal Gas Law and engine volumetric parameters.

    Formula (for a 4-stroke engine):
        MAF = (VE * MAP * Vd * RPM) / (R_air * T * 2)

    where:
        VE       = Volumetric Efficiency (0–1)
        MAP      = Manifold Absolute Pressure (Pa)
        Vd       = Engine Displacement (m³)
        RPM      = Engine revolutions per minute
        R_air    = Specific gas constant for air (287.058 J/kg·K)
        T        = Intake Air Temperature (Kelvin)
        Division by 2 accounts for the 4-stroke cycle (intake once every 2 revolutions).

    Args:
        rpm (float): Engine speed in revolutions per minute.
        intake_temp_c (float): Intake air temperature in Celsius.
        intake_pressure_kpa (float): Manifold absolute pressure in kilopascals (kPa).
        displacement_l (float): Engine displacement in liters (default = 1.6 L).
        ve (float): Volumetric efficiency (default = 0.85).

    Returns:
        float | None: Estimated MAF in grams per second (g/s), or None if data is insufficient.
    """
    try:
        if rpm is None or intake_temp_c is None or intake_pressure_kpa is None:
            return None

        R_air = 287.058  # J/(kg·K)
        T_K = intake_temp_c + 273.15
        MAP_Pa = intake_pressure_kpa * 1000.0
        Vd_m3 = displacement_l / 1000.0  # liters → cubic meters

        maf_kg_s = (ve * MAP_Pa * Vd_m3 * rpm) / (R_air * T_K * 2.0)
        maf_g_s = maf_kg_s * 1000.0  # convert to g/s
        return max(0.0, maf_g_s)

    except Exception:
        return None

def calc_emission_rate(maf, fuel_type):
    """Calculate the emission rate in g/s"""
    
    # constants for gasoline or alcohol fuel
    co2_per_liter = 2310 if fuel_type=='gasoline' else 1510       #g/L
    air_fuel_ratio = 14.7 if fuel_type=='gasoline' else 9.0
    density = 737 if fuel_type=='gasoline' else 789 #g/L

    if maf is None:
        maf = 0
        
    rate = (maf*co2_per_liter)/(air_fuel_ratio*density) #g/s
    return rate


def convert_emission_rate(emission, speed):
    """Transform emission rate from g/s to g/km"""
    # km/h to km/s
    speed= speed* 0.000277778
    
    if speed == 0:
        speed = 0.1
    return emission/speed #g/km

def calculate_emissions_maf_afr(dados):
    # if maf not in dados
    if "maf" not in dados:
        print("calculating without maf")
        dados["maf"] = estimate_maf(dados["rpm"], dados["intake_temp"], dados["intake_pressure"], dados["cc"])

    # calculate emission rate
    dados["co2_emission"] = calc_emission_rate(dados["maf"], dados["fuel_type"])

    # calculate emission rate per km
    dados["co2_emission_per_km"] = convert_emission_rate(dados["co2_emission"], dados["speed"])

    return dados