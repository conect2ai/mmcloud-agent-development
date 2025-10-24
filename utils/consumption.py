# consumption.py

from __future__ import annotations
from typing import Optional
from utils.emissions import estimate_maf

# -------------------------------
# Volumetric efficiency curves
# -------------------------------

def volumetric_efficiency_1_0L(engine_rpm: float, is_rpm_gt_3584: int) -> float:
    """
    Piecewise-linear VE curve for 1.0L engines (returns VE as FRACTION 0..1).

    The original formula returns a percentage-like value which may exceed 1.
    Here we normalize to a fraction by dividing by 100 if needed.
    """
    ve = (((0.0025 * engine_rpm) + 85) * abs(is_rpm_gt_3584 - 1)) + (((-0.0025 * engine_rpm) + 110) * is_rpm_gt_3584)
    return ve / 100.0 if ve > 1.0 else ve

def volumetric_efficiency_1_6L(engine_rpm: float, is_rpm_gt_3584: int) -> float:
    """
    Piecewise-linear VE curve for 1.6L engines (returns VE as FRACTION 0..1).
    """
    ve = (((0.004 * engine_rpm) + 85) * abs(is_rpm_gt_3584 - 1)) + (((-0.004 * engine_rpm) + 110) * is_rpm_gt_3584)
    return ve / 100.0 if ve > 1.0 else ve

def volumetric_efficiency_2_0L(engine_rpm: float, is_rpm_gt_3584: int) -> float:
    """
    Piecewise-linear VE curve for 2.0L engines (returns VE as FRACTION 0..1).
    """
    ve = (((0.005 * engine_rpm) + 81) * abs(is_rpm_gt_3584 - 1)) + (((-0.005 * engine_rpm) + 117) * is_rpm_gt_3584)
    return ve / 100.0 if ve > 1.0 else ve


def _ve_from_displacement(rpm: Optional[float], vdm_l: Optional[float]) -> Optional[float]:
    """
    Pick a VE (fraction 0..1) based on engine displacement and RPM.
    Falls back to a reasonable default if displacement is unknown.
    """
    if rpm is None:
        return 0.85  # generic fallback

    is_rpm_gt_3584 = 1 if rpm > 3584 else 0
    if vdm_l is None:
        return 0.85

    if abs(vdm_l - 1.0) < 1e-6:
        return volumetric_efficiency_1_0L(rpm, is_rpm_gt_3584)
    if abs(vdm_l - 1.6) < 1e-6:
        return volumetric_efficiency_1_6L(rpm, is_rpm_gt_3584)
    if abs(vdm_l - 2.0) < 1e-6:
        return volumetric_efficiency_2_0L(rpm, is_rpm_gt_3584)

    # Unknown displacement: use a safe default
    return 0.85

# -------------------------------
# Instant fuel consumption
# -------------------------------

def instant_fuel_consumption(
    vss_kmh: float,
    rpm: float | None = None,
    map_value: float | None = None,
    iat: float | None = None,
    vdm: float | None = None,
    maf: float | None = None,
    mm: float = 28.97,
    r: float = 8.314,
    combustivel: str = 'Gasoline'
) -> float:
    """
    Compute instant fuel economy in kilometers per liter (km/L).

    This function supports two modes:
      1) Direct MAF: if 'maf' (g/s) is provided, it uses it directly.
      2) Speed-Density fallback: if MAF is missing and (rpm, MAP, IAT, displacement) are provided,
         it estimates MAF using the ideal gas + volumetric efficiency model.

    Args:
        vss_kmh: Vehicle speed in km/h.
        rpm: Engine speed in RPM. Required only if MAF is absent.
        map_value: Manifold absolute pressure (MAP) in kPa. Required only if MAF is absent.
        iat: Intake air temperature. Accepts °C or °K.
             If iat < 200, it's assumed to be Celsius and converted to Kelvin internally.
        vdm: Engine displacement in liters (e.g., 1.0, 1.6, 2.0). Required only if MAF is absent.
        maf: Mass air flow in g/s. If provided, the function will not estimate MAF.
        mm: Mean molecular mass of air (kept for backward-compatibility; not used in the new SD path).
        r: Universal gas constant (kept for backward-compatibility; not used in the new SD path).
        combustivel: 'Gasoline' or 'Ethanol' (case-insensitive).

    Returns:
        Instant fuel economy in km/L.

    Raises:
        ValueError: If neither MAF nor the required SD parameters (rpm, map_value, iat, vdm) are provided.
    """
    # Normalize fuel string
    fuel = (combustivel or "Gasoline").strip().capitalize()
    if fuel not in ("Gasoline", "Ethanol"):
        raise ValueError("Invalid fuel type. Use 'Gasoline' or 'Ethanol'.")

    # Convert speed to mph (for the MPG constants you already use)
    vss_mih = (vss_kmh or 0.0) / 1.60934
    if vss_mih == 0.0:
        # Avoid div-by-zero MPG; a tiny floor helps continuity when the car stops.
        vss_mih = 0.1

    # If MAF is missing, try to estimate it via speed-density
    if maf is None:
        if rpm is None or map_value is None or iat is None or vdm is None:
            raise ValueError("MAF or (RPM, MAP, IAT, and VDM) are required to compute fuel consumption.")

        # Detect IAT units: assume °C if < 200, else it's already Kelvin
        if iat < 200.0:
            iat_c = iat
        else:
            iat_c = iat - 273.15

        # Volumetric efficiency (fraction 0..1) from your displacement-specific curves
        ve = _ve_from_displacement(rpm, vdm)

        # Estimate MAF in g/s
        maf = estimate_maf(
            rpm=float(rpm),
            intake_temp_c=float(iat_c),
            intake_pressure_kpa=float(map_value),
            displacement_l=float(vdm),
            ve=float(ve)
        )

    # Fuel-specific MPG constant (as in your original code)
    # These constants map mpg = C * (mph / maf_gps)
    if fuel == 'Gasoline':
        C = 7.107
    else:  # 'Ethanol'
        C = 8.56984

    # Guard against zero/negative maf
    maf_gps = float(maf or 0.0)
    if maf_gps <= 0.0:
        maf_gps = 0.1  # small floor to avoid division by zero

    mpg = C * (vss_mih / maf_gps)
    km_per_l = mpg * 0.4251  # mpg → km/L
    return km_per_l