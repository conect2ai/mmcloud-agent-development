# utils/heading.py
def update_heading_deg(prev_deg: float, gyro_z_dps: float, dt_s: float) -> float:
    return (prev_deg + gyro_z_dps * dt_s) % 360.0

def heading_deg_to_cardinal_pt(deg: float) -> str:
    idx = int((deg + 22.5) // 45) % 8
    return ["N", "N", "L", "L", "S", "S", "O", "O"][idx]