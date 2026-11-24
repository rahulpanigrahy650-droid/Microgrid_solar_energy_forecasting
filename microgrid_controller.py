from dataclasses import dataclass


@dataclass
class BatteryState:
    capacity_kwh: float = 13.5      # e.g. one Tesla-Powerwall-sized unit
    soc_kwh: float = 6.75           # current stored energy (starts at 50%)
    max_charge_kw: float = 5.0
    max_discharge_kw: float = 5.0
    min_soc_pct: float = 10.0       # never fully drain the battery
    max_soc_pct: float = 100.0

    @property
    def soc_pct(self) -> float:
        return round(100 * self.soc_kwh / self.capacity_kwh, 1)


def decide(solar_kw: float, load_kw: float, battery: BatteryState, dt_hours: float = 1.0) -> dict:
    """
    Given one time-step of forecast solar generation and load demand,
    decide the micro-grid action and update the battery state in place.

    Returns a dict describing the decision (used by the API + dashboard).
    """
    net_kw = solar_kw - load_kw  # positive => surplus, negative => deficit
    min_soc_kwh = battery.capacity_kwh * battery.min_soc_pct / 100
    max_soc_kwh = battery.capacity_kwh * battery.max_soc_pct / 100

    action = "direct_supply"
    grid_import_kw = 0.0
    battery_flow_kw = 0.0  # positive = charging, negative = discharging

    if net_kw > 0.05:
        # Surplus solar -> charge battery with the extra
        chargeable = min(net_kw, battery.max_charge_kw)
        room_kwh = max_soc_kwh - battery.soc_kwh
        actual_charge_kw = min(chargeable, room_kwh / dt_hours) if dt_hours > 0 else 0
        battery.soc_kwh = min(max_soc_kwh, battery.soc_kwh + actual_charge_kw * dt_hours)
        battery_flow_kw = actual_charge_kw
        action = "charging" if actual_charge_kw > 0.05 else "direct_supply"

    elif net_kw < -0.05:
        # Shortfall -> try the battery first, then the grid
        deficit_kw = -net_kw
        available_kwh = battery.soc_kwh - min_soc_kwh
        dischargeable_kw = min(deficit_kw, battery.max_discharge_kw, available_kwh / dt_hours if dt_hours > 0 else 0)
        dischargeable_kw = max(dischargeable_kw, 0)
        battery.soc_kwh = max(min_soc_kwh, battery.soc_kwh - dischargeable_kw * dt_hours)
        battery_flow_kw = -dischargeable_kw

        remaining_deficit = deficit_kw - dischargeable_kw
        if remaining_deficit > 0.05:
            grid_import_kw = remaining_deficit
            action = "grid_supply" if dischargeable_kw <= 0.05 else "battery_plus_grid"
        else:
            action = "discharging"
    else:
        action = "direct_supply"

    return {
        "action": action,
        "solar_kw": round(solar_kw, 3),
        "load_kw": round(load_kw, 3),
        "net_kw": round(net_kw, 3),
        "battery_flow_kw": round(battery_flow_kw, 3),
        "grid_import_kw": round(grid_import_kw, 3),
        "battery_soc_pct": battery.soc_pct,
        "battery_soc_kwh": round(battery.soc_kwh, 3),
    }


if __name__ == "__main__":
    # Quick smoke test simulating one synthetic day
    import numpy as np

    battery = BatteryState()
    hours = np.arange(24)
    solar_profile = np.clip(6.5 * np.exp(-((hours - 12.5) ** 2) / 10), 0, None)
    load_profile = 3 + 2 * np.exp(-((hours - 8) ** 2) / 4) + 3 * np.exp(-((hours - 19.5) ** 2) / 6)

    print(f"{'hr':>3} {'solar':>7} {'load':>7} {'action':>16} {'batt%':>7} {'grid':>6}")
    for h, s, l in zip(hours, solar_profile, load_profile):
        result = decide(s, l, battery)
        print(f"{h:>3} {s:>7.2f} {l:>7.2f} {result['action']:>16} {result['battery_soc_pct']:>7.1f} {result['grid_import_kw']:>6.2f}")
