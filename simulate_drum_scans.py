"""
simulate_drum_scans.py
======================

Simulates 4 months of line-laser scans of the shell of a dry magnetic drum
separator while the shell slowly wears, and writes them to a raw CSV file
that looks like a real plant data-logger export.

One row  = one laser profile, summarised, every 5 minutes.
Target   = sim_true_min_thickness_mm (only a simulation can know it).

What is simulated
-----------------
Process  : drum speed, feed rate, ore type, planned and unplanned stops
Wear     : Archard-type law, wear per hour ~ k * abrasiveness * load * speed
Signals  : mean and maximum distance laser -> shell surface, laser intensity,
           share of valid profile points, vibration, temperatures, humidity
Noise    : eccentricity and out-of-roundness, thermal expansion, dirty laser
           window, moisture, material build-up, reflections, vibration jitter,
           sensor remount offset, sensor resolution
Faults   : power outage, laser error codes (-999), frozen sensor, lost rows,
           duplicated rows, vibration sensor dropouts, wrong temperature unit,
           date format change after a firmware update, humidity stored as text

All machine dimensions and rates are GENERIC values chosen for teaching.
They are not taken from any manufacturer.

Outputs
-------
data/raw/drum_scans_raw.csv                 the "plant export" (sep=';', decimal=',')
data/simulation_truth/simulation_events.csv answer key: every injected event

Run from the repository root:   python src/simulate_drum_scans.py
Requires: numpy, pandas
"""

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Settings you may want to change
# ---------------------------------------------------------------------------
SEED = 42                    # same seed -> exactly the same file every time
SENSOR = "gocator"           # "gocator" or "smart_camera" (see SENSORS below)

START = "2026-02-02 00:00"   # first scan (local time)
END = "2026-06-01 23:55"     # last scan  (local time)
TZ = "Europe/Vienna"         # logger runs on local time -> DST jump in March
STEP_MIN = 5                 # minutes between scans
DT_H = STEP_MIN / 60         # the same step in hours

# Drum and sensor geometry (generic)
DRUM_RADIUS_MM = 600.0       # drum diameter 1200 mm
SHELL_T0_MM = 6.0            # new shell thickness
STANDOFF_MM = 250.0          # laser -> shell surface distance, new shell, 20 degC
BRACKET_MM = 250.0           # length of the steel bracket holding the laser
ALPHA_SHELL = 16e-6          # thermal expansion, austenitic stainless (1/K)
ALPHA_BRACKET = 12e-6        # thermal expansion, carbon steel (1/K)
T_REF_C = 20.0               # temperature at which the laser was zeroed

# Wear law: wear rate (mm/h) = K_WEAR * abrasiveness * feed/80 * speed/25
K_WEAR = 6.5e-4              # gives about 0.04-0.10 mm per 100 h
GROOVE_FACTOR = 0.72         # mean wear across the width / wear at the deepest groove

# Ore types: (abrasiveness, nominal feed t/h, drum speed rpm)
ORES = {
    "fine":   (0.8, 60.0, 28.0),
    "coarse": (1.3, 90.0, 22.0),
    "slag":   (2.0, 75.0, 20.0),
}

# Laser options: noise of the mean / max distance (mm) and height resolution (mm)
# Values are assumptions for a comparison, not datasheet numbers.
SENSORS = {
    "gocator":      {"noise_mean": 0.004, "noise_max": 0.012, "resolution": 0.01},
    "smart_camera": {"noise_mean": 0.030, "noise_max": 0.100, "resolution": 0.24},
}

ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = ROOT / "data" / "raw" / "drum_scans_raw.csv"
EVENTS_FILE = ROOT / "data" / "simulation_truth" / "simulation_events.csv"

LASER_COLS = ["laser_intensity", "valid_points_pct", "dist_mean_mm", "dist_max_mm"]


# ---------------------------------------------------------------------------
# 2. Small helpers
# ---------------------------------------------------------------------------
def add_event(events, kind, start, end, details=""):
    """Append one injected event to the answer key."""
    events.append({"event": kind,
                   "start": start.strftime("%Y-%m-%d %H:%M"),
                   "end": end.strftime("%Y-%m-%d %H:%M"),
                   "details": details})


def pick_episodes(rng, free, n_episodes, min_rows, max_rows, gap=12):
    """Pick random episodes that do not overlap each other.

    free      : boolean array, True where an episode is allowed
    gap       : rows kept free around each episode (12 rows = 1 h)
    Returns the episodes as a sorted list of (start, stop) row positions
    (stop is not included) and an updated copy of `free`.
    """
    free = free.copy()
    episodes = []
    tries = 0
    while len(episodes) < n_episodes:
        tries += 1
        if tries > 100_000:
            raise RuntimeError("No room left for more episodes")
        length = int(rng.integers(min_rows, max_rows + 1))
        start = int(rng.integers(0, len(free) - length))
        stop = start + length
        if free[start:stop].all():
            episodes.append((start, stop))
            free[max(0, start - gap):stop + gap] = False
    return sorted(episodes), free


def ar1_noise(rng, n, sigma, phi):
    """Slowly wandering noise (AR(1) process): x[i] = phi*x[i-1] + e[i]."""
    e = rng.normal(0, sigma * np.sqrt(1 - phi**2), n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


# ---------------------------------------------------------------------------
# 3. The plant: stops, ore type, feed, speed
# ---------------------------------------------------------------------------
def simulate_operation(time, rng, events):
    n = len(time)
    hour = time.hour + time.minute / 60
    df = pd.DataFrame({"time": time})
    df["day"] = (time - time[0]).total_seconds() / 86400

    # Planned maintenance: every Tuesday 06:00-12:00, one long stop 06:00-18:00
    major_day = pd.Timestamp("2026-04-14", tz=TZ).date()
    is_tuesday = time.dayofweek == 1
    is_major = np.array([t.date() == major_day for t in time])
    maintenance = is_tuesday & (hour >= 6) & (hour < 12)
    maintenance = maintenance | (is_major & (hour >= 6) & (hour < 18))
    df["maintenance"] = maintenance
    df["major_maintenance"] = is_major & (hour >= 6) & (hour < 18)

    running = ~maintenance

    # Power outage: plant and logger are off for 30 h
    s = int(np.searchsorted(df["day"], 97.3))
    e = s + 360
    running[s:e] = False
    df["power_outage"] = False
    df.loc[s:e - 1, "power_outage"] = True
    add_event(events, "power_outage_rows_missing", time[s], time[e - 1],
              "plant stopped and no rows logged")

    # Unplanned stops (blockages, belt trips): about one every 4 days
    stops, _ = pick_episodes(rng, running, 30, 3, 96)
    for s, e in stops:
        running[s:e] = False
        add_event(events, "unplanned_stop", time[s], time[e - 1],
                  f"{(e - s) * STEP_MIN} min")
    df["running"] = running

    # Ore campaigns of 5-12 days
    ore = np.empty(n, dtype=object)
    i = 0
    previous = None
    while i < n:
        choices = [o for o in ORES if o != previous]
        name = rng.choice(choices)
        length = int(rng.uniform(5, 12) * 288)
        ore[i:i + length] = name
        add_event(events, "ore_campaign", time[i], time[min(i + length, n) - 1], name)
        previous = name
        i += length
    df["ore_type"] = ore

    abrasiveness = np.array([ORES[o][0] for o in ore])
    feed_nominal = np.array([ORES[o][1] for o in ore])
    speed_set = np.array([ORES[o][2] for o in ore])

    feed = feed_nominal * (1 + ar1_noise(rng, n, 0.06, 0.995))
    feed = feed + rng.normal(0, 1.0, n)
    speed = speed_set + rng.normal(0, 0.15, n)

    # Short feed interruptions: drum turns, no material on it
    interruptions, _ = pick_episodes(rng, running, 15, 2, 18)
    for s, e in interruptions:
        feed[s:e] = 0.0
        add_event(events, "feed_interruption", time[s], time[e - 1], "drum runs empty")

    feed[~running] = 0.0
    speed[~running] = 0.0
    df["feed_rate_t_h"] = np.clip(feed, 0, None)
    df["drum_speed_rpm"] = np.clip(speed, 0, None)
    df["abrasiveness"] = abrasiveness
    return df


# ---------------------------------------------------------------------------
# 4. Environment: hall temperature, humidity, shell temperature
# ---------------------------------------------------------------------------
def simulate_environment(df, rng, events):
    n = len(df)
    hour = df["time"].dt.hour + df["time"].dt.minute / 60
    day = df["day"].to_numpy()

    # Hall temperature: cold February -> mild June, warmest at 15:00
    ambient = 6 + 14 * day / day.max() + 3 * np.sin(2 * np.pi * (hour - 9) / 24)
    ambient = ambient.to_numpy() + ar1_noise(rng, n, 0.8, 0.99)

    # Relative humidity: higher at night, plus rainy spells of 1-3 days
    humidity = 58 + 8 * np.cos(2 * np.pi * (hour - 5) / 24).to_numpy()
    humidity = humidity + ar1_noise(rng, n, 4, 0.995)
    wet_spells, _ = pick_episodes(rng, np.ones(n, bool), 12, 288, 864, gap=288)
    for s, e in wet_spells:
        ramp = np.sin(np.linspace(0, np.pi, e - s))    # rises and falls smoothly
        humidity[s:e] += 30 * ramp
        add_event(events, "wet_period", df["time"][s], df["time"][e - 1], "high humidity")
    humidity = np.clip(humidity, 25, 98)

    # Shell temperature: eddy currents warm the rotating shell; it follows
    # with a time constant of 1.5 h (first-order lag)
    speed = df["drum_speed_rpm"].to_numpy()
    target = ambient + 16 * (speed / 25) ** 1.5
    shell = np.zeros(n)
    shell[0] = ambient[0]
    lag = 1 - np.exp(-DT_H / 1.5)
    for i in range(1, n):
        shell[i] = shell[i - 1] + (target[i] - shell[i - 1]) * lag
    shell = shell + rng.normal(0, 0.15, n)

    df["ambient_temp_c"] = ambient
    df["humidity_pct"] = humidity
    df["shell_temp_c"] = shell
    return df


# ---------------------------------------------------------------------------
# 5. Wear of the shell (the quantity we want to monitor)
# ---------------------------------------------------------------------------
def simulate_wear(df, rng):
    rate = (K_WEAR * df["abrasiveness"] * df["feed_rate_t_h"] / 80
            * df["drum_speed_rpm"] / 25)
    # Wear does not grow perfectly smoothly: random factor with mean 1
    increment = rate.to_numpy() * DT_H * rng.gamma(4.0, 1 / 4.0, len(df))
    df["wear_peak_mm"] = np.cumsum(increment)
    df["wear_mean_mm"] = GROOVE_FACTOR * df["wear_peak_mm"]
    df["sim_true_min_thickness_mm"] = SHELL_T0_MM - df["wear_peak_mm"]
    return df


# ---------------------------------------------------------------------------
# 6. Dirt on the laser window and vibration
# ---------------------------------------------------------------------------
def simulate_dirt(df, rng, events):
    """Dust builds up on the window while running (faster when dry).

    The window is cleaned at the start of planned maintenance, but the
    cleaning is sometimes forgotten.
    """
    n = len(df)
    dirt = np.zeros(n)
    maint = df["maintenance"].to_numpy()
    running = df["running"].to_numpy()
    humidity = df["humidity_pct"].to_numpy()
    feed = df["feed_rate_t_h"].to_numpy()
    for i in range(1, n):
        dirt[i] = dirt[i - 1]
        if running[i]:
            dust = np.clip((75 - humidity[i]) / 40, 0.3, 1.5)
            dirt[i] += 0.0035 * DT_H * dust * (feed[i] / 80 + 0.2)
        maintenance_starts = maint[i] and not maint[i - 1]
        if maintenance_starts:
            if df["major_maintenance"].iloc[i] or rng.random() < 0.8:
                dirt[i] = rng.uniform(0, 0.03)
            else:
                add_event(events, "window_cleaning_skipped", df["time"][i],
                          df["time"][i], "dirt keeps building up")
    df["dirt_level"] = np.clip(dirt, 0, 1)
    return df


def simulate_vibration(df, rng, events):
    """Vibration velocity RMS at the laser bracket (mm/s).

    A bearing slowly deteriorates from day 52 and is replaced during the
    major maintenance.
    """
    speed = df["drum_speed_rpm"].to_numpy()
    feed = df["feed_rate_t_h"].to_numpy()
    running = df["running"].to_numpy()
    day = df["day"].to_numpy()

    vib = 1.3 * speed / 25 + 0.004 * feed + rng.normal(0, 0.2, len(df))
    vib[~running] = 0.15 + rng.normal(0, 0.03, (~running).sum())

    major = df.index[df["major_maintenance"]][0]
    major_day = day[major]
    bearing = (day >= 52) & (day < major_day)
    growth = 1 + 4 * ((day[bearing] - 52) / (major_day - 52)) ** 2
    vib[bearing] = vib[bearing] * growth
    add_event(events, "bearing_deterioration", df["time"][np.argmax(bearing)],
              df["time"][major - 1], "vibration grows up to about 5x, fixed at major maintenance")
    df["vibration_rms_mm_s"] = np.clip(vib, 0.05, None)
    return df


# ---------------------------------------------------------------------------
# 7. Laser signals: true geometry + every kind of disturbance
# ---------------------------------------------------------------------------
def simulate_laser(df, rng, events):
    n = len(df)
    sensor = SENSORS[SENSOR]
    running = df["running"].to_numpy()

    # Drum angle at the moment of the scan (encoder). Scans are not
    # synchronised with the rotation, so the angle is random while running
    # and frozen while the drum stands still.
    angle = rng.uniform(0, 360, n)
    for i in range(1, n):
        if not running[i]:
            angle[i] = angle[i - 1]
    rad = np.deg2rad(angle)
    eccentricity = 0.10 * np.cos(rad - np.deg2rad(40))          # 0.10 mm run-out
    out_of_round = 0.035 * np.cos(2 * rad - np.deg2rad(110))     # oval shell
    geometry = eccentricity + out_of_round

    # Thermal expansion: a warm shell grows towards the laser, a warm bracket
    # moves the laser towards the shell. Both shorten the measured distance.
    thermal = -(ALPHA_SHELL * DRUM_RADIUS_MM * (df["shell_temp_c"] - T_REF_C)
                + ALPHA_BRACKET * BRACKET_MM * (df["ambient_temp_c"] - T_REF_C))
    thermal = thermal.to_numpy()

    # Dirt on the window: longer apparent distance, more scatter
    dirt = df["dirt_level"].to_numpy()

    # Moisture: water film on the shell from about 82 % RH
    wet = np.clip((df["humidity_pct"].to_numpy() - 82) / 14, 0, 1)

    # Material build-up on the shell (more likely with fine ore when wet),
    # 0.2-0.8 mm thick, lasts 1-8 h until it drops off
    layer = np.zeros(n)
    sticky = (df["ore_type"].to_numpy() == "fine") & (wet > 0.2)
    buildups, _ = pick_episodes(rng, sticky, 10, 12, 96, gap=24)
    for s, e in buildups:
        layer[s:e] = rng.uniform(0.2, 0.8)
        add_event(events, "material_buildup", df["time"][s], df["time"][e - 1],
                  f"{layer[s]:.2f} mm layer")

    # Sensor remounted during the major maintenance: new zero point
    major = df.index[df["major_maintenance"]][0]
    offset = np.zeros(n)
    offset[major:] = 0.35
    add_event(events, "sensor_remounted", df["time"][major], df["time"][major],
              "+0.35 mm offset from here on")

    # Vibration jitter: displacement RMS = velocity RMS / (2*pi*f), f ~ 25 Hz
    jitter = df["vibration_rms_mm_s"].to_numpy() / (2 * np.pi * 25)
    sd_mean = np.sqrt((sensor["noise_mean"] * (1 + 3 * dirt)) ** 2 + jitter**2)
    sd_max = np.sqrt((sensor["noise_max"] * (1 + 3 * dirt)) ** 2 + jitter**2)

    dist_mean = (STANDOFF_MM + df["wear_mean_mm"].to_numpy() + geometry + thermal
                 + 0.12 * dirt + offset - layer + rng.normal(0, 1, n) * sd_mean)
    dist_max = (STANDOFF_MM + 0.08 + df["wear_peak_mm"].to_numpy() + geometry + thermal
                + 0.25 * dirt + offset - 0.5 * layer + rng.normal(0, 1, n) * sd_max)

    # Reflections from a wet or shiny surface: single very long readings
    p_reflection = 0.002 + 0.03 * wet + 0.001 * df["vibration_rms_mm_s"].to_numpy()
    hit = rng.random(n) < p_reflection
    dist_max[hit] += rng.uniform(1, 6, hit.sum())

    # Particle passing through the laser line: shorter mean distance
    hit = rng.random(n) < 0.002
    dist_mean[hit] -= rng.uniform(0.5, 3, hit.sum())

    # Height resolution of the sensor
    dist_max = np.round(dist_max / sensor["resolution"]) * sensor["resolution"]

    valid = 99.3 - 55 * dirt**1.3 - 20 * wet - 3 * layer + rng.normal(0, 0.6, n)
    intensity = 820 * (1 - 0.55 * dirt) * (1 - 0.35 * wet) + rng.normal(0, 15, n)

    df["drum_angle_deg"] = angle
    df["dist_mean_mm"] = dist_mean
    df["dist_max_mm"] = dist_max
    df["valid_points_pct"] = np.clip(valid, 0, 100)
    df["laser_intensity"] = np.clip(intensity, 0, None)
    return df


# ---------------------------------------------------------------------------
# 8. The data logger: everything that goes wrong between sensor and file
# ---------------------------------------------------------------------------
def build_raw_export(df, rng, events):
    n = len(df)
    raw = df.copy()
    t = raw["time"]

    # Machine hour counter of the PLC: hours run before this scan
    running = raw["running"].to_numpy()
    raw["operating_hours"] = 18430 + (np.cumsum(running) - running) * DT_H

    # Manual ultrasonic gauge reading once per planned maintenance (09:00)
    is_reading = raw["maintenance"] & (t.dt.hour == 9) & (t.dt.minute == 0)
    raw["ut_ref_thickness_mm"] = np.nan
    raw.loc[is_reading, "ut_ref_thickness_mm"] = (
        raw.loc[is_reading, "sim_true_min_thickness_mm"]
        + rng.normal(0, 0.04, is_reading.sum()))

    # Rows that never reach the file: power outage and lost logger buffers.
    # They are chosen first so that no other fault is hidden inside them.
    outage = raw["power_outage"].to_numpy()
    lost, free = pick_episodes(rng, ~outage, 6, 4, 60)

    # Laser reports an error code
    errors, free = pick_episodes(rng, free, 8, 2, 36)
    for s, e in errors:
        raw.loc[s:e - 1, LASER_COLS] = -999
        add_event(events, "laser_error_code", t[s], t[e - 1], "laser columns = -999")

    # Laser hangs and repeats its last value
    frozen, free = pick_episodes(rng, free, 4, 12, 36)
    for s, e in frozen:
        for col in LASER_COLS:
            raw.loc[s:e - 1, col] = raw.loc[s, col]
        add_event(events, "laser_frozen", t[s], t[e - 1], "laser columns repeat one value")

    # Vibration sensor: loose cable -> empty cells; once stuck at 0.0 while
    # the drum is running (a real reading is never 0.0 then)
    vib_free = ~outage
    for s, e in lost:
        vib_free[max(0, s - 12):e + 12] = False
    missing, vib_free = pick_episodes(rng, vib_free, 3, 24, 120)
    for s, e in missing:
        raw.loc[s:e - 1, "vibration_rms_mm_s"] = np.nan
        add_event(events, "vibration_missing", t[s], t[e - 1], "empty cells")
    stuck, _ = pick_episodes(rng, vib_free & running, 1, 60, 60)
    s, e = stuck[0]
    raw.loc[s:e - 1, "vibration_rms_mm_s"] = 0.0
    add_event(events, "vibration_stuck_zero", t[s], t[e - 1], "reads 0.0")

    # Replacement thermometer configured in Fahrenheit for about one week
    wrong_unit = (raw["day"] >= 88) & (raw["day"] < 95)
    raw.loc[wrong_unit, "shell_temp_c"] = raw.loc[wrong_unit, "shell_temp_c"] * 9 / 5 + 32
    add_event(events, "shell_temp_in_fahrenheit", t[wrong_unit].iloc[0],
              t[wrong_unit].iloc[-1], "values are degF, column says degC")

    # Rounding as the logger would store the values
    decimals = {"drum_speed_rpm": 2, "feed_rate_t_h": 1, "ambient_temp_c": 1,
                "shell_temp_c": 1, "humidity_pct": 1, "vibration_rms_mm_s": 2,
                "drum_angle_deg": 1, "laser_intensity": 0, "valid_points_pct": 1,
                "dist_mean_mm": 4, "dist_max_mm": 3, "ut_ref_thickness_mm": 2,
                "operating_hours": 2, "sim_true_min_thickness_mm": 4}
    raw = raw.round(decimals)

    # Humidity sensor sends text such as "63,4 %" and "--" when it fails
    humidity_text = raw["humidity_pct"].map(lambda v: f"{v:.1f} %".replace(".", ","))
    broken = rng.random(n) < 0.004
    humidity_text[broken] = "--"
    raw["humidity_pct"] = humidity_text

    # Timestamps: firmware update on day 40 changes the date format for 2 weeks
    firmware = (raw["day"] >= 40) & (raw["day"] < 54)
    raw["timestamp"] = np.where(firmware,
                                t.dt.strftime("%d.%m.%Y %H:%M:%S"),
                                t.dt.strftime("%Y-%m-%d %H:%M:%S"))
    add_event(events, "date_format_changed", t[firmware].iloc[0], t[firmware].iloc[-1],
              "dd.mm.yyyy HH:MM:SS")
    raw["scan_id"] = np.arange(1, n + 1)

    # Row order of the file: lost rows, power outage, re-sent duplicates
    order = []
    position = 0
    for s, e in lost:
        add_event(events, "rows_lost", t[s], t[e - 1], "logger buffer lost")
        order.extend(range(position, s))          # rows before the gap
        n_dup = int(rng.integers(6, 13))
        order.extend(range(s - n_dup, s))         # logger re-sends old rows
        add_event(events, "rows_duplicated", t[s - n_dup], t[s - 1],
                  f"{n_dup} rows re-sent after the gap")
        position = e                              # skip the lost rows
    order.extend(range(position, n))
    order = [i for i in order if not outage[i]]
    raw = raw.iloc[order]

    columns = ["timestamp", "scan_id", "operating_hours", "ore_type",
               "drum_speed_rpm", "feed_rate_t_h", "ambient_temp_c", "shell_temp_c",
               "humidity_pct", "vibration_rms_mm_s", "drum_angle_deg",
               "laser_intensity", "valid_points_pct", "dist_mean_mm", "dist_max_mm",
               "ut_ref_thickness_mm", "sim_true_min_thickness_mm"]
    return raw[columns]


# ---------------------------------------------------------------------------
# 9. Run everything
# ---------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(SEED)
    events = []
    time = pd.date_range(START, END, freq=f"{STEP_MIN}min", tz=TZ)

    df = simulate_operation(time, rng, events)
    df = simulate_environment(df, rng, events)
    df = simulate_wear(df, rng)
    df = simulate_dirt(df, rng, events)
    df = simulate_vibration(df, rng, events)
    df = simulate_laser(df, rng, events)
    raw = build_raw_export(df, rng, events)

    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    EVENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RAW_FILE, sep=";", decimal=",", index=False)
    pd.DataFrame(events).sort_values("start").to_csv(EVENTS_FILE, index=False)

    print(f"Sensor: {SENSOR}")
    print(f"Wrote {len(raw):,} rows x {raw.shape[1]} columns to {RAW_FILE}")
    print(f"Wrote {len(events)} events to {EVENTS_FILE}")
    print(f"Shell thickness: {SHELL_T0_MM:.2f} mm -> "
          f"{df['sim_true_min_thickness_mm'].iloc[-1]:.2f} mm "
          f"after {df['running'].sum() * DT_H:,.0f} running hours")


if __name__ == "__main__":
    main()
