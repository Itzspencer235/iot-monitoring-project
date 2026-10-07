#!/usr/bin/env python3
"""
IoT Temperature & Humidity Monitor + LED Control + Linear Regression model.
Everything in one file: sensor reading, fan/heater LED control, CSV logging,
and a simple AI model (Linear Regression) that predicts the temperature trend.

Wiring (BCM numbering):
    DHT11 data -> GPIO4
    Fan LED    -> GPIO17 (through a ~330 ohm resistor)
    Heater LED -> GPIO27 (through a ~330 ohm resistor)

Fan LED turns ON when temperature >= MAX_TEMP.
Heater LED turns ON when temperature <= MIN_TEMP.
Both turn OFF once the temperature is back between MIN_TEMP and MAX_TEMP.

Every PREDICT_EVERY readings, the script fits a Linear Regression line
(T = m*t + c) on the data collected so far, prints the equation, and saves
a graph (trend.png) showing the measured points, the regression line and a
short forecast.

Run with SIMULATE=1 to test on a PC without any hardware:
    SIMULATE=1 python3 monitor.py
"""
import csv
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

SIMULATE = os.getenv("SIMULATE", "0") == "1"

# --- ThingSpeak (cloud) ---
THINGSPEAK_WRITE_KEY = "HNHUJE5C8DCHIVKV"
THINGSPEAK_CHANNEL_ID = "3518851"

# --- Thresholds: change these two values to whatever your project needs ---
MIN_TEMP = 30    # degrees C - at or below this, the heater LED turns ON
MAX_TEMP =    # degrees C - at or above this, the fan LED turns ON

INTERVAL = int(os.getenv("INTERVAL_S", "15"))        # seconds between readings
PREDICT_EVERY = int(os.getenv("PREDICT_EVERY", "10"))  # run the AI model every N readings
FORECAST_MIN = float(os.getenv("FORECAST_MIN", "30"))  # how far ahead to forecast (minutes)

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()

CSV_FILE = HERE / "readings.csv"
GRAPH_FILE = HERE / "trend.png"


# ------------------------------- hardware setup -----------------------------
class DummyLed:
    def __init__(self, name):
        self.name = name
        self.value = False

    def on(self):
        self.value = True

    def off(self):
        self.value = False


if SIMULATE:
    dht = None
    fan_led = DummyLed("fan")
    heater_led = DummyLed("heater")
    print("SIMULATE=1: using random readings, no real hardware")
else:
    import adafruit_dht
    import board
    from gpiozero import LED

    dht = adafruit_dht.DHT11(board.D4, use_pulseio=False)
    fan_led = LED(17)
    heater_led = LED(27)


def read_sensor():
    """Return (temp_c, humidity_pct) or None if the read failed."""
    if SIMULATE:
        import random
        return round(random.uniform(20, 32), 1), round(random.uniform(40, 70), 1)
    for _ in range(5):
        try:
            t, h = dht.temperature, dht.humidity
            if t is not None and h is not None:
                return float(t), float(h)
        except (RuntimeError, OverflowError):
            pass  # DHT11 misses a read or hits a timing overflow, just retry
        time.sleep(2)
    return None


def control(temp):
    """Fan LED on at/above MAX_TEMP, heater LED on at/below MIN_TEMP, both off between."""
    if temp >= MAX_TEMP:
        heater_led.off()
        fan_led.on()
    elif temp <= MIN_TEMP:
        fan_led.off()
        heater_led.on()
    else:
        fan_led.off()
        heater_led.off()


# --------------------------------- cloud --------------------------------------
def send_to_cloud(temp, hum, predicted_temp=None):
    """Upload one reading (and optionally the latest AI prediction) to ThingSpeak."""
    if not THINGSPEAK_WRITE_KEY:
        return False
    params = {
        "api_key": THINGSPEAK_WRITE_KEY,
        "field1": temp,
        "field2": hum,
        "field3": int(bool(fan_led.value)),
        "field4": int(bool(heater_led.value)),
    }
    if predicted_temp is not None:
        params["field5"] = round(predicted_temp, 2)    # enable Field 5 = "Predicted Temp" on ThingSpeak
    try:
        r = requests.get("https://api.thingspeak.com/update", params=params, timeout=10)
        return r.ok and r.text.strip() != "0"    # ThingSpeak returns "0" when rejected
    except requests.RequestException as err:
        print(f"[cloud] upload failed: {err}")
        return False


# --------------------------------- logging -----------------------------------
def save_reading(temp, hum):
    new_file = not CSV_FILE.exists()
    with CSV_FILE.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp", "temperature_c", "humidity_pct"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), temp, hum])


def count_rows():
    if not CSV_FILE.exists():
        return 0
    with CSV_FILE.open() as f:
        return max(sum(1 for _ in f) - 1, 0)  # minus header


last_prediction = None  # most recent "temperature in FORECAST_MIN minutes" value


# ------------------------------ AI model (Linear Regression) -----------------
def run_prediction():
    """Fit T = m*t + c on readings.csv, print the equation, save trend.png."""
    global last_prediction
    rows = []
    with CSV_FILE.open() as f:
        for row in csv.DictReader(f):
            rows.append((datetime.fromisoformat(row["timestamp"]), float(row["temperature_c"])))
    if len(rows) < 5:
        return

    rows.sort(key=lambda r: r[0])
    t0 = rows[0][0]
    x = np.array([(t - t0).total_seconds() / 60 for t, _ in rows])    # minutes
    y = np.array([temp for _, temp in rows])

    m, c = np.polyfit(x, y, 1)                       # Linear Regression (degree 1)
    y_pred = m * x + c
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0

    t_future = x[-1] + FORECAST_MIN
    temp_future = m * t_future + c
    last_prediction = temp_future

    print(f"[AI model] T = {m:.4f}*t + {c:.2f}   (R^2={r2:.3f})   "
          f"predicted in {FORECAST_MIN:.0f} min: {temp_future:.2f}C")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.scatter(x, y, s=10, alpha=0.6, label="Measured temperature")
        xs = np.linspace(x.min(), t_future, 100)
        plt.plot(xs, m * xs + c, "r", label=f"T = {m:.4f}t + {c:.2f}")
        plt.axvline(x[-1], color="gray", ls=":", label="Now")
        plt.axhline(MIN_TEMP, color="blue", ls=":", label=f"MIN_TEMP {MIN_TEMP}C")
        plt.axhline(MAX_TEMP, color="orange", ls=":", label=f"MAX_TEMP {MAX_TEMP}C")
        plt.xlabel("Time since first reading (minutes)")
        plt.ylabel("Temperature (C)")
        plt.title("Linear Regression Temperature Trend")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(GRAPH_FILE, dpi=150)
        plt.close()
        print(f"[AI model] graph saved to {GRAPH_FILE}")
    except ImportError:
        print("[AI model] matplotlib not installed, skipping graph (equation above still valid)")


# --------------------------------- main loop ----------------------------------
def main():
    print(f"MIN_TEMP={MIN_TEMP}C  MAX_TEMP={MAX_TEMP}C  interval={INTERVAL}s  "
          f"predicting every {PREDICT_EVERY} readings  (Ctrl+C to stop)")
    try:
        while True:
            reading = read_sensor()
            if reading is None:
                print("Sensor read failed, skipping this cycle")
            else:
                temp, hum = reading
                control(temp)
                save_reading(temp, hum)
                if count_rows() % PREDICT_EVERY == 0:
                    run_prediction()
                cloud_ok = send_to_cloud(temp, hum, last_prediction)
                print(f"T={temp:.1f}C  H={hum:.1f}%  fan_led={'ON' if fan_led.value else 'off'}  "
                      f"heater_led={'ON' if heater_led.value else 'off'}  "
                      f"predicted={f'{last_prediction:.1f}C' if last_prediction is not None else '-'}  "
                      f"cloud={'ok' if cloud_ok else ('skip' if not THINGSPEAK_WRITE_KEY else 'FAIL')}")
            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        pass
    finally:
        fan_led.off()
        heater_led.off()
        if dht is not None:
            dht.exit()
        print("Stopped, LEDs off")


if __name__ == "__main__":
    main()
  
