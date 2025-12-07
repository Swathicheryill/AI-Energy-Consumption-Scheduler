"""
ai_energy_scheduler.py

End-to-end prototype:
- Data loading & preprocessing
- Custom RL environment
- PPO training
- Streamlit dashboard

Usage:
  1) Train RL agent:
       python ai_energy_scheduler.py --train
  2) Run dashboard:
       streamlit run ai_energy_scheduler.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing  import MinMaxScaler

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO

import streamlit as st
import joblib
import os

# =========================
# CONFIG
# =========================

DATA_PATH = Path("data/household_power_consumption.txt")  # Kaggle/UCI file path
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "rl_scheduler_ppo"  # SB3 will save as .zip
SCALER_PATH = MODEL_DIR / "scaler.joblib"


# =========================
# DATA LOADING & PREPROCESSING
# =========================

def load_and_preprocess(resample_to_1min: bool = True):
    """
    Load the UCI Household Electric Power Consumption dataset and preprocess it.
    - Clean missing values.
    - (Optional) Resample to 1-minute resolution to ensure consistent day lengths.
    - Create time features and price signal.
    - Build simple flexible/base load features.
    - Scale features for RL.
    """
    df = pd.read_csv(
        DATA_PATH,
        sep=';',
        na_values='?',
        low_memory=False
    )

    # Combine date and time
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'],
                                    format='%d/%m/%Y %H:%M:%S', errors='coerce')
    df = df.dropna(subset=['datetime']).sort_values('datetime').set_index('datetime')

    # Convert numeric columns
    num_cols = [
        'Global_active_power', 'Global_reactive_power', 'Voltage',
        'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'
    ]
    for c in num_cols:
        # coerce errors -> NaN then we will interpolate
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Basic cleaning: drop rows where active power missing, then interpolate other numeric columns
    df = df.dropna(subset=['Global_active_power'])
    df[num_cols] = df[num_cols].interpolate().fillna(method='bfill').fillna(method='ffill')

    # If requested, resample to exact 1-minute frequency to ensure day slices of consistent length
    if resample_to_1min:
        # create minute index spanning observed range
        full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='T')
        df = df.reindex(full_idx)
        # forward/backfill numeric columns
        df[num_cols] = df[num_cols].interpolate().fillna(method='bfill').fillna(method='ffill')

    # Time features (true hour/day derived from index)
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek

    # Simple tariff: off-peak / mid / peak
    def price_func(h):
        if 18 <= h <= 22:
            return 0.30  # peak
        elif 8 <= h < 18:
            return 0.20  # mid
        else:
            return 0.10  # off-peak

    df['price_raw'] = df['hour'].apply(price_func)

    # Save a copy of raw active power (kW) and price for reward calculation
    df['Global_active_power_kW'] = df['Global_active_power']
    df['price_signal'] = df['price_raw']

    # Simple flexible vs base load using sub-meterings
    df['flexible_base'] = df['Sub_metering_1'] + df['Sub_metering_2']
    df['hp_load'] = df['Sub_metering_3']  # treat as controllable high-power appliance

    # Scale features for RL state
    feat_cols = [
        'Global_active_power_kW', 'Global_reactive_power', 'Voltage',
        'Global_intensity', 'flexible_base', 'hp_load',
        # NOTE: we don't scale 'hour'/'day_of_week' here for simplicity of state (we include raw hour/day)
        # but to keep state bounded, we'll include normalized hour/day in [0,1]
        'hour', 'day_of_week', 'price_signal'
    ]

    # Normalize hour and day_of_week to 0..1 to keep observation_space bounded
    df['hour_norm'] = df['hour'] / 23.0
    df['dow_norm'] = df['day_of_week'] / 6.0
    # Replace 'hour' and 'day_of_week' entries in feat cols with their normalized names
    feat_cols = [
        'Global_active_power_kW', 'Global_reactive_power', 'Voltage',
        'Global_intensity', 'flexible_base', 'hp_load',
        'hour_norm', 'dow_norm', 'price_signal'
    ]

    scaler = MinMaxScaler()
    df[feat_cols] = scaler.fit_transform(df[feat_cols])

    return df, scaler, feat_cols


# =========================
# RL ENVIRONMENT
# =========================

class EnergySchedulerEnv(gym.Env):
    """
    Simple RL environment for one controllable high-power appliance.

    State:
      - Scaled features (power, voltage, time, price, etc.)
      - Remaining required runtime fraction (0-1).
    Action:
      - 0 = OFF, 1 = ON.

    Reward:
      - Negative of (energy cost + peak penalty + comfort penalty),
        to encourage shifting use away from peak price while
        still running the appliance enough minutes per day.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        feat_cols,
        day_length=24 * 60,
        required_runtime=120,
        lambda_peak=0.2,
        lambda_comfort=0.1
    ):
        super().__init__()
        self.df = df.reset_index(drop=False)  # use integer iloc, but keep original timestamps as a column
        self.feat_cols = feat_cols
        self.day_length = day_length
        self.required_runtime = required_runtime
        self.lambda_peak = lambda_peak
        self.lambda_comfort = lambda_comfort

        # Observation: features + remaining runtime fraction
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(len(feat_cols) + 1,),
            dtype=np.float32
        )
        # Action: ON/OFF
        self.action_space = spaces.Discrete(2)

        self.current_step = None
        self.start_idx = None
        self.remaining_runtime = None
        self.max_power_seen = None
        self.rng = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        if len(self.df) <= self.day_length:
            self.start_idx = 0
        else:
            self.start_idx = int(self.rng.integers(0, len(self.df) - self.day_length))

        self.current_step = 0
        self.remaining_runtime = float(self.required_runtime)
        self.max_power_seen = 0.0

        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self):
        idx = self.start_idx + self.current_step
        # Safety: clamp index
        idx = min(idx, len(self.df) - 1)
        row = self.df.iloc[idx][self.feat_cols].values.astype(np.float32)
        remaining_frac = np.array(
            [max(self.remaining_runtime, 0) / max(self.required_runtime, 1.0)],
            dtype=np.float32
        )
        return np.concatenate([row, remaining_frac], axis=0)

    def step(self, action):
        idx = self.start_idx + self.current_step
        # If we've gone past data length, terminate
        if idx >= len(self.df):
            # Episode ended unexpectedly due to lack of data
            terminated = True
            truncated = False
            obs = self._get_obs()
            reward = 0.0
            info = {"cost": 0.0, "power": 0.0, "remaining_runtime": float(self.remaining_runtime),
                    "max_power_seen": float(self.max_power_seen)}
            return obs, reward, terminated, truncated, info

        row = self.df.iloc[idx]

        on = 1 if int(action) == 1 else 0

        # Denormalized power and price (we stored raw copies)
        base_power_kW = row['Global_active_power_kW']
        price = row['price_signal']

        # Assume high-power appliance drains 1 kW when ON (simple proxy).
        # Note: base_power_kW is normalized (0..1) — but we kept original scale under a different column earlier.
        # If you want actual kW values in cost, don't scale 'Global_active_power_kW' before cost calculation.
        # For simplicity, we'll use the *denormalized* original power if present in df; else fall back:
        if 'Global_active_power' in self.df.columns:
            # If resampled, this column exists (we filled it earlier)
            true_base_kW = float(row.get('Global_active_power', base_power_kW))
        else:
            true_base_kW = float(base_power_kW)

        power_with_hp = true_base_kW + 1.0 * on
        self.max_power_seen = max(self.max_power_seen, power_with_hp)

        # Cost per minute (kWh = kW * (1/60 h))
        cost = power_with_hp * float(price) * (1.0 / 60.0)

        # Comfort: penalize unmet runtime if time is running out.
        steps_left = self.day_length - (self.current_step + 1)
        comfort_penalty = 0.0
        unmet_after = max(self.remaining_runtime - on, 0)
        if steps_left < self.remaining_runtime:
            comfort_penalty = (self.remaining_runtime - steps_left) / max(self.required_runtime, 1.0)

        # Stabilize peak penalty: ratio of observed peak to a running baseline (avoid tiny denom)
        denom = max(true_base_kW + 1e-3, 1e-3)
        peak_ratio = (self.max_power_seen / denom)
        # Bound peak_ratio to avoid extremely large penalties
        peak_ratio = min(peak_ratio, 100.0)

        reward = -(
            cost
            + self.lambda_peak * peak_ratio
            + self.lambda_comfort * comfort_penalty
        )

        # Update runtime requirement
        if on and self.remaining_runtime > 0:
            self.remaining_runtime -= 1

        self.current_step += 1
        terminated = self.current_step >= self.day_length
        truncated = False

        obs = self._get_obs()
        info = {
            "cost": float(cost),
            "power": float(power_with_hp),
            "remaining_runtime": float(self.remaining_runtime),
            "max_power_seen": float(self.max_power_seen),
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        pass


# =========================
# TRAINING
# =========================

def train_rl(total_timesteps: int = 50_000):
    df, scaler, feat_cols = load_and_preprocess()

    env = EnergySchedulerEnv(
        df=df,
        feat_cols=feat_cols,
        day_length=24 * 60,
        required_runtime=120,
        lambda_peak=0.2,
        lambda_comfort=0.1
    )

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=total_timesteps)

    # SB3 saves as MODEL_PATH.zip
    model.save(str(MODEL_PATH))
    joblib.dump(scaler, SCALER_PATH)
    print("Training complete. Model and scaler saved.")


# =========================
# STREAMLIT DASHBOARD
# =========================

@st.cache_data
def _st_load_data():
    df, scaler, feat_cols = load_and_preprocess()
    return df, feat_cols


@st.cache_resource
def _st_load_model():
    # Load only if model file exists to avoid unclear exceptions from SB3
    zip_path = str(MODEL_PATH) + ".zip"
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Model not found at {zip_path}")
    model = PPO.load(str(MODEL_PATH))
    return model


def run_dashboard():
    st.set_page_config(page_title="AI Energy Scheduler", layout="wide")
    st.title("AI Energy Consumption Scheduler Dashboard")

    # Load data & model
    df, feat_cols = _st_load_data()
    try:
        model = _st_load_model()
    except Exception as e:
        st.error(f"RL model not found or failed to load. Run: python ai_energy_scheduler.py --train\nDetails: {e}")
        return

    st.sidebar.header("Simulation Controls")

    # Select a day (use available dates)
    available_dates = sorted(pd.to_datetime(df.index).date)
    default_date = available_dates[0] if available_dates else pd.Timestamp.now().date()
    date_choice = st.sidebar.date_input("Select day", value=default_date)

    # Filter a single day (df index is DatetimeIndex)
    # Our df from preprocessing may have been reset, so we rely on original timestamps if available
    if 'index' in df.columns:
        timestamps = pd.to_datetime(df['index'])
    else:
        timestamps = pd.to_datetime(df.index)

    day_mask = timestamps.date == pd.to_datetime(date_choice).date()
    day_df = df[day_mask].copy()
    if len(day_df) == 0:
        st.warning("No data for selected date.")
        return

    strategy = st.sidebar.selectbox(
        "Scheduling strategy",
        ["RL Scheduler", "Naive (Always ON 18–22)"]
    )

    required_runtime = st.sidebar.slider(
        "Required runtime (minutes/day)",
        min_value=30,
        max_value=240,
        value=120,
        step=30
    )

    # Build environment limited to this day
    env = EnergySchedulerEnv(
        df=day_df,
        feat_cols=feat_cols,
        day_length=len(day_df),
        required_runtime=required_runtime,
        lambda_peak=0.2,
        lambda_comfort=0.1
    )

    def run_policy(policy="RL"):
        obs, info = env.reset()
        costs, powers, actions, hours, prices = [], [], [], [], []
        for _ in range(env.day_length):
            if policy == "RL":
                # model.predict may return array or scalar; extract int
                action_arr, _ = model.predict(obs, deterministic=True)
                action = int(action_arr) if hasattr(action_arr, "__len__") else int(action_arr)
            else:
                idx = env.start_idx + env.current_step
                # get true hour from timestamp column
                ts = env.df.iloc[min(idx, len(env.df)-1)].get('index', None)
                if ts is None:
                    # fallback: if 'index' column is missing, try to reconstruct from original values
                    # attempt to use numeric index as hour 0
                    h = int(env.df.iloc[min(idx, len(env.df)-1)].get('hour', 0))
                else:
                    h = pd.to_datetime(ts).hour

                if 18 <= h <= 22 and env.remaining_runtime > 0:
                    action = 1
                else:
                    action = 0

            obs, reward, terminated, truncated, info = env.step(action)
            actions.append(int(action))
            costs.append(info["cost"])
            powers.append(info["power"])

            # Use true hour from index (if present)
            idx_for_hour = env.start_idx + env.current_step - 1
            idx_for_hour = max(0, min(idx_for_hour, len(env.df)-1))
            ts = env.df.iloc[idx_for_hour].get('index', None)
            if ts is not None:
                true_hour = pd.to_datetime(ts).hour
            else:
                true_hour = int(env.df.iloc[idx_for_hour].get('hour', 0))
            hours.append(true_hour)

            prices.append(float(env.df.iloc[idx_for_hour].get('price_signal', 0.0)))

            if terminated or truncated:
                break

        res_df = pd.DataFrame({
            "step": np.arange(len(costs)),
            "hour": hours,
            "action_on": actions,
            "power_kW": powers,
            "cost": costs,
            "price": prices
        })
        return res_df

    if strategy == "RL Scheduler":
        sim_df = run_policy(policy="RL")
    else:
        sim_df = run_policy(policy="Naive")

    total_cost = sim_df["cost"].sum()
    peak_power = sim_df["power_kW"].max() if len(sim_df) else 0.0
    runtime_actual = int(sim_df["action_on"].sum()) if len(sim_df) else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Total cost (relative)", f"{total_cost:.6f}")
    col2.metric("Peak power (kW approx.)", f"{peak_power:.3f}")
    col3.metric("Runtime (min)", f"{runtime_actual} / {required_runtime}")

    st.subheader("Power profile over the day")
    if len(sim_df):
        st.line_chart(sim_df.set_index("step")[["power_kW"]])
    else:
        st.write("No simulation data to show.")

    st.subheader("Appliance schedule (ON = 1, OFF = 0)")
    if len(sim_df):
        st.area_chart(sim_df.set_index("step")[["action_on"]])

    st.subheader("Price signal")
    if len(sim_df):
        st.line_chart(sim_df.set_index("step")[["price"]])

    st.subheader("Simulation data")
    st.dataframe(sim_df)


# =========================
# MAIN
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the RL scheduler instead of running Streamlit dashboard.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50_000,
        help="Number of training timesteps.",
    )
    args, unknown = parser.parse_known_args()

    if args.train:
        train_rl(total_timesteps=args.timesteps)
    else:
        # If run via `streamlit run ai_energy_scheduler.py`, Streamlit will call run_dashboard().
        run_dashboard()


if __name__ == "__main__":
    main()
