import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import random

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(page_title="GridShift AI", layout="wide")
np.random.seed(42)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("Simulation Controls")

days = st.sidebar.slider("Simulation Length (days)", 30, 365, 180)
demand_scale = st.sidebar.slider("Demand Scaling", 0.5, 2.0, 1.2)
gas_volatility = st.sidebar.slider("Gas Price Volatility", 0.0, 2.0, 1.0)
reserve_threshold = st.sidebar.slider("Stress Threshold", 0.01, 0.30, 0.10)
enable_agent = st.sidebar.checkbox("Enable AI Agent", value=True)

# =========================================================
# DATA MODELS
# =========================================================
@st.cache_data
def generate_demand(days, scale):
    hours = days * 24
    idx = pd.date_range("2025-01-01", periods=hours, freq="h")

    seasonal = 15 + 10 * np.cos((idx.dayofyear / 365) * 2 * np.pi)

    daily = (
        5
        + 4 * np.sin((idx.hour - 7) / 24 * 2 * np.pi)
        + 4 * np.sin((idx.hour - 19) / 24 * 2 * np.pi)
    )

    noise = np.random.normal(0, 1.5, hours)
    demand = np.clip((seasonal + daily + noise) * scale, 10, None)

    return pd.DataFrame({"demand": demand}, index=idx)


@st.cache_data
def generation_stack():
    plants = pd.DataFrame({
        "type": ["nuclear", "coal", "gas_cc", "gas_peaker", "wind", "solar"],
        "capacity": [25, 20, 35, 15, 10, 5],
        "cost": [10, 30, 45, 120, 5, 3],
        "emissions_rate": [0.0, 1.0, 0.4, 0.7, 0.0, 0.0],
    })

    return plants.sort_values("cost")


def renewable_profile(idx):
    solar = np.maximum(0, np.sin((idx.hour - 7) / 12 * np.pi)) * 0.6
    wind = 0.4 + 0.2 * np.sin(idx.dayofyear / 365 * 2 * np.pi)

    return solar, wind


@st.cache_data
def gas_prices(idx_length, dayofyear, volatility):
    base = 3
    seasonal = 2 * np.cos(dayofyear / 365 * 2 * np.pi)
    noise = np.random.normal(0, volatility, idx_length)

    return base + seasonal + noise


# =========================================================
# DISPATCH ENGINE
# =========================================================
def dispatch(demand_df, plants, volatility):
    idx = demand_df.index
    solar_cf, wind_cf = renewable_profile(idx)
    gas = gas_prices(len(idx), idx.dayofyear.to_numpy(), volatility)

    results = []

    for i, t in enumerate(idx):
        demand = demand_df.iloc[i]["demand"]

        used_supply = 0
        total_available_capacity = 0
        cost = 0
        emissions = 0

        for _, p in plants.iterrows():
            if p["type"] == "solar":
                avail = p["capacity"] * solar_cf[i]
                marginal = p["cost"]

            elif p["type"] == "wind":
                avail = p["capacity"] * wind_cf[i]
                marginal = p["cost"]

            elif "gas" in p["type"]:
                avail = p["capacity"]
                marginal = p["cost"] * (gas[i] / 3)

            else:
                avail = p["capacity"]
                marginal = p["cost"]

            total_available_capacity += avail

            used = max(0, min(avail, demand - used_supply))

            used_supply += used
            cost += used * marginal
            emissions += used * p["emissions_rate"]

        reserve = (
            (total_available_capacity - demand) / demand
            if demand > 0
            else 0
        )

        results.append({
            "demand": demand,
            "supply": used_supply,
            "available_capacity": total_available_capacity,
            "reserve": reserve,
            "gas_price": gas[i],
            "cost": cost,
            "emissions": emissions,
        })

    return pd.DataFrame(results, index=idx)


# =========================================================
# RL AGENT - LOAD SHIFTING
# =========================================================
class RLAgent:
    def __init__(self):
        self.q = {}

        # Percent of demand shifted from peak/stressed hours
        self.actions = [0.00, 0.05, 0.10, 0.15]

        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.2

    def get_state(self, reserve, hour, threshold):
        if reserve < threshold:
            reserve_state = "low"
        elif reserve < threshold * 1.5:
            reserve_state = "medium"
        else:
            reserve_state = "high"

        if 16 <= hour <= 21:
            time_state = "peak"
        elif 0 <= hour <= 6:
            time_state = "offpeak"
        else:
            time_state = "normal"

        return (reserve_state, time_state)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)

        return max(
            self.actions,
            key=lambda action: self.q.get((state, action), 0)
        )

    def update(self, state, action, reward, next_state):
        old_value = self.q.get((state, action), 0)

        future_value = max(
            self.q.get((next_state, action), 0)
            for action in self.actions
        )

        self.q[(state, action)] = old_value + self.alpha * (
            reward + self.gamma * future_value - old_value
        )


def find_shift_target(reserves, index, hours, threshold):
    """
    Finds a safer future hour to move demand into.
    Prefers overnight and midday hours with high reserve.
    """

    end = min(index + 24, len(reserves) - 1)
    candidates = []

    for j in range(index + 1, end + 1):
        hour = hours[j]

        is_offpeak = 0 <= hour <= 6
        is_midday = 10 <= hour <= 15
        has_good_reserve = reserves[j] > threshold * 1.5

        if has_good_reserve and (is_offpeak or is_midday):
            candidates.append(j)

    if not candidates:
        return None

    return max(candidates, key=lambda j: reserves[j])


def train_agent(demand_df, plants, agent, threshold, volatility, episodes=100):
    original_demand = demand_df["demand"].values.copy()

    best_demand = original_demand.copy()
    best_score = float("-inf")

    baseline_result = dispatch(demand_df, plants, volatility)
    baseline_reserves = baseline_result["reserve"].values.copy()
    hours = demand_df.index.hour.to_numpy()

    for _ in range(episodes):
        adjusted_demand = original_demand.copy()
        reserves = baseline_reserves.copy()

        for i in range(len(adjusted_demand) - 1):
            hour = hours[i]
            current_reserve = reserves[i]
            is_peak_hour = 16 <= hour <= 21

            state = agent.get_state(current_reserve, hour, threshold)
            action = agent.choose_action(state)

            shift_target = find_shift_target(
                reserves=reserves,
                index=i,
                hours=hours,
                threshold=threshold
            )

            reward = 0

            if shift_target is not None and (current_reserve < threshold or is_peak_hour):
                shifted_amount = adjusted_demand[i] * action

                adjusted_demand[i] -= shifted_amount
                adjusted_demand[shift_target] += shifted_amount

                new_reserve = current_reserve + action
                target_reserve = reserves[shift_target] - action

                before_stress = current_reserve < threshold
                after_fixed = new_reserve >= threshold
                caused_new_stress = target_reserve < threshold

                if before_stress and after_fixed and not caused_new_stress:
                    reward = 25
                elif is_peak_hour and action > 0 and not caused_new_stress:
                    reward = 15
                elif caused_new_stress:
                    reward = -20
                elif action == 0:
                    reward = -2
                else:
                    reward = 3

            else:
                if current_reserve < threshold:
                    reward = -5
                else:
                    reward = 2

            next_hour = hours[i + 1]

            next_state = agent.get_state(
                reserves[i + 1],
                next_hour,
                threshold
            )

            agent.update(state, action, reward, next_state)

        test_df = pd.DataFrame(
            {"demand": adjusted_demand},
            index=demand_df.index
        )

        test_result = dispatch(test_df, plants, volatility)

        stress_events = (test_result["reserve"] < threshold).sum()
        peak_demand = test_result["demand"].max()
        total_cost = test_result["cost"].sum()
        total_emissions = test_result["emissions"].sum()

        score = (
            -1000 * stress_events
            -10 * peak_demand
            -0.01 * total_cost
            -5 * total_emissions
        )

        if score > best_score:
            best_score = score
            best_demand = adjusted_demand.copy()

    return best_demand


# =========================================================
# PLOTTING
# =========================================================
def plot_series(x, y1, y2=None, name1="A", name2="B", title=""):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x,
        y=y1,
        name=name1
    ))

    if y2 is not None:
        fig.add_trace(go.Scatter(
            x=x,
            y=y2,
            name=name2
        ))

    fig.update_layout(
        title=title,
        hovermode="x unified",
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=30, label="1m", step="day", stepmode="backward"),
                    dict(step="all"),
                ]
            ),
            rangeslider=dict(visible=True),
            type="date",
        ),
    )

    return fig


def plot_load_shifting(baseline_df, agent_df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=baseline_df.index,
        y=baseline_df["demand"],
        name="Baseline Demand",
        line=dict(width=2)
    ))

    fig.add_trace(go.Scatter(
        x=agent_df.index,
        y=agent_df["demand"],
        name="AI-Shifted Demand",
        line=dict(width=2)
    ))

    fig.add_trace(go.Bar(
        x=agent_df.index,
        y=agent_df["load_shift"],
        name="Load Shift Amount",
        opacity=0.35
    ))

    fig.update_layout(
        title="Demand Response: AI Reduces Peaks and Moves Load to Safer Hours",
        hovermode="x unified",
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=30, label="1m", step="day", stepmode="backward"),
                    dict(step="all"),
                ]
            ),
            rangeslider=dict(visible=True),
            type="date",
        ),
        yaxis_title="Demand / Shifted Load"
    )

    return fig


def plot_cost(baseline_df, agent_df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=baseline_df.index,
        y=baseline_df["cost"],
        name="Baseline Cost"
    ))

    fig.add_trace(go.Scatter(
        x=agent_df.index,
        y=agent_df["cost"],
        name="AI Cost"
    ))

    fig.update_layout(
        title="Cost Over Time: AI Avoids Expensive Peak Generation",
        hovermode="x unified",
        yaxis_title="Cost"
    )

    return fig


def plot_emissions(baseline_df, agent_df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=baseline_df.index,
        y=baseline_df["emissions"],
        name="Baseline Emissions"
    ))

    fig.add_trace(go.Scatter(
        x=agent_df.index,
        y=agent_df["emissions"],
        name="AI Emissions"
    ))

    fig.update_layout(
        title="Carbon Emissions Over Time",
        hovermode="x unified",
        yaxis_title="CO₂ Emissions"
    )

    return fig


# =========================================================
# RUN PIPELINE
# =========================================================
demand_df = generate_demand(days, demand_scale)
plants = generation_stack()

baseline_df = dispatch(demand_df, plants, gas_volatility)

if enable_agent:
    agent = RLAgent()

    adjusted = train_agent(
        demand_df=demand_df,
        plants=plants,
        agent=agent,
        threshold=reserve_threshold,
        volatility=gas_volatility,
        episodes=100
    )

    agent_demand_df = pd.DataFrame(
        {"demand": adjusted},
        index=demand_df.index
    )

    agent_df = dispatch(agent_demand_df, plants, gas_volatility)

else:
    agent_df = baseline_df.copy()


# =========================================================
# LOAD SHIFT CALCULATIONS
# =========================================================
agent_df["load_shift"] = baseline_df["demand"] - agent_df["demand"]

agent_df["shift_type"] = np.where(
    agent_df["load_shift"] > 0,
    "Reduced Peak Load",
    np.where(
        agent_df["load_shift"] < 0,
        "Added Off-Peak Load",
        "No Shift"
    )
)

change_details = agent_df[agent_df["load_shift"].abs() > 0.001].copy()

change_details["baseline_demand"] = baseline_df.loc[
    change_details.index,
    "demand"
]

change_details["ai_demand"] = agent_df.loc[
    change_details.index,
    "demand"
]

change_details["baseline_reserve"] = baseline_df.loc[
    change_details.index,
    "reserve"
]

change_details["ai_reserve"] = agent_df.loc[
    change_details.index,
    "reserve"
]

change_details["baseline_cost"] = baseline_df.loc[
    change_details.index,
    "cost"
]

change_details["ai_cost"] = agent_df.loc[
    change_details.index,
    "cost"
]

change_details["baseline_emissions"] = baseline_df.loc[
    change_details.index,
    "emissions"
]

change_details["ai_emissions"] = agent_df.loc[
    change_details.index,
    "emissions"
]

change_details["energy_change"] = (
    change_details["ai_demand"] - change_details["baseline_demand"]
)

change_details["cost_change"] = (
    change_details["ai_cost"] - change_details["baseline_cost"]
)

change_details["emissions_change"] = (
    change_details["ai_emissions"] - change_details["baseline_emissions"]
)

change_details_table = change_details[
    [
        "shift_type",
        "baseline_demand",
        "ai_demand",
        "energy_change",
        "baseline_reserve",
        "ai_reserve",
        "baseline_cost",
        "ai_cost",
        "cost_change",
        "baseline_emissions",
        "ai_emissions",
        "emissions_change",
    ]
].round(4)


# =========================================================
# METRIC CALCULATIONS
# =========================================================
baseline_stress = (baseline_df["reserve"] < reserve_threshold).sum()
agent_stress = (agent_df["reserve"] < reserve_threshold).sum()

baseline_peak = baseline_df["demand"].max()
agent_peak = agent_df["demand"].max()

total_baseline_energy = baseline_df["demand"].sum()
total_agent_energy = agent_df["demand"].sum()

total_peak_reduction = agent_df.loc[
    agent_df["load_shift"] > 0,
    "load_shift"
].sum()

total_offpeak_refill = abs(
    agent_df.loc[
        agent_df["load_shift"] < 0,
        "load_shift"
    ].sum()
)

baseline_total_cost = baseline_df["cost"].sum()
agent_total_cost = agent_df["cost"].sum()

cost_savings = baseline_total_cost - agent_total_cost
percent_savings = (
    cost_savings / baseline_total_cost
) * 100 if baseline_total_cost > 0 else 0

baseline_total_emissions = baseline_df["emissions"].sum()
agent_total_emissions = agent_df["emissions"].sum()

emissions_reduction = baseline_total_emissions - agent_total_emissions
percent_emissions_reduction = (
    emissions_reduction / baseline_total_emissions
) * 100 if baseline_total_emissions > 0 else 0

energy_removed = change_details.loc[
    change_details["energy_change"] < 0,
    "energy_change"
].sum()

energy_added = change_details.loc[
    change_details["energy_change"] > 0,
    "energy_change"
].sum()


# =========================================================
# UI
# =========================================================
st.title("⚡ GridShift AI")
st.caption(
    "A grid-aware AI agent that shifts flexible demand to reduce peak load, "
    "lower operating costs, and reduce emissions."
)

st.subheader("Demand: Baseline vs AI-Shifted")
st.plotly_chart(
    plot_series(
        baseline_df.index,
        baseline_df["demand"],
        agent_df["demand"],
        "Baseline Demand",
        "AI-Shifted Demand",
        "Demand"
    ),
    use_container_width=True
)

st.subheader("AI Load Shifting: Peak Reduction vs Off-Peak Refill")
st.plotly_chart(
    plot_load_shifting(baseline_df, agent_df),
    use_container_width=True
)

st.subheader("Supply")
st.plotly_chart(
    plot_series(
        baseline_df.index,
        baseline_df["supply"],
        agent_df["supply"],
        "Baseline Supply",
        "AI Supply",
        "Supply"
    ),
    use_container_width=True
)

st.subheader("Reserve Margin")
fig_reserve = plot_series(
    baseline_df.index,
    baseline_df["reserve"],
    agent_df["reserve"],
    "Baseline Reserve",
    "AI Reserve",
    "Reserve Margin"
)

fig_reserve.add_hline(
    y=reserve_threshold,
    line_dash="dash",
    annotation_text="Stress Threshold"
)

st.plotly_chart(fig_reserve, use_container_width=True)

st.subheader("Cost Over Time")
st.plotly_chart(
    plot_cost(baseline_df, agent_df),
    use_container_width=True
)

st.subheader("Carbon Emissions Over Time")
st.plotly_chart(
    plot_emissions(baseline_df, agent_df),
    use_container_width=True
)


# =========================================================
# METRICS UI
# =========================================================
st.subheader("Grid Performance Metrics")

c1, c2, c3 = st.columns(3)

c1.metric("Baseline Stress Events", baseline_stress)
c2.metric("AI Stress Events", agent_stress)
c3.metric("Stress Events Reduced", baseline_stress - agent_stress)

c4, c5, c6 = st.columns(3)

c4.metric("Baseline Peak Demand", round(baseline_peak, 2))
c5.metric("AI Peak Demand", round(agent_peak, 2))
c6.metric("Peak Demand Reduced", round(baseline_peak - agent_peak, 2))

c7, c8, c9 = st.columns(3)

c7.metric("Total Peak Load Reduced", round(total_peak_reduction, 2))
c8.metric("Total Off-Peak Load Refilled", round(total_offpeak_refill, 2))
c9.metric("Total Energy Change", round(total_agent_energy - total_baseline_energy, 4))

st.subheader("Cost Savings Impact")

c10, c11, c12 = st.columns(3)

c10.metric("Baseline Cost", round(baseline_total_cost, 2))
c11.metric("AI Cost", round(agent_total_cost, 2))
c12.metric("Cost Savings", round(cost_savings, 2))

c13 = st.columns(1)[0]
c13.metric("Percent Cost Savings (%)", round(percent_savings, 2))

st.subheader("Carbon Emissions Impact")

c14, c15, c16 = st.columns(3)

c14.metric("Baseline Emissions", round(baseline_total_emissions, 2))
c15.metric("AI Emissions", round(agent_total_emissions, 2))
c16.metric("Emissions Reduced", round(emissions_reduction, 2))

c17 = st.columns(1)[0]
c17.metric(
    "Percent Emissions Reduction (%)",
    round(percent_emissions_reduction, 2)
)

st.subheader("Energy Preservation Check")

c18, c19, c20 = st.columns(3)

c18.metric("Energy Removed from Peaks", round(abs(energy_removed), 4))
c19.metric("Energy Added Elsewhere", round(energy_added, 4))
c20.metric("Net Energy Change", round(energy_added + energy_removed, 4))


# =========================================================
# EXPLANATION + TABLE
# =========================================================
if enable_agent:
    if agent_stress < baseline_stress:
        st.success(
            "AI improves the grid by shifting demand away from stressed hours."
        )
    elif agent_stress == baseline_stress and agent_peak < baseline_peak:
        st.info(
            "AI reduced peak demand while preserving total energy."
        )
    elif agent_stress == baseline_stress:
        st.info(
            "AI preserved energy, but stress events stayed the same."
        )
    else:
        st.warning(
            "AI did not improve stress events under these settings."
        )

if cost_savings > 0:
    st.success(
        "AI reduces grid operating costs by avoiding expensive peak generation."
    )

if emissions_reduction > 0:
    st.success(
        "AI reduces carbon emissions by shifting demand away from fossil-heavy hours."
    )

st.subheader("AI Load Shift Details")

st.caption(
    "This table shows each hour where the AI changed demand. "
    "Negative energy_change means demand was reduced from that hour. "
    "Positive energy_change means demand was added to that hour."
)

st.dataframe(
    change_details_table,
    use_container_width=True
)