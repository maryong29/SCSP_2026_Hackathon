# SCSP_2026_Hackathon
The rapid growth of AI and data centers is putting unprecedented strain on energy grids. As highlighted in CNBC article on AI data centers and Pennsylvania Republicans 2026 election, rising electricity demand is already becoming a political and infrastructure challenge.
Today’s grids are reactive, not intelligent:
- Peak demand causes grid stress and outages
- Energy is underutilized during off-peak hours
- No adaptive system exists to balance demand in real time

I built a grid-aware reinforcement learning agent that dynamically shifts electricity demand to stabilize the grid.
Instead of static rules, our AI:
- Detects high-risk (low reserve) periods
- Reduces demand during stress events
- Redistributes load to off-peak hours
- Learns optimal behavior over time

Core Idea
We model the grid as an environment and train an RL agent to optimize it.
State: Grid reserve level
Action: Adjust demand (-10%, 0%, +10%)
Reward:
- Penalty if grid is stressed
- Reward if grid is stable
Over time, the agent learns:
- “Shift demand away from dangerous periods.”

System Architecture:

Grid Data → RL Agent → Demand Adjustment → Updated Grid State → Reward → Learning Loop

Key Features:

- Peak demand reduction
- Load shifting to off-peak hours
- Cost savings simulation
- Carbon emissions reduction (optional add-on)
- Self-improving RL model

In this demo, we simulate grid conditions and show how the AI agent dynamically adjusts demand in real time to prevent stress events.

How to Run

</> Bash
git clone https://github.com/your-repo/grid-ai
cd grid-ai
pip install -r requirements.txt
python app.py

If using Streamlit

</> Bash
streamlit run app.py

**Tech Stack**

- Python
- Pandas / NumPy
- Reinforcement Learning (Q-learning)
- Streamlit for UI
- Used a simulated dataset. I started with a simulated environment to rapidly prototype and validate the RL strategy. This lets us test extreme scenarios like grid stress events, which are harder to capture in static datasets.

Future Improvements
- Real-time grid data integration
- Deep Reinforcement Learning (DQN)
- Integration with smart grids / utilities
- Dynamic pricing optimization
- Scaling for data center demand

Why this matters
As AI infrastructure scales, energy demand will become one of the biggest bottlenecks. This project demonstrates how AI itself can be used to make energy systems more resilient, efficient, and sustainable.

Author:
Lizzie Ong
