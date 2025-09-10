# ARON Long-Term AMR Model — Part III: Infection Transmission Dynamics

## Overview
This module simulates the daily spread of respiratory infections over time in a simulated population.
It builds on the population characteristics (Part I) and initial health state setup (Part II) and serves as the core engine of the long-term AMR model.
The simulation accounts for treatment type, resistance development, immunity, and seasonal variation in transmission risk.

## Objectives
- Simulate daily infection dynamics for resistant and non-resistant pathogens
- Model treatment allocation and its effect on illness duration and resistance development
- Track each individual's health state over time
- Identify within host selection of resistance cases following inappropriate treatment
- Generate outputs for incidence, prevalence, and health state trends

## Parameters (adjustable at the top of the script)
- `duration_illness_sens`: base duration of non-resistant infections (days)
- `ab_reduction_factor`: multiplier for illness duration reduction with appropriate antibiotics
- `resistance_extension_factor`: multiplier for extended illness duration due to resistance
- `duration_immunity`: number of immune days following recovery
- `P_within_host_selec`: probability of resistance following inappropriate antibiotics
- `w_prevalence`, `gamma_prevalence`, `epsilon_prevalence`: parameters for dynamic prevalence-based transmission probabilities
- `state_columns`: list of exclusive health states assigned to individuals

## Key Steps
1. **Read inputs**
   - Loads initial health states and baseline infection probabilities from Part I and II

2. **Daily simulation loop**
   - Calculates adjusted infection probabilities per individual based on prevalence and season
   - Draws new infections and allocates treatment randomly (none, appropriate, or inappropriate)
   - Assigns daily health state values based on illness progression and immunity
   - Models within host selection of resistance transitions

3. **Summarize and visualize results**
   - Aggregates daily and weekly infection counts
   - Tracks incidence by treatment type and resistance status
   - Visualizes seasonal infection curves and state distributions

## Output
Saved in `/Transmission/Dynamics/`:
- Daily individual-level health state data (`Simulation_Dynamics_Year_<n>_Days_1_to_365.csv`)
- Daily state counts (`Daily_Counts_Year<n>.csv`)
- Combined dataset across years
- Weekly incidence summary (`incidenties_per_week_per_state.csv`)
- Figures:
  - Daily health states per year (`plot_year_<n>.png`)
  - Weekly incidence per state, seasonally ordered (`incidentieplot_week25_start_year_<n>.png`)

This module enables long-term estimation of the ARON intervention’s impact on infection and resistance dynamics.
