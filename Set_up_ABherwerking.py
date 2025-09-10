import pandas as pd
import numpy as np
import os

# ================================= MODULE II: INITIAL HEALTH STATE SETUP =================================
# This module prepares the initial health state allocation for a simulated population (Year 1).
# It assigns each individual to a starting health state and simulates the evolution of that state over 365 days.
# The result is a daily record of health states per individual, ready for use in dynamic transmission modeling.

# ----------------------------------------- CONFIGURATION --------------------------------------------------

# Simulation start settings
start_year = 1
start_day = 1

# Parameters for illness duration, treatment effect, resistance development and immunity
duration_illness_sens = 7   # base illness duration for non-resistant infections
ab_reduction_factor = 0.8   # reduction factor when appropriate antibiotics are used
resistance_extension_factor = 1.56  # illness duration increase due to resistance
duration_immunity = 180     # duration of immunity after recovery
P_within_host_selec = 0.12  # probability of within-host selection after inappropriate treatment

# Starting proportions for each health state
mu_values = {
    "Susceptible": 0.996,
    "I_NR_no_AB": 0.001,
    "I_NR_AB_appropriate": 0.00137,
    "I_NR_AB_inappropriate": 0.00063,
    "I_R_no_AB": 0.000054,
    "I_R_AB_appropriate": 0.00002466,
    "I_R_AB_inappropriate": 0.00001134,
    "Immune": 0.0
}

# File paths
base_path_out = "/Users/erinndhulster/Desktop/Simulation AMR/Transmission/Set-up"
file_path = "/Users/erinndhulster/Desktop/Simulation AMR/Population Model/Year_1_population_herwerkt_ARON.csv"

# ----------------------------------------- LOAD POPULATION -------------------------------------------------

# Load individual identifiers
population_df = pd.read_csv(file_path, sep=';', decimal=',')
n = len(population_df)
population_df = population_df[["Identifier"]].copy()

# ----------------------------------------- INITIAL STATE ALLOCATION ----------------------------------------

# Convert proportions into absolute counts and assign initial health states randomly
states = list(mu_values.keys())
counts = (np.array(list(mu_values.values())) * n).astype(int)
counts[0] += n - counts.sum()  # adjustment to ensure total adds up

# Create a shuffled list of health states
init_states = np.repeat(states, counts)
np.random.shuffle(init_states)
status_matrix = pd.get_dummies(init_states)

# Ensure all expected columns are present, even if not assigned
for state in states:
    if state not in status_matrix:
        status_matrix[state] = 0
status_matrix = status_matrix[states]  # maintain consistent order

# Add state columns to population
population_df = pd.concat([population_df, status_matrix], axis=1)

# Assign infection markers and within-host selection
population_df["NonRes_Inf"] = population_df[["I_NR_no_AB", "I_NR_AB_appropriate", "I_NR_AB_inappropriate"]].sum(axis=1).astype(bool).astype(int)
population_df["Res_Inf"] = population_df[["I_R_no_AB", "I_R_AB_appropriate", "I_R_AB_inappropriate"]].sum(axis=1).astype(bool).astype(int)
population_df["Within host selection?"] = 0
inapp_idx = population_df["I_NR_AB_inappropriate"] == 1
population_df.loc[inapp_idx, "Within host selection?"] = np.random.binomial(1, P_within_host_selec, size=inapp_idx.sum())

# ----------------------------------------- SIMULATE STATE OVER TIME -----------------------------------------

# Generate daily state data for each individual across all 365 days
all_days = []
for day in range(1, 366):
    df = population_df.copy()
    df.insert(0, "Day", day)
    df.insert(0, "Year", start_year)

    if day > 1:
        # Reset daily health states and flags
        df[["Within host selection?", "NonRes_Inf", "Res_Inf"]] = 0
        df[states] = 0

        ill = duration_illness_sens - 1
        immune = duration_immunity
        def active(t1, t2): return t1 < day <= t2

        # Simulate each scenario
        idx = population_df["I_NR_no_AB"] == 1
        df.loc[idx, "I_NR_no_AB"] = (day <= start_day + ill)
        df.loc[idx, "Immune"] = active(start_day + ill, start_day + ill + immune)

        idx = population_df["I_NR_AB_appropriate"] == 1
        ill2 = int(duration_illness_sens * ab_reduction_factor) - 1
        df.loc[idx, "I_NR_AB_appropriate"] = (day <= start_day + ill2)
        df.loc[idx, "Immune"] = active(start_day + ill2, start_day + ill2 + immune)

        idx = population_df["I_NR_AB_inappropriate"] == 1
        whs = population_df["Within host selection?"] == 1

        nr_only = idx & ~whs
        ill3 = ill
        df.loc[nr_only, "I_NR_AB_inappropriate"] = (day <= start_day + ill3)
        df.loc[nr_only, "Immune"] = active(start_day + ill3, start_day + ill3 + immune)

        whs_only = idx & whs
        ill4 = ill
        res = int(duration_illness_sens * resistance_extension_factor)
        df.loc[whs_only, "I_NR_AB_inappropriate"] = (day <= start_day + ill4)
        df.loc[whs_only, "I_R_no_AB"] = active(start_day + ill4, start_day + ill4 + res)
        df.loc[whs_only, "Immune"] = active(start_day + ill4 + res, start_day + ill4 + res + immune)

        for st, dur in zip(["I_R_no_AB", "I_R_AB_inappropriate"],
                           [int(duration_illness_sens * resistance_extension_factor) - 1] * 2):
            idx = population_df[st] == 1
            df.loc[idx, st] = (day <= start_day + dur)
            df.loc[idx, "Immune"] = active(start_day + dur, start_day + dur + immune)

        idx = population_df["I_R_AB_appropriate"] == 1
        dur = int(duration_illness_sens * ab_reduction_factor * resistance_extension_factor) - 1
        df.loc[idx, "I_R_AB_appropriate"] = (day <= start_day + dur)
        df.loc[idx, "Immune"] = active(start_day + dur, start_day + dur + immune)

        idx = population_df["Immune"] == 1
        df.loc[idx, "Immune"] = (day <= start_day + immune - 1)

    all_days.append(df)

# ----------------------------------------- EXPORT TO CSV ----------------------------------------------------

# Merge all daily data into a single DataFrame
final_df = pd.concat(all_days, ignore_index=True)

# Convert booleans to 0/1 integers
final_df = final_df.applymap(lambda x: int(x) if isinstance(x, bool) else x)

# Replace all-0 rows with blanks for health states and infection markers (i.e., no state assigned that day)
health_columns = states + ["NonRes_Inf", "Res_Inf", "Within host selection?"]
mask_no_assignment = (final_df[health_columns].sum(axis=1) == 0)
final_df.loc[mask_no_assignment, health_columns] = ""

# Save the dataset
csv_path = os.path.join(base_path_out, f"Simulation_Setup_Year_{start_year}_Days_1_to_365_ARON.csv")
final_df.to_csv(csv_path, index=False, sep=';', decimal=',')

# Feedback
print(f"✔ Simulation data exported as CSV to: {csv_path}")
print(f"\nSimulation range: Year {start_year}, Days 1–365")
print("\nInitial health state counts:")
for s in states:
    print(f"{s}: {status_matrix[s].sum()}")
