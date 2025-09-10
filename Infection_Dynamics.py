import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# ================================= MODULE III: DYNAMIC TRANSMISSION SIMULATION ==============================
# This module performs the daily dynamic simulation of infection transmission across the simulation population.
# Based on input parameters and infection probabilities, it tracks individual transitions between health states
# over time and simulates within-host selection. Results include incidence summaries and plots for each simulated year.

# ----------------------------------------- CONFIGURATION --------------------------------------------------
# Define input and output directories
base_path_in = "/Users/erinndhulster/Desktop/Simulation AMR/Transmission/Set-up/"
base_path_out = "/Users/erinndhulster/Desktop/Simulation AMR/Transmission/Dynamics/"
base_pop_path = "/Users/erinndhulster/Desktop/Simulation AMR/Population Model/"

# Seasonality per month, relative to average infection load
SEASONALITY = {
    1:  1.18,  # Jan
    2:  1.13,  # Feb
    3:  1.20,  # Mar
    4:  0.81,  # Apr
    5:  0.87,  # May
    6:  0.66,  # Jun
    7:  0.49,  # Jul
    8:  0.40,  # Aug
    9:  0.92,  # Sep
    10: 1.18,  # Oct
    11: 1.32,  # Nov
    12: 1.53   # Dec
}

# Set parameters for illness duration, resistance development, and immunity
duration_illness_sens = 7
duration_immunity = 180
ab_reduction_factor = 0.8
resistance_extension_factor = 1.56
P_within_host_selec = 0.12  # probability of within-host selection after inappropriate treatment
w_prevalence = 0.2
gamma_prevalence = 0.5
epsilon_prevalence = 0.05

# Antibiotic use correctness
P_APPROPRIATE = 0.63
P_INAPPROPRIATE = 0.37

# Infection probabilities (uniform across individuals)
BASE_INF_PROB_NONRES = 0.0012
BASE_INF_PROB_RES = 0.00003

# Define the health states used in the simulation
state_columns = [
    "Susceptible",
    "I_NR_no_AB", "I_NR_AB_appropriate", "I_NR_AB_inappropriate",
    "I_R_no_AB", "I_R_AB_appropriate", "I_R_AB_inappropriate",
    "Immune"
]

# Function to generate all necessary file paths for a given simulation year
def generate_file_paths(year):
    return {
        "setup": os.path.join(base_path_in, f"Simulation_Setup_Year_{year}_Days_1_to_365.csv"),
        "output_csv": os.path.join(base_path_out, f"Simulation_Dynamics_Year_{year}_Days_1_to_365_herwerkt.csv"),
        "counts_csv": os.path.join(base_path_out, f"Daily_Counts_Year{year}_herwerkt.csv"),
        "baseline": os.path.join(base_pop_path, f"Year_{year}_population_herwerkt.csv")
    }

# Initialize lists to store simulation results
all_outputs = []
all_counts = []

# Core function to run the transmission simulation for a single year
def run_simulation_for_year(year):
    print(f"\n▶️ Start simulatie jaar {year}")
    paths = generate_file_paths(year)

    # Load input simulation data (baseline infection probabilities are fixed and not loaded)
    input_df = pd.read_csv(paths["setup"], sep=';', decimal=',')

    # Prepare simulation structures
    days = sorted(input_df["Day"].unique())
    identifiers = input_df["Identifier"].unique()
    id_to_idx = {ident: i for i, ident in enumerate(identifiers)}

    n_ind, n_days, n_states = len(identifiers), len(days), len(state_columns)
    state_arr = np.zeros((n_ind, n_days, n_states), dtype=int)
    nonres_inf = np.zeros((n_ind, n_days), dtype=int)
    res_inf = np.zeros((n_ind, n_days), dtype=int)

    # Initialize state matrix with data from setup file
    for row in input_df.itertuples(index=False):
        i = id_to_idx.get(row.Identifier)
        if i is None:
            continue
        t = row.Day - 1
        for k, st in enumerate(state_columns):
            val = getattr(row, st)
            if pd.notna(val):
                state_arr[i, t, k] = int(val)
        if hasattr(row, "NonRes_Inf") and row.NonRes_Inf == 1:
            nonres_inf[i, t] = 1
        if hasattr(row, "Res_Inf") and row.Res_Inf == 1:
            res_inf[i, t] = 1

    # Preload AB probabilities once per year (also needed if only resistant infections occur on a day)
    ab_probs = pd.read_csv(os.path.join(base_pop_path, f"Year_{year}_population_herwerkt.csv"),
                           sep=';', decimal=',')
    ab_map = ab_probs.set_index("Identifier")["AB_Probability"].to_dict()

    # Start daily simulation loop
    for t, day in enumerate(days):
        # Use fixed baseline probabilities on the first day
        if t == 0:
            p_non = np.full(n_ind, BASE_INF_PROB_NONRES)
            p_res = np.full(n_ind, BASE_INF_PROB_RES)
        else:
            # Update probabilities based on current prevalence and seasonality
            prev = state_arr[:, t - 1, :]
            N = n_ind
            preval_non = prev[:, 1:4].sum() / N
            preval_res = prev[:, 4:7].sum() / N
            pf_non = epsilon_prevalence + (1 - epsilon_prevalence) * (preval_non ** gamma_prevalence)
            pf_res = epsilon_prevalence + (1 - epsilon_prevalence) * (preval_res ** gamma_prevalence)
            # Determine month based on simulation day (approximate; adjust if needed)
            day_of_year = day
            month = (day_of_year - 1) // 30 + 1
            month = min(month, 12)  # Cap at 12
            season = SEASONALITY.get(month, 1)
            p_non = BASE_INF_PROB_NONRES * ((1 - w_prevalence) + w_prevalence * pf_non) * season
            p_res = BASE_INF_PROB_RES * ((1 - w_prevalence) + w_prevalence * pf_res) * season
            p_non = np.full(n_ind, p_non)
            p_res = np.full(n_ind, p_res)

        # Sample new infections
        draw_non = np.random.binomial(1, p_non)
        draw_res = np.random.binomial(1, p_res)

        # Resolve double infections: only allow one type of infection per person
        both = draw_non & draw_res
        flip = np.random.rand(n_ind) < 0.5
        draw_non[both & flip] = 0
        draw_res[both & ~flip] = 0

        # Prevent reinfection of already infected individuals
        already = state_arr[:, t, :].sum(axis=1) > 0
        draw_non[already] = 0
        draw_res[already] = 0

        # ——— 2-staps antibiotica-toewijzing voor niet-resistente infecties ———
        idx_nr = np.where(draw_non == 1)[0]
        if idx_nr.size:
            # 1) AB-kans per individu
            p_ab = np.array([ab_map.get(ident, 0.4) for ident in identifiers])[idx_nr]
            got_ab = np.random.binomial(1, p_ab)

            idx_no_ab = idx_nr[got_ab == 0]
            idx_ab = idx_nr[got_ab == 1]

            # 2) binnen de AB-groep: appropriate vs inappropriate
            split = np.random.choice([0, 1], size=idx_ab.size, p=[P_APPROPRIATE, P_INAPPROPRIATE])
            idx_app = idx_ab[split == 0]
            idx_inapp = idx_ab[split == 1]

            # – geen AB:
            if idx_no_ab.size:
                e = min(n_days, t + duration_illness_sens)
                state_arr[idx_no_ab, t:e, state_columns.index("I_NR_no_AB")] = 1
                state_arr[idx_no_ab, e:e + duration_immunity, state_columns.index("Immune")] = 1
                nonres_inf[idx_no_ab, t] = 1

            # – passende AB:
            if idx_app.size:
                e = min(n_days, t + int(round(duration_illness_sens * ab_reduction_factor)))
                state_arr[idx_app, t:e, state_columns.index("I_NR_AB_appropriate")] = 1
                state_arr[idx_app, e:e + duration_immunity, state_columns.index("Immune")] = 1
                nonres_inf[idx_app, t] = 1

            # – ongepaste AB (met kans op within-host selection):
            if idx_inapp.size:
                # fase 1: non-resistant ziekteperiode
                state_arr[idx_inapp, t:t + duration_illness_sens, state_columns.index("I_NR_AB_inappropriate")] = 1
                nonres_inf[idx_inapp, t] = 1

                # fase 2: within-host selection
                flags2 = np.random.binomial(1, P_within_host_selec, size=idx_inapp.size)
                idx_whs = idx_inapp[flags2 == 1]
                idx_rec = idx_inapp[flags2 == 0]

                if idx_whs.size:
                    s2 = t + duration_illness_sens
                    if s2 < n_days:  # guard
                        e2 = min(n_days, s2 + int(duration_illness_sens * resistance_extension_factor))
                        state_arr[idx_whs, s2:e2, state_columns.index("I_R_no_AB")] = 1
                        res_inf[idx_whs, s2] = 1
                        state_arr[idx_whs, e2:min(n_days, e2 + duration_immunity),
                                  state_columns.index("Immune")] = 1

                if idx_rec.size:
                    r0 = t + duration_illness_sens
                    if r0 < n_days:
                        state_arr[idx_rec, r0:r0 + duration_immunity,
                                  state_columns.index("Immune")] = 1

        # ——— 2-staps antibiotica-toewijzing voor resistente infecties ———
        idx_r = np.where((draw_res == 1) & (draw_non == 0))[0]
        if idx_r.size:
            # AB-kans per individu
            p_ab_r = np.array([ab_map.get(ident, 0.4) for ident in identifiers])[idx_r]
            got_ab_r = np.random.binomial(1, p_ab_r)

            idx_no_r = idx_r[got_ab_r == 0]
            idx_ab_r = idx_r[got_ab_r == 1]

            # split appropriate/inappropriate
            split_r = np.random.choice([0, 1], size=idx_ab_r.size, p=[P_APPROPRIATE, P_INAPPROPRIATE])
            idx_app_r = idx_ab_r[split_r == 0]
            idx_inapp_r = idx_ab_r[split_r == 1]

            # geen AB bij R
            if idx_no_r.size:
                e = min(n_days, t + int(round(duration_illness_sens * resistance_extension_factor)))
                state_arr[idx_no_r, t:e, state_columns.index("I_R_no_AB")] = 1
                state_arr[idx_no_r, e:e + duration_immunity, state_columns.index("Immune")] = 1
                res_inf[idx_no_r, t] = 1

            # passende AB bij R
            if idx_app_r.size:
                e = min(n_days,
                        t + int(round(duration_illness_sens * resistance_extension_factor * ab_reduction_factor)))
                state_arr[idx_app_r, t:e, state_columns.index("I_R_AB_appropriate")] = 1
                state_arr[idx_app_r, e:e + duration_immunity, state_columns.index("Immune")] = 1
                res_inf[idx_app_r, t] = 1

            # onpaste AB bij R
            if idx_inapp_r.size:
                e = min(n_days, t + int(round(duration_illness_sens * resistance_extension_factor)))
                state_arr[idx_inapp_r, t:e, state_columns.index("I_R_AB_inappropriate")] = 1
                state_arr[idx_inapp_r, e:e + duration_immunity, state_columns.index("Immune")] = 1
                res_inf[idx_inapp_r, t] = 1

        idx_s = np.where((draw_non == 0) & (draw_res == 0) & (~already))[0]
        state_arr[idx_s, t, state_columns.index("Susceptible")] = 1

        if not np.all(state_arr[:, t, :].sum(axis=1) == 1):
            raise ValueError(f"State sum mismatch op dag {day}")

    # Convert simulation output to DataFrame
    ind_idx = np.repeat(identifiers, n_days)
    day_idx = np.tile(days, n_ind)

    flat_state = state_arr.reshape(-1, n_states)
    flat_nonres = nonres_inf.ravel()
    flat_res = res_inf.ravel()

    out_df = pd.DataFrame(flat_state, columns=state_columns)
    out_df["Identifier"] = ind_idx
    out_df["Day"] = day_idx
    out_df["Year"] = year
    out_df["NonRes_Inf"] = flat_nonres
    out_df["Res_Inf"] = flat_res
    out_df.to_csv(paths["output_csv"], index=False)
    counts = out_df.groupby("Day")[state_columns].sum().reset_index()
    counts.to_csv(paths["counts_csv"], index=False)

    # Store results
    all_outputs.append(out_df)
    counts["Year"] = year
    all_counts.append(counts)

    # Print yearly infection summary
    print(f"\n📊 Year {year}: Total incidence")
    print(f" - Non-resistant infections: {out_df['NonRes_Inf'].sum()}")
    print(f" - Resistant infections:     {out_df['Res_Inf'].sum()}")

    nr_inc = out_df[out_df["NonRes_Inf"] == 1]
    print("\nNon-resistant infections:")
    print(f" - No AB:     {(nr_inc['I_NR_no_AB'] == 1).sum()}")
    print(f" - Appropriate AB:   {(nr_inc['I_NR_AB_appropriate'] == 1).sum()}")
    print(f" - Inappropriate AB: {(nr_inc['I_NR_AB_inappropriate'] == 1).sum()}")

    r_inc = out_df[out_df["Res_Inf"] == 1]
    print("\nResistant infections:")
    print(f" - No AB:     {(r_inc['I_R_no_AB'] == 1).sum()}")
    print(f" - Appropriate AB:   {(r_inc['I_R_AB_appropriate'] == 1).sum()}")
    print(f" - Inappropriate AB: {(r_inc['I_R_AB_inappropriate'] == 1).sum()}")

    # Estimate number of within-host selection cases (transition I_NR_AB_inappropriate -> I_R_no_AB)
    idx_inapp = state_columns.index("I_NR_AB_inappropriate")
    idx_rnoab = state_columns.index("I_R_no_AB")
    within_host_selection_count = 0
    for i in range(n_ind):
        for t2 in range(1, n_days):
            if state_arr[i, t2 - 1, idx_inapp] == 1 and state_arr[i, t2, idx_rnoab] == 1:
                within_host_selection_count += 1
                break

    # ========================== SETUP FOR NEXT YEAR ==========================
    if year < 5:  # Adjust this if you want to simulate more than 5 years
        # 1. Load final state of the current year
        df_end = out_df[out_df["Day"] == out_df["Day"].max()].copy()
        ids_current = set(df_end["Identifier"])

        # 2. Load population file for the next year
        pop_next_path = os.path.join(base_pop_path, f"Year_{year + 1}_population_herwerkt.csv")
        pop_next = pd.read_csv(pop_next_path, sep=';', decimal=',')
        ids_next = set(pop_next["Identifier"])

        # 3. Determine continuing, new, and deceased individuals
        deceased_ids = ids_current - ids_next
        new_ids = ids_next - ids_current
        continuing_ids = ids_current & ids_next

        print(f"\n📦 Setup for year {year+1}:")
        print(f" - Deceased:   {len(deceased_ids)}")
        print(f" - New:        {len(new_ids)}")
        print(f" - Continuing: {len(continuing_ids)}")

        # 4. For continuing individuals: copy end state to day 1 of next year
        continuing_df = df_end[df_end["Identifier"].isin(continuing_ids)].copy()
        continuing_df["Year"] = year + 1
        continuing_df["Day"] = 1
        continuing_df["NonRes_Inf"] = 0
        continuing_df["Res_Inf"] = 0

        # 5. For new individuals: create empty rows with Susceptible = 1 on day 1
        columns = out_df.columns
        new_rows = []
        for _, row in pop_next.iterrows():
            if row["Identifier"] not in new_ids:
                continue
            empty = {col: np.nan for col in columns}
            empty.update({
                "Identifier": row["Identifier"],
                "Year": year + 1,
                "Day": 1,
                "Susceptible": 1,
                "NonRes_Inf": 0,
                "Res_Inf": 0,
            })
            new_rows.append(empty)
        new_df = pd.DataFrame(new_rows)

        # 6. Combine both groups
        setup_df = pd.concat([continuing_df, new_df], ignore_index=True)

        # 7. Add empty rows for days 2–365
        extra_rows = []
        for _, r in setup_df.iterrows():
            for d in range(2, 366):
                empty = {col: np.nan for col in columns}
                empty.update({"Identifier": r["Identifier"], "Year": year + 1, "Day": d})
                extra_rows.append(empty)
        setup_df = pd.concat([setup_df, pd.DataFrame(extra_rows)], ignore_index=True)

        # 8. Sort and export setup for next year
        setup_df.sort_values(by=["Identifier", "Day"], inplace=True)
        outpath = os.path.join(base_path_in, f"Simulation_Setup_Year_{year + 1}_Days_1_to_365.csv")
        setup_df.to_csv(outpath, index=False, sep=';', decimal=',')
        print(f"✅ Setup for year {year+1} saved to: {outpath}")

    print(f"\n💥 Within-host selection cases in year {year}: {within_host_selection_count}")
    print(f"✅ Year {year} completed.")


for year in range(1, 6):
    run_simulation_for_year(year)

# ----------------------------------------- MERGE AND EXPORT OUTPUTS ----------------------------------------
# Combine all individual outputs and save to CSV
combined_df = pd.concat(all_outputs, ignore_index=True)
combined_df.to_csv(os.path.join(base_path_out, "Simulation_Dynamics_Years_1_to_2.csv"), index=False)

# Combine daily state counts and save to CSV
combined_counts = pd.concat(all_counts, ignore_index=True)
combined_counts.to_csv(os.path.join(base_path_out, "Daily_Counts_Years_1_to_2.csv"), index=False)

# ----------------------------------------- VISUALIZE STATE TRAJECTORIES -------------------------------------
# Generate and save line plots of health states for each year
for year in range(1, 6):
    year_counts = combined_counts[combined_counts["Year"] == year]
    plt.figure(figsize=(14, 6))
    for st in state_columns[1:-1]:  # Optionally exclude "Susceptible" and "Immune"
        plt.plot(year_counts["Day"], year_counts[st], label=st)
    plt.xlabel("Day")
    plt.ylabel("Number of individuals")
    plt.title(f"Health states over time (Year {year})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(base_path_out, f"plot_year_{year}.png"))
    plt.close()

# ----------------------------------------- WEEKLY INCIDENCE EXPORT ------------------------------------------
# Export weekly incidence for each state to CSV
incidentie_states = ["I_NR_no_AB", "I_NR_AB_appropriate", "I_NR_AB_inappropriate",
                     "I_R_no_AB", "I_R_AB_appropriate", "I_R_AB_inappropriate"]

# Function to center weeks (week 1 = mid-year)
def shift_weeks_centered(df, total_weeks=52):
    mid_week = total_weeks // 2 + 1
    df["Shifted_Week"] = ((df["Week"] - mid_week) % total_weeks) + 1
    return df

alle_jaren_resultaat = []

# Calculate weekly incidence per health state
for year in range(1, 6):
    df = combined_df[combined_df["Year"] == year].copy()
    df["Week"] = ((df["Day"] - 1) // 7) + 1

    for state in incidentie_states:
        df_sorted = df.sort_values(by=["Identifier", "Day"]).copy()
        df_sorted["New"] = df_sorted.groupby("Identifier")[state].diff().fillna(df_sorted[state])
        df_sorted["New"] = (df_sorted["New"] == 1).astype(int)

        weekly_counts = df_sorted.groupby("Week")["New"].sum().reset_index(name="Count")
        all_weeks = pd.DataFrame({"Week": np.arange(1, 53)})
        weekly_counts = all_weeks.merge(weekly_counts, on="Week", how="left").fillna(0)
        weekly_counts["Count"] = weekly_counts["Count"].astype(int)

        weekly_counts = shift_weeks_centered(weekly_counts)
        weekly_counts["State"] = state
        weekly_counts["Year"] = year

        alle_jaren_resultaat.append(weekly_counts)

# Combine all years and save result
result = pd.concat(alle_jaren_resultaat, ignore_index=True)
result.to_csv("/Users/erinndhulster/Desktop/Simulation AMR/Transmission/Dynamics/incidenties_per_week_per_state.csv", index=False)

# ----------------------------------------- WEEKLY INCIDENCE PLOTS -------------------------------------------
# Create line plots of incidence per state, reordered from week 25 to 24
ordered_weeks = list(range(25, 53)) + list(range(1, 25))

for year in result["Year"].unique():
    df_year = result[result["Year"] == year]
    plt.figure(figsize=(14, 6))

    for state in df_year["State"].unique():
        df_state = df_year[df_year["State"] == state][["Week", "Count"]].copy()
        full_weeks = pd.DataFrame({"Week": ordered_weeks})
        df_state = full_weeks.merge(df_state, on="Week", how="left").fillna(0)
        df_state["Count"] = df_state["Count"].astype(float)

        x_vals = list(range(len(ordered_weeks)))
        y_vals = df_state["Count"].tolist()

        plt.plot(x_vals, y_vals, label=state)

    plt.xticks(ticks=range(len(ordered_weeks)), labels=ordered_weeks, rotation=45)
    plt.xlabel("Week")
    plt.ylabel("Incidence per 100,000 PY")
    plt.title(f"Incidence per state per 100,000 person-years (PY) – Year {year}")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"/Users/erinndhulster/Desktop/Simulation AMR/Transmission/Dynamics/incidentieplot_week25_start_year_{year}.png")
    plt.close()
