import pandas as pd
import numpy as np
import os

# =============================================== MODULE I: POPULATION GENERATION SETUP ===============================================
# This module generates a dynamic, demographically realistic simulation population for use in the ARON model.

# =============================================== CONFIGURATION ===============================================

# Set the output directory for storing population simulation files
OUTPUT_FOLDER = "/Users/erinndhulster/Desktop/Simulation AMR/Population Model/"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ----------------- Simulation Setup -----------------
NUM_PEOPLE = 10000  # Number of individuals to simulate
START_AVG_AGE = 40  # Target average age of the population
NUM_YEARS = 5  # Number of simulation years
MORTALITY_RATE = 0.0096  # Annual mortality rate
BIRTH_RATE = 0.0099  # Annual birth rate

# ----------------- Demographic Distributions -----------------
GENDER_DISTRIBUTION = [0.4948, 0.5052]  # Male, Female
AGE_DISTRIBUTION = {
    (0, 5): 5.24, (5, 10): 5.30, (10, 15): 5.33, (15, 20): 5.35, (20, 25): 5.38,
    (25, 30): 5.36, (30, 35): 5.41, (35, 40): 5.43, (40, 45): 5.52, (45, 50): 5.48,
    (50, 55): 5.50, (55, 60): 5.52, (60, 65): 5.47, (65, 70): 5.38, (70, 75): 5.29,
    (75, 80): 5.10, (80, 85): 4.80, (85, 90): 4.37, (90, 95): 3.30, (95, 100): 1.47
}

# Antibiotic prescription probabilities by age and gender
AB_PROBABILITIES = {
    (0, 12, 0): 0.3567,  # Male 0-12
    (13, 120, 0): 0.3752,  # Male 13+
    (0, 12, 1): 0.3364,  # Female 0-12
    (13, 120, 1): 0.3828   # Female 13+
}

# ========================================== DEMOGRAPHIC SIMULATOR ==========================================

def generate_ages(n, target_avg):
    ages = []
    for (mn, mx), pct in AGE_DISTRIBUTION.items():
        count = int(round(n * pct / 100))
        ages += [np.random.randint(mn, mx) + np.random.rand() for _ in range(count)]
    ages = ages[:n]
    cur = np.mean(ages)
    while abs(cur - target_avg) > 0.01:
        i = np.random.randint(0, n)
        ages[i] += 1 if cur < target_avg else -1
        ages[i] = max(0, min(99.999, ages[i]))
        cur = np.mean(ages)
    return ages

def assign_ab_probability(age, gender):
    for (min_age, max_age, g), prob in AB_PROBABILITIES.items():
        if min_age <= age <= max_age and gender == g:
            return prob
    return 0.0

def create_initial_population(n):
    df = pd.DataFrame({
        "Identifier": np.arange(1, n + 1),
        "Age": generate_ages(n, START_AVG_AGE),
        "Gender": np.random.choice([0, 1], size=n, p=GENDER_DISTRIBUTION),
        "Mortality": np.zeros(n, dtype=int)
    })
    df["AB_Probability"] = df.apply(lambda row: assign_ab_probability(row["Age"], row["Gender"]), axis=1)
    olds = df[df["Age"] >= 85].index
    deaths = int(n * MORTALITY_RATE)
    dead_idx = np.random.choice(olds if len(olds) >= deaths else df.index, deaths, replace=False)
    df.loc[dead_idx, "Mortality"] = 1
    return df

def simulate_year(pop):
    pop = pop[pop["Mortality"] == 0].copy()
    pop["Age"] += 1
    olds = pop[pop["Age"] >= 85].index
    deaths = int(len(pop) * MORTALITY_RATE)
    dead_idx = np.random.choice(olds if len(olds) >= deaths else pop.index, deaths, replace=False)
    pop.loc[dead_idx, "Mortality"] = 1
    births = int(len(pop) * BIRTH_RATE)
    new = pd.DataFrame({
        "Identifier": np.arange(pop["Identifier"].max() + 1, pop["Identifier"].max() + 1 + births),
        "Age": np.random.rand(births),
        "Gender": np.random.choice([0, 1], size=births, p=GENDER_DISTRIBUTION),
        "Mortality": np.zeros(births, dtype=int)
    })
    new["AB_Probability"] = new.apply(lambda row: assign_ab_probability(row["Age"], row["Gender"]), axis=1)
    pop["AB_Probability"] = pop.apply(lambda row: assign_ab_probability(row["Age"], row["Gender"]), axis=1)
    return pd.concat([pop, new], ignore_index=True)

# ========================================== RUN SIMULATION ==========================================

population = create_initial_population(NUM_PEOPLE)
for yr in range(1, NUM_YEARS + 1):
    population = simulate_year(population)
    population.to_csv(os.path.join(OUTPUT_FOLDER, f"Year_{yr}_population_herwerkt_ARON.csv"), index=False, sep=';', decimal=',')
    print(f"✔ Year {yr} saved.")

print("✔ All population files created.")