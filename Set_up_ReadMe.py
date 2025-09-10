# ARON Long-Term AMR Model — Part II: Initial Health State Setup

## Overview
This module assigns an initial health state to each individual in the simulated population on Day 1 of Year 1. For those in an infectious state, the health state is extended over time to reflect the natural course of illness and immunity. This setup provides the starting conditions for the dynamic infection transmission model in Part III.

## Objectives
- Distribute the population across mutually exclusive health states based on predefined proportions
- Simulate the day-by-day continuation of illness and immunity for infected individuals
- Identify individuals with selected resistance following inappropriate treatment

## Parameters (adjustable at the top of the script)
- `duration_illness_sens`: base illness duration (days) for non-resistant infections
- `ab_reduction_factor`: illness duration reduction for appropriately treated infections
- `resistance_extension_factor`: illness duration increase for resistant infections
- `duration_immunity`: duration of temporary immunity after recovery
- `P_within_host_selec`: probability that inappropriate antibiotic treatment results in resistance
- `mu_values`: initial distribution of the population across health states

## Key Steps
1. **Initial health state assignment**
   Each individual is randomly assigned to one of the eight predefined health states (e.g., susceptible, various infection types, immune) based on population-level proportions.

2. **Daily state progression**
   For individuals starting in an infectious state, their health state is extended over time:
   - Infection lasts a fixed or adjusted number of days (depending on resistance and treatment)
   - Followed by a temporary immune period
   - Transitions such as selected resistance are modeled if applicable

3. **Formatting output**
   Each individual's health state is tracked for every day of Year 1, with relevant infection flags (e.g., *NonRes_Inf*, *Res_Inf*) and resistance emergence.

## Output
A CSV file is generated for Year 1, in which:
- Individuals are distributed across health states on Day 1
- Those in infectious states have their health trajectory simulated for each day of the year
- Infection flags and resistance indicators are included

This file forms the input for the transmission dynamics simulation developed in Part III of the model.

Saved in `/Transmission/Set-up/`:
- `Simulation_Setup_Year_1_Days_1_to_365.csv`
