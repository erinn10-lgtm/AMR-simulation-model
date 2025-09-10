# ARON Long-Term AMR Model — Part I: Population Simulation

## Overview
This module generates a simulated population that reflects the Belgian demographic structure and evolves over a 5-year period. The resulting dataset forms the basis for estimating infection probabilities and simulating the long-term impact of the ARON intervention on antimicrobial resistance (AMR).

## Objectives
- Simulate a demographically realistic population (age, gender)
- Apply annual demographic changes: aging, mortality, and births
- Calculate individual-level probabilities of non-resistant and resistant infection based on population characteristics

## Parameters (adjustable at the top of the script)
- `NUM_PEOPLE`: total population size (e.g., 100,000)
- `START_AVG_AGE`: target average baseline age
- `NUM_YEARS`: number of simulation years
- `MORTALITY_RATE` and `BIRTH_RATE`: annual demographic change rates

## Key Steps
1. **Initial population creation**: based on Belgian age distribution,and gender.
2. **Annual updates**: simulates deaths (among oldest), aging, and births.
3. **Infection probability calculation**: assigns a risk score per individual using demographic modifiers.
4. **Export**: for each year, generates an Excel files with demographic data

## Output
Saved in `/Population Model/`:
- Year-by-year Excel files with demographic and infection probability data
