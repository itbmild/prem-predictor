# Premier League Predictor
An end to end Machine Learning Pipeline that predicts premier league match outcomes using historical data.

# Architecture
- Data Ingestion (raw CSV)
- Data Cleaning and Preprocessing
- Feature Engineering
- Model Training
- Model Evaluation
- Model registry / logging

## Data Ingestion
Uses historical data from the 2008/2009 season through to the 2024/2025 season.
Train/val/test split proportions can be adjusted in config/config.yaml.

## Features
Features are split into 3 categories; Rolling Window, Head to Head, Previous Season.

### Rolling Window (last X games)
These features are aggregations (sums or means) over the previous 5 games of the chosen team for a specific metric.

List of Rolling window features:
  - Form: Average points earned
  - AVG_YC: Average Yellow cards
  - AVG_RC: Average Red cards

### Previous Season

Features based on the overall performance of chosen team in the previous season.

List of Previous Season features:
  - Points (PTS)
  - Wins (W)
  - Draws (D)
  - Losses (L)
  - Goals For (GF)
  - Goals Against (GA)
  - Yellow Cards (SSN_YC)
  - Red Cards (SSN_RC)
  - Average Shots on target per game (AVG_SOT)

### Head To Head

Features based on the chosen team against a specific opponent.
For example, if a specific match is Manchester City vs Arsenal, these features are aggregated over the previous X games between Manchester City and Arsenal.

Head to Head features:
  - Form (average)
  - Goals (average)




