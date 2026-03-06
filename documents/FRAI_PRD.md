# Product Requirement Document (PRD) - FPAI System

## 1. Business Objective
To build a quantitative analysis tool for football betting that identifies value bets (EV > 0) by predicting match outcomes (1X2, Over/Under, Corners) using historical and real-time data.

## 2. Target Markets
* **Initial**: English Premier League (EPL).
* **Expansion**: Major European leagues (Bundesliga, La Liga, etc.).
* **Betting Types**: Match Result (1X2), Total Goals (Over/Under 2.5), and Corners.

## 3. Core Logic & Rules
### 3.1 Betting Decision (The Value Formula)
A "Value Bet" is identified when the model's predicted probability ($P$) multiplied by the bookmaker's decimal odds ($O$) is greater than 1.
$$EV = (P \times O) - 1$$
* **Decision**: Only output recommendations where $EV > 0$.

### 3.2 Data Integrity Rules
* **Missing Data Policy**: If critical data points (e.g., closing odds, key match stats) are missing, the system MUST skip the match to avoid skewed predictions.
* **League Isolation**: Feature calculations (rolling averages) should be performed within the same league context to maintain data consistency.

### 3.3 Evaluation & Backtesting
* **Rolling Backtest**: The system must support a rolling window approach. Use 1 year of data for the backtest window, retraining the model as new match results become available.
* **Performance Metrics**: ROI, Win Rate, and Brier Score.

## 4. Workflow
1.  **Ingestion**: Fetch historical/live data.
2.  **Processing**: Calculate rolling features and validate data.
3.  **Prediction**: Generate probabilities for upcoming matches.
4.  **Action**: Calculate EV and display recommendations via CLI/Dashboard.