# 2026 F1 Championship Predictor

A machine learning pipeline that simulates the full 2026 Formula 1 season and predicts driver and constructor championship standings. Trained on real 2024 race data using the FastF1 API.

---

## Sample output

```
🏆 2026 DRIVER WORLD CHAMPIONSHIP - FINAL STANDINGS

🥇 1   RUS      Mercedes        439 pts
🥈 2   ANT      Mercedes        321 pts
🥉 3   NOR      McLaren         281 pts
   4   VER      Red Bull        226 pts
   5   LAW      Red Bull        184 pts
```

---

## How it works

1. **Data collection** — pulls lap times, positions, and race results from all 2024 races via the FastF1 API
2. **Feature engineering** — extracts driver performance metrics, team consistency, circuit-specific tendencies
3. **Model training** — trains a baseline ML model on 2024 results
4. **Season simulation** — simulates all 24 races of the 2026 calendar, applying regulation change adjustments
5. **Championship calculation** — aggregates points across all races to produce final standings

---

## Tech stack

- **Python** — core language
- **FastF1** — official F1 timing and telemetry data API
- **Pandas** — data processing and feature engineering
- **scikit-learn** — ML model training and prediction
- **Matplotlib** — performance visualizations

---

## Getting started

```bash
git clone https://github.com/roqaya25/f1-predictions
cd f1-predictions
pip install fastf1 pandas scikit-learn matplotlib
python championship_predictor.py
```

Note: First run downloads race data which may take a few minutes. Subsequent runs use cached data.

---

## Files

- `championship_predictor.py` — main simulation pipeline
- `f1_enhanced.py` — enhanced feature engineering
- `f1-1.py` — exploratory analysis and model experimentation
- `output.txt` — sample prediction output

---

## Notes

- Predictions are based on 2024 historical data with adjustments for 2026 regulation changes
- Re-run after each real 2026 race for updated predictions
- High uncertainty until pre-season testing data is available

---

## Author

**Roqaya Elrahwan** — Computer Engineering @ University of Toronto  
[GitHub](https://github.com/roqaya25)
