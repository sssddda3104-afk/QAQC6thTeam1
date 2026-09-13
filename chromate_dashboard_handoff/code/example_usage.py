from pathlib import Path
import pandas as pd
from dashboard_inference import current_snapshot, predict_final_lot, predict_early_warning

BASE = Path(__file__).resolve().parent.parent
raw = pd.read_csv(BASE / "data" / "dashboard_sensor_timeseries.csv")
raw["Timestamp"] = pd.to_datetime(raw["Timestamp"])

lot = raw[(raw["Date"] == "2021-09-08") & (raw["Lot"] == 20)].sort_values("Timestamp")

# 약 60% checkpoint(41개 point)
print(current_snapshot(lot.iloc[:41]))
print(predict_early_warning(lot.iloc[:41]))

# 완료 LOT
print(predict_final_lot(lot))
