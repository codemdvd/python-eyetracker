import pandas as pd
from eyetrk.calib.fitting import fit_dataframe

df = pd.read_csv("runs/cf8b850e-d8fb-4d2d-869c-c7746868e5c8/samples_mpiris.csv")

# берём только калибровку:
df = df[df["task_name"] == "calibration"]  # или по event/stim_id, как у тебя

# ВАЖНО: убиваем прямые таргеты
df = df.drop(columns=["target_x_px", "target_y_px"], errors="ignore")

out = fit_dataframe(df, width=1280, height=720, per_stim_median=False)
print(out.diag)
print(out.per_stim)



