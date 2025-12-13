import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("runs/cf8b850e-d8fb-4d2d-869c-c7746868e5c8/samples_mpiris.csv")
valid = df[df["validity"] == 0].copy()

plt.scatter(valid["y_norm"], valid["target_y_px"], s=2)
plt.xlabel("y_norm")
plt.ylabel("target_y_px")
plt.show()