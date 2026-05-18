# run_realtime.py
from __future__ import annotations

import csv
import json
import os
import time
import uuid

from eyetrk.adapters.mpiris import MpirisAdapter
from eyetrk.core.types import SAMPLE_CSV_FIELDS, CalibModel, Sample, sample_to_csv_row


def load_model(path: str) -> CalibModel | None:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return CalibModel(**data)


def main():
    session_id = str(uuid.uuid4())
    out_dir = os.path.join("runs", session_id)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "samples_mpiris.csv")

    adapter = MpirisAdapter()
    adapter.initialize(
        {
            "fps": 30,
            "camera_index": 0,
            "width": 1280,
            "height": 720,
            "flip_x": False,
            "flip_y": False,
            "head_comp": False,
            "min_conf": 0.7,
            "max_jump_norm": 0.12,
            "gain_x": 1.2,
            "gain_y": 1.0,
            "auto_gain": False,
            "cam_unmirror": True,
        }
    )

    model = load_model("models/calib_model.json")
    if model:
        adapter.set_external_model(model)
        print("[run_realtime] calib model loaded")
    else:
        print("[run_realtime] no calib model, x_px/y_px will stay normalized")

    f = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=SAMPLE_CSV_FIELDS)
    writer.writeheader()

    def on_sample(sample: Sample):
        writer.writerow(sample_to_csv_row(sample))

    adapter.start_stream(on_sample, session_id=session_id)
    print(f"[run_realtime] logging to {csv_path} | session_id={session_id}")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        adapter.stop()
        f.close()
        print("[run_realtime] stopped")


if __name__ == "__main__":
    main()
