import time
import cv2
import gxipy as gx


def main():
    dm = gx.DeviceManager()
    dev_num, dev_info_list = dm.update_device_list()

    if dev_num == 0:
        raise RuntimeError("Camera not found")

    cam = dm.open_device_by_index(1)

    cam.TriggerMode.set(gx.GxSwitchEntry.OFF)
    cam.ExposureAuto.set(gx.GxAutoEntry.OFF)
    cam.GainAuto.set(gx.GxAutoEntry.OFF)

    cam.PixelFormat.set(gx.GxPixelFormatEntry.MONO8)
    cam.ExposureTime.set(5000.0)
    cam.Gain.set(12.0)

    cam.stream_on()

    frame_count = 0
    t0 = time.perf_counter()
    last_print = t0

    try:
        while True:
            img = cam.data_stream[0].get_image(timeout=1000)
            if img is None:
                continue

            frame = img.get_numpy_array()
            if frame is None:
                continue

            frame_count += 1

            # Превью может быть медленнее реального FPS, это нормально
            preview = cv2.resize(frame, (960, 720))
            cv2.imshow("Daheng camera preview", preview)

            now = time.perf_counter()
            if now - last_print >= 1.0:
                fps = frame_count / (now - t0)
                print(f"Average FPS: {fps:.2f}")
                last_print = now

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cam.stream_off()
        cam.close_device()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()