import os
import time
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame
import dataclasses
from dataclasses import dataclass
import cv2

@dataclass
class StimConfig:
    width: int = 1280
    height: int = 720
    marker_px: int = 16
    bg: tuple[int, int, int] = (255, 255, 255)  # white background
    fg: tuple[int, int, int] = (255, 0, 0)      # red marker
    fullscreen: bool = False

class StimEngine:
    def __init__(self, cfg: StimConfig = StimConfig()):
        pygame.init()
        pygame.event.clear()
        fullscreen_flags = pygame.NOFRAME | pygame.SCALED
        info = pygame.display.Info()

        if cfg.fullscreen:
            flags = fullscreen_flags
            req_w, req_h = info.current_w or cfg.width, info.current_h or cfg.height
        else:
            flags = 0
            req_w, req_h = cfg.width, cfg.height
            req_w = min(req_w, info.current_w)
            req_h = min(req_h, info.current_h)

        self.screen = pygame.display.set_mode((req_w, req_h), flags)
        pygame.event.clear()
        real_w, real_h = self.screen.get_size()
        self.cfg = dataclasses.replace(cfg, width=real_w, height=real_h)
        pygame.display.set_caption("Calibration")

        auto_marker = int(min(self.cfg.width, self.cfg.height) * 0.03)
        self.marker_px = max(16, auto_marker, self.cfg.marker_px)

        self.active_xy_px: tuple[int,int] | None = None
        try:
            pygame.font.init()
            self._font = pygame.font.SysFont("Segoe UI", 20)
        except Exception:
            self._font = None

    @property
    def size(self) -> tuple[int,int]:
        return self.screen.get_size()

    def _draw(self):
        self.screen.fill(self.cfg.bg)
        if self.active_xy_px:
            radius = max(2, self.marker_px // 2)
            pygame.draw.circle(self.screen, self.cfg.fg, self.active_xy_px, radius)
        pygame.display.flip()

    def _draw_ready(
        self,
        msg: str,
        frame_surface=None,
        frame_scale: float = 0.68,
        head_scale: float = 0.42,
        hint: str | None = None,
        show_alignment_overlay: bool = True,
    ):
        w, h = self.screen.get_size()
        if frame_surface:
            self.screen.blit(frame_surface, (0, 0))
        else:
            self.screen.fill((10, 12, 20))

        frame_w = int(w * max(0.4, min(0.9, frame_scale)))
        frame_h = int(h * max(0.4, min(0.9, frame_scale)))
        x0 = (w - frame_w) // 2
        y0 = max(0, (h - frame_h) // 2 - int(h * 0.05))  # lift frame slightly up

        if show_alignment_overlay:
            overlay = pygame.Surface((w, h), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 0))
            pygame.draw.rect(overlay, (56, 189, 248, 70), (x0 - 12, y0 - 12, frame_w + 24, frame_h + 24), width=1, border_radius=22)
            pygame.draw.rect(overlay, (34, 197, 94, 180), (x0, y0, frame_w, frame_h), width=3, border_radius=18)
            # head silhouette (ellipse)
            head_w = int(frame_w * max(0.28, min(0.9, head_scale * 0.85)))
            head_h = int(frame_h * max(0.42, min(1.0, head_scale + 0.16)))
            hx0 = (w - head_w) // 2
            hy0 = y0 + int(frame_h * 0.12)
            pygame.draw.ellipse(overlay, (255, 255, 255, 35), (hx0, hy0, head_w, head_h), width=2)
            self.screen.blit(overlay, (0, 0))

        if self._font:
            text = self._font.render(msg, True, (226, 232, 240))
            rect = text.get_rect(center=(w // 2, y0 + frame_h + 32))
            self.screen.blit(text, rect)
            hint_text = hint or "Press SPACE or ENTER to begin"
            hint = self._font.render(hint_text, True, (148, 163, 184))
            rect2 = hint.get_rect(center=(w // 2, rect.bottom + 18))
            self.screen.blit(hint, rect2)
        pygame.display.flip()

    def wait_for_ready(
        self,
        msg: str = "Fit your face in the frame and press Space to start",
        cam_index: int | None = 0,
        mirror_preview: bool = False,
        unmirror_cam: bool = True,
        frame_scale: float = 0.82,
        head_scale: float = 0.6,
        auto_start_ms: int | None = None,
        show_alignment_overlay: bool = True,
        camera_source: str = "webcam",
    ):
        """Blocks until space/enter is pressed. Shows camera preview with silhouette if possible. ESC quits."""
        waiting = True
        clock = pygame.time.Clock()
        if auto_start_ms is not None and auto_start_ms <= 0:
            return
        deadline_ms = None
        if auto_start_ms is not None:
            deadline_ms = time.time() * 1000 + auto_start_ms

        cap = None
        frame_surface_cache: pygame.Surface | None = None
        if camera_source == "daheng":
            try:
                from eyetrk.adapters.daheng_capture import DahengCapture
                cap = DahengCapture()
                if not cap.isOpened():
                    cap = None
            except Exception:
                cap = None
        elif cam_index is not None:
            try:
                cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap.release()
                    cap = cv2.VideoCapture(cam_index, cv2.CAP_ANY)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.height)
                    cap.set(cv2.CAP_PROP_FPS, 30)
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
                    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                cap = None

        try:
            while waiting:
                frame_surface = None
                if cap and cap.isOpened():
                    ok, frame = cap.read()
                    if ok:
                        if unmirror_cam and camera_source != "daheng":
                            frame = cv2.flip(frame, 1)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        if mirror_preview:
                            frame_rgb = cv2.flip(frame_rgb, 1)
                        try:
                            sw, sh = self.screen.get_size()
                            frame_rgb = cv2.resize(frame_rgb, (sw, sh), interpolation=cv2.INTER_NEAREST)
                            if frame_surface_cache is None:
                                frame_surface_cache = pygame.Surface((sw, sh))
                            pygame.surfarray.blit_array(frame_surface_cache, frame_rgb.transpose(1, 0, 2))
                            frame_surface = frame_surface_cache
                        except Exception:
                            frame_surface = None
                msg_text = msg
                hint_text = None
                if deadline_ms is not None:
                    remaining_ms = max(0, int(deadline_ms - time.time() * 1000))
                    if remaining_ms <= 0:
                        waiting = False
                        break
                    remaining_s = remaining_ms / 1000.0
                    msg_text = f"{msg} (auto-start in {remaining_s:.1f}s)"
                    hint_text = "Press SPACE or ENTER to start now"
                self._draw_ready(
                    msg_text,
                    frame_surface,
                    frame_scale=frame_scale,
                    head_scale=head_scale,
                    hint=hint_text,
                    show_alignment_overlay=show_alignment_overlay,
                )

                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        pygame.quit()
                        raise SystemExit
                    if ev.type == pygame.KEYDOWN:
                        if ev.key in (pygame.K_SPACE, pygame.K_RETURN):
                            waiting = False
                            break
                        if ev.key == pygame.K_ESCAPE:
                            pygame.quit()
                            raise SystemExit
                clock.tick(60)
        finally:
            if cap is not None:
                cap.release()
                time.sleep(0.25)

    def show_point(self, x_norm: float, y_norm: float):
        w, h = self.screen.get_size()
        x = int(round(x_norm * w))
        y = int(round(y_norm * h))
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        self.active_xy_px = (x, y)
        self._draw()

    def hide(self):
        self.active_xy_px = None
        self._draw()

    def tick(self, sleep_ms: int = 5):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                pygame.quit()
                raise SystemExit
        time.sleep(sleep_ms / 1000.0)

    def close(self):
        pygame.quit()

    def show_message(
        self,
        title: str,
        subtitle: str | None = None,
        duration_ms: int = 0,
        bg: tuple[int, int, int] = (10, 12, 20),
        fg: tuple[int, int, int] = (226, 232, 240),
        sub_fg: tuple[int, int, int] = (148, 163, 184),
    ):
        """Show a centered message for duration_ms (0 = just draw once)."""
        w, h = self.screen.get_size()
        self.screen.fill(bg)
        if self._font:
            text = self._font.render(title, True, fg)
            rect = text.get_rect(center=(w // 2, h // 2 - 10))
            self.screen.blit(text, rect)
            if subtitle:
                sub = self._font.render(subtitle, True, sub_fg)
                sub_rect = sub.get_rect(center=(w // 2, rect.bottom + 24))
                self.screen.blit(sub, sub_rect)
        pygame.display.flip()
        if duration_ms > 0:
            end = time.time() * 1000 + duration_ms
            clock = pygame.time.Clock()
            while time.time() * 1000 < end:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                        pygame.quit()
                        raise SystemExit
                clock.tick(30)
