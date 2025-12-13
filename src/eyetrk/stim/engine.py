import time
import pygame
import dataclasses
from dataclasses import dataclass

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
        fullscreen_flags = pygame.FULLSCREEN | pygame.SCALED
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
        real_w, real_h = self.screen.get_size()
        self.cfg = dataclasses.replace(cfg, width=real_w, height=real_h)
        pygame.display.set_caption("Calibration")

        auto_marker = int(min(self.cfg.width, self.cfg.height) * 0.02)
        self.marker_px = max(12, auto_marker, self.cfg.marker_px)

        self.active_xy_px: tuple[int,int] | None = None

    @property
    def size(self) -> tuple[int,int]:
        return self.screen.get_size()

    def _draw(self):
        self.screen.fill(self.cfg.bg)
        if self.active_xy_px:
            radius = max(2, self.marker_px // 2)
            pygame.draw.circle(self.screen, self.cfg.fg, self.active_xy_px, radius)
        pygame.display.flip()

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
