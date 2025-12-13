
import pygame, time
class StimEngine:
    def __init__(self, width=1280, height=720, marker_px=14):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        self.w, self.h, self.r = width, height, marker_px//2
        self.active = None
    def show_point(self, x_norm: float, y_norm: float, color=(255,255,255)):
        self.active = (int(x_norm*self.w), int(y_norm*self.h))
        self._draw(color)
    def hide(self):
        self.active = None
        self._draw()
    def _draw(self, color=(255,255,255)):
        self.screen.fill((0,0,0))
        if self.active:
            pygame.draw.circle(self.screen, color, self.active, self.r)
        pygame.display.flip()
    def tick(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); raise SystemExit
        time.sleep(0.005)
