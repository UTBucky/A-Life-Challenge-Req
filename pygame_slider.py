import pygame

class Slider:
    def __init__(self, x, y, height, min_val, max_val, start_val):
        self.track_rect = pygame.Rect(x, y, 10, height)
        self.knob_height = 10
        self.knob_rect = pygame.Rect(
            x - 5,
            y + int((1 - (start_val - min_val) / (max_val - min_val)) * (height - self.knob_height)),
            20,
            self.knob_height
        )
        self.min_val = min_val
        self.max_val = max_val
        self.value = start_val
        self.dragging = False
        self.font = pygame.font.SysFont(None, 24)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.knob_rect.collidepoint(event.pos):
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            new_y = max(self.track_rect.top, min(event.pos[1] - self.knob_height // 2, self.track_rect.bottom - self.knob_height))
            self.knob_rect.y = new_y
            ratio = 1 - (self.knob_rect.y - self.track_rect.y) / (self.track_rect.height - self.knob_height)
            self.value = int(self.min_val + ratio * (self.max_val - self.min_val))

    def draw(self, surface):
        pygame.draw.rect(surface, (200, 200, 200), self.track_rect)
        pygame.draw.rect(surface, (100, 200, 255), self.knob_rect)

        # Draw tick rate label above the slider
        label = self.font.render(f"{self.value} FPS", True, (255, 255, 255))
        surface.blit(label, (self.track_rect.centerx - 25, self.track_rect.top - 25))

    def get_value(self):
        return self.value
