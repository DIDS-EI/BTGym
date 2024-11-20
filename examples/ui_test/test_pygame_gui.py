import pygame
import pygame_gui

pygame.init()

window_size = (800, 600)
screen = pygame.display.set_mode(window_size)
manager = pygame_gui.UIManager(window_size)

clock = pygame.time.Clock()

# 创建按钮和文本框
button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((350, 275), (100, 50)),
                                    text='Submit',
                                    manager=manager)
text_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((350, 200), (100, 50)),
                                                manager=manager)

running = True
while running:
    time_delta = clock.tick(60)/1000.0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        manager.process_events(event)

    manager.update(time_delta)

    screen.fill((255, 255, 255))
    manager.draw_ui(screen)

    pygame.display.update()