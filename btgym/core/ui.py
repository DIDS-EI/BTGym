import time
import pygame
import pygame_gui
from btgym.utils.logger import log
# import matplotlib
# matplotlib.use('Agg')

class UI:
    def __init__(self, controller, simulator):
        self.controller = controller
        self.simulator = simulator

        pygame.init()

        window_size = (800, 600)
        self.screen = pygame.display.set_mode(window_size)
        self.manager = pygame_gui.UIManager(window_size)

        self.clock = pygame.time.Clock()
        self.last_step_time = time.time()
        self.last_log_time = time.time()
        self.step_interval = 1/60
        # 创建按钮和文本框
        self.button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((350, 275), (100, 50)),
            text='Submit', manager=self.manager)
        self.text_entry = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((350, 200), (100, 50)),
            manager=self.manager)
        self.text_entry.set_text("putting_shoes_on_rack")

    def init_trav_map(self):
        trav_map = self.simulator.get_trav_map()
        self.trav_map = trav_map


    def step(self):
        current_time = time.time()
        if current_time - self.last_step_time < self.step_interval:
            return
        self.last_step_time = current_time

        # self.update_info()
        self.render()

        # self.log_step_info()

    def log_step_info(self):
        current_time = time.time()
        if current_time - self.last_log_time < self.step_log_interval:
            return
        self.last_log_time = current_time
        # log(f"UI info: {self.info}")


    # def update_info(self):
    #     self.info['robot_pos'] = self.simulator.get_robot_pos()

    def render(self):
        time_delta = self.clock.tick(60)/1000.0
        for event in pygame.event.get():

            if event.type == pygame.USEREVENT:
                if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                    if event.ui_element == self.button:
                        self.on_submit()  # 调用关联的函数

            self.manager.process_events(event)

        self.manager.update(time_delta)
    
        self.screen.fill((255, 255, 255))
        self.manager.draw_ui(self.screen)

        pygame.display.update()

    def on_submit(self):
        text = self.text_entry.text
        log(f"UI: Button Submit {text}")

        try:
            task_index = int(text)
            self.simulator.load_behavior_task_by_index(task_index)
        except:
            if text == '':
                task_index = 0
            else:
                self.simulator.load_behavior_task_by_name(text)
        # self.simulator.load_task_by_index(task_index)
        self.controller.reset()
        self.controller.do_task(text)
