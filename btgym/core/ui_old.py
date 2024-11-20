import math
import time
from btgym.utils.logger import log
import pygame
import matplotlib.pyplot as plt
import matplotlib
import cv2
import numpy as np
import pygame_gui
matplotlib.use('Agg')

class UI:
    def __init__(self, controller, simulator):
        self.controller = controller
        self.simulator = simulator

        self.text_area_scale = 0.4375

        self.window_width = 800
        self.window_height = 800
        self.text_area_width = self.window_width * self.text_area_scale
        self.window_width = self.window_width + self.text_area_width

        self.last_ui_step_time = 0
        self.ui_step_interval = 1/60


        self.last_log_time = 0
        self.step_log_interval = 2

        self.init_pygame()
        log("UI step log interval: {}".format(self.step_log_interval))

        self.info = {'scene_name': 'default', 'robot_pos': [0,0,0]}
        self.info['scene_name'] = self.simulator.get_scene_name()

        # self.init_trav_map()

    def init_trav_map(self):
        trav_map = self.simulator.get_trav_map()
        self.trav_map = trav_map

    def init_pygame(self):
        pygame.init()
        pygame.display.init()

        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("OmniBT UI")
        self.clock = pygame.time.Clock()
        self.screen.fill((255, 255, 255))  # 设置白色背景
        pygame.display.flip()  # 更新显示

        # 初始化文本框和按钮
        self.input_box = pygame.Rect(10, 10, 140, 32)
        self.button_box = pygame.Rect(160, 10, 80, 32)
        self.button_color = (0, 128, 0)
        self.text = ''
        self.font = pygame.font.Font(None, 32)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    print(self.text)  # 在按下回车时打印文本框内容
                    self.text = ''
                elif event.key == pygame.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if self.button_box.collidepoint(event.pos):
                    print("Button clicked!")  # 在按钮被点击时执行操作


    def step(self):
        current_time = time.time()
        if current_time - self.last_ui_step_time < self.ui_step_interval:
            return
        self.last_ui_step_time = current_time


        self.update_info()
        self.render()

        self.log_step_info()

    def log_step_info(self):
        current_time = time.time()
        if current_time - self.last_log_time < self.step_log_interval:
            return
        self.last_log_time = current_time
        # log(f"UI info: {self.info}")


    def update_info(self):
        self.info['robot_pos'] = self.simulator.get_robot_pos()

    def render(self):
        return
        trav_map = self.simulator.get_trav_map()
        robot_map_point = trav_map.world_to_map((self.info['robot_pos'][0], self.info['robot_pos'][1]))
        self.trav_map_img = trav_map.floor_map[0].numpy()
        # 将单通道灰度图转换为RGB格式
        img = self.trav_map_img.T  # 转置x,y坐标以适应pygame显示
        # 翻转y坐标
        img = img[:,::-1]
        ui_x, ui_y = int(robot_map_point[1]), int(robot_map_point[0])
        # log(f"robot_map_point: {robot_map_point}")
        size = 4  # 智能体大小

        # 创建surface并画圆
        img_surface = pygame.surfarray.make_surface(img)
        # y坐标需要翻转,使用img.shape[0]-y来获取翻转后的y坐标
        pygame.draw.circle(img_surface, (255,0,0), (ui_x,img.shape[0]-ui_y), size)  # 直接在img_surface上画圆
        # pygame.draw.circle(img_surface, (255,0,0), (img.shape[0]-y, x), size)  # 直接在img_surface上画圆

        img_size = img_surface.get_size()[0]
        img_text_area_size = img_size * self.text_area_scale

        bg = pygame.Surface((img_text_area_size + img_size, img_size))
        bg.convert()
        bg.fill((255, 255, 255))

        bg.blit(img_surface, (img_text_area_size, 0))

        bg = pygame.transform.smoothscale(bg, (self.window_width, self.window_height))

        # create text area
        font_size = 22
        font = pygame.freetype.SysFont(pygame.font.get_default_font(), font_size)

        def draw_text_lines(text_lines):
            num_lines = len(text_lines)
            line_height = self.window_height // (num_lines + 1)
            y_offset = (self.window_height - line_height * num_lines) // 2

            for i, line in enumerate(text_lines):
                text_rect = font.get_rect(line, size=font_size)
                text_rect.centerx = self.text_area_width // 2
                text_rect.y = y_offset + i * line_height
                font.render_to(bg, text_rect, line, size=font_size, fgcolor=(0, 0, 0))


        robot_pos = self.info['robot_pos']
        robot_pos = f"{robot_pos[0]:.2f}, {robot_pos[1]:.2f}, {robot_pos[2]:.2f}"
        # 文本内容
        text_lines = [
            f"Scene Name: {self.info['scene_name']}",
            f"Robot Pos:",
            f"({robot_pos})"
        ]

        # 绘制文本
        draw_text_lines(text_lines)

        self.screen.blit(bg, (0, 0))
        pygame.event.pump()
        self.clock.tick(60)
        pygame.display.flip()
