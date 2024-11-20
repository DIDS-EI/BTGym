import time
from btgym.utils.logger import log
import pygame

class UI:
    def __init__(self, ctrl_sender_comm, ctrl_reciver_comm):
        self.ctrl_sender_comm = ctrl_sender_comm
        self.ctrl_reciver_comm = ctrl_reciver_comm
        self.ctrl_reciver_comm.set_owner(self)
        self.text_area_scale = 0.4375

        self.window_width = 800
        self.window_height = 800
        self.text_area_width = self.window_width * self.text_area_scale
        self.window_width = self.window_width + self.text_area_width

        self.init_pygame()
        self.img = None
        log('UI init finished')
        self.info = {'scene_name': 'default', 'robot_pos': [0,0,0]}
        self.run()

    def run(self):
        while True:
            self.step()
            time.sleep(0.01)

    def init_pygame(self):
        pygame.init()
        pygame.display.init()


        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("OmniBT UI")
        self.clock = pygame.time.Clock()
        self.screen.fill((255, 255, 255))  # 设置白色背景
        pygame.display.flip()  # 更新显示

    def step(self):
        self.ctrl_reciver_comm.deal_functions()

        self.render(self.img)

    def render(self,img):
        if img is None:
            return
        # draw img
        img = img[:, :, ::-1]  # BGR转RGB
        img = img.transpose((1, 0, 2))  # 转置以适应pygame格式
        img_surface = pygame.surfarray.make_surface(img)

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
            f"Robot Position:\n({robot_pos})"
        ]

        # 绘制文本
        draw_text_lines(text_lines)

        self.screen.blit(bg, (0, 0))
        pygame.event.pump()
        self.clock.tick(60)
        pygame.display.flip()

    def set_image(self, image):
        log("UI set_image")
        self.img = image

    def update_info(self, info):
        self.info.update(info)
