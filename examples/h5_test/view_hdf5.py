import pygame
import pygame_gui
import h5py
import numpy as np
from pathlib import Path
import cv2
import os

class HDF5Viewer:
    def __init__(self):
        pygame.init()
        self.window_size = (1280, 720)
        self.window = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption('HDF5 Data Viewer')
        
        # Load HDF5 file first to get total samples
        self.hdf5_path = Path(__file__).parent / 'robot_data.hdf5'
        self.load_hdf5()
        
        # Get total number of samples
        self.total_samples = self.h5_file.attrs['total_samples']
        
        # Create UIManager
        self.manager = pygame_gui.UIManager(self.window_size)
        self.clock = pygame.time.Clock()
        
        # Create UI elements
        self.setup_ui()
        
        # Create surfaces for images
        self.rgb_surface = None
        self.depth_surface = None
        self.mask_surface = None
    
    def setup_ui(self):
        # Create left list panel
        self.list_panel = pygame_gui.elements.UIPanel(
            relative_rect=pygame.Rect((0, 0), (250, 720)),
            manager=self.manager
        )
        
        # Create data list with actual number of samples
        self.data_list = pygame_gui.elements.UISelectionList(
            relative_rect=pygame.Rect((10, 10), (230, 700)),
            item_list=[f"Sample {i:05d}" for i in range(self.total_samples)],
            manager=self.manager,
            container=self.list_panel
        )
        
        # Create right display panel
        self.display_panel = pygame_gui.elements.UIPanel(
            relative_rect=pygame.Rect((260, 0), (1020, 720)),
            manager=self.manager
        )
        
        # Create text display area
        self.text_height = 30
        y_offset = 10
        
        # Add total samples counter
        self.total_samples_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_offset), (980, self.text_height)),
            text=f"Total Samples: {self.total_samples}",
            manager=self.manager,
            container=self.display_panel
        )
        
        y_offset += self.text_height + 10
        
        # Task description
        self.task_desc_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_offset), (980, self.text_height)),
            text="Task Description: ",
            manager=self.manager,
            container=self.display_panel
        )
        
        # Image display area
        self.image_size = (320, 240)
        y_offset += self.text_height + 10
        self.image_y_offset = y_offset
        y_offset += self.image_size[1]
        
        # Image labels
        self.rgb_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((10, y_offset), (100, self.text_height)),
            text="RGB Image",
            manager=self.manager,
            container=self.display_panel
        )
        
        self.depth_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((340, y_offset), (100, self.text_height)),
            text="Depth Image",
            manager=self.manager,
            container=self.display_panel
        )
        
        self.mask_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((670, y_offset), (100, self.text_height)),
            text="Mask Image",
            manager=self.manager,
            container=self.display_panel
        )
        
        # Robot state
        y_offset += 30
        self.robot_state_label = pygame_gui.elements.UITextBox(
            relative_rect=pygame.Rect((10, y_offset), (980, 100)),
            html_text="Robot State: ",
            manager=self.manager,
            container=self.display_panel
        )

    def load_hdf5(self):
        self.h5_file = h5py.File(self.hdf5_path, 'r')
    
    def process_image(self, img_data, is_rgb=False):
        # Resize image
        img = cv2.resize(img_data, self.image_size)
        
        if is_rgb:
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # Convert to 3 channel image
            img = np.stack([img] * 3, axis=-1)
        
        # Create pygame surface
        surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
        return surface
        
    def update_display(self, sample_idx):
        sample_group = self.h5_file['data'][f'sample_{sample_idx:05d}']
        
        # Update task description
        task_desc = sample_group['task_description'][()].decode('utf-8')
        self.task_desc_label.set_text(f"Task Description: {task_desc}")
        
        # Update images
        # RGB image
        rgb_img = sample_group['rgb'][()]
        self.rgb_surface = self.process_image(rgb_img, is_rgb=True)
        
        # Depth image
        depth_img = sample_group['depth'][()]
        depth_img = (depth_img * 255).astype(np.uint8)
        self.depth_surface = self.process_image(depth_img)
        
        # Mask image
        mask_img = sample_group['object_mask'][()]
        mask_img = (mask_img * 25).astype(np.uint8)
        self.mask_surface = self.process_image(mask_img)
        
        # Update robot state
        joints = sample_group['robot_joints'][()]
        pose = sample_group['gripper_pose'][()]
        state_text = (
            f"Joint Angles: {', '.join([f'{j:.3f}' for j in joints])}<br>"
            f"End Effector Pose: Position({pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f}), "
            f"Quaternion({pose[3]:.3f}, {pose[4]:.3f}, {pose[5]:.3f}, {pose[6]:.3f})"
        )
        self.robot_state_label.html_text = state_text
        self.robot_state_label.rebuild()
    
    def draw_images(self):
        if self.rgb_surface:
            self.window.blit(self.rgb_surface, (270, self.image_y_offset))
        if self.depth_surface:
            self.window.blit(self.depth_surface, (600, self.image_y_offset))
        if self.mask_surface:
            self.window.blit(self.mask_surface, (930, self.image_y_offset))
        
    def run(self):
        running = True
        while running:
            time_delta = self.clock.tick(60)/1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame_gui.UI_SELECTION_LIST_NEW_SELECTION:
                    if event.ui_element == self.data_list:
                        selected_idx = int(event.text.split()[1])
                        self.update_display(selected_idx)

                self.manager.process_events(event)

            self.manager.update(time_delta)
            
            self.window.fill((255, 255, 255))
            self.manager.draw_ui(self.window)
            self.draw_images()
            pygame.display.update()

        pygame.quit()
        self.h5_file.close()

if __name__ == '__main__':
    viewer = HDF5Viewer()
    viewer.run()