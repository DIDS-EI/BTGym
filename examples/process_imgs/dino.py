import os
import numpy as np
import torch
import cv2
from torch.nn.functional import interpolate
from kmeans_pytorch import kmeans
# from utils import filter_points_by_bounds
from sklearn.cluster import MeanShift

folder_path = os.path.dirname(os.path.abspath(__file__))

patch_size = 14


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').eval().to(device)


rgb = cv2.imread(os.path.join(folder_path, "camera_0_rgb.png"))
H, W, _ = rgb.shape
patch_h = int(H // patch_size)
patch_w = int(W // patch_size)
new_H = patch_h * patch_size
new_W = patch_w * patch_size
transformed_rgb = cv2.resize(rgb, (new_W, new_H))
transformed_rgb = transformed_rgb.astype(np.float32) / 255.0  # float32 [H, W, 3]
# shape info
shape_info = {
    'img_h': H,
    'img_w': W,
    'patch_h': patch_h,
    'patch_w': patch_w,
}


img_tensors = torch.from_numpy(transformed_rgb).permute(2, 0, 1).unsqueeze(0).to(device)  # float32 [1, 3, H, W]
assert img_tensors.shape[1] == 3, "unexpected image shape"
features_dict = dinov2.forward_features(img_tensors)

print(features_dict)