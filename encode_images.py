import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
import random
import time
from moviepy.editor import VideoFileClip

import os
import sys
sys.path.append('/home/lgeng/iih')

from iih.models.timm_encoders.timm_ssl import TimmSSL


VIDEO_FILENAME = "compressed_video.mp4"
TENSOR_FILENAME = "images_embedding.pt"

def get_encoder():
    encoder = TimmSSL()
    encoder.load_state_dict(torch.load("hpr_model.pt")['model'])
    encoder.eval()
    for param in encoder.parameters():
            param.requires_grad = False
    return encoder

def extract_all_frames(video_path):
    frames = []
    with VideoFileClip(video_path) as video:
        for frame in video.iter_frames():
            frames.append(frame)
    frames_array = np.array(frames)
    return torch.from_numpy(frames_array).float().permute(0, 3, 1, 2)

def dir_encode_images(encoder, base_path):
    subdirs = sorted(os.listdir(base_path))
    subdirs = [d for d in subdirs if os.path.isdir(os.path.join(base_path, d))]
    for i, subdir in enumerate(subdirs):
        subdir_path = os.path.join(base_path, subdir)
        video_path = os.path.join(subdir_path, VIDEO_FILENAME)
        tensor_path = os.path.join(subdir_path, TENSOR_FILENAME)

        frames = extract_all_frames(video_path)
        encoded_frames = encoder(frames)
        torch.save(encoded_frames, tensor_path)
        print(f"{i+1}/{len(subdirs)}: {subdir} encoded")

if __name__ == "__main__":
    encoder = get_encoder()
    path_1 = "/home/lgeng/hrdc/pour_extracted/Pick_and_Place/cool/Env1"
    path_2 = '/home/lgeng/hrdc/pour_failure_extracted/Door_Opening/my_home/Env1'
    path_3 = '/home/lgeng/hrdc/new_pour_failure_extracted/Door_Opening/my_home/Env1'
    # dir_encode_images(encoder, path_1)
    # dir_encode_images(encoder, path_2)
    # dir_encode_images(encoder, path_3)