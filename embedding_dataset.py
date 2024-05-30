import os
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import copy
import yaml
from PIL import Image
from pathlib import Path
import random
import time
from moviepy.editor import VideoFileClip

VIDEO_FILENAME = "compressed_video.mp4"
TENSOR_FILENAME = "images_embedding.pt"


class EmbeddingDataset(Dataset):
    def __init__(self, data_name, set_type, success_data_path, unsuccess_data_path, negatives_num=5):
        self.data_name = data_name
        self.success_data_path = success_data_path
        self.unsuccess_data_path = unsuccess_data_path
        self.set_type = set_type
        self.negatives_num = negatives_num

        self._get_success_embedding_paths()
        self._get_unsuccess_embedding_paths()
        self._get_success_video_paths()
        self._get_unsuccess_video_paths()
        self._get_max_length()

    def _get_max_length(self):
        self.max_length = 0
        for video_path in self.success_video_paths + self.unsuccess_video_paths:
            with VideoFileClip(video_path) as video:
                length = int(video.fps * video.duration)
                if length > self.max_length:
                    self.max_length = length

    def __getitem__(self, idx):
        if self.set_type == 'train':
            if idx < len(self.success_paths):
                query = self.success_paths[idx]
                positive_key = random.choice(self.success_paths)
                while positive_key == query:
                    positive_key = random.choice(self.success_paths)
                negative_keys = random.sample(self.unsuccess_paths, self.negatives_num)

            else:
                query = self.unsuccess_paths[idx - len(self.success_paths)]
                positive_key = random.choice(self.unsuccess_paths)
                while positive_key == query:
                    positive_key = random.choice(self.unsuccess_paths)
                negative_keys = random.sample(self.success_paths, self.negatives_num)
            
            # start_time = time.time()
            
            query_embeddings = torch.load(query)
            positive_embeddings = torch.load(positive_key)
            negative_embeddings = [torch.load(negative) for negative in negative_keys]

            query_embeddings = F.pad(query_embeddings, (0, 0, self.max_length - query_embeddings.size(0), 0))
            positive_embeddings = F.pad(positive_embeddings, (0, 0, self.max_length - positive_embeddings.size(0), 0))
            combined_embeddings = torch.stack([query_embeddings, positive_embeddings], dim=0) # [B, L, d_model]
            negative_embeddings = torch.stack([F.pad(negative, (0, 0, self.max_length - negative.size(0), 0)) for negative in negative_embeddings], dim=0)

            # end_time = time.time()
            # print("Time taken for getting one batch and embeddings:", end_time - start_time)
            return {
                'traj_1': combined_embeddings, # [2, L, embed_dim]
                'traj_2': negative_embeddings
            }
        
        elif self.set_type == 'valid':
            label = random.choice([1, -1])
            if idx < len(self.success_paths):
                traj1_path = self.success_paths[idx]

                if label == 1:
                    traj2_path = random.choice(self.success_paths)
                    while traj2_path == traj1_path:
                        traj2_path = random.choice(self.success_paths)
                else:
                    traj2_path = random.choice(self.unsuccess_paths)

            else:
                traj1_path = self.unsuccess_paths[idx - len(self.success_paths)]

                if label == 1:
                    traj2_path = random.choice(self.unsuccess_paths)
                    while traj2_path == traj1_path:
                        traj2_path = random.choice(self.unsuccess_paths)
                else:
                    traj2_path = random.choice(self.success_paths)

        
            traj1 = torch.load(traj1_path)
            traj2 = torch.load(traj2_path)
            traj1 = F.pad(traj1, (0, 0, self.max_length - traj1.size(0), 0)).unsqueeze(0)
            traj2 = F.pad(traj2, (0, 0, self.max_length - traj2.size(0), 0)).unsqueeze(0)
            
            return {'traj_1': traj1, 'traj_2': traj2, 'label': label}
    
    def __len__(self):
        return len(self.success_paths) + len(self.unsuccess_paths)

    def _get_success_embedding_paths(self):
        self.success_paths = []

        subdirs = sorted(os.listdir(self.success_data_path))
        subdirs = [d for d in subdirs if os.path.isdir(os.path.join(self.success_data_path, d))]
        for i, subdir in enumerate(subdirs):
            print("getting points for", subdir)
            subdir_path = os.path.join(self.success_data_path, subdir)
            embedding_path = os.path.join(subdir_path, TENSOR_FILENAME)
            self.success_paths.append(embedding_path)
    
    def _get_unsuccess_embedding_paths(self):
        self.unsuccess_paths = []

        subdirs = sorted(os.listdir(self.unsuccess_data_path))
        subdirs = [d for d in subdirs if os.path.isdir(os.path.join(self.unsuccess_data_path, d))]
        for i, subdir in enumerate(subdirs):
            print("getting points for", subdir)
            subdir_path = os.path.join(self.unsuccess_data_path, subdir)
            embedding_path = os.path.join(subdir_path, TENSOR_FILENAME)
            self.unsuccess_paths.append(embedding_path)

    def _get_success_video_paths(self):
        self.success_video_paths = []

        subdirs = sorted(os.listdir(self.success_data_path))
        subdirs = [d for d in subdirs if os.path.isdir(os.path.join(self.success_data_path, d))]
        for i, subdir in enumerate(subdirs):
            print("getting points for", subdir)
            subdir_path = os.path.join(self.success_data_path, subdir)
            video_path = os.path.join(subdir_path, VIDEO_FILENAME)
            self.success_video_paths.append(video_path)
    
    def _get_unsuccess_video_paths(self):
        self.unsuccess_video_paths = []

        subdirs = sorted(os.listdir(self.unsuccess_data_path))
        subdirs = [d for d in subdirs if os.path.isdir(os.path.join(self.unsuccess_data_path, d))]
        for i, subdir in enumerate(subdirs):
            print("getting points for", subdir)
            subdir_path = os.path.join(self.unsuccess_data_path, subdir)
            video_path = os.path.join(subdir_path, VIDEO_FILENAME)
            self.unsuccess_video_paths.append(video_path)