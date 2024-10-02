import clip
import torch
import torch.nn as nn
import faiss
import cv2
from PIL import Image
from tqdm import tqdm
import warnings
import pandas as pd
import numpy as np
import requests
from io import BytesIO

class GIFFrameRetriever():
    def get_gif_frames(self, video_url):
        cap = cv2.VideoCapture(video_url)
        framerate = int(cap.get(cv2.CAP_PROP_FPS))
        framerate=max(1,framerate)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration=frame_count/framerate
        frames = []
        count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            reshaped_frame=cv2.resize(frame,(512,512))
            rgb_frame = cv2.cvtColor(reshaped_frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)

        cap.release()
        frames = frames[::2]
        return frames

    def create_faiss_index(self, frames, model, preprocess, device):
        index = faiss.IndexFlatL2(model.visual.output_dim)  
        images = torch.stack([preprocess(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device) for frame in frames])
        images = images.reshape(-1, *images.shape[2:])
        with torch.no_grad():
            image_features = model.encode_image(images[:len(images)]).to(dtype=torch.float32) 
            index.add(image_features.cpu().numpy())
        return index

    def search_similar_frames(self, text, model, preprocess, index, device, top_k=3):
        text_input = clip.tokenize([text]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_input).to(dtype=torch.float32)
            distances, indices = index.search(text_features.cpu().numpy(), top_k)
        return indices

    def retrieve_images_from_gif(self, video_path, model, preprocess, text, top_k=10):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        frames=self.get_gif_frames(video_path)

        faiss_index = self.create_faiss_index(frames, model, preprocess, device)
        similar_frame_indices = self.search_similar_frames(text, model, preprocess, faiss_index, device, top_k)

        matched_frames = [frames[i] for i in similar_frame_indices[0]]
        return matched_frames



