from distutils.command.config import config
import json
import os
import random

from torch.utils.data import Dataset
import torch
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption
import os
from torchvision.transforms.functional import hflip, resize

import math
import random
from random import random as rand

class DGM4_Dataset(Dataset):
    def __init__(self, config, ann_file, transform, max_words=30, is_train=True):

        # Image root that ann['image'] paths are relative to. Configurable so the
        # same metadata can be used wherever DGM4 lives; defaults to the layout
        # described in the README ('../../datasets').
        self.root_dir = config.get('image_root', '../../datasets')
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        if 'dataset_division' in config:
            self.ann = self.ann[:int(len(self.ann)/config['dataset_division'])]

        self.transform = transform
        self.max_words = max_words
        self.image_res = config['image_res']

        # Lowercasing should match the text-encoder casing assumption.
        # ``bert-base-uncased`` expects lowercase; ``microsoft/deberta-v3-base``
        # is case-sensitive and benefits from preserving entity casing.
        # Default tracks the text_backbone if 'text_lowercase' is not given.
        if 'text_lowercase' in config:
            self.lowercase = bool(config['text_lowercase'])
        else:
            self.lowercase = config.get('text_backbone', 'deberta').lower() == 'bert'

        self.is_train = is_train

        # ---- VLM distillation cache (training split only) ----
        # The VLM provides auxiliary soft targets during training only; the
        # evaluation/test path never consumes them, so we attach the 8th
        # return value strictly to the training dataset. When disabled the
        # dataset returns the original 7-tuple, byte-for-byte unchanged.
        self.vlm_enabled = bool(config.get('vlm_distill', False)) and is_train
        self.vlm_cache = None
        if self.vlm_enabled:
            from dataset.vlm_cache import VLMCache
            self.vlm_cache = VLMCache(
                config.get('vlm_cache_file'), max_words=self.max_words, verbose=True
            )

    def __len__(self):
        return len(self.ann)

    def get_bbox(self, bbox):
        xmin, ymin, xmax, ymax = bbox
        w = xmax - xmin
        h = ymax - ymin
        return int(xmin), int(ymin), int(w), int(h)    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        img_dir = ann['image']    
        image_dir_all = f'{self.root_dir}/{img_dir}'

        try:
            image = Image.open(image_dir_all).convert('RGB')   
        except Warning:
            raise ValueError("### Warning: fakenews_dataset Image.open")   
                         
        W, H = image.size
        has_bbox = False
        try:
            x, y, w, h = self.get_bbox(ann['fake_image_box'])
            has_bbox = True
        except:
            fake_image_box = torch.tensor([0, 0, 0, 0], dtype=torch.float)

        do_hflip = False
        if self.is_train:
            if rand() < 0.5:
                # flipped applied
                image = hflip(image)
                do_hflip = True

            image = resize(image, [self.image_res, self.image_res], interpolation=Image.BICUBIC)
        image = self.transform(image)
            
        if has_bbox:
            # flipped applied
            if do_hflip:  
                x = (W - x) - w  # W is w0

            # resize applied
            x = self.image_res / W * x
            w = self.image_res / W * w
            y = self.image_res / H * y
            h = self.image_res / H * h

            center_x = x + 1 / 2 * w
            center_y = y + 1 / 2 * h

            fake_image_box = torch.tensor([center_x / self.image_res, 
                        center_y / self.image_res,
                        w / self.image_res, 
                        h / self.image_res],
                        dtype=torch.float)

        label = ann['fake_cls']
        caption = pre_caption(ann['text'], self.max_words, lowercase=self.lowercase)
        fake_text_pos = ann['fake_text_pos']

        fake_text_pos_list = torch.zeros(self.max_words)

        for i in fake_text_pos:
            if i<self.max_words:
                fake_text_pos_list[i]=1

        if self.vlm_enabled:
            # caption is the exact pre_caption() string the text encoder sees,
            # so its sha1 matches the offline cache key.
            vlm_target = self.vlm_cache.build_target(img_dir, caption)
            return image, label, caption, fake_image_box, fake_text_pos_list, W, H, vlm_target

        return image, label, caption, fake_image_box, fake_text_pos_list, W, H
