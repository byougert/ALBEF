import json
import os
import random

from torch.utils.data import Dataset

from PIL import Image
from PIL import ImageFile

import utils

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from dataset.utils import pre_caption


class re_split_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):
        self.ann = []
        self.imgs = json.load(open(ann_file, 'r'))['images']
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}

        n = 0
        for img in self.imgs:
            if img['split'] == 'train':
                img_name = img['filename']
                for an in img['sentences']:
                    self.ann.append({'image': img_name,
                                     'caption': an['raw'],
                                     'image_id': an['imgid']})
                    if an['imgid'] not in self.img_ids.keys():
                        self.img_ids[an['imgid']] = n
                        n += 1

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        image_path = os.path.join(self.image_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        caption = pre_caption(ann['caption'], self.max_words)

        return image, caption, self.img_ids[ann['image_id']]


class re_split_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, split, max_words=30):
        self.ann_imgs = json.load(open(ann_file, 'r'))['images']
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words

        # mapping total file to split file
        self.img_order2id = {}
        self.img_id2order = {}
        self.text_order2id = {}
        self.text_id2order = {}
        self.ann = []
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}

        txt_order = 0
        img_order = 0
        for img in self.ann_imgs:
            if img['split'] == split:
                self.image.append(img['filename'])
                self.img2txt[img_order] = []
                self.img_id2order[img['imgid']] = img_order
                self.img_order2id[img_order] = img['imgid']
                caps = []
                for caption in img['sentences']:
                    caps.append(caption['raw'])
                    self.text.append(pre_caption(caption['raw'], self.max_words))
                    self.img2txt[img_order].append(txt_order)
                    self.txt2img[txt_order] = img_order
                    self.text_id2order[caption['sentid']] = txt_order
                    self.text_order2id[txt_order] = caption['sentid']
                    txt_order += 1
                self.ann.append({'image': img['filename'],
                                 'caption': caps})
                img_order += 1

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):

        image_path = os.path.join(self.image_root, self.ann[index]['image'])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, index


class re_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.img_ids = {}   
        
        n = 0
        for ann in self.ann:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        
        caption = pre_caption(ann['caption'], self.max_words) 

        return image, caption, self.img_ids[ann['image_id']]
    
    

class re_eval_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root, max_words=30):        
        self.ann = json.load(open(ann_file,'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words 
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.ann):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,self.max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        return len(self.image)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.ann[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = self.transform(image)  

        return image, index
      
        

class pretrain_dataset(Dataset):
    def __init__(self, ann_file, transform, max_words=30):        
        self.ann = []
        for f in ann_file:
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.max_words = max_words
        
        
    def __len__(self):
        return len(self.ann)
    

    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        if type(ann['caption']) == list:
            caption = pre_caption(random.choice(ann['caption']), self.max_words)
        else:
            caption = pre_caption(ann['caption'], self.max_words)
      
        image = Image.open(ann['image']).convert('RGB')   
        image = self.transform(image)
                
        return image, caption
            

    
