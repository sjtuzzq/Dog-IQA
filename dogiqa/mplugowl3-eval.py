import os
import time
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def str2bool(s:str):
    if s.lower() == 'true':
        return True
    else:
        return False

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--idx', type=int, default=0)
parser.add_argument('--total', type=int, default=1)
parser.add_argument('--gpu', type=str, default='0')
# parser.add_argument('--total', action='store_true')
parser.add_argument('--bbox', type=str2bool, default=True)
parser.add_argument('--quality_only', type=str2bool, default=True)
parser.add_argument('--standard', type=str, default='word', choices=['number', 'word', 'sentence', 'number_2'])
parser.add_argument('--suffix', type=str, default='doge')
#parser.add_argument('--json_src', type=str, default='/home/zzq/zzq/SAMs/segment-anything-2/code-lk/koniq_seg_data_v4')
parser.add_argument('--json_src', type=str, default='/data/user/zzq/SAMs/segment-anything-2/code-lk/doge')
parser.add_argument('--n_word', type=int, default=7)
#parser.add_argument('--image_root', type=str, default='/data/dataset/IQA/koniq')
parser.add_argument('--image_root', type=str, default='/data/user/zzq/SAMs/segment-anything-2/code-lk/doge')


args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


    

import json
import tqdm

import numpy as np
import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor
from decord import VideoReader, cpu
from modeling_mplugowl3 import mPLUGOwl3Model
from configuration_mplugowl3 import mPLUGOwl3Config
from words_systems_queries import get_system_and_query

from PIL import Image
import requests
import copy
import torch

import numpy as np
import pycocotools.mask as maskutil
import skimage.measure as skm

import cv2

def draw_red_ellipse(image, top, bottom, left, right):
    red_color = (0, 0, 255)

    # 计算椭圆的中心点、长轴和短轴的长度
    center = ((left + right) // 2, (top + bottom) // 2)  # 椭圆中心点
    axes = ((right - left) // 2, (bottom - top) // 2)    # (长轴半径, 短轴半径)

    # 绘制椭圆，angle=0 表示椭圆不旋转，startAngle 和 endAngle 分别为椭圆的起始角度和结束角度
    cv2.ellipse(image, center, axes, angle=0, startAngle=0, endAngle=360, color=red_color, thickness=2)
    
    return image


def draw_red_box(image, top, bottom, left, right):
    # 定义红色 (B, G, R) 颜色
    red_color = (0, 0, 255)

    # 绘制矩形框, (left, top) 是左上角，(right, bottom) 是右下角
    cv2.rectangle(image, (left, top), (right, bottom), red_color, 2)
    
    return image


def get_masked_image(image, coco_mask, bbox=True, circle=True):
    binary_mask = maskutil.decode(coco_mask)
    
    if circle:
        labeled_mask = skm.label(binary_mask)
        regions = skm.regionprops(labeled_mask)

        real_min_row = None
        real_min_col = None
        real_max_row = None
        real_max_col = None
        for region in regions:
            min_row, min_col, max_row, max_col = region.bbox
            if real_min_row == None:
                real_min_row = min_row
            else:
                real_min_row = min(real_min_row, min_row)
                
            if real_min_col == None:
                real_min_col = min_col
            else:
                real_min_col = min(real_min_col, min_col)
                
            if real_max_row == None:
                real_max_row = max_row
            else:
                real_max_row = max(real_max_row, max_row)
                
            if real_max_col == None:
                real_max_col = max_col
            else:
                real_max_col = max(real_max_col, max_col)
    
    if not bbox:
        masked_image = np.where(binary_mask[:, :, None], image, 0)
    else:
        masked_image = image

    # 确定bounding box
    labeled_mask = skm.label(binary_mask)
    regions = skm.regionprops(labeled_mask)

    real_min_row = None
    real_min_col = None
    real_max_row = None
    real_max_col = None
    for region in regions:
        min_row, min_col, max_row, max_col = region.bbox
        if real_min_row == None:
            real_min_row = min_row
        else:
            real_min_row = min(real_min_row, min_row)
            
        if real_min_col == None:
            real_min_col = min_col
        else:
            real_min_col = min(real_min_col, min_col)
            
        if real_max_row == None:
            real_max_row = max_row
        else:
            real_max_row = max(real_max_row, max_row)
            
        if real_max_col == None:
            real_max_col = max_col
        else:
            real_max_col = max(real_max_col, max_col)
            
    cropped_image = masked_image[real_min_row:real_max_row, real_min_col:real_max_col]
    
    return Image.fromarray(cropped_image)

def main():
    args = parser.parse_args()
    
    dry_run = False
    
    json_root = args.json_src
    image_root = args.image_root
    # json_save_root = '/data/user/zzq/results/mplug-owl3-result'
    if args.n_word == -1:
        num_word = ''
    else:
        num_word = f'-{args.n_word}'
        
    json_save_root = f'/data/user/zzq/results/bbox{args.bbox}-quality{args.quality_only}{num_word}-{args.standard}-{args.suffix}'
    
    os.makedirs(json_save_root, exist_ok=True)

    model_path = 'mPLUG/mPLUG-Owl3-7B-240728'
    print('start build model')
    config = mPLUGOwl3Config.from_pretrained(model_path)
    model = mPLUGOwl3Model.from_pretrained(model_path, attn_implementation='sdpa', torch_dtype=torch.half)
    model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = model.init_processor(tokenizer)
    print('build model')
    if not args.quality_only:
        assess_words = [
            'quality',
            'brightness',
            'contrast',
            'noise',
            'sharpness',
            'clearness',
        ]
    else:
        assess_words = [
            'quality',
        ]
    print(f'{args.quality_only=}')
    
    image_names = os.listdir(image_root)
    image_names.sort()
    
    chunk_size = len(image_names) // args.total
    chunks = [image_names[i:i+chunk_size] for i in range(0, len(image_names), chunk_size)]
    
    used_chunk = chunks[args.idx]
    print(f'chunk id:{args.idx}')

    json_names = [f'{image_name[:-4]}.json' for image_name in used_chunk]

    image_dirs = [os.path.join(image_root, image_name) for image_name in used_chunk]
    json_dirs = [os.path.join(json_root, json_name) for json_name in json_names]

    print('start eval.')
    if args.idx == 0:
        pbar = tqdm.tqdm(zip(image_dirs, json_dirs), total=chunk_size)
    else:
        pbar = zip(image_dirs, json_dirs)
    
    for image_dir, json_dir in pbar:
        if not os.path.exists(json_dir):
            continue
        
        json_name = os.path.split(json_dir)[1]
        json_result_save = os.path.join(json_save_root, json_name)
        
        if os.path.exists(json_result_save):
            continue
        
        
        if args.idx != 0:
            print(image_dir)
            
        with open(json_dir, 'r') as f:
            data = json.load(f)
            masks = [anno['segmentation'] for anno in data['annotations']]
        
        image = Image.open(image_dir).convert('RGB')
        image_array = np.array(image)
            
        image_result = {
            'image_name':os.path.split(image_dir)[1],
            'n_mask':len(masks),
            'mask_assess_result':[]
        }

        for mask in masks:
            mask_assess_result = {}
            cropped_image = get_masked_image(image_array, mask, bbox=args.bbox, circle=args.circle)
            
            cropped_image = [cropped_image]
    
            for assess_word in assess_words:
                system, query = get_system_and_query(assess_word, standard=args.standard, n_word=args.n_word)
                
                messages = [
                    {"role": "user", "content": f""" {system}
                        <|image|>
                        {query}"""},
                    {"role": "assistant", "content": ""}
                        ]

                inputs = processor(messages, images=cropped_image, videos=None)
            
                inputs.to('cuda')
                inputs.update({
                    'tokenizer': tokenizer,
                    'max_new_tokens':100,
                    'decode_text':True,
                    # 'logits_mode':False,
                })


                outputs = model.generate(**inputs)[0][0]
                mask_assess_result[assess_word] = outputs
                #print("outputs:",outputs)
            image_result['mask_assess_result'].append(mask_assess_result)
    
        
        with open(json_result_save, 'w') as f:
            json.dump(image_result, f)

if __name__ == '__main__':
    main()