import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--idx', type=int, default=0)
parser.add_argument('--total', type=int, default=1)
#parser.add_argument('--model_path', type=str, default='/data/user/zzq/train/mPLUG-Owl/mPLUG-Owl3/iic/mPLUG-Owl3-7B-240728')
parser.add_argument('--model_path', type=str, default='/data/user/zzq/train/checkpoints/mplug-owl3-7b-chat/v4-20241012-171402/checkpoint-2375-merged')
parser.add_argument('--dataset', type=str, default='kadid', choices=['spaq', 'koniq', 'livec', 'agiqa', 'kadid'])
parser.add_argument('--save_root', type=str, default='/data/user/zzq/train/results/koniq2kadid.json')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import json
import tqdm
import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr, pearsonr
import scipy.io as scio
import traceback
import math
from transformers import TextStreamer
from transformers import AutoTokenizer, AutoProcessor

from modeling_mplugowl3 import mPLUGOwl3Model
from configuration_mplugowl3 import mPLUGOwl3Config
from words_systems_queries import get_system_and_query


def get_score_gt(file_path):
    result = {}
    
    if 'spaq' in file_path:
        # print('spaq')
        with open(file_path, 'r') as f:
            data = f.readlines()[1:]
        name = [line.split(',')[0] for line in data]
        score_gt = [float(line.split(',')[1]) for line in data]
        result['name'] = name
        result['score'] = score_gt
    elif 'livec' in file_path:
        # print('livec')
        name_path = os.path.join(file_path, 'AllImages_release.mat')
        score_path = os.path.join(file_path, 'AllMOS_release.mat')
        
        name=scio.loadmat(name_path)['AllImages_release']
        score=scio.loadmat(score_path)['AllMOS_release'][0]
        
        name = [str(n[0][0]) for n in name]
        
        result['name'] = name
        result['score'] = score
    elif 'koniq' in file_path:
        # print('koniq')
        with open(file_path, 'r') as f:
            data = f.readlines()[1:]
        name = [line.split(',')[0].replace('\"', '') for line in data]
        score_gt = [float(line.split(',')[7]) for line in data]
        result['name'] = name
        result['score'] = score_gt
        
    elif 'agiqa' in file_path:
        with open(file_path, 'r') as f:
            data = f.readlines()[1:]
        name = [line.split(',')[0] for line in data]
        score_gt = [float(line.split(',')[-4]) for line in data]
        result['name'] = name
        result['score'] = score_gt
    
    elif 'kadid' in file_path:
        with open(file_path, 'r') as f:
            data = f.readlines()[1:]
        name = [line.split(',')[0] for line in data]
        score_gt = [float(line.split(',')[2]) for line in data]
        result['name'] = name
        result['score'] = score_gt
    
    return result

def get_srcc_plcc(result_gt, result_llm, remove_zero=False):
    gt_dict = {}
    for name, score in zip(result_gt['name'], result_gt['score']):
        name:str
        name = name.split('.')[0]
        gt_dict[name]=score
        
    llm_dict = {}
    for name, score in zip(result_llm['name'], result_llm['score']):
        name = name.split('.')[0]
        llm_dict[name]=score
    
    score_gt = []
    score_llm = []
    score_llm_total = []
    
    zero_cnt = 0
    
    for name, gt_score in gt_dict.items():
        
        llm_score = llm_dict.get(name, 1)
        if llm_score == 0:
            zero_cnt += 1
            if remove_zero:
                continue
        score_gt.append(gt_score)
        score_llm.append(llm_score)

    srcc = spearmanr(score_gt, score_llm)[0]
    plcc = pearsonr(score_gt, score_llm)[0]
    
    return srcc, plcc

def main(args):
    gt_path = {
        'spaq':'/home/zzq/data/results/spaq_gt/MOS and Image attribute scores.csv',
        'livec':'/home/zzq/zzq/others/iqa-metric/gt/livec',
        'koniq':'/home/zzq/zzq/others/iqa-metric/gt/koniq/koniq10k_scores_and_distributions.csv',
        'agiqa':'/home/zzq/zzq/others/iqa-metric/gt/agiqa/data.csv',
        'kadid':'/home/zzq/zzq/others/iqa-metric/gt/kadid/dmos.csv',
        }[args.dataset]
    result_gt = get_score_gt(gt_path)
    
    image_root = {
        'spaq':'/data/dataset/IQA/spaq',
        'livec':'/data/dataset/IQA/livec',
        'koniq':'/data/dataset/IQA/koniq',
        'agiqa':'/data/dataset/IQA/agiqa',
        'kadid':'/data/dataset/IQA/kadid',
        }[args.dataset]
    save_root = args.save_root
    
    config = mPLUGOwl3Config.from_pretrained(args.model_path)
    model = mPLUGOwl3Model.from_pretrained(args.model_path, attn_implementation='sdpa', torch_dtype=torch.half)
    model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    processor = model.init_processor(tokenizer)
    
    image_names = os.listdir(image_root)
    image_names.sort()
    chunk_size = len(image_names) // args.total
    chunks = [image_names[i:i+chunk_size] for i in range(0, len(image_names), chunk_size)]
    used_chunk = chunks[args.idx]
    image_dirs = [os.path.join(image_root, image_name) for image_name in used_chunk]
    
    system, query = get_system_and_query('quality', standard='word',n_word=7)
    messages = [
        {"role": "user", "content": f""" {system}
        <|image|>
        {query}"""},
        {"role": "assistant", "content": ""}
            ]
    
    if args.idx == 0:
        pbar = tqdm.tqdm(image_dirs, total=chunk_size)
    else:
        pbar = image_dirs
    
    image_results=[]
    result_llm = {
        'name':[],
        'score':[]
    }
    
    for image_dir in pbar:
        image_result = {
            'name':os.path.split(image_dir)[1],
            'score':""
        }
        result_llm['name'].append(image_result['name'])
        image = [Image.open(image_dir).convert('RGB')]
        system, query = get_system_and_query('quality', standard='word',n_word=7)
        messages = [
            {"role": "user", "content": f""" {system}
            <|image|>
            {query}"""},
            {"role": "assistant", "content": ""}
            ]
        inputs = processor(messages, images=image, videos=None)
        inputs.to('cuda')
        inputs.update({
                'tokenizer': tokenizer,
                'max_new_tokens':100,
                'decode_text':True,
            })
        output = model.generate(**inputs)[0][0]
        try:
            score = int(output)
        except Exception as e:
            print(image_result['name'], e)
            score = 1
        image_result['score']=score
        result_llm['score'].append(score)
        image_results.append(image_result)
    with open(save_root, 'w') as f:
        json.dump(image_results, f)

    srcc, plcc = get_srcc_plcc(result_gt, result_llm, remove_zero=True)
    print('srcc:',srcc)
    print('plcc:',plcc)
    

if __name__ == '__main__':
    main(args)