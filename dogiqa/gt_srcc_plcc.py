from copy import copy
import json
import math
import os
import numpy as np
from scipy.stats import spearmanr, pearsonr
import scipy.io as scio
import traceback

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