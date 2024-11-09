from copy import copy
import json
import math
import os
import numpy as np
from scipy.stats import spearmanr, pearsonr
import scipy.io as scio
import traceback

total_dirs = [
    '/home/zzq/data/results/internlm-1.0-result',
    '/home/zzq/data/results/internlm-2.0-result',
    '/home/zzq/data/results/llava-7b-result',
    '/home/zzq/data/results/llava-13b-result',
    '/home/zzq/data/results/llava-next-result',
    '/home/zzq/data/results/mplug-owl-result',
    '/home/zzq/data/results/mplug-owl2-result',
    '/home/zzq/data/results/mplug-owl3-result',
]

mplug_dirs = [
    '/home/zzq/data/results/mplug-owl3-result-bboxTrue-qualityonlyTrue-sentence',
    '/home/zzq/data/results/mplug-owl3-result-total-qualityonlyTrue-number',
    '/home/zzq/data/results/mplug-owl3-result-total-qualityonlyTrue-word',
    '/home/zzq/data/results/mplug-owl3-result-total-qualityonlyTrue-sentence',
    '/home/zzq/data/results/mplug-owl3-result-bboxTrue-qualityonlyTrue-word',
    '/home/zzq/zzq/results/mplug-owl3-result-bboxTrue-qualityonlyTrue-word-22',
    '/home/zzq/zzq/results/mplug-owl3-result-bboxTrue-qualityonlyTrue-word-215',
]



def get_model_name(dir_path):
    # return '-'.join(os.path.split(dir_path)[1].split('-')[:-1])
    return os.path.split(dir_path)[1]

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

    
def get_score_llm_total(dir_path):
    json_names = os.listdir(dir_path)
    json_names.sort()
    scores = [1 for _ in range(len(json_names))]
    for idx, json_name in enumerate(json_names):
        json_path = os.path.join(dir_path, json_name)
        with open(json_path, 'r') as f:
            data = json.load(f)
            try:
                score = float(data['assess_result']['quality'])
            except Exception as e:
                score:str = data['assess_result']['quality']
                if score.endswith('</s>'):
                    score = float(score[0])
                else:
                    print(json_name, e)
                    score = 1
            # score = float(['assess_result']['quality'])
            scores[idx] =score
    return {
        'name':json_names,
        'score':scores
    }


def get_score_llm_mask(dir_path, reduction='mean', area_json_dir = '', lambda1=0.0, lambda2=0.0, add_total=False, switch_score=True, switch_pair=(1,2)):
    result = {}
    log_dir = r'stat_res'
    
    model_name = get_model_name(dir_path)
    error_log_dir = os.path.join(log_dir, model_name+'.txt')
    f_error = open(error_log_dir, 'w') 
    
    json_names = os.listdir(dir_path)
    json_names.sort()
    result['name'] = json_names
    
    if switch_score:
        print(f'{switch_pair=}')
        def switch_func(score, switch_pair=switch_pair):
            if score == switch_pair[0]:
                return switch_pair[1]
            elif score == switch_pair[1]:
                return switch_pair[0]
            else:
                return score
    
    scores = [0 for _ in range(len(json_names))]

    mask_scores = [[] for _ in range(len(json_names))]
    mask_areas = [[] for _ in range(len(json_names))]
    
    for idx, json_name in enumerate(json_names):
        try:
            json_path = os.path.join(dir_path, json_name)
            with open(json_path, 'r') as f:
                data = json.load(f)
                # idx = int(data['image_name'][:-4]) - 1
                scores_mask = []
                
                for res in data['mask_assess_result']:
                    res:dict
                    try:
                        score = float(res['quality'])
                    except Exception as e:
                        print('level 2',json_name, e, file=f_error)
                        score:str = res['quality']
                        if score.endswith('</s>'):
                            score = float(score[0])
                        else:
                            score = 1

                    scores_mask.append(score)
                
                if switch_score:
                    scores_mask = [switch_func(s) for s in scores_mask]
                    
            if reduction == 'mean':
                scores[idx] = np.mean(scores_mask)
                mask_scores[idx] = scores_mask
                
            elif reduction == 'area':
                area_json_path = os.path.join(area_json_dir, json_name)
                if add_total:
                    json_path_total = os.path.join('/home/zzq/zzq/results/mplug-owl3-result-total', json_name)
                    with open(json_path_total, 'r') as f:
                        data = json.load(f)
                        total_s = float(data['assess_result']['quality'])

                areas = []
                with open(area_json_path, 'r') as f:
                    data = json.load(f)
                    for seg in data['annotations']:
                        seg:dict
                        
                        if seg.get('area', False):
                            areas.append(seg['area'])
                    if add_total:
                        total_area = data['image']['width'] * data['image']['height']

                            
                weights = np.array(areas) 
                # weights2 = np.array(ss)          
                # weights = weights1 * weights2
                scores[idx] = np.average(scores_mask, weights=weights) + \
                    lambda1 * len(areas)
                
                # scores[idx] = lambda1 * len(areas)
                if add_total:
                    scores[idx] += lambda2 * total_s
                    
                mask_scores[idx] = list(np.array(scores_mask) * np.array(areas) / np.sum(areas) * len(scores_mask))
                mask_areas[idx] = copy(areas)
                
        except Exception as e:
            print('level 1',json_name, e, file=f_error)
            traceback.print_exc(file=f_error)
    f_error.close()

    result['mask_score'] = mask_scores
    result['score'] = scores
    result['area'] = mask_areas

    return result
    
    
def get_score_llm(dir_path, mode, area_json_dir='', lambda1=0, lambda2=0, use_total=True, switch_score=False, switch_pair=(1,2)):
    if mode == 'total':
        result = get_score_llm_total(dir_path)
    elif mode == 'mask-mean':
        result = get_score_llm_mask(dir_path, reduction='mean')
    elif mode == 'mask-area':
        result = get_score_llm_mask(dir_path, reduction='area', area_json_dir=area_json_dir, lambda1=lambda1, lambda2=lambda2, switch_score=switch_score, switch_pair=switch_pair)

    return result

def get_srcc_plcc(result_gt, result_llm, result_llm_total=None, remove_zero=False, lambda3=0.0):
    gt_dict = {}
    for name, score in zip(result_gt['name'], result_gt['score']):
        name:str
        name = name.split('.')[0]
        gt_dict[name]=score
        
    llm_dict = {}
    for name, score in zip(result_llm['name'], result_llm['score']):
        name = name.split('.')[0]
        llm_dict[name]=score
    
    if result_llm_total != None:
        llm_dict_total = {}
        for name, score in zip(result_llm_total['name'], result_llm_total['score']):
            name = name.split('.')[0]
            llm_dict_total[name]=score
    
    score_gt = []
    score_llm = []
    score_llm_total = []
    
    zero_cnt = 0
    # print(len(result_llm['score']))
    # print(len(result_gt['score']))
    
    for name, gt_score in gt_dict.items():
        
        llm_score = llm_dict.get(name, 1)
        if llm_score == 0:
            zero_cnt += 1
            if remove_zero:
                continue
        score_gt.append(gt_score)
        score_llm.append(llm_score)
        if result_llm_total != None:
            llm_score_total = llm_dict_total.get(name, 1)
            score_llm_total.append(llm_score_total)
            
    if result_llm_total != None:
        score_llm = np.array(score_llm)
        score_llm_total = np.array(score_llm_total)
        
        score_llm = lambda3 * score_llm + score_llm_total 
    # print(len(score_gt))
    # print(len(score_llm))
    
    # score_llm = np.array(score_llm)
    # score_llm = np.round ((score_llm - score_llm.min()) / (score_llm.max() - score_llm.min()) * 10)
    
    srcc = spearmanr(score_gt, score_llm)[0]
    plcc = pearsonr(score_gt, score_llm)[0]
    
    
    return srcc, plcc

def main(args):
    spaq_dirs = [
        '/data/user/zzq/results/bboxTrue-qualityTrue-7-word-spaq-circle-1016',
        '/home/zzq/zzq/results/bboxTrue-qualityTrue-3-word-v4',
        '/home/zzq/zzq/results/bboxTrue-qualityTrue-5-word-v4',
        '/home/zzq/zzq/results/bboxTrue-qualityTrue-7-word-v4',#word
        '/home/zzq/zzq/results/bboxTrue-qualityTrue-9-word-v4',
        '/home/zzq/zzq/results/bboxTrue-qualityTrue-7-word-spaq-v4-pv2',
        '/home/zzq/zzq/results/bboxFalse-qualityTrue-7-word-spaq-v4',#mask
        '/home/zzq/zzq/results/bboxTrue-qualityTrue-7-sentence_2-spaq-v4',#sentence
        '/home/zzq/zzq/results/bboxTrue-qualityTrue-7-number-spaq-v4',#number
        '/home/zzq/zzq/results/mplug-owl3-result-total-qualityonlyFalse-word-9-spaq',
        '/home/zzq/zzq/results/mplug-owl3-result-total-qualityonlyFalse-word-5-spaq',
        '/home/zzq/zzq/results/mplug-owl3-result-total-qualityonlyFalse-word-7-spaq',#total
    ]

    livec_dirs = [
        '/home/zzq/zzq/results/bboxTrue-qualityTrue-3-word-livec-v4',
        '/home/zzq/zzq/results/bboxTrue-qualityTrue-5-word-livec-v4',
        '/home/zzq/zzq/results/bboxTrue-qualityTrue-7-word-livec-v4',
        '/home/zzq/zzq/results/bboxTrue-qualityTrue-9-word-livec-v4',
        '/home/zzq/zzq/results/bboxTrue-qualityTrue-7-word-livec-v4-pv2',
        '/data/user/zzq/results/mplugowl3-spaq2livec',
        '/home/zzq/zzq/results/mplug-owl3-result-total-qualityonlyFalse-word-7-livec',
    ]

    koniq_dirs = [
        #'/home/zzq/zzq/results/bboxTrue-qualityTrue-3-word-koniq-v4',
        '/home/zzq/zzq/results/bboxTrue-qualityTrue-5-word-koniq-v4',
        '/home/zzq/zzq/results/bboxTrue-qualityTrue-7-word-koniq-v4',
        '/home/zzq/zzq/results/bboxTrue-qualityTrue-9-word-koniq-v4',
        '/home/zzq/zzq/results/bboxTrue-qualityTrue-7-word-koniq-v4-pv2',
        '/data/user/zzq/results/mplugowl3-spaq2koniq',
        '/home/zzq/zzq/results/mplug-owl3-result-total-qualityonlyFalse-word-7-koniq',
    ]
    
    agiqa_dirs = [
        '/data/user/zzq/results/bboxTrue-qualityTrue-7-word-agiqa-circle-1016',
        '/home/zzq/zzq/results/bboxTrue-qualityTrue-3-word-agiqa-v4',
        '/home/zzq/zzq/results/bboxTrue-qualityTrue-5-word-agiqa-v4',
        '/home/zzq/zzq/results/bboxTrue-qualityTrue-7-word-agiqa-v4',
        '/home/zzq/zzq/results/bboxTrue-qualityTrue-9-word-agiqa-v4',
        '/home/zzq/zzq/results/bboxTrue-qualityTrue-7-word-agiqa-v4-pv2',
        '/data/user/zzq/results/bboxFalse-qualityTrue-7-word-agiqa-v4',
        '/data/user/zzq/results/bboxTrue-qualityTrue-7-sentence_2-agiqa-v4',
        '/data/user/zzq/results/bboxTrue-qualityTrue-7-number-agiqa-v4',
        '/home/zzq/zzq/results/mplug-owl3-result-total-qualityonlyFalse-word-5-agiqa',
        '/data/user/zzq/results/mplugowl3-spaq2agiqa',
        '/home/zzq/zzq/results/mplug-owl3-result-total-qualityonlyFalse-word-7-agiqa',
    ]
    kadid_dirs = [
        '/home/zzq/zzq/results/bboxTrue-qualityTrue-3-word-kadid-v4',
        '/home/zzq/zzq/results/bboxTrue-qualityTrue-5-word-kadid-v4',
        '/home/zzq/zzq/results/bboxTrue-qualityTrue-7-word-kadid-v4',
        '/home/zzq/zzq/results/bboxTrue-qualityTrue-9-word-kadid-v4',
        '/data/user/zzq/results/mplugowl3-spaq2kadid',
        '/home/zzq/zzq/results/bboxTrue-qualityTrue-7-word-kadid-v4-pv2',
        '/home/zzq/zzq/results/mplug-owl3-result-total-qualityonlyFalse-word-7-kadid',
    ]
    
    assess_result_dirs = {
        'spaq':spaq_dirs,
        'livec':livec_dirs,
        'koniq':koniq_dirs,
        'agiqa':agiqa_dirs,
        'kadid':kadid_dirs,
    }
    
    llm_dir = assess_result_dirs[args.gt_path][args.path_idx]
    model_name = get_model_name(llm_dir)
    

    gt_path = {
        'spaq':'/home/zzq/data/results/spaq_gt/MOS and Image attribute scores.csv',
        'livec':'/home/zzq/zzq/others/iqa-metric/gt/livec',
        'koniq':'/home/zzq/zzq/others/iqa-metric/gt/koniq/koniq10k_scores_and_distributions.csv',
        'agiqa':'/home/zzq/zzq/others/iqa-metric/gt/agiqa/data.csv',
        'kadid':'/home/zzq/zzq/others/iqa-metric/gt/kadid/dmos.csv',
    }[args.gt_path]

    area_json_dir = {
        'spaq':r'/home/zzq/zzq/SAMs/segment-anything-2/code-lk/spaq_seg_data_v4',
        'livec':r'/home/zzq/zzq/SAMs/segment-anything-2/code-lk/livec_seg_data_v4',
        'koniq':r'/home/zzq/zzq/SAMs/segment-anything-2/code-lk/koniq_seg_data_v4',
        'agiqa':r'/home/zzq/zzq/SAMs/segment-anything-2/code-lk/agiqa_seg_data_v4',
        'kadid':r'/home/zzq/zzq/SAMs/segment-anything-2/code-lk/kadid_seg_data_v4',
    }[args.gt_path]

    result_gt = get_score_gt(gt_path)
    result_llm = get_score_llm(
            llm_dir,
            mode=args.mode, 
            area_json_dir=area_json_dir,
            lambda1=args.lambda1, 
            lambda2=args.lambda2,
            switch_score=args.switch_score, 
            switch_pair=args.switch_pair)
    if args.add_total:
        total_path = {
        'spaq':'/home/zzq/zzq/results/mplug-owl3-result-total-qualityonlyFalse-word-7-spaq',
        'livec':'/home/zzq/zzq/results/mplug-owl3-result-total-qualityonlyFalse-word-7-livec',
        'koniq':'/home/zzq/zzq/results/mplug-owl3-result-total-qualityonlyFalse-word-7-koniq',
        'agiqa':'/home/zzq/zzq/results/mplug-owl3-result-total-qualityonlyFalse-word-7-agiqa',
        'kadid':'/home/zzq/zzq/results/mplug-owl3-result-total-qualityonlyFalse-word-7-kadid',
        }[args.gt_path]
        result_llm_total = get_score_llm(
            total_path,
            mode='total'
            )
    else:
        result_llm_total = None
        

    print(len(result_gt['name']))
    print(len(result_llm['score']))
    #print(len(result_llm_total['score']))
    
    srcc, plcc = get_srcc_plcc(result_gt, result_llm, result_llm_total, remove_zero=args.remove_zero, lambda3=args.lambda3)
    with open('stat_res/a_llm_res.txt', 'a') as f:
        
        # print(f'{model_name}\t{args.mode}\t{srcc:.5f}\t{plcc:.5f}', file=f)
        print(f'{model_name}\t{args.mode}\t{srcc:.5f}\t{plcc:.5f}\t{(srcc+plcc)/2:.5f}', file=f)
        print(f'{model_name}\t{args.mode}\t{srcc:.5f}\t{plcc:.5f}\t{(srcc+plcc)/2:.5f}')
        # print(f'{(srcc+plcc)/2:.5f}')
    
    # if args.save_llm:
    #     np.save(f'result/{model_name}.npy',np.array(score_llm))
    #     if args.return_mask_score:
    #         with open(f'result/{model_name}-mask-{args.mode[-4:]}.json', 'w') as f:
    #             json.dump(mask_scores, f)
            # np.save(f'result/{model_name}-mask.npy',np.array(mask_scores))

def theoretically_max(args):
    gt_path = {
        'spaq':'/home/zzq/data/results/spaq_gt/MOS and Image attribute scores.csv',
        'livec':'/home/zzq/zzq/others/iqa-metric/gt/livec',
        'koniq':'/home/zzq/zzq/others/iqa-metric/gt/koniq/koniq10k_scores_and_distributions.csv',
        'agiqa':'/home/zzq/zzq/others/iqa-metric/gt/agiqa/data.csv',
        'kadid':'/home/zzq/zzq/others/iqa-metric/gt/kadid/dmos.csv',
        }[args.gt_path]
    
    result_gt = get_score_gt(gt_path)
    score_gt = np.array(result_gt['score'])
    
    score_round = np.round(score_gt / (score_gt.max()-score_gt.min()) * args.n_words)
    result_round = {'score':score_round, 'name':result_gt['name']}
    srcc, plcc = get_srcc_plcc(result_gt, result_round, remove_zero=False)
    
    print(f'{args.gt_path}\t{srcc=:.5f},{plcc=:.5f},{(srcc+plcc)/2:.5f}')

if __name__ == '__main__':
    def str2bool(s:str):
        return s.lower() == 'true'

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='mask-area', choices=['total', 'mask-mean', 'mask-area'])
    parser.add_argument('--gt_path', type=str, default='agiqa', choices=['spaq', 'koniq', 'livec', 'agiqa', 'kadid'])
    parser.add_argument('--path_idx', type=int, default=-1)
    parser.add_argument('--save_llm', action='store_true')
    parser.add_argument('--return_mask_score', action='store_true')
    parser.add_argument('--remove_zero', action='store_true')
    # parser.add_argument('--lambda1', type=float, default=7/31)
    parser.add_argument('--lambda1', type=float, default=0.1)
    parser.add_argument('--lambda2', type=float, default=0.5)
    parser.add_argument('--switch_score', action='store_true')
    parser.add_argument('--switch_pair',type=float, nargs='+')
    parser.add_argument('--add_total',type=str2bool, default=True)
    parser.add_argument('--lambda3', type=float, default=1)
    # switch_pair=(1,2)
    # parser.add_argument('--gt_path', type=str, default='/home/zzq/data/results/spaq_gt/MOS and Image attribute scores.csv')
    
    parser.add_argument('--n_words',type=float, default=5)
    
    args = parser.parse_args()
    # args.switch_pair = [float(s) for s in args.switch_pair]
    main(args)
    
    # theoretically_max(args)