def str2bool(s:str):
    if s.lower() == 'true':
        return True
    else:
        return False

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import json
import tqdm
import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr, pearsonr
import scipy.io as scio
import traceback
import math
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
from transformers import TextStreamer
from transformers import AutoTokenizer, AutoProcessor

from mplug_owl3.modeling_mplugowl3 import mPLUGOwl3Model
from mplug_owl3.configuration_mplugowl3 import mPLUGOwl3Config

from words_systems_queries import get_system_and_query
from gt_srcc_plcc import get_score_gt, get_srcc_plcc

def process_images(gpu_id, image_paths, model_path, result_queue, progress_queue, standard, n_word):
    try:
        torch.cuda.set_device(int(gpu_id))
        #os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        model = mPLUGOwl3Model.from_pretrained(model_path, attn_implementation='sdpa', torch_dtype=torch.half)
        model.eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        processor = model.init_processor(tokenizer)
        results = []
        system, query = get_system_and_query(standard, n_word)
        messages = [{"role": "user", "content": f""" {system}
                    <|image|>
                    {query}"""},
                    {"role": "assistant", "content": ""}]
        for image_path in image_paths:
            image_result = {
                'name':os.path.split(image_path)[1],
                'score':""
            }
            image = Image.open(image_path).convert("RGB")
            messages = [{"role": "user", "content": f""" {system}
                    <|image|>
                    {query}"""},
                    {"role": "assistant", "content": ""}]
            inputs = processor(messages, images=[image], videos=None).to('cuda')
            inputs.update({
                        'tokenizer': tokenizer,
                        'max_new_tokens':100,
                        'decode_text':True,})
            with torch.no_grad():
                outputs = model.generate(**inputs)[0][0]
            #print(outputs)
            try:
                score = int(outputs)
            except Exception as e:
                print(image_result['name'], e)
                score = 1
            image_result['score']=score
            results.append(image_result)
            progress_queue.put(1)
        result_queue.put(results)
    except Exception as e:
        print(f"Error in process_images on GPU {gpu_id}: {e}")
        traceback.print_exc()
        progress_queue.put(1)

def main(args):
    #print(f"Available GPUs: {torch.cuda.device_count()}")
    gt_path = {
        'spaq':'gt/spaq/MOS and Image attribute scores.csv',
        'livec':'gt/livec',
        'koniq':'gt/koniq/koniq10k_scores_and_distributions.csv',
        'agiqa':'gt/agiqa/data.csv',
        'kadid':'gt/kadid/dmos.csv',
        }[args.dataset]
    result_gt = get_score_gt(args.data_dir+gt_path)
    
    image_root = args.data_dir + args.dataset
    os.makedirs(args.result_dir, exist_ok=True)
    save_root = args.result_dir + args.dataset + '-test.json'
    
    image_paths = [os.path.join(image_root, img) for img in os.listdir(image_root)]
    gpu_ids = list(map(int, args.gpu.split(',')))
    num_gpus = len(gpu_ids)
    chunk_size = len(image_paths) // num_gpus
    
    processes = []
    result_queue = multiprocessing.Queue()
    progress_queue = multiprocessing.Queue()
    for i, gpu_id in enumerate(gpu_ids):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i != num_gpus - 1 else len(image_paths)
        p = multiprocessing.Process(target=process_images, args=(gpu_id, image_paths[start_idx:end_idx], args.model_path, result_queue, progress_queue, args.standard, args.n_word))
        p.start()
        processes.append(p)
        
    with tqdm.tqdm(total=len(image_paths)) as pbar:
        for _ in range(len(image_paths)):
            progress_queue.get() 
            pbar.update(1) 
    
    for p in processes:
        p.join()
        
    results = []
    while not result_queue.empty():
        results.extend(result_queue.get())
    with open(save_root, 'w') as f:
        json.dump(results, f)
    
    result_llm = {
        'name':[result['name'] for result in results],
        'score':[result['score'] for result in results]
    }   
    srcc, plcc = get_srcc_plcc(result_gt, result_llm, remove_zero=True)
    print('srcc:',srcc)
    print('plcc:',plcc)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='2,3')

    parser.add_argument('--model_path', type=str, default='mPLUG/mPLUG-Owl3-7B-240728')
    parser.add_argument('--dataset', type=str, default='livec', choices=['spaq', 'koniq', 'livec', 'agiqa', 'kadid'])
    parser.add_argument('--data_dir', type=str, default='datasets/')
    parser.add_argument('--result_dir', type=str, default='results/')

    parser.add_argument('--bbox', type=str2bool, default=True)
    parser.add_argument('--standard', type=str, default='word', choices=['number', 'word', 'sentence', 'number_decimal','word_only'])
    parser.add_argument('--n_word', type=int, default=7)

    args = parser.parse_args()
    multiprocessing.set_start_method('spawn', force=True)
    main(args)