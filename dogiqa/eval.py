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
import pycocotools.mask as mask_util
import traceback
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
from transformers import TextStreamer
from transformers import AutoTokenizer, AutoProcessor

from mplug_owl3.modeling_mplugowl3 import mPLUGOwl3Model
from mplug_owl3.configuration_mplugowl3 import mPLUGOwl3Config

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

from words_systems_queries import get_system_and_query
from gt_srcc_plcc import get_score_gt, get_srcc_plcc
from get_mask import get_final_mask, get_masked_image

def process_images(gpu_id, image_paths, model_path, sam2_path, result_queue, progress_queue, bbox=True, standard='word', n_word=7, lambda1=0, lambda2=0):
    try:
        torch.cuda.set_device(int(gpu_id))
        
        sam2_checkpoint = sam2_path
        model_cfg = "sam2_hiera_l.yaml"
        device_name = 'cuda'
        sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device_name, apply_postprocessing=True)
        min_area = 1000
        mask_generator = SAM2AutomaticMaskGenerator(
            model=sam2,
            points_per_side=8,
            points_per_batch=128,
            pred_iou_thresh=0.9,
            stability_score_thresh=0.8,
            stability_score_offset=0.7,
            crop_n_layers=0,
            box_nms_thresh=0.9,
            crop_n_points_downscale_factor=1.2,
            min_mask_region_area=1000,
            use_m2m=False,
            output_mode='coco_rle'
        )
        
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
            image_name = os.path.split(image_path)[1]
            image_result = {
                'name':image_name,
                'score':0
            }
            image = Image.open(image_path).convert("RGB")
            messages = [{"role": "user", "content": f""" {system}
                    <|image|>
                    {query}"""},
                    {"role": "assistant", "content": ""}]
            inputs = processor(messages, images=[image], videos=None).to('cuda')
            inputs.update({
                        'tokenizer': tokenizer,
                        'max_new_tokens':1,
                        'decode_text':True,})
            with torch.no_grad():
                outputs = model.generate(**inputs)[0][0]
            #print(outputs)
            try:
                total_score = int(outputs)
            except Exception as e:
                print(image_result['name'], e)
                total_score = 1
            
            image = np.array(image)
            masks = mask_generator.generate(image)
            final_mask = get_final_mask(image, masks, min_area)
            for mask in final_mask:
                cropped_image = get_masked_image(image, mask['segmentation'], bbox=bbox)
                messages = [{"role": "user", "content": f""" {system}
                        <|image|>
                        {query}"""},
                        {"role": "assistant", "content": ""}]
                inputs = processor(messages, images=[cropped_image], videos=None).to('cuda')
                inputs.update({
                            'tokenizer': tokenizer,
                            'max_new_tokens':1,
                            'decode_text':True,})
                with torch.no_grad():
                    outputs = model.generate(**inputs)[0][0]
                #print(outputs)
                try:
                    score = int(outputs)
                except Exception as e:
                    print(image_result['name'], e)
                    score = 1
                image_result['score']+=score*mask['area']
            area_sum = sum([mask['area'] for mask in final_mask])
            image_result['score'] = image_result['score']/area_sum + lambda1*len(final_mask) + lambda2*total_score
            results.append(image_result)
            progress_queue.put(1)
        #print(f"Results on GPU {gpu_id}: {results}")
        result_queue.put(results)
    except Exception as e:
        print(f"Error in process_images on GPU {gpu_id}: {e}")
        traceback.print_exc()
        progress_queue.put(1)
        result_queue.put(None)

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
    save_root = args.result_dir + args.dataset + '.json'
    
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
        p = multiprocessing.Process(target=process_images, args=(gpu_id, image_paths[start_idx:end_idx], args.model_path, args.sam2_path, result_queue, progress_queue, args.bbox, args.standard, args.n_word, args.lambda1, args.lambda2))
        p.start()
        processes.append(p)
        
    total_images = len(image_paths)
    pbar = tqdm.tqdm(total=total_images)

    while True:
        if not progress_queue.empty():
            progress_queue.get()
            pbar.update(1)  # 更新进度条

        if not result_queue.empty():
            result = result_queue.get()
            if result is None:
                print("Error detected, terminating all processes.")
                for p in processes:
                    p.terminate()
                exit(1)
            results.extend(result)

        # 检查是否所有进程都已完成
        if pbar.n >= total_images:
            break
    
    for p in processes:
        p.join()
        
    results = []
    while not result_queue.empty():
        result = result_queue.get()
        if result is not None:
            results.extend(result)
    #print(f"Final results: {results}")
    with open(save_root, 'w') as f:
        json.dump(results, f)
    #print(f"Results saved to {save_root}")
    
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
    parser.add_argument('--gpu', type=str, default='1,2,3')

    parser.add_argument('--model_path', type=str, default='mPLUG/mPLUG-Owl3-7B-240728')
    parser.add_argument('--sam2_path', type=str, default='segment-anything-2/checkpoints/sam2_hiera_large.pt')
    parser.add_argument('--dataset', type=str, default='spaq', choices=['spaq', 'koniq', 'livec', 'agiqa', 'kadid'])
    parser.add_argument('--data_dir', type=str, default='datasets/')
    parser.add_argument('--result_dir', type=str, default='results/')

    parser.add_argument('--bbox', type=str2bool, default=True)
    parser.add_argument('--standard', type=str, default='word', choices=['number', 'word', 'sentence', 'number_decimal','word_only'])
    parser.add_argument('--n_word', type=int, default=7)
    parser.add_argument('--lambda1', type=int, default=0.1)
    parser.add_argument('--lambda2', type=int, default=0.5)

    args = parser.parse_args()
    multiprocessing.set_start_method('spawn', force=True)
    main(args)