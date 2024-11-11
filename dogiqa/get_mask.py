import numpy as np
import pycocotools.mask as mask_util
from PIL import Image
import os        
import skimage.measure as skm

def get_final_mask(image, masks, min_area):        
    final_mask = []
    for mask in masks:
        if mask['area'] >= min_area:
            final_mask.append(mask)
    remain_area = image.shape[0]*image.shape[1]
    if len(final_mask) == 0:
        m = np.array(np.ones(image.shape[:2]), dtype=np.bool_)
        rle = mask_util.encode(np.asfortranarray(m.astype(np.bool_)))
        rle['counts'] = rle['counts'].decode('utf-8')
        final_mask=[{'segmentation':rle, 'area':remain_area}]
    else:
        combined_mask = None
        
        for mask in final_mask:
            rle = mask['segmentation']
            m = np.array(mask_util.decode(rle), dtype=np.bool_)
                
            if combined_mask is None:
                combined_mask = m
            else:
                combined_mask = np.logical_or(combined_mask, m)
            remain_area -= mask['area']
        
        # print(rle)
        if remain_area >= min_area:
            inverse_mask = np.logical_not(combined_mask)
            rle = mask_util.encode(np.asfortranarray(inverse_mask.astype(np.bool_)))
            rle['counts'] = rle['counts'].decode('utf-8')
            final_mask.append({'segmentation':rle, 'area':remain_area})
        
    return final_mask

def get_masked_image(image, coco_mask, bbox=True):
    binary_mask = mask_util.decode(coco_mask)
    
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

