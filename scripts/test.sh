#test dogiqa on different datasets
python dogiqa/eval.py --dataset spaq
python dogiqa/eval.py --dataset koniq
python dogiqa/eval.py --dataset livec
python dogiqa/eval.py --dataset agiqa
python dogiqa/eval.py --dataset kadid

#set different parameters for dogiqa
python dogiqa/eval.py --Bbox True --standard word --n_word 7

#set GPU and paths
python dogiqa/eval.py \
  --gpu 0 \
  --model_path mPLUG/mPLUG-Owl3-7B-240728 \
  --sam2_path segment-anything-2/checkpoints/sam2_hiera_large.pt \
  --dataset spaq \
  --data_dir datasets/ \
  --result_dir results/