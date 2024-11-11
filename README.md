<div align="center">
<p align="center"> <img src="figs/logo.png" width="200px"> </p>
</div>

# Dog-IQA: Standard-guided Zero-shot MLLM for Mix-grained Image Quality Assessment
PyTorch code for our paper "Dog-IQA: Standard-guided Zero-shot MLLM for Mix-grained Image Quality Assessment"


[Kai Liu](https://kai-liu001.github.io/), [Ziqing Zhang](), [Wenbo Li](https://fenglinglwb.github.io/), [Renjing Pei](https://orcid.org/0000-0001-7513-6576), [Fenglong Song](https://scholar.google.com/citations?hl=zh-CN&pli=1&user=WYDVk5oAAAAJ), [Xiaohong Liu](https://jhc.sjtu.edu.cn/~xiaohongliu/), [Linghe Kong](https://www.cs.sjtu.edu.cn/~linghe.kong/), and [Yulun Zhang](http://yulunzhang.com/)

"Dog-IQA: Standard-guided Zero-shot MLLM for Mix-grained Image Quality Assessment", arXiv, 2024

[[arXiv](https://arxiv.org/abs/2410.02505)] [[supplementary material](https://github.com/Kai-Liu001/Dog-IQA/releases/tag/v1)] [visual results] 

#### 🔥🔥🔥 News

- **2024-10-03:** Add pipeline figure and results. 
- **2024-10-01:** This repo is released! 🎉🎉🎉

---

> **Abstract:** Image quality assessment (IQA) serves as the golden standard for all models' performance in nearly all computer vision fields. However, it still suffers from poor out-of-distribution generalization ability and expensive training costs. To address these problems, we propose Dog-IQA, a standard-guided zero-shot mix-grained IQA method, which is training-free and utilizes the exceptional prior knowledge of multimodal large language models (MLLMs). To obtain accurate IQA scores, namely scores consistent with humans, we design an MLLM-based inference pipeline that imitates human experts. In detail, Dog-IQA applies two techniques. First, Dog-IQA objectively scores with specific standards that utilize MLLM's behavior pattern and minimize the influence of subjective factors. Second, Dog-IQA comprehensively takes local semantic objects and the whole image as input and aggregates their scores, leveraging local and global information. Our proposed Dog-IQA achieves state-of-the-art (SOTA) performance compared with training-free methods, and competitive performance compared with training-based methods in cross-dataset scenarios. Our code and models will be available at https://github.com/Kai-Liu001/Dog-IQA.

---

<p align="center">
  <img width="900" src="figs/pipeline.png">
</p>

<p align="center">
  <img width="900" src="figs/human.png">
</p>

---

The radar plot in Figure 1 of the main paper shows that our proposed Dog-IQA outperforms all previous training-free IQA methods in all five datasets in terms of both Spearman Rank Correlation Coefficient (SRCC) and Pearson Linear Correlation Coefficient (PLCC).

<p align="center">
  <img width="400" src="figs/radar-zeroshot.png">
</p>

---

## ⚒ TODO

* [x] Release code

## <a name="dependencies"></a>📦 Dependencies

- Python 3.10
- PyTorch 2.4.1+cu121

```bash
# Clone the github repo and go to the default directory 'Dog-IQA'.
git clone https://github.com/Kai-Liu001/Dog-IQA.git
conda create -n dogiqa python=3.10
conda activate dogiqa
cd Dog-IQA
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
git clone https://github.com/facebookresearch/sam2.git
cd segment-anything-2
pip install -e .
```

## 🔗 Contents

1. [Dependencies](#-dependencies)
2. [Datasets](#-datasets)
3. [Evaluation](#-evaluation)
4. [Results](#-results)
5. [Citation](#-citation)
6. [Acknowledgements](#-acknowledgements)

## <a name="datasets"></a>📑 Datasets

Used testing sets can be downloaded as follows:

| Testing Set | Download Link |
|---------|---------|
| [SPAQ](https://github.com/h4nwei/SPAQ) (10k smartphone photography) + [LIVEC](https://live.ece.utexas.edu/research/ChallengeDB/index.html) (1k in-the-wild images) + [KonIQ](https://github.com/subpic/koniq) (10k in-the-wild images) + [AGIQA](https://github.com/lcysyzxdxc/AGIQA-3k-Database) (3k AI-generated images) + [KADID](https://database.mmsp-kn.de/kadid-10k-database.html) (10k artificially distorted​ images) |[Google Drive]() / [Baidu Disk]()|

Download testing datasets and put them into the corresponding folders of `datasets/`.

## <a name="evaluation"></a>🎯 Evaluation

- Download the pre-trained mPLUG-Owl3 from [ModelScope](https://modelscope.cn/models/iic/mPLUG-Owl3-7B-240728) or [Huggingface](https://huggingface.co/mPLUG/mPLUG-Owl3-7B-240728).

- Download the pre-trained SAM2 by running the following script:

```bash
sh segment-anything-2/checkpionts/download_ckpts.sh
```

- Download testing datasets from [Google drive]() or [Baidu Drive]() and place them in `datasets/`.

- Run the following script to test Dog-IQA on SPAQ. More scripts can be found in `scripts/test.sh`

```bash
python dogiqa/eval.py --dataset spaq 
```
  
- The output is in `results/`.

## <a name="results"></a>🔎 Results

We achieve SOTA performance on various dataset compared with training-free approaches.

<details>
<summary>Comparisons with Training-free methods. (click to expand)</summary>

- Quantitative comparisons in Table 2 of the main paper


<p align="center">
  <img width="900" src="figs/t1.png">
</p>

</details>
<details>
<summary>Comparisons with Training-based methods. (click to expand)</summary>

- Quantitative comparisons in Table 3 of the main paper


<p align="center">
  <img width="900" src="figs/t2.png">
</p>

</details>
<details>
<summary>Visual Results. (click to expand)</summary>

- Visual result in Figure 5 of the main paper.

<p align="center">
  <img width="900" src="figs/vis-2.png">
</p>

- Visual result in Figure 2 of the supplementary material.

<p align="center">
  <img width="900" src="figs/vis_final2.png">
</p>
</details>

## <a name="citation"></a>📎 Citation

If you find the code helpful in your resarch or work, please cite the following paper(s).

TBD.

## <a name="acknowledgements"></a>💕💖💕 Acknowledgements

Thanks to [mPLUG-Owl3](https://github.com/X-PLUG/mPLUG-Owl) and [SAM2](https://github.com/facebookresearch/sam2) for their outstanding models.