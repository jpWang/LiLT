# LiLT (ACL 2022)

This is the official PyTorch implementation of the ACL 2022 paper: "[LiLT: A Simple yet Effective Language-Independent Layout Transformer for Structured Document Understanding](https://arxiv.org/abs/2202.13669)".

<img src="./figs/framework.png" alt="framework"/>

LiLT is pre-trained on the visually-rich documents of a single language (English) and can be directly fine-tuned on other languages with the corresponding off-the-shelf monolingual/multilingual pre-trained textual models. We hope the public availability of this work can help document intelligence researches.

## Installation

For CUDA 11.X: 

~~~bash
conda create -n liltfinetune python=3.7
conda activate liltfinetune
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.0 -c pytorch
python -m pip install detectron2==0.5 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
git clone https://github.com/jpWang/LiLT
cd LiLT
pip install -r requirements.txt
pip install -e .
~~~

Or check [Detectron2](https://github.com/facebookresearch/detectron2/releases)/[PyTorch](https://pytorch.org/get-started/previous-versions/) versions and modify the command lines accordingly.

## Datasets

In this repository, we provide the fine-tuning codes for [FUNSD](https://guillaumejaume.github.io/FUNSD/) and [XFUND](https://github.com/doc-analysis/XFUND). 

You can download our pre-processed data (~1.2GB) from [here](https://1drv.ms/u/s!Ahd-h7H5akVZeZQvKieg8g5THV8?e=mBRnxw), and put the unzipped `xfund&funsd/` under `LiLT/`. 

## Models

| Model                         | Language  | Size  | Download     | 
| ----------------------------- | --------- | ----- | ------------ |
| `lilt-roberta-en-base`        | EN        | 293MB | [OneDrive](https://1drv.ms/u/s!Ahd-h7H5akVZfhPVHQQ1tOypA48?e=nraHn3)    | 
| `lilt-infoxlm-base`           | MUL       | 846MB | [OneDrive](https://1drv.ms/u/s!Ahd-h7H5akVZfeIhAQ8KHELRvcc?e=WS1P82)    |
| `lilt-only-base`              | None      | 21MB  | [OneDrive](https://1drv.ms/u/s!Ahd-h7H5akVZfEIRbCmcWKjhoSM?e=6tMGbe)    | 

If you want to combine the pre-trained LiLT with the *RoBERTa*s of **other languages**, please download  `lilt-only-base` and use `gen_weight_roberta_like.py` to generate your own pre-trained weight.

For example, combine `lilt-only-base` with English `roberta-base`:

~~~bash
mkdir roberta-en-base
wget https://huggingface.co/roberta-base/resolve/main/config.json -O roberta-en-base/config.json
wget https://huggingface.co/roberta-base/resolve/main/pytorch_model.bin -O roberta-en-base/pytorch_model.bin
python gen_weight_roberta_like.py \
     --lilt lilt-only-base/pytorch_model.bin \
     --text roberta-en-base/pytorch_model.bin \
     --config roberta-en-base/config.json \
     --out lilt-roberta-en-base
~~~

Or combine `lilt-only-base` with `microsoft/infoxlm-base`:

~~~bash
mkdir infoxlm-base
wget https://huggingface.co/microsoft/infoxlm-base/resolve/main/config.json -O infoxlm-base/config.json
wget https://huggingface.co/microsoft/infoxlm-base/resolve/main/pytorch_model.bin -O infoxlm-base/pytorch_model.bin
python gen_weight_roberta_like.py \
     --lilt lilt-only-base/pytorch_model.bin \
     --text infoxlm-base/pytorch_model.bin \
     --config infoxlm-base/config.json \
     --out lilt-infoxlm-base
~~~


## Fine-tuning


### Semantic Entity Recognition on FUNSD

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 examples/run_funsd.py \
        --model_name_or_path lilt-roberta-en-base \
        --tokenizer_name roberta-base \
        --output_dir ser_funsd_lilt-roberta-en-base \
        --do_train \
        --do_predict \
        --max_steps 2000 \
        --per_device_train_batch_size 8 \
        --warmup_ratio 0.1 \
        --fp16
```

### Language-specific (For example, ZH) Semantic Entity Recognition on XFUND

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 examples/run_xfun_ser.py \
        --model_name_or_path lilt-infoxlm-base \
        --tokenizer_name xlm-roberta-base \
        --output_dir ls_ser_xfund_zh_lilt-infoxlm-base \
        --do_train \
        --do_eval \
        --lang zh \
        --max_steps 2000 \
        --per_device_train_batch_size 16 \
        --warmup_ratio 0.1 \
        --fp16
```

### Language-specific (For example, ZH) Relation Extraction on XFUND

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 examples/run_xfun_re.py \
        --model_name_or_path lilt-infoxlm-base \
        --tokenizer_name xlm-roberta-base \
        --output_dir ls_re_xfund_zh_lilt-infoxlm-base \
        --do_train \
        --do_eval \
        --lang zh \
        --max_steps 5000 \
        --per_device_train_batch_size 8 \
        --learning_rate  6.25e-6 \
        --warmup_ratio 0.1 \
        --fp16
```


## Results

### Semantic Entity Recognition on FUNSD
<img src="./figs/funsd.png" width=500 alt="funsd"/>

### Language-specific Fine-tuning on XFUND
<img src="./figs/ls_xfund.png" alt="ls_xfund"/>

### Cross-lingual Zero-shot Transfer on XFUND
<img src="./figs/cl_xfund.png" alt="cl_xfund"/>

### Multitask Fine-tuning on XFUND
<img src="./figs/mt_xfund.png" alt="mt_xfund"/>



## Acknowledge

The repository benefits greatly from [unilm/layoutlmft](https://github.com/microsoft/unilm/tree/master/layoutlmft). Thanks a lot for their excellent work.

## Citation
If our paper helps your research, please cite it in your publication(s):
```
@inproceedings{wang2022LiLT,
  title={LiLT: A Simple yet Effective Language-Independent Layout Transformer for Structured Document Understanding},
  author={Wang, Jiapeng and Jin, Lianwen and Ding, Kai},
  booktitle={ACL},
  year={2022}
  }
```

## Feedback
Suggestions and discussions are greatly welcome. Please contact the authors by sending email to `eejpwang@mail.scut.edu.cn`.
