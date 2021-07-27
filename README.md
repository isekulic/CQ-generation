# CQ-generation
Code for ICTIR 2021 paper "Towards Facet-Driven Generation of Clarifying Questions for Conversational Search". 

## Training
You can run training with:
```
python run.py --model_name 'gpt2' --use_faceted_data 1 --my_faceted_data 'data/ClariQ-FKw.tsv' 
```

For more control over hyperparameters please check out argparse arguments in `run.py`.

## Inference
Given an initial query and facet keywords, the model will generate a clarifying question. Run inference with:
```
python run.py --model_name 'gpt2' --test_mode 1 --test_ckp 'gtp2_ckpt_epoch=6.ckpt' --use_faceted_data 1 --my_faceted_data 'data/ClariQ-FKw.tsv' 
```

Text generation is controlable with several parameters in `run.py`, including:
* temperature,
* top_k,
* top_p,
* min_output_len,
* max_output_len.

### Pre-trained model
You can download fine-tuned GPT-2 model [here](https://drive.switch.ch/index.php/s/cWDW7orKF0YjfsX).

## Cite
If you found this paper useful please cite our ICTIR 2021 paper:
```
@inproceedings{sekulic2021towards,
author = {Sekuli\'c, Ivan and Aliannejadi, Mohammad and Crestani, Fabio},
title = {Towards Facet-Driven Generation of Clarifying Questions for Conversational Search,
year = {2021},
publisher = {Association for Computing Machinery},
booktitle = {Proceedings of the 2021 ACM SIGIR on International Conference on Theory of Information Retrieval},
location = {Virtual Event},
series = {ICTIR '21}
}
```
