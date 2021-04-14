# Leftmost Segment Determination for Chunking

In NAACL-2021, we introduce a fast and effective approach for sequence segmentation tasks (e.g., Chinese POS tagging).
This repo. contains the main implementation of our method.

## Setup

Two steps. Firstly, create a folder named "dataset" (containing {train, dev, & test}.txt)
and the data format is
```
中      NR
美      NR
在      P
沪      NR
签订    VV
高      JJ
科技    NN
合作    NN
协议    NN
```

Secondly, download [evaluation script](https://www.clips.uantwerpen.be/conll2000/chunking/conlleval.txt)
and rename it as "conlleval.pl". 

## Training and Test
```
CUDA_VISIBLE_DEVICES=0 python main.py \
    --data_dir dataset \
    --check_dir save \
    --script_path conlleval.pl
```

## Citation
```
@inproceedings{li-etal-2021-neural,
    title = "Neural Sequence Segmentation as Determining the Leftmost Segments",
    author = "Li, Yangming and Liu, Lemao and Yao, Kaisheng",
    booktitle = "Proceedings of the 2021 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    publisher = "Association for Computational Linguistics",
}
```
