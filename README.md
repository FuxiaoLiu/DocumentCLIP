# DocumentCLIP: Linking Figures and Main Body Text in Reflowed Documents
Fuxiao Liu, Chris Tensmeyer, Hao Tan

In this work, we apply the contrastive learning algorithm to determine the document-internal connections between specific figures and body Text. Our model can be applied to Adobe Liquid mode to improve the reading experience on the smartphone.

Model Preview:
![DocumentCLIP Model!](./model.png)

## Requirements
Clone this repo and build the environment

```
conda env create -f DCLIP.yml --name DCLIP
conda activate DCLIP
```

## Data

- Download the latest Wikipedia<sup>[1](#footnote1)</sup> dump `enwiki-*-pages-articles.xml.bz2` from https://dumps.wikimedia.org/backup-index.html.

### Data Preprocess
Follow the instruction in [preprocess.txt](./Data/instruction.txt)

The final output of this tool is a collection of Wikipedia articles, which are separated by directories, i.e., one directory per one article. The structure in each directory is below:

```
${ARTICLE_NAME}/
  |-- doc.xml : an article text marked with section and image position information
  |-- doc.json : JSON format converted from doc.xml
  |-- *.jpeg : image files in JPEG format
  |-- info/ : additional information stored
        |-- *.license : license information of images
        |-- removed.json : modified parts from the original text
``` 

### Statistic of Our Dataset
![DocumentCLIP dataset!](./statistic.png)

## Model Running

### Sample single-process running code:
```bash
CUDA_VISIBLE_DEVICES=0 python -m training1.main     --save-frequency 1     --zeroshot-frequency 1     --report-to tensorboard     --train-data="./data/validation_wiki.csv"      --val-data="./data/validation_wiki.csv"      --csv-img-key filepath     --csv-caption-key title     --warmup 10000     --batch-size=32     --lr=0.001    --wd=0.1     --epochs=30     --workers=8
```

Note: `imagenet-val` is the path to the *validation* set of ImageNet for zero-shot evaluation, not the training set!
You can remove this argument if you do not want to perform zero-shot evaluation on ImageNet throughout training. 

### Multi-GPU with Single-Node

We make use of `torchrun` to launch distributed jobs. The following launches a
a job on a node of 4 GPUs:

```bash
    
torchrun --nproc_per_node=4 --rdzv_endpoint=$HOSTE_NODE_ADDR -m training1.main \ --train-data="./data/validation_wiki.csv"      --val-data="./data/validation_wiki.csv"   --warmup 10000     --batch-size=64    --lr=0.001   --wd=0.1     --epochs=30     --workers=4
```
### Inference
The checkpoint of our model will be shared upon request. please email: fl3es@umd.edu.

## Notice
This repository and tool are licensed under [Apache 2.0](./LICENSE). 

<a name="footnote1">1</a>: Wikimedia Wordmark is a trademark of the Wikimedia Foundation and is used with the permission of the Wikimedia Foundation. We are not endorsed by or affiliated with the Wikimedia Foundation.
