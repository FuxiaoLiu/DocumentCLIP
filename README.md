# DocumentCLIP: Linking Figures and Main Body Text in Reflowed Documents

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

## Data Preprocess
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



