# [A Benchmarking Study of Embedding-based Entity Alignment for Knowledge Graphs](http://www.vldb.org/pvldb/vol13/p2326-sun.pdf)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg?style=flat-square)](https://github.com/nju-websoft/OpenEA/issues)
[![License](https://img.shields.io/badge/License-GPL-lightgrey.svg?style=flat-square)](https://github.com/nju-websoft/OpenEA/blob/master/LICENSE)
[![language-python3](https://img.shields.io/badge/Language-Python3-blue.svg?style=flat-square)](https://www.python.org/)
[![made-with-Tensorflow](https://img.shields.io/badge/Made%20with-Tensorflow-orange.svg?style=flat-square)](https://www.tensorflow.org/)
[![Paper](https://img.shields.io/badge/VLDB%202020-PDF-yellow.svg?style=flat-square)](http://www.vldb.org/pvldb/vol13/p2326-sun.pdf)

> Entity alignment seeks to find entities in different knowledge graphs (KGs) that refer to the same real-world object. Recent advancement in KG embedding impels the advent of embedding-based entity alignment, which encodes entities in a continuous embedding space and measures entity similarities based on the learned embeddings. In this paper, we conduct a comprehensive experimental study of this emerging field. This study surveys 23 recent embedding-based entity alignment approaches and categorizes them based on their techniques and characteristics. We further observe that current approaches use different datasets in evaluation, and the degree distributions of entities in these datasets are inconsistent with real KGs. Hence, we propose a new KG sampling algorithm, with which we generate a set of dedicated benchmark datasets with various heterogeneity and distributions for a realistic evaluation. This study also produces an open-source library, which includes 12 representative embedding-based entity alignment approaches. We extensively evaluate these approaches on the generated datasets, to understand their strengths and limitations. Additionally, for several directions that have not been explored in current approaches, we perform exploratory experiments and report our preliminary findings for future studies. The benchmark datasets, open-source library and experimental results are all accessible online and will be duly maintained. 

*** **UPDATE** ***

- Aug. 1, 2021: We release the source code for [entity alignment with dangling cases](https://sunzequn.github.io/articles/acl2021_dbp2.pdf).

- June 29, 2021: We release the [DBP2.0](https://github.com/nju-websoft/OpenEA/tree/master/dbp2.0) dataset for [entity alignment with dangling cases](https://sunzequn.github.io/articles/acl2021_dbp2.pdf).

- Jan. 8, 2021: The results of AliNet on OpenEA datasets are avaliable at [Google docs](https://docs.google.com/spreadsheets/d/1P_MX8V7zOlZjhHlEMiXbXlIaMGSJT1Gh_gZWe4yIQBY/edit?usp=sharing).

- Nov. 30, 2020: We release **a new version (v2.0) of the OpenEA dataset**, where the URIs of DBpedia and YAGO entities are encoded to resovle the [name bias](https://www.aclweb.org/anthology/2020.emnlp-main.515.pdf) issue. It is strongly recommended to use the [v2.0 dataset](https://figshare.com/articles/dataset/OpenEA_dataset_v1_1/19258760/3) for evaluating attribute-based entity alignment methods, such that the results can better reflect the robustness of these methods in real-world situation.

- Sep. 24, 2020: add AliNet.

## Table of contents
1. [Library for Embedding-based Entity Alignment](#library-for-embedding-based-entity-alignment)
    1. [Overview](#overview)
    2. [Getting Started](#getting-started)
        1. [Code Package Description](#package-description)
        2. [Dependencies](#dependencies)
        3. [Installation](#installation)
        4. [Usage](#usage)
2. [KG Sampling Method and Datasets](#kg-sampling-method-and-datasets)
    1. [Iterative Degree-based Sampling](#iterative-degree-based-sampling)
    2. [Dataset Overview](#dataset-overview)
    2. [Dataset Description](#dataset-description)
3. [Experiment and Results](#experiment-and-results)
    1. [Experiment Settings](#experiment-settings)
    2. [Detailed Results](#detailed-results)
4. [License](#license)
5. [Citation](#citation)

## Library for Embedding-based Entity Alignment

### Overview

We use [Python](https://www.python.org/) and [Tensorflow](https://www.tensorflow.org/) to develop an open-source library, namely **OpenEA**, for embedding-based entity alignment. The software architecture is illustrated in the following Figure. 

<p>
  <img width="70%" src="https://github.com/nju-websoft/OpenEA/blob/master/docs/stack.png" />
</p>

The design goals and features of OpenEA include three aspects, i.e., loose coupling, functionality and extensibility, and off-the-shelf solutions.

* **Loose coupling**. The implementations of embedding and alignment modules are independent to each other. OpenEA provides a framework template with pre-defined input and output data structures to make the three modules as an integral pipeline. Users can freely call and combine different techniques in these modules.

* **Functionality and extensibility**. OpenEA implements a set of necessary functions as its underlying components, including initialization functions, loss functions and negative sampling methods in the embedding module; combination and learning strategies in the interaction mode; as well as distance metrics and alignment inference strategies in the alignment module. On top of those, OpenEA also provides a set of flexible and high-level functions with configuration options to call the underlying components. In this way, new functions can be easily integrated by adding new configuration options.

* **Off-the-shelf solutions**. To facilitate the use of OpenEA in diverse scenarios, we try our best to integrate or re-build a majority of existing embedding-based entity alignment approaches. Currently, OpenEA has integrated the following embedding-based entity alignment approaches:
    1. **MTransE**: [Multilingual Knowledge Graph Embeddings for Cross-lingual Knowledge Alignment](https://www.ijcai.org/proceedings/2017/0209.pdf). IJCAI 2017.
    1. **IPTransE**: [Iterative Entity Alignment via Joint Knowledge Embeddings](https://www.ijcai.org/proceedings/2017/0595.pdf). IJCAI 2017.
    1. **JAPE**: [Cross-Lingual Entity Alignment via Joint Attribute-Preserving Embedding](https://link.springer.com/chapter/10.1007/978-3-319-68288-4_37). ISWC 2017.
    1. **KDCoE**: [Co-training Embeddings of Knowledge Graphs and Entity Descriptions for Cross-lingual Entity Alignment](https://www.ijcai.org/proceedings/2018/0556.pdf). IJCAI 2018.
    1. **BootEA**: [Bootstrapping Entity Alignment with Knowledge Graph Embedding](https://www.ijcai.org/proceedings/2018/0611.pdf). IJCAI 2018.
    1. **GCN-Align**: [Cross-lingual Knowledge Graph Alignment via Graph Convolutional Networks](https://www.aclweb.org/anthology/D18-1032). EMNLP 2018.
    1. **AttrE**: [Entity Alignment between Knowledge Graphs Using Attribute Embeddings](https://people.eng.unimelb.edu.au/jianzhongq/papers/AAAI2019_EntityAlignment.pdf). AAAI 2019.
    1. **IMUSE**: [Unsupervised Entity Alignment Using Attribute Triples and Relation Triples](https://link.springer.com/content/pdf/10.1007%2F978-3-030-18576-3_22.pdf). DASFAA 2019.
    1. **SEA**: [Semi-Supervised Entity Alignment via Knowledge Graph Embedding with Awareness of Degree Difference](https://dl.acm.org/citation.cfm?id=3313646). WWW 2019.
    1. **RSN4EA**: [Learning to Exploit Long-term Relational Dependencies in Knowledge Graphs](http://proceedings.mlr.press/v97/guo19c/guo19c.pdf). ICML 2019.
    1. **MultiKE**: [Multi-view Knowledge Graph Embedding for Entity Alignment](https://www.ijcai.org/proceedings/2019/0754.pdf). IJCAI 2019.
    1. **RDGCN**: [Relation-Aware Entity Alignment for Heterogeneous Knowledge Graphs](https://www.ijcai.org/proceedings/2019/0733.pdf). IJCAI 2019.
    1. **AliNet**: [Knowledge Graph Alignment Network with Gated Multi-hop Neighborhood Aggregation](https://aaai.org/ojs/index.php/AAAI/article/view/5354). AAAI 2020.
    
* OpenEA has also integrated the following relationship embedding models and two attribute embedding models (AC2Vec and Label2vec ) in the embedding module:
    1. **TransH**: [Knowledge Graph Embedding by Translating on Hyperplanes](https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/view/8531/8546). AAAI 2014.
    1. **TransR**: [Learning Entity and Relation Embeddings for Knowledge Graph Completion](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9571/9523). AAAI 2015.
    1. **TransD**: [Knowledge Graph Embedding via Dynamic Mapping Matrix](https://aclweb.org/anthology/P15-1067). ACL 2015.
    1. **HolE**: [Holographic Embeddings of Knowledge Graphs](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12484/11828). AAAI 2016.
    1. **ProjE**: [ProjE: Embedding Projection for Knowledge Graph Completion](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/view/14279/13906). AAAI 2017.
    1. **ConvE**: [Convolutional 2D Knowledge Graph Embeddings](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/17366/15884). AAAI 2018.
    1. **SimplE**: [SimplE Embedding for Link Prediction in Knowledge Graphs](https://papers.nips.cc/paper/7682-simple-embedding-for-link-prediction-in-knowledge-graphs.pdf). NeurIPS 2018.
    1. **RotatE**: [RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space](https://openreview.net/pdf?id=HkgEQnRqYQ). ICLR 2019.

### Getting Started
These instructions cover how to get a copy of the library and how to install and run it on your local machine for development and testing purposes. It also provides an overview of the package structure of the source code.

#### Package Description

```
src/
├── openea/
│   ├── approaches/: package of the implementations for existing embedding-based entity alignment approaches
│   ├── models/: package of the implementations for unexplored relationship embedding models
│   ├── modules/: package of the implementations for the framework of embedding module, alignment module, and their interaction
│   ├── expriment/: package of the implementations for evalution methods
```

#### Dependencies
* Python 3.x (tested on Python 3.6)
* Tensorflow 1.x (tested on Tensorflow 1.8 and 1.12)
* Scipy
* Numpy
* Graph-tool or igraph or NetworkX
* Pandas
* Scikit-learn
* Matching==0.1.1
* Gensim

#### Installation
We recommend creating a new conda environment to install and run OpenEA. You should first install tensorflow-gpu (tested on 1.8 and 1.12), graph-tool (tested on 2.27 and 2.29,  the latest version would cause a bug), and python-igraph using conda:
```bash
conda create -n openea python=3.6
conda activate openea
conda install tensorflow-gpu==1.12
conda install -c conda-forge graph-tool==2.29
conda install -c conda-forge python-igraph
```

Then, OpenEA can be installed using pip with the following steps:

```bash
git clone https://github.com/nju-websoft/OpenEA.git OpenEA
cd OpenEA
pip install -e .
```

#### Usage
The following is an example about how to use OpenEA in Python (We assume that you have already downloaded our datasets and configured the hyperparameters as in the [examples](https://github.com/nju-websoft/OpenEA/tree/master/run/args).)
```python
import openea as oa

model = oa.kge_model.TransE
args = load_args("hyperparameter file folder")
kgs = read_kgs_from_folder("data folder")
model.set_args(args)
model.set_kgs(kgs)
model.init()
model.run()
model.test()
model.save()

```
More examples are available [here](https://github.com/nju-websoft/OpenEA/tree/master/run)

To run the off-the-shelf approaches on our datasets and reproduce our experiments, change into the ./run/ directory and use the following script:

```bash
python main_from_args.py "predefined_arguments" "dataset_name" "split"
```

For example, if you want to run BootEA on D-W-15K (V1) using the first split, please execute the following script:

```bash
python main_from_args.py ./args/bootea_args_15K.json D_W_15K_V1 721_5fold/1/
```

## KG Sampling Method and Datasets
As the current widely-used datasets are quite different from real-world KGs, we present a new dataset sampling algorithm to generate a benchmark dataset for embedding-based entity alignment.

### Iterative Degree-based Sampling
The proposed iterative degree-based sampling (IDS) algorithm simultaneously deletes entities in two source KGs with reference alignment until achieving the desired size, meanwhile retaining a similar degree distribution of the sampled dataset as the source KG. The following figure describes the sampling procedure. 

<p>
  <img width="50%" src="https://github.com/nju-websoft/OpenEA/blob/master/docs/KG_sampling.png" />
</p>

### Dataset Overview

We choose three well-known KGs as our sources: DBpedia (2016-10),Wikidata (20160801) and YAGO3. Also, we consider two cross-lingual versions of DBpedia: English--French and English--German. We follow the conventions in JAPE and BootEA to generate datasets of two sizes with 15K and 100K entities, using the IDS algorithm:

*#* Entities | Languages | Dataset names
:---: | :---: | :---: 
15K | Cross-lingual | EN-FR-15K, EN-DE-15K
15K | English | D-W-15K, D-Y-15K
100K | Cross-lingual | EN-FR-100K, EN-DE-100K
100K | English-lingual | D-W-100K, D-Y-100K

The v1.1 datasets used in this paper can be downloaded from [figshare](https://figshare.com/articles/dataset/OpenEA_dataset_v1_1/19258760/2), [Dropbox](https://www.dropbox.com/s/nzjxbam47f9yk3d/OpenEA_dataset_v1.1.zip?dl=0) or [Baidu Wangpan](https://pan.baidu.com/s/1Wb4xMds3PT0IaKCJrPR8Lw) (password: 9feb). (**Note that**, we have fixed a minor format issue in YAGO of our v1.0 datasets. Please download our v1.1 datasets from the above links and use this version for evaluation.)

(**Recommended**) The v2.0 datasets can be downloaded from [figshare](https://figshare.com/articles/dataset/OpenEA_dataset_v1_1/19258760/3), [Dropbox](https://www.dropbox.com/s/xfehqm4pcd9yw0v/OpenEA_dataset_v2.0.zip?dl=0) or [Baidu Wangpan](https://pan.baidu.com/s/19RlM9OqwhIz4Lnogrp74tg) (password: nub1). 



### Dataset Statistics
We generate two versions of datasets for each pair of KGs to be aligned. V1 is generated by directly using the IDS algorithm. For V2, we first randomly delete entities with low degrees (d <= 5) in the source KG to make the average degree doubled, and
then execute IDS to fit the new KG. The statistics of the datasets are shown below.  

<p>
  <img src="https://github.com/nju-websoft/OpenEA/blob/master/docs/Dataset_Statistics.png" />
</p>

### Dataset Description
We hereby take the EN_FR_15K_V1 dataset as an example to introduce the files in each dataset. In the *721_5fold* folder, we divide the reference entity alignment into five disjoint folds, each of which accounts for 20% of the total alignment. For each fold, we pick this fold (20%) as training data and leave the remaining (80%) for validation (10%) and testing (70%). The directory structure of each dataset is listed as follows:

```
EN_FR_15K_V1/
├── attr_triples_1: attribute triples in KG1
├── attr_triples_2: attribute triples in KG2
├── rel_triples_1: relation triples in KG1
├── rel_triples_2: relation triples in KG2
├── ent_links: entity alignment between KG1 and KG2
├── 721_5fold/: entity alignment with test/train/valid (7:2:1) splits
│   ├── 1/: the first fold
│   │   ├── test_links
│   │   ├── train_links
│   │   └── valid_links
│   ├── 2/
│   ├── 3/
│   ├── 4/
│   ├── 5/
```

## Experiment and Results

### Experiment Settings
The common hyper-parameters used for OpenEA are shown below.

<table style="text-align:center">
    <tr>
        <td style="text-align:center"></td>
        <th style="text-align:center">15K</th>
        <th style="text-align:center">100K</th>
    </tr>
    <tr>
        <td style="text-align:center">Batch size for rel. triples</td>
        <td style="text-align:center">5,000</td>
        <td style="text-align:center">20,000</td>
    </tr>
    <tr>
        <td style="text-align:center">Termination condition</td>
        <td style="text-align:center" colspan="2">Early stop when the Hits@1 score begins to drop on <br>
            the validation sets, checked every 10 epochs.</td>
    </tr>
    <tr>
        <td style="text-align:center">Max. epochs</td>
        <td style="text-align:center" colspan="2">2,000</td>
    </tr>
</table>

Besides, it is well-recognized to split a dataset into training, validation and test sets. 
The details are shown below.

| *#* Ref. alignment | *#* Training | *#* Validation | *#* Test |
|:------------------:|:------------:|:--------------:|:--------:|
|        15K         |    3,000     |     1,500      |  10,500  |
|        100K        |    20,000    |     10,000     |  70,000  |

We use Hits@m (m = 1, 5, 10, 50), mean rank (MR) and mean reciprocal rank (MRR) as the evaluation metrics.  Higher Hits@m and MRR scores as well as lower MR scores indicate better performance.

### Detailed Results
The detailed and supplementary experimental results are list as follows:

#### Detailed results of current approaches on the 15K datasets
[**detailed_results_current_approaches_15K.csv**](https://github.com/nju-websoft/OpenEA/blob/master/docs/detailed_results_current_approaches_15K.csv)

#### Detailed results of current approaches on the 100K datasets
[**detailed_results_current_approaches_100K.csv**](https://github.com/nju-websoft/OpenEA/blob/master/docs/detailed_results_current_approaches_100K.csv)

#### Running time (sec.) of current approaches
[**running_time.csv**](https://github.com/nju-websoft/OpenEA/blob/master/docs/running_time.csv)

### Unexplored KG Embedding Models

#### Detailed results of unexplored KG embedding models on the 15K datasets
[**detailed_results_unexplored_models_15K.csv**](https://github.com/nju-websoft/OpenEA/blob/master/docs/detailed_results_unexplored_models_15K.csv)

#### Detailed results of unexplored KG embedding models on the 100K datasets
[**detailed_results_unexplored_models_100K.csv**](https://github.com/nju-websoft/OpenEA/blob/master/docs/detailed_results_unexplored_models_100K.csv)

## License
This project is licensed under the GPL License - see the [LICENSE](LICENSE) file for details

## Citation
If you find the benchmark datasets, the OpenEA library or the experimental results useful, please kindly cite the following paper:
```
@article{OpenEA,
  author    = {Zequn Sun and
               Qingheng Zhang and
               Wei Hu and
               Chengming Wang and
               Muhao Chen and
               Farahnaz Akrami and
               Chengkai Li},
  title     = {A Benchmarking Study of Embedding-based Entity Alignment for Knowledge Graphs},
  journal   = {Proceedings of the VLDB Endowment},
  volume    = {13},
  number    = {11},
  pages     = {2326--2340},
  year      = {2020},
  url       = {http://www.vldb.org/pvldb/vol13/p2326-sun.pdf}
}
```

If you use the DBP2.0 dataset, please kindly cite the following paper:
```
@inproceedings{DBP2,
  author    = {Zequn Sun and
               Muhao Chen and
               Wei Hu},
  title     = {Knowing the No-match: Entity Alignment with Dangling Cases},
  booktitle = {ACL},
  year      = {2021}
}
```

