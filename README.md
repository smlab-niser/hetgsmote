# HetGSMOTE: A SMOTE-Based Oversampling Framework for Heterogeneous Graphs

Graph Neural Networks (GNNs) have proven effective for learning from graph-structured data, with heterogeneous graphs (HetGs) gaining particular prominence for their ability to model diverse real-world systems through multiple node and edge types. However, **class imbalance**—where certain node classes are significantly underrepresented—presents a critical challenge for node classification on HetGs.

This work introduces **HetGSMOTE**, a novel oversampling framework that extends SMOTE-based techniques to heterogeneous graphs by incorporating **node type, edge type**, and **metapath** information into synthetic sample generation. It constructs a **content- and neighbor-type-aggregated embedding space**, generates synthetic minority nodes, and trains **specialized edge generators** for each node type to preserve relational structure.

HetGSMOTE outperforms state-of-the-art baselines across multiple benchmark datasets and heterogeneous GNN architectures—especially under **extreme class imbalance scenarios**.

---

## 📦 Setup

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 📊 Datasets Overview

The datasets used include varying types of nodes, edges, and levels of imbalance. The labeled node type is marked with `*`.

| Dataset      | Node Types           | #Nodes | Edge Types                | #Edges | Classes (Minority) |
|--------------|----------------------|--------|----------------------------|--------|---------------------|
| **AMiner-AII** | author*, paper, venue | 20,171 | author-paper, paper-paper, paper-venue | 70,212 | 4 (3) |
| **IMDb**     | movie*, actor, director | 4,666  | movie-actor, movie-director | 18,656 | 3 (2) |
| **DBLP**     | paper*, author, term, conference | 14,328 | author-paper, paper-term, paper-conference | 119,783 | 4 (3) |
| **PubMed**   | nodes                 | 63,109 | edges                      | 244,986| 8 (6) |

---


## 📂 Folder Structure

```bash
25-hetero-smote/
├── aminer_train/
│   └── data/
│   └── train/
│       ├── han/
│       ├── hgnn/
│       └── magnn/
├── imdb_train/
├── dblp_train/
├── pubmed_train/
├── requirements.txt
├── metric.py
├── smote.py
├── args.py
├── results.md
└── README.md
```

---

## 🚀 Results Reproducibility

### ➤ Imbalance Ratio and Training Size Experiments
```bash
python imdb_train/train/han/trainer.py
```

### ➤ Upsampling Ratio Experiments
```bash
python imdb_train/train/han/trainer_up.py
```

- Replace `han` in the path with `hgnn` or `magnn` to switch models.
- Replace `imdb_train` with `aminer_train`, `dblp_train`, or `pubmed_train` to run on other datasets.
- For `aminer_train` and `pubmed_train`, `trainer.py` file runs both the tests.
- These files include both training and evaluation code for all the list of parameters, iterations and metrics.

---


## 📄 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{ansad2025hetgsmote,
  title={Het{GSMOTE}: Oversampling for Heterogeneous Graphs},
  author={Adhilsha Ansad and Deependra Singh and Subhankar Mishra and Rucha Bhalchandra Joshi},
  booktitle={Northern Lights Deep Learning Conference 2026},
  year={2025},
  url={https://openreview.net/forum?id=rLzas2xQGs}
}
```

The citation information will be updated upon the release of the official publication.


---

 <!-- args.py: Defines hyperparameters and other parameters

raw_Data_process: Reads and processes raw data into files

DeepWalk.py: Uses het_random_walks, which is a list of random walks, in Word2Vec model to generate learned feature embeddings 
            for nodes of all types. Output: node_net_embedding.txt

data_generator.py: (a: author, p: paper, v: venue)

  A)                  Data File                            |         Matrix                                                           
    1) a_p_list_train.txt- a:{List of written papers}        ---- a_p_list_train 
    2) p_a_list_train.txt- p:{List of authors} (Same as 1)   ---- p_a_list_train
    3) p_p_citation_list.txt- p:{List of cited papers}       ---- p_p_cite_list_train
    4) v_p_list_train.txt- v:{List of papers}                ---- v_p_list_train 
    5) p_v.txt- p:v                                          ---- p_v

    # Read these to corresponding numpy matrices: a_p_list_train, p_a_list_train, p_p_cite_list_train, v_p_list_train, and
    # p_neigh_list_train: [[p, {List of author neighbours}, {List of cited papers}, {List of venue neighbour}] ...]

  B)                  Data File                           |         Matrix 
    1) p_abstract_embed.txt- p {embedding of dim 128}       ---- p_abstract_embed  
    2) p_title_embed.txt- p {embedding of dim 128}          ---- p_title_embed
    3) node_net_embedding- {a/p/v} {embedding of dim 128}   ---- a_net_embed, p_net_embed, v_net_embed
 
  C) p_v_net_embed: [p v_net_embed[v]], each row index corresponding to p, the row is filled with net_embed of corresponding v. 

  D) p_a_net_embed: [p mean(a_net_embed[{List of author neighbours}])]: Vanilla aggregated neighbour author embedding to p

  E) p_ref_net_embed: [p mean(p_net_embed[{List of cited papers}])]: : vanilla aggregated neighbour paper embedding to p

  F) a_text_embed: Shape[a, 3*embed_dim/(3*128)] Stores the abstract info of top 3 neighbour papers
    1) if len(written papers) > 3: [[a {p_abstract_embed[Neighbour 1]}, {p_abstract_embed[Neighbour 2]}, {For 3}] ...]
    2) if len(written papers) < 3 (not empty): Concat one embedding twice or thrice

  G) v_text_embed: [[v {p_abstract_embed[Neighbour 1]}, {p_abstract_embed[Neighbour 2]}, {Upto 5}] ...] Use p_v

  H) Use het_neigh_train.txt- a/p/v {Neighbours from random walk}
    1) a_neigh_list_train: [[a {List of author neighbours}][a {List of papers}][a {List of venue neighbours}]] 
	2) p_neigh_list_train: " (Similar to above)
    3) v_neigh_list_train: "
    4) a_neigh_list_train_top, p_neigh_list_train_top, v_neigh_list_train_top: Only neighbours above threshold are taken: [a=10,p=10,v=3]
 
  I) train_id_list = [[] for i in range(3)] Stores ids at each row a/pv. Uses a_neigh_list_train, p_neigh_list_train, v_neigh_list_train

--------------------> 