# Clinical Trial Retrieval System  
**Using Two-Stage Hybrid Retrieval, Query Expansion, and LLM-based Reranking**

_Spring Semester 2024-25_

---

##  Overview

This repository develops an advanced **clinical trial retrieval system** combining **sparse retrieval**, **semantic reranking**, and **query expansion techniques** to improve medical document retrieval accuracy. 

It integrates:
- Sparse Retrieval (BM25)
- Dense Semantic Retrieval (Sentence-BERT)
- LLM-based Reranking (Mistral-7B)
- Query Expansion Techniques (NLS, KS, RM3, T5)


---

##  Features

| Feature | Description |
|:--------|:------------|
| **NLS (Natural Language Summarization)** | Summarizes queries by selecting key medical sentences. |
| **KS (Keyword Summarization)** | Generates focused queries by extracting important keywords. |
| **RM3 (Relevance Model 3)** | Traditional probabilistic query expansion technique for better retrieval. |
| **T5 Summarization** | Uses the T5 Transformer to summarize medical queries and documents. |
| **BM25 Retrieval** | Classical sparse retrieval using probabilistic term weighting. |
| **Sentence-BERT Reranking** | Re-ranks top documents based on semantic similarity. |
| **Mistral-7B Reranking** | Large Language Model-based deep contextual reranking. |


---

##  Installation

To run this project, follow these steps:

### Clone the repository:
```bash
git clone https://github.com/subhaorku/Clinical_Trial_Indexing_Retrieval_System.git
cd Clinical_Trial_Indexing_Retrieval_System

Environment Setup:
Create a dedicated environment and install all dependencies:

bash
Copy
Edit
conda create --name disease python=3.8
conda activate disease
conda update -n base -c defaults conda
conda install -c conda-forge openjdk=11
conda env config vars set JAVA_HOME=$(dirname $(dirname $(which java)))
conda deactivate
conda activate disease
```

## Project Structure

File/Folder	Description
extraction_of_doc.py	XML Parsing and metadata extraction
topics.xml	Input topics (patient queries)
Index/	BM25 Index files (large, ignored from Git)
Dataset/	Clinical trial dataset (ignored from Git)
outputs/	Result files (TREC and JSON formats)
Unipd.py	Query expansion techniques implementation
.gitignore	To avoid pushing large files and datasets


## Usage
You can run the scripts in different modes based on your task:

Retrieval and Reranking:
To run the hybrid retrieval and reranking system:

bash
Copy
Edit
python main.py Run1
Run1: BM25 + Sentence-BERT + Hybrid scoring

Run2: BM25 + LLM (Mistral-7B) reranking

Run3, Run4: Other configurations (customized runs)



## Results Summary
BM25 retrieval provides strong baseline results.

Sentence-BERT improves semantic matching between patient queries and documents.

Mistral-7B LLM-based reranking shows excellent contextual understanding.

Query Expansion Techniques significantly enhance coverage and recall.

Hybrid scoring balances classic and deep-learning-based retrieval models for best performance.

Eligibility Filtering ensures only trials matching patient criteria are ranked high.


