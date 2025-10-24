# `dataset_builder_wiki` Module

## 1. Overview

The `dataset_builder_wiki` module is a sophisticated data engineering pipeline designed for the systematic construction of graph-based datasets derived from the Wikipedia knowledge base. The primary objective of this module is to transform the semi-structured hyperlink topology of Wikipedia articles into a structured, tensor-based format amenable to advanced graph-based machine learning and Natural Language Processing (NLP) research.

The module facilitates the exploration of semantic relationships between concepts as represented by the connectivity of Wikipedia articles, providing a robust foundation for tasks such as node classification, link prediction, and graph-level analysis.

## 2. Methodological Workflow

The dataset generation process is articulated as a multi-stage, sequential pipeline, ensuring modularity, reproducibility, and computational efficiency.

### 2.1. Stage 1: Graph Extraction from Wikipedia

-   **Input**: A curated list of seed Wikipedia article titles, specified in `wikipedia_articles.txt`. These titles serve as the initial nodes for the graph construction process.

-   **Process**: The module employs a Breadth-First Search (BFS) algorithm, implemented in `wiki_crowler_bfs.py`, to systematically traverse the hyperlink structure of Wikipedia, originating from each seed article. For each seed, a graph is constructed where nodes correspond to Wikipedia articles and edges represent hyperlinks between them. The graph extraction for multiple seed articles is parallelized using a multi-threaded approach managed by `wiki_dataset_builder.py` to optimize performance.

-   **Output**: A collection of JSON files. Each file encapsulates the graph structure (nodes and edges) corresponding to the local neighborhood of a seed article.

### 2.2. Stage 2: Tensor-based Dataset Conversion

-   **Input**: The set of JSON files representing the extracted graphs from the preceding stage.

-   **Process**: This stage, orchestrated by `wiki_graph_json_to_tensor.py`, is responsible for the semantic enrichment and structural transformation of the graphs. Each node's textual label (the article title) is converted into a high-dimensional vector embedding using a pre-trained sentence-transformer model from the Hugging Face model hub. The entire graph is then converted into a `torch_geometric.data.Data` object, which includes the node feature tensor and the edge index tensor. This computationally intensive process is also parallelized to ensure efficiency.

-   **Output**: A directory of `.pt` files. Each file contains a serialized PyTorch Geometric `Data` object, representing a single graph with its associated node embeddings and connectivity information, ready for consumption by a machine learning model.

### 2.3. Stage 3: Data Loading for Model Training

-   **Input**: The directory containing the pre-processed `.pt` tensor files.

-   **Process**: The `wiki_torch_loader.py` script provides a custom PyTorch `Dataset` implementation (`WikiGraphDataset` and `WikiBaseDataset`). These classes facilitate the efficient loading and batching of the graph data. The implementation includes essential functionalities for partitioning the dataset into training and testing subsets, a critical step for rigorous model evaluation.

-   **Output**: PyTorch `DataLoader` objects that can be seamlessly integrated into a standard PyTorch or PyTorch Geometric model training and evaluation pipeline.

## 3. Core Components

-   `create_dataset_main.py`: Serves as the master script that orchestrates the end-to-end execution of the data generation pipeline, from initial graph extraction to final tensor conversion.

-   `wiki_crowler_bfs.py`: Implements the core BFS web crawling logic for the extraction of graph data from Wikipedia.

-   `wiki_dataset_builder.py`: Manages the parallelized construction of multiple graphs from the list of seed articles, producing the intermediate JSON representation.

-   `wiki_graph_json_to_tensor.py`: Handles the conversion of the JSON-formatted graphs into semantically enriched tensor-based `Data` objects.

-   `wiki_torch_loader.py`: Provides the necessary data loader classes for interfacing the processed dataset with a PyTorch-based machine learning framework.

-   `wikipedia_articles.txt`: A configuration file containing a newline-separated list of seed Wikipedia article titles that form the basis of the dataset.

## 4. Usage

To generate the complete dataset, execute the main orchestration script from the command line:

```bash
python create_dataset_main.py
```

This will initiate the full pipeline, and the final tensor dataset will be stored in the designated output directory.
