# `gnn_networks` Module: An Advanced Framework for Graph-Based Link Prediction

## 1. Overview

The `gnn_networks` module provides a comprehensive and rigorous framework for the implementation, training, and in-depth analysis of Graph Neural Networks (GNNs), with a specific focus on Graph Attention Networks (GATs) tailored for the complex task of link prediction. This module is engineered not merely to build predictive models, but to facilitate a deep scientific inquiry into the capabilities of GNNs to understand and reconstruct the semantic and structural topology of graphs. The entire workflow, from model definition to results analysis, is designed to adhere to the highest standards of academic research, emphasizing reproducibility, detailed metric tracking, and advanced model evaluation.

## 2. Core Research Objective

The central scientific objective of this module is to develop and critically evaluate GNN architectures that can effectively infer latent relational structures within semantic graphs. The problem is approached from two distinct but complementary perspectives:

1.  **Structural Link Prediction**: Assessing the model's capacity to predict missing edges based on the explicit topological information of a given graph.
2.  **Semantic Relationship Reconstruction**: Evaluating the model's ability to infer connections based solely on the semantic content of the nodes, independent of a pre-existing graph structure.

## 3. Architectural Implementations

The module features an iterative development of GAT architectures, allowing for a comparative analysis of different design choices.

### 3.1. `GatModelV1`

This model serves as the foundational baseline architecture. Its core is the `DeepGATBlockV1`, a deep GAT block inspired by the Transformer encoder.

-   **Architecture**: Comprises a stack of layers, each containing a multi-head `GATConv` sub-module and a feed-forward network.
-   **Components**: Utilizes standard `torch.nn.ReLU` as the activation function and `torch.nn.LayerNorm` for normalization.
-   **Purpose**: Establishes a robust performance benchmark for the link prediction task.

### 3.2. `GatModelV2`

This model represents an architectural evolution, incorporating more advanced components to push the boundaries of performance and training efficiency.

-   **Architecture**: Follows the same deep block structure but is parameterized with more sophisticated functions via the `DeepGATBlockV2` module.
-   **Key Enhancements**:
    -   **`SwiGLU` Activation**: Replaces ReLU with the Swish-Gated Linear Unit (`swiglu_func_v2.py`), a more advanced activation function known to improve performance in Transformer-based models.
    -   **`RMSNorm`**: Employs Root Mean Square Normalization instead of LayerNorm, which can offer improved computational efficiency and performance.
-   **Purpose**: To investigate the impact of state-of-the-art neural network components in the context of graph attention models.

## 4. Methodological Workflow

The module encapsulates an end-to-end pipeline for model development and evaluation, orchestrated primarily by `trainer.py`.

1.  **Data Ingestion and Preprocessing**: The framework loads graph datasets (from the `dataset_builder_wiki` module) and splits them into training and testing sets.
2.  **Model Training**: The training loop performs forward and backward passes, optimizing the model parameters using the `BCEWithLogitsLoss` function and the `AdamW` optimizer. Negative sampling is employed during loss calculation for efficiency.
3.  **Dual-Metric Monitoring**: Throughout training, the model's performance is continuously monitored using two distinct AUC metrics (detailed below).
4.  **Model Evaluation**: After training, the model's generalization capability is assessed on a held-out test set using the same comprehensive set of metrics.

## 5. Evaluation Metrics and Analysis

A cornerstone of this module is its sophisticated evaluation strategy, which moves beyond standard metrics.

### 5.1. Link Prediction AUC (Area Under the ROC Curve)

This is the standard metric for the link prediction task. It measures the model's ability to correctly discriminate between true (existing) and false (non-existing) edges, providing a quantitative assessment of its performance on the known graph structure.

### 5.2. Agnostic AUC (A-AUC)

This is a novel, custom-defined metric designed to evaluate the model's intrinsic understanding of the semantic content of the nodes, independent of the graph's explicit topology.

-   **Methodology**: The A-AUC is computed by using the cosine similarity of the nodes' initial feature embeddings as a proxy for a "semantic ground truth". The model's predicted edge probabilities (from the `encode` method) are then evaluated against this semantic similarity matrix.
-   **Significance**: A high A-AUC score indicates that the model has learned to recognize semantic relationships between nodes, even if they are not directly connected in the original graph. This is a powerful measure of the model's ability to generalize and perform reasoning over the node content.

### 5.3. Qualitative Analysis via Graph Reconstruction (`llm_graph_tester.py`)

To further probe the model's capabilities, a unique evaluation scenario is implemented. A trained GNN is tasked with reconstructing a graph's entire edge structure given only a set of nodes generated by a Large Language Model (LLM). This serves as a practical test of the model's ability to perform knowledge graph completion from scratch, a task that closely mimics real-world applications where only entities are known and relationships must be inferred.

## 6. Experiment Management and Reproducibility

The module is equipped with a robust system for experiment tracking and analysis.

-   **Metric Exportation (`metric_exporter.py`)**: All training and testing metrics, including epoch-wise performance and configuration parameters, are systematically serialized to JSON files and stored in the `metric outputs/` directory.
-   **Visualization (`plot_utils.py`)**: The framework automatically generates high-resolution plots for training dynamics (loss, AUC, A-AUC vs. steps) and final test results. These visualizations are saved in the `plots/` directory, enabling immediate analysis of model behavior.
-   **Model Checkpointing (`model_dumps/`)**: Trained model weights are periodically and finally saved to the `model_dumps/` directory, ensuring that valuable experimental artifacts are preserved for future use and analysis.

## 7. Module Components

-   `gnn_model_v1/`: Contains the implementation of the baseline `GatModelV1`.
-   `gnn_model_v2/`: Contains the implementation of the advanced `GatModelV2`.
-   `trainer.py`: The main orchestration script for training and evaluating the models.
-   `llm_graph_tester.py`: A script for evaluating the model's ability to reconstruct graphs from LLM-generated nodes.
-   `metric_exporter.py`: Utilities for exporting performance metrics to JSON.
-   `plot_utils.py`: Utilities for generating plots of experimental results.
-   `model_dumps/`: Directory for storing trained model checkpoints.
-   `metric outputs/`: Directory for storing JSON files with detailed experiment metrics.
-   `plots/`: Directory for storing generated plots.