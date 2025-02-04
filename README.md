# Social Media NLP Project

This repository contains the project for the 'Social Media' module of the "IR-NLP & Social Media" course at DIMES. The project focuses on Natural Language Processing (NLP) techniques applied to social media data, specifically Wikipedia articles.

## Project Structure

The project is organized into the following directories:

* **custom_logger:** Contains a custom logger configuration for consistent logging throughout the project.

* **dataset_builder_wiki:** Contains scripts for building the dataset from Wikipedia articles.
    * `wiki_crowler_bfs.py`: Crawls Wikipedia pages using Breadth-First Search (BFS) to collect article content.
    * `wiki_dataset_builder.py`: Processes the crawled articles, extracts relevant information, and builds the dataset.
    * `wiki_graph_json_to_tensor.py`: Converts the dataset from JSON format to PyTorch tensors.
    * `create_dataset_main.py`: Main script to orchestrate the dataset creation process.
    * `wikipedia_articles.txt`: List of seed Wikipedia articles for crawling.
    * `wiki_torch_loader.py`: Creates a PyTorch DataLoader for the generated dataset.
    * `final_dataset/json`: Stores the final dataset in JSON format.

* **gnn_networks:** Contains code for training and evaluating Graph Neural Networks (GNNs).
    * `trainer.py`: Implements the training loop and evaluation logic for GNN models.
    * `plot_utils.py`: Provides utility functions for plotting training metrics and visualizations.
    * `metric_exporter.py`: Exports training and evaluation metrics to JSON files.
    * `llm_graph_tester.py`: Tests the trained GNN models using a Large Language Model (LLM).
    * `gnn_model_v1`: Contains the implementation of the first version of the GNN model.
        * `gat_net_module_v1.py`: Defines the GAT network architecture for version 1.
        * `gat_block_v1.py`: Implements the GAT block for version 1.
    * `gnn_model_v2`: Contains the implementation of the second version of the GNN model.
        * `gat_net_module_v2.py`: Defines the GAT network architecture for version 2.
        * `gat_block_v2.py`: Implements the GAT block for version 2.
        * `swiglu_func_v2.py`: Implements the SwiGLU activation function for version 2.
    * `metric outputs`: Stores the training and evaluation metrics in JSON format.
    * `model_dumps`: Stores the trained GNN models in PyTorch format.
    * `plots`: Stores plots generated during training and evaluation.

* **llm_dataset:** Contains scripts for generating additional graph nodes using an LLM.
    * `generate_llm_graph_nodes.py`: Generates new nodes for the graph based on LLM outputs.
* **latex:** Contains the LaTeX source code for the project report.
    * `assignment`: Contains the LaTeX files, bibliography, and compiled PDF for the assignment report.


## Dependencies

The project requires the following Python libraries:

* torch
* transformers
* numpy
* pandas
* networkx
* matplotlib
* seaborn
* scikit-learn
* tqdm
* wandb

You can install the dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Running the Project

To run the project, you can follow these steps:

1. **Build the dataset:**
   ```bash
   python dataset_builder_wiki/create_dataset_main.py
   ```
2. **Train and test a GNN model:**
   ```bash
   python gnn_networks/train_gnn_main.py
   ```
3. **Evaluate the trained model:**
   ```bash
   python gnn_networks/llm_graph_tester.py
   ```

You can modify the configuration parameters in the respective scripts to customize the dataset creation, model training, and evaluation process.

## Project Report
The project report is available in the `latex/assignment` directory as LaTeX and PDF compiled format.