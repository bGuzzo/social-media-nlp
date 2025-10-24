"""
This script is dedicated to the analysis of graph structures stored in JSON format.
It provides functionalities to compute a comprehensive set of graph-theoretic metrics,
including degree, clustering coefficient, path lengths, and modularity. The script is designed
to process a directory of JSON graph files, calculate metrics for each graph, and export
the results into separate JSON files for further analysis.

The primary functionalities of this script are:
1.  **Graph Loading**: It reads graph data from JSON files, where nodes and edges are defined.
2.  **Metric Computation**: For each graph, it calculates:
    - Number of nodes and edges.
    - Average node degree.
    - Average clustering coefficient.
    - Average shortest path length (for connected components).
    - Modularity using the Louvain method for community detection.
    - Graph density.
    - Connectivity of the graph.
3.  **Metric Export**: The computed metrics are saved in a structured JSON format, making them
    readily available for subsequent data analysis and visualization tasks.

This script is an essential tool for the preliminary analysis of the graph dataset, providing
quantitative insights into the topological properties of the graphs.
"""

import networkx as nx
import numpy as np
import sys, os
import logging
import community as co
import json
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_logger.logger_config import get_logger

log: logging.Logger = get_logger(name=__name__)

# Directory containing the JSON graph dataset.
JSON_DATASET_DIR = "/home/bruno/Documents/GitHub/social-media-nlp/dataset_builder_wiki/final_dataset/json"
# Directory where the computed graph metrics will be exported.
JSON_METRIC_EXPORT_DIR = "/home/bruno/Documents/GitHub/social-media-nlp/graph_metrics/json_metrics_density"

def export_metrics(graph: nx.Graph) -> dict[str, float]:
    """
    Computes a set of standard graph metrics for a given NetworkX graph.

    This function calculates the number of edges and nodes, average degree, average clustering coefficient,
    average shortest path length, and modularity. It includes handling for special cases such as
    empty or disconnected graphs to ensure robust computation.

    Args:
        graph (nx.Graph): The input NetworkX graph for which to compute metrics.

    Returns:
        dict[str, float]: A dictionary containing the computed graph metrics.
    """
    log.info("Computing graph metrics")

    num_edges = graph.number_of_edges()
    num_nodes = graph.number_of_nodes()

    # Handle empty graph case to avoid division by zero errors
    log.info("Computing average degree")
    if num_nodes == 0:  
        avg_degree = 0
    else:
        degrees = dict(graph.degree())  # Get node degree map
        avg_degree = sum(degrees.values()) / num_nodes # Compute average

    log.info("Computing average shortest path length")
    if num_nodes <=1:
        avg_path_length = 0
    else:
        try:
            avg_path_length = nx.average_shortest_path_length(graph)
        except nx.NetworkXError: 
            # Throws exception when graph is not connected
            # Use infinity as average path lenght
            avg_path_length = float('inf')

    log.info("Computing average clustering")
    if num_nodes <= 2:
        avg_clustering = 0
    else:
        avg_clustering = nx.average_clustering(graph)

    log.info("Computing modularity")
    if num_nodes <= 1 or num_edges == 0:
        modularity = 0
    else:
        try:           
            # Use Louvain method for community detection
            partition = co.best_partition(graph)  # Compute communities
            modularity = co.modularity(partition, graph)
        except Exception as e:
            log.error(f"Error calculating modularity: {e}. Modularity set to None")
            modularity = None
    
    is_connected = nx.is_connected(graph)
    results = {
        "num_edges": num_edges,
        "num_nodes": num_nodes,
        "average_degree": avg_degree,
        "average_clustering": avg_clustering,
        "average_path_length": avg_path_length,
        "modularity": modularity,
        "is_connected": is_connected,
    }

    log.info(f"Graph metrics computed: {results}")
    return results

def __export_density(graph: nx.Graph) -> dict[str, float]:
    """
    Computes the density of a given NetworkX graph.

    Args:
        graph (nx.Graph): The input NetworkX graph.

    Returns:
        dict[str, float]: A dictionary containing the graph density.
    """
    densitiy = nx.density(graph)
    return {
        "density": densitiy
    }


def __json_to_graph(json_graph: dict) -> nx.Graph:
    """
    Constructs a NetworkX graph from a JSON object representing a graph.

    The JSON object should contain 'nodes' and 'edges' keys. Each node is expected to have an 'id' and 'label',
    and each edge is expected to have a 'source' and 'target' referencing node IDs.

    Args:
        json_graph (dict): A dictionary representing the graph in JSON format.

    Returns:
        nx.Graph: The constructed NetworkX graph.
    """
    
    graph = nx.Graph()
    node_idx_map = {}

    for node in json_graph["nodes"]:
        node_idx_map[node["id"]] = node["label"]
        graph.add_node(node["label"])
    
    for edge in json_graph["edges"]:
        source_node_id = edge["source"]
        target_node_id = edge["target"]
        graph.add_edge(node_idx_map[source_node_id], node_idx_map[target_node_id])

    log.info(f"Build NetworX graph from JSON with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    return graph

def __export_metrics_json(filename: str) -> None:
    """
    Loads a graph from a JSON file, computes its metrics, and exports them to another JSON file.

    Args:
        filename (str): The name of the JSON graph file to process.
    """
    log.info(f"Processing {filename}")
    
    filepath = os.path.join(JSON_DATASET_DIR, filename)
    json_graph = json.load(open(filepath))
    graph = __json_to_graph(json_graph)
    metrics = export_metrics(graph)
    log.info(f"Metrics for {filename}: {metrics}")
    
    graph_name = filename.split(".")[0]
    metric_file_name = f"{graph_name}_metrics.json"
    metrics["filename"] = filename
    metric_json_path = os.path.join(JSON_METRIC_EXPORT_DIR, metric_file_name)
    str_metrics = json.dumps(metrics, indent=4)
    with open(metric_json_path, "w") as file:    
        file.write(str_metrics)
        file.close()
    
    log.info(f"Metrics exported to {metric_json_path}")


def __export_density_json(filename: str) -> None:
    """
    Loads a graph from a JSON file, computes its density, and exports it to a JSON file.

    Args:
        filename (str): The name of the JSON graph file to process.
    """
    log.info(f"Processing density for {filename}")
    
    filepath = os.path.join(JSON_DATASET_DIR, filename)
    json_graph = json.load(open(filepath))
    graph = __json_to_graph(json_graph)
    metrics = __export_density(graph)
    log.info(f"Metrics for {filename}: {metrics}")
    
    graph_name = filename.split(".")[0]
    metric_file_name = f"{graph_name}_density.json"
    metrics["filename"] = filename
    metric_json_path = os.path.join(JSON_METRIC_EXPORT_DIR, metric_file_name)
    str_metrics = json.dumps(metrics, indent=4)
    with open(metric_json_path, "w") as file:    
        file.write(str_metrics)
        file.close()
    
    log.info(f"Density exported to {metric_json_path}")
    

def main_singlethread():
    """
    Main function to iterate through all JSON graph files in the dataset directory,
    compute their metrics, and export them. This version runs in a single thread.
    """
    for filename in tqdm(os.listdir(JSON_DATASET_DIR), desc="JSON graphs loop"):
            if filename.endswith(".json"):
                __export_density_json(filename)
                # __export_metrics_json(filename)

if __name__ == "__main__":
    # This script block executes the main function to process the graph dataset.
    # It is configured to run in a single thread.
    main_singlethread()
