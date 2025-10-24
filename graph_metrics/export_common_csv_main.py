"""
This script is focused on analyzing the interconnectivity of a graph dataset by identifying
and quantifying the number of common nodes shared between different graphs.

The script operates in two main phases:
1.  **Common Node Computation and JSON Export**:
    - It begins by loading all graphs from a specified JSON dataset directory into memory
      as NetworkX graph objects.
    - For each graph in the dataset, it computes the number of its nodes that are also present
      in at least one other graph within the same dataset.
    - The result of this computation, which is the count of common nodes for each graph,
      is then exported to a separate JSON file.

2.  **CSV Aggregation**:
    - After the common node counts have been calculated and saved for all graphs, the script
      proceeds to read all the generated JSON files.
    - It then aggregates this information into a single, consolidated CSV file.
      This CSV provides a tabular summary of the common node counts for each graph in the dataset,
      facilitating a high-level analysis of dataset-wide node overlap.

This analysis is valuable for understanding the diversity and redundancy of the information
contained within the graph dataset, which can inform decisions related to dataset curation
and model training strategies.
"""

import sys, os
import logging
import json
from tqdm import tqdm
import csv
import networkx as nx

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_logger.logger_config import get_logger

log: logging.Logger = get_logger(name=__name__)

# Directory containing the JSON graph dataset.
JSON_DATASET_DIR = "/home/bruno/Documents/GitHub/social-media-nlp/dataset_builder_wiki/final_dataset/json"
# Directory to store the intermediate JSON files with common node counts.
JOSN_OUT_DIR = "/home/bruno/Documents/GitHub/social-media-nlp/graph_metrics/json_common_nodes"
# The final CSV file to which the aggregated common node counts will be written.
CSV_OUT_FILE = "/home/bruno/Documents/GitHub/social-media-nlp/graph_metrics/csv_common_nodes/csv_common_nodes.csv"


def __json_to_graph(json_graph: dict) -> nx.Graph:
    """
    Constructs a NetworkX graph from a JSON object representing a graph.

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

def __compue_common_nodes(curr_graph: nx.Graph, remaining_graphs: list[nx.Graph]) -> int:
    """
    Computes the number of nodes in the current graph that are also present in other graphs.

    Args:
        curr_graph (nx.Graph): The graph for which to count common nodes.
        remaining_graphs (list[nx.Graph]): A list of other graphs in the dataset to compare against.

    Returns:
        int: The total count of common nodes.
    """
    common_nodes = 0
    for cur_node in curr_graph.nodes:    
        for other_graph in remaining_graphs:
            
            if curr_graph == other_graph:
                continue
            
            if cur_node in other_graph.nodes:
                common_nodes += 1
                break
                
    log.info(f"Found {common_nodes} common nodes")
    return common_nodes

def __compute_json_common_nodes():
    """
    Orchestrates the computation of common nodes for all graphs in the dataset and exports the results to JSON files.
    """
    graphs_map: dict[str, nx.Graph] = {}
    for filename in tqdm(os.listdir(JSON_DATASET_DIR), desc="JSON graphs loop"):
            if filename.endswith(".json"):
                log.info(f"Loading NetworkX graphs, processing {filename}")
                json_graph = json.load(open(os.path.join(JSON_DATASET_DIR, filename)))
                graph = __json_to_graph(json_graph)
                graphs_map[filename] = graph
    log.info(f"Loaded {len(graphs_map)} graphs")
    
    for key, graph in tqdm(graphs_map.items(), total=len(graphs_map), desc="Graphs loop"):
        log.info(f"Computing common nodes count for {key}")
        common_nodes_count = __compue_common_nodes(graph, list(graphs_map.values()))
        log.info(f"Common nodes count for {key}: {common_nodes_count}")
        json_dict = {
            "filename": key,
            "common_nodes_count": common_nodes_count
        }
        json_str = json.dumps(json_dict, indent=4)
        json_out_filename = "common_nodes_count_" + key
        with open(os.path.join(JOSN_OUT_DIR, json_out_filename), "w") as file:
            file.write(json_str)
            file.close()
        log.info(f"Dict {json_dict} exported to {json_out_filename}")
        
def __export_common_labels_csv(
    json_common_mnetrics_file: str = JOSN_OUT_DIR, 
    csv_out_file: str  = CSV_OUT_FILE
) -> None:
    """
    Aggregates common node count data from JSON files into a single CSV file.

    Args:
        json_common_mnetrics_file (str, optional): The directory containing the JSON files with common node counts.
                                                    Defaults to JOSN_OUT_DIR.
        csv_out_file (str, optional): The path to the output CSV file. Defaults to CSV_OUT_FILE.
    """
    
    csv_dicts_list: list[dict[str, int]] = []
    for filename in tqdm(os.listdir(json_common_mnetrics_file), desc="JSON metrics loop"):
        if filename.endswith(".json"):
            log.info(f"Reading {filename}")
            json_dict = json.load(open(os.path.join(json_common_mnetrics_file, filename)))
            csv_dicts_list.append(json_dict)
    
    with open(csv_out_file, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["filename", "common_nodes_count"])
        writer.writeheader()
        writer.writerows(csv_dicts_list)
        csv_file.flush()
        csv_file.close()


if __name__ == "__main__":
    # This script block executes the two main phases of the common node analysis:
    # 1. Compute common nodes and export to individual JSON files.
    # 2. Aggregate the JSON data into a single CSV file.
    __compute_json_common_nodes()
    __export_common_labels_csv()
