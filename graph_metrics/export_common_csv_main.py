import networkx as nx
import numpy as np
import sys, os
import logging
import community as co
import json
from tqdm import tqdm
import csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_logger.logger_config import get_logger

log: logging.Logger = get_logger(name=__name__)

JSON_DATASET_DIR = "/home/bruno/Documents/GitHub/social-media-nlp/dataset_builder_wiki/final_dataset/json"
JOSN_OUT_DIR = "/home/bruno/Documents/GitHub/social-media-nlp/graph_metrics/json_common_nodes"
CSV_OUT_FILE = "/home/bruno/Documents/GitHub/social-media-nlp/graph_metrics/csv_common_nodes/csv_common_nodes.csv"


def __json_to_graph(json_graph: dict) -> nx.Graph:
    
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
    common_nodes = 0
    for cur_node in curr_graph.nodes:    
        for other_graph in remaining_graphs:
            
            if curr_graph == other_graph:
                # log.info("Found same graph, skipping")
                continue
            
            if cur_node in other_graph.nodes:
                common_nodes += 1
                break
                
    log.info(f"Found {common_nodes} common nodes")
    return common_nodes

def __compute_json_common_nodes():
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
    __compute_json_common_nodes()
    __export_common_labels_csv()