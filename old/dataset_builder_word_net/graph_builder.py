import logging
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset
import networkx as nx
import matplotlib.pyplot as plt
import json

# Import parent script
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_logger.logger_config import get_logger

log: logging.Logger = get_logger(name=__name__)

# Download WordNet data (if not already downloaded)
nltk.download("wordnet")

# Default  graph size
DEF_MAX_NODES = 500

# Default depth limit
DEF_MAX_DEPTH = 5


def json_dump(graph: nx.Graph):
    node_ids_counter = 0
    node_id_map = {str : int}
    
    nodes_list = []
    edges_list = []
    
    for node in graph.nodes:
        node_id_map[str(node)] = node_ids_counter
        nodes_list.append({
            "id": node_ids_counter,
            "label": node
        })
        node_ids_counter += 1
    
    for edge in graph.edges:
        edges_list.append({
            "source": node_id_map[str(edge[0])],
            "target": node_id_map[str(edge[1])]
        })
    
    json_graph = {
        "nodes": nodes_list,
        "edges": edges_list
    }
    
    return json.dumps(json_graph)


def create_graph(
    root_word: str, max_nodes: int = DEF_MAX_NODES, max_depth: int = DEF_MAX_DEPTH
) -> nx.Graph:
    graph = nx.Graph()
    visited = set()

    # for word in start_words:
    synsets = wn.synsets(root_word)
    for synset in synsets:
        log.info(f"First level synset: {synset.name()}")
        add_related_nodes(graph=graph, synset=synset, visited=visited, max_nodes=max_nodes, max_depth=max_depth, depth=0)
        if len(graph.nodes) >= max_nodes:
            break
    
    log.info(f"Graph created with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    return graph


def add_related_nodes(
    graph: nx.Graph,
    synset: Synset,
    visited: set,
    max_nodes: int,
    max_depth: int,
    depth: int,
):
    if synset in visited:
        log.info(f"Already visited {synset.name()}")
        return

    if depth > max_depth:
        log.info(f"Reached depth limit {depth}")
        return

    if len(graph.nodes) > max_nodes:
        log.info(f"Reached graph size limit {len(graph.nodes)}")
        return

    visited.add(synset)
    label_name_synset = str(synset.name()).split(".")[0].replace("_", " ")
    graph.add_node(label_name_synset, label=label_name_synset)

    # Explore different semantic relationships
    for relation in [
        synset.hyponyms,
        synset.hypernyms,
        synset.member_holonyms,
        synset.part_holonyms,
        synset.substance_holonyms,
        synset.member_meronyms,
        synset.part_meronyms,
        synset.substance_meronyms,
        synset.entailments,
        synset.causes,
        synset.also_sees,
        synset.similar_tos,
        synset.attributes,
    ]:
        for related_synset in relation():
            
            if len(graph.nodes) > max_nodes:
                log.info(f"Reached graph size limit {len(graph.nodes)}")
                return
            
            label_name_related_synset = (
                str(related_synset.name()).split(".")[0].replace("_", " ")
            )
            graph.add_edge(label_name_synset, label_name_related_synset)
            add_related_nodes(graph, related_synset, visited, max_nodes, max_depth, depth + 1)


# Test only
if __name__ == "__main__":
    # Generate the graph
    graph = create_graph(root_word="poverty")

    # log.info(f"JSON Graph \n{json_dump(graph)}")

    # Visualize the graph (optional)
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(graph)
    nx.draw(
        graph, pos, with_labels=True, node_size=1500, node_color="skyblue", font_size=8
    )
    plt.title("Semantic Graph")
    plt.show()
