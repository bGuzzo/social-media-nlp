import wikipedia
import networkx as nx
import logging
import matplotlib.pyplot as plt
import json

# Import parent script
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_logger.logger_config import get_logger

log: logging.Logger = get_logger(name=__name__)

# Default  graph size
DEF_MAX_NODES = 20

# Default depth limit
DEF_MAX_DEPTH = 1


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


def crawl(
    graph: nx.Graph, 
    visited: set, 
    page_title: str, 
    max_nodes: int, 
    max_depth: int, 
    current_depth: int
):
    if len(graph.nodes) > max_nodes:
        log.info(f"Reached graph size limit {len(graph.nodes)}")
        return
    
    if current_depth > max_depth:
        log.info(f"Reached depth limit {current_depth}")
        return
    
    if page_title in visited:
        log.info(f"Already visited {page_title}")
        return

    # Check if the page exists
    try:
        page = wikipedia.page(page_title)
    except wikipedia.exceptions.PageError as e:
        log.error(f"Error: Page '{page_title}' not found.", e)
        return
    except wikipedia.exceptions.DisambiguationError as e:
        log.error(f"Disambiguation Error for '{page_title}': {e.options}", e)
        return
    
    visited.add(page.title)
    graph.add_node(page.title)

    for link_title in page.links:
        
        # Check if the graph cloud be expanded
        if len(graph.nodes) > max_nodes:
            log.info(f"Reached graph size limit {len(graph.nodes)}")
            return
        
        graph.add_edge(page.title, link_title)
        crawl(
            graph=graph, 
            visited=visited, 
            page_title=link_title, 
            max_nodes=max_nodes, 
            max_depth=max_depth, 
            current_depth=current_depth + 1
        )


def extract_graph(
    root_article_title: str, 
    max_nodes: int = DEF_MAX_NODES, 
    max_depth: int = DEF_MAX_DEPTH
) -> nx.Graph:
    
    graph = nx.Graph()
    visited = set()

    crawl(graph, visited, root_article_title, max_nodes, max_depth, 0)
    return graph

# Test only
if __name__ == "__main__":
    # SGD as root article
    graph = extract_graph(root_article_title="Sustainable Development Goals")
    
    # Draw the graph (adjust layout as needed)
    plt.figure(figsize=(12, 12))
    nx.draw(graph, with_labels=True, node_size=300, font_size=8, node_color="skyblue")
    plt.title("Wikipedia Article Link Graph")
    plt.show()