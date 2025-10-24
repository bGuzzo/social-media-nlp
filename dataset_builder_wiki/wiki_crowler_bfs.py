"""
This script is the heart of the data acquisition phase of the project. It implements a web
crawler that systematically explores the hyperlink structure of Wikipedia to construct a
graph dataset. The crawler employs a Breadth-First Search (BFS) algorithm, which is
particularly well-suited for this task as it prioritizes the exploration of the immediate
neighborhood of a root article, leading to a more coherent and thematically focused graph.

The script's main functionalities are:

1.  **Breadth-First Search (BFS) Crawler (`__crawl`)**:
    - This is the core of the script, a recursive function that implements the BFS traversal.
    - Starting from a root article, it explores the links on the page, adding new articles
      to a queue to be visited. This process continues until a specified maximum number of
      nodes is reached or the queue is exhausted.
    - The BFS approach ensures that the resulting graph is a connected component centered
      around the root article, which is a desirable property for the subsequent graph
      analysis and modeling tasks.

2.  **Graph Construction (`extract_graph`)**:
    - This function orchestrates the entire graph extraction process for a single root article.
    - It initializes the graph, the queue, and the set of visited nodes, and then kicks off
      the BFS crawling process.
    - It also includes important error handling and consistency checks, such as verifying
      that the resulting graph is connected.

3.  **Wikipedia Page Validation (`__check_title`)**:
    - To ensure the quality of the dataset, this function validates each potential node (i.e.,
      article title) to confirm that it corresponds to a valid, non-disambiguation Wikipedia page.

4.  **JSON Serialization (`json_dump`)**:
    - Once a graph is constructed, this function serializes it into a structured JSON format.
    - This format is human-readable and can be easily parsed by other components of the data
      pipeline.

This script is a prime example of how web crawling techniques can be leveraged to create
rich, structured datasets from the vast and semi-structured information available on the web.
"""

import wikipedia
from wikipedia.wikipedia import WikipediaPage
import networkx as nx
import logging
import matplotlib.pyplot as plt
import json
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_logger.logger_config import get_logger

log: logging.Logger = get_logger(name=__name__)

# Default maximum number of nodes for the graph
DEF_MAX_NODES = 20000

def json_dump(graph: nx.Graph) -> str:
    """
    Serializes a NetworkX graph into a JSON string.

    Args:
        graph (nx.Graph): The graph to be serialized.

    Returns:
        str: The JSON representation of the graph.
    """
    node_ids_counter = 0
    node_id_map = {}
    
    nodes_list, edges_list = [], []
    
    for node in graph.nodes:
        node_id_map[str(node)] = node_ids_counter
        nodes_list.append({"id": node_ids_counter, "label": node})
        node_ids_counter += 1
    
    for edge in graph.edges:
        edges_list.append({"source": node_id_map[str(edge[0])], "target": node_id_map[str(edge[1])]})
    
    return json.dumps({"nodes": nodes_list, "edges": edges_list}, indent=4)

def __add_edge(graph: nx.Graph, page_src: str, page_dst: str) -> None:
    """
    Adds an edge to the graph with consistency checks.

    Args:
        graph (nx.Graph): The graph to which the edge will be added.
        page_src (str): The source node of the edge.
        page_dst (str): The destination node of the edge.
    """
    if page_src == page_dst:
        log.warning(f"Avoided adding self-node link ({page_src} -> {page_dst})")
        return
    
    if graph.has_edge(page_src, page_dst) or graph.has_edge(page_dst, page_src):
        log.info(f"Edge ({page_src} -> {page_dst}) already present, not adding it")
        return
    
    graph.add_edge(page_src, page_dst)

def __check_title(page_title: str) -> WikipediaPage:
    """
    Checks if a page title corresponds to a valid Wikipedia page.

    Args:
        page_title (str): The title of the page to check.

    Returns:
        WikipediaPage: The WikipediaPage object if the page is valid.

    Raises:
        wikipedia.exceptions.PageError: If the page is not found.
        wikipedia.exceptions.DisambiguationError: If the page is a disambiguation page.
    """
    try:
        return wikipedia.page(page_title)
    except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError) as e:
        log.error(f"Error processing page '{page_title}': {e}")
        raise

def __crawl(graph: nx.Graph, queue: list, visited: set, max_nodes: int):
    """
    The recursive core of the Breadth-First Search (BFS) crawler.

    Args:
        graph (nx.Graph): The graph being constructed.
        queue (list): The queue of articles to visit.
        visited (set): A set of visited article titles.
        max_nodes (int): The maximum number of nodes to include in the graph.
    """
    if not queue or len(graph.nodes) >= max_nodes:
        log.info(f"Stopping crawl. Queue empty: {not queue}, Node limit reached: {len(graph.nodes) >= max_nodes}")
        return

    page_title = queue.pop(0)
    
    try:
        page = __check_title(page_title)
        for link_title in page.links:
            if len(graph.nodes) >= max_nodes:
                break
            
            if link_title not in visited:
                visited.add(link_title)
                queue.append(link_title)
                graph.add_node(link_title)
            __add_edge(graph, page_title, link_title)

    except Exception:
        log.error(f"Skipping page '{page_title}' due to an error.")

    __crawl(graph, queue, visited, max_nodes)

def extract_graph(root_article_title: str, max_nodes: int = DEF_MAX_NODES) -> nx.Graph:
    """
    Orchestrates the extraction of a single graph from Wikipedia using BFS.

    Args:
        root_article_title (str): The title of the root article for the graph.
        max_nodes (int, optional): The maximum number of nodes in the graph. Defaults to DEF_MAX_NODES.

    Returns:
        nx.Graph: The extracted NetworkX graph.
    """
    __check_title(root_article_title)
    
    graph = nx.Graph()
    graph.add_node(root_article_title)
    visited = {root_article_title}
    queue = [root_article_title]
    
    __crawl(graph, queue, visited, max_nodes)
    
    if not nx.is_connected(graph):
        raise Exception("Graph is not connected")
    
    log.info(f"Graph created with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    return graph

if __name__ == "__main__":
    # This block demonstrates the usage of the extract_graph function and visualizes the result.
    graph = extract_graph(root_article_title="Sustainability")
    
    plt.figure(figsize=(12, 12))
    nx.draw(graph, with_labels=False, node_size=30, font_size=8, node_color="skyblue")
    plt.title("Wikipedia graph from root word 'Sustainability'")
    plt.show()
