import logging
from graph_builder import create_graph, json_dump

# Import parent script
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_logger.logger_config import get_logger

log: logging.Logger = get_logger(name=__name__)

FOLDER_PATH = (
    "/home/bruno/Documents/GitHub/social-media-nlp/dataset_builder/json_graph_dataset"
)

NUM_NODES = 500
DEPTH_LIMIT = 5

SDG_WORDS = [
    "poverty",
    "hunger",
    "health",
    "education",
    "gender",
    "water",
    "energy",
    "economy",
    "innovation",
    "inequality",
    "cities",
    "consumption",
    "climate",
    "oceans",
    "biodiversity",
    "peace",
    "justice",
    "partnership",
    "sustainability",
    "development",
]

if __name__ == "__main__":
    for word in SDG_WORDS:
        log.info(f"Processing word {word}")
        graph = create_graph(
            root_word=word, max_depth=DEPTH_LIMIT, max_nodes=NUM_NODES
        )
        json_graph = json_dump(graph)
        with open(f"{FOLDER_PATH}/{word}.json", "w") as file:
            file.write(json_graph)
            file.close()
            log.info(f"File {word}.json created")
