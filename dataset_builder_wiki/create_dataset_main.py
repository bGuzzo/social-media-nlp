import sys
import os
from wiki_dataset_builder import create_dataset
from wiki_graph_json_to_tensor import convert_jsons_to_tensors
import logging

# Import parent script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_logger.logger_config import get_logger

log: logging.Logger = get_logger(name=__name__)

MAX_NODES = 5000
JSON_FOLDER_PATH = "/home/bruno/Documents/GitHub/social-media-nlp/dataset_builder_wiki/final_dataset/json"
TESNOR_FOLDER_PATH = "/home/bruno/Documents/GitHub/social-media-nlp/dataset_builder_wiki/final_dataset/tensor"
HF_PHARESE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# Genearted with Gemini 1.5 Pro
# Prompt: Generate a JSON list of 50 Wikipedia pages title about {topic}.
WIKI_ARTICLES = [
    
    # SDG 1: Poverty
    "Poverty",
    "Poverty threshold",
    "Poverty in the United States",
    "Poverty in India",
    "Poverty in Africa",
    "Poverty in China",
    "Poverty in the United Kingdom",
    "Poverty in Canada",
    "Poverty in Australia",
    "Poverty in Brazil",
    "Poverty in Mexico",
    "Poverty in South Africa",
    "Poverty in Nigeria",
    "Poverty in Pakistan",
    "Poverty in Bangladesh",
    "Poverty in the Philippines",
    "Poverty in Egypt",
    "Poverty in Ethiopia",
    "Poverty in Afghanistan",
    "Poverty in Yemen",
    "Poverty in the Democratic Republic of the Congo",
    "Poverty in Tanzania",
    "Poverty in Kenya",
    "Poverty in Uganda",
    "Poverty in Sudan",
    "Poverty in Mozambique",
    "Poverty in Madagascar",
    "Poverty in Angola",
    "Poverty in Malawi",
    "Poverty in Zambia",
    "Poverty in Zimbabwe",
    "Poverty in Haiti",
    "Poverty in Guatemala",
    "Poverty in Honduras",
    "Poverty in El Salvador",
    "Poverty in Nicaragua",
    "Poverty in Costa Rica",
    "Poverty in Panama",
    "Poverty in the Dominican Republic",
    "Poverty in Puerto Rico",
    "Poverty in Jamaica",
    "Poverty in Trinidad and Tobago",
    "Poverty in Guyana",
    "Poverty in Suriname",
    "Poverty in Belize",
    "Poverty in the Bahamas",
    "Poverty in Barbados",
    "Poverty in Saint Lucia",
    "Poverty in Grenada",
    "Poverty in Saint Vincent and the Grenadines",
    "Poverty in Antigua and Barbuda"
]

def main():
    log.info("Start creating wiki dataset")
    create_dataset(
        wiki_articles=WIKI_ARTICLES,
        folder_path=JSON_FOLDER_PATH,
        max_nodes=MAX_NODES
    )
    log.info("Wiki dataset created successfully")
    
    log.info("Start converting wiki dataset to tensor")
    convert_jsons_to_tensors(
        json_folder=JSON_FOLDER_PATH,
        tensor_folder=TESNOR_FOLDER_PATH
    )
    log.info("Wiki dataset converted to tensor successfully")

if __name__ == "__main__":
    main()
