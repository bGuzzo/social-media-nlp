import sys
import os
from wiki_dataset_builder import create_dataset
from wiki_graph_json_to_tensor import convert_jsons_to_tensors
import logging

# Import parent script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_logger.logger_config import get_logger

log: logging.Logger = get_logger(name=__name__)

MAX_NODES = 20000
JSON_FOLDER_PATH = "/home/bruno/Documents/GitHub/social-media-nlp/dataset_builder_wiki/final_dataset/json"
TESNOR_FOLDER_PATH = "/home/bruno/Documents/GitHub/social-media-nlp/dataset_builder_wiki/final_dataset/tensor"
HF_PHARESE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# Genearted with Gemini 1.5 Pro
# Prompt: Generate a JSON list of 50 Wikipedia pages title about {topic}.
WIKI_ARTICLES = [
    
    # Sustainability
    "Sustainability",
    "Sustainable development",
    "Sustainable Development Goals",
    "Sustainable agriculture",
    "Sustainable design",
    "Sustainable energy",
    "Sustainable city",
    "Renewable and Sustainable Energy Reviews",
    "Sustainable tourism",
    "Sustainable living",
    "Sustainable yield",
    "Sustainable business",
    "Sustainable architecture",
    "Sustainable transport",
    "Sustainable gardening",
    "Sustainable management",
    
    # Green economy/Economy
    "Green economy",
    "Economy",
    "Blue economy",
    "Green growth",
    "Circular economy",
    "Green economy policies in Canada",
    "European Green Deal",
    "Green New Deal",
    "Ecological economics",
    "Low-carbon economy",
    "Green-collar worker",
    "Extractivism",
    "Climate of Ethiopia",
    "Ceres Power",
    "Green bond",
    "Recycling",
    "Economy of West Bengal",
    "Eco-socialism",
    "Economic system",
    "Green Energy Act, 2009",
    "Sustainable finance",
    "Malaysian Green Transition",
    "Economy of Ontario",
    "Economy of China",
    "International Labour Organization",
    "Water conservation",
    "Economy of the United Kingdom",
    "World Environment Day",
    "Atom economy",
    "Eco-capitalism",
    "Economy of Milan",
    "Economy of India",
    
    # Renewable energy
    "Renewable energy",
    "Renewable energy in India",
    "Renewable energy commercialization",
    "Renewable energy in China",
    "Variable renewable energy",
    "100% renewable energy",
    "Renewable energy in the United States",
    "Renewable energy in the European Union",
    "Non-renewable resource",
    "Renewable energy in Germany",
    "Energy transition",
    "Renewable energy in Australia",
    "Renewable energy in Russia",
    "Renewable resource",
    "Renewable fuels",
    "Nuclear power",
    
    # Poverty
    "Poverty",
    "Poverty threshold",
    "Poverty in the United States",
    "Poverty in India",
    "Poverty reduction",
    "Extreme poverty",
    "Poverty Point",
    "Poverty porn",
    "Cycle of poverty",
    "Poverty in China",
    "Causes of poverty",
    "Corporate poverty",
    "Poverty in Bangladesh",
    "Poverty in Africa",
    "Poverty in the Philippines",
    "Poverty in Pakistan",
    "Diseases of poverty",
    "Measuring poverty",
    "Poverty in Kenya",
    "Poverty of the stimulus",
    "Child poverty",
    "Multidimensional Poverty Index",
    "Poverty in South America",
    "Poverty in the United Kingdom",
    "Reservation poverty",
    "War on poverty",
    "Cost of poverty",
    "Poverty in Canada",
    "Poverty Island",
    "Poverty in Italy",
    
    # Climate change
    "Climate change",
    "Climate change denial",
    "Effects of climate change",
    "Climate change mitigation",
    "Climate variability and change",
    "Causes of climate change",
    "Climate change feedbacks",
    "Intergovernmental Panel on Climate Change",
    "Climate change in California",
    "Scientific consensus on climate change",
    "Politics of climate change",
    "Climate change adaptation",
    "Climate change and agriculture",
    "United Nations Climate Change Conference",
    "Paris Agreement",
    "2024 in climate change",
    "United Nations Framework Convention on Climate Change",
    "Climate change in the United States",
    "Climate change in Australia",
    "Effects of climate change on agriculture",
    "Copernicus Climate Change Service",
    "Deforestation and climate change"
]

def main():
    log.info(f"Create dataset of {len(WIKI_ARTICLES)} articles root nodes")
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
