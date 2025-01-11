import logging
from datasets import load_dataset, Dataset, Features, Value
import os

# Create & configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def chunk_text(
    text: str,
    chunk_size: int = 2048,
    overlap=256,
    tokenizer=None,
    max_tokens: int = 2048,
) -> list[str]:
    """
    Splits a text into overlapping chunks of a specified size (in words),
    ensuring that words are not split between chunks.

    Args:
        text: The text to be split.
        chunk_size: The desired size of each chunk in words.
        overlap: The number of words to overlap between consecutive chunks.

    Returns:
        A list of text chunks.
    """
    words = text.split()
    chunks = []

    logger.info(
        f"Splitting text of {len(words)} words into chunks of {chunk_size} words, with {overlap} words of overlap"
    )

    last_index = 0
    for i in range(0, len(words) - chunk_size + 1, chunk_size - overlap):
        logger.info(
            f"Processing chunk from position {i} to position {i + chunk_size - 1}"
        )
        chunks.append(" ".join(words[i : i + chunk_size]))
        last_index = i + chunk_size - overlap

    # If small chunk is reamining
    if last_index < len(words):
        # Append the remaining words
        chunks.append(" ".join(words[last_index:]))
        logger.info(
            f"Last chunk size: {len(words[last_index:])} words, starting from index {last_index}"
        )

    logger.info(f"Text of {len(words)} words split into {len(chunks)} chunks")

    if tokenizer:
        logger.info(f"Checking if each text chunk is below {max_tokens} tokens")
        for chunk in chunks:
            tokens = tokenizer.tokenize(chunk)
            if len(tokens) > max_tokens:
                logger.error(f"Found token sequence with {len(tokens)}.")
                raise Exception(f"Found chunk with {len(tokens)} tokens. Dataset not valid.")
            # logger.info(f"Valid text chunk of words {len(chunk.split())} converted into tokens {len(tokens)}")
                
    return chunks


def read_txt_file_to_string(file_path):
    """Reads a txt file and returns its content as a string.

    Args:
        file_path: The path to the txt file.

    Returns:
        The content of the file as a string.
    """
    try:
        with open(file_path, "r") as file:
            file_content = file.read()
            logger.info(f"Read {len(file_content)} characters from {file_path}")
        return file_content
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None


def load_text_folder_dataset(
    data_dir: str,
    chunk_size: int,
    overlap: int,
    field_name: str = "text",
    split_name: str = "train",
    tokenizer=None,
    max_tokens: int = 2048,
) -> Dataset:
    logger.info(
        f"""Loadeding text dataset for folder {data_dir}, chunk size {chunk_size}, 
        overlap {overlap}, field name {field_name}, split name {split_name}"""
    )
    data_files = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".txt")
    ]

    file_content = [read_txt_file_to_string(f) for f in data_files]

    dataset = Dataset.from_dict(
        mapping={
            field_name: [
                item
                for sublist in [
                    chunk_text(f, chunk_size, overlap, tokenizer, max_tokens)
                    for f in file_content
                ]
                for item in sublist
            ]
        },
        split=split_name,
    )

    logger.info(
        f"Loaded dataset with {len(dataset)} samples. Dataset object: \n{dataset}\n"
    )

    return dataset


# if __name__ == "__main__":
#     data_dir = "/home/bruno/Documents/GitHub/social-media-nlp/training/dataset_txt"
#     dataset = load_text_folder_dataset(data_dir=data_dir, chunk_size=1024, overlap=128)
#     print(dataset)
#     logger.info(f"First 5 rows of dataset: \n{dataset[:5]}")
