from sentence_transformers import SentenceTransformer
from torch import Tensor

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")


def encode_phrase(phrase:str) -> Tensor:
    return model.encode(phrase)





# The sentences to encode
# sentences = [
#     "The weather is lovely today.",
#     "It's so sunny outside!",
#     "He drove to the stadium.",
# ]

# 2. Calculate embeddings by calling model.encode()
# embeddings = model.encode("no poverty")
# print(f"Shape of embeddings: {embeddings.shape}")
# print(embeddings)
# [3, 384]

# 3. Calculate the embedding similarities
# similarities = model.similarity(embeddings, embeddings)
# print(similarities)
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])