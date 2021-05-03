import torch
from torch.nn.functional import cosine_similarity
from transformers import RobertaTokenizer, RobertaModel


class BertEmbedding:
    def __init__(self, model_folder: str):
        self._tokenizer = RobertaTokenizer.from_pretrained(model_folder)
        self._model = RobertaModel.from_pretrained(model_folder)

    def embed(self, input_query: str):
        inputs = self._tokenizer(input_query, return_tensors="pt")
        output = self._model(**inputs)
        embedding = torch.mean(output.last_hidden_state, 1).squeeze()
        return embedding

    @staticmethod
    def cosine_sim(embedding1, embedding2):
        return cosine_similarity(embedding1, embedding2, dim=0).item()
