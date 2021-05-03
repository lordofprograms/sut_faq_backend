import os
from typing import List
from elasticsearch import Elasticsearch


class ElasticSearchWorker:
    def __init__(self):
        """class for vector and full-text search through elastic"""
        self._es = Elasticsearch([f"{os.environ['ELASTIC_IP']}:9200"],
                                 http_auth=(os.environ['ELASTIC_USERNAME'], os.environ['ELASTIC_PASSWORD']),
                                 verify_certs=False)

    def search_by_vector(self, query_vector: List[float], vector_field: str = "answer_vector", output_size: int = 3,
                         index: str = "qa_index") -> List[dict]:
        script_query = {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": f"cosineSimilarity(params.query_vector, '{vector_field}') + 1.0",
                    "params": {"query_vector": query_vector}
                }
            }
        }
        response = self._es.search(
            index=index,
            body={
                "size": output_size,
                "query": script_query,
                "_source": {"includes": ["question", "answer"]}
            }
        )
        qa_docs = []
        for hit in response["hits"]["hits"]:
            score, source = hit["_score"] - 1.0, hit["_source"]
            # if round(score, 2) > 0.7:
            qa_docs.append({"score": score, "qa_doc": source})
            # print({"score": score, "qa_doc": source})
        return qa_docs

    def get_total(self, index: str = "emotions") -> int:
        """return total rows of the given index(db) in elasticsearch"""
        res = self._es.search(index=index, body={"query": {"match_all": {}}})
        return res['hits']['total']['value']
