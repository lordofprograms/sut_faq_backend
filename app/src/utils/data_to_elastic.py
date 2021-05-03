import gc
import os
import sys
from time import sleep
from typing import List
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from elasticsearch.exceptions import RequestError

sys.path.insert(0, ".")
from app.src.embedding import BertEmbedding


def get_qa_data(file_path: str, embedding: BertEmbedding) -> List[dict]:
    df = pd.read_csv(file_path)
    qa_records = df.to_dict('records')
    for item in qa_records:
        item['question_vector'] = embedding.embed(item['cleaned_question']).tolist()
        item['answer_vector'] = embedding.embed(item['cleaned_answer']).tolist()
        del item['cleaned_question']
        del item['cleaned_answer']
    return qa_records


def write_data(es: Elasticsearch, index_config: str, index_name: str = "qa_index", data: List[dict] = None):
    with open(index_config) as index_scheme_file:
        source = index_scheme_file.read().strip()
        try:
            if es.indices.exists(index=index_name):
                es.indices.delete(index=index_name, ignore=[400, 404])
                print(f"Delete {index_name}")
            es.indices.create(index=index_name, body=source)
            print(f"Successfully created {index_name}")
        except RequestError as e:
            print(f"You have problems with your db:{str(e)}")

    if data:
        response = bulk(es, data, index=index_name)
        print(response)
    else:
        print("Data shouldn't be None")


def qa_index_exists(es: Elasticsearch, index_name: str = "qa_index"):
    return es.indices.exists(index_name)


if __name__ == '__main__':
    sleep(30)
    elastic = Elasticsearch([f"{os.environ['ELASTIC_IP']}:9200"],
                            http_auth=(os.environ['ELASTIC_USERNAME'], os.environ['ELASTIC_PASSWORD']),
                            verify_certs=False)
    if not qa_index_exists(elastic) or os.environ['UPDATE_DB'] == 'true':
        faq_file_path = "app/data/dut_faq_dataset_extended.csv"
        bert_embedding = BertEmbedding("youscan/ukr-roberta-base")
        qa_data = get_qa_data(faq_file_path, bert_embedding)

        write_data(elastic, index_config="app/config/es_qa_index.json", data=qa_data)
        del bert_embedding
        gc.collect()
    else:
        print('Already have qa data')
