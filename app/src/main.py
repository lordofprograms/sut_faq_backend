from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.src.embedding import BertEmbedding
from app.src.es_worker import ElasticSearchWorker


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins="https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bert_embedding = BertEmbedding("youscan/ukr-roberta-base")
es_worker = ElasticSearchWorker()


@app.get("/api/healthcheck")
async def health_check():
    return "OK"


@app.get("/api/ask")
async def ask(question: str):
    # question = question.lower().replace("?", "")
    question = question.replace("?", "")
    question_vector = bert_embedding.embed(question).tolist()
    results = es_worker.search_by_vector(question_vector, vector_field="question_vector")
    return results

# def search_query_es(query: str, embedding: BertEmbedding, es_worker: ElasticSearchWorker):
#     # TODO maybe add tf-idf or similar tool to remove frequent words
#     # lowering and remove ?
#     query = query.lower().replace("?", "")
#
#     query_vector = embedding.embed(query).tolist()
#     # results = es_worker.search_by_vector(query_vector)
#     results = es_worker.search_by_vector(query_vector, vector_field="question_vector")
#     return results
#
#
# if __name__ == '__main__':
#     start_time = time()
#     # TODO try clean data: remove frequent words combinations and question mark
#     # maybe add query preprocessing
#     bert_embedding = BertEmbedding("youscan/ukr-roberta-base")
#     esw = ElasticSearchWorker()
#     question = "Чи є курси медичної допомоги?"
#     search_query_es(question, bert_embedding, esw)
#
#     # start_time = time()
#     # original_question = "Де я зможу працювати після закінчення університету?"
#     # answer = "Детальну інформацію про працевлаштування після закінчення університету Ви можете знайти за посиланням"
#     # query = "Робота після універу?"
#     # weird_query = "Яку якість освіти дає університет?"
#
#     # query_embedding = bert_embedding.embed(query)
#     # print(f"Creating first embedding time is {time() - start_time}")
#     # answer_embedding = bert_embedding.embed(answer)
#     # print(f"Creating second embedding time is {time() - start_time}")
#     #
#     # sim_res = bert_embedding.cosine_sim(answer_embedding, query_embedding)
#     # print(f"Comparing time: {time() - start_time}")
#     # print(sim_res)
