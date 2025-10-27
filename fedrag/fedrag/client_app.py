"""fedrag: A Flower Federated RAG app."""

from flwr.app import ConfigRecord, Context, Message, RecordDict
from flwr.clientapp import ClientApp
import numpy as np

from fedrag.retriever import Retriever
from fedrag.reranker_model import RerankerModel, title_similarity

# Flower ClientApp
app = ClientApp()


@app.query()
def query(msg: Message, context: Context):
    """Handle a server query by retrieving top-k documents from the local FAISS index.

    The function extracts the question, question ID, corpus name, and k-NN value
    from the message config, performs retrieval using `Retriever`, and returns
    documents, scores, and titles back to the server.
    """

    node_id = context.node_id

    # Extract question
    question = str(msg.content["config"]["question"])
    question_id = str(msg.content["config"]["question_id"])

    # Extract corpus name
    corpus_name = str(msg.content["config"]["corpus_name"])
    print("corpus_name : %s" % corpus_name)
    # Initialize retrieval system
    retriever = Retriever()
    # Use the knn value for retrieving the closest-k documents to the query
    knn = int(msg.content["config"]["knn"])
    retrieved_docs = retriever.query_faiss_index(corpus_name, question, knn)

    # Create lists with the computed scores and documents
    scores = [doc["score"] for doc_id, doc in retrieved_docs.items()]
    documents = [doc["content"] for doc_id, doc in retrieved_docs.items()]
    titles = [doc["title"] for doc_id, doc in retrieved_docs.items()]
    print(
        "ClientApp: {} - Question ID: {} - Retrieved: {} documents.".format(
            node_id, question_id, len(documents)
        )
    )
    # Federated reranker local training step (optional)
    train_payload = msg.content["config"].get("reranker_train_payload")
    if train_payload:
        # train_payload: dict with keys 'pairs' and optional 'weights'
        # pairs: List of items with fields: doc_score_norm, title, question_title, label
        try:
            pairs = train_payload.get("pairs", [])
            model_weights = train_payload.get("weights")
            model = RerankerModel(model_weights)
            X, y = [], []
            for p in pairs:
                s = float(p.get("doc_score_norm", 0.0))
                sim = float(title_similarity(p.get("question_title"), p.get("title")))
                X.append([s, sim])
                y.append(float(p.get("label", 0.0)))
            if X:
                model.train(np.array(X, dtype="float32"), np.array(y, dtype="float32"), lr=0.1, epochs=3)
            train_reply = {"reranker_weights": model.get_weights()}
        except Exception:
            train_reply = {"reranker_weights": train_payload.get("weights") if train_payload else None}
    else:
        train_reply = {"reranker_weights": None}

    # Create reply record with retrieved documents and optional updated weights.
    docs_n_scores = ConfigRecord(
        {
            "documents": documents,
            "scores": scores,
            "titles": titles,
            "reranker_weights": train_reply.get("reranker_weights"),
        }
    )
    reply_record = RecordDict({"docs_n_scores": docs_n_scores})

    # Return message
    return Message(reply_record, reply_to=msg)
