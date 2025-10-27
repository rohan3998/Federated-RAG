"""fedrag: A Flower Federated RAG app."""

import hashlib
import json
import os
import time
from collections import defaultdict
from itertools import cycle
from time import sleep
from difflib import SequenceMatcher

import numpy as np
from flwr.app import ConfigRecord, Context, Message, MessageType, RecordDict
from flwr.serverapp import Grid, ServerApp

from fedrag.llm_querier import LLMQuerier
from fedrag.task import index_exists
from fedrag.reranker_model import RerankerModel, title_similarity


def node_online_loop(grid: Grid) -> list[int]:
    node_ids = []
    while not node_ids:
        # Get IDs of nodes available
        node_ids = grid.get_node_ids()
        # Wait if no node is available
        sleep(1)
    return node_ids


def get_hash(doc):
    # Create and return an SHA-256 hash for the given document
    return hashlib.sha256(doc.encode())


def calculate_title_similarity(title1: str, title2: str) -> float:
    """Calculate similarity between two titles using SequenceMatcher."""
    if not title1 or not title2:
        return 0.0
    
    # Normalize titles for comparison
    t1 = title1.lower().strip()
    t2 = title2.lower().strip()
    
    # Use SequenceMatcher for similarity
    similarity = SequenceMatcher(None, t1, t2).ratio()
    
    # Check for exact substring matches (boost score)
    if t1 in t2 or t2 in t1:
        similarity = max(similarity, 0.8)
    
    # Check for word overlap
    words1 = set(t1.split())
    words2 = set(t2.split())
    if words1 and words2:
        word_overlap = len(words1.intersection(words2)) / max(len(words1), len(words2))
        similarity = max(similarity, word_overlap * 0.7)
    
    return similarity


def rerank_documents(
    documents: list[str], 
    titles: list[str], 
    scores: list[float], 
    question_title: str = None,
    title_weight: float = 0.6,
    score_weight: float = 0.4,
    knn: int = 5
) -> tuple[list[str], list[str], list[float]]:
    """
    Rerank documents based on original scores and title similarity.
    
    Args:
        documents: List of document contents
        titles: List of document titles
        scores: List of original retrieval scores
        question_title: Expected title from question.json
        title_weight: Weight for title similarity (0.0 to 1.0)
        score_weight: Weight for original scores (0.0 to 1.0)
    
    Returns:
        Tuple of (reranked_documents, reranked_titles, reranked_scores)
    """
    if not documents or len(documents) != len(titles) or len(documents) != len(scores):
        return documents, titles, scores
    
    # Normalize original scores to 0-1 range
    if scores:
        min_score = min(scores)
        max_score = max(scores)
        if max_score > min_score:
            normalized_scores = [(s - min_score) / (max_score - min_score) for s in scores]
        else:
            normalized_scores = [1.0] * len(scores)
    else:
        normalized_scores = [1.0] * len(scores)
    
    # Calculate title similarities if question title is provided
    title_similarities = []
    if question_title:
        for title in titles:
            similarity = calculate_title_similarity(question_title, title)
            title_similarities.append(similarity)
    else:
        title_similarities = [0.0] * len(titles)
    
    # Calculate combined scores
    combined_scores = []
    for i in range(len(documents)):
        combined_score = (
            score_weight * normalized_scores[i] + 
            title_weight * title_similarities[i]
        )
        combined_scores.append(combined_score)
    
    # Create tuples for sorting
    doc_data = list(zip(documents, titles, scores, combined_scores, title_similarities))
    
    # Sort by combined score (descending)
    doc_data.sort(key=lambda x: x[3], reverse=True)
    
    # Extract sorted data
    reranked_documents = [item[0] for item in doc_data]
    reranked_titles = [item[1] for item in doc_data]
    reranked_scores = [item[2] for item in doc_data]
    
    # Print reranking info
    print(f"🔄 Reranking {len(documents)} documents using title similarity")
    if question_title:
        print(f"🎯 Target title: '{question_title}'")
        
        # Calculate how many documents changed position
        position_changes = 0
        for i, (doc, title, orig_score, combined_score, title_sim) in enumerate(doc_data):
            original_pos = documents.index(doc)
            if original_pos != i:
                position_changes += 1
        
        print(f"📈 Reranking impact: {position_changes}/{len(documents)} documents changed position")
        
        # Show top 3 reranked results with detailed scores
        print("🏆 Top reranked documents:")
        for i in range(min(3, len(doc_data))):
            title_sim = doc_data[i][4]
            combined_score = doc_data[i][3]
            orig_score = doc_data[i][2]
            original_pos = documents.index(doc_data[i][0]) + 1
            movement = f"↑{original_pos - (i+1)}" if original_pos > i+1 else f"↓{(i+1) - original_pos}" if original_pos < i+1 else "="
            
            print(f"   {i+1}. {doc_data[i][1][:40]}...")
            print(f"      Original pos: #{original_pos} {movement} | Orig score: {orig_score:.3f}")
            print(f"      New score: {combined_score:.3f}")
    else:
        print("ℹ️  No target title provided - using original ranking")
    
    return reranked_documents[:knn], reranked_titles[:knn], reranked_scores[:knn]


def rerank_documents_with_model(
    documents: list[str],
    titles: list[str],
    scores: list[float],
    question_title: str | None,
    model_weights: list[float] | None,
    knn: int,
) -> tuple[list[str], list[str], list[float]]:
    """Rerank documents using a trained linear reranker model.

    Falls back to score-only if no weights provided.
    """
    if not documents or len(documents) != len(titles) or len(documents) != len(scores):
        return documents, titles, scores

    # Normalize retrieval scores to [0,1]
    min_s, max_s = float(min(scores)), float(max(scores))
    if max_s > min_s:
        norm_scores = [(s - min_s) / (max_s - min_s) for s in scores]
    else:
        norm_scores = [1.0] * len(scores)

    # Compute title sims
    sims = [title_similarity(question_title, t) if question_title else 0.0 for t in titles]

    if not model_weights:
        # No trained model yet; just return score sort
        combined = list(zip(documents, titles, scores, norm_scores))
        combined.sort(key=lambda x: x[3], reverse=True)
        docs_sorted = [c[0] for c in combined][:knn]
        titles_sorted = [c[1] for c in combined][:knn]
        orig_scores_sorted = [c[2] for c in combined][:knn]
        print("ℹ️  No trained reranker found: using retrieval scores only")
        return docs_sorted, titles_sorted, orig_scores_sorted

    model = RerankerModel(model_weights)
    X = np.array(list(zip(norm_scores, sims)), dtype=np.float32)
    logits = model.predict_scores(X)

    combined = list(zip(documents, titles, scores, logits))
    combined.sort(key=lambda x: x[3], reverse=True)
    docs_sorted = [c[0] for c in combined][:knn]
    titles_sorted = [c[1] for c in combined][:knn]
    orig_scores_sorted = [c[2] for c in combined][:knn]

    print("🔄 Reranking with trained reranker model (linear)")
    return docs_sorted, titles_sorted, orig_scores_sorted


def load_question_metadata(question_text: str) -> dict:
    """
    Load question metadata from question.json file.
    
    Args:
        question_text: The question text to match
    
    Returns:
        Dictionary with question metadata including title
    """
    try:
        with open("question.json", 'r', encoding='utf-8') as f:
            loaded = json.load(f)

        # Handle single object format
        if isinstance(loaded, dict):
            if loaded.get("title"):
                stored_question = loaded.get("question", "").strip().lower()
                current_question = question_text.strip().lower()
                if stored_question == current_question or stored_question in current_question or current_question in stored_question:
                    return loaded
                else:
                    print(f"⚠️  Using title from question.json despite question mismatch")
                    print(f"   Current: {question_text[:50]}...")
                    print(f"   Stored: {loaded.get('question', 'N/A')[:50]}...")
                    return loaded
            return {}

        # Handle list of objects format
        if isinstance(loaded, list):
            current_question = question_text.strip().lower()
            best_match = None
            for item in loaded:
                if not isinstance(item, dict):
                    continue
                q = str(item.get("question", "")).strip().lower()
                if q == current_question:
                    return item
                if q and (q in current_question or current_question in q):
                    best_match = item
            if best_match:
                print("⚠️  Using closest title from question.json list based on partial match")
                return best_match
            return {}
        
        return {}
    
    except FileNotFoundError:
        # Don't print warning if file doesn't exist - this is normal for question.txt mode
        return {}
    except json.JSONDecodeError as e:
        print(f"⚠️  Error parsing question.json: {e}")
        return {}
    except Exception as e:
        print(f"⚠️  Error loading question.json: {e}")
        return {}


def merge_documents(documents, scores, titles, knn, k_rrf=0, reverse_sort=False) -> tuple[list[str], list[str]]:
    """Merge and rank documents from all clients using score sort or RRF.

    Args:
        documents: Concatenated list of documents from clients.
        scores: Corresponding retrieval scores.
        titles: Titles for each document.
        knn: Number of final documents to keep after merge.
        k_rrf: RRF hyperparameter; 0 disables RRF and uses pure score sorting.
        reverse_sort: Sort scores descending if True (for metrics where higher is better).

    Returns:
        Tuple of (docs, titles) after merging and truncation to k.
    """
    RRF_dict = defaultdict(dict)
    sorted_scores = np.array(scores).argsort()
    if reverse_sort:  # from larger to smaller scores
        sorted_scores = sorted_scores[::-1]
    sorted_documents = [documents[i] for i in sorted_scores]
    sorted_titles = [titles[i] for i in sorted_scores]
    for i in range(min(3, len(sorted_titles))):
        doc_preview = sorted_documents[i][:60].replace('\n', ' ').strip() + "..."
        print(f"   {i+1}. {sorted_titles[i]} (score: {scores[sorted_scores[i]]:.4f})")
        print(f"      Preview: {doc_preview}")
    if k_rrf == 0:
        # If k_rff is not set then simply return the
        # sorted documents based on their retrieval score
        return sorted_documents, sorted_titles
    else:
        for doc_idx, doc in enumerate(sorted_documents):
            # Given that some returned results/documents could be extremely
            # large we cannot use the original document as a dictionary key.
            # Therefore, we first hash the returned string/document to a
            # representative hash code, and we use that code as a key for
            # the final RRF dictionary. We follow this approach, because a
            # document could  have been retrieved twice by multiple clients
            # but with different scores and depending on these scores we need
            # to maintain its ranking
            doc_hash = get_hash(doc)
            RRF_dict[doc_hash]["rank"] = 1 / (k_rrf + doc_idx + 1)
            RRF_dict[doc_hash]["doc"] = doc
            RRF_dict[doc_hash]["title"] = titles[doc_idx]

        RRF_docs = sorted(RRF_dict.values(), key=lambda x: x["rank"], reverse=True)
        docs = [rrf_res["doc"] for rrf_res in RRF_docs][
            :knn
        ]  # select the final top-k / k-nn
        titles = [rrf_res["title"] for rrf_res in RRF_docs][:knn]
        return docs, titles


def submit_question(
    grid: Grid,
    question: str,
    question_id: str,
    knn: int,
    node_ids: list,
    corpus_names_iter: iter,
):

    messages = []
    # Send the same Message to each connected node (which run `ClientApp` instances)
    for node_idx, node_id in enumerate(node_ids):
        # The payload of a Message is of type RecordDict
        # https://flower.ai/docs/framework/ref-api/flwr.common.RecordDict.html
        # which can carry different types of records. We'll use a ConfigRecord object
        # We need to create a new ConfigRecord() object for every node, otherwise
        # if we just override a single key, e.g., corpus_name, the grid will send
        # the same object to all nodes.
        config_record = ConfigRecord()
        config_record["question"] = question
        config_record["question_id"] = question_id
        config_record["knn"] = knn
        # Round-Robin assignment of corpus to individual clients
        # by infinitely looping over the corpus names.
        config_record["corpus_name"] = next(corpus_names_iter)

        # Optionally send reranker train payload if available in environment
        # Expect a JSON file path via ENV: FEDRAG_RERANKER_TRAIN_JSON
        train_json_path = os.environ.get("FEDRAG_RERANKER_TRAIN_JSON")
        if train_json_path and os.path.exists(train_json_path):
            try:
                with open(train_json_path, "r", encoding="utf-8") as f:
                    train_payload = json.load(f)
                config_record["reranker_train_payload"] = train_payload
                print("📦 Sent reranker training payload to clients")
            except Exception as e:
                print(f"⚠️  Failed loading training payload: {e}")

        task_record = RecordDict({"config": config_record})
        message = Message(
            content=task_record,
            message_type=MessageType.QUERY,  # target `query` method in ClientApp
            dst_node_id=node_id,
            group_id=str(question_id),
        )
        messages.append(message)
        print(f"   📡 Querying client {node_idx+1} (corpus: {config_record['corpus_name']})")
    
    # Send messages and wait for all results
    replies = grid.send_and_receive(messages)
    print(f"📨 Received {len(replies)}/{len(messages)} results from federated clients")

    documents, scores, titles = [], [], []
    client_weights = []
    client_results = []
    for i, reply in enumerate(replies):
        if reply.has_content():
            client_docs = reply.content["docs_n_scores"]["documents"]
            client_scores = reply.content["docs_n_scores"]["scores"]
            client_titles = reply.content["docs_n_scores"]["titles"]
            client_w = reply.content["docs_n_scores"].get("reranker_weights")
            
            documents.extend(client_docs)
            scores.extend(client_scores)
            titles.extend(client_titles)
            if client_w and isinstance(client_w, list) and len(client_w) == 3:
                try:
                    client_weights.append([float(x) for x in client_w])
                except Exception:
                    pass
            
            # Track results per client for detailed logging
            client_results.append({
                'client_id': i+1,
                'num_docs': len(client_docs),
                'top_score': max(client_scores) if client_scores else 0,
                'top_title': client_titles[0] if client_titles else "N/A"
            })
    
    # Show per-client results summary
    print("📊 Results per client:")
    for result in client_results:
        print(f"   Client {result['client_id']}: {result['num_docs']} docs, "
              f"top score: {result['top_score']:.4f}, top title: {result['top_title']}")

    # Aggregate reranker weights (simple average)
    aggregated_weights = None
    if client_weights:
        n = len(client_weights)
        summed = [0.0, 0.0, 0.0]
        for w in client_weights:
            summed[0] += w[0]
            summed[1] += w[1]
            summed[2] += w[2]
        aggregated_weights = [summed[0]/n, summed[1]/n, summed[2]/n]
        print(f"🧮 Aggregated reranker weights from {n} clients: {aggregated_weights}")

    return documents, scores, titles, aggregated_weights


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    node_ids = node_online_loop(grid)

    # k-reciprocal-rank-fusion is used by the server to merge
    # the results returned by the clients
    k_rrf = 0
    # k-nearest-neighbors for document retrieval at each client
    knn = int(context.run_config["k-nn"])
    corpus_names = context.run_config["clients-corpus-names"].split("|")
    corpus_names = [c.lower() for c in corpus_names]  # make them lower case
    # Before we start the execution of the FedRAG pipeline,
    # we need to make sure we have downloaded the corpus and
    # created the respective indices
    index_exists(corpus_names)
    # Create corpus iterator
    corpus_names_iter = cycle(corpus_names)
    # Use OpenAI model - default to gpt-4, but allow override from config
    model_name = context.run_config.get("server-llm-model", "gpt-4")
    use_gpu = context.run_config.get("server-llm-use-gpu", False)
    use_gpu = True if str(use_gpu).lower() == "true" else False

    llm_querier = LLMQuerier(model_name, use_gpu)
    
    print("=" * 60)
    print("🎯 FEDRAG FILE-BASED MODE")
    print("=" * 60)
    print(f"✅ Connected to {len(node_ids)} federated clients")
    print(f"✅ Corpus distribution: {corpus_names}")
    print(f"✅ LLM Model: {model_name}")
    print(f"✅ Retrieval settings: k-nn={knn}, k-rrf={k_rrf}")
    print("=" * 60)
    
    # Read questions from file - try question.json first, then question.txt
    questions = []
    questions_source = None
    
    # Try question.json first (supports single object or list of objects)
    try:
        with open("question.json", 'r', encoding='utf-8') as f:
            question_data = json.load(f)
            if isinstance(question_data, dict) and question_data.get("question"):
                questions = [question_data["question"]]
                questions_source = "question.json"
                print(f"📄 Loaded question from question.json with title: {question_data.get('title', 'N/A')}")
            elif isinstance(question_data, list):
                extracted = [q.get("question") for q in question_data if isinstance(q, dict) and q.get("question")]
                if extracted:
                    questions = extracted
                    questions_source = "question.json"
                    print(f"📄 Loaded {len(questions)} questions from question.json (list mode)")
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    
    # If no questions from JSON, try question.txt
    if not questions:
        try:
            with open("question.txt", 'r', encoding='utf-8') as f:
                questions = [line.strip() for line in f.readlines() if line.strip()]
                questions_source = "question.txt"
                print(f"📄 Loaded {len(questions)} questions from question.txt")
        except FileNotFoundError:
            pass
    
    # If still no questions, show error
    if not questions:
        print("❌ No questions found. Please create either:")
        print("   • question.json with format: {'question': 'your question', 'title': 'expected title'}")
        print("   • question.txt with questions, one per line")
        return
        
    print("=" * 60)
    
    question_count = 0
    
    # Process each question from the file
    for user_question in questions:
        try:
            question_count += 1
            q_id = f"user_question_{question_count}"
            
            print(f"\n🔎 Processing question {question_count}/{len(questions)}: {user_question}")
            print("📡 Querying federated clients...")
            
            # Start timer
            q_st = time.time()
            
            # Submit question to federated clients
            docs, scores, titles, agg_weights = submit_question(
                grid, user_question, q_id, knn, node_ids, corpus_names_iter
            )
            
            print(f"📚 Retrieved {len(docs)} documents from {len(node_ids)} clients")
            
            # Optionally persist aggregated weights for future runs
            if agg_weights:
                try:
                    RerankerModel.save(agg_weights, os.environ.get("FEDRAG_RERANKER_WEIGHTS", "reranker_weights.json"))
                    print("💾 Saved aggregated reranker weights")
                except Exception as e:
                    print(f"⚠️  Failed to save reranker weights: {e}")

            # Merge documents from all clients
            merged_docs, merged_titles = merge_documents(docs, scores, titles, knn, k_rrf)
            print(f"🔗 Merged to top-{len(merged_docs)} most relevant documents")
            
            # Show original ranking before re-ranking with document previews
            print("📋 Original ranking (after merge):")
            for i, (title, doc) in enumerate(zip(merged_titles[:5], merged_docs[:5]), 1):  # Show top 5
                doc_preview = doc[:100].replace('\n', ' ').strip() + "..." if len(doc) > 100 else doc
                print(f"   {i}. {title}")
                print(f"      Preview: {doc_preview}")
            if len(merged_titles) > 5:
                print(f"   ... and {len(merged_titles) - 5} more documents")
            
            # Load question metadata for title-based reranking
            question_metadata = load_question_metadata(user_question)
            question_title = question_metadata.get("title", None)
            
            # Rerank documents: prefer trained model if available
            model_weights_path = os.environ.get("FEDRAG_RERANKER_WEIGHTS", "reranker_weights.json")
            model_weights = RerankerModel.load(model_weights_path)
            if model_weights:
                # Use trained model
                merged_scores_dummy = [1.0 - (i * 0.1) for i in range(len(merged_docs))]
                reranked_docs, reranked_titles, reranked_scores = rerank_documents_with_model(
                    merged_docs, merged_titles, merged_scores_dummy, question_title, model_weights, knn
                )
            elif question_title:
                # Fallback to heuristic title-aware reranking
                merged_scores = [1.0 - (i * 0.1) for i in range(len(merged_docs))]
                reranked_docs, reranked_titles, reranked_scores = rerank_documents(
                    merged_docs, merged_titles, merged_scores, question_title, knn=knn
                )
                
                # Calculate title similarities for display
                title_similarities = []
                for title in merged_titles:
                    similarity = calculate_title_similarity(question_title, title)
                    title_similarities.append(similarity)
                
                reranked_title_similarities = []
                for title in reranked_titles:
                    similarity = calculate_title_similarity(question_title, title)
                    reranked_title_similarities.append(similarity)
                
                # Show detailed comparison between original and reranked
                print(f"\n📊 Detailed Ranking Comparison (Target: '{question_title}'):")
                print("=" * 80)
                
                # Show if any actual reranking occurred
                ranking_changed = merged_titles != reranked_titles
                if ranking_changed:
                    print("🔄 RERANKING OCCURRED - Documents were reordered!")
                else:
                    print("📌 NO RERANKING - Original order maintained")
                
                print("\nOriginal Ranking:")
                for i in range(min(5, len(merged_titles))):
                    title = merged_titles[i]
                    doc_preview = merged_docs[i][:80].replace('\n', ' ').strip() + "..."
                    sim_score = title_similarities[i]
                    print(f"   {i+1}. {title}")
                    print(f"      Similarity: {sim_score:.3f} | Preview: {doc_preview}")
                
                print("\nRe-ranked Results:")
                for i in range(min(5, len(reranked_titles))):
                    title = reranked_titles[i]
                    doc_preview = reranked_docs[i][:80].replace('\n', ' ').strip() + "..."
                    sim_score = reranked_title_similarities[i]
                    
                    # Find original position
                    try:
                        original_pos = merged_titles.index(title) + 1
                        movement = "same" if original_pos == i + 1 else f"moved from #{original_pos}"
                    except ValueError:
                        movement = "new"
                    
                    print(f"   {i+1}. {title} ({movement})")
                    print(f"      Similarity: {sim_score:.3f} | Preview: {doc_preview}")
                
                print("=" * 80)
                
                # Use reranked results
                merged_docs, merged_titles = reranked_docs, reranked_titles
            else:
                print("🔄 No title metadata or trained reranker found, using original ranking")
            
            # For interactive mode, we'll create a simple format without multiple choice
            # since we don't have predefined options
            options = {"A": "Based on the medical literature"}
            
            # Generate answer using LLM
            print("🤖 Generating answer with LLM...")
            prompt, predicted_answer = llm_querier.answer(
                user_question, merged_docs, options, "medical"
            )
            
            # Calculate response time
            q_et = time.time()
            response_time = q_et - q_st
            
            # Display results
            print("\n" + "=" * 60)
            print("📖 FEDRAG ANSWER")
            print("=" * 60)
            
            
            print(f"\n🎯 Question: {user_question}")
            
            # For interactive mode, we'll show the full LLM response instead of just the option
            # Extract the actual answer from the prompt
            if predicted_answer:
                print(f"🤖 Answer: {predicted_answer}")
            else:
                # If no specific answer was extracted, show the generated response
                print("🤖 Answer: Based on the consulted medical literature, here's what I found:")
                for doc in merged_docs[:2]:  # Show content from top 2 most relevant docs
                    relevant_content = doc[:200].replace('\n', ' ') + "..."
                    print(f"   • {relevant_content}")
            
            print(f"\n⏱️  Response time: {response_time:.2f} seconds")
            print("=" * 60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error processing question: {e}")
            print("Continuing with next question...")
            continue
    
    print("\n" + "=" * 60)
    print("🎉 PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"✅ Successfully processed {question_count}/{len(questions)} questions")
    print("👋 Thank you for using FedRAG!")
