from utils.predict_intent import predict_intent
from llm_pipeline import generate_response
from rag_pipeline import RAGPipeline

rag = RAGPipeline()

def run_pipeline(query, chat_history=None):
    if chat_history is None:
        chat_history = []

    chat_history = [
        msg for msg in chat_history
        if isinstance(msg, dict) and "role" in msg and "content" in msg
    ]

    intent, intent_conf = predict_intent(query)

    rag_result = rag.get_best_answer(query)

    response = generate_response(
        query=query,
        rag_result=rag_result,
        chat_history=chat_history
    )

    return {
        "response": response,
        "intent": intent,
        "intent_confidence": intent_conf,
        "rag_confidence": rag_result.get("confidence", 0)
    }