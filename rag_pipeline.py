from sentence_transformers import SentenceTransformer
import faiss
import json

class RAGPipeline:
    def __init__(self):
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.qa_data = self.load_qa("./dataset/qa_pairs.json")
        self.index, self.embeddings = self.build_index(self.qa_data)

    def load_qa(self, qa_path):
        with open(qa_path, "r") as f:
            data = json.load(f)

        required_keys = ["question", "answer", "category"]
        for item in data:
            for key in required_keys:
                if key not in item:
                    raise ValueError(f"Missing key {key} in QA data")

        return data

    def build_index(self, qa_data):
        questions = [item["question"] for item in qa_data]

        embeddings = self.embed_model.encode(
            questions,
            convert_to_numpy=True,
            normalize_embeddings=True 
        )

        dim = embeddings.shape[1]

        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        return index, embeddings

    def retrieve(self, query, k=3):
        query_vec = self.embed_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            item = self.qa_data[idx]

            results.append({
                "question": item["question"],
                "answer": item["answer"],
                "category": item.get("category"),
                "policy_ref": item.get("policy_ref"),
                "risk_level": item.get("risk_level"),
                "suggested_action": item.get("suggested_action"),
                "score": float(score)
            })

        return results

    def get_best_answer(self, query):
        results = self.retrieve(query, k=3)
        best = results[0]

        return {
            "answer": best["answer"],
            "policy_ref": best["policy_ref"],
            "risk_level": best["risk_level"],
            "suggested_action": best["suggested_action"],
            "confidence": best["score"]
        }