import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

class LocalVectorStore:
    def __init__(self, model_path="models/minilm.onnx"):
        # We use Hugging Face for tokenization, but run the AI on Snapdragon
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        
        # Force the MiniLM model onto the Hexagon NPU
        providers = [
            ("QNNExecutionProvider", {"backend_path": "QnnHtp.dll"}),
            "CPUExecutionProvider"
        ]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.chunks = []
        self.embeddings = []

    def build_index(self, text, chunk_size=200):
        """Splits the document into smaller pieces and generates NPU embeddings for each."""
        words = text.split()
        self.chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        self.embeddings = [self.embed(chunk) for chunk in self.chunks]

    def embed(self, text):
        """Converts text into numerical vectors using the ONNX model."""
        inputs = self.tokenizer(text, return_tensors="np", padding=True, truncation=True, max_length=256)
        
        ort_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "token_type_ids": inputs.get("token_type_ids", np.zeros_like(inputs["input_ids"]))
        }
        
        outputs = self.session.run(None, ort_inputs)
        # Apply mean pooling to the output tensor
        embedding = outputs[0].mean(axis=1).flatten()
        return embedding / np.linalg.norm(embedding)

    def search(self, query, top_k=2):
        """Finds the most relevant document chunks using Cosine Similarity."""
        query_vector = self.embed(query)
        similarities = [np.dot(query_vector, doc_vector) for doc_vector in self.embeddings]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.chunks[i] for i in top_indices]