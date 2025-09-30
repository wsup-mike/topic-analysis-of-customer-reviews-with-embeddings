import os, numpy as np, pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from scipy.spatial import distance

load_dotenv()
EMBEDDING_MODEL = "text-embedding-3-small"

# 1) Data
reviews = pd.read_csv("data/samples/womens_clothing_e-commerce_reviews.csv")
review_texts = reviews["Review Text"].dropna().astype(str)

# 2) Chroma (let it embed for us)
chroma_client = chromadb.PersistentClient()
collection = chroma_client.create_collection(
    name="review_embeddings",
    embedding_function=OpenAIEmbeddingFunction(
        model_name=EMBEDDING_MODEL, api_key=os.environ["OPENAI_API_KEY"])
)
collection.add(documents=review_texts.tolist(),
               ids=[str(i) for i in range(len(review_texts))])

# 3) Get vectors back for t-SNE (small subset if large)
vecs = np.array(collection.get(include=["embeddings"])["embeddings"])
sample_idx = np.random.choice(len(vecs), size=min(800, len(vecs)), replace=False)
vecs_sample = vecs[sample_idx]

tsne = TSNE(n_components=2, random_state=0, init="random", learning_rate="auto")
pts = tsne.fit_transform(vecs_sample)
plt.figure(figsize=(10,7))
plt.scatter(pts[:,0], pts[:,1], alpha=0.5)
plt.title("t-SNE of Review Embeddings")
plt.show()

# 4) Topic centroids
categories = ["Quality", "Fit", "Style", "Comfort"]
openai_client = OpenAI()
cat_resp = openai_client.embeddings.create(input=categories, model=EMBEDDING_MODEL)
cat_embs = [d.embedding for d in cat_resp.data]

def categorize_feedback(text_emb, cat_embs):
    sims = [{"distance": distance.cosine(text_emb, c), "i": i} for i, c in enumerate(cat_embs)]
    return categories[min(sims, key=lambda x: x["distance"])["i"]]

feedback_categories = [categorize_feedback(e, cat_embs) for e in vecs]

# 5) Similarity search
def find_similar_reviews(query, n=3):
    res = collection.query(query_texts=[query], n_results=n)
    return res["documents"][0]

print(find_similar_reviews("Absolutely wonderful - silky and sexy and comfortable", 3))

# clean up if you want a fresh run next time:
# chroma_client.delete_collection(name="review_embeddings")