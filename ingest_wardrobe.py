import json
import chromadb
from chromadb.utils import embedding_functions

def setup_wardrobe_vector_db():
    print("1. Initializing ChromaDB persistent client...")
    # PersistentClient saves data to disk (in ./chroma_db folder)
    client = chromadb.PersistentClient(path="./chroma_db")

    # Use a lightweight, standard embedding model (runs fast on CPU)
    embedding_func = embedding_functions.DefaultEmbeddingFunction()

    # Get or create the 'wardrobe' collection
    collection = client.get_or_create_collection(
        name="user_wardrobe",
        embedding_function=embedding_func
    )

    print("2. Reading wardrobe.json...")
    with open("wardrobe.json", "r") as f:
        wardrobe_items = json.load(f)

    ids = []
    documents = []
    metadatas = []

    for item in wardrobe_items:
        ids.append(item["id"])
        
        # The 'document' is the natural language text converted to a vector embedding
        doc_text = f"{item['name']} - Category: {item['category']}, Color: {item['color']}, Warmth: {item['warmth']}, Style: {item['style']}"
        documents.append(doc_text)
        
        # Metadata enables hard filtering (e.g. filter by color or category)
        metadatas.append({
            "name": item["name"],
            "category": item["category"],
            "color": item["color"],
            "warmth": item["warmth"],
            "style": item["style"]
        })

    print(f"3. Ingesting {len(documents)} items into vector collection...")
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )
    print("Ingestion complete!\n")

    # 4. Run a quick verification search
    test_query = "something warm for winter weather"
    print(f"4. Running test semantic query: '{test_query}'")
    results = collection.query(
        query_texts=[test_query],
        n_results=2
    )

    print("\n--- Top Matches ---")
    for doc, meta, distance in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
        print(f"- Item: {meta['name']} (Warmth: {meta['warmth']}, Color: {meta['color']}) | Distance: {distance:.4f}")

if __name__ == "__main__":
    setup_wardrobe_vector_db()