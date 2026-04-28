import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA

# -------------------------------
# Load PDF
# -------------------------------
loader = PyPDFLoader('NCERT-Class-10-History.pdf')
docs = loader.load()

# -------------------------------
# Text Split
# -------------------------------
split = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=10
)

chunks = split.split_documents(docs)

print("Total chunks:", len(chunks))
print(chunks[0].page_content)

# -------------------------------
# Convert to text
# -------------------------------
texts = []
for doc in chunks:
    texts.append(doc.page_content)

# -------------------------------
# Create IDs
# -------------------------------
ids = []
for i in range(len(texts)):
    ids.append(str(i))

# -------------------------------
# Embeddings
# -------------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

print("\nGenerating embeddings (this may take a minute)...")
embeddings = model.encode(texts).tolist()

# -------------------------------
# ChromaDB (Persistent)
# -------------------------------
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="History")

collection.add(
    documents=texts,
    embeddings=embeddings,
    ids=ids
)

print("Data stored with embeddings ✅")

# -------------------------------
# Ollama LLM Setup
# -------------------------------
llm = Ollama(model="llama3:8b", temperature=0.7)

# -------------------------------
# Interactive Query Loop
# -------------------------------
print("\n" + "="*50)
print("🤖 History RAG - Ask anything!")
print("Type 'quit' to exit")
print("="*50)

# Check if running in interactive mode
if not sys.stdin.isatty():
    print("\n⚠️  Not running in interactive mode. Exiting...")
    print("Please run this script directly in a terminal: ./venv/bin/python app.py")
    sys.exit(0)

while True:
    try:
        query = input("\n❓ Enter your question: ").strip()
    except EOFError:
        print("\nGoodbye! 👋")
        break

    if query.lower() == 'quit':
        print("Goodbye! 👋")
        break

    if not query:
        continue

    # Get query embedding
    query_embedding = model.encode([query]).tolist()

    # Retrieve relevant documents
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    retrieved_docs = results['documents'][0]

    # Build context
    context = "\n\n".join(retrieved_docs)

    # Create prompt for LLM
    prompt = f"""You are a helpful history assistant. Answer the question based on the context below.
If the answer is not in the context, say "I don't have information about this in the provided documents."

Context:
{context}

Question: {query}

Answer:"""

    # Get answer from Ollama
    print("\n🤖 Answer: ", end="", flush=True)
    answer = llm.invoke(prompt)
    print(answer)
