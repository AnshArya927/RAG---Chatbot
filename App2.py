from flask import Flask, render_template, request, jsonify, session, send_file
import warnings
warnings.filterwarnings("ignore")
import os
import uuid
import markdown
import sqlite3
from werkzeug.utils import secure_filename
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader 
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'

# --- CONFIGURATION ---
VECTOR_STORE_PATH = "vectorstore_gemini"
NOTES_PATH = "notes"
LLM_MODEL = "gemini-1.5-flash-latest"
ALLOWED_EXTS = {".pdf", ".txt", ".md"}

# Global variables
rag_chain = None
vector_store = None
is_initialized = False

# ---------------------- DATABASE SETUP ----------------------
DB_FILE = "chat_history.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_message TEXT,
            bot_response TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()  # initialize DB at app start

# ---------------------- HELPER FUNCTIONS ----------------------
def setup_api_key():
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = "API KEY here"  
    return True

def create_or_load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if os.path.exists(VECTOR_STORE_PATH):
        return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    
    loader = DirectoryLoader(NOTES_PATH, glob="**/*.txt")
    docs = loader.load()
    
    if not docs:
        dummy_doc = [Document(page_content="This is a placeholder. Add your own notes.")]
        return FAISS.from_documents(dummy_doc, embeddings)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)
    vector_store = FAISS.from_documents(split_docs, embeddings)
    vector_store.save_local(VECTOR_STORE_PATH)
    return vector_store

def create_rag_chain(vector_store):
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2, convert_system_message_to_human=True)
    
    prompt = ChatPromptTemplate.from_template("""
Use the following context to answer the user's question as helpfully and accurately as possible.
If the answer isn't directly in the context, feel free to suggest ideas or related insights.
Format your response with proper line breaks and use markdown formatting where appropriate.
Use bullet points or numbered lists when listing items.

Context:
{context}

Question: {input}
""")
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    return retrieval_chain

def process_markdown(text):
    md = markdown.Markdown(extensions=['nl2br', 'tables', 'fenced_code'])
    return md.convert(text)

def initialize_chatbot():
    global rag_chain, vector_store, is_initialized 
    try:
        setup_api_key()
        vector_store = create_or_load_vector_store()
        rag_chain = create_rag_chain(vector_store)
        is_initialized = True
        return True
    except Exception as e:
        print(f"Initialization error: {e}")
        return False

def save_chat(session_id, user_message, bot_response):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO chat_history (session_id, user_message, bot_response)
        VALUES (?, ?, ?)
    ''', (session_id, user_message, bot_response))
    conn.commit()
    conn.close()

def get_chat_history(session_id, limit=50):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        SELECT user_message, bot_response, timestamp
        FROM chat_history
        WHERE session_id=?
        ORDER BY timestamp ASC
        LIMIT ?
    ''', (session_id, limit))
    chats = c.fetchall()
    conn.close()
    return chats

# ---------------------- ROUTES ----------------------
@app.route('/')
def index():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    global rag_chain, is_initialized
    if not is_initialized and not initialize_chatbot():
        return jsonify({'error': 'Failed to initialize chatbot.'}), 500
    
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        show_sources = data.get('show_sources', False)
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Get RAG response
        response = rag_chain.invoke({"input": user_message})
        processed_response = process_markdown(response["answer"])
        
        # Save chat history
        save_chat(session['session_id'], user_message, processed_response)
        
        result = {'response': processed_response}
        if show_sources:
            result['sources'] = [
                {
                    'content': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    'source': doc.metadata.get('source', 'N/A')
                } 
                for doc in response.get("context", [])
            ]
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def history():
    session_id = session.get('session_id')
    if not session_id:
        return jsonify([])
    chats = get_chat_history(session_id)
    return jsonify(chats)

@app.route('/upload', methods=['POST'])
def upload():
    global vector_store, rag_chain, is_initialized
    if not is_initialized and not initialize_chatbot():
        return jsonify({'error': 'Failed to initialize chatbot.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    f = request.files['file']
    if f.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_EXTS:
        return jsonify({'error': f'Unsupported file type: {ext}'}), 400

    os.makedirs(NOTES_PATH, exist_ok=True)
    safe_name = secure_filename(f.filename)
    save_path = os.path.join(NOTES_PATH, safe_name)
    f.save(save_path)

    try:
        if ext == ".pdf":
            loader = PyPDFLoader(save_path)
        else:
            loader = TextLoader(save_path, encoding="utf-8")
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(docs)

        vector_store.add_documents(split_docs)
        vector_store.save_local(VECTOR_STORE_PATH)
        return jsonify({'ok': True, 'file': safe_name, 'chunks_added': len(split_docs)})
    except Exception as e:
        return jsonify({'error': f'Failed to process file: {e}'}), 500

@app.route('/status')
def status():
    return jsonify({'initialized': is_initialized})

if __name__ == '__main__':
    os.makedirs(NOTES_PATH, exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
