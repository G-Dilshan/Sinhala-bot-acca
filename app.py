import os
import logging
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import google.generativeai as genai
from utils.document_processor import DocumentProcessor
from utils.vector_store import VectorStore
from utils.sinhala_transliterator import SinhalaTransliterator

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize components
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
document_processor = DocumentProcessor()
vector_store = VectorStore()
transliterator = SinhalaTransliterator()

class SinhalaRAG:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.vector_store = vector_store
        self.transliterator = transliterator
        self.setup_knowledge_base()
    
    def setup_knowledge_base(self):
        """Initialize the knowledge base from PDF documents"""
        uploads_dir = 'uploads'
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)
            logger.warning(f"Created {uploads_dir} directory. Please add your PDF files.")
            return
        
        pdf_files = [f for f in os.listdir(uploads_dir) if f.endswith('.pdf')]
        
        if not pdf_files:
            logger.warning("No PDF files found in uploads directory")
            return
        
        logger.info(f"Processing {len(pdf_files)} PDF files...")
        
        # Check if vector store already exists
        if not self.vector_store.collection_exists():
            documents = []
            for pdf_file in pdf_files:
                file_path = os.path.join(uploads_dir, pdf_file)
                logger.info(f"Processing {pdf_file}...")
                
                # Extract text from PDF
                text_chunks = document_processor.extract_text_from_pdf(file_path)
                
                # Add metadata
                for i, chunk in enumerate(text_chunks):
                    documents.append({
                        'text': chunk,
                        'source': pdf_file,
                        'chunk_id': i,
                        'document_type': self.get_document_type(pdf_file)
                    })
            
            # Create vector store
            if documents:
                self.vector_store.create_collection(documents)
                logger.info(f"Successfully processed {len(documents)} text chunks")
            else:
                logger.error("No documents were processed")
    
    def get_document_type(self, filename):
        """Extract document type from filename"""
        if 'LKAS' in filename:
            return 'ගිණුම්කරණ ප්‍රමිතිය'  # Accounting Standard
        elif 'SLFRS' in filename:
            return 'මූල්‍ය වාර්තාකරණ ප්‍රමිතිය'  # Financial Reporting Standard
        return 'ප්‍රමිතිය'  # Standard
    
    def process_query(self, user_query):
        """Process user query and return response"""
        try:
            # Convert English-typed Sinhala to proper Sinhala if needed
            processed_query = self.transliterator.process_query(user_query)
            
            # Retrieve relevant documents
            relevant_docs = self.vector_store.search(processed_query, k=5)
            
            if not relevant_docs:
                return "මට ඔබේ ප්‍රශ්නයට අදාළ තොරතුරු සොයා ගැනීමට නොහැකි විය. කරුණාකර වෙනත් ප්‍රශ්නයක් අසන්න."
            
            # Prepare context
            context = self.prepare_context(relevant_docs)
            
            # Generate response
            response = self.generate_response(processed_query, context)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return "ක්‍රමලේඛයේ දෝෂයක් සිදු වී ඇත. කරුණාකර නැවත උත්සාහ කරන්න."
    
    def prepare_context(self, relevant_docs):
        """Prepare context from relevant documents"""
        context_parts = []
        
        for doc in relevant_docs:
            source_info = f"ප්‍රභවය: {doc['metadata']['source']}"
            doc_type = doc['metadata']['document_type']
            content = doc['text']
            
            context_parts.append(f"{source_info} ({doc_type}):\n{content}\n")
        
        return "\n".join(context_parts)
    
    def generate_response(self, query, context):
        """Generate response using Gemini model"""
        prompt = f"""
        ඔබ සිංහල ගිණුම්කරණ ප්‍රමිතීන් සහ මූල්‍ය වාර්තාකරණ ප්‍රමිතීන් පිළිබඳ විශේෂඥයෙකි. 
        
        පරිශීලක ප්‍රශ්නය: {query}
        
        අදාළ ප්‍රලේඛන තොරතුරු:
        {context}
        
        උපදෙස්:
        
        1. සම්පූර්ණයෙන්ම සිංහල භාෂාවෙන් පිළිතුරු දෙන්න
        2. ප්‍රලේඛනවල ඇති නිශ්චිත තොරතුරු මත පදනම්ව පිළිතුරු දෙන්න
        3. නිවැරදි සහ ගැඹුරු පිළිතුරක් ලබා දෙන්න
        4. ගිණුම්කරණ සිද්ධාන්තවල විස්තර සහ උදාහරණ ඇතුළත් කරන්න
        5. සම්බන්ධ LKAS/SLFRS අංක සඳහන් කරන්න
        6. අවශ්‍ය නම් පියවරෙන් පියවර පැහැදිලි කිරීමක් ලබා දෙන්න
        7. ප්‍රශ්නයට සෘජු පිළිතුරක් ලබා දෙන්න
        8. සන්දර්භයෙන් පිටත දේවල් ගැන ඇහුවොත්, සන්දර්භය ගැන ප්‍රශ්න අහන්න ආචාරශීලීව කියන්න.

        පිළිතුර:
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "ප්‍රතිචාරය ජනනය කිරීමේදී දෝෂයක් සිදු විය. කරුණාකර නැවත උත්සාහ කරන්න."

# Initialize RAG system
rag_system = SinhalaRAG()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'ප්‍රශ්නයක් ලබා දෙන්න'}), 400
        
        logger.info(f"Processing question: {question}")
        answer = rag_system.process_query(question)
        
        return jsonify({
            'answer': answer,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error in ask_question: {str(e)}")
        return jsonify({
            'error': 'අභ්‍යන්තර දෝෂයක් සිදු විය',
            'status': 'error'
        }), 500

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'message': 'සේවාව ක්‍රියාත්මකයි'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)