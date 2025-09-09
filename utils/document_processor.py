import PyPDF2
import re
import logging
from typing import List

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def extract_text_from_pdf(self, file_path: str) -> List[str]:
        """Extract text from PDF and return chunks"""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():  # Only add non-empty pages
                            text += f"\n--- පිටුව {page_num + 1} ---\n"
                            text += page_text
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num + 1}: {str(e)}")
                        continue
                
                if not text.strip():
                    logger.error(f"No text extracted from {file_path}")
                    return []
                
                # Clean and chunk the text
                cleaned_text = self.clean_text(text)
                chunks = self.create_chunks(cleaned_text)
                
                logger.info(f"Extracted {len(chunks)} chunks from {file_path}")
                return chunks
                
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            return []
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might interfere with processing
        text = re.sub(r'[^\u0D80-\u0DFF\u0020-\u007E\n\r\t]', '', text)
        
        # Fix common OCR errors in Sinhala text
        text = text.replace('ා', 'ා')  # Fix vowel marks
        text = text.replace('ී', 'ී')
        text = text.replace('ු', 'ු')
        text = text.replace('ූ', 'ූ')
        text = text.replace('ෙ', 'ෙ')
        text = text.replace('ේ', 'ේ')
        text = text.replace('ො', 'ො')
        text = text.replace('ෝ', 'ෝ')
        text = text.replace('ෞ', 'ෞ')
        
        # Remove page headers/footers that might be repetitive
        text = re.sub(r'පිටුව \d+', '', text)
        text = re.sub(r'--- පිටුව \d+ ---', '\n', text)
        
        return text.strip()
    
    def create_chunks(self, text: str) -> List[str]:
        """Create overlapping chunks from text"""
        if not text:
            return []
        
        # Split by sentences first (using Sinhala sentence endings)
        sentences = re.split(r'[.!?។]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    
                    # Create overlap with previous chunk
                    if self.chunk_overlap > 0:
                        words = current_chunk.split()
                        overlap_words = words[-min(self.chunk_overlap // 10, len(words)):]
                        current_chunk = ' '.join(overlap_words) + ' ' + sentence
                    else:
                        current_chunk = sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += ' ' + sentence if current_chunk else sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Filter out very short chunks
        chunks = [chunk for chunk in chunks if len(chunk) > 50]
        
        return chunks
    
    def extract_key_terms(self, text: str) -> List[str]:
        """Extract key Sinhala accounting terms"""
        key_terms = [
            'ගිණුම්කරණ', 'ප්‍රමිතිය', 'මූල්‍ය', 'ප්‍රකාශන', 'වත්කම්', 'වගකීම්', 
            'හිමිකම', 'ආදායම්', 'වියදම්', 'ලාභ', 'අලාභ', 'තොග', 'ක්ෂය',
            'ප්‍රතිපාදන', 'මුදල්', 'ප්‍රවාහ', 'කල්බදු', 'අයභාරය', 'ගනුදෙනුකරු',
            'ධාරණ', 'වටිනාකම', 'පිරිවැය', 'සාධාරණ', 'අගය', 'ගිවිසුම්',
            'කාර්යසාධන', 'බැඳීම්', 'ප්‍රතයාගණන', 'භාවිත', 'අයිතිය',
            'දේපළ', 'පිරියත', 'උපකරණ', 'අසම්භාව්‍ය'
        ]
        
        found_terms = []
        for term in key_terms:
            if term in text:
                found_terms.append(term)
        
        return found_terms