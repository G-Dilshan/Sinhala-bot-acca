import re
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class SinhalaTransliterator:
    def __init__(self):
        # English to Sinhala transliteration mapping
        self.transliteration_map = {
            # Common accounting terms
            'accounting': 'ගිණුම්කරණ',
            'ginumkarana': 'ගිණුම්කරණ',
            'ginumkaranaya': 'ගිණුම්කරණය',
            'standard': 'ප්‍රමිතිය',
            'pramiithiya': 'ප්‍රමිතිය',
            'financial': 'මූල්‍ය',
            'moolya': 'මූල්‍ය',
            'statement': 'ප්‍රකාශනය',
            'prakashana': 'ප්‍රකාශන',
            'prakashanaya': 'ප්‍රකාශනය',
            'assets': 'වත්කම්',
            'wathkam': 'වත්කම්',
            'liabilities': 'වගකීම්',
            'wagakeem': 'වගකීම්',
            'equity': 'හිමිකම',
            'himikam': 'හිමිකම',
            'revenue': 'ආදායම්',
            'aadayam': 'ආදායම්',
            'income': 'ආදායම්',
            'expenses': 'වියදම්',
            'wiyadam': 'වියදම්',
            'profit': 'ලාභ',
            'labha': 'ලාභ',
            'loss': 'අලාභ',
            'alabha': 'අලාභ',
            'inventory': 'තොග',
            'thoga': 'තොග',
            'depreciation': 'ක්ෂය',
            'kshaya': 'ක්ෂය',
            'provision': 'ප්‍රතිපාදන',
            'prathipaadana': 'ප්‍රතිපාදන',
            'cash': 'මුදල්',
            'mudal': 'මුදල්',
            'flow': 'ප්‍රවාහ',
            'prawaha': 'ප්‍රවාහ',
            'lease': 'කල්බදු',
            'kalbadu': 'කල්බදු',
            'customer': 'ගනුදෙනුකරු',
            'ganudenukaraya': 'ගනුදෙනුකරු',
            'carrying': 'ධාරණ',
            'dharana': 'ධාරණ',
            'value': 'වටිනාකම',
            'watinaakama': 'වටිනාකම',
            'cost': 'පිරිවැය',
            'piriwaya': 'පිරිවැය',
            'fair': 'සාධාරණ',
            'saadarana': 'සාධාරණ',
            'contract': 'ගිවිසුම්',
            'giwisuma': 'ගිවිසුම්',
            'performance': 'කාර්යසාධන',
            'kaaryasadhana': 'කාර්යසාධන',
            'obligation': 'බැඳීම්',
            'baendeem': 'බැඳීම්',
            'revaluation': 'ප්‍රතයාගණන',
            'prathyaagana': 'ප්‍රතයාගණන',
            'right': 'අයිතිය',
            'ayithiya': 'අයිතිය',
            'use': 'භාවිත',
            'bhaawitha': 'භාවිත',
            'property': 'දේපළ',
            'depala': 'දේපළ',
            'plant': 'පිරියත',
            'piriyatha': 'පිරියත',
            'equipment': 'උපකරණ',
            'upakarana': 'උපකරණ',
            'contingent': 'අසම්භාව්‍ය',
            'asambhaawya': 'අසම්භාව්‍ය',
            
            # Common question words
            'what': 'මොකක්ද',
            'mokakda': 'මොකක්ද',
            'how': 'කොහොමද',
            'kohomada': 'කොහොමද',
            'when': 'කවදාද',
            'kawadaada': 'කවදාද',
            'where': 'කොහේද',
            'koheda': 'කොහේද',
            'why': 'ඇයි',
            'ayee': 'ඇයි',
            'which': 'කුමන',
            'kumana': 'කුමන',
            
            # LKAS/SLFRS references
            'lkas': 'ශ්‍රී ලංකා ගිණුම්කරණ ප්‍රමිතිය',
            'slfrs': 'ශ්‍රී ලංකා මූල්‍ය වාර්තාකරණ ප්‍රමිතිය',
        }
        
        # Common Sinhala phrases in English transliteration
        self.phrase_map = {
            'ginumkaranaya kohomada': 'ගිණුම්කරණය කොහොමද',
            'moolya prakashana': 'මූල්‍ය ප්‍රකාශන',
            'wathkam saha wagakeem': 'වත්කම් සහ වගකීම්',
            'labha alaba prakashana': 'ලාභ අලාභ ප්‍රකාශන',
            'mudal prawaha': 'මුදල් ප්‍රවාහ',
            'himikam wenas weema': 'හිමිකම වෙනස් වීම',
            'thoga watinaakama': 'තොග වටිනාකම',
            'depala piriyatha upakarana': 'දේපළ පිරියත උපකරණ',
            'kshaya kireema': 'ක්ෂය කිරීම',
            'prathipaadana haduna gaaneema': 'ප්‍රතිපාදන හඳුනා ගැනීම',
        }
    
    def process_query(self, query: str) -> str:
        """Process user query and convert English-typed Sinhala to proper Sinhala"""
        if not query:
            return query
        
        # Convert to lowercase for matching
        query_lower = query.lower()
        
        # Check if query is already in Sinhala
        if self._is_sinhala_text(query):
            return query
        
        # Check for phrase matches first
        for eng_phrase, sin_phrase in self.phrase_map.items():
            if eng_phrase in query_lower:
                query_lower = query_lower.replace(eng_phrase, sin_phrase)
        
        # Replace individual words
        words = query_lower.split()
        translated_words = []
        
        for word in words:
            # Remove punctuation for matching
            clean_word = re.sub(r'[^\w]', '', word)
            
            if clean_word in self.transliteration_map:
                translated_words.append(self.transliteration_map[clean_word])
            else:
                # Check for partial matches
                translated = self._find_partial_match(clean_word)
                translated_words.append(translated if translated else word)
        
        result = ' '.join(translated_words)
        
        # If significant translation occurred, return translated version
        if self._translation_score(query, result) > 0.3:
            logger.info(f"Translated query: '{query}' -> '{result}'")
            return result
        
        # Otherwise return original
        return query
    
    def _is_sinhala_text(self, text: str) -> bool:
        """Check if text contains significant Sinhala characters"""
        sinhala_chars = len(re.findall(r'[\u0D80-\u0DFF]', text))
        total_chars = len(re.findall(r'[\w]', text))
        
        return total_chars > 0 and (sinhala_chars / total_chars) > 0.5
    
    def _find_partial_match(self, word: str) -> str:
        """Find partial matches for transliteration"""
        for eng_word, sin_word in self.transliteration_map.items():
            if word in eng_word or eng_word in word:
                if len(word) > 3:  # Only for longer words
                    return sin_word
        return word
    
    def _translation_score(self, original: str, translated: str) -> float:
        """Calculate how much of the text was translated"""
        if original == translated:
            return 0.0
        
        original_words = set(original.lower().split())
        translated_sinhala_chars = len(re.findall(r'[\u0D80-\u0DFF]', translated))
        total_chars = len(translated)
        
        if total_chars == 0:
            return 0.0
        
        return translated_sinhala_chars / total_chars
    
    def get_common_terms(self) -> Dict[str, str]:
        """Get dictionary of common accounting terms"""
        return self.transliteration_map.copy()
    
    def add_custom_mapping(self, english: str, sinhala: str):
        """Add custom transliteration mapping"""
        self.transliteration_map[english.lower()] = sinhala
        logger.info(f"Added custom mapping: {english} -> {sinhala}")
    
    def suggest_sinhala_terms(self, partial_query: str) -> List[str]:
        """Suggest Sinhala terms based on partial English input"""
        suggestions = []
        partial_lower = partial_query.lower()
        
        for eng_term, sin_term in self.transliteration_map.items():
            if partial_lower in eng_term or eng_term.startswith(partial_lower):
                suggestions.append(sin_term)
        
        return suggestions[:5]  # Return top 5 suggestions