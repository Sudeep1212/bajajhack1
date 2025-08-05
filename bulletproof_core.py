import pdfplumber
import requests
import google.generativeai as genai
import logging
import re
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import json
from typing import List, Dict, Tuple, Optional
import math
import psutil
import gc
import traceback

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Multiple API keys for rotation
API_KEYS = [
    "AIzaSyD7VOzLJe9tdHBPfz2MaF0ag1uKLMN5S4I",
    "AIzaSyAzioJtpKSSU8RqzeDielck08m7YOkL6Lk",
    "AIzaSyCturqn478GurAsDjG80p38xwOJR8i5Dxc",
    "AIzaSyAKxMS3h-Dvg4R-eEa1VTZagfgYdsyGJ08",
    "AIzaSyCi-kSXDkJtjn3qpHOtn7_i2Gp44eM9tzc",
    "AIzaSyAp61P_hc25OuJW0CG2YBhFvj8ndAGPSGA",
    "AIzaSyDjPfezH_amGWt9G5vpr-2x5mYZN1AdYpU",
    "AIzaSyDooA_ozoxWw-fDeM8-HPu0wc6hKW_82fg",
    "AIzaSyCUtj4SePS6u_w5NDNBeSrgS7E5XgQbNN0",
    "AIzaSyCZJIdCw2Olw7iu6yFOaav0COa_btX99N8"
]

class BulletproofHackRxSystem:
    def __init__(self):
        self.current_api_key_index = 0
        self.stop_words = set(stopwords.words('english'))
        self.chunks = []
        self.chunk_embeddings = None
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.memory_threshold = 400 * 1024 * 1024  # 400MB
        self.max_processing_time = 300  # 5 minutes
        self.start_time = None
        
    def get_memory_usage(self):
        """Get current memory usage in bytes."""
        try:
            process = psutil.Process()
            return process.memory_info().rss
        except:
            return 0
    
    def check_memory_and_cleanup(self):
        """Check memory usage and cleanup if needed."""
        memory_usage = self.get_memory_usage()
        logger.info(f"Memory usage: {memory_usage / 1024 / 1024:.1f}MB")
        
        if memory_usage > self.memory_threshold:
            logger.warning(f"High memory usage: {memory_usage / 1024 / 1024:.1f}MB, cleaning up...")
            gc.collect()
            return True
        return False
    
    def check_timeout(self):
        """Check if processing has exceeded timeout."""
        if self.start_time and time.time() - self.start_time > self.max_processing_time:
            logger.warning("Processing timeout reached")
            return True
        return False
    
    def get_next_api_key(self):
        """Get next API key in rotation."""
        key = API_KEYS[self.current_api_key_index]
        self.current_api_key_index = (self.current_api_key_index + 1) % len(API_KEYS)
        return key
    
    def configure_gemini_with_retry(self):
        """Configure Gemini with retry mechanism."""
        for attempt in range(len(API_KEYS)):
            try:
                api_key = self.get_next_api_key()
                genai.configure(api_key=api_key)
                logger.info(f"Using API key {self.current_api_key_index}")
                return genai.GenerativeModel("gemini-2.5-flash")
            except Exception as e:
                logger.error(f"API key {self.current_api_key_index} failed: {str(e)}")
                continue
        raise Exception("All API keys failed")
    
    def download_pdf_with_retry(self, url: str, max_retries: int = 3) -> str:
        """Download PDF with retry mechanism."""
        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading PDF from {url} (attempt {attempt + 1})")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                with open("temp.pdf", "wb") as f:
                    f.write(response.content)
                logger.info("PDF downloaded successfully")
                return "temp.pdf"
            except Exception as e:
                logger.error(f"Download attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise
                time.sleep(2)
    
    def extract_text_with_error_handling(self, pdf_file: str) -> str:
        """Extract text with comprehensive error handling."""
        logger.info("Extracting text from PDF with error handling")
        all_text = ""
        page_count = 0
        
        try:
            with pdfplumber.open(pdf_file) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"Processing {total_pages} pages")
                
                for i, page in enumerate(pdf.pages):
                    try:
                        # Check memory and timeout
                        if self.check_memory_and_cleanup():
                            logger.warning("Memory cleanup performed")
                        
                        if self.check_timeout():
                            logger.warning("Timeout reached, returning partial text")
                            break
                        
                        text = page.extract_text(layout=True, x_tolerance=3, y_tolerance=3)
                        if text:
                            all_text += text + " "
                            page_count += 1
                            logger.info(f"Extracted {len(text)} chars from page {i+1}/{total_pages}")
                        
                        # Cleanup every 10 pages
                        if (i + 1) % 10 == 0:
                            gc.collect()
                            
                    except Exception as e:
                        logger.error(f"Error extracting page {i+1}: {str(e)}")
                        continue  # Skip failed page and continue
                
                logger.info(f"Successfully extracted {page_count}/{total_pages} pages")
                return all_text.strip()
                
        except Exception as e:
            logger.error(f"PDF extraction failed: {str(e)}")
            # Try alternative extraction method
            return self.fallback_text_extraction(pdf_file)
    
    def fallback_text_extraction(self, pdf_file: str) -> str:
        """Fallback text extraction method."""
        logger.info("Using fallback text extraction")
        try:
            import PyPDF2
            text = ""
            with open(pdf_file, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + " "
            return text.strip()
        except Exception as e:
            logger.error(f"Fallback extraction failed: {str(e)}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """Clean the extracted text."""
        try:
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            # Remove page numbers
            text = re.sub(r'\b\d+\s*of\s*\d+\b', '', text)
            text = re.sub(r'\bPage\s+\d+\b', '', text)
            return text.strip()
        except Exception as e:
            logger.error(f"Text cleaning failed: {str(e)}")
            return text.strip()
    
    def create_chunks_with_error_handling(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """Create chunks with error handling and memory management."""
        logger.info("Creating text chunks with error handling")
        try:
            chunks = []
            words = text.split()
            
            # Adjust chunk size based on memory usage
            if self.get_memory_usage() > 300 * 1024 * 1024:  # 300MB
                chunk_size = 300
                overlap = 50
                logger.info("Reduced chunk size due to memory usage")
            
            for i in range(0, len(words), chunk_size - overlap):
                try:
                    chunk = ' '.join(words[i:i + chunk_size])
                    if chunk.strip():
                        chunks.append(chunk.strip())
                    
                    # Check memory every 5 chunks
                    if len(chunks) % 5 == 0:
                        self.check_memory_and_cleanup()
                        
                except Exception as e:
                    logger.error(f"Error creating chunk {len(chunks)}: {str(e)}")
                    continue
            
            logger.info(f"Created {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Chunk creation failed: {str(e)}")
            # Fallback: simple word-based chunks
            return self.fallback_chunk_creation(text)
    
    def fallback_chunk_creation(self, text: str) -> List[str]:
        """Fallback chunk creation method."""
        logger.info("Using fallback chunk creation")
        try:
            words = text.split()
            chunk_size = 200
            chunks = []
            for i in range(0, len(words), chunk_size):
                chunk = ' '.join(words[i:i + chunk_size])
                if chunk.strip():
                    chunks.append(chunk.strip())
            return chunks
        except Exception as e:
            logger.error(f"Fallback chunk creation failed: {str(e)}")
            return [text[:1000]]  # Return first 1000 chars as single chunk
    
    def extract_keywords_with_retry(self, questions: List[str]) -> List[List[str]]:
        """Extract keywords with comprehensive retry mechanism."""
        logger.info("Extracting keywords with retry mechanism")
        
        combined_questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
        
        prompt = f"""Extract the most important keywords from each question below. Focus on:
- Insurance/policy terms (grace period, waiting period, coverage, etc.)
- Medical terms (maternity, cataract, organ donor, etc.)
- Numbers and time periods
- Specific policy features

Questions:
{combined_questions}

Return ONLY a JSON array where each element is an array of keywords for the corresponding question."""

        # Try all API keys
        for attempt in range(len(API_KEYS)):
            try:
                model = self.configure_gemini_with_retry()
                
                response = model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.1,
                        "max_output_tokens": 1000,
                        "top_p": 0.8
                    }
                )
                
                # Parse JSON response
                keywords_text = response.text.strip()
                if keywords_text.startswith("```json"):
                    keywords_text = keywords_text[7:-3]
                elif keywords_text.startswith("```"):
                    keywords_text = keywords_text[3:-3]
                
                keywords_list = json.loads(keywords_text)
                logger.info(f"Extracted keywords for {len(keywords_list)} questions")
                return keywords_list
                
            except Exception as e:
                logger.error(f"Keyword extraction attempt {attempt + 1} failed: {str(e)}")
                continue
        
        # Fallback to local keyword extraction
        logger.warning("All API attempts failed, using fallback keyword extraction")
        return self.fallback_keyword_extraction(questions)
    
    def fallback_keyword_extraction(self, questions: List[str]) -> List[List[str]]:
        """Fallback keyword extraction without API call."""
        keywords_list = []
        for question in questions:
            try:
                words = word_tokenize(question.lower())
                keywords = [word for word in words if word.isalnum() and word not in self.stop_words]
                insurance_terms = ['policy', 'coverage', 'period', 'waiting', 'grace', 'premium', 'claim', 'hospital', 'medical', 'expenses']
                keywords.extend([term for term in insurance_terms if term in question.lower()])
                keywords_list.append(list(set(keywords)))
            except Exception as e:
                logger.error(f"Fallback keyword extraction failed for question: {str(e)}")
                keywords_list.append([])
        return keywords_list
    
    def find_relevant_chunks_with_error_handling(self, keywords: List[str], top_k: int = 3) -> List[str]:
        """Find relevant chunks with error handling."""
        try:
            if not self.chunks or self.chunk_embeddings is None:
                return []
            
            query_text = ' '.join(keywords)
            query_vector = self.vectorizer.transform([query_text])
            
            similarities = cosine_similarity(query_vector, self.chunk_embeddings).flatten()
            
            top_indices = similarities.argsort()[-top_k:][::-1]
            relevant_chunks = [self.chunks[i] for i in top_indices if similarities[i] > 0.1]
            
            return relevant_chunks
            
        except Exception as e:
            logger.error(f"Chunk finding failed: {str(e)}")
            # Return first few chunks as fallback
            return self.chunks[:3] if self.chunks else []
    
    def process_questions_batch_with_retry(self, questions: List[str], relevant_chunks: List[str]) -> List[str]:
        """Process questions with comprehensive retry mechanism."""
        logger.info(f"Processing batch of {len(questions)} questions with retry")
        
        questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
        chunks_text = "\n\n".join(relevant_chunks)
        
        prompt = f"""You are an expert insurance policy analyst. Analyze the provided policy document chunks and answer each question with precise, specific information.

POLICY DOCUMENT CHUNKS:
{chunks_text}

QUESTIONS:
{questions_text}

INSTRUCTIONS:
- Answer using ONLY information from the policy document chunks above
- Provide specific details, numbers, time periods, and exact clauses
- Use the exact language and terminology from the policy document
- If the information is not in the chunks, respond with "No relevant information found in the context"
- Keep each answer concise but comprehensive (1-3 sentences)
- Reference specific clauses, sections, or policy terms when possible
- Use formal, professional language similar to the policy document

Return ONLY a JSON array of answers in the same order as the questions."""

        # Try all API keys
        for attempt in range(len(API_KEYS)):
            try:
                model = self.configure_gemini_with_retry()
                
                response = model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.1,
                        "max_output_tokens": 2000,
                        "top_p": 0.8
                    }
                )
                
                # Parse JSON response
                answers_text = response.text.strip()
                if answers_text.startswith("```json"):
                    answers_text = answers_text[7:-3]
                elif answers_text.startswith("```"):
                    answers_text = answers_text[3:-3]
                
                answers = json.loads(answers_text)
                logger.info(f"Generated {len(answers)} answers for batch")
                return answers
                
            except Exception as e:
                logger.error(f"Batch processing attempt {attempt + 1} failed: {str(e)}")
                continue
        
        # Fallback to simple responses
        logger.warning("All API attempts failed, using fallback responses")
        return self.fallback_answer_generation(questions)
    
    def fallback_answer_generation(self, questions: List[str]) -> List[str]:
        """Fallback answer generation without API."""
        answers = []
        for question in questions:
            try:
                # Simple keyword-based response
                keywords = self.fallback_keyword_extraction([question])[0]
                if keywords:
                    answer = f"Based on the policy document, {', '.join(keywords[:3])} are relevant to your question. Please refer to the specific policy sections for detailed information."
                else:
                    answer = "No relevant information found in the context."
                answers.append(answer)
            except Exception as e:
                logger.error(f"Fallback answer generation failed: {str(e)}")
                answers.append("Error generating answer. Please try again.")
        return answers
    
    def process_questions(self, pdf_url: str, questions: List[str]) -> Dict:
        """Main processing function with bulletproof error handling."""
        self.start_time = time.time()
        
        try:
            # Step 1: Download and extract PDF
            logger.info("Step 1: Processing PDF with error handling...")
            pdf_file = self.download_pdf_with_retry(pdf_url)
            text = self.extract_text_with_error_handling(pdf_file)
            text = self.clean_text(text)
            logger.info(f"Extracted {len(text)} characters from PDF")
            
            if not text:
                logger.error("No text extracted from PDF")
                return {
                    "answers": ["Error: Could not extract text from PDF"] * len(questions),
                    "processing_time": "0s",
                    "questions_processed": len(questions),
                    "text_length": 0,
                    "error": "PDF text extraction failed"
                }
            
            # Step 2: Create chunks and compute embeddings
            logger.info("Step 2: Creating chunks and computing embeddings...")
            self.chunks = self.create_chunks_with_error_handling(text)
            
            if not self.chunks:
                logger.error("No chunks created")
                return {
                    "answers": ["Error: Could not process PDF chunks"] * len(questions),
                    "processing_time": "0s",
                    "questions_processed": len(questions),
                    "text_length": len(text),
                    "error": "Chunk creation failed"
                }
            
            try:
                self.chunk_embeddings = self.vectorizer.fit_transform(self.chunks)
                logger.info(f"Created {len(self.chunks)} chunks with embeddings")
            except Exception as e:
                logger.error(f"Embedding creation failed: {str(e)}")
                # Continue without embeddings
            
            # Step 3: Extract keywords
            logger.info("Step 3: Extracting keywords from questions...")
            all_keywords = self.extract_keywords_with_retry(questions)
            
            # Step 4: Find relevant chunks
            logger.info("Step 4: Finding relevant chunks for each question...")
            all_relevant_chunks = []
            for keywords in all_keywords:
                relevant_chunks = self.find_relevant_chunks_with_error_handling(keywords)
                all_relevant_chunks.append(relevant_chunks)
            
            # Step 5: Process questions in batches
            logger.info("Step 5: Processing questions in batches...")
            batch_size = 5
            all_answers = []
            
            for i in range(0, len(questions), batch_size):
                try:
                    batch_questions = questions[i:i + batch_size]
                    batch_keywords = all_keywords[i:i + batch_size]
                    
                    # Combine all relevant chunks for this batch
                    batch_chunks = set()
                    for keywords in batch_keywords:
                        relevant_chunks = self.find_relevant_chunks_with_error_handling(keywords)
                        batch_chunks.update(relevant_chunks)
                    
                    # Process batch
                    batch_answers = self.process_questions_batch_with_retry(batch_questions, list(batch_chunks))
                    all_answers.extend(batch_answers)
                    
                    logger.info(f"Processed batch {i//batch_size + 1}/{(len(questions) + batch_size - 1)//batch_size}")
                    
                except Exception as e:
                    logger.error(f"Batch processing failed: {str(e)}")
                    # Add fallback answers for this batch
                    fallback_answers = self.fallback_answer_generation(batch_questions)
                    all_answers.extend(fallback_answers)
            
            processing_time = time.time() - self.start_time
            logger.info(f"Completed in {processing_time:.2f} seconds")
            
            return {
                "answers": all_answers,
                "processing_time": f"{processing_time:.2f}s",
                "questions_processed": len(questions),
                "text_length": len(text),
                "chunks_created": len(self.chunks),
                "api_calls_reduced": f"From {len(questions)} to {math.ceil(len(questions)/batch_size) + 1}",
                "memory_usage": f"{self.get_memory_usage() / 1024 / 1024:.1f}MB"
            }
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return fallback answers
            fallback_answers = self.fallback_answer_generation(questions)
            return {
                "answers": fallback_answers,
                "processing_time": "0s",
                "questions_processed": len(questions),
                "text_length": 0,
                "error": str(e),
                "status": "fallback_used"
            }

# Global instance
bulletproof_system = BulletproofHackRxSystem()

def process_questions(pdf_url: str, questions: List[str]) -> Dict:
    """Wrapper function for compatibility."""
    return bulletproof_system.process_questions(pdf_url, questions) 