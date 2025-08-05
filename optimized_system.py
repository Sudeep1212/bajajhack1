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
from typing import List, Dict, Tuple
import math

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

# Configure Gemini
genai.configure(api_key="AIzaSyCfPtyPVOXvbO1nUnoIlKB81xBV_SL6drQ")

class OptimizedHackRxSystem:
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        self.stop_words = set(stopwords.words('english'))
        self.chunks = []
        self.chunk_embeddings = None
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
    
    def download_pdf(self, url: str) -> str:
        """Download PDF from URL."""
        logger.info(f"Downloading PDF from {url}")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open("temp.pdf", "wb") as f:
                f.write(response.content)
            logger.info("PDF downloaded successfully")
            return "temp.pdf"
        except Exception as e:
            logger.error(f"Failed to download PDF: {str(e)}")
            raise
    
    def extract_text(self, pdf_file: str) -> str:
        """Extract text from PDF."""
        logger.info("Extracting text from PDF")
        try:
            with pdfplumber.open(pdf_file) as pdf:
                all_text = ""
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text(layout=True, x_tolerance=3, y_tolerance=3)
                    if text:
                        all_text += text + " "
                        logger.info(f"Extracted {len(text)} chars from page {i+1}")
                return all_text.strip()
        except Exception as e:
            logger.error(f"PDF extraction failed: {str(e)}")
            raise
    
    def clean_text(self, text: str) -> str:
        """Clean the extracted text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers
        text = re.sub(r'\b\d+\s*of\s*\d+\b', '', text)
        text = re.sub(r'\bPage\s+\d+\b', '', text)
        return text.strip()
    
    def create_chunks(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Create overlapping chunks from text."""
        logger.info("Creating text chunks for efficient processing")
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def extract_keywords_from_questions(self, questions: List[str]) -> List[List[str]]:
        """Extract keywords from all questions using ONE Gemini API call."""
        logger.info("Extracting keywords from questions using Gemini")
        
        # Combine all questions for single API call
        combined_questions = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
        
        prompt = f"""Extract the most important keywords from each question below. Focus on:
- Insurance/policy terms (grace period, waiting period, coverage, etc.)
- Medical terms (maternity, cataract, organ donor, etc.)
- Numbers and time periods
- Specific policy features

Questions:
{combined_questions}

Return ONLY a JSON array where each element is an array of keywords for the corresponding question. Example:
[
  ["grace period", "premium payment", "thirty days"],
  ["waiting period", "pre-existing diseases", "36 months"],
  ...
]

Keywords should be specific and relevant to finding answers in insurance documents."""

        try:
            response = self.model.generate_content(
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
            logger.error(f"Keyword extraction failed: {str(e)}")
            # Fallback: simple keyword extraction
            return self.fallback_keyword_extraction(questions)
    
    def fallback_keyword_extraction(self, questions: List[str]) -> List[List[str]]:
        """Fallback keyword extraction without API call."""
        keywords_list = []
        for question in questions:
            # Extract important words
            words = word_tokenize(question.lower())
            keywords = [word for word in words if word.isalnum() and word not in self.stop_words]
            # Add insurance-specific terms
            insurance_terms = ['policy', 'coverage', 'period', 'waiting', 'grace', 'premium', 'claim', 'hospital', 'medical', 'expenses']
            keywords.extend([term for term in insurance_terms if term in question.lower()])
            keywords_list.append(list(set(keywords)))
        return keywords_list
    
    def find_relevant_chunks(self, keywords: List[str], top_k: int = 3) -> List[str]:
        """Find most relevant chunks using cosine similarity."""
        if not self.chunks or self.chunk_embeddings is None:
            return []
        
        # Create query vector
        query_text = ' '.join(keywords)
        query_vector = self.vectorizer.transform([query_text])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.chunk_embeddings).flatten()
        
        # Get top-k most similar chunks
        top_indices = similarities.argsort()[-top_k:][::-1]
        relevant_chunks = [self.chunks[i] for i in top_indices if similarities[i] > 0.1]
        
        return relevant_chunks
    
    def process_questions_batch(self, questions: List[str], relevant_chunks: List[str]) -> List[str]:
        """Process a batch of questions with relevant chunks using Gemini."""
        logger.info(f"Processing batch of {len(questions)} questions")
        
        # Combine questions and chunks
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

Return ONLY a JSON array of answers in the same order as the questions. Example:
[
  "A grace period of thirty days is provided for premium payment...",
  "There is a waiting period of thirty-six months...",
  ...
]"""

        try:
            response = self.model.generate_content(
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
            logger.error(f"Batch processing failed: {str(e)}")
            return ["Error generating answer. Please try again."] * len(questions)
    
    def process_questions(self, pdf_url: str, questions: List[str]) -> Dict:
        """Main processing function with optimized approach."""
        start_time = time.time()
        
        try:
            # Step 1: Download and extract PDF
            logger.info("Step 1: Processing PDF...")
            pdf_file = self.download_pdf(pdf_url)
            text = self.extract_text(pdf_file)
            text = self.clean_text(text)
            logger.info(f"Extracted {len(text)} characters from PDF")
            
            # Step 2: Create chunks and compute embeddings
            logger.info("Step 2: Creating chunks and computing embeddings...")
            self.chunks = self.create_chunks(text)
            if self.chunks:
                self.chunk_embeddings = self.vectorizer.fit_transform(self.chunks)
            logger.info(f"Created {len(self.chunks)} chunks with embeddings")
            
            # Step 3: Extract keywords from all questions (ONE API call)
            logger.info("Step 3: Extracting keywords from questions...")
            all_keywords = self.extract_keywords_from_questions(questions)
            
            # Step 4: Find relevant chunks for each question
            logger.info("Step 4: Finding relevant chunks for each question...")
            all_relevant_chunks = []
            for keywords in all_keywords:
                relevant_chunks = self.find_relevant_chunks(keywords)
                all_relevant_chunks.append(relevant_chunks)
            
            # Step 5: Process questions in batches of 5
            logger.info("Step 5: Processing questions in batches...")
            batch_size = 5
            all_answers = []
            
            for i in range(0, len(questions), batch_size):
                batch_questions = questions[i:i + batch_size]
                batch_keywords = all_keywords[i:i + batch_size]
                
                # Combine all relevant chunks for this batch
                batch_chunks = set()
                for keywords in batch_keywords:
                    relevant_chunks = self.find_relevant_chunks(keywords)
                    batch_chunks.update(relevant_chunks)
                
                # Process batch
                batch_answers = self.process_questions_batch(batch_questions, list(batch_chunks))
                all_answers.extend(batch_answers)
                
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(questions) + batch_size - 1)//batch_size}")
            
            processing_time = time.time() - start_time
            logger.info(f"Completed in {processing_time:.2f} seconds")
            
            return {
                "answers": all_answers,
                "processing_time": f"{processing_time:.2f}s",
                "questions_processed": len(questions),
                "text_length": len(text),
                "chunks_created": len(self.chunks),
                "api_calls_reduced": f"From {len(questions)} to {math.ceil(len(questions)/batch_size) + 1}"
            }
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            raise

# Global instance
optimized_system = OptimizedHackRxSystem()

def process_questions(pdf_url: str, questions: List[str]) -> Dict:
    """Wrapper function for compatibility."""
    return optimized_system.process_questions(pdf_url, questions)

# Test function
def test_optimized_system():
    """Test the optimized system."""
    pdf_url = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
    
    questions = [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
    
    result = process_questions(pdf_url, questions)
    
    print("\n" + "="*60)
    print("OPTIMIZED HACKRX 6.0 SYSTEM RESULTS:")
    print("="*60)
    
    for i, (question, answer) in enumerate(zip(questions, result['answers']), 1):
        print(f"\n{i}. Question: {question}")
        print(f"   Answer: {answer}")
        print(f"   Length: {len(answer)} chars")
        
        # Quality check
        is_good = len(answer) > 50 and "No relevant information" not in answer
        has_specifics = any(word in answer.lower() for word in ['days', 'months', 'years', 'percent', '%', 'clause', 'section', 'policy', 'coverage', 'limit'])
        status = "✅ Excellent" if is_good and has_specifics else "✅ Good" if is_good else "❌ Needs improvement"
        print(f"   Status: {status}")
    
    print(f"\n📊 Performance:")
    print(f"   Processing Time: {result['processing_time']}")
    print(f"   Text Length: {result['text_length']} chars")
    print(f"   Chunks Created: {result['chunks_created']}")
    print(f"   API Calls: {result['api_calls_reduced']}")

if __name__ == "__main__":
    test_optimized_system() 