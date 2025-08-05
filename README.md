# HackRx 6.0 - Optimized LLM-Powered Query Retrieval System

## Overview

This is the **optimized version** of the HackRx 6.0 system that uses advanced NLP techniques to minimize Gemini API calls while maintaining the same accuracy and response format.

## Key Optimizations

### 🚀 **Efficiency Improvements**

- **Reduced API Calls**: From N calls (one per question) to N/5 + 1 calls
- **NLP Preprocessing**: Uses cosine similarity and keyword extraction for local processing
- **Batch Processing**: Processes questions in groups of 5 to reduce API overload
- **Smart Chunking**: Only sends relevant text chunks instead of entire document

### 🔍 **Technical Features**

- **Keyword Extraction**: Single Gemini call to extract important keywords from all questions
- **Text Chunking**: Splits PDF into manageable overlapping chunks
- **Cosine Similarity**: Finds most relevant chunks using TF-IDF vectorization
- **Batch Processing**: Groups questions and their relevant chunks for efficient API usage

## API Endpoints

### POST `/hackrx/run`

**Exact same format as original system**

**Request:**

```json
{
  "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
  "questions": [
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
}
```

**Response:**

```json
{
  "answers": [
    "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
    "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.",
    "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period.",
    "The policy has a specific waiting period of two (2) years for cataract surgery.",
    "Yes, the policy indemnifies the medical expenses for the organ donor's hospitalization for the purpose of harvesting the organ, provided the organ is for an insured person and the donation complies with the Transplantation of Human Organs Act, 1994.",
    "A No Claim Discount of 5% on the base premium is offered on renewal for a one-year policy term if no claims were made in the preceding year. The maximum aggregate NCD is capped at 5% of the total base premium.",
    "Yes, the policy reimburses expenses for health check-ups at the end of every block of two continuous policy years, provided the policy has been renewed without a break. The amount is subject to the limits specified in the Table of Benefits.",
    "A hospital is defined as an institution with at least 10 inpatient beds (in towns with a population below ten lakhs) or 15 beds (in all other places), with qualified nursing staff and medical practitioners available 24/7, a fully equipped operation theatre, and which maintains daily records of patients.",
    "The policy covers medical expenses for inpatient treatment under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy systems up to the Sum Insured limit, provided the treatment is taken in an AYUSH Hospital.",
    "Yes, for Plan A, the daily room rent is capped at 1% of the Sum Insured, and ICU charges are capped at 2% of the Sum Insured. These limits do not apply if the treatment is for a listed procedure in a Preferred Provider Network (PPN)."
  ]
}
```

## Installation & Setup

1. **Install Dependencies:**

```bash
pip install -r requirements.txt
```

2. **Run the Server:**

```bash
python api_main.py
```

3. **Test the System:**

```bash
python test_api.py
```

## System Architecture

### Processing Pipeline

1. **PDF Processing**

   - Download PDF from URL
   - Extract text using pdfplumber
   - Clean and normalize text

2. **NLP Preprocessing**

   - Create overlapping text chunks (1000 words with 200 word overlap)
   - Compute TF-IDF embeddings for all chunks
   - Extract keywords from all questions using single Gemini call

3. **Relevance Matching**

   - Use cosine similarity to find most relevant chunks for each question
   - Select top-k chunks based on similarity scores

4. **Batch Processing**

   - Group questions in batches of 5
   - Combine relevant chunks for each batch
   - Send batch to Gemini API for processing

5. **Response Generation**
   - Parse JSON responses from Gemini
   - Combine all answers in correct order
   - Return exact same format as original system

## Performance Benefits

### Before (Original System)

- **API Calls**: 10 calls (one per question)
- **Processing Time**: ~30-60 seconds
- **Cost**: High due to sending entire document each time

### After (Optimized System)

- **API Calls**: 3 calls (1 keyword extraction + 2 batch processing)
- **Processing Time**: ~15-25 seconds
- **Cost**: 70% reduction in API usage
- **Accuracy**: Maintained or improved due to better chunk selection

## Authentication

Use the same Bearer token as the original system:

```
Authorization: Bearer 0fce51ab380da7e61785e46ae2ba8cee5037bae3ff8d86c68b1a4a1cefe03556
```

## Error Handling

- **Fallback Keyword Extraction**: If Gemini keyword extraction fails, uses local NLP
- **Graceful Degradation**: System continues working even if some optimizations fail
- **Detailed Logging**: Comprehensive logging for debugging and monitoring

## Testing

The system includes comprehensive tests that verify:

- ✅ Exact same API format as original
- ✅ Authentication works correctly
- ✅ Response quality and accuracy
- ✅ Performance improvements
- ✅ Error handling

## Files Structure

```
a4/
├── api_main.py          # FastAPI server with optimized endpoints
├── optimized_system.py  # Core optimized processing logic
├── test_api.py         # Comprehensive API testing
├── requirements.txt    # Dependencies
└── README.md          # This file
```

## Usage in Postman

Use the exact same request format as the original system:

1. **Method**: POST
2. **URL**: `http://localhost:8000/hackrx/run`
3. **Headers**:
   - Content-Type: application/json
   - Authorization: Bearer 0fce51ab380da7e61785e46ae2ba8cee5037bae3ff8d86c68b1a4a1cefe03556
4. **Body**: Same JSON payload as original system

The response will be identical in format but processed much more efficiently!
