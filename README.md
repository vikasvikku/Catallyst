# Synopsis Scorer App

A privacy-conscious Gen AI-powered application that evaluates the quality of a synopsis based on an uploaded article.

## Features

- Upload article (.txt or .pdf) and synopsis (.txt) files
- Get a comprehensive score out of 100
- Receive qualitative feedback on synopsis quality
- Privacy-first approach with text anonymization
- Multiple scoring options: Local NLP, Groq (llama3-70b), or Gemini (gemini-2.0-flash)
- Visual score breakdown

## Privacy Protection

This application prioritizes privacy:
- Named Entity Recognition to identify and replace sensitive information
- Pattern matching to detect and mask emails and phone numbers
- No data storage after processing
- Only anonymized content is sent to external APIs

## Scoring Methodology

The synopsis is evaluated on three key dimensions:
- **Content Coverage (50 points)**: How well the synopsis captures main ideas and key points
- **Clarity (25 points)**: How clear, concise, and well-organized the synopsis is
- **Coherence (25 points)**: How well the synopsis flows logically and maintains relationships between ideas

## Setup Instructions

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
3. Create a `.env` file with your API keys (optional):
   ```
   GROQ_API_KEY=your_groq_api_key
   GEMINI_API_KEY=your_gemini_api_key
   ```
4. Run the application:
   ```
   streamlit run app.py
   ```

## Technologies Used

- Streamlit for the web interface
- spaCy for NLP and anonymization
- PyPDF2 for PDF text extraction
- Groq and Google Generative AI for advanced scoring
- Scikit-learn for NLP-based scoring
- Matplotlib for visualization
