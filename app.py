import streamlit as st
import os
import re
import tempfile
import uuid
from pathlib import Path
import PyPDF2
import google.generativeai as genai
from groq import Groq
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Load NLP model for anonymization
try:
    nlp = spacy.load("en_core_web_sm")
except:
    st.warning("Downloading NLP model for anonymization...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Set page configuration
st.set_page_config(
    page_title="Synopsis Scorer", 
    page_icon="📝", 
    layout="wide"
)

# Custom CSS styling
st.markdown("""
<style>
.main {
    padding: 2rem;
    max-width: 1200px;
    margin: 0 auto;
}
.stButton button {
    background-color: #4CAF50;
    color: white;
    font-weight: bold;
    padding: 0.5rem 1rem;
    border-radius: 5px;
}
.feedback-box {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 5px;
    border-left: 5px solid #4CAF50;
    margin: 1rem 0;
}
.score-display {
    font-size: 2rem;
    text-align: center;
    font-weight: bold;
    margin: 1rem 0;
}
.privacy-note {
    font-style: italic;
    font-size: 0.8rem;
    color: #6c757d;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'article_text' not in st.session_state:
    st.session_state.article_text = ""
if 'synopsis_text' not in st.session_state:
    st.session_state.synopsis_text = ""

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

# Function to read text from uploaded file
def read_text_file(file):
    try:
        content = file.read().decode('utf-8')
        return content
    except Exception as e:
        st.error(f"Error reading text file: {e}")
        return ""

# Function to anonymize text
def anonymize_text(text):
    doc = nlp(text)
    
    # Replace named entities with generic placeholders
    anonymized_text = text
    entities_map = {}
    
    for ent in doc.ents:
        if ent.text not in entities_map:
            if ent.label_ == "PERSON":
                placeholder = f"[PERSON_{len([e for e in entities_map.values() if 'PERSON' in e])+1}]"
            elif ent.label_ in ["ORG", "GPE", "LOC"]:
                placeholder = f"[ORGANIZATION_{len([e for e in entities_map.values() if 'ORGANIZATION' in e])+1}]"
            elif ent.label_ == "DATE":
                placeholder = f"[DATE_{len([e for e in entities_map.values() if 'DATE' in e])+1}]"
            else:
                placeholder = f"[ENTITY_{len([e for e in entities_map.values() if 'ENTITY' in e])+1}]"
            entities_map[ent.text] = placeholder
    
    # Sort entities by length (descending) to replace longer phrases first
    for entity, placeholder in sorted(entities_map.items(), key=lambda x: len(x[0]), reverse=True):
        anonymized_text = anonymized_text.replace(entity, placeholder)
    
    # Handle email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    anonymized_text = re.sub(email_pattern, '[EMAIL]', anonymized_text)
    
    # Handle phone numbers
    phone_pattern = r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
    anonymized_text = re.sub(phone_pattern, '[PHONE]', anonymized_text)
    
    return anonymized_text

# Function to calculate NLP-based score
def calculate_nlp_score(article_text, synopsis_text):
    # Process texts
    article_doc = nlp(article_text)
    synopsis_doc = nlp(synopsis_text)
    
    # Extract non-stopwords
    article_words = [token.text.lower() for token in article_doc if not token.is_stop and token.is_alpha]
    synopsis_words = [token.text.lower() for token in synopsis_doc if not token.is_stop and token.is_alpha]
    
    # Calculate keyword coverage
    article_unique = set(article_words)
    synopsis_unique = set(synopsis_words)
    common_keywords = article_unique.intersection(synopsis_unique)
    keyword_coverage = len(common_keywords) / len(article_unique) if article_unique else 0
    
    # Calculate cosine similarity
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([article_text, synopsis_text])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except:
        cosine_sim = 0
    
    # Length ratio (penalize if synopsis is too short or too long compared to article)
    ideal_ratio = 0.3  # synopsis should be ~30% of article length
    actual_ratio = len(synopsis_text) / len(article_text) if len(article_text) > 0 else 0
    length_score = 1 - min(abs(actual_ratio - ideal_ratio) / ideal_ratio, 1)
    
    # Calculate sentence structure similarity
    article_sent_lengths = [len(sent) for sent in article_doc.sents]
    synopsis_sent_lengths = [len(sent) for sent in synopsis_doc.sents]
    
    avg_article_sent_len = np.mean(article_sent_lengths) if article_sent_lengths else 0
    avg_synopsis_sent_len = np.mean(synopsis_sent_lengths) if synopsis_sent_lengths else 0
    
    sent_structure_sim = 1 - min(abs(avg_synopsis_sent_len - avg_article_sent_len) / avg_article_sent_len, 1) if avg_article_sent_len > 0 else 0
    
    # Calculate final score components
    content_coverage = (keyword_coverage * 0.7 + cosine_sim * 0.3) * 50  
    clarity_score = sent_structure_sim * 25 
    coherence_score = length_score * 25  
    
    
    content_coverage = round(content_coverage, 1)
    clarity_score = round(clarity_score, 1)
    coherence_score = round(coherence_score, 1)
    
    total_score = content_coverage + clarity_score + coherence_score
    total_score = min(max(total_score, 0), 100)  # Ensure between 0-100
    
    return {
        "total_score": round(total_score, 1),
        "content_coverage": content_coverage,
        "clarity": clarity_score,
        "coherence": coherence_score
    }


def get_groq_score(anonymized_article, anonymized_synopsis, api_key):
    try:
        client = Groq(api_key=api_key)
        
        prompt = f"""
        You are a professional synopsis evaluator. I will provide you with an original article and a synopsis of that article.
        Your task is to evaluate how well the synopsis captures the essence of the article.

        Original Article:
        {anonymized_article}

        Synopsis:
        {anonymized_synopsis}

        Evaluate the synopsis based on these criteria:
        1. Content Coverage (50 points): How well does the synopsis capture the main ideas, key points, and important details of the original article?
        2. Clarity (25 points): How clear, concise, and well-organized is the synopsis?
        3. Coherence (25 points): How well does the synopsis flow logically and maintain the relationships between ideas?

        Provide your evaluation in the following JSON format:
        {{
            "content_coverage": [score out of 50],
            "clarity": [score out of 25],
            "coherence": [score out of 25],
            "total_score": [sum of all scores],
            "feedback": [2-3 sentences of qualitative feedback on the synopsis]
        }}
        
        Your response should ONLY include the JSON object, nothing else.
        """
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a professional synopsis evaluator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        result = response.choices[0].message.content
        # Try to extract JSON from the response
        result = result.strip()
        if result.startswith("```json"):
            result = result[7:]
        if result.endswith("```"):
            result = result[:-3]
        
        import json
        try:
            result_json = json.loads(result)
            return result_json
        except:
            st.error("Failed to parse JSON response from Groq API")
            return {
                "content_coverage": 0,
                "clarity": 0,
                "coherence": 0,
                "total_score": 0,
                "feedback": "Error evaluating synopsis. Please try again."
            }
            
    except Exception as e:
        st.error(f"Error with Groq API: {e}")
        return {
            "content_coverage": 0,
            "clarity": 0,
            "coherence": 0, 
            "total_score": 0,
            "feedback": "Error evaluating synopsis. Please try again."
        }


def get_gemini_score(anonymized_article, anonymized_synopsis, api_key):
    try:
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        prompt = f"""
        You are a professional synopsis evaluator. I will provide you with an original article and a synopsis of that article.
        Your task is to evaluate how well the synopsis captures the essence of the article.

        Original Article:
        {anonymized_article}

        Synopsis:
        {anonymized_synopsis}

        Evaluate the synopsis based on these criteria:
        1. Content Coverage (50 points): How well does the synopsis capture the main ideas, key points, and important details of the original article?
        2. Clarity (25 points): How clear, concise, and well-organized is the synopsis?
        3. Coherence (25 points): How well does the synopsis flow logically and maintain the relationships between ideas?

        Provide your evaluation in the following JSON format:
        {{
            "content_coverage": [score out of 50],
            "clarity": [score out of 25],
            "coherence": [score out of 25],
            "total_score": [sum of all scores],
            "feedback": [2-3 sentences of qualitative feedback on the synopsis]
        }}
        
        Your response should ONLY include the JSON object, nothing else.
        """
        
        response = model.generate_content(prompt)
        result = response.text
        
        # Try to extract JSON from the response
        result = result.strip()
        if result.startswith("```json"):
            result = result[7:]
        if result.endswith("```"):
            result = result[:-3]
        
        import json
        try:
            result_json = json.loads(result)
            return result_json
        except:
            st.error("Failed to parse JSON response from Gemini API")
            return {
                "content_coverage": 0,
                "clarity": 0,
                "coherence": 0,
                "total_score": 0,
                "feedback": "Error evaluating synopsis. Please try again."
            }
            
    except Exception as e:
        st.error(f"Error with Gemini API: {e}")
        return {
            "content_coverage": 0,
            "clarity": 0,
            "coherence": 0,
            "total_score": 0,
            "feedback": "Error evaluating synopsis. Please try again."
        }

# Function to create radar chart for score components
def create_score_chart(scores):
    categories = ['Content Coverage', 'Clarity', 'Coherence']
    normalized_scores = [
        scores['content_coverage'] / 50 * 100,  
        scores['clarity'] / 25 * 100,
        scores['coherence'] / 25 * 100
    ]
    
    # Number of variables
    N = len(categories)
    
    # What will be the angle of each axis in the plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Normalize scores to 0-1 for the chart
    normalized_scores_for_chart = [s / 100 for s in normalized_scores]
    normalized_scores_for_chart += normalized_scores_for_chart[:1]  # Close the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    # Draw one axis per variable and add labels
    plt.xticks(angles[:-1], categories, size=12)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["25%", "50%", "75%"], size=10)
    plt.ylim(0, 1)
    
    # Plot data
    ax.plot(angles, normalized_scores_for_chart, linewidth=2, linestyle='solid')
    
    # Fill area
    ax.fill(angles, normalized_scores_for_chart, alpha=0.1)
    
    # Add title
    plt.title("Synopsis Score Breakdown", size=16, pad=20)
    
    return fig

# Main application layout
st.title("Synopsis Scorer App")
st.subheader("Evaluate the quality of your article synopsis with AI")

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["Upload Files", "Evaluation Results", "Privacy Information"])

with tab1:
    # File upload section
    st.header("Upload Your Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Original Article")
        article_file = st.file_uploader("Choose a .txt or .pdf file", type=["txt", "pdf"], key="article_upload")
        
    with col2:
        st.subheader("Upload Synopsis")
        synopsis_file = st.file_uploader("Choose a .txt file", type=["txt"], key="synopsis_upload")
    
    # API/Model selection
    st.subheader("Select Model")
    api_provider = st.selectbox(
        "Choose model for evaluation:",
        ["Groq (llama-3.3-70b-versatile)", "Gemini (gemini-2.0-flash)", "Local NLP Only"]
    )
    
    # Get API keys from environment variables only - no frontend input
    groq_api_key = os.getenv("GROQ_API_KEY", "")
    gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    
    # Submit button
    submit_col1, submit_col2, submit_col3 = st.columns([1, 2, 1])
    with submit_col2:
        submit_button = st.button("Evaluate Synopsis", use_container_width=True)

# Handle file processing and scoring
if submit_button:
    if not article_file:
        st.error("Please upload an article file")
    elif not synopsis_file:
        st.error("Please upload a synopsis file")
    elif api_provider == "Groq (llama3-70b)" and not groq_api_key:
        st.error("Missing Groq API key in environment variables. Please add it to your .env file.")
    elif api_provider == "Gemini (gemini-2.0-flash)" and not gemini_api_key:
        st.error("Missing Gemini API key in environment variables. Please add it to your .env file.")
    else:
        with st.spinner("Processing files and evaluating synopsis..."):
            # Extract text from article
            if article_file.name.endswith('.pdf'):
                article_text = extract_text_from_pdf(article_file)
            else:
                article_text = read_text_file(article_file)
            
            # Extract text from synopsis
            synopsis_text = read_text_file(synopsis_file)
            
            if not article_text or not synopsis_text:
                st.error("Error processing files. Please check file formats and try again.")
            else:
                st.session_state.article_text = article_text
                st.session_state.synopsis_text = synopsis_text
                
                # Calculate local NLP score
                nlp_scores = calculate_nlp_score(article_text, synopsis_text)
                
                # Anonymize texts before sending to external APIs
                anonymized_article = anonymize_text(article_text)
                anonymized_synopsis = anonymize_text(synopsis_text)
                
                # Get AI score based on provider
                ai_scores = None
                if api_provider == "Groq (llama3-70b)":
                    ai_scores = get_groq_score(anonymized_article, anonymized_synopsis, groq_api_key)
                elif api_provider == "Gemini (1.5 Pro)":
                    ai_scores = get_gemini_score(anonymized_article, anonymized_synopsis, gemini_api_key)
                
                # Store results in session state
                st.session_state.nlp_scores = nlp_scores
                st.session_state.ai_scores = ai_scores
                st.session_state.analysis_complete = True
                st.session_state.api_provider = api_provider
                
                # Instead of using experimental_rerun, use tabs navigation
                st.success("Analysis complete! View results in the Evaluation Results tab.")

# Display results in the evaluation tab
with tab2:
    if st.session_state.analysis_complete:
        st.header("Synopsis Evaluation Results")
        
        # Display scores
        results_col1, results_col2 = st.columns([2, 1])
        
        with results_col1:
            if st.session_state.api_provider != "Local NLP Only" and st.session_state.ai_scores:
                st.subheader("AI-based Evaluation")
                
                ai_score = st.session_state.ai_scores["total_score"]
                st.markdown(f"""
                <div class="score-display">
                    Score: {ai_score}/100
                </div>
                """, unsafe_allow_html=True)
                
                # Display detailed scores
                st.markdown("### Score Breakdown")
                st.markdown(f"""
                - **Content Coverage**: {st.session_state.ai_scores['content_coverage']}/50
                - **Clarity**: {st.session_state.ai_scores['clarity']}/25
                - **Coherence**: {st.session_state.ai_scores['coherence']}/25
                """)
                
                # Display feedback
                st.markdown("### Qualitative Feedback")
                st.markdown(f"""
                <div class="feedback-box">
                {st.session_state.ai_scores['feedback']}
                </div>
                """, unsafe_allow_html=True)
                
                # Display chart
                st.markdown("### Visual Breakdown")
                chart = create_score_chart(st.session_state.ai_scores)
                st.pyplot(chart)
            else:
                st.subheader("NLP-based Evaluation")
                
                nlp_score = st.session_state.nlp_scores["total_score"]
                st.markdown(f"""
                <div class="score-display">
                    Score: {nlp_score}/100
                </div>
                """, unsafe_allow_html=True)
                
                # Display detailed scores
                st.markdown("### Score Breakdown")
                st.markdown(f"""
                - **Content Coverage**: {st.session_state.nlp_scores['content_coverage']}/50
                - **Clarity**: {st.session_state.nlp_scores['clarity']}/25
                - **Coherence**: {st.session_state.nlp_scores['coherence']}/25
                """)
                
                # Display chart
                st.markdown("### Visual Breakdown")
                chart = create_score_chart(st.session_state.nlp_scores)
                st.pyplot(chart)
                
                # Local NLP feedback
                st.markdown("### Automated Feedback")
                
                # Generate simple automated feedback
                content_quality = "excellent" if st.session_state.nlp_scores['content_coverage'] > 40 else "good" if st.session_state.nlp_scores['content_coverage'] > 30 else "fair" if st.session_state.nlp_scores['content_coverage'] > 20 else "poor"
                clarity_quality = "excellent" if st.session_state.nlp_scores['clarity'] > 20 else "good" if st.session_state.nlp_scores['clarity'] > 15 else "fair" if st.session_state.nlp_scores['clarity'] > 10 else "poor"
                coherence_quality = "excellent" if st.session_state.nlp_scores['coherence'] > 20 else "good" if st.session_state.nlp_scores['coherence'] > 15 else "fair" if st.session_state.nlp_scores['coherence'] > 10 else "poor"
                
                feedback = f"The synopsis shows {content_quality} content coverage of the main article. The clarity is {clarity_quality}, and the coherence between ideas is {coherence_quality}."
                
                st.markdown(f"""
                <div class="feedback-box">
                {feedback}
                </div>
                """, unsafe_allow_html=True)
        
        with results_col2:
            st.subheader("Text Statistics")
            st.markdown(f"""
            **Article:**
            - Length: {len(st.session_state.article_text)} characters
            - Words: {len(st.session_state.article_text.split())} words
            
            **Synopsis:**
            - Length: {len(st.session_state.synopsis_text)} characters
            - Words: {len(st.session_state.synopsis_text.split())} words
            
            **Synopsis/Article Ratio:**
            - {round(len(st.session_state.synopsis_text) / len(st.session_state.article_text) * 100, 1)}% of original length
            """)
    else:
        st.info("Upload files and click 'Evaluate Synopsis' to see results here")

# Privacy information tab
with tab3:
    st.header("Privacy Protection")
    st.subheader("How We Protect Your Data")
    
    st.markdown("""
    This application is designed with privacy at its core. Here's how we protect your data:
    
    ### Anonymization Process
    1. **Named Entity Recognition**: We use NLP to identify and replace names, organizations, dates, and other sensitive information with generic placeholders.
    2. **Pattern Matching**: We detect and mask email addresses and phone numbers.
    3. **No Data Storage**: No uploaded content is stored after processing is complete.
    
    ### External API Usage
    When using Groq or Gemini APIs:
    - Only anonymized versions of your content are sent to these services.
    - API calls are made directly from your browser session.
    - We do not log or retain any of the anonymized content.
    
    ### Local Processing Option
    - The "Local NLP Only" option processes everything on your local machine without using external APIs.
    - This provides an additional layer of privacy for sensitive documents.
    
    ### Best Practices
    - Use generic text for testing if you're concerned about privacy.
    - Consider using the local processing option for highly sensitive materials.
    - Remember that while we anonymize data, 100% perfect anonymization is difficult to guarantee.
    """)
    
    st.markdown("""
    <div class="privacy-note">
    Note: This application was created as a demonstration for the Catallyst Gen AI engineer assignment and implements privacy best practices as requested in the assignment brief.
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; font-size: 0.8rem;">
© 2024 Synopsis Scorer App | Created for Catallyst Gen AI Assignment
</div>
""", unsafe_allow_html=True)
