   # Synopsis Scorer: Methodology and Privacy Strategy

   ## Scoring Methodology

   The Synopsis Scorer evaluates submissions using a 100-point scale across three dimensions:

   1. **Content Coverage (50 points)**
      - Measures how well the synopsis captures the main ideas and key points from the original article
      - Uses keyword extraction, TF-IDF vectorization, and cosine similarity to quantify content overlap
      - Weights important concepts higher than common words through stop word removal

   2. **Clarity (25 points)**
      - Evaluates the clarity, conciseness, and organization of the synopsis
      - Analyzes sentence structure, length, and complexity
      - Compares average sentence length to identify overly complex or simplistic writing

   3. **Coherence (25 points)**
      - Assesses logical flow and relationship maintenance between ideas
      - Evaluates the synopsis-to-article length ratio (ideally around 30%)
      - Ensures the synopsis is neither too brief nor too verbose

   When using AI models (Groq or Gemini), these dimensions are evaluated through prompt engineering that instructs the model to assess each category separately and provide a detailed breakdown.

   ## Privacy Protection Strategy

   The application implements a multi-layered privacy protection strategy:

   1. **Named Entity Recognition**
      - Uses spaCy's NLP model to identify sensitive information
      - Replaces names, organizations, locations, and dates with generic placeholders
      - Maintains unique identifiers for each entity type (e.g., [PERSON_1], [ORGANIZATION_2])

   2. **Pattern-Based Anonymization**
      - Applies regex pattern matching to detect and mask:
        - Email addresses → [EMAIL]
        - Phone numbers → [PHONE]
        - Other potentially identifiable patterns

   3. **Data Handling**
      - No content is stored after processing is complete
      - All processing happens in-memory
      - No user data is logged or retained

   4. **API Integration**
      - Only anonymized versions of content are sent to external APIs
      - API keys are stored in environment variables, not in the code
      - Users can opt for local-only processing for maximum privacy

   This comprehensive approach ensures that personally identifiable information is protected while still allowing for accurate synopsis evaluation.