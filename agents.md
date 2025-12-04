Agent 1 – Input Cleaning
Cleans and normalizes the student’s answer before processing.


Removes punctuation, stop words, and unnecessary whitespace.


Handles casing and simple text normalization (e.g., stemming or lemmatization).



Agent 2 – Knowledge Retrieval
Retrieves relevant knowledge for the question.


Source: key points or teacher rubrics.


Retrieval method: TF-IDF or semantic keyword overlap between the question and the rubric.


Output: the most relevant rubric entries or key points that describe what a correct answer should contain.



Agent 3 – Semantic Similarity
Computes semantic similarity between the cleaned student answer and the retrieved rubric/key points.


Uses embeddings (e.g., all-MiniLM-L6-v2).


Stores the student answer and question ID in the database for audit and analytics.


If similarity is above a “very certain” threshold (e.g., 0.8), grading can be inferred directly.


Otherwise, forwards the case to the RAG reasoning agent.



Agent 4 – RAG-Based Grading (LLM Reasoning)
If the similarity score is low, this agent invokes an LLM (e.g., Phi-3-mini) to reason about the answer using retrieved rubric knowledge.
Prompt Template
You are a {subject} teacher in a secondary school.  
Your task is to grade the student’s answer on a scale from **1 to {max_score}**,  
where the total score equals the **sum of the weights of all key points** in the rubric.  

Rubric / Key Points:  
{rubric_or_keypoints}

Question:  
{question}

Student’s Answer:  
{answer}

Evaluate how much information from the rubric is present in the student’s answer  
and assign a single numeric grade (no text explanation).  

**Return only the number out of {max_score}.**


Agent 5 – Feedback Generator (Optional)
Optionally, the system can produce constructive feedback for the student.

| Instance Type   | GPU           | vCPUs | RAM   | Approx. Hourly Cost (USD) | Approx. Daily Cost (24h) | Inference Speed (per prompt, ~150–200 tokens) | Notes                                                                                             |
| --------------- | ------------- | ----- | ----- | ------------------------- | ------------------------ | --------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| **g4dn.xlarge** | 1×T4 (16GB)   | 4     | 16 GB | ~$0.53/hr                 | **~$12.70/day**          | ~3–6 seconds                                  | Good balance of cost and performance. Recommended for small models (Phi-3-mini, Mistral 7B, etc.) |
| **g5.xlarge**   | 1×A10G (24GB) | 4     | 16 GB | ~$1.20/hr                 | **~$28.80/day**          | ~1–3 seconds                                  | Faster inference, more VRAM—smoother parallel requests and RAG pipelines.                         |


