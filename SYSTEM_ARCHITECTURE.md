## System Architecture

System Architecture v3 - AlgebraKit - Answer Evaluator

```
+------------------------+
|    User (Student)      |
|  - types answer        |
+-----------+------------+
            |
            v
+------------------------+
|   React UI (Frontend)  |
| - shows question       |
| - collects answer      |
| - shows result         |
| - teacher adds Qs:     |
|   • manual (single)    |
|   • bulk from text     |
+-----------+------------+
            |
     HTTP / JSON (POST /answer, GET /question)
            |
            v
+------------------------+
|   FastAPI API Layer    |
|  (/question, /answer)  |
+-----------+------------+
            |
            v
+------------------------+
| Text Processing        |
|  - normalize           |
|  - chunk split         |
|    (. ? ! ; \n)        |
+-----------+------------+
            |
            v
+------------------------+
| Embedding Service      |
| - embed student        |
|   answer chunks        |
+-----------+------------+
            |
            v
+------------------------+      +------------------------+
| Vector DB / Reference  |<---->| Original Answer Chunks |
| - stored key-point &   |      | - teacher / rubric     |
|   reference embeddings |      |   chunks (text)        |
+-----------+------------+      +------------------------+
            |
            v
+------------------------+
| Cosine Similarity      |
| - compare student      |
|   chunks vs reference  |
|   embeddings           |
+-----------+------------+
            |
            v
+------------------------+
| NLI Model              |
| - check entailment /   |
|   contradiction        |
|   between matched      |
|   chunks               |
+-----------+------------+
            |
            v
+------------------------+
| LLM Verification       |
| - review grey-area     |
|   similarity/NLI cases |
| - adjust matches       |
+-----------+------------+
            |
            v
+------------------------+
|  Score + Feedback      |
+-----------+------------+
            |
            v
+------------------------+
| React UI shows result  |
+------------------------+
```

###  Components

1. **User (Student)** – person entering the answer to be evaluated.
2. **React UI (Frontend)** – web interface that shows the question, collects the answer, and displays the final result.
3. **FastAPI API Layer** – backend HTTP API (`/question`, `/answer`) that receives requests from the frontend.
4. **Text Processing** – normalizes the student answer and splits it into chunks using `. ? ! ; \n`.
5. **Embedding Service** – converts the student answer chunks into vector representations.
6. **Original Answer Chunks** – teacher/rubric reference chunks (text) for the correct answer.
7. **Vector DB / Reference Embeddings** – stored embeddings of the reference chunks used for comparison.
8. **Cosine Similarity** – computes similarity scores between student chunk embeddings and reference embeddings.
9. **NLI Model** – performs entailment/contradiction checks between matched chunks to refine understanding.
10. **LLM Verification** – reviews grey-area similarity/NLI cases and adjusts matches where needed.
11. **Score + Feedback** – final numeric score and textual feedback produced from all upstream signals.
12. **React UI (Result View)** – frontend view that presents the score and feedback back to the student.

