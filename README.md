# AI Banking Support System

## Objective

* Build an intelligent AI-powered banking assistant using RAG & NLP
* Understand and classify user queries
* Generate contextual responses using LLMs
* Suggest **actionable steps** (block card, escalate, request documents, etc.)
* Build an interactive chatbot UI using Streamlit

---

## Project Structure

```
.
├── dataset/
├── models/
├── requirements.txt
├── understanding_data.ipynb
├── nlp_layer.ipynb
├── rag_pipeline.py
├── llm_pipeline.py
├── main_pipeline.py
├── app.py

````

---

## Repository

GitHub Repository:  
https://github.com/omkar-mandhare26/ai-powered-banking-support-and-fraud-intelligence-system

---
## Dataset

Dataset Link:  
https://drive.google.com/drive/folders/1VqSXY26XDWjqr3EQedSIcKIvzI5ypWKd

---

## Project Setup & Execution

### Step 1: Clone Repository

```bash
https://github.com/omkar-mandhare26/ai-powered-banking-support-and-fraud-intelligence-system.git
cd ai-powered-banking-support-and-fraud-intelligence-system
````

---

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate     # macOS/Linux
venv\Scripts\activate        # Windows
```

---

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Step 4: Run Data Processing

```bash
# Open and run
understanding_data.ipynb
```

---

### Step 5: Run Application

```bash
streamlit run app.py
```

---

## Access the App

```
http://localhost:8501
```