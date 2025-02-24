# **PubMed RAG: AI-Powered Biomedical Paper Retrieval**  

**PubMed RAG** is an advanced **Retrieval-Augmented Generation (RAG) system** that enables intelligent searching and interaction with **PubMed biomedical research papers**. It utilizes **ChromaDB** for efficient storage and retrieval of embeddings and integrates **OpenAI 4o** to provide insightful, context-aware responses based on user queries.  

---

## **🔬 Features**  
- **📝 Scrapes and indexes PubMed biomedical research papers** for structured data retrieval.  
- **🔍 Efficient search and retrieval** powered by **ChromaDB** and **Llama embeddings**.  
- **🧠 AI-Powered Interactions** through **OpenAI 4o**, enabling users to query and interact with research data.  
- **⚡ Fast and Scalable ETL Pipeline** built with **Langchain** to process large volumes of research data.  
- **☁️ Cloud-based deployment** for seamless access and scalability.  

---

## **🛠 Tech Stack**  
- **Python** for core development  
- **ChromaDB** for vector storage and retrieval  
- **Llama embeddings** for text representation  
- **OpenAI 4o** for natural language interaction  
- **Langchain** for building the ETL pipeline  
- **FastAPI / Flask** (Optional) for deploying the API  

---

## **🚀 Installation & Setup**  

### **1️⃣ Clone the Repository**  
```sh
git clone https://github.com/Akilan19/PubMed_RAG.git
cd PubMed_RAG
```

### **2️⃣ Create a Virtual Environment**  
```sh
python3 -m venv envi
source envi/bin/activate  # On Windows use: envi\Scripts\activate
```

### **3️⃣ Install Dependencies**  
```sh
pip install -r requirements.txt
```

### **4️⃣ Set Up Environment Variables**  
```sh
OPENAI_API_KEY=your_openai_api_key
```

### **5️⃣ Run the Pipeline**  
```sh
python main.py  # Scrapes papers and stores embeddings in ChromaDB
```

# 🛠 Usage
- Step 1: Run the data ingestion pipeline to fetch and store papers.
- Step 2: Use the querying interface to interact with the model.
- Step 3: Get AI-enhanced answers based on biomedical research.
