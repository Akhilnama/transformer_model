# 🧠 Transformer Language Model from Scratch

End-to-end implementation of a **decoder-only Transformer Language Model** built from first principles using PyTorch.

This project demonstrates deep understanding of modern LLM systems including:

- Attention mechanism mathematics
- Transformer architecture engineering
- Training dynamics and convergence
- Inference system design
- Interactive AI application development

---

## 🚀 Key Features

### 🔬 Model Engineering
- Character-level tokenizer implementation
- Scaled dot-product causal self-attention
- Multi-Head Attention with head projection
- Residual connections and LayerNorm stabilization
- Learnable positional embeddings
- Temperature-controlled autoregressive sampling

### 🏋️ Training Pipeline
- Custom dataset batching strategy
- Negative Log Likelihood optimization
- Adam optimizer training loop
- Periodic checkpoint saving
- Automated qualitative text sampling
- Training convergence monitoring

### ⚡ Inference System
- FastAPI production-ready inference endpoint
- Configurable generation parameters
- Efficient model loading and request validation

### 💬 Interactive Chat UI
- Streamlit-based conversational interface
- Adjustable sampling temperature & length
- Session memory for conversation flow

### 📚 Retrieval Augmented Generation (RAG)
- Sentence-Transformer embedding generation
- FAISS vector database indexing
- Retriever-Generator pipeline integration

### 🐳 Deployment Engineering
- Dockerized inference service
- Modular project packaging
- Scalable API architecture

---

## 🧩 System Architecture
User Prompt
↓
Streamlit Chat UI
↓
FastAPI Inference Layer
↓
Transformer Language Model
↓
(Optional) RAG Retrieval Pipeline
↓
Generated Text Response

## 📁 Project Structure
transformer_model/
│
├── src/
│ ├── attention.py
│ ├── transformer.py
│ ├── model.py
│ ├── train.py
│ ├── generate.py
│
├── api/
│ └── app.py
│
├── ui/
│ └── app.py
│
├── rag/
│ ├── embed.py
│ ├── vectordb.py
│ └── pipeline.py
│
├── checkpoints/
├── Dockerfile
└── requirements.txt

## ▶️ Running the Project

### Setup Environment
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt


### Train Model
python -m src.train


### Start Inference API
python -m uvicorn api.app:app --reload


### Launch Chat UI
streamlit run ui/app.py


---

## 📈 Future Improvements

- KV-Cache optimized decoding
- Flash-Attention integration
- Instruction fine-tuning pipeline
- Distributed GPU training
- Token streaming inference
- Experiment tracking dashboards

---

## 👨‍💻 Author

**Akhil Nama**  
Credit Risk Model Developer → Aspiring AI Engineer  

## 🏗️ Transformer Block Flow
Input Tokens
↓
Token Embedding + Positional Encoding
↓
Multi-Head Self Attention
↓
Residual Connection + LayerNorm
↓
Feed Forward Network
↓
Residual Connection + LayerNorm
↓
Output Logits → Softmax → Next Token


Logs:
python -m src.train
  0%|                                                                                                           | 0/5000 [00:00<?, ?it/s]loss: 4.342959403991699
 10%|█████████▋                                                                                       | 500/5000 [02:54<25:40,  2.92it/s]loss: 2.147707223892212
 20%|███████████████████▏                                                                            | 1000/5000 [05:46<23:14,  2.87it/s]loss: 1.7897803783416748
 30%|████████████████████████████▊                                                                   | 1500/5000 [08:38<19:59,  2.92it/s]loss: 1.6008719205856323
 40%|██████████████████████████████████████▍                                                         | 2000/5000 [11:30<17:12,  2.91it/s]loss: 1.537873387336731
 50%|████████████████████████████████████████████████                                                | 2500/5000 [14:23<14:33,  2.86it/s]loss: 1.4753987789154053
 60%|█████████████████████████████████████████████████████████▌                                      | 3000/5000 [17:15<11:26,  2.92it/s]loss: 1.4221601486206055
 70%|███████████████████████████████████████████████████████████████████▏                            | 3500/5000 [20:07<08:33,  2.92it/s]loss: 1.3678864240646362
 80%|████████████████████████████████████████████████████████████████████████████▊                   | 4000/5000 [23:00<05:44,  2.90it/s]loss: 1.3101907968521118
 90%|██████████████████████████████████████████████████████████████████████████████████████▍         | 4500/5000 [25:52<02:49,  2.95it/s]loss: 1.3117079734802246
100%|████████████████████████████████████████████████████████████████████████████████████████████████| 5000/5000 [28:44<00:00,  2.90it/s]