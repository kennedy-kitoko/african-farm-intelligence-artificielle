# 🌍 African Farm Artificial Intelligence / Intelligence Artificielle Agricole Africaine  

## 🇬🇧 English  

### 📌 Project Overview  
This repository contains the **fine-tuning of Llama 3.1 (8B)** using **Unsloth** on a **custom agricultural dataset from the Democratic Republic of Congo (DRC)**.  
The objective is to build an **agricultural expert assistant (AGRO-IA)** capable of answering farmers’ questions about crops, pests, soil management, irrigation, and sustainable practices.  

### 🔬 Key Features  
- Fine-tuned **Llama 3.1 8B** with **Unsloth 2× faster LoRA**.  
- **Dataset**: 901 Q/A pairs in **French**, focused on Congolese & African agriculture.  
- Covers major crops (maize, cassava, rice, beans, tomatoes, bananas, cocoa, coffee).  
- Practical and low-cost solutions (compost, neem, biochar, crop rotation).  
- **Loss reduced from 2.57 → 0.69** after 120 training steps.  
- Deployable as a **local chatbot** with **Ollama** or **LM Studio**.  

### ⚙️ Tech Stack  
- **Model**: Llama 3.1 8B Instruct  
- **Framework**: [Unsloth](https://unsloth.ai)  
- **Training**: Google Colab (T4 GPU, 16 GB VRAM)  
- **Fine-tuning method**: LoRA (Low Rank Adaptation)  
- **Output format**: GGUF (compatible with Ollama/LM Studio)  

### 📊 Results  
- Training steps: **120 (~2 epochs)**  
- Final loss: **0.69**  
- Adapted answers for African farmers (local diseases, tropical climate).  

### 🚀 How to Use  
1. Clone the repo:  
   ```bash
   git clone https://github.com/your-username/african-farm-artificial-intelligence.git
   cd african-farm-artificial-intelligence
