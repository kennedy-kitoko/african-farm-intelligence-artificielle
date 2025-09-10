🌍 African Farm Artificial Intelligence / Intelligence Artificielle Agricole Africaine
🇬🇧 English
📌 Project Overview

This repository contains the fine-tuning of Llama 3.1 (8B) using Unsloth on a custom agricultural dataset from the Democratic Republic of Congo (DRC).
The objective is to build an agricultural expert assistant (AGRO-IA) capable of answering farmers’ questions about crops, pests, soil management, irrigation, and sustainable practices.

🔬 Key Features

Fine-tuned Llama 3.1 8B with Unsloth 2× faster LoRA.

Dataset: 901 Q/A pairs in French, focused on Congolese & African agriculture.

Covers major crops (maize, cassava, rice, beans, tomatoes, bananas, cocoa, coffee).

Practical and low-cost solutions (compost, neem, biochar, crop rotation).

Loss reduced from 2.57 → 0.69 after 120 training steps.

Deployable as a local chatbot with Ollama or LM Studio.

⚙️ Tech Stack

Model: Llama 3.1 8B Instruct

Framework: Unsloth

Training: Google Colab (T4 GPU, 16 GB VRAM)

Fine-tuning method: LoRA (Low Rank Adaptation)

Output format: GGUF (compatible with Ollama/LM Studio)

📊 Results

Training steps: 120 (~2 epochs)

Final loss: 0.69

Adapted answers for African farmers (local diseases, tropical climate).

🚀 How to Use

Clone the repo:

git clone https://github.com/your-username/african-farm-artificial-intelligence.git
cd african-farm-artificial-intelligence


Install dependencies:

pip install -r requirements.txt


Run fine-tuning (example):

python train.py


Export the model to GGUF and load it with Ollama or LM Studio.

📌 Future Work

Expand dataset to 50k+ entries (French, Lingala, Swahili).

Add economic aspects (yield, cost estimation).

Deploy as a mobile chatbot for farmers.

🇫🇷 Français
📌 Aperçu du projet

Ce dépôt contient le recalibrage (fine-tuning) du modèle Llama 3.1 (8B) à l’aide de Unsloth sur un jeu de données agricoles congolais.
L’objectif est de créer un assistant agricole intelligent (AGRO-IA) capable de répondre aux questions des agriculteurs sur les cultures, les ravageurs, la fertilisation, l’irrigation et les pratiques durables.

🔬 Points clés

Recalibrage de Llama 3.1 8B avec Unsloth 2× plus rapide (LoRA).

Dataset : 901 paires Q/R en français, centrées sur l’agriculture congolaise et africaine.

Couvre les principales cultures (maïs, manioc, riz, haricot, tomate, bananier, cacao, café).

Solutions pratiques et peu coûteuses (compost, neem, biochar, rotation des cultures).

Perte réduite de 2.57 → 0.69 après 120 étapes d’entraînement.

Déployable en chatbot local avec Ollama ou LM Studio.

⚙️ Pile technologique

Modèle : Llama 3.1 8B Instruct

Framework : Unsloth

Entraînement : Google Colab (T4 GPU, 16 Go VRAM)

Méthode : LoRA (Low Rank Adaptation)

Format de sortie : GGUF (compatible Ollama/LM Studio)

📊 Résultats

Étapes d’entraînement : 120 (~2 époques)

Perte finale : 0.69

Réponses adaptées aux agriculteurs africains (maladies locales, climat tropical).

🚀 Utilisation

Cloner le dépôt :

git clone https://github.com/your-username/african-farm-artificial-intelligence.git
cd african-farm-artificial-intelligence


Installer les dépendances :

pip install -r requirements.txt


Lancer l’entraînement :

python train.py


Exporter le modèle en GGUF et le charger avec Ollama ou LM Studio.

📌 Travaux futurs

Élargir le dataset à 50k+ entrées (français, lingala, swahili).

Ajouter des données économiques (rendement, coûts).

Déployer en chatbot mobile pour les agriculteurs.
