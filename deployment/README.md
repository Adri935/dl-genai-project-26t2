---
title: Smart MCQ Solver - ELECTRA AI
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
short_description: A Multiple Choice Question solver using fine-tuned ELECTRA.
---

# 🧠 Smart MCQ Solver (ELECTRA Fine-Tuned Model)

An intelligent, deep-learning powered Multiple Choice Question (MCQ) Solver hosted on Gradio and fine-tuned on multiple-choice reasoning datasets using **ELECTRA** (`google/electra-base-discriminator`).

## 🌟 Features
- **High Performance**: Fine-tuned `google/electra-base-discriminator` model achieving **MAP@3: 0.9787** and **Validation Accuracy: 96.4%**.
- **Top Answer Prediction**: Identifies the correct option (A, B, C, D, or E).
- **Top-3 Ranked Predictions**: Displays confidence scores and visual probability distributions across candidate choices.
- **Modern Responsive UI**: Clean glassmorphism styling with presets for instant evaluation.

## 🚀 Hugging Face Space Deployment
To deploy this application to Hugging Face Spaces:
1. Create a new **Gradio Space** on [Hugging Face](https://huggingface.co/new-space).
2. Upload `app.py`, `requirements.txt`, `README.md`, and `best_electra_fine-tuned_model.pt`.
3. Space will automatically install dependencies and launch the Gradio application!
