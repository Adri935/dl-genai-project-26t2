import os
import torch
import numpy as np
import pandas as pd
import gradio as gr
import spaces
from transformers import AutoTokenizer, AutoModelForMultipleChoice

# ---------------------------------------------------------
# 1. Device Setup & Model Loading
# ---------------------------------------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_MODEL_NAME = 'google/electra-base-discriminator'
CHECKPOINT_NAME = 'best_electra_fine-tuned_model.pt'

print(f"Loading tokenizer from '{BASE_MODEL_NAME}'...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

print(f"Initializing AutoModelForMultipleChoice from '{BASE_MODEL_NAME}'...")
model = AutoModelForMultipleChoice.from_pretrained(BASE_MODEL_NAME)

# Load fine-tuned weights
current_dir = os.path.dirname(os.path.abspath(__file__))
ckpt_path = os.path.join(current_dir, CHECKPOINT_NAME)

if not os.path.exists(ckpt_path):
    # Try current working directory as fallback
    ckpt_path = CHECKPOINT_NAME

if os.path.exists(ckpt_path):
    print(f"Loading weights from checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    if isinstance(state_dict, dict):
        if "model_state_dict" in state_dict:
            model.load_state_dict(state_dict["model_state_dict"])
        elif "state_dict" in state_dict:
            model.load_state_dict(state_dict["state_dict"])
        else:
            model.load_state_dict(state_dict)
    else:
        model = state_dict
    print("Fine-tuned weights loaded successfully!")
else:
    print(f"WARNING: Checkpoint '{CHECKPOINT_NAME}' not found in '{current_dir}'. Using base pretrained weights.")

model.to(DEVICE)
model.eval()

# ---------------------------------------------------------
# 2. Prediction Pipeline
# ---------------------------------------------------------
OPTION_LABELS = ['A', 'B', 'C', 'D', 'E']

@spaces.GPU
def solve_mcq(prompt, option_a, option_b, option_c, option_d, option_e):
    """
    Predicts the correct answer and calculates top-3 candidate probabilities
    for a given question prompt and 5 options.
    """
    # Clean inputs
    prompt_str = str(prompt).strip() if prompt else ""
    opts = [str(opt).strip() if opt else "" for opt in [option_a, option_b, option_c, option_d, option_e]]
    
    if not prompt_str:
        return (
            "<div class='error-box'>⚠️ Please enter a question prompt.</div>",
            {},
            pd.DataFrame(columns=["Option", "Text", "Confidence (%)"])
        )
    
    empty_opts = [OPTION_LABELS[i] for i, opt in enumerate(opts) if not opt]
    if empty_opts:
        return (
            f"<div class='error-box'>⚠️ Please provide text for option(s): {', '.join(empty_opts)}.</div>",
            {},
            pd.DataFrame(columns=["Option", "Text", "Confidence (%)"])
        )
    
    # Tokenize 5 (Prompt, Option) pairs
    first_sentences = [prompt_str] * 5
    second_sentences = opts

    encoding = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        max_length=256,
        padding="max_length",
        return_tensors="pt"
    )

    inputs = {k: v.unsqueeze(0).to(DEVICE) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(0).cpu().numpy() # shape (5,)
        
    # Softmax probabilities
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    
    top_idx = int(np.argmax(probs))
    best_option_label = OPTION_LABELS[top_idx]
    best_option_text = opts[top_idx]
    best_confidence = probs[top_idx] * 100.0

    # Top-3 predictions dict for gr.Label
    top_indices = np.argsort(probs)[::-1]
    top_3_dict = {
        f"Option {OPTION_LABELS[i]}: {opts[i][:35]}{'...' if len(opts[i]) > 35 else ''}": float(probs[i])
        for i in top_indices[:3]
    }
    
    # Complete dataframe breakdown
    df_data = []
    for rank, idx in enumerate(top_indices, 1):
        df_data.append({
            "Rank": f"#{rank}",
            "Option": f"Option {OPTION_LABELS[idx]}",
            "Text": opts[idx],
            "Confidence (%)": f"{probs[idx] * 100.0:.2f}%"
        })
    df_breakdown = pd.DataFrame(df_data)

    # HTML Card for Predicted Answer
    best_card_html = f"""
    <div class='result-card'>
        <div class='result-header'>
            <span class='badge-winner'>🏆 TOP PREDICTED ANSWER</span>
            <span class='confidence-pill'>Confidence: {best_confidence:.1f}%</span>
        </div>
        <div class='winner-content'>
            <div class='winner-option-circle'>{best_option_label}</div>
            <div class='winner-details'>
                <div class='winner-label'>Option {best_option_label}</div>
                <div class='winner-text'>{best_option_text}</div>
            </div>
        </div>
    </div>
    """
    
    return best_card_html, top_3_dict, df_breakdown

# ---------------------------------------------------------
# 3. Custom CSS Styling
# ---------------------------------------------------------
CUSTOM_CSS = """
/* Theme and layout tweaks */
body {
    background-color: #0f172a;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}

.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
}

.header-banner {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    color: white;
}

.header-title {
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0 0 8px 0;
    background: linear-gradient(90deg, #ffffff, #c7d2fe);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.header-subtitle {
    font-size: 1.05rem;
    color: #a5b4fc;
    margin: 0 0 16px 0;
}

.badge-container {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}

.metric-badge {
    background: rgba(255, 255, 255, 0.12);
    border: 1px solid rgba(255, 255, 255, 0.2);
    backdrop-filter: blur(8px);
    color: #e0e7ff;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
}

.result-card {
    background: linear-gradient(135deg, #064e3b 0%, #047857 100%);
    border: 1px solid #10b981;
    border-radius: 16px;
    padding: 24px;
    color: white;
    box-shadow: 0 10px 25px -5px rgba(16, 185, 129, 0.25);
    margin-bottom: 16px;
}

.result-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
}

.badge-winner {
    background: rgba(255, 255, 255, 0.2);
    color: #ecfdf5;
    padding: 4px 12px;
    border-radius: 12px;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.5px;
}

.confidence-pill {
    background: #10b981;
    color: #064e3b;
    font-weight: 800;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.9rem;
}

.winner-content {
    display: flex;
    align-items: center;
    gap: 20px;
}

.winner-option-circle {
    width: 64px;
    height: 64px;
    border-radius: 50%;
    background: #ffffff;
    color: #047857;
    font-size: 2.2rem;
    font-weight: 900;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    flex-shrink: 0;
}

.winner-details {
    flex-grow: 1;
}

.winner-label {
    font-size: 0.95rem;
    color: #a7f3d0;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.winner-text {
    font-size: 1.35rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.3;
}

.error-box {
    background: #450a0a;
    border: 1px solid #dc2626;
    color: #fca5a5;
    padding: 16px;
    border-radius: 12px;
    font-weight: 600;
}
"""

# ---------------------------------------------------------
# 4. Building Gradio UI Interface
# ---------------------------------------------------------
with gr.Blocks(theme=gr.themes.Soft(), css=CUSTOM_CSS, title="Smart MCQ Solver - ELECTRA AI") as demo:
    
    # Header Banner
    gr.HTML("""
    <div class='header-banner'>
        <h1 class='header-title'>⚡ Smart MCQ Solver</h1>
        <p class='header-subtitle'>AI-Powered Multiple Choice Question Answering System using fine-tuned ELECTRA Base</p>
        <div class='badge-container'>
            <span class='metric-badge'>🎯 Architecture: ELECTRA-Base (fine-tuned)</span>
            <span class='metric-badge'>✅ Validation MAP@3: 0.9787</span>
            <span class='metric-badge'>📊 Leaderboard Score: 0.75893</span>
            <span class='metric-badge'>🚀 PyTorch & Transformers</span>
        </div>
    </div>
    """)
    
    with gr.Row(equal_height=False):
        # Left Column: Inputs
        with gr.Column(scale=6):
            gr.Markdown("### 📝 Input Question & Options")
            
            prompt_input = gr.Textbox(
                label="Question / Prompt",
                placeholder="Enter the question prompt here...",
                lines=4,
                max_lines=8
            )
            
            with gr.Group():
                gr.Markdown("#### Candidate Options (Provide 5 Options)")
                opt_a = gr.Textbox(label="Option A", placeholder="First candidate answer...")
                opt_b = gr.Textbox(label="Option B", placeholder="Second candidate answer...")
                opt_c = gr.Textbox(label="Option C", placeholder="Third candidate answer...")
                opt_d = gr.Textbox(label="Option D", placeholder="Fourth candidate answer...")
                opt_e = gr.Textbox(label="Option E", placeholder="Fifth candidate answer...")
            
            with gr.Row():
                predict_btn = gr.Button("⚡ Predict Best Option", variant="primary", scale=2)
                clear_btn = gr.ClearButton(
                    components=[prompt_input, opt_a, opt_b, opt_c, opt_d, opt_e],
                    value="🗑️ Clear Inputs",
                    scale=1
                )
        
        # Right Column: Output & Confidence
        with gr.Column(scale=6):
            gr.Markdown("### 🎯 Model Prediction Results")
            
            output_card = gr.HTML(
                value="""
                <div style='background: rgba(255, 255, 255, 0.05); border: 2px dashed rgba(255, 255, 255, 0.2); border-radius: 16px; padding: 32px; text-align: center; color: #94a3b8;'>
                    <div style='font-size: 2.5rem; margin-bottom: 8px;'>💡</div>
                    <div style='font-size: 1.1rem; font-weight: 600;'>Enter a question and 5 options on the left</div>
                    <div style='font-size: 0.9rem; margin-top: 4px;'>Click "Predict Best Option" to view model prediction and confidence rankings.</div>
                </div>
                """
            )
            
            top3_label = gr.Label(
                label="📊 Top-3 Candidate Confidence Distribution",
                num_top_classes=3
            )
            
            df_output = gr.Dataframe(
                label="📋 Full Option Rankings & Probabilities",
                headers=["Rank", "Option", "Text", "Confidence (%)"],
                interactive=False
            )

    # Event Bindings
    predict_btn.click(
        fn=solve_mcq,
        inputs=[prompt_input, opt_a, opt_b, opt_c, opt_d, opt_e],
        outputs=[output_card, top3_label, df_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
