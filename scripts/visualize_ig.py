

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import LayerIntegratedGradients
from captum.attr import visualization as viz


def construct_input_and_baseline(text, tokenizer, max_length=128):
    """Create original input and baseline (all PAD tokens)."""
    
    text_ids = tokenizer.encode(text, max_length=max_length, truncation=True, add_special_tokens=False)
    
    input_ids = [tokenizer.cls_token_id] + text_ids + [tokenizer.sep_token_id]
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    baseline_input_ids = [tokenizer.cls_token_id] + \
                         [tokenizer.pad_token_id] * len(text_ids) + \
                         [tokenizer.sep_token_id]
    
    return (
        torch.tensor([input_ids]),
        torch.tensor([baseline_input_ids]),
        all_tokens
    )


def summarize_attributions(attributions):
    """Sum over embedding dimension and normalize."""
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


def interpret_text(text, model, tokenizer, lig, true_class=None, device="cpu"):
    """Run Integrated Gradients and visualize with Captum."""
    
    input_ids, baseline_input_ids, all_tokens = construct_input_and_baseline(text, tokenizer)
    input_ids = input_ids.to(device)
    baseline_input_ids = baseline_input_ids.to(device)
    
    # Get prediction FIRST (needed for target)
    with torch.no_grad():
        outputs = model(input_ids)
        pred_class = torch.argmax(outputs.logits, dim=-1).item()
        pred_prob = torch.softmax(outputs.logits, dim=-1)[0, pred_class].item()
    
    # Compute attributions with target
    attributions, delta = lig.attribute(
        inputs=input_ids,
        baselines=baseline_input_ids,
        target=pred_class,  # Must specify target!
        return_convergence_delta=True,
        internal_batch_size=1
    )
    
    attributions_sum = summarize_attributions(attributions)
    
    if true_class is None:
        true_class = pred_class
    
    # Create visualization record
    score_vis = viz.VisualizationDataRecord(
        word_attributions=attributions_sum,
        pred_prob=pred_prob,
        pred_class=pred_class,
        true_class=true_class,
        attr_class=text,
        attr_score=attributions_sum.sum(),
        raw_input_ids=all_tokens,
        convergence_score=delta
    )
    
    # Print summary
    label = "Positive" if pred_class == 1 else "Negative"
    print(f"\n{'='*60}")
    print(f"Text: {text}")
    print(f"Prediction: {label} ({pred_prob:.1%})")
    print(f"True class: {true_class}")
    print(f"{'='*60}")
    
    # Get HTML and save to file (viz.visualize_text only works in Jupyter)
    html_obj = viz.visualize_text([score_vis])
    
    # Save HTML to file and open in browser
    import tempfile
    import webbrowser
    from pathlib import Path
    
    html_path = Path("reports/figures/ig_visualization.html")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_obj.data)
    
    print(f"\nSaved visualization to: {html_path}")
    webbrowser.open(f"file://{html_path.absolute()}")
    
    return attributions_sum, all_tokens


def main():
    parser = argparse.ArgumentParser(description="Visualize IG for BERT using Captum")
    parser.add_argument("--text", type=str, default="This movie is superb",
                        help="Text to interpret")
    parser.add_argument("--true-class", type=int, default=None,
                        help="Ground truth class (0=negative, 1=positive)")
    parser.add_argument("--model", type=str,
                        default="distilbert-base-uncased-finetuned-sst-2-english",
                        help="Model name or path")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    model.eval()
    
    # Define forward function and initialize IG
    def forward_func(inputs):
        return model(inputs).logits
    
    lig = LayerIntegratedGradients(forward_func, model.get_input_embeddings())
    
    print("Computing Integrated Gradients...")
    interpret_text(args.text, model, tokenizer, lig, args.true_class, device)


if __name__ == "__main__":
    main()
