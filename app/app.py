from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import torch
from PIL import Image

from src.dataset import preprocess_image
from src.gradcam import generate_gradcam
from src.models import create_model

APP_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = APP_ROOT / "checkpoints"
CONFIG_PATH = APP_ROOT / "configs" / "config.yaml"

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required to read the application configuration.") from exc

with CONFIG_PATH.open("r", encoding="utf-8") as cfg_file:
    CONFIG = yaml.safe_load(cfg_file)

CLASS_NAMES = CONFIG["experiment"]["class_names"]
DISCLAIMER = CONFIG["app"]["disclaimer"]


def get_checkpoint_choices() -> list[str]:
    if not CHECKPOINT_DIR.exists():
        return []
    supported_suffixes = {".pt", ".pth", ".ckpt"}
    return sorted(p.name for p in CHECKPOINT_DIR.iterdir() if p.is_file() and p.suffix.lower() in supported_suffixes)


def _safe_state_dict(candidate: Any) -> dict[str, torch.Tensor]:
    if isinstance(candidate, dict):
        if "state_dict" in candidate and isinstance(candidate["state_dict"], dict):
            return candidate["state_dict"]
        if "model_state_dict" in candidate and isinstance(candidate["model_state_dict"], dict):
            return candidate["model_state_dict"]
        if all(isinstance(v, torch.Tensor) for v in candidate.values()):
            return candidate
    raise ValueError(
        "Unsupported checkpoint format. Expected a raw state_dict or a dictionary containing 'state_dict'/'model_state_dict'."
    )


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path, architecture: str) -> None:
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}. Please place the model file in the checkpoints directory.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and checkpoint.get("architecture"):
        checkpoint_architecture = str(checkpoint["architecture"]).lower().strip()
        if checkpoint_architecture != architecture.lower().strip():
            raise ValueError(
                f"Checkpoint architecture '{checkpoint_architecture}' does not match selected architecture '{architecture}'."
            )
    state_dict = _safe_state_dict(checkpoint)

    model_load_result = model.load_state_dict(state_dict, strict=False)
    if model_load_result.missing_keys or model_load_result.unexpected_keys:
        raise ValueError(
            "Checkpoint state_dict is incompatible with the selected architecture. "
            f"Missing: {model_load_result.missing_keys}; Unexpected: {model_load_result.unexpected_keys}"
        )

    model.to(device)
    model.eval()


def predict_tuberculosis(image: Image.Image, architecture: str, checkpoint_name: str):
    if not checkpoint_name:
        raise gr.Error("Please select a checkpoint before running inference.")

    checkpoint_path = CHECKPOINT_DIR / checkpoint_name
    if not checkpoint_path.exists():
        raise gr.Error(f"Checkpoint '{checkpoint_name}' was not found in the checkpoints directory.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(architecture, num_classes=2, pretrained=False)
    try:
        load_checkpoint(model, checkpoint_path, architecture)
    except Exception as exc:
        raise gr.Error(f"Checkpoint could not be loaded: {exc}") from exc

    image_rgb = image.convert("RGB")
    processed = preprocess_image(np.asarray(image_rgb, dtype=np.uint8)).to(device)

    with torch.no_grad():
        logits = model(processed)
        probabilities = torch.softmax(logits, dim=1)[0]

    normal_probability = float(probabilities[0].item())
    tb_probability = float(probabilities[1].item())
    confidence = float(probabilities.max().item())
    predicted_index = int(probabilities.argmax().item())
    predicted_label = CLASS_NAMES[predicted_index]

    gradcam_output = generate_gradcam(model, image_rgb, architecture, target_class=predicted_index)
    gradcam_image = gradcam_output["overlay"]

    return (
        predicted_label,
        f"{normal_probability:.4f}",
        f"{tb_probability:.4f}",
        f"{confidence:.4f}",
        gradcam_image,
    )


def build_app() -> gr.Blocks:
    with gr.Blocks(title="TB CXR Research Prototype") as demo:
        gr.Markdown("# Tuberculosis Chest X-Ray Screening Research Prototype")
        gr.Markdown(
            "RESEARCH PROTOTYPE ONLY — NOT FOR CLINICAL DIAGNOSIS\n\n"
            "Grad-CAM is an exploratory model-attribution visualization and is not validated clinical ground truth."
        )

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Chest X-Ray Image", type="pil")
                architecture_dropdown = gr.Dropdown(
                    choices=["resnet50", "densenet121", "efficientnet_b0"],
                    value="resnet50",
                    label="Architecture",
                )
                checkpoint_dropdown = gr.Dropdown(
                    choices=get_checkpoint_choices(),
                    value=(get_checkpoint_choices()[0] if get_checkpoint_choices() else None),
                    label="Checkpoint",
                    allow_custom_value=False,
                )
                predict_button = gr.Button("Predict")
            with gr.Column():
                prediction_output = gr.Textbox(label="Predicted class")
                normal_probability = gr.Textbox(label="Normal probability")
                tb_probability = gr.Textbox(label="TB-consistent probability")
                confidence_output = gr.Textbox(label="Confidence")
                gradcam_output = gr.Image(label="Grad-CAM visualization")

        predict_button.click(
            fn=predict_tuberculosis,
            inputs=[image_input, architecture_dropdown, checkpoint_dropdown],
            outputs=[prediction_output, normal_probability, tb_probability, confidence_output, gradcam_output],
        )

        return demo


def main() -> None:
    demo = build_app()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
