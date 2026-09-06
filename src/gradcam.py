from __future__ import annotations

import cv2
import numpy as np
import torch

from .dataset import preprocess_image


class GradCAM:
    """Hook-based Grad-CAM implementation for the supported transfer-learning models.

    This is an exploratory attribution tool for research visualization. It is not
    validated clinical ground truth or a lesion-localization method.
    """

    def __init__(self, model: torch.nn.Module, architecture: str, target_layer: torch.nn.Module | None = None):
        self.model = model.eval()
        self.architecture = architecture.lower().strip()
        self._target_layer = self._resolve_target_layer(target_layer)
        self._feature_map: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handles = []
        self._register_hooks()

    def _resolve_target_layer(self, target_layer: torch.nn.Module | None) -> torch.nn.Module:
        if target_layer is not None:
            self._check_layer_validity(target_layer)
            return target_layer

        if self.architecture == "resnet50":
            if not hasattr(self.model, "layer4"):
                raise AttributeError("The selected model does not expose 'layer4'.")
            layer = self.model.layer4[-1]
        elif self.architecture == "densenet121":
            if not hasattr(self.model, "features") or not hasattr(self.model.features, "denseblock4"):
                raise AttributeError("The selected model does not expose 'features.denseblock4'.")
            layer = self.model.features.denseblock4
        elif self.architecture == "efficientnet_b0":
            if not hasattr(self.model, "features") or not hasattr(self.model.features, "__getitem__"):
                raise AttributeError("The selected model does not expose a valid 'features' block.")
            layer = self.model.features[-1]
        else:
            raise ValueError(f"Unsupported architecture for Grad-CAM: {self.architecture}")

        self._check_layer_validity(layer)
        return layer

    def _check_layer_validity(self, layer: torch.nn.Module) -> None:
        if layer is None:
            raise ValueError("Target layer is None.")
        if not isinstance(layer, torch.nn.Module):
            raise TypeError("Target layer must be a torch.nn.Module instance.")

        sample = layer
        if not hasattr(sample, "weight") and not hasattr(sample, "features") and not hasattr(sample, "conv"):
            # Some valid modules have nested outputs but not a direct weight attribute.
            pass

    def _hook_fn(self, module: torch.nn.Module, input_tensor: tuple[torch.Tensor, ...], output_tensor: torch.Tensor):
        self._feature_map = output_tensor

    def _backward_hook_fn(self, module: torch.nn.Module, grad_input: tuple[torch.Tensor, ...], grad_output: tuple[torch.Tensor, ...]):
        self._gradients = grad_output[0]

    def _register_hooks(self):
        self._remove_hooks()

        self._handles.append(self._target_layer.register_forward_hook(self._hook_fn))
        self._handles.append(self._target_layer.register_full_backward_hook(self._backward_hook_fn))

    def _remove_hooks(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __call__(self, input_tensor: torch.Tensor, target_class: int | None = None) -> dict:
        if input_tensor.dim() != 4:
            raise ValueError(f"Expected a batch tensor of shape [B, C, H, W], received {tuple(input_tensor.shape)}")

        self.model.zero_grad(set_to_none=True)
        self._feature_map = None
        self._gradients = None

        logits = self.model(input_tensor)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        logits[:, target_class].backward(retain_graph=False)

        if self._feature_map is None or self._gradients is None:
            raise RuntimeError("Grad-CAM failed to capture activations and gradients from the target layer.")

        activations = self._feature_map.detach()
        gradients = self._gradients.detach()

        if activations.dim() != 4:
            raise ValueError(
                "The target layer does not produce a 4D activation tensor required for Grad-CAM. "
                f"Received shape: {tuple(activations.shape)}"
            )

        if activations.shape[-2] <= 1 or activations.shape[-1] <= 1:
            raise ValueError(
                "The target layer activation map is not spatially large enough for Grad-CAM. "
                f"Received shape: {tuple(activations.shape)}"
            )

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam_max = cam.amax(dim=(2, 3), keepdim=True)
        cam = cam / (cam_max + 1e-8)

        heatmap = cam[0, 0].cpu().numpy()
        heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)
        heatmap = heatmap.astype(np.float32)
        normalized = heatmap / (np.max(heatmap) + 1e-8)

        original_shape = tuple(input_tensor.shape[-2:])
        resized = cv2.resize(normalized, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
        resized = np.clip(resized, 0.0, 1.0)

        self._remove_hooks()
        return {
            "class_index": target_class,
            "heatmap": resized,
            "activation_shape": tuple(activations.shape),
        }

    def __del__(self):
        self._remove_hooks()


def generate_gradcam(
    model: torch.nn.Module,
    image_input,
    architecture: str,
    target_class: int | None = None,
) -> dict:
    """Generate a Grad-CAM heatmap and overlay for a single image.

    Returns a dictionary containing:
    - original: RGB numpy image array
    - heatmap: normalized 2D heatmap in [0, 1]
    - overlay: RGB overlay image
    """
    if isinstance(image_input, (str, np.ndarray, tuple)) or hasattr(image_input, "size"):
        original_rgb = np.asarray(image_input.convert("RGB"), dtype=np.uint8) if hasattr(image_input, "convert") else np.asarray(image_input)
        if original_rgb.ndim == 2:
            original_rgb = cv2.cvtColor(original_rgb, cv2.COLOR_GRAY2RGB)
        elif original_rgb.ndim == 3 and original_rgb.shape[2] == 4:
            original_rgb = cv2.cvtColor(original_rgb, cv2.COLOR_RGBA2RGB)
        elif original_rgb.ndim == 3 and original_rgb.shape[2] == 1:
            original_rgb = cv2.cvtColor(original_rgb[:, :, 0], cv2.COLOR_GRAY2RGB)
    else:
        original_rgb = np.asarray(image_input, dtype=np.uint8)
        if original_rgb.ndim == 2:
            original_rgb = cv2.cvtColor(original_rgb, cv2.COLOR_GRAY2RGB)
        elif original_rgb.ndim == 3 and original_rgb.shape[2] == 4:
            original_rgb = cv2.cvtColor(original_rgb, cv2.COLOR_RGBA2RGB)

    if original_rgb.ndim == 2:
        original_rgb = cv2.cvtColor(original_rgb, cv2.COLOR_GRAY2RGB)

    processed = preprocess_image(image_input)
    processed = processed.to(next(model.parameters()).device)

    gradcam = GradCAM(model, architecture)
    result = gradcam(processed, target_class=target_class)
    heatmap = result["heatmap"]

    color_map = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
    overlay = color_map.astype(np.float32) / 255.0
    base = original_rgb.astype(np.float32) / 255.0
    if base.shape[:2] != overlay.shape[:2]:
        overlay = cv2.resize(overlay, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_LINEAR)
    blended = 0.35 * base + 0.65 * overlay

    return {
        "original": original_rgb,
        "heatmap": heatmap,
        "overlay": (blended * 255.0).astype(np.uint8),
    }
