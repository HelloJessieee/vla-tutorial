"""Small VLM encoders for the educational π₀-style policy."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor


class VLMEncoder(ABC, nn.Module):
    @property
    @abstractmethod
    def hidden_size(self) -> int: ...

    @abstractmethod
    def encode(
        self,
        images: list[Image.Image],
        instructions: list[str],
    ) -> torch.Tensor:
        """Return pooled hidden (B, H)."""


class CLIPEncoder(VLMEncoder):
    """Default teaching backend — stable on transformers 5.x, ~150M."""

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32", freeze: bool = True) -> None:
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id)
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
        self._hidden = self.model.config.projection_dim * 2

    @property
    def hidden_size(self) -> int:
        return self._hidden

    def encode(self, images: list[Image.Image], instructions: list[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        inputs = self.processor(
            text=instructions,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        out = self.model(**inputs)
        img = out.image_embeds
        txt = out.text_embeds
        return torch.cat([img, txt], dim=-1)


class SmolVLMEncoder(VLMEncoder):
    def __init__(self, model_id: str, freeze: bool = True) -> None:
        super().__init__()
        from transformers import SmolVLMForConditionalGeneration, SmolVLMProcessor

        self.processor = SmolVLMProcessor.from_pretrained(model_id)
        self.model = SmolVLMForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float32
        )
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
        cfg = getattr(self.model.config, "text_config", self.model.config)
        self._hidden = cfg.hidden_size

    @property
    def hidden_size(self) -> int:
        return self._hidden

    def encode(self, images: list[Image.Image], instructions: list[str]) -> torch.Tensor:
        messages = [
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": t}]}]
            for t in instructions
        ]
        prompts = [
            self.processor.apply_chat_template(m, add_generation_prompt=False, tokenize=False)
            for m in messages
        ]
        inputs = self.processor(text=prompts, images=images, return_tensors="pt", padding=True)
        device = next(self.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        core = getattr(self.model, "model", self.model)
        outputs = core(**inputs, output_hidden_states=True, return_dict=True)
        hidden = outputs.hidden_states[-1]
        attn = inputs.get("attention_mask")
        if attn is not None:
            last_idx = attn.sum(dim=1) - 1
            batch_idx = torch.arange(hidden.size(0), device=hidden.device)
            return hidden[batch_idx, last_idx]
        return hidden[:, -1]


class MiniMindEncoder(VLMEncoder):
    """MiniMind2-Small-V (~26M) via trust_remote_code — good for CN teaching demos."""

    def __init__(self, model_id: str, freeze: bool = True) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float32)
        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()
        hidden = getattr(self.model.config, "hidden_size", None)
        if hidden is None:
            hidden = getattr(self.model.config, "n_embd", 512)
        self._hidden = int(hidden)

    @property
    def hidden_size(self) -> int:
        return self._hidden

    def encode(self, images: list[Image.Image], instructions: list[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        # MiniMind VLM API: model.forward(images=..., input_ids=...) — best-effort
        texts = [f"<|image_pad|>{t}" for t in instructions]
        if hasattr(self.model, "encode_image_text"):
            return self.model.encode_image_text(images, texts, device=device)

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        ).to(device)
        if hasattr(self.model, "get_image_features"):
            # stack image tensors if model exposes vision path
            import torchvision.transforms.functional as TF

            tensors = []
            for img in images:
                t = TF.to_tensor(img.resize((256, 256))).to(device)
                tensors.append(t)
            pixel = torch.stack(tensors)
            out = self.model(input_ids=inputs["input_ids"], pixel_values=pixel, output_hidden_states=True)
        else:
            out = self.model(**inputs, output_hidden_states=True)

        if hasattr(out, "last_hidden_state"):
            hidden = out.last_hidden_state
        elif hasattr(out, "hidden_states"):
            hidden = out.hidden_states[-1]
        else:
            raise RuntimeError("MiniMind model output has no hidden states; use smolvlm backend")
        return hidden[:, -1]


def build_encoder(backbone: str, model_id: str, freeze: bool) -> VLMEncoder:
    if backbone == "clip":
        return CLIPEncoder(model_id or "openai/clip-vit-base-patch32", freeze=freeze)
    if backbone.startswith("minimind"):
        return MiniMindEncoder(model_id, freeze=freeze)
    if backbone == "smolvlm":
        return SmolVLMEncoder(model_id, freeze=freeze)
    return CLIPEncoder(model_id or "openai/clip-vit-base-patch32", freeze=freeze)
