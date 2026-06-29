import copy
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50


RESNET_FREEZE_STAGES = ("none", "stem", "layer1", "layer2", "layer3", "all")


def cosine_mse_loss(p, z):
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return 2 - 2 * (p * z.detach()).sum(dim=-1).mean()


def load_local_state_dict(weights_path: str):
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict):
        if "state_dict" in state:
            state = state["state_dict"]
        elif "model" in state:
            state = state["model"]

    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("model."):
            key = key[len("model."):]
        cleaned[key] = value
    return cleaned


def freeze_resnet_until(backbone: nn.Module, freeze_until: str):
    if freeze_until not in RESNET_FREEZE_STAGES:
        raise ValueError(
            f"Unknown ResNet freeze stage '{freeze_until}'. "
            f"Choose from {RESNET_FREEZE_STAGES}."
        )

    stage_modules = {
        "none": [],
        "stem": ["conv1", "bn1"],
        "layer1": ["conv1", "bn1", "layer1"],
        "layer2": ["conv1", "bn1", "layer1", "layer2"],
        "layer3": ["conv1", "bn1", "layer1", "layer2", "layer3"],
        "all": ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"],
    }[freeze_until]

    for module_name in stage_modules:
        module = getattr(backbone, module_name)
        for parameter in module.parameters():
            parameter.requires_grad = False
    return stage_modules


class FeatureMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        latent_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class ResNetImageEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        pretrained: bool = True,
        weights_path: Optional[str] = None,
        freeze_backbone: bool = False,
        freeze_backbone_until: str = "none",
    ):
        super().__init__()

        if weights_path:
            backbone = resnet50(weights=None)
            backbone.load_state_dict(load_local_state_dict(weights_path), strict=True)
            self.pretrained_source = str(Path(weights_path).expanduser().resolve())
        elif pretrained:
            backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            self.pretrained_source = "torchvision_imagenet1k"
        else:
            backbone = resnet50(weights=None)
            self.pretrained_source = "none"

        feature_dim = int(backbone.fc.in_features)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.frozen_backbone_modules = []

        if freeze_backbone:
            freeze_backbone_until = "all"
        self.frozen_backbone_modules = freeze_resnet_until(
            self.backbone,
            freeze_backbone_until,
        )

        self.projector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.projector(features)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            for module_name in self.frozen_backbone_modules:
                getattr(self.backbone, module_name).eval()
        return self


class CrossModalAggregator(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        depth: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, tokens, observed_mask):
        if observed_mask.dtype != torch.bool:
            observed_mask = observed_mask.bool()

        encoded = self.blocks(tokens, src_key_padding_mask=~observed_mask)
        encoded = self.norm(encoded)
        weights = observed_mask.float().unsqueeze(-1)
        return (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class TargetConditionedPredictor(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, context_latent, target_token):
        return self.net(torch.cat([context_latent, target_token], dim=-1))


class ClassificationHead(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        out_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class MultiModalImageJEPA(nn.Module):
    """
    Raw-image multimodal JEPA with impute-then-aggregate classification.

    Image modalities are encoded with ResNet50 backbones and feature modalities
    are encoded with MLPs. During classification, missing modality latents are
    synthesized by the JEPA predictor before the completed latent set is
    aggregated by the classifier.
    """

    def __init__(
        self,
        input_dims: Dict[str, Optional[int]],
        image_modalities=("cc", "mlo"),
        latent_dim: int = 256,
        hidden_dim: int = 512,
        aggregator_depth: int = 2,
        aggregator_heads: int = 4,
        predictor_hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        num_classes: Optional[int] = None,
        classifier_hidden_dim: Optional[int] = None,
        classifier_dropout: float = 0.1,
        pretrained_image_encoder: bool = True,
        image_weights_path: Optional[str] = None,
        freeze_image_backbone: bool = False,
        freeze_image_backbone_until: str = "none",
    ):
        super().__init__()
        self.modality_names = list(input_dims.keys())
        self.image_modalities = set(image_modalities)
        self.num_modalities = len(self.modality_names)
        self.latent_dim = int(latent_dim)
        self.num_classes = None if num_classes is None else int(num_classes)

        self.online_encoders = nn.ModuleDict()
        for name in self.modality_names:
            if name in self.image_modalities:
                self.online_encoders[name] = ResNetImageEncoder(
                    latent_dim=latent_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    pretrained=pretrained_image_encoder,
                    weights_path=image_weights_path,
                    freeze_backbone=freeze_image_backbone,
                    freeze_backbone_until=freeze_image_backbone_until,
                )
            else:
                if input_dims[name] is None:
                    raise ValueError(f"Feature modality {name} needs an input dim.")
                self.online_encoders[name] = FeatureMLP(
                    in_dim=int(input_dims[name]),
                    latent_dim=latent_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )

        self.target_encoders = copy.deepcopy(self.online_encoders)
        for parameter in self.target_encoders.parameters():
            parameter.requires_grad = False

        self.modality_embed = nn.Embedding(self.num_modalities, latent_dim)
        self.target_modality_token = nn.Embedding(self.num_modalities, latent_dim)
        self.aggregator = CrossModalAggregator(
            latent_dim=latent_dim,
            depth=aggregator_depth,
            num_heads=aggregator_heads,
            dropout=dropout,
        )
        self.predictor = TargetConditionedPredictor(
            latent_dim=latent_dim,
            hidden_dim=predictor_hidden_dim or hidden_dim,
            dropout=dropout,
        )
        self.classifier = None
        if self.num_classes is not None:
            out_dim = 1 if self.num_classes == 2 else self.num_classes
            self.classifier = ClassificationHead(
                latent_dim=latent_dim,
                out_dim=out_dim,
                hidden_dim=classifier_hidden_dim or hidden_dim,
                dropout=classifier_dropout,
            )

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoders.eval()
        return self

    @torch.no_grad()
    def update_target_encoders(self, momentum: float):
        online_params = list(self.online_encoders.parameters())
        target_params = list(self.target_encoders.parameters())
        torch._foreach_mul_(target_params, momentum)
        torch._foreach_add_(target_params, online_params, alpha=1.0 - momentum)

        online_buffers = dict(self.online_encoders.named_buffers())
        target_buffers = dict(self.target_encoders.named_buffers())
        for name, target_buffer in target_buffers.items():
            online_buffer = online_buffers[name]
            if target_buffer.dtype.is_floating_point:
                target_buffer.mul_(momentum).add_(online_buffer, alpha=1.0 - momentum)
            else:
                target_buffer.copy_(online_buffer)

    def encode_online(self, modalities: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            name: self.online_encoders[name](modalities[name])
            for name in self.modality_names
        }

    @torch.no_grad()
    def encode_target(self, modalities: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            name: self.target_encoders[name](modalities[name])
            for name in self.modality_names
        }

    def stack_modalities(self, latents: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.stack([latents[name] for name in self.modality_names], dim=1)

    def context_from_latents(self, online_latents, modality_mask):
        latent_stack = self.stack_modalities(online_latents)
        modality_ids = torch.arange(self.num_modalities, device=latent_stack.device)
        tokens = latent_stack + self.modality_embed(modality_ids).unsqueeze(0)
        return self.aggregator(tokens, modality_mask.bool())

    def classify_from_latents(self, online_latents, modality_mask):
        if self.classifier is None:
            raise RuntimeError("MultiModalImageJEPA was built without num_classes.")
        context_latent = self.context_from_latents(online_latents, modality_mask)
        return self.classifier(context_latent)

    def complete_missing_latents(
        self,
        online_latents: Dict[str, torch.Tensor],
        modality_mask: torch.Tensor,
    ):
        modality_mask = modality_mask.bool()
        if bool((modality_mask.sum(dim=1) < 1).any()):
            raise RuntimeError("At least one observed modality is required.")

        latent_stack = self.stack_modalities(online_latents)
        batch_size = latent_stack.size(0)
        device = latent_stack.device

        context_latent = self.context_from_latents(online_latents, modality_mask)
        modality_ids = torch.arange(self.num_modalities, device=device)
        target_tokens = self.target_modality_token(modality_ids)
        predicted_stack = torch.stack(
            [
                self.predictor(
                    context_latent,
                    target_tokens[index].expand(batch_size, -1),
                )
                for index in range(self.num_modalities)
            ],
            dim=1,
        )

        completed_stack = torch.where(
            modality_mask.unsqueeze(-1),
            latent_stack,
            predicted_stack,
        )
        completed_latents = {
            name: completed_stack[:, index]
            for index, name in enumerate(self.modality_names)
        }
        completed_mask = torch.ones_like(modality_mask, dtype=torch.bool)
        return completed_latents, completed_mask

    def classify(self, modalities, modality_mask):
        online_latents = self.encode_online(modalities)
        completed_latents, completed_mask = self.complete_missing_latents(
            online_latents,
            modality_mask,
        )
        return self.classify_from_latents(completed_latents, completed_mask)

    def forward(
        self,
        modalities: Dict[str, torch.Tensor],
        modality_mask: torch.Tensor,
        mode: str = "classify",
        target_indices: Optional[torch.Tensor] = None,
        noise_std: float = 0.01,
        feature_dropout: float = 0.0,
        inter_weight: float = 1.0,
        intra_weight: float = 1.0,
        labels: Optional[torch.Tensor] = None,
        criterion=None,
        num_classes: Optional[int] = None,
        cls_weight: float = 1.0,
        lambda_smd: float = 1.0,
    ):
        if mode == "classify":
            return self.classify(modalities, modality_mask)

        if mode == "jepa_loss":
            return compute_multimodal_image_jepa_loss(
                model=self,
                modalities=modalities,
                modality_mask=modality_mask,
                target_indices=target_indices,
                noise_std=noise_std,
                feature_dropout=feature_dropout,
                inter_weight=inter_weight,
                intra_weight=intra_weight,
            )

        if mode == "train_loss":
            if labels is None or criterion is None or num_classes is None:
                raise ValueError(
                    "labels, criterion, and num_classes are required for train_loss."
                )

            jepa_losses = compute_multimodal_image_jepa_loss(
                model=self,
                modalities=modalities,
                modality_mask=modality_mask,
                target_indices=target_indices,
                noise_std=noise_std,
                feature_dropout=feature_dropout,
                inter_weight=inter_weight,
                intra_weight=intra_weight,
            )

            logits_fusion = self.classify(modalities, modality_mask)
            cls_fusion_loss = compute_supervised_loss(
                logits_fusion,
                labels,
                num_classes,
                criterion,
            )

            single_losses = []
            for modality_index in range(self.num_modalities):
                keep = modality_mask[:, modality_index]
                if not bool(keep.any()):
                    continue

                observed_mask = torch.zeros_like(modality_mask[keep])
                observed_mask[:, modality_index] = True
                kept_modalities = {
                    name: value[keep]
                    for name, value in modalities.items()
                }
                logits_single = self.classify(kept_modalities, observed_mask)
                single_losses.append(
                    compute_supervised_loss(
                        logits_single,
                        labels[keep],
                        num_classes,
                        criterion,
                    )
                )

            cls_single_loss = (
                torch.stack(single_losses).sum()
                if single_losses
                else torch.zeros((), device=labels.device)
            )
            cls_loss = cls_fusion_loss + lambda_smd * cls_single_loss
            loss = jepa_losses["loss"] + cls_weight * cls_loss
            return {
                "loss": loss,
                "jepa_loss": jepa_losses["loss"].detach(),
                "inter_loss": jepa_losses["inter_loss"],
                "intra_loss": jepa_losses["intra_loss"],
                "cls_loss": cls_loss.detach(),
                "cls_fusion_loss": cls_fusion_loss.detach(),
                "cls_single_loss": cls_single_loss.detach(),
                "target_indices": jepa_losses["target_indices"],
                "observed_mask": jepa_losses["observed_mask"],
            }

        raise ValueError(f"Unknown forward mode: {mode}")

    def predict_target(self, online_latents, modality_mask, target_indices):
        latent_stack = self.stack_modalities(online_latents)
        batch_size = latent_stack.size(0)
        device = latent_stack.device

        modality_ids = torch.arange(self.num_modalities, device=device)
        observed_mask = modality_mask.bool().clone()
        observed_mask[torch.arange(batch_size, device=device), target_indices] = False

        if bool((observed_mask.sum(dim=1) < 1).any()):
            raise RuntimeError(
                "Every sample needs at least one observed modality after target sampling."
            )

        tokens = latent_stack + self.modality_embed(modality_ids).unsqueeze(0)
        context_latent = self.aggregator(tokens, observed_mask)
        target_token = self.target_modality_token(target_indices)
        pred_target_latent = self.predictor(context_latent, target_token)
        return pred_target_latent, observed_mask


def augment_features(
    modalities: Dict[str, torch.Tensor],
    noise_std: float = 0.05,
    feature_dropout: float = 0.05,
    training: bool = True,
) -> Dict[str, torch.Tensor]:
    if not training:
        return {name: value for name, value in modalities.items()}

    augmented = {}
    for name, value in modalities.items():
        x = value
        if feature_dropout > 0:
            keep = torch.rand_like(x).ge(feature_dropout).float()
            x = x * keep
        if noise_std > 0:
            x = x + torch.randn_like(x) * noise_std
        augmented[name] = x
    return augmented


def sample_target_indices(modality_mask: torch.Tensor) -> torch.Tensor:
    modality_mask = modality_mask.bool()
    if bool((modality_mask.sum(dim=1) < 2).any()):
        raise RuntimeError(
            "Target sampling requires at least two available modalities per sample."
        )

    scores = torch.rand(
        modality_mask.shape,
        device=modality_mask.device,
        dtype=torch.float32,
    )
    scores = scores.masked_fill(~modality_mask, -1.0)
    return scores.argmax(dim=1)


def gather_by_target(latent_stack: torch.Tensor, target_indices: torch.Tensor) -> torch.Tensor:
    gather_index = target_indices.view(-1, 1, 1).expand(
        -1,
        1,
        latent_stack.size(-1),
    )
    return latent_stack.gather(dim=1, index=gather_index).squeeze(1)


def compute_supervised_loss(logits, labels, num_classes: int, criterion):
    if num_classes == 2:
        return criterion(logits.squeeze(1), labels.float())
    return criterion(logits, labels)


def compute_multimodal_image_jepa_loss(
    model: MultiModalImageJEPA,
    modalities: Dict[str, torch.Tensor],
    modality_mask: torch.Tensor,
    target_indices: Optional[torch.Tensor] = None,
    noise_std: float = 0.01,
    feature_dropout: float = 0.0,
    inter_weight: float = 1.0,
    intra_weight: float = 1.0,
):
    aug_online = augment_features(
        modalities,
        noise_std=noise_std,
        feature_dropout=feature_dropout,
        training=model.training,
    )
    aug_target = augment_features(
        modalities,
        noise_std=noise_std,
        feature_dropout=feature_dropout,
        training=model.training,
    )

    online_latents = model.encode_online(aug_online)
    with torch.no_grad():
        target_latents = model.encode_target(aug_target)

    online_stack = model.stack_modalities(online_latents)
    target_stack = model.stack_modalities(target_latents)

    modality_mask = modality_mask.bool()
    if target_indices is None:
        target_indices = sample_target_indices(modality_mask)

    pred_target, observed_mask = model.predict_target(
        online_latents=online_latents,
        modality_mask=modality_mask,
        target_indices=target_indices,
    )
    target_latent = gather_by_target(target_stack, target_indices)
    inter_loss = cosine_mse_loss(pred_target, target_latent)

    intra_terms = []
    for modality_index in range(model.num_modalities):
        available = modality_mask[:, modality_index]
        if bool(available.any()):
            intra_terms.append(
                cosine_mse_loss(
                    online_stack[available, modality_index],
                    target_stack[available, modality_index],
                )
            )

    if not intra_terms:
        raise RuntimeError("No available modalities found for intra-modality loss.")

    intra_loss = torch.stack(intra_terms).mean()
    loss = inter_weight * inter_loss + intra_weight * intra_loss
    return {
        "loss": loss,
        "inter_loss": inter_loss.detach(),
        "intra_loss": intra_loss.detach(),
        "target_indices": target_indices.detach(),
        "observed_mask": observed_mask.detach(),
    }
