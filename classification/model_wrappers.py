'''
Wrapper classes for some foundation models to make them
compatible with forward_intermediates calls to get intermediate
model features to run DAC.
'''

from typing import Optional, List, Tuple, Union
from open_clip import create_model_from_pretrained
from torchvision.transforms.functional import to_pil_image
from transformers import CLIPVisionModel, CLIPImageProcessor
import torch


def feature_take_indices(
    num_features: int,
    indices: Optional[Union[int, List[int]]] = None,
    as_set: bool = False,
) -> Tuple[List[int], int]:
    """Determine the absolute feature indices to 'take' from.

    Note: This function can be called in forwar() so must be torchscript compatible,
    which requires some incomplete typing and workaround hacks.

    Args:
        num_features: total number of features to select from
        indices: indices to select,
          None -> select all
          int -> select last n
          list/tuple of int -> return specified (-ve indices specify from end)
        as_set: return as a set

    Returns:
        List (or set) of absolute (from beginning) indices, Maximum index
    """
    if indices is None:
        indices = num_features  # all features if None

    if isinstance(indices, int):
        # convert int -> last n indices
        take_indices = [num_features - indices + i for i in range(indices)]
    else:
        take_indices: List[int] = []
        for i in indices:
            idx = num_features + i if i < 0 else i
            take_indices.append(idx)

    if not torch.jit.is_scripting() and as_set:
        return set(take_indices), max(take_indices)

    return take_indices, max(take_indices)


class OpenClipModelWrapper(torch.nn.Module):
    def __init__(self, checkpoint_name, num_classes, freeze_encoder):
        super().__init__()
        self.model, self.preprocess = create_model_from_pretrained(checkpoint_name)
        if hasattr(self.model, "text"):
            del self.model.text
        if hasattr(self.model.visual, "output_dim"):
            output_dim = self.model.visual.output_dim
        elif hasattr(self.model.visual, "head") and hasattr(
            self.model.visual.head, "proj"
        ):
            output_dim = self.model.visual.head.proj.out_features
        elif hasattr(self.model.visual, "proj"):
            output_dim = self.model.visual.proj.shape[1]
        else:
            # fallback case assume no projection
            output_dim = self.model.visual.trunk.embed_dim
        # else:
        # output_dim = self.model.visual.output_dim if hasattr(self.model.visual, 'output_dim') else self.model.visual.trunk.embed_dim
        self.model.visual.fc = torch.nn.Linear(output_dim, out_features=num_classes)

        self.freeze_encoder = freeze_encoder

        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.visual.fc.parameters():
                param.requires_grad = True
            self.model.visual.fc.train()

    def forward_head(self, features):
        return self.model.visual.fc(features)

    def forward_features(self, x):
        if self.freeze_encoder:
            with torch.no_grad():
                f = self.model.encode_image(x)
        else:
            f = self.model.encode_image(x)
        return f

    def forward_intermediates(
        self, x: torch.Tensor, indices: Optional[Union[int, List[int]]] = None
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Forward features that returns intermediates.

        Args:
            x: Input image tensor
            indices: Take last n blocks if int, all if None, select matching indices if sequence
            return_prefix_tokens: Return both prefix and spatial intermediate tokens
            stop_early: Stop iterating over blocks when last desired intermediate hit
            output_fmt: Shape of intermediate feature outputs
            intermediates_only: Only return intermediate features
        Returns:

        """
        try:
            last_trunk, intermediates_trunk = (
                self.model.visual.trunk.forward_intermediates(x, indices=indices)
            )
            pooled_last_feats = self.model.visual.trunk.forward_head(last_trunk)
            return self.model.visual.head(pooled_last_feats), intermediates_trunk

        except AttributeError:
            intermediates = []
            take_indices, _ = feature_take_indices(
                len(self.model.visual.transformer.resblocks), indices
            )

            # forward pass
            B, _, height, width = x.shape
            x = self.model.visual.conv1(x)  # shape = [*, width, grid, grid]
            H, W = x.shape[-2], x.shape[-1]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

            # class embeddings and positional embeddings
            x = torch.cat(
                [
                    (
                        self.model.visual.class_embedding.view(1, 1, -1).expand(
                            x.shape[0], -1, -1
                        )
                    ).to(x.dtype),
                    x,
                ],
                dim=1,
            )
            # shape = [*, grid ** 2 + 1, width]
            x = x + self.model.visual.positional_embedding.to(x.dtype)

            x = self.model.visual.patch_dropout(x)
            x = self.model.visual.ln_pre(x)

            blocks = self.model.visual.transformer.resblocks
            for i, blk in enumerate(blocks):
                x = blk(x)
                if i in take_indices:
                    # normalize intermediates with final norm layer if enabled
                    intermediates.append(x)

            # process intermediates
            # split prefix (e.g. class, distill) and spatial feature tokens
            intermediates = [y[:, 1:] for y in intermediates]
            # reshape to BCHW output format
            intermediates = [
                y.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                for y in intermediates
            ]

            if self.model.visual.attn_pool is not None:
                if self.model.visual.attn_pool_contrastive is not None:
                    # This is untested, WIP pooling that should match paper
                    x = self.model.visual.ln_post(
                        x
                    )  # TBD LN first or separate one after each pool?
                    tokens = self.model.visual.attn_pool(x)
                    if self.model.visual.attn_pool_type == "parallel":
                        pooled = self.model.visual.attn_pool_contrastive(x)
                    else:
                        assert self.model.visual.attn_pool_type == "cascade"
                        pooled = self.model.visual.attn_pool_contrastive(tokens)
                else:
                    # this is the original OpenCLIP CoCa setup, does not match paper
                    x = self.model.visual.attn_pool(x)
                    x = self.model.visual.ln_post(x)
                    pooled, tokens = self.model.visual._global_pool(x)
            elif self.model.visual.final_ln_after_pool:
                pooled, tokens = self.model.visual._global_pool(x)
                pooled = self.model.visual.ln_post(pooled)
            else:
                x = self.model.visual.ln_post(x)
                pooled, tokens = self.model.visual._global_pool(x)

            if hasattr(self.model.visual, "proj"):
                pooled = pooled @ self.model.visual.proj
            elif hasattr(self.model.visual, "head"):
                pooled = self.model.visual.head(pooled)
            return pooled, intermediates


class ClipWrapper(torch.nn.Module):
    def __init__(self, num_classes, freeze_encoder=True):
        super().__init__()

        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

        def _preprocess(x):
            return processor(images=x, return_tensors="pt")["pixel_values"].squeeze(0)

        self.preprocess = _preprocess

        self.model.eval()
        self.model.fc = torch.nn.Linear(self.model.config.hidden_size, num_classes)

        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.fc.parameters():
                param.requires_grad = True
            self.model.fc.train()

    def forward_features(self, x):
        if self.freeze_encoder:
            with torch.no_grad():
                f = self.model(x).pooler_output
        else:
            f = self.model(x).pooler_output
        return f

    def forward_intermediates(self, data, indices):
        outputs = self.model(data, output_hidden_states=True)
        intermediates = [y[:, 1:] for y in outputs["hidden_states"][-indices:]]
        # reshape to BCHW output format
        intermediates = [
            y.reshape(1, 7, 7, -1).permute(0, 3, 1, 2).contiguous()
            for y in intermediates
        ]
        return outputs.pooler_output, intermediates

    def forward_head(self, features):
        return self.model.fc(features)
