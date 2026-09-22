import torch
from torch import nn
from typing import Optional, Dict  # For type annotations
from collections import OrderedDict
# from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """
        Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.parameter.Parameter(torch.ones(hidden_size))
        self.bias = nn.parameter.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: Optional[torch.Tensor] = None, dropout: float = 0.0):
    # def __init__(self, d_model: int, n_head: int, attn_mask: Optional[torch.Tensor] = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout)
        # self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        if self.attn_mask is not None:
            self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device)
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TemporalTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: Optional[torch.Tensor] = None, dropout: float = 0.0):
    # def __init__(self, width: int, layers: int, heads: int, attn_mask: Optional[torch.Tensor] = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, dropout) for _ in range(layers)])
        # self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks((x))


class VideoHead(nn.Module):
    def __init__(self, video_head: str, clip_state_dict, dropout: float = 0.0):
    # def __init__(self, video_head: str, clip_state_dict):
        super().__init__()
        self.video_head = video_head
        assert video_head in ["None", "TTrans"]

        if self.video_head == "TTrans":
            context_length = clip_state_dict["positional_embedding"].shape[0]
            embed_dim = clip_state_dict["text_projection"].shape[1]
            # vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
            transformer_width = clip_state_dict["ln_final.weight"].shape[0]
            transformer_heads = transformer_width // 64

            # transformer_layers = len(
            #     set(k.split(".")[2]
            #         for k in clip_state_dict
            #         if k.startswith("transformer.resblocks")
            #     )
            # )

            self.frame_position_embeddings = nn.Embedding(context_length, embed_dim)
            self.pos_drop = nn.Dropout(p=dropout)  # Testing with dropout
            # self.transformer = TemporalTransformer(width=embed_dim, layers=2, heads=transformer_heads, dropout=dropout)
            self.transformer = TemporalTransformer(width=embed_dim, layers=6, heads=transformer_heads, dropout=dropout)

        self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if "beta" in dir(module) and "gamma" in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x):
        b, t, c = x.size()
        x = x.contiguous()
        if self.video_head == "None":
            pass

        elif self.video_head == "TTrans":
            x_original = x
            seq_length = t
            position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0).expand(x.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            x = x + frame_position_embeddings
            x = self.pos_drop(x) # Dropout test
            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = x.type(x_original.dtype) + x_original
        else:
            raise ValueError(f"Unknown temporal modeling header: {self.video_head}")
        return x.mean(dim=1, keepdim=False)


class VideoCLIP(nn.Module):
    def __init__(self, clip_model, video_head: VideoHead, n_seg: int, text_projector: torch.Tensor, reductor: Optional[Dict[str, torch.Tensor]] = None):
        super(VideoCLIP, self).__init__()
        self.visual = clip_model.visual
        self.fusion_model = video_head
        self.n_seg = n_seg
        self.text_projector = text_projector / text_projector.norm(dim=-1, keepdim=True)  # Ensuring normalized text projector
        self.logit_scale = clip_model.logit_scale
        # self.logit_scale.requires_grad_(False)
        self.setup_reductor(reductor)

    def forward(self, image):
        image_emb = self.encode_image(image)
        image_emb = self._reduction_transform(image_emb)
        image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
        return self.logit_scale.exp() * (image_emb @ self.text_projector.t())  # Logits

    def encode_image(self, image):
        n_images = image.size(0)
        batch_size = n_images // self.n_seg
        image_emb = self.visual(image)
        if image_emb.size(0) == batch_size:
            # Fallback in the case we are only processing one image in the batch
            return image_emb
        else:
            # Applying the temporal transformer
            image_emb = image_emb.view(batch_size, self.n_seg, -1)
            image_emb = self.fusion_model(image_emb)
            return image_emb

    def setup_reductor(self, reductor: Optional[Dict[str, torch.Tensor]]):
        self.centerer = None
        if reductor is None:
            self._reduction_type = None
            self.reductor = None
            return

        self._reduction_type = reductor["reductor_type"]
        # TODO: Check the reductor dictionary creation in file "projector.py"
        if self._reduction_type in ["pca", "lda"]:
            self.centerer = reductor["centerer"].clone().detach()  # Centerer required in the case of PCA and LDA

        if self._reduction_type in ["pca", "nca"]:
            self.reductor = reductor["reductor"].T.clone().detach()  # Reduction matrix
        else:
            self.reductor = reductor["reductor"].clone().detach()  # Reduction matrix, only this is required for LDA transformation

    def _reduction_transform(self, data):
        """ Reduction over "data" with a linear transformation (PCA, LDA or NCA) """
        if self._reduction_type is None:
            return data

        # reduce = data.to(dtype=torch.float32)  # Only this is required for NCA transformation
        # reductor = self.reductor["reductor"].to(device=data.device, dtype=torch.float32)  # Reduction matrix, only this is required for LDA transformation

        if self.centerer is not None:
            centered_data = data - self.centerer

        reduced_data = torch.matmul(centered_data, self.reductor)

        return reduced_data


class VideoCLIPConv(nn.Module):
    def __init__(self, clip_model, video_head: VideoHead, n_seg):
        super(VideoCLIPConv, self).__init__()
        self.visual = clip_model.visual
        self.fusion_model = video_head
        self.n_seg = n_seg
        self.logit_scale = clip_model.logit_scale
        self.reductor = nn.Conv1d(512, 264, 1)

    def forward(self, image, text_emb):
        image_emb = self.encode_image(image)
        image_emb = self.reductor(image_emb.unsqueeze(-1)).squeeze(-1)
        image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_emb @ text_emb.t()
        return logits

    def encode_image(self, image):
        # Batch size
        bt = image.size(0)
        b = bt // self.n_seg
        image_emb = self.visual(image)
        if image_emb.size(0) == b:  # joint
            return image_emb
        else:
            image_emb = image_emb.view(b, self.n_seg, -1)
            image_emb = self.fusion_model(image_emb)
            return image_emb


class VideoClassifier(nn.Module):
    """
    IMPORTANT: This class is intended to use the ViT pretrained
    backbone inside the CLIP model with a video head and a linear
    layer to classify video
    """

    def __init__(self, clip_model, video_head, n_seg, num_classes):
        super(VideoClassifier, self).__init__()
        self.visual = clip_model.visual
        # self.visual = vit
        self.fusion_model = video_head
        self.classifier = nn.Linear(768, num_classes)
        self.n_seg = n_seg

    def forward(self, image):
        image_emb = self.encode_image(image)
        logits = self.classifier(image_emb)
        return logits

    def encode_image(self, image):
        # Batch size
        bt = image.size(0)
        b = bt // self.n_seg
        image_emb = self.visual(image)
        # print(f"Embedding shape: {image_emb.shape[-1]}")
        if image_emb.size(0) == b:  # joint
            return image_emb
        else:
            image_emb = image_emb.view(b, self.n_seg, -1)
            image_emb = self.fusion_model(image_emb)
            return image_emb.contiguous()

