from typing import Optional, Dict
import torch
from torch import nn
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
    def __init__(self, d_model: int, n_head: int, attn_mask: Optional[torch.Tensor] = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
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
    def __init__(self, width: int, layers: int, heads: int, attn_mask: Optional[torch.Tensor] = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks((x))


class VideoHead(nn.Module):
    def __init__(self, vid_head, clip_state_dict):
        super().__init__()
        self.vid_header = vid_head
        assert vid_head in ["None", "Transf"]

        if self.vid_header == "Transf":
            embed_dim = clip_state_dict["text_projection"].shape[1]
            # embed_dim = 768

            context_length = clip_state_dict["positional_embedding"].shape[0]
            vocab_size = clip_state_dict["token_embedding.weight"].shape[0]
            transformer_width = clip_state_dict["ln_final.weight"].shape[0]
            transformer_heads = transformer_width // 64

            transformer_layers = len(
                set(k.split(".")[2] for k in clip_state_dict if k.startswith("transformer.resblocks")))

            self.frame_position_embeddings = nn.Embedding(context_length, embed_dim)

            self.transformer = TemporalTransformer(width=embed_dim, layers=6, heads=transformer_heads)
            print('layer=6')

        self.apply(self.init_weights)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            if 'beta' in dir(module) and 'gamma' in dir(module):
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
        if self.vid_header == "None":
            pass

        elif self.vid_header == "Transf":
            x_original = x
            seq_length = t
            position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0).expand(x.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            x = x + frame_position_embeddings

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = x.type(x_original.dtype) + x_original
        else:
            raise ValueError('Unknown temporal modeling header: {}'.format(self.vid_header))
        return x.mean(dim=1, keepdim=False)


class VideoCLIP(nn.Module):
    def __init__(self, clip_model, video_head: VideoHead, n_seg, reductor: Optional[Dict[str, torch.Tensor]] = None):
        super(VideoCLIP, self).__init__()
        self.visual = clip_model.visual
        self.fusion_model = video_head
        self.n_seg = n_seg
        self.logit_scale = clip_model.logit_scale
        self.reductor = reductor

    def forward(self, image, text_emb):
        image_emb = self.encode_image(image)
        # image_emb = self.pytorch_pca_transform(image_emb)
        image_emb = self.reduction_transform(image_emb)
        image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        text_emb = text_emb.to(dtype=image_emb.dtype)
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

    # def pytorch_pca_transform(self, data):
    #     if self.reductor is None:
    #         return data
    #
    #     data_centered = data - self.reductor["mean"]
    #     transformed_data = torch.matmul(data_centered, self.reductor["components"].T).to(dtype=torch.float32)
    #     return transformed_data

    def reduction_transform(self, data):
        """Reduction over 'data' with a linear transformation (PCA, LDA or NCA)"""
        if self.reductor is None:
            return data

        reduce = data.to(dtype=torch.float32)  # Only this is required for NCA transformation
        reductor = self.reductor["reductor"].to(device=data.device, dtype=torch.float32)  # Reduction matrix, only this is required for LDA transformation

        if self.reductor["reductor_type"] in ["pca", "lda"]:
            reduce = reduce - self.reductor["centerer"].to(device=data.device, dtype=torch.float32)  # Centered data in the case of PCA and LDA

        if self.reductor["reductor_type"] in ["pca", "nca"]:
            reductor = reductor.T

        transformed_data = torch.matmul(reduce, reductor).to(dtype=torch.float32)

        return transformed_data


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


class ViT(nn.Module):
    def __init__(self, img_size=224, patch_size=32, dim=768, depth=12, heads=12, mlp_dim=3072):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size

        num_patches = (img_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size

        # Patch embedding
        self.patch_embed = nn.Linear(patch_dim, dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        # Transformer encoder (old PyTorch API)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            activation="relu"   # old PyTorch does not have GELU by default
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size

        # Extract patches
        patches = x.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(B, -1, 3 * p * p)

        # Patch embedding
        x = self.patch_embed(patches)

        # CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Add positional embeddings
        x = x + self.pos_embed  # [B, seq, dim]

        # Transform to the format expected by old TransformerEncoder
        x = x.transpose(0, 1)  # [seq, B, dim]

        # Transformer
        x = self.transformer(x)

        # Back to [B, seq, dim]
        x = x.transpose(0, 1)

        # Return CLS token
        return x[:, 0]


class VideoClassifier(nn.Module):
    """
    IMPORTANT: This class is only for use the ViT pretrained backbone inside the CLIP model
    """

    def __init__(self, clip_model, video_header, n_seg, num_classes):
        super(VideoClassifier, self).__init__()
        self.visual = clip_model.visual
        # self.visual = vit
        self.fusion_model = video_header
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
