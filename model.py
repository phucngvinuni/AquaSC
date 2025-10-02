# model.py
import math
from typing import Dict, Optional
import torch
import torch.nn as nn
import numpy as np
from functools import partial
from timm.models.registry import register_model
from digital_channel import transmit_and_receive_indices_batch as transmit_digital_indices # Dòng mới

from model_util import (
    ViTEncoder_Van, ViTDecoder_ImageReconstruction,
    HierarchicalQuantizer, Channels, FeatureImportanceTransformer, _cfg
)
# --- BƯỚC 1: IMPORT MODULE KÊNH KỸ THUẬT SỐ ---
# Giả định bạn đã tạo file digital_channel.py
from digital_channel import transmit_and_receive_indices_batch

class ViT_Reconstruction_Model(nn.Module):
    def __init__(self,
                 img_size: int = 224,
                 patch_size: int = 16,
                 encoder_in_chans: int = 3,
                 encoder_embed_dim: int = 768,
                 encoder_depth: int = 12,
                 encoder_num_heads: int = 12,
                 decoder_embed_dim: int = 256,
                 decoder_depth: int = 4,
                 decoder_num_heads: int = 4,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 norm_layer=nn.LayerNorm,
                 init_values: float = 0.0,
                 use_learnable_pos_emb: bool = False,
                 quantizer_dim: int = 512,
                 bits_vq_high: int = 12,
                 bits_vq_low: int = 8,
                 quantizer_commitment_cost: float = 0.25,
                 fim_embed_dim: int = 128,
                 fim_depth: int = 2,
                 fim_num_heads: int = 4,
                 fim_drop_rate: float = 0.1,
                 fim_routing_threshold: float = 0.6,
                 **kwargs):
        super().__init__()

        self.fim_routing_threshold = fim_routing_threshold
        effective_patch_size = patch_size

        self.img_encoder = ViTEncoder_Van(
            img_size=img_size, patch_size=effective_patch_size, in_chans=encoder_in_chans,
            embed_dim=encoder_embed_dim, depth=encoder_depth, num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate,
            drop_path_rate=drop_path_rate, norm_layer=norm_layer, init_values=init_values,
            use_learnable_pos_emb=use_learnable_pos_emb
        )

        self.full_image_num_patches_h = self.img_encoder.patch_embed.patch_shape[0]
        self.full_image_num_patches_w = self.img_encoder.patch_embed.patch_shape[1]
        num_total_patches_in_image = self.img_encoder.patch_embed.num_patches

        self.img_decoder = ViTDecoder_ImageReconstruction(
            patch_size=effective_patch_size, num_total_patches=num_total_patches_in_image,
            embed_dim=decoder_embed_dim, depth=decoder_depth, num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer, init_values=init_values
        )

        self.encoder_to_channel_proj = nn.Linear(encoder_embed_dim, quantizer_dim)
        self.norm_before_quantizer = nn.LayerNorm(quantizer_dim)
        
        self.quantizer_high = HierarchicalQuantizer(
            num_embeddings=2**bits_vq_high,
            embedding_dim=quantizer_dim,
            commitment_cost=quantizer_commitment_cost
        )
        self.quantizer_low = HierarchicalQuantizer(
            num_embeddings=2**bits_vq_low,
            embedding_dim=quantizer_dim,
            commitment_cost=quantizer_commitment_cost
        )
        
        # Channel simulator này giờ chỉ được dùng cho "analog proxy" trong lúc training
        self.channel_simulator = Channels() 
        self.channel_to_decoder_proj = nn.Linear(quantizer_dim, decoder_embed_dim)

        self.fim_module = FeatureImportanceTransformer(
            input_dim=encoder_embed_dim, fim_embed_dim=fim_embed_dim, fim_depth=fim_depth,
            fim_num_heads=fim_num_heads, drop_rate=fim_drop_rate,
            norm_layer=partial(norm_layer, eps=1e-6)
        )
        self.current_vq_loss = torch.tensor(0.0, dtype=torch.float32)

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None: nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0); nn.init.constant_(module.weight, 1.0)

    def forward(self,
            img: torch.Tensor,
            bm_pos: Optional[torch.Tensor] = None,
            _eval: bool = False,
            eval_snr_db: float = 30.0,
            train_snr_db_min: float = 10.0,
            train_snr_db_max: float = 25.0,
            **kwargs
           ) -> Dict[str, torch.Tensor]:

        B = img.shape[0]
        is_training = self.training and not _eval
        
        if is_training:
            current_snr_db = (torch.rand(1).item() * (train_snr_db_max - train_snr_db_min)) + train_snr_db_min
        else:
            current_snr_db = eval_snr_db

        if bm_pos is None:
            encoder_input_mask_bool = torch.zeros(
                B, self.img_encoder.patch_embed.num_patches, dtype=torch.bool, device=img.device
            )
        else:
            encoder_input_mask_bool = bm_pos

        # 1. Encoder
        x_encoded_tokens = self.img_encoder(img, encoder_input_mask_bool)
        
        # 2. FIM -> Routing Mask
        fim_raw_logits = self.fim_module(x_encoded_tokens)
        fim_scores = torch.sigmoid(fim_raw_logits)
        route_to_high_mask = (fim_scores.squeeze(-1) > self.fim_routing_threshold) # [B, Np]

        # 3. Project features
        x_proj_normalized = self.norm_before_quantizer(self.encoder_to_channel_proj(x_encoded_tokens))

        # --- BƯỚC 4: LƯỢNG TỬ HÓA VÀ TRUYỀN TIN (Logic có phân nhánh) ---
        if is_training:
            # --- TRAINING PATH (ANALOG PROXY) ---
            final_tokens_for_channel = torch.zeros_like(x_proj_normalized)
            total_vq_loss = torch.tensor(0.0, device=img.device)
            
            # Xử lý các token quan trọng (High-Fidelity)
            high_tokens_mask_flat = route_to_high_mask.flatten()
            if high_tokens_mask_flat.any():
                q_high, vq_loss_high, _, _ = self.quantizer_high(x_proj_normalized.view(-1, x_proj_normalized.size(-1))[high_tokens_mask_flat])
                final_tokens_for_channel[route_to_high_mask] = q_high
                total_vq_loss += (vq_loss_high * fim_scores.squeeze(-1)[route_to_high_mask]).mean()

            # Xử lý các token không quan trọng (Low-Fidelity)
            low_tokens_mask = ~route_to_high_mask
            low_tokens_mask_flat = low_tokens_mask.flatten()
            if low_tokens_mask_flat.any():
                q_low, vq_loss_low, _, _ = self.quantizer_low(x_proj_normalized.view(-1, x_proj_normalized.size(-1))[low_tokens_mask_flat])
                final_tokens_for_channel[low_tokens_mask] = q_low
                total_vq_loss += (vq_loss_low * (1 - fim_scores.squeeze(-1)[low_tokens_mask])).mean()
            
            self.current_vq_loss = total_vq_loss / 2.0
            
            # Truyền VECTORS qua kênh ANALOG mô phỏng
            noise_power_variance = 10**(-current_snr_db / 10.0)
            tokens_after_channel = self.channel_simulator.Rayleigh(final_tokens_for_channel, noise_power_variance)
            x_for_decoder_input = self.channel_to_decoder_proj(tokens_after_channel)

        else: # is_training is False --> EVALUATION PATH (DIGITAL PIPELINE)
            if B > 1:
                # Giữ lại cảnh báo này vì pipeline digital thường xử lý batch=1
                print("Warning: Digital evaluation path might be slow with batch size > 1. Forcing batch size to 1 is recommended for evaluation.")

                        # Tạo tensor để chứa các chỉ số và số bit cho từng token
            all_indices = torch.zeros_like(route_to_high_mask, dtype=torch.long)
            bits_per_index = torch.zeros_like(route_to_high_mask, dtype=torch.long)
            
            # Khởi tạo loss và output STE
            total_vq_loss = torch.tensor(0.0, device=img.device)
            quantized_ste_output = torch.zeros_like(x_proj_normalized)

            bits_high = int(np.log2(self.quantizer_high.num_embeddings))
            bits_low = int(np.log2(self.quantizer_low.num_embeddings))

            # Xử lý các token quan trọng (High-Fidelity)
            if route_to_high_mask.any():
                high_tokens = x_proj_normalized[route_to_high_mask]
                q_high_ste, vq_loss_high, _, indices_high = self.quantizer_high(high_tokens)
                
                quantized_ste_output[route_to_high_mask] = q_high_ste
                all_indices[route_to_high_mask] = indices_high
                bits_per_index[route_to_high_mask] = bits_high
                # Weight VQ loss bằng FIM scores
                total_vq_loss += (vq_loss_high * fim_scores.squeeze(-1)[route_to_high_mask]).mean()

            # Xử lý các token không quan trọng (Low-Fidelity)
            low_tokens_mask = ~route_to_high_mask
            if low_tokens_mask.any():
                low_tokens = x_proj_normalized[low_tokens_mask]
                q_low_ste, vq_loss_low, _, indices_low = self.quantizer_low(low_tokens)

                quantized_ste_output[low_tokens_mask] = q_low_ste
                all_indices[low_tokens_mask] = indices_low
                bits_per_index[low_tokens_mask] = bits_low
                # Weight VQ loss
                total_vq_loss += (vq_loss_low * (1 - fim_scores.squeeze(-1)[low_tokens_mask])).mean()

            self.current_vq_loss = total_vq_loss / 2.0 # Trung bình loss từ 2 quantizer

            # 4.5. **TRUYỀN QUA KÊNH DIGITAL**
            # Chú ý: Cần đảm bảo `transmit_digital_indices` có thể xử lý batch > 1
            # Nếu không, cần thêm `if B > 1: raise NotImplementedError...`
            recovered_indices = transmit_digital_indices(all_indices, bits_per_index, snr_db=current_snr_db)

            # 5. Khôi phục vector từ các chỉ số đã nhận được (bên thu)
            tokens_after_channel = torch.zeros_like(x_proj_normalized)
            for i in range(B): # Lặp qua từng ảnh trong batch để tra cứu sổ mã
                rec_idx_i = recovered_indices[i]
                route_high_i = route_to_high_mask[i]
                
                # Lấy các vector từ sổ mã high
                tokens_after_channel[i, route_high_i] = self.quantizer_high.embedding(rec_idx_i[route_high_i])
                
                # Lấy các vector từ sổ mã low
                tokens_after_channel[i, ~route_high_i] = self.quantizer_low.embedding(rec_idx_i[~route_high_i])
                
            # 6. Decoder
            # *** Quan trọng: Để STE hoạt động, gradient phải chảy qua quantized_ste_output ***
            # Chúng ta sẽ dùng một "mánh" nhỏ:
            # Luồng xuôi dùng `tokens_after_channel` (kết quả thực tế sau kênh)
            # Luồng ngược sẽ dùng `quantized_ste_output` (để gradient có thể đi qua)
            x_for_decoder_input_fw = self.channel_to_decoder_proj(tokens_after_channel)
            x_for_decoder_input_bw = self.channel_to_decoder_proj(quantized_ste_output)
            
            # Áp dụng STE cho toàn bộ khối kênh + VQ
            x_for_decoder_input = x_for_decoder_input_fw + (x_for_decoder_input_bw - x_for_decoder_input_fw).detach()
        
        reconstructed_image = self.img_decoder(
            x_vis_tokens=x_for_decoder_input,
            encoder_mask_boolean=encoder_input_mask_bool,
            full_image_num_patches_h=self.full_image_num_patches_h,
            full_image_num_patches_w=self.full_image_num_patches_w,
        )

        output_dict = {
            'reconstructed_image': reconstructed_image,
            'vq_loss': self.current_vq_loss,
            'fim_importance_scores': fim_raw_logits
        }
        return output_dict

@register_model
def ViT_Reconstruction_Model_Default(pretrained: bool = False, **kwargs) -> ViT_Reconstruction_Model:
    model_defaults = dict(
        patch_size=16, encoder_in_chans=3,
        encoder_embed_dim=384, encoder_depth=6, encoder_num_heads=6,
        decoder_embed_dim=192, decoder_depth=3, decoder_num_heads=3, # Decoder ViT blocks
        mlp_ratio=4.0, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        quantizer_dim=256, # Common dimension for both VQs
        bits_vq_high=10,   # Default bits for high-importance VQ
        bits_vq_low=6,     # Default bits for low-importance VQ
        quantizer_commitment_cost=0.25,
        init_values=0.0, use_learnable_pos_emb=False,
        drop_rate=0.0, drop_path_rate=0.1,
        fim_embed_dim=128, fim_depth=2, fim_num_heads=4, fim_drop_rate=0.1,
        fim_routing_threshold=0.6, # Default routing threshold
    )
    
    final_model_constructor_args = model_defaults.copy()

    # Resolve img_size
    if 'input_size' in kwargs: final_model_constructor_args['img_size'] = kwargs['input_size']
    elif 'img_size' in kwargs: final_model_constructor_args['img_size'] = kwargs['img_size']
    else: final_model_constructor_args['img_size'] = 224

    # Override defaults with any relevant keys from kwargs
    for key in final_model_constructor_args.keys():
        if key in kwargs:
            final_model_constructor_args[key] = kwargs[key]
    
    model = ViT_Reconstruction_Model(**final_model_constructor_args)
    model.default_cfg = _cfg() # Define or import _cfg appropriately
    if pretrained:
        print("Warning: `pretrained=True` but no pretrained weight loading implemented in factory.")
    return model