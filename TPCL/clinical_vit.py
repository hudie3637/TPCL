import torch
import torch.nn as nn
from medclip import MedCLIPVisionModelViT, MedCLIPModel
from mode import PhaseAwareAttentionFusionNetwork
from base_config import Config

class ClinicalViT(nn.Module):
    def __init__(self, max_text_length=512, chunk_size=256, overlap_size=50):
        super().__init__()
        # 期相集合
        self.PHASES = ['SP', 'AP', 'VP', 'DP']
        self.PHASE_DIM = 32
        self.DATA_FEAT_DIM = 512
        
        # 文本分块参数
        self.max_text_length = max_text_length
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

        # 通道适配器
        self.phase_channel_adapter = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 视觉编码器
        self.vision_model = MedCLIPVisionModelViT(checkpoint=Config.medclip_vision_checkpoint)
        for param in self.vision_model.parameters():
            param.requires_grad = False

        # 期相感知注意力融合网络
        self.phase_aware_fusion = PhaseAwareAttentionFusionNetwork(in_channels=512, num_phases=4)

        # 文本编码器
        self.text_model = MedCLIPModel().text_model
        for param in self.text_model.parameters():
            param.requires_grad = False
        self.TEXT_EMB_DIM = self.text_model.model.embeddings.word_embeddings.embedding_dim

        # 文本特征投影到视觉特征空间
        self.text_proj = nn.Linear(self.TEXT_EMB_DIM, 512)

        # 期相可学习向量
        self.phase_embeddings = nn.ParameterDict({
            phase: nn.Parameter(torch.randn(self.PHASE_DIM)) for phase in self.PHASES
        })
        # 期相prompt映射到文本embedding空间
        self.prompt_mlp_text = nn.Sequential(
            nn.Linear(self.PHASE_DIM, self.TEXT_EMB_DIM),
            nn.ReLU(),
            nn.Linear(self.TEXT_EMB_DIM, self.TEXT_EMB_DIM)
        )
        # 期相prompt映射到结构化数据空间
        self.prompt_mlp_data = nn.Sequential(
            nn.Linear(self.PHASE_DIM, self.DATA_FEAT_DIM),
            nn.ReLU(),
            nn.Linear(self.DATA_FEAT_DIM, self.DATA_FEAT_DIM)
        )

        # 临床数据编码器
        self.data_encoder = nn.Sequential(
            nn.Linear(len(Config.special_clinical_features) + self.DATA_FEAT_DIM, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 融合分类器
        self.classifier = nn.Sequential(
            nn.Linear(512 + self.TEXT_EMB_DIM + 512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, Config.num_classes)
        )

    def split_text_into_chunks(self, text, tokenizer):
        """
        将长文本分成多个重叠的块
        """
        # 首先对整个文本进行tokenization
        tokens = tokenizer.tokenize(text)
        
        if len(tokens) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
            chunks.append(chunk_text)
            
            # 计算下一个块的起始位置（考虑重叠）
            start = end - self.overlap_size
            if start >= len(tokens):
                break
        
        return chunks

    def extract_text_features_with_chunks(self, text, processor, clip_model, device, dtype):
        """
        将长文本分成多个块，分别提取特征后合并
        """
        # 获取tokenizer用于分块
        tokenizer = processor.tokenizer
        
        # 分块处理
        text_chunks = self.split_text_into_chunks(text, tokenizer)
        
        if len(text_chunks) == 1:
            # 如果只有一个块，直接处理
            inputs = processor(text=[text_chunks[0]], return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                text_features = clip_model.get_text_features(**inputs)
                text_features = text_features.to(dtype=dtype)
            return text_features
        
        # 多个块的情况
        chunk_features = []
        
        for chunk in text_chunks:
            inputs = processor(text=[chunk], return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                chunk_feat = clip_model.get_text_features(**inputs)
                chunk_feat = chunk_feat.to(dtype=dtype)
                chunk_features.append(chunk_feat)
        
        # 合并所有块的特征
        chunk_features = torch.stack(chunk_features, dim=0)  # (num_chunks, feature_dim)
        
        # 使用平均池化合并特征
        text_features = torch.mean(chunk_features, dim=0, keepdim=True)  # (1, feature_dim)
        
        return text_features

    def forward(self, pixel_values, input_ids, attention_mask, special_features, phase_types):
        """
        pixel_values: (B, 4, 1, H, W) 四期相CT
        input_ids: (B, seq_len)
        attention_mask: (B, seq_len)
        special_features: (B, feat_num)
        phase_type: List[List[str]]，每个batch样本的各期相类型
        """
        B, P, C, H, W = pixel_values.shape
        assert P == 4, "Input must contain 4 phases (SP, AP, VP, DP)"
        device = pixel_values.device

        # ------- 1. 四期相图像特征提取 -------
        phase_features = []
        for phase_idx in range(P):
            single_phase = pixel_values[:, phase_idx]  # (B, 1, H, W)
            processed_phase = self.phase_channel_adapter(single_phase)  # (B, 3, H, W)
            vis_output = self.vision_model(pixel_values=processed_phase)  # (B, 512)
            phase_feat = vis_output.pooler_output if hasattr(vis_output, 'pooler_output') else vis_output
            phase_features.append(phase_feat.view(B, 1, 512, 1, 1))
        phase_features = torch.cat(phase_features, dim=1)  # (B, 4, 512, 1, 1)

        # 期相感知注意力融合
        fused_vis_features = self.phase_aware_fusion(phase_features)  # (B, 512, 1, 1)
        vis_features = fused_vis_features.squeeze(-1).squeeze(-1)  # (B, 512)

        # ------- 2. 期相上下文向量 -------
        batch_phase_prompts = []
        for phases in phase_types:  # phases: List[str]，长度4
            batch_phase_prompts.append(self.phase_embeddings[phases[0]])
        phase_prompts = torch.stack(batch_phase_prompts, dim=0).to(device)  # (B, PHASE_DIM)

        # ------- 3. 文本特征提取（期相prompt拼接） -------
        phase_prompts_text = self.prompt_mlp_text(phase_prompts)  # (B, TEXT_EMB_DIM)
        embedding_layer = self.text_model.model.embeddings
        text_embeds = embedding_layer(input_ids)  # (B, seq_len, TEXT_EMB_DIM)
        phase_prompts_text = phase_prompts_text.unsqueeze(1)  # (B, 1, TEXT_EMB_DIM)
        text_embeds = torch.cat([phase_prompts_text, text_embeds], dim=1)  # (B, seq_len+1, TEXT_EMB_DIM)
        attention_mask = torch.cat([torch.ones((B, 1), device=device), attention_mask], dim=1)  # (B, seq_len+1)

        # 动态裁剪到模型最大长度
        max_length = getattr(self.text_model.model.config, 'max_position_embeddings', 512)
        if text_embeds.size(1) > max_length:
            text_embeds = text_embeds[:, :max_length, :]
            attention_mask = attention_mask[:, :max_length]

        # 检查长度一致性
        if text_embeds.size(1) != attention_mask.size(1):
            raise ValueError(f"text_embeds和attention_mask长度不一致: {text_embeds.size(1)} vs {attention_mask.size(1)}")
        if text_embeds.size(1) > max_length:
            print(f"[警告] text_embeds长度超出模型最大长度: {text_embeds.size(1)} > {max_length}")

        text_output = self.text_model.model(
            inputs_embeds=text_embeds,
            attention_mask=attention_mask
        )
        text_features = text_output.pooler_output if hasattr(text_output, "pooler_output") else text_output.last_hidden_state[:, 0]  # (B, TEXT_EMB_DIM)
        # --- 新增：文本特征投影到512维 ---
        text_features_proj = self.text_proj(text_features)  # (B, 512)

        # ------- 4. 结构化数据特征提取（期相prompt拼接） -------
        phase_prompts_data = self.prompt_mlp_data(phase_prompts)  # (B, DATA_FEAT_DIM)
        data_input = torch.cat([special_features, phase_prompts_data], dim=1)  # (B, feat_num+DATA_FEAT_DIM)
        data_features = self.data_encoder(data_input)  # (B, 512)

        # ------- 5. 多模态特征融合与分类 -------
        fused_features = torch.cat([vis_features, text_features, data_features], dim=1)
        outputs = self.classifier(fused_features)

        # 返回 text_features_proj 供相似度计算
        return outputs, vis_features, text_features_proj, data_features

    # 新增方法：用于外部调用的文本特征提取
    def get_text_features_with_chunks(self, text, processor, clip_model, device, dtype):
        """
        外部调用接口：将长文本分成多个块，分别提取特征后合并
        """
        return self.extract_text_features_with_chunks(text, processor, clip_model, device, dtype)
