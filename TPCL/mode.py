import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class PhaseAwareAttentionFusionNetwork(nn.Module):
    """期相感知注意力融合网络，将特征差序列通过Transformer处理并生成统一权重"""
    def __init__(self, in_channels=512, num_phases=4):
        super().__init__()
        self.num_phases = num_phases
        self.in_channels = in_channels
        
        # 时序建模：Transformer编码器处理特征差序列
        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(d_model=in_channels, nhead=8, dim_feedforward=2048),
            num_layers=2
        )
        
        # 特征差处理
        self.diff_projection = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        
        # 生成统一注意力权重
        self.weight_generator = nn.Sequential(
            nn.Conv2d(in_channels * (num_phases-1), in_channels//2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels//2, num_phases-1, kernel_size=1),  # 为每个期相生成权重
            nn.Softmax(dim=1)  # 在期相维度归一化
        )
        
        # 最终融合层
        self.final_projection = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

    def forward(self, phase_features):
        """
        输入: (B, num_phases, in_channels, H, W)
        输出: 融合后的特征 (B, in_channels, H, W)
        """
        B, P, C, H, W = phase_features.shape
        
        # 提取平扫期(SP)特征
        sp_feature = phase_features[:, 0].clone()  # (B, C, H, W)
        
        # 计算特征差序列：[AP-SP, VP-SP, DP-SP]
        diff_sequence = []
        for i in range(1, P):  # i=1,2,3 (AP, VP, DP)
            diff = phase_features[:, i] - sp_feature  # 计算差
            diff = self.diff_projection(diff)  # 投影到相同维度
            diff_sequence.append(diff)
        
        # 将特征差序列组织为Transformer输入格式
        # 形状: (sequence_length, B, C*H*W)
        sequence = torch.stack(diff_sequence, dim=0)  # (3, B, C, H, W)
        sequence = sequence.reshape(P-1, B, C, H*W).permute(0, 1, 3, 2).reshape(P-1, B*H*W, C)
        
        # 通过Transformer处理特征差序列
        transformer_output = self.transformer(sequence)  # (3, B*H*W, C)
        
        # 恢复为特征图格式
        transformer_output = transformer_output.reshape(P-1, B, H*W, C).permute(1, 0, 3, 2).reshape(B, P-1, C, H, W)
        
        # 拼接所有期相的特征差
        concat_features = torch.cat([transformer_output[:, i] for i in range(P-1)], dim=1)  # (B, 3*C, H, W)
        
        # 生成统一注意力权重
        attention_weights = self.weight_generator(concat_features)  # (B, 3, H, W)
        
        # 加权融合特征
        fused_feature = torch.zeros(B, C, H, W).to(phase_features.device)
        for i in range(P-1):
            # 为每个期相应用对应的注意力权重
            weighted_feature = transformer_output[:, i] * attention_weights[:, [i]]  # (B, C, H, W)
            fused_feature += weighted_feature
        
        # 最终投影
        fused_feature = self.final_projection(fused_feature)
        
        return fused_feature