import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from medclip import MedCLIPProcessor
from base_config import Config
import json
from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold

class MultiPhaseCTDataset(Dataset):
    def __init__(self, processor, mode='train', fold=0, n_splits=5, phase_aware=False):
        self.processor = processor
        self.mode = mode
        self.fold = fold
        self.n_splits = n_splits

        self.phase_aware = phase_aware
        self.phase_order = ['SP', 'AP', 'VP', 'DP']
        self.phase2idx = {p: i for i, p in enumerate(self.phase_order)}
        self.transform = self._get_transforms()
        self.label_df = self._prepare_labels()
        
        # 使用 StratifiedKFold 进行分层采样
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        indices = list(skf.split(self.label_df, self.label_df['label']))
        train_idx, val_idx = indices[self.fold]

        if self.mode == 'train':
            self.label_df = self.label_df.iloc[train_idx]
        elif self.mode == 'val':
            self.label_df = self.label_df.iloc[val_idx]
        else:
            raise ValueError(f"不支持的 mode：{self.mode}，必须是 'train' 或 'val'")

        self.samples = self._build_samples()
        self.tokenizer = MedCLIPProcessor().tokenizer

        # 加载肿瘤分析结果 JSON
        with open(Config.tumor_analysis_path, "r", encoding="utf-8") as f:
            tumor_analysis_data = json.load(f)

        self.tumor_text_info = {}
        for entry in tumor_analysis_data:
            patient_id = str(entry["patient_id"])
            phase = entry["phase"]
            description = entry["description"]
            if patient_id not in self.tumor_text_info:
                self.tumor_text_info[patient_id] = {}
            self.tumor_text_info[patient_id][phase] = description

    def _get_transforms(self):
        return transforms.Compose([
            transforms.Resize((Config.img_size, Config.img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def _prepare_labels(self):
        df = pd.read_excel(Config.label_path)
        expanded = []
        for _, row in df.iterrows():
            rad_id = str(row['放射号'])
            timepoints = self._get_valid_timepoints(rad_id)
            if not timepoints:
                print(f"警告：放射号 {rad_id} 没有找到任何有效的时间点。跳过该样本。")
                continue
            last_tp = sorted(timepoints)[-1]
            for tp in timepoints:
                new_row = row.copy()
                new_row['timepoint'] = tp
                new_row['label'] = self._get_label(row, tp, last_tp)
                expanded.append(new_row)
        return pd.DataFrame(expanded)

    def _get_valid_timepoints(self, rad_id):
        """返回所有有效的时间点（phase_aware控制是否每个相位都必须存在）"""
        timepoints = []
        for folder in os.listdir(Config.data_root):
            if folder.startswith(f"{rad_id}+"):
                tp = folder.split('+')[1]
                if self.phase_aware:
                    # 严格检查四个相位都存在且不为空
                    valid = True
                    for phase in self.phase_order:
                        phase_dir = os.path.join(Config.data_root, folder, phase)
                        if not (os.path.exists(phase_dir) and len(os.listdir(phase_dir)) > 0):
                            valid = False
                            break
                    if valid:
                        timepoints.append(tp)
                else:
                    # 只要能找到这个时间点就行
                    timepoints.append(tp)
        return sorted(set(timepoints))

    def _extract_special_features(self, row):
        features = []
        for feat in Config.special_clinical_features:
            value = row.get(feat, 0)
            value = 0 if pd.isna(value) else value
            features.append(value)
        return torch.tensor(features, dtype=torch.float32)

    def _get_label(self, row, timepoint, last_tp):
        if timepoint == last_tp:
            return 1 if row['病理反应'] in ['PR', 'CR'] else 0
        else:
            return 0

    def _build_samples(self):
        samples = []
        for _, row in self.label_df.iterrows():
            patient_id = str(row['放射号'])
            for phase in ['AP', 'DP', 'SP', 'VP']:
                samples.append((row, phase))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 从 samples 中获取数据
        row, phase = self.samples[idx]
        
        # row 是 pandas.Series，直接通过列名访问
        patient_id = str(row['放射号'])
        timepoint = row['timepoint']
        label = torch.tensor(row['label'], dtype=torch.long)

        # 准备期相图像
        phase_imgs = []
        phase_types = ['AP', 'DP', 'SP', 'VP']
        phase_indices = []
        for phase in self.phase_order:
            # 拼接为 .../rad_id+timepoint/phase
            phase_dir = os.path.join(Config.data_root, f"{patient_id}+{timepoint}", phase)
            if os.path.exists(phase_dir) and len(os.listdir(phase_dir)) > 0:
                slices = sorted(glob.glob(os.path.join(phase_dir, "*.jpg")))
                if slices:
                    mid_slice = slices[len(slices) // 2]
                    img = Image.open(mid_slice).convert('L')
                else:
                    img = Image.new('L', (Config.img_size, Config.img_size), 0)
            else:
                img = Image.new('L', (Config.img_size, Config.img_size), 0)
            img = self.transform(img)
            phase_imgs.append(img)
            phase_types.append(phase)
            phase_indices.append(self.phase2idx[phase])
        pixel_values = torch.stack(phase_imgs, dim=0)  # [4, 1,P, H, W]

        # 生成文本描述
        prompt_text = self._generate_prompt(row)

        # 编码文本
        encoding = self.tokenizer(
            prompt_text,
            max_length=512,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
            add_special_tokens=True
        )
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)

        # 确保编码长度为 512
        if input_ids.shape[0] != 512:
            print(f"[异常] input_ids shape: {input_ids.shape}, pid={patient_id}, prompt: {prompt_text}")
            input_ids = input_ids[:512]
        if attention_mask.shape[0] != 512:
            print(f"[异常] attention_mask shape: {attention_mask.shape}, pid={patient_id}")
            attention_mask = attention_mask[:512]

        assert input_ids.shape[0] == 512, f"input_ids shape: {input_ids.shape}, pid={patient_id}"
        assert attention_mask.shape[0] == 512, f"attention_mask shape: {attention_mask.shape}, pid={patient_id}"

        # 提取特殊临床特征
        special_features = self._extract_special_features(row)

        return {
            'pixel_values': pixel_values,  # [4, 1, H, W]
            'input_ids': input_ids,        # [512]
            'attention_mask': attention_mask, # [512]
            'special_features': special_features,  # [N]
            'label': label,
            'phase_types': phase_types,    # ['SP', 'AP', 'VP', 'DP']
            'phase_indices': torch.tensor(phase_indices, dtype=torch.long),  # [4]
            'patient_id': patient_id
        }

    def _generate_prompt(self, row):
        patient_id = str(row['放射号'])
        phases = ['AP', 'DP', 'SP', 'VP']
        components = []

        # 添加每个期相的文本描述
        for phase in phases:
            if patient_id in self.tumor_text_info and phase in self.tumor_text_info[patient_id]:
                components.append(f"{phase}期相：{self.tumor_text_info[patient_id][phase]}")
            else:
                components.append(f"{phase}期相：未找到相关信息。")

        # 合并文本描述
        prompt = " ".join(components)
        return prompt
