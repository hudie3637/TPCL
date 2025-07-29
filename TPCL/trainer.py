import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from medclip import MedCLIPProcessor
from data_loader import MultiPhaseCTDataset
from clinical_vit import ClinicalViT
from metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import label_binarize
import numpy as np
from sklearn.metrics import roc_curve, auc

# 自定义collate_fn
# 1. 自定义collate_fn，保证phase_types等list[str]字段不会丢失
def custom_collate_fn(batch):
    elem = batch[0]
    collated = {}
    for key in elem:
        if isinstance(elem[key], torch.Tensor):
            collated[key] = torch.stack([d[key] for d in batch])
        elif isinstance(elem[key], list) and isinstance(elem[key][0], str):
            collated[key] = [d[key] for d in batch]
        else:
            collated[key] = [d[key] for d in batch]
    return collated

def compute_pairwise_clip_loss(features_list, criterion):
    n = len(features_list)
    device = features_list[0].device
    total_loss = 0
    count = 0
    labels_same = torch.arange(features_list[0].size(0), device=device)
    for i in range(n):
        for j in range(i + 1, n):
            f1 = features_list[i] / features_list[i].norm(dim=1, keepdim=True)
            f2 = features_list[j] / features_list[j].norm(dim=1, keepdim=True)
            logits_1_2 = torch.matmul(f1, f2.T)
            logits_2_1 = logits_1_2.T
            loss_1_2 = criterion(logits_1_2, labels_same)
            loss_2_1 = criterion(logits_2_1, labels_same)
            total_loss += (loss_1_2 + loss_2_1) / 2
            count += 1
    return total_loss / count if count > 0 else 0

def compute_lalign_loss(vis_features, text_features, data_features):
    vis_features = vis_features / vis_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    data_features = data_features / data_features.norm(dim=1, keepdim=True)
    dist_vis_text = torch.sqrt(torch.sum((vis_features - text_features) ** 2, dim=1))
    dist_vis_data = torch.sqrt(torch.sum((vis_features - data_features) ** 2, dim=1))
    dist_text_data = torch.sqrt(torch.sum((text_features - data_features) ** 2, dim=1))
    return (torch.mean(dist_vis_text) + torch.mean(dist_vis_data) + torch.mean(dist_text_data)) / 3

def compute_lxuniform_loss(vis_features, text_features, data_features):
    batch_size = vis_features.size(0)
    device = vis_features.device
    vis_features = vis_features / vis_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
    data_features = data_features / data_features.norm(dim=1, keepdim=True)
    loss = 0
    for j in range(batch_size):
        for k in range(batch_size):
            if j != k:
                loss += torch.exp(-2 * torch.norm(vis_features[j] - text_features[k], p=2)) + \
                        torch.exp(-2 * torch.norm(vis_features[j] - data_features[k], p=2)) + \
                        torch.exp(-2 * torch.norm(text_features[j] - data_features[k], p=2))
    return torch.log(1 / (batch_size * (batch_size - 1)) * loss)

def compute_luniform_loss(vis_features, text_features, data_features):
    def compute_single_modality_loss(features):
        batch_size = features.size(0)
        features = features / features.norm(dim=1, keepdim=True)
        loss = 0
        for j in range(batch_size):
            for k in range(batch_size):
                loss += torch.exp(-2 * torch.norm(features[j] - features[k], p=2))
        return torch.log(1 / (batch_size * batch_size) * loss)
    l_vis = compute_single_modality_loss(vis_features)
    l_text = compute_single_modality_loss(text_features)
    l_data = compute_single_modality_loss(data_features)
    return (l_vis + l_text + l_data) / 3

def compute_phase_consistency_loss(vis_features, text_features, data_features, patient_ids):
    device = vis_features.device
    loss = 0
    count = 0
    patient_ids = np.array(patient_ids) if not isinstance(patient_ids, np.ndarray) else patient_ids
    unique_patients = np.unique(patient_ids)
    for pid in unique_patients:
        idxs = np.where(patient_ids == pid)[0]
        if len(idxs) < 2:
            continue
        vis = vis_features[idxs]
        text = text_features[idxs[0]].unsqueeze(0)
        data = data_features[idxs[0]].unsqueeze(0)
        loss += torch.mean((vis - text) ** 2) + torch.mean((vis - data) ** 2)
        count += 1
    return loss / count if count > 0 else torch.tensor(0., device=device)


class CTTrainer:
    def __init__(self, config):
        self.config = config
        
        # # 使用配置中的随机种子设置
        # self.config.set_seed()
        
        self.processor = MedCLIPProcessor()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_weight = 0.01
        self.lalign_weight = 0.005
        self.luniform_weight = 0.005
        self.lxuniform_weight = 0.005
        self.phase_consistency_weight = 0.01

 
        self.model = ClinicalViT().to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr * 10,
            weight_decay=self.config.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        self.clip_criterion = nn.CrossEntropyLoss()

    def train_epoch(self, epoch, train_loader):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch in progress_bar:
            inputs = {
                'pixel_values': batch['pixel_values'].to(self.device),
                'input_ids': batch['input_ids'].to(self.device),
                'attention_mask': batch['attention_mask'].to(self.device),
                'special_features': batch['special_features'].to(self.device),
                'phase_types': batch['phase_types'],  # 直接传递
            }
            labels = batch['label'].to(self.device)
            patient_ids = batch['patient_id']

            outputs, vis_features, text_features, data_features = self.model(
                pixel_values=inputs['pixel_values'],
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                special_features=inputs['special_features'],
                phase_types=inputs['phase_types']  # 直接传递
            )
            features_list = [vis_features, text_features, data_features]
            l_clip = compute_pairwise_clip_loss(features_list, self.clip_criterion)
            l_align = compute_lalign_loss(vis_features, text_features, data_features)
            l_xuniform = compute_lxuniform_loss(vis_features, text_features, data_features)
            l_uniform = compute_luniform_loss(vis_features, text_features, data_features)

            ce_loss = self.criterion(outputs, labels)

            phase_consistency_loss = compute_phase_consistency_loss(
                vis_features, text_features, data_features, patient_ids
            )

            total_loss_epoch = ce_loss + self.clip_weight * l_clip + \
                            self.phase_consistency_weight * phase_consistency_loss + self.lxuniform_weight * l_xuniform + \
                            self.lalign_weight * l_align + \
            		self.luniform_weight * l_uniform
            self.optimizer.zero_grad()
            total_loss_epoch.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            total_loss += total_loss_epoch.item()
            progress_bar.set_postfix({
                'Total Loss': f"{total_loss_epoch.item():.4f}",
                'CE Loss': f"{ce_loss.item():.4f}",
                'CLIP Loss': f"{l_clip.item():.4f}",
                'LAlign Loss': f"{l_align.item():.4f}",
                'LUniform Loss': f"{l_uniform.item():.4f}",
                'LXUniform Loss': f"{l_xuniform.item():.4f}",
                'PhaseCons Loss': f"{phase_consistency_loss.item():.4f}"
            })

        return total_loss / len(train_loader)

    def evaluate(self, loader):
        self.model.eval()
        all_preds, all_probs, all_labels = [], [], []
        results = []

        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating"):
                inputs = {
                    'pixel_values': batch['pixel_values'].to(self.device),
                    'input_ids': batch['input_ids'].to(self.device),
                    'attention_mask': batch['attention_mask'].to(self.device),
                    'special_features': batch['special_features'].to(self.device),
                    'phase_types': batch['phase_types'],
                }
                outputs, _, _, _ = self.model(
                    pixel_values=inputs['pixel_values'],
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    special_features=inputs['special_features'],
                    phase_types=inputs['phase_types']
                )
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1).cpu().numpy()
                labels = batch['label'].cpu().numpy()
                for pred, label, prob in zip(preds, labels, probs.cpu().numpy()):
                    results.append({
                        '真实标签': label,
                        '预测标签': pred,
                        '预测概率': prob
                    })
                all_probs.append(probs.cpu().numpy())
                all_preds.extend(preds)
                all_labels.extend(labels)

        all_probs = np.vstack(all_probs)
        results_df = pd.DataFrame(results)
        results_df.to_csv("predictions_vs_labels.csv", index=False, encoding="utf-8")
        print("预测值和真实值的对比已保存到 predictions_vs_labels.csv 文件中。")

        return self._compute_metrics(all_labels, all_preds, all_probs)

    def _compute_metrics(self, y_true, y_pred, y_prob):
        metrics = {
            'accuracy': round(accuracy_score(y_true, y_pred), 4),
            'precision': round(precision_score(y_true, y_pred, average='macro'), 4),
            'recall': round(recall_score(y_true, y_pred, average='macro'), 4),
            'f1_score': round(f1_score(y_true, y_pred, average='macro'), 4),
            'roc_auc': round(roc_auc_score(y_true, y_prob[:, 1]), 4)
        }
        return metrics

    def run(self):
        all_metrics = []

        for fold in range(self.config.n_splits):
            # # 为每个折设置相同的随机种子以确保一致性
            # self.config.set_seed()
            
            print(f"\n正在处理第 {fold + 1}/{self.config.n_splits} 折交叉验证...")
            
            train_dataset = MultiPhaseCTDataset(self.processor, mode='train', fold=fold, n_splits=self.config.n_splits)
            val_dataset = MultiPhaseCTDataset(self.processor, mode='val', fold=fold, n_splits=self.config.n_splits)

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=4,
                drop_last=True,
                collate_fn=custom_collate_fn  # 使用自定义的 collate 函数
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=4,
                collate_fn=custom_collate_fn  # 使用自定义的 collate 函数
            )


            for epoch in range(self.config.epochs):
                train_loss = self.train_epoch(epoch, train_loader)
                print(f"Fold {fold + 1}, Epoch {epoch + 1}/{self.config.epochs}, Train Loss: {train_loss:.4f}")

            print("\nEvaluating on validation set...")
            metrics = self.evaluate(val_loader)
            print(f"Fold {fold + 1} Metrics:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision (Macro): {metrics['precision']:.4f}")
            print(f"Recall (Macro): {metrics['recall']:.4f}")
            print(f"F1 Score (Macro): {metrics['f1_score']:.4f}")
            print(f"AUC: {metrics['roc_auc']:.4f}")
            all_metrics.append(metrics)

        # 计算平均指标值
        avg_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0]}
        print("\n平均指标值:")
        print(f"Average Accuracy: {avg_metrics['accuracy']:.4f}")
        print(f"Average Precision (Macro): {avg_metrics['precision']:.4f}")
        print(f"Average Recall (Macro): {avg_metrics['recall']:.4f}")
        print(f"Average F1 Score (Macro): {avg_metrics['f1_score']:.4f}")
        print(f"Average AUC: {avg_metrics['roc_auc']:.4f}")

        return avg_metrics



