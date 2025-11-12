"""
Date: 2025-08-04
"""
import torch
import numpy as np
from tqdm import *
from joblib import load
from utils.metrics import *
from datetime import datetime
from pyhealth.tokenizer import Tokenizer
from pyhealth.datasets import  MIMIC4Dataset, MIMIC3Dataset
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from pyhealth.datasets import split_by_patient, get_dataloader
from pyhealth.medcode import InnerMap
import os
import matplotlib.pyplot as plt


def get_label_tokenizer(label_tokens):
    special_tokens = []
    label_tokenizer = Tokenizer(
        label_tokens,
        special_tokens=special_tokens,
    )
    return label_tokenizer

def batch_to_multihot(label, num_labels: int) -> torch.tensor:

    multihot = torch.zeros((len(label), num_labels))
    for i, l in enumerate(label):
        multihot[i, l] = 1
    return multihot

def prepare_labels(
        labels,
        label_tokenizer: Tokenizer,
    ) -> torch.Tensor:
    labels_index = label_tokenizer.batch_encode_2d(
        labels, padding=False, truncation=False
    )
    num_labels = label_tokenizer.get_vocabulary_size()
    labels = batch_to_multihot(labels_index, num_labels)
    return labels

def parse_datetimes(datetime_strings):
    # print(datetime_strings)
    return [datetime.strptime(dt_str, "%Y-%m-%d %H:%M") for dt_str in datetime_strings]

def timedelta_to_str(tdelta):
    days = tdelta.days
    seconds = tdelta.seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return days * 1440 + hours * 60 + minutes

def convert_to_relative_time(datetime_strings):
    datetimes = parse_datetimes(datetime_strings)
    base_time = min(datetimes)
    return [timedelta_to_str(dt - base_time) for dt in datetimes]

def load_dataset(dataset, root , tables=["diagnoses_icd", "procedures_icd", "prescriptions"], task_fn = None, dev = False):
    if dataset=='mimic3':
        dataset = MIMIC3Dataset(
            root = root,
            dev = dev,
            tables = ['DIAGNOSES_ICD', 'PROCEDURES_ICD', 'PRESCRIPTIONS'],
            code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
            refresh_cache=False,
        )
    elif dataset == 'mimic4':
        dataset = MIMIC4Dataset(
            root=root,
            dev=dev,
            tables=tables,
            code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
            refresh_cache=False,
        )
    else:
        return load(root)
    return dataset.set_task(task_fn=task_fn)

def get_init_tokenizers(task_dataset, keys = ['cond_hist', 'procedures', 'drugs']):
    Tokenizers = {key: Tokenizer(tokens=task_dataset.get_all_tokens(key), special_tokens=["<pad>"]) for key in keys}
    return Tokenizers

def get_parent_tokenizers(task_dataset, keys = ['cond_hist', 'procedures']):
    parent_tokenizers = {}
    dictionary = {'cond_hist':InnerMap.load("ICD9CM"), 'procedures':InnerMap.load("ICD9PROC")}
    for feature_key in keys:
        assert feature_key in dictionary.keys()
        tokens = task_dataset.get_all_tokens(feature_key)
        parent_tokens = set()
        for token in tokens:
            try:
                parent_tokens.update(dictionary[feature_key].get_ancestors(token))
            except:
                continue
        parent_tokenizers[feature_key + '_parent'] = Tokenizer(tokens=list(parent_tokens), special_tokens=["<pad>"])
    return parent_tokenizers

# 自适应温度
def adaptive_temperature(logits):
    probs = F.softmax(logits, dim=1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
    tau = 1.0 + 2.0 * entropy / torch.max(entropy)
    return tau.unsqueeze(1)

# 权重增强
def compute_weights(logits_aux, labels):
    preds = (logits_aux.sigmoid() > 0.5).float()
    misclassified = (preds != labels).float()
    weights = 1.0 + 0.5 * misclassified
    return weights

def train_phase_one(data_loader, model, label_tokenizer, optimizer, device):
    train_loss = 0
    for data in data_loader:
        model.train()
        optimizer.zero_grad()
        # 统一数据访问方式
        if type(data) == dict:
            label = prepare_labels(data['conditions'], label_tokenizer).to(device)
        else:
            label = prepare_labels(data[0]['conditions'], label_tokenizer).to(device)
        out, _, _ = model(data)
        loss = F.binary_cross_entropy_with_logits(out,label)
        # y_prob = torch.sigmoid(out)
        loss.backward()
        optimizer.step()
        train_loss += loss.detach().cpu().numpy()
    return  train_loss

def valid_phase_one(data_loader, model, label_tokenizer, device):
    val_loss= 0
    with torch.no_grad():
        for data in data_loader:
            model.eval()
            # 统一数据访问方式
            if type(data) == dict:
                label = prepare_labels(data['conditions'], label_tokenizer).to(device)
            else:
                label = prepare_labels(data[0]['conditions'], label_tokenizer).to(device)
            out, _, _ = model(data)
            loss = F.binary_cross_entropy_with_logits(out,label)
            val_loss += loss.detach().cpu().numpy()
    return val_loss



def test_phase_one(data_loader, model, label_tokenizer, device, show_progress=False):
    y_t_all, y_p_all = [], []
    with torch.no_grad():
        for data in tqdm(data_loader, desc=" Testing ", leave=False, disable=not show_progress):
            model.eval()
            # 统一数据访问方式
            if type(data) == dict:
                label = prepare_labels(data['conditions'], label_tokenizer).to(device)
            else:
                label = prepare_labels(data[0]['conditions'], label_tokenizer).to(device)
            out, _, _ = model(data)
            y_t = label.cpu().numpy()
            y_p = torch.sigmoid(out).detach().cpu().numpy()
            y_t_all.append(y_t)
            y_p_all.append(y_p)
        y_true = np.concatenate(y_t_all, axis=0)
        y_prob = np.concatenate(y_p_all, axis=0)
    return y_true, y_prob

def evaluate(data_loader, model, label_tokenizer, device, show_progress=False):
    y_t_all, y_p_all = [], []
    with torch.no_grad():
        for data in tqdm(data_loader, desc=" Testing ", leave=False, disable=not show_progress):
            model.eval()
            # 统一数据访问方式
            if type(data) == dict:
                label = prepare_labels(data['conditions'], label_tokenizer).to(device)
            else:
                label = prepare_labels(data[0]['conditions'], label_tokenizer).to(device)
            out, _, _ = model(data)
            y_t = label.cpu().numpy()
            y_p = torch.sigmoid(out).detach().cpu().numpy()
            y_t_all.append(y_t)
            y_p_all.append(y_p)
        y_true = np.concatenate(y_t_all, axis=0)
        y_prob = np.concatenate(y_p_all, axis=0)
        metric_code = code_level(y_true, y_prob)[1]
        metric_visit = visit_level(y_true, y_prob)[1]
    return metric_code, metric_visit

def train_phase_two(train_aux_loader, main_model, aux_model1, aux_model2, label_tokenizer, contrast, optimizer_aux, device):
    aux_model1.train()
    aux_model2.train()
    total_contrast_loss = 0.0
    total_pred_loss = 0.0

    for data in train_aux_loader:
        seq_batch, graph_batch, drug_batch, procedure_batch, _ = data
        x_main = (seq_batch, graph_batch)
        x_aux1 = drug_batch.to(device)
        x_aux2 = procedure_batch.to(device)
        # 统一数据访问方式
        if type(data) == dict:
            label = prepare_labels(data['conditions'], label_tokenizer).to(device)
        else:
            label = prepare_labels(data[0]['conditions'], label_tokenizer).to(device)
        # 主视图编码（仅表征）
        with torch.no_grad():
            logit_main, pe, z_main = main_model(x_main)
        # 辅视图输出 logits + representation
        logit_aux1, z_aux1 = aux_model1(x_aux1, pe)
        logit_aux2, z_aux2 = aux_model2(x_aux2, pe)
        # 表征对齐损失
        contrast_loss1 = contrast.info_nce_loss(z_aux1, z_main)
        contrast_loss2 = contrast.info_nce_loss(z_aux2, z_main)
        contrast_loss = contrast_loss1 + contrast_loss2
        #预测损失
        weights = compute_weights(logit_main, label)
        pred_loss1 = F.binary_cross_entropy_with_logits(logit_aux1, label, weights)
        pred_loss2 = F.binary_cross_entropy_with_logits(logit_aux2, label, weights)
        pred_loss = pred_loss1 + pred_loss2
        #计算总损失
        total_loss = 0.05 * contrast_loss + pred_loss

        optimizer_aux.zero_grad()
        total_loss.backward()
        optimizer_aux.step()
        total_contrast_loss += contrast_loss.detach().cpu().numpy()
        total_pred_loss += pred_loss.detach().cpu().numpy()

    return  total_contrast_loss, total_pred_loss


def test_phase_two(data_loader, model, main_model, label_tokenizer, device, show_progress=False):
    y_t_all, y_p_all = [], []
    with torch.no_grad():
        for data in tqdm(data_loader, desc=" Testing ", leave=False, disable=not show_progress):
            seq_batch, graph_batch, drug_batch, procedure_batch, _ = data
            x_main = (seq_batch, graph_batch)
            model.eval()
            # 统一数据访问方式
            if type(data) == dict:
                label = prepare_labels(data['conditions'], label_tokenizer).to(device)
            else:
                label = prepare_labels(data[0]['conditions'], label_tokenizer).to(device)
            with torch.no_grad():
                logit_main, pe, z_main = main_model(x_main)
            out, _ = model(drug_batch, pe)
            y_t = label.cpu().numpy()
            y_p = torch.sigmoid(out).detach().cpu().numpy()
            y_t_all.append(y_t)
            y_p_all.append(y_p)
        y_true = np.concatenate(y_t_all, axis=0)
        y_prob = np.concatenate(y_p_all, axis=0)
    return y_true, y_prob

def train_phase_kd(aux_models, aux_names, stage, train_aux_loader,  main_model,
                       lambda_kd, lambda_repr, label_tokenizer, contrast, optimizer, device):
    total_contrast_loss = 0.0
    total_pred_loss = 0.0
    total_kd_loss = 0.0
    aux_model = aux_models[stage]
    aux_name = aux_names[stage]

    for data in train_aux_loader:
        main_model.train()
        optimizer.zero_grad()
        seq_batch, graph_batch, drug_batch, procedure_batch, _ = data
        x_main = (seq_batch, graph_batch)
        logit_main, pe, z_main = main_model(x_main)

        with torch.no_grad():
            x_aux = drug_batch.to(device) if aux_name == 'treatment' else procedure_batch.to(device)
            logit_aux, z_aux = aux_model(x_aux, pe)
        if type(data) == dict:
            label = prepare_labels(data['conditions'], label_tokenizer).to(device)
        else:
            label = prepare_labels(data[0]['conditions'], label_tokenizer).to(device)
        # 损失项
        weights = compute_weights(logit_aux, label)
        loss_label = F.binary_cross_entropy_with_logits(logit_main, label, weight=weights)
        loss_repr = contrast.info_nce_loss(z_main, z_aux.detach())
        tau = 2.0
        # tau = adaptive_temperature(logit_aux)
        yT = F.softmax(logit_aux / tau, dim=1).detach()
        yS = F.log_softmax(logit_main / tau, dim=1)
        # loss_kd = F.kl_div(yS, yT, reduction='batchmean') * (torch.mean(tau) ** 2)
        loss_kd = F.kl_div(yS, yT, reduction='batchmean') * (tau ** 2)
        loss = loss_label + lambda_repr * loss_repr + lambda_kd * loss_kd
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_kd_loss += loss_kd.detach().cpu().numpy()
        total_pred_loss += loss_label.detach().cpu().numpy()
        total_contrast_loss += loss_repr.detach().cpu().numpy()
    return  total_pred_loss, total_contrast_loss, total_kd_loss


class LossVisualizer:

    def __init__(self, save_dir='./fig', model_name='model', dataset='dataset'):

        self.save_dir = save_dir
        self.model_name = model_name
        self.dataset = dataset
        self.train_losses = []
        self.val_losses = []
        os.makedirs(self.save_dir, exist_ok=True)

    def add_train_val_loss(self, train_loss, val_loss):
        """添加每个epoch的训练和验证损失"""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

    def plot_and_save(self, dpi=300):
        """绘制并保存损失曲线图"""
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.train_losses) + 1)
        # 绘制训练和验证曲线
        plt.plot(epochs, self.train_losses,
                 label='Train Loss', color='royalblue', linewidth=2, alpha=0.8)
        plt.plot(epochs, self.val_losses,
                 label='Validation Loss', color='darkorange', linewidth=2, alpha=0.8)

        # 标注最佳验证损失点
        best_val_idx = np.argmin(self.val_losses)
        best_val_epoch = epochs[best_val_idx]
        best_val_loss = self.val_losses[best_val_idx]
        plt.scatter(
            best_val_epoch,
            best_val_loss,
            color='gold',
            marker='*',
            s=150,
            edgecolor='black',
            label=f'Best Val Loss (Epoch {best_val_epoch})',
            zorder=3
        )
        # 图表装饰
        plt.title(f'Training Progress ({self.model_name} on {self.dataset})', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend(frameon=True, loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()

        # 保存文件（自动生成唯一文件名）
        fig_path = os.path.join(
            self.save_dir,
            f'loss_curve_{self.model_name}_{self.dataset}.png'
        )
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        return fig_path

    def save_loss_data(self, path):
        """保存损失数据到文件"""
        data = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        torch.save(data, path)

    @classmethod
    def load_loss_data(cls, path):
        """从文件加载损失数据"""
        data = torch.load(path)
        visualizer = cls()
        visualizer.train_losses = data['train_losses']
        visualizer.val_losses = data['val_losses']
        return visualizer


