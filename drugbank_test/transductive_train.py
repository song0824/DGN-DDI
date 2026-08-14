import os
import sys
import time
import contextlib
import platform
import shutil
import copy
import json
import hashlib

# 尽早输出，避免长时间 import 时控制台看似“卡死”
print("[DGN-DDI] 启动中，正在加载 PyTorch / PyG（首次可能需 1–3 分钟）...", flush=True)

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import logging
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, f1_score, recall_score
)
from tqdm import tqdm
from typing import List, Tuple
import random
import argparse

# 导入您项目中的其他模块
from data_preprocessing import DDIDataLoader, DrugDataset, DrugDataLoader, merge_ddi_batches
from models import DGN_DDI
from custom_loss import SigmoidLoss, FocalLoss, PairwiseHingeLoss, AdaptiveLoss

print("[DGN-DDI] 依赖加载完成，进入训练主程序。", flush=True)

# ================== 配置 ==================
# 配置日志
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def _format_hms(seconds: float) -> str:
    s = int(max(0, seconds))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _torch_compile_available() -> tuple:
    """
    检查 torch.compile(inductor) 是否可在当前环境安全使用。
    Windows 上 inductor 生成 C++ 内核需要 MSVC cl.exe；PyG 动态图也会导致大量 graph break。
    """
    if not hasattr(torch, "compile"):
        return False, "当前 PyTorch 版本不支持 torch.compile"
    if platform.system() == "Windows" and shutil.which("cl") is None:
        return False, (
            "Windows 未检测到 MSVC 编译器 cl.exe（torch.compile/inductor 依赖 Visual Studio C++ 构建工具）"
        )
    return True, ""


def _resolve_torch_compile(config: dict) -> None:
    """在训练开始前关闭不可用的 compile，避免首个 step 因 inductor 编译失败而崩溃。"""
    if not config.get("compile_model"):
        return
    ok, reason = _torch_compile_available()
    if ok:
        return
    logger.warning("compile_model 已自动关闭: %s", reason)
    config["compile_model"] = False


_TRAIN_CONFIG_SIGNATURE_KEYS = (
    "hidden_dim",
    "kge_dim",
    "heads_out_feat_params",
    "blocks_params",
    "loss_fn",
    "focal_alpha",
    "focal_gamma",
    "focal_balance_by_counts",
    "sigmoid_label_smoothing",
    "hinge_margin",
    "neg_ent",
    "hard_neg_mode",
    "hard_neg_ratio",
    "hard_neg_max_train_samples",
    "num_candidates_per_pos",
    "hard_neg_selection_mode",
    "hard_neg_pool_strategy",
    "min_recall_for_threshold",
)


def _stable_json_dumps(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _training_signature_payload(config: dict) -> dict:
    return {k: copy.deepcopy(config.get(k)) for k in _TRAIN_CONFIG_SIGNATURE_KEYS}


def get_training_config_signature(config: dict) -> str:
    payload = _training_signature_payload(config)
    return hashlib.sha1(_stable_json_dumps(payload).encode("utf-8")).hexdigest()[:12]


def _unwrap_collate_batch(batch):
    if len(batch) == 3:
        return batch[0], batch[1], batch[2]
    return batch[0], batch[1], None


def _safe_binary_auc(y_true, y_score) -> float:
    """
    当标签只有一类时，sklearn 的 roc_auc_score 会抛异常。
    这里统一降级为 0.5，避免训练/验证中断。
    """
    try:
        yt = np.asarray(y_true)
        ys = _sanitize_scores_np(y_score)
        if yt.size == 0 or len(np.unique(yt)) < 2:
            return 0.5
        return float(roc_auc_score(yt, ys))
    except Exception:
        return 0.5


def _sanitize_scores_np(y_score) -> np.ndarray:
    """将 NaN/Inf 分数替换为有限值，避免 sklearn 指标计算崩溃。"""
    ys = np.asarray(y_score, dtype=np.float64)
    if ys.size == 0:
        return ys
    bad = ~np.isfinite(ys)
    if bad.any():
        ys = np.nan_to_num(ys, nan=0.0, posinf=50.0, neginf=-50.0)
    return ys


def _safe_aupr(y_true, y_score) -> float:
    """AUPR 的安全版本：标签单类或分数含 NaN/Inf 时降级为 0.0。"""
    try:
        yt = np.asarray(y_true)
        ys = _sanitize_scores_np(y_score)
        if yt.size == 0 or len(np.unique(yt)) < 2:
            return 0.0
        return float(average_precision_score(yt, ys))
    except Exception:
        return 0.0


def _sanitize_score_tensor(scores: torch.Tensor, stage: str = "") -> torch.Tensor:
    """清理模型输出分数，并记录 NaN/Inf 数量。"""
    if scores is None or scores.numel() == 0:
        return scores
    nan_n = int(torch.isnan(scores).sum().item())
    inf_n = int(torch.isinf(scores).sum().item())
    if nan_n or inf_n:
        logger.warning(
            "[%s] model scores contain non-finite values: nan=%d inf=%d / %d",
            stage or "eval", nan_n, inf_n, scores.numel(),
        )
    return torch.nan_to_num(scores, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)


def _warn_if_model_nonfinite(model: torch.nn.Module, stage: str = "") -> None:
    """检测参数/梯度是否出现 NaN/Inf，便于定位数值发散。"""
    bad_params = []
    for name, p in model.named_parameters():
        if p is None or not p.requires_grad:
            continue
        if not torch.isfinite(p).all():
            bad_params.append(name)
    if bad_params:
        logger.warning(
            "[%s] non-finite model parameters (%d): %s%s",
            stage or "model",
            len(bad_params),
            ", ".join(bad_params[:5]),
            " ..." if len(bad_params) > 5 else "",
        )


def _parse_hhmm(value: str):
    """解析 HH:MM / HHMM，返回 (hour, minute)。"""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if ":" in s:
        hh, mm = s.split(":", 1)
    elif len(s) == 4 and s.isdigit():
        hh, mm = s[:2], s[2:]
    else:
        raise ValueError(f"Invalid time format: {value!r}, expected HH:MM")
    hour, minute = int(hh), int(mm)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"Invalid time value: {value!r}")
    return hour, minute


def _next_stop_deadline(stop_at_hhmm: str, now: float = None) -> float:
    """
    计算下一次停止时刻的 unix timestamp。
    若当前已过今日 stop_at，则落到次日同一时刻（适配 20:30->08:00 跨夜窗口）。
    """
    hour, minute = _parse_hhmm(stop_at_hhmm)
    import datetime as _dt
    now_dt = _dt.datetime.fromtimestamp(now if now is not None else time.time())
    stop_dt = now_dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if stop_dt <= now_dt:
        stop_dt = stop_dt + _dt.timedelta(days=1)
    return stop_dt.timestamp()


def _atomic_torch_save(obj, path: str) -> None:
    """先写临时文件再 replace，避免强杀导致半截 checkpoint。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def get_default_config():
    """获取默认配置，包含模型和训练的超参数"""
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _drugbank_dir = os.path.join(_script_dir, "drugbank")
    _fold0 = os.path.join(_drugbank_dir, "fold0")
    return {
        # 数据与路径（相对当前脚本目录）
        'train_csv': os.path.join(_fold0, "train.csv"),
        'val_csv': os.path.join(_fold0, "val.csv"),
        'test_csv': os.path.join(_fold0, "test.csv"),
        'save_dir': os.path.join(_script_dir, "checkpoints"),
        'log_dir': os.path.join(_script_dir, "logs"),
        'resume_from_checkpoint': '',
        'allow_resume_config_mismatch': False,

        # 数据处理
        'neg_ent': 3,
        'subset_size': 5000,
        'train_ratio': 0.7,
        'val_ratio': 0.1,

        'hidden_dim': 256,
        'kge_dim': 256,
        'heads_out_feat_params': [64, 64, 64],
        'blocks_params': [4, 4, 4],

        # 训练超参数
        'batch_size': 64,
        'epochs': 20,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'lr_decay_rate': 0.98,
        'grad_clip_value': 2.0,
        'loss_fn': 'focal',
        'focal_alpha': 0.5,
        'focal_gamma': 2.0,
        'focal_balance_by_counts': True,
        'sigmoid_label_smoothing': 0.0,
        'hinge_margin': 1.0,
        'min_recall_for_threshold': 0.0,

        # 硬负
        'use_hard_negative_sampling': True,
        'hard_neg_mode': 'subset_epoch',
        'hard_neg_start_epoch': 3,
        'hard_neg_frequency': 3,
        'num_candidates_per_pos': 5,
        'hard_neg_ratio': 0.7,
        'hard_neg_max_train_samples': 10000,
        'online_hard_candidates': 4,

        'use_amp': True,
        'compile_model': False,
        'gradient_accumulation_steps': 1,

        # 训练加速：启动时预热二部图内存缓存（需先运行 precompute_bipartite_cache.py）
        'warm_bipartite_cache': True,
        'warm_bipartite_max_pairs': None,
        'warm_bipartite_splits': ["train", "val", "test"],
        'csv_max_rows': None,

        # 早停与保存（与 round_patience 对齐：连续 3 epoch 无提升停）
        'patience': 3,
        'save_freq': 10,
        'random_seed': 42,
        # 中途可续训：按 batch / 时间落盘；到 stop_at 也写 mid-epoch checkpoint
        'ckpt_every_batches': 50,
        'ckpt_every_minutes': 20,

        # 多轮 = 每 epoch 一轮：训完即测试记录；连续 3 个 epoch 无明显提升则停止
        'stop_at': '',
        'skip_final_test': False,
        'record_test_results': True,
        'progress_file': '',
        'test_results_dir': '',
        'test_every_n_epochs': 1,
        'epochs_per_round': 1,        # 每个 epoch 就是一轮
        'max_rounds': 200,
        'round_patience': 3,          # 连续 3 个 epoch 无明显提升则停止
        'min_round_improve': 1e-4,
        'test_after_each_round': True,
        'hard_neg_selection_mode': 'per_positive',
        'hard_neg_pool_strategy': 'random_subset',
    }


def _apply_hard_negative_schedule(config: dict, max_train_samples: int) -> None:
    """前 hard_neg_start_epoch 轮纯随机；之后每 hard_neg_frequency 轮 subset_epoch 预生成硬负。"""
    config["use_hard_negative_sampling"] = True
    config["hard_neg_mode"] = "subset_epoch"
    config["hard_neg_start_epoch"] = 3
    config["hard_neg_frequency"] = 3
    config["hard_neg_max_train_samples"] = max_train_samples


def apply_training_profile(config: dict, profile: str) -> None:
    """
    训练阶段配置：smoke 子集快验；full 全数据长训。
    与命令行 --profile 配合；全量前若不想续接烟测权重请用 --fresh。
    """
    p = (profile or "smoke").lower().strip()
    if p == "smoke":
        config["subset_size"] = 5000
        config["epochs"] = 20
        config["patience"] = 3
        # 烟测优先稳定与吞吐：减负样本数并使用更稳健的损失。
        config["neg_ent"] = 1
        config["loss_fn"] = "sigmoid"
        config["sigmoid_label_smoothing"] = 0.1
        # 烟测仅预热训练子集涉及的药物对，避免启动时长时间无输出
        config["warm_bipartite_max_pairs"] = 5000
        config["warm_bipartite_splits"] = ["train"]
        config["csv_max_rows"] = 20000
        # 烟测用小 batch + 轻量 block，显著降低每 step 串行药物对数与注意力开销
        config["batch_size"] = 32
        config["blocks_params"] = [2, 2, 2]
        config["heads_out_feat_params"] = [128, 128, 128]  # 2*128=256，与 kge_dim 一致
        config["epochs_per_round"] = 1
        config["round_patience"] = 3
        config["test_every_n_epochs"] = 1
        config["hard_neg_selection_mode"] = "per_positive"
        _apply_hard_negative_schedule(config, max_train_samples=5000)
    elif p == "full":
        config["subset_size"] = None
        config["epochs"] = 200
        config["patience"] = 6
        config["csv_max_rows"] = None
        # 默认改为更贴近排序任务的 sigmoid 目标；focal/hinge 可通过 CLI 覆盖。
        # neg_ent=3: 与旧版持平，避免二部图磁盘I/O成为速度瓶颈（每批查找对数×67%增量影响大）
        config["neg_ent"] = 3
        config["batch_size"] = 48
        config["compile_model"] = True
        config["loss_fn"] = "sigmoid"
        config["focal_alpha"] = 0.75
        config["focal_gamma"] = 2.0
        config["sigmoid_label_smoothing"] = 0.0
        config["hinge_margin"] = 1.0
        config["lr"] = 2e-4
        config["weight_decay"] = 5e-6
        config["min_recall_for_threshold"] = 0.0
        config["blocks_params"] = [4, 4, 4]
        config["heads_out_feat_params"] = [64, 64, 64]
        # 全量二部图靠磁盘缓存，避免启动时预热 15 万对
        config["warm_bipartite_cache"] = False
        # 每 epoch 一轮；放宽 patience，减少把缓慢上升误判为平台期。
        config["epochs_per_round"] = 1
        config["round_patience"] = 8
        config["max_rounds"] = 200
        config["test_after_each_round"] = True
        config["test_every_n_epochs"] = 5
        _apply_hard_negative_schedule(config, max_train_samples=20000)
        # 从第2个 epoch 开始使用更广覆盖的硬负样本，降低硬负比例避免过拟合噪声候选。
        config["hard_neg_start_epoch"] = 1
        config["hard_neg_frequency"] = 1
        config["num_candidates_per_pos"] = 16
        config["hard_neg_ratio"] = 0.5
        config["hard_neg_selection_mode"] = "per_positive"
        config["hard_neg_pool_strategy"] = "random_subset"
    else:
        raise ValueError(f"Unknown profile {profile!r}, expected 'smoke' or 'full'")

    if p == "full":
        _check_bipartite_cache_for_full(config)

    _resolve_torch_compile(config)


def _check_bipartite_cache_for_full(config: dict) -> None:
    """全量训练前检查二部图磁盘缓存，缺失会显著拖慢 collate。"""
    train_csv = config.get("train_csv", "")
    bank = os.path.dirname(os.path.dirname(os.path.abspath(train_csv)))
    cache_dir = os.path.join(bank, "bipartite_cache")
    n_pt = 0
    if os.path.isdir(cache_dir):
        try:
            n_pt = sum(1 for f in os.listdir(cache_dir) if f.endswith(".pt"))
        except OSError:
            pass
    if n_pt < 1000:
        logger.warning(
            "全量训练建议先运行: python precompute_bipartite_cache.py "
            "(当前 bipartite_cache 仅 %d 个 .pt，collate 将大量现场计算二部图)",
            n_pt,
        )
    else:
        logger.info("bipartite_cache 已就绪: %d 个 .pt 文件", n_pt)


# ================== 硬负样本采样器 ==================

class HardNegativeSampler:
    """
    用于在训练过程中动态生成硬负样本的采样器。
    它会根据模型对候选负样本的预测分数来选择最难区分的样本。
    """

    def __init__(
            self,
            model: DGN_DDI,
            ddi_loader: DDIDataLoader,
            neg_ent: int,
            device: torch.device
    ):
        """
        Args:
            model: 待用于采样的DGN-DDI模型。
            ddi_loader: 包含药物图数据和ID映射的DDIDataLoader实例。
            neg_ent: 每个正样本需要采样的负样本数量。
            device: 训练设备（cuda或cpu）。
        """
        self.model = model
        self.ddi_loader = ddi_loader
        self.neg_ent = neg_ent
        self.device = device
        self.selection_mode = "per_positive"

        # 获取所有可能的药物ID列表，用于生成候选负样本
        self.drug_ids = list(ddi_loader.int_to_drug_id.values())
        logger.info(f"HardNegativeSampler initialized with {len(self.drug_ids)} drug candidates.")

    def get_candidate_negatives(
            self,
            pos_triples: List[Tuple[str, str, str]],
            num_candidates: int
    ) -> List[Tuple[str, str, str]]:
        """
        为给定的正样本列表生成一批随机候选负样本。
        """
        candidate_negatives = []
        for h, t, r in pos_triples:
            for _ in range(num_candidates):
                # 随机选择一个头或尾部进行替换
                if random.random() < 0.5:
                    neg_h = random.choice(self.drug_ids)
                    if neg_h == h and len(self.drug_ids) > 1:
                        for _retry in range(4):
                            neg_h = random.choice(self.drug_ids)
                            if neg_h != h:
                                break
                    neg_t = t
                else:
                    neg_h = h
                    neg_t = random.choice(self.drug_ids)
                    if neg_t == t and len(self.drug_ids) > 1:
                        for _retry in range(4):
                            neg_t = random.choice(self.drug_ids)
                            if neg_t != t:
                                break
                candidate_negatives.append((neg_h, neg_t, r))
        return candidate_negatives

    def sample_hard_negatives_from_batch(
            self,
            pos_triples: List[Tuple[str, str, str]],
            num_candidates_per_pos: int = 20
    ) -> List[Tuple[str, str, str]]:
        """
        根据模型预测分数，从候选样本中筛选出硬负样本。
        优化版本：复用临时DataLoader，避免重复初始化
        """
        if not pos_triples:
            return []

        # 1. 生成候选负样本
        candidate_neg_triples = self.get_candidate_negatives(
            pos_triples, num_candidates=num_candidates_per_pos
        )

        if not candidate_neg_triples:
            return []

        # 2. 批量评估候选负样本
        self.model.eval()
        candidate_scores = []

        # 复用临时数据集和DataLoader（仅初始化一次）
        if not hasattr(self, '_temp_loader'):
            # 初始化临时数据集（__new__ 不会走 __init__，需补齐 collate 所需属性）
            self._temp_dataset = DrugDataset.__new__(DrugDataset)
            self._temp_dataset.data_loader = self.ddi_loader
            self._temp_dataset.neg_ent = 0
            self._temp_dataset.drug_ids = np.array(self.drug_ids)
            self._temp_dataset.return_pos_triples = False
            self._temp_dataset.hard_neg_ratio = 0.0
            self._temp_dataset.use_hard_negatives = False
            self._temp_dataset.hard_negatives_cache = {}
            self._temp_dataset._drug_data_cache = {}
            self._temp_dataset.tri_list = []
            # 空关系统计，避免 collate 负采样依赖缺失
            self._temp_dataset.ALL_TRUE_H_WITH_TR = {}
            self._temp_dataset.ALL_TRUE_T_WITH_HR = {}
            self._temp_dataset.FREQ_REL = {}
            self._temp_dataset.ALL_H_WITH_R = {}
            self._temp_dataset.ALL_T_WITH_R = {}
            self._temp_dataset.ALL_HEAD_PER_TAIL = {}
            self._temp_dataset.ALL_TAIL_PER_HEAD = {}
            self._temp_loader = DrugDataLoader(self._temp_dataset, batch_size=128, shuffle=False)

        # 更新临时数据集的三元组列表（避免重复创建数据集）
        self._temp_dataset.tri_list = candidate_neg_triples
        if not hasattr(self._temp_dataset, "_drug_data_cache") or self._temp_dataset._drug_data_cache is None:
            self._temp_dataset._drug_data_cache = {}

        with torch.no_grad():
            for batch in self._temp_loader:  # 复用已创建的temp_loader
                if len(batch) == 3:
                    pos_tri, neg_tri, _ = batch
                else:
                    pos_tri, neg_tri = batch
                if pos_tri[0] is None:
                    continue

                # 处理候选负样本（代码不变）
                pos_h, pos_t, pos_r, pos_bg = pos_tri
                pos_h = pos_h.to(self.device)
                pos_t = pos_t.to(self.device)
                pos_r = pos_r.to(self.device)
                pos_bg = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                          for k, v in pos_bg.items()}

                scores = self.model(pos_h, pos_t, pos_r, pos_bg)
                candidate_scores.append(scores.cpu())

        self.model.train()  # 切换回训练模式

        if not candidate_scores:
            return []

        all_scores = torch.cat(candidate_scores)
        if all_scores.numel() == 0:
            return []

        selection_mode = str(self.selection_mode).lower().strip()
        if selection_mode == "per_positive":
            hard_negatives = []
            span = max(1, int(num_candidates_per_pos))
            per_pos_topk = max(1, int(self.neg_ent))
            for idx, _ in enumerate(pos_triples):
                start = idx * span
                end = min(start + span, len(candidate_neg_triples))
                if start >= end:
                    continue
                local_scores = all_scores[start:end]
                k = min(per_pos_topk, local_scores.numel())
                top_local = torch.topk(local_scores, k=k, largest=True).indices.tolist()
                for local_idx in top_local:
                    hard_negatives.append(candidate_neg_triples[start + int(local_idx)])
            return hard_negatives

        # 兼容旧逻辑：对整批候选做全局 top-k。
        num_hard_negatives = min(len(pos_triples) * self.neg_ent, len(all_scores))
        if num_hard_negatives > 0:
            top_k_indices = torch.topk(all_scores, k=num_hard_negatives, largest=True).indices
            hard_negatives = [candidate_neg_triples[i] for i in top_k_indices]
            return hard_negatives

        return []


# ================== 改进的数据集类 ==================

class HardNegativeAwareDataset(DrugDataset):
    """支持硬负样本采样的数据集"""

    def __init__(self, csv_path, data_loader, neg_ent=1, disjoint_split=True, shuffle=True,
                 return_pos_triples=False, hard_neg_ratio=0.7, csv_max_rows=None):
        super().__init__(
            csv_path, data_loader, neg_ent, disjoint_split, shuffle,
            return_pos_triples=return_pos_triples, hard_neg_ratio=hard_neg_ratio,
            csv_max_rows=csv_max_rows,
        )
        self.use_hard_negatives = False
        self.hard_negatives_cache = {}  # 缓存硬负样本

    def set_hard_negatives(self, hard_negatives_dict):
        """设置硬负样本缓存"""
        self.hard_negatives_cache = hard_negatives_dict
        self.use_hard_negatives = len(hard_negatives_dict) > 0

    def enable_hard_negatives(self):
        """启用硬负样本采样"""
        self.use_hard_negatives = True

    def disable_hard_negatives(self):
        """禁用硬负样本采样"""
        self.use_hard_negatives = False

    def _generate_neg_samples_with_hard(self, h, t, r, hard_neg_ratio=0.7):
        """结合硬负样本和随机负样本的生成策略"""
        total_neg = self.neg_ent

        # 如果启用硬负样本且有缓存
        if self.use_hard_negatives and (h, t, r) in self.hard_negatives_cache:
            hard_negs = self.hard_negatives_cache[(h, t, r)]
            num_hard = min(int(total_neg * hard_neg_ratio), len(hard_negs))
            num_random = total_neg - num_hard

            # 硬负样本
            selected_hard = random.sample(hard_negs, num_hard) if num_hard > 0 else []

            # 随机负样本
            if num_random > 0:
                random_neg_heads, random_neg_tails = self._generate_neg_samples(h, t, r)
                # 调整数量
                if len(random_neg_heads) + len(random_neg_tails) > num_random:
                    total_random = len(random_neg_heads) + len(random_neg_tails)
                    keep_heads = int(len(random_neg_heads) * num_random / total_random)
                    random_neg_heads = random_neg_heads[:keep_heads]
                    random_neg_tails = random_neg_tails[:num_random - keep_heads]
            else:
                random_neg_heads, random_neg_tails = [], []

            # 合并硬负样本和随机负样本
            hard_neg_heads = [neg_h for neg_h, neg_t, neg_r in selected_hard if neg_t == t and neg_r == r]
            hard_neg_tails = [neg_t for neg_h, neg_t, neg_r in selected_hard if neg_h == h and neg_r == r]

            fh = list(hard_neg_heads) + list(random_neg_heads)
            ft = list(hard_neg_tails) + list(random_neg_tails)
            return np.array(fh, dtype=object), np.array(ft, dtype=object)
        else:
            # 回退到标准负样本生成
            return self._generate_neg_samples(h, t, r)

    def _neg_sample_heads_tails(self, h, t, r):
        if self.use_hard_negatives and (h, t, r) in self.hard_negatives_cache:
            return self._generate_neg_samples_with_hard(h, t, r, self.hard_neg_ratio)
        return self._generate_neg_samples(h, t, r)


# ================== 训练器 ==================

class DDITrainer:
    def __init__(self, config):
        self.config = config
        self.config_signature = get_training_config_signature(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        logger.info("Training config signature: %s", self.config_signature)

        # 1. 初始化数据加载器 (所有数据的基础)
        self.ddi_loader = DDIDataLoader()

        # 2. 加载数据集
        self.load_data()

        # 3. 初始化模型（关系统一用 ddi_loader.rel_total）
        rel_n = max(1, int(getattr(self.ddi_loader, "rel_total", 0) or 0))
        self.model = DGN_DDI(
            hidden_dim=self.config['hidden_dim'],
            kge_dim=self.config['kge_dim'],
            rel_total=rel_n,
            heads_out_feat_params=self.config['heads_out_feat_params'],
            blocks_params=self.config['blocks_params'],
            drug_graph_dict=self.ddi_loader.drug_graph_dict,
            int_to_drug_id=self.ddi_loader.int_to_drug_id,
            device=self.device
        ).to(self.device)
        if self.config.get("compile_model"):
            ok, reason = _torch_compile_available()
            if ok:
                try:
                    self.model = torch.compile(self.model)
                    logger.info("torch.compile applied to model")
                except Exception as e:
                    logger.warning("torch.compile skipped: %s", e)
            else:
                logger.warning("torch.compile skipped: %s", reason)

        # 4. 初始化优化器和学习率调度器
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config['lr'],
                                     weight_decay=self.config['weight_decay'])
        # warmup 前3个epoch线性升温(0.1→1.0×lr)，之后cosine退火到 eta_min
        _warmup_epochs = 3
        _total_epochs = max(self.config['epochs'], _warmup_epochs + 1)
        _warmup_sched = LinearLR(
            self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=_warmup_epochs
        )
        _cosine_sched = CosineAnnealingLR(
            self.optimizer, T_max=max(1, _total_epochs - _warmup_epochs), eta_min=1e-6
        )
        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[_warmup_sched, _cosine_sched],
            milestones=[_warmup_epochs]
        )

        # 5. 初始化损失函数
        self.loss_fn = self._get_loss_function(self.config['loss_fn']).to(self.device)

        # AMP
        self.use_amp = bool(self.config.get("use_amp", True)) and self.device.type == "cuda"
        if self.use_amp:
            from torch.amp import GradScaler, autocast
            self.scaler = GradScaler("cuda")
            self._autocast = lambda: autocast("cuda", enabled=True)
        else:
            self.scaler = None
            self._autocast = contextlib.nullcontext

        # 6. 硬负采样器：online_batch 必用；subset_epoch 仅当 use_hard_negative_sampling
        hnm = self.config.get("hard_neg_mode", "off")
        need_sampler = hnm == "online_batch" or (
            hnm == "subset_epoch" and self.config.get("use_hard_negative_sampling", False)
        )
        if need_sampler:
            self.hard_neg_sampler = HardNegativeSampler(
                model=self.model,
                ddi_loader=self.ddi_loader,
                neg_ent=self.config['neg_ent'],
                device=self.device
            )
            self.hard_neg_sampler.selection_mode = str(
                self.config.get("hard_neg_selection_mode", "per_positive")
            )
            logger.info("Hard negative sampler initialized (mode=%s)", hnm)
        else:
            self.hard_neg_sampler = None

        self.best_val_auc = 0.0
        self.best_val_aupr = 0.0
        self.patience_counter = 0
        self.best_val_f1 = 0.0
        self.best_threshold_for_test = 0.5
        # 多轮训练状态：训一轮→测试→继续，直到无明显提升
        self.current_round = 0
        self.rounds_without_improve = 0
        self.best_round_score = 0.0  # 以 Val AUC 为主分数
        self.training_converged = False
        self.epoch_in_round = 0
        self.global_epoch = 0
        # 中途续训：从 checkpoint 恢复的 batch 偏移（同一 epoch 内）
        self._resume_batch = 0
        self._epoch_in_progress = False
        self._stop_deadline = None

    def _get_loss_function(self, loss_name):
        """根据配置返回损失函数实例"""
        if loss_name == 'sigmoid':
            return SigmoidLoss(
                label_smoothing=float(self.config.get("sigmoid_label_smoothing", 0.0))
            )
        elif loss_name == 'focal':
            return FocalLoss(
                alpha=float(self.config.get("focal_alpha", 0.5)),
                gamma=float(self.config.get("focal_gamma", 2.0)),
                balance_by_counts=bool(self.config.get("focal_balance_by_counts", True)),
            )
        elif loss_name == 'hinge':
            return PairwiseHingeLoss(margin=float(self.config.get("hinge_margin", 1.0)))
        elif loss_name == 'adaptive':
            return AdaptiveLoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_name}")

    def load_data(self):
        """加载训练、验证和测试数据集"""
        logger.info("Loading DDI datasets...")
        for csv_path in [self.config['train_csv'], self.config['val_csv'], self.config['test_csv']]:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Dataset file {csv_path} not found")

        tr_meta = self.config.get("hard_neg_mode") == "online_batch"
        csv_cap = self.config.get("csv_max_rows")
        self.train_dataset = HardNegativeAwareDataset(
            self.config['train_csv'], self.ddi_loader,
            neg_ent=self.config['neg_ent'], shuffle=True,
            return_pos_triples=tr_meta,
            hard_neg_ratio=float(self.config.get("hard_neg_ratio", 0.7)),
            csv_max_rows=csv_cap,
        )
        self.val_dataset = DrugDataset(
            self.config['val_csv'], self.ddi_loader,
            neg_ent=self.config['neg_ent'], shuffle=False,
            csv_max_rows=csv_cap,
        )
        self.test_dataset = DrugDataset(
            self.config['test_csv'], self.ddi_loader,
            neg_ent=self.config['neg_ent'], shuffle=False,
            csv_max_rows=csv_cap,
        )

        # 数据子集处理
        subset_size = self.config.get('subset_size')
        if subset_size and subset_size > 0:
            logger.info(f"Using a subset of the data for training: {subset_size} samples.")
            subset_size = min(subset_size, len(self.train_dataset))
            self.train_dataset.tri_list = self.train_dataset.tri_list[:subset_size]

            val_subset_size = int(subset_size * (self.config['val_ratio'] / self.config['train_ratio']))
            val_subset_size = min(val_subset_size, len(self.val_dataset))
            self.val_dataset.tri_list = self.val_dataset.tri_list[:val_subset_size]

            test_subset_size = val_subset_size
            test_subset_size = min(test_subset_size, len(self.test_dataset))
            self.test_dataset.tri_list = self.test_dataset.tri_list[:test_subset_size]

            logger.info(
                f"Active subset sizes - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")

        use_pin = self.device.type == "cuda"
        self.train_loader = DrugDataLoader(
            self.train_dataset, batch_size=self.config['batch_size'], shuffle=True, pin_memory=use_pin
        )
        self.val_loader = DrugDataLoader(
            self.val_dataset, batch_size=self.config['batch_size'], shuffle=False, pin_memory=use_pin
        )
        self.test_loader = DrugDataLoader(
            self.test_dataset, batch_size=self.config['batch_size'], shuffle=False, pin_memory=use_pin
        )

        logger.info(
            f"Data loaded - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")

        if self.config.get("warm_bipartite_cache", True):
            max_pairs = self.config.get("warm_bipartite_max_pairs")
            split_map = {
                "train": self.train_dataset,
                "val": self.val_dataset,
                "test": self.test_dataset,
            }
            warm_splits = self.config.get("warm_bipartite_splits", ["train", "val", "test"])
            for name in warm_splits:
                ds = split_map.get(name)
                if ds is None:
                    continue
                logger.info("Bipartite warm [%s]: scanning pairs (max_pairs=%s)...", name, max_pairs)
                total, loaded = ds.warm_bipartite_cache(max_pairs=max_pairs)
                logger.info(
                    "Bipartite warm [%s]: %d unique pairs, %d loaded into memory cache",
                    name, total, loaded,
                )

    def _hard_neg_cache_path(self, epoch: int) -> str:
        return os.path.join(
            self.config["save_dir"],
            f"hard_neg_cache_{self.config_signature}_e{epoch}.pt",
        )

    def _save_hard_neg_cache(self, epoch: int, hard_negatives_dict: dict) -> None:
        path = self._hard_neg_cache_path(epoch)
        try:
            _atomic_torch_save(hard_negatives_dict, path)
            logger.info("Hard-neg cache saved: %s (%d keys)", path, len(hard_negatives_dict))
        except Exception as e:
            logger.warning("Failed to save hard-neg cache %s: %s", path, e)

    def _load_hard_neg_cache(self, epoch: int) -> bool:
        path = self._hard_neg_cache_path(epoch)
        if not os.path.exists(path):
            return False
        try:
            hard_negatives_dict = torch.load(path, map_location="cpu", weights_only=False)
            self.train_dataset.set_hard_negatives(hard_negatives_dict)
            logger.info(
                "Loaded hard-neg cache for epoch %d from %s (%d keys)",
                epoch, path, len(hard_negatives_dict),
            )
            return True
        except Exception as e:
            logger.warning("Failed to load hard-neg cache %s: %s", path, e)
            return False

    def _train_index_order(self, epoch: int):
        """同一 epoch 内确定性打乱，便于 mid-epoch 按 batch 精确续训。"""
        n = len(self.train_dataset)
        g = torch.Generator()
        g.manual_seed(int(self.config.get("random_seed", 42)) + int(epoch) * 10007)
        return torch.randperm(n, generator=g).tolist()

    def _sample_epoch_subset(self, triples, cap: int, epoch: int):
        """对硬负候选池做可复现随机抽样，避免始终只看前 N 条三元组。"""
        if cap <= 0 or len(triples) <= cap:
            return list(triples)
        rng = random.Random(int(self.config.get("random_seed", 42)) + int(epoch) * 9973)
        return rng.sample(list(triples), k=cap)

    def _make_resume_train_loader(self, epoch: int, start_batch: int = 0):
        """
        从 start_batch 起构造本 epoch 剩余数据的 loader（跳过已训 batch，避免重 collate）。
        返回 (loader, global_start_batch, total_batches)；无剩余则 loader=None。
        """
        from torch.utils.data import Subset, DataLoader

        bs = max(1, int(self.config["batch_size"]))
        order = self._train_index_order(epoch)
        total_batches = (len(order) + bs - 1) // bs if order else 0
        start_batch = max(0, min(int(start_batch), total_batches))
        remaining = order[start_batch * bs:]
        if not remaining:
            return None, start_batch, total_batches
        subset = Subset(self.train_dataset, remaining)
        use_pin = self.device.type == "cuda"
        loader = DataLoader(
            subset,
            batch_size=bs,
            shuffle=False,
            collate_fn=self.train_dataset.collate_fn,
            num_workers=0,
            pin_memory=use_pin,
        )
        return loader, start_batch, total_batches

    def _should_regenerate_subset_hard(self, epoch) -> bool:
        """仅 hard_neg_mode==subset_epoch 时，按频率重算硬负缓存。"""
        if not self.config.get("use_hard_negative_sampling", False):
            return False
        if self.config.get("hard_neg_mode", "off") != "subset_epoch":
            return False
        if self.hard_neg_sampler is None:
            return False
        if epoch < self.config.get("hard_neg_start_epoch", 5):
            return False
        if epoch % self.config.get("hard_neg_frequency", 3) != 0:
            return False
        return True

    def _generate_hard_negatives_for_epoch(self, epoch):
        """为当前epoch生成硬负样本（限量 train 子集，避免全量数天）。"""
        if not self._should_regenerate_subset_hard(epoch) or self.hard_neg_sampler is None:
            return
        logger.info(f"Generating hard negatives for epoch {epoch} (subset_epoch mode)...")
        current_train_loader = self.train_loader
        pos_triples = self.train_dataset.tri_list.copy()
        cap = int(self.config.get("hard_neg_max_train_samples", 10_000))
        if cap > 0 and len(pos_triples) > cap:
            logger.info(f"Limiting hard-neg pool from {len(pos_triples)} to {cap} triples")
            pool_strategy = str(self.config.get("hard_neg_pool_strategy", "random_subset")).lower().strip()
            if pool_strategy == "front_slice":
                pos_triples = pos_triples[:cap]
            else:
                pos_triples = self._sample_epoch_subset(pos_triples, cap, epoch)
        # 批量生成硬负样本
        batch_size = 100  # 控制批次大小以避免内存问题
        hard_negatives_dict = {}
        total_batches = (len(pos_triples) + batch_size - 1) // batch_size
        # 新增：用于统计时间的变量
        total_start_time = time.time()  # 总生成开始时间

        for i in range(0, len(pos_triples), batch_size):
            batch_num = i // batch_size + 1
            # 记录当前批次开始时间
            batch_start_time = time.time()

            batch_pos = pos_triples[i:i + batch_size]
            try:
                # 直接使用批次三元组生成硬负样本，不通过DataLoader
                hard_negs = self.hard_neg_sampler.sample_hard_negatives_from_batch(
                    batch_pos,
                    num_candidates_per_pos=self.config.get('num_candidates_per_pos', 5)
                )
                # 将硬负样本按正样本分组
                for pos_triple in batch_pos:
                    h, t, r = pos_triple
                    # 找到属于这个正样本的硬负样本
                    relevant_hard_negs = [
                        (nh, nt, nr) for nh, nt, nr in hard_negs
                        if (nh == h and nr == r) or (nt == t and nr == r)
                    ]
                    if relevant_hard_negs:
                        hard_negatives_dict[(h, t, r)] = relevant_hard_negs

                # 新增：计算单批次耗时 + 剩余时间预估
                batch_end_time = time.time()
                batch_elapsed = batch_end_time - batch_start_time  # 单批次耗时（秒）
                processed_batches = batch_num  # 已处理批次
                remaining_batches = total_batches - processed_batches  # 剩余批次
                avg_batch_time = (batch_end_time - total_start_time) / processed_batches  # 平均批次耗时（秒）
                remaining_time = remaining_batches * avg_batch_time  # 剩余时间（秒）

                logger.info(
                    f"HardNeg Batch {batch_num:4d}/{total_batches:4d} | "
                    f"Batch Time: {batch_elapsed:.1f}s | "
                    f"Avg Batch Time: {avg_batch_time:.1f}s | "
                    f"Remaining Time: {_format_hms(remaining_time)} | "
                    f"Processed Pos Samples: {i + len(batch_pos):6d}/{len(pos_triples):6d}"
                )

            except Exception as e:
                # 异常时也打印进度，便于定位问题批次
                batch_end_time = time.time()
                batch_elapsed = batch_end_time - batch_start_time
                logger.warning(
                    f"Failed to generate hard negatives for Batch {batch_num:4d}/{total_batches:4d} | "
                    f"Batch Time: {batch_elapsed:.1f}s | Error: {str(e)}"
                )
                continue

        # 更新训练数据集的硬负样本，不重新初始化DataLoader
        self.train_dataset.set_hard_negatives(hard_negatives_dict)
        self._save_hard_neg_cache(epoch, hard_negatives_dict)
        # 新增：总生成耗时统计
        total_elapsed = time.time() - total_start_time
        logger.info(
            f"Generated hard negatives for epoch {epoch} | "
            f"Total Batches: {total_batches} | "
            f"Valid HardNeg Pairs: {len(hard_negatives_dict)} | "
            f"Total Time: {_format_hms(total_elapsed)}"
        )
        # 确保训练加载器保持原来的状态
        self.train_loader = current_train_loader

    def train_epoch(self, epoch, stop_deadline=None):
        """
        训练一个 epoch：subset 硬负 / online / AMP / 梯度累积。
        支持 mid-epoch 续训、按 batch/时间落盘、到 stop_at 优雅暂停。
        返回 (avg_loss, metrics, interrupted)。
        """
        hnm = self.config.get("hard_neg_mode", "off")
        start_batch = int(getattr(self, "_resume_batch", 0) or 0)
        self._resume_batch = 0
        self._epoch_in_progress = False

        if stop_deadline is not None and time.time() >= stop_deadline:
            logger.info("⏰ stop_at reached before epoch %d work. Pause.", epoch)
            self.save_checkpoint(
                epoch, checkpoint_type="last",
                epoch_in_progress=True, resume_batch=start_batch,
            )
            self._write_progress(
                epoch - 1,
                status="stopped_by_schedule_mid_epoch",
                extra={
                    "stop_at": self.config.get("stop_at"),
                    "epoch_in_progress": True,
                    "current_epoch": epoch,
                    "resume_batch": start_batch,
                    "resume_hint": "Next night continues same epoch from resume_batch",
                },
            )
            return 0.0, {"auc": 0.5, "p_loss": 0.0, "n_loss": 0.0}, True

        if hnm == "subset_epoch" and self._should_regenerate_subset_hard(epoch):
            if start_batch > 0 and self._load_hard_neg_cache(epoch):
                self.train_dataset.use_hard_negatives = True
                logger.info(
                    "Epoch %d: reuse hard-neg cache (resume from batch %d)",
                    epoch, start_batch,
                )
            else:
                self._generate_hard_negatives_for_epoch(epoch)
                self.train_dataset.use_hard_negatives = True
                logger.info(f"Epoch {epoch}: subset hard negatives refreshed")
        elif hnm == "subset_epoch" and getattr(self.train_dataset, "hard_negatives_cache", None):
            self.train_dataset.use_hard_negatives = len(self.train_dataset.hard_negatives_cache) > 0
        else:
            self.train_dataset.use_hard_negatives = False

        train_loader, start_batch, total_batches = self._make_resume_train_loader(epoch, start_batch)
        if train_loader is None:
            logger.info(
                "Epoch %d: no remaining batches (start_batch=%d/%d).",
                epoch, start_batch, total_batches,
            )
            return 0.0, {"auc": 0.5, "p_loss": 0.0, "n_loss": 0.0}, False

        if start_batch > 0:
            logger.info("Resuming epoch %d from batch %d/%d", epoch, start_batch, total_batches)

        if stop_deadline is not None and time.time() >= stop_deadline:
            logger.info("⏰ stop_at reached after hard-neg / before train loop (epoch %d). Pause.", epoch)
            self.save_checkpoint(
                epoch, checkpoint_type="last",
                epoch_in_progress=True, resume_batch=start_batch,
            )
            self._write_progress(
                epoch - 1,
                status="stopped_by_schedule_mid_epoch",
                extra={
                    "stop_at": self.config.get("stop_at"),
                    "epoch_in_progress": True,
                    "current_epoch": epoch,
                    "resume_batch": start_batch,
                    "total_batches": total_batches,
                    "resume_hint": "Next night continues same epoch from resume_batch",
                },
            )
            return 0.0, {"auc": 0.5, "p_loss": 0.0, "n_loss": 0.0}, True

        self.model.train()
        total_loss, total_p_loss, total_n_loss = 0, 0, 0
        all_scores, all_labels = [], []
        accum = max(1, int(self.config.get("gradient_accumulation_steps", 1)))
        self.optimizer.zero_grad(set_to_none=True)
        n_batches = 0

        ckpt_every_batches = max(0, int(self.config.get("ckpt_every_batches", 50) or 0))
        ckpt_every_minutes = float(self.config.get("ckpt_every_minutes", 20) or 0)
        ckpt_every_secs = max(0.0, ckpt_every_minutes * 60.0)
        last_ckpt_time = time.time()
        interrupted = False
        global_step = start_batch - 1
        local_step = -1

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:3d} [Train]",
            leave=False,
            initial=start_batch,
            total=total_batches,
        )
        for local_step, batch in enumerate(progress_bar):
            global_step = start_batch + local_step

            if stop_deadline is not None and time.time() >= stop_deadline:
                interrupted = True
                logger.info(
                    "⏰ Reached stop_at mid-epoch %d at batch %d/%d. Saving & pause.",
                    epoch, global_step, total_batches,
                )
                break

            pos_tri, neg_tri, pos_meta = _unwrap_collate_batch(batch)
            if pos_tri[0] is None:
                continue
            n_batches += 1

            pos_h, pos_t, pos_r, pos_bg = pos_tri
            pos_h = pos_h.to(self.device, non_blocking=True)
            pos_t = pos_t.to(self.device, non_blocking=True)
            pos_r = pos_r.to(self.device, non_blocking=True)
            pos_bg = {
                k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in pos_bg.items()
            }

            neg_h, neg_t, neg_r, neg_bg = neg_tri
            if neg_h is not None and neg_h.num_graphs > 0:
                neg_h = neg_h.to(self.device, non_blocking=True)
                neg_t = neg_t.to(self.device, non_blocking=True)
                neg_r = neg_r.to(self.device, non_blocking=True)
                neg_bg = {
                    k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in neg_bg.items()
                }
                merged_tri, n_pos = merge_ddi_batches(
                    (pos_h, pos_t, pos_r, pos_bg),
                    (neg_h, neg_t, neg_r, neg_bg),
                )
            else:
                merged_tri = (pos_h, pos_t, pos_r, pos_bg)
                n_pos = pos_r.size(0)

            with self._autocast():
                mh, mt, mr, mbg = merged_tri
                batch_scores = _sanitize_score_tensor(self.model(mh, mt, mr, mbg), stage=f"Train-e{epoch}")
                p_scores = batch_scores[:n_pos]
                n_scores = batch_scores[n_pos:] if batch_scores.size(0) > n_pos else torch.tensor([], device=self.device)

                if (
                    hnm == "online_batch"
                    and pos_meta
                    and self.hard_neg_sampler is not None
                    and epoch >= self.config.get("hard_neg_start_epoch", 0)
                ):
                    K = int(self.config.get("online_hard_candidates", 4))
                    dlist = self.hard_neg_sampler.drug_ids
                    candidates = []
                    for h, t, r in pos_meta:
                        for _ in range(K):
                            if random.random() < 0.5:
                                candidates.append((random.choice(dlist), t, r))
                            else:
                                candidates.append((h, random.choice(dlist), r))
                    if candidates:
                        pos_only = self.train_dataset.collate_positives_only(candidates)
                        if pos_only and pos_only[0] is not None:
                            ph, pt, pr, pbg = pos_only
                            ph, pt, pr = ph.to(self.device), pt.to(self.device), pr.to(self.device)
                            pbg = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                                   for k, v in pbg.items()}
                            c_scores = self.model(ph, pt, pr, pbg)
                            n_scores = torch.cat([n_scores, c_scores], dim=0) if n_scores.numel() else c_scores

                loss, p_loss, n_loss = self.loss_fn(p_scores, n_scores)
                loss = loss / accum

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (local_step + 1) % accum == 0:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["grad_clip_value"])
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.item() * accum)
            total_p_loss += p_loss.item()
            total_n_loss += n_loss.item()

            all_scores.append(p_scores.detach().cpu())
            all_labels.append(torch.ones_like(p_scores.detach().cpu()))
            if n_scores.numel() > 0:
                all_scores.append(n_scores.detach().cpu())
                all_labels.append(
                    torch.zeros(n_scores.shape[0], dtype=torch.float32, device="cpu")
                )

            hard_neg_status = (
                f"{hnm}" if hnm != "off" and self.train_dataset.use_hard_negatives
                else ("online" if hnm == "online_batch" else "rand")
            )
            progress_bar.set_postfix(
                Loss=f"{loss.item() * accum:.4f}",
                P_Loss=f"{p_loss.item():.4f}",
                N_Loss=f"{n_loss.item():.4f}",
                Neg=hard_neg_status,
            )

            next_batch = global_step + 1
            due_by_batch = ckpt_every_batches > 0 and next_batch % ckpt_every_batches == 0
            due_by_time = ckpt_every_secs > 0 and (time.time() - last_ckpt_time) >= ckpt_every_secs
            if due_by_batch or due_by_time:
                self.save_checkpoint(
                    epoch,
                    checkpoint_type="last",
                    epoch_in_progress=True,
                    resume_batch=next_batch,
                )
                last_ckpt_time = time.time()
                self._write_progress(
                    epoch - 1,
                    status="mid_epoch_checkpoint",
                    extra={
                        "epoch_in_progress": True,
                        "current_epoch": epoch,
                        "resume_batch": next_batch,
                        "total_batches": total_batches,
                        "current_round": self.current_round,
                        "epoch_in_round": self.epoch_in_round,
                    },
                )

        if n_batches > 0 and local_step >= 0 and (local_step + 1) % accum != 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["grad_clip_value"])
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        if interrupted:
            # break 发生在该 batch 训练之前，故从 global_step 续训
            resume_at = max(start_batch, global_step)
            self.save_checkpoint(
                epoch,
                checkpoint_type="last",
                epoch_in_progress=True,
                resume_batch=resume_at,
            )
            self._write_progress(
                epoch - 1,
                status="stopped_by_schedule_mid_epoch",
                extra={
                    "stop_at": self.config.get("stop_at"),
                    "epoch_in_progress": True,
                    "current_epoch": epoch,
                    "resume_batch": resume_at,
                    "total_batches": total_batches,
                    "current_round": self.current_round,
                    "epoch_in_round": self.epoch_in_round,
                    "resume_hint": "Next night continues same epoch from resume_batch",
                },
            )
            avg_loss = total_loss / max(1, n_batches)
            return avg_loss, {
                "auc": 0.5,
                "p_loss": total_p_loss / max(1, n_batches),
                "n_loss": total_n_loss / max(1, n_batches),
            }, True

        avg_loss = total_loss / max(1, n_batches)
        all_scores_t = torch.cat(all_scores) if all_scores else torch.tensor([], dtype=torch.float32)
        all_labels_t = torch.cat(all_labels) if all_labels else torch.tensor([], dtype=torch.float32)
        auc = _safe_binary_auc(all_labels_t.cpu().numpy(), all_scores_t.cpu().numpy())
        return avg_loss, {
            "auc": float(auc),
            "p_loss": total_p_loss / max(1, n_batches),
            "n_loss": total_n_loss / max(1, n_batches),
        }, False

    def _evaluate(self, data_loader, stage='Val'):
        """评估模型（用于验证和测试）"""
        self.model.eval()
        total_loss, total_p_loss, total_n_loss = 0, 0, 0
        all_scores, all_labels = [], []

        progress_bar = tqdm(data_loader, desc=f"Epoch N/A [{stage}]", leave=False)
        with torch.no_grad():
            for batch in progress_bar:
                pos_tri, neg_tri, _ = _unwrap_collate_batch(batch)

                if pos_tri[0] is None:
                    continue

                # 正样本 + 负样本合并为单次 forward
                pos_h, pos_t, pos_r, pos_bg = pos_tri
                pos_h, pos_t, pos_r = pos_h.to(self.device), pos_t.to(self.device), pos_r.to(self.device)
                pos_bg = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in pos_bg.items()}

                neg_h, neg_t, neg_r, neg_bg = neg_tri
                if neg_h is not None and neg_h.num_graphs > 0:
                    neg_h, neg_t, neg_r = neg_h.to(self.device), neg_t.to(self.device), neg_r.to(self.device)
                    neg_bg = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in neg_bg.items()}
                    merged_tri, n_pos = merge_ddi_batches(
                        (pos_h, pos_t, pos_r, pos_bg),
                        (neg_h, neg_t, neg_r, neg_bg),
                    )
                else:
                    merged_tri = (pos_h, pos_t, pos_r, pos_bg)
                    n_pos = pos_r.size(0)

                mh, mt, mr, mbg = merged_tri
                batch_scores = _sanitize_score_tensor(self.model(mh, mt, mr, mbg), stage=stage)
                p_scores = batch_scores[:n_pos]
                n_scores = batch_scores[n_pos:] if batch_scores.size(0) > n_pos else torch.tensor([], device=self.device)

                loss, p_loss, n_loss = self.loss_fn(p_scores, n_scores)

                total_loss += loss.item()
                total_p_loss += p_loss.item()
                total_n_loss += n_loss.item()

                all_scores.append(p_scores.cpu())
                all_labels.append(torch.ones_like(p_scores.cpu()))
                if n_scores.numel() > 0:
                    all_scores.append(n_scores.cpu())
                    all_labels.append(torch.zeros_like(n_scores.cpu()))

        n_loader = max(1, len(data_loader))
        avg_loss = total_loss / n_loader
        if not all_scores:
            metrics = {
                "auc": 0.5,
                "aupr": 0.0,
                "accuracy": 0.0,
                "f1_score": 0.0,
                "p_loss": total_p_loss / n_loader,
                "n_loss": total_n_loss / n_loader,
                "best_threshold": 0.5,
            }
            return avg_loss, metrics

        all_scores = torch.cat(all_scores)
        all_labels = torch.cat(all_labels)

        raw_nan = int(torch.isnan(all_scores).sum().item())
        raw_inf = int(torch.isinf(all_scores).sum().item())
        if raw_nan or raw_inf:
            logger.warning(
                "[%s] aggregated scores before metrics: nan=%d inf=%d / %d",
                stage, raw_nan, raw_inf, all_scores.numel(),
            )
        all_scores = _sanitize_score_tensor(all_scores, stage=f"{stage}-agg")

        all_probabilities = torch.sigmoid(all_scores)
        y_true_np = all_labels.cpu().numpy()
        prob_np = all_probabilities.cpu().numpy()
        min_recall = float(self.config.get("min_recall_for_threshold", 0.0))

        best_t, best_f1, best_mrec, best_acc = 0.5, -1.0, 0.0, -1.0
        for t in np.linspace(0.0, 1.0, 101):
            y_pred_b = (prob_np > t).astype(int)
            rec = float(recall_score(y_true_np, y_pred_b, zero_division=0))
            if rec < min_recall:
                continue
            f1v = float(f1_score(y_true_np, y_pred_b, zero_division=0))
            accv = float(accuracy_score(y_true_np, y_pred_b))
            if (
                f1v > best_f1
                or (abs(f1v - best_f1) <= 1e-9 and accv > best_acc)
                or (abs(f1v - best_f1) <= 1e-9 and abs(accv - best_acc) <= 1e-9 and abs(t - 0.5) < abs(best_t - 0.5))
            ):
                best_f1, best_t, best_mrec, best_acc = f1v, t, rec, accv

        logger.info(
            f"[{stage}] best threshold on prob (F1={best_f1:.4f}): {best_t:.4f} "
            f"(min_recall>={min_recall}, recall@thr={best_mrec:.4f}, acc@thr={best_acc:.4f})"
        )
        y_pred_best = (all_probabilities > best_t).long()

        y_true_np = all_labels.cpu().numpy()
        y_score_np = all_scores.cpu().numpy()
        metrics = {
            "auc": _safe_binary_auc(y_true_np, y_score_np),
            "aupr": _safe_aupr(y_true_np, y_score_np),
            "accuracy": float(accuracy_score(y_true_np, y_pred_best.cpu().numpy())),
            "f1_score": float(f1_score(y_true_np, y_pred_best.cpu().numpy(), zero_division=0)),
            "p_loss": total_p_loss / n_loader,
            "n_loss": total_n_loss / n_loader,
            "best_threshold": best_t,
        }

        return avg_loss, metrics

    def validate_epoch(self, epoch):
        return self._evaluate(self.val_loader, stage='Val')

    def test(self, threshold=0.5):
        """使用给定阈值评估测试集；优先 best_model，其次 last_epoch。不覆盖多轮收敛状态。"""
        logger.info(f"Starting final testing with threshold: {threshold:.4f}...")
        # 测试加载权重时保留当前多轮/收敛状态，避免被旧 best checkpoint 冲掉
        _round_state = {
            "current_round": self.current_round,
            "rounds_without_improve": self.rounds_without_improve,
            "best_round_score": self.best_round_score,
            "training_converged": self.training_converged,
            "epoch_in_round": self.epoch_in_round,
            "global_epoch": self.global_epoch,
            "best_val_f1": self.best_val_f1,
            "best_val_auc": self.best_val_auc,
            "best_val_aupr": self.best_val_aupr,
            "best_threshold_for_test": self.best_threshold_for_test,
            "patience_counter": self.patience_counter,
        }
        best_model_path = os.path.join(self.config['save_dir'], 'best_model.pth')
        last_epoch_path = os.path.join(self.config['save_dir'], 'last_epoch.pth')
        ckpt_used = None
        if os.path.exists(best_model_path):
            try:
                self.load_checkpoint(best_model_path)
                ckpt_used = best_model_path
            except RuntimeError as e:
                logger.warning("Skip loading best_model for test due to signature mismatch: %s", e)
        elif os.path.exists(last_epoch_path):
            try:
                self.load_checkpoint(last_epoch_path)
                ckpt_used = last_epoch_path
                logger.info("No best_model.pth yet; testing with last_epoch.pth")
            except RuntimeError as e:
                logger.warning("Skip loading last_epoch for test due to signature mismatch: %s", e)
        else:
            logger.warning("No checkpoint found for testing; using in-memory model weights.")

        for k, v in _round_state.items():
            setattr(self, k, v)

        self.model.eval()
        all_scores, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="[Test]"):
                pos_tri, neg_tri, _ = _unwrap_collate_batch(batch)
                if pos_tri[0] is None:
                    continue

                pos_h, pos_t, pos_r, pos_bg = pos_tri
                pos_h, pos_t, pos_r = pos_h.to(self.device), pos_t.to(self.device), pos_r.to(self.device)
                pos_bg = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in pos_bg.items()}

                neg_h, neg_t, neg_r, neg_bg = neg_tri
                if neg_h is not None and neg_h.num_graphs > 0:
                    neg_h, neg_t, neg_r = neg_h.to(self.device), neg_t.to(self.device), neg_r.to(self.device)
                    neg_bg = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in neg_bg.items()}
                    merged_tri, n_pos = merge_ddi_batches(
                        (pos_h, pos_t, pos_r, pos_bg),
                        (neg_h, neg_t, neg_r, neg_bg),
                    )
                else:
                    merged_tri = (pos_h, pos_t, pos_r, pos_bg)
                    n_pos = pos_r.size(0)

                mh, mt, mr, mbg = merged_tri
                batch_scores = torch.sigmoid(
                    _sanitize_score_tensor(self.model(mh, mt, mr, mbg), stage="Test")
                )
                p_scores = batch_scores[:n_pos]
                n_scores = batch_scores[n_pos:] if batch_scores.size(0) > n_pos else torch.tensor([], device=self.device)

                all_scores.append(p_scores.cpu())
                all_labels.append(torch.ones_like(p_scores.cpu()))
                if n_scores.numel() > 0:
                    all_scores.append(n_scores.cpu())
                    all_labels.append(torch.zeros_like(n_scores.cpu()))

        if not all_scores:
            logger.warning("No valid test batches; cannot compute metrics.")
            return None

        all_scores = torch.cat(all_scores)
        all_labels = torch.cat(all_labels)
        y_pred = (all_scores > threshold).long()
        y_true_np = all_labels.cpu().numpy()
        y_score_np = all_scores.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()

        return {
            'auc': _safe_binary_auc(y_true_np, y_score_np),
            'aupr': _safe_aupr(y_true_np, y_score_np),
            'accuracy': float(accuracy_score(y_true_np, y_pred_np)),
            'f1_score': float(f1_score(y_true_np, y_pred_np, zero_division=0)),
            'threshold': float(threshold),
            'checkpoint_used': ckpt_used or 'in_memory',
        }

    def _save_session_test_results(self, test_results: dict, session_status: str, completed_epoch: int) -> str:
        """将会话测试结果写入最新 JSON + 追加 JSONL 历史。"""
        import json
        import datetime as _dt

        out_dir = self.config.get("test_results_dir") or self.config.get("log_dir") or "."
        os.makedirs(out_dir, exist_ok=True)
        stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record = {
            "recorded_at": stamp,
            "session_status": session_status,
            "completed_epoch": int(completed_epoch),
            "train_config_signature": self.config_signature,
            "best_val_auc": float(self.best_val_auc),
            "best_val_aupr": float(self.best_val_aupr),
            "best_val_f1": float(self.best_val_f1),
            "best_threshold_for_test": float(self.best_threshold_for_test),
            "test_metrics": test_results or {},
        }

        latest_path = os.path.join(out_dir, "latest_test_results.json")
        history_path = os.path.join(out_dir, "nightly_test_history.jsonl")
        try:
            with open(latest_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            with open(history_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            logger.info("Test results saved: %s | history append: %s", latest_path, history_path)
            return latest_path
        except Exception as e:
            logger.warning("Failed to save test results: %s", e)
            return ""

    def _run_and_record_test(self, session_status: str, completed_epoch: int) -> dict:
        """跑测试并落盘（夜间到点停 / 正常训完均可）。"""
        if self.config.get("skip_final_test"):
            logger.info("skip_final_test=True; skip session test.")
            return None

        logger.info(
            "Running session test (status=%s, epoch=%d, threshold=%.4f)...",
            session_status, completed_epoch, self.best_threshold_for_test,
        )
        try:
            test_results = self.test(threshold=self.best_threshold_for_test)
        except Exception:
            logger.exception("Session test failed; training progress is still saved.")
            return None

        if test_results:
            logger.info(
                "Session Test Results - AUC: %.4f | AUPR: %.4f | Acc: %.4f | F1: %.4f | thr: %.4f",
                test_results.get("auc", 0.0),
                test_results.get("aupr", 0.0),
                test_results.get("accuracy", 0.0),
                test_results.get("f1_score", 0.0),
                test_results.get("threshold", self.best_threshold_for_test),
            )
            if self.config.get("record_test_results", True):
                path = self._save_session_test_results(test_results, session_status, completed_epoch)
                self._write_progress(
                    completed_epoch,
                    status=f"{session_status}_tested",
                    extra={"latest_test_results": path, "test_metrics": test_results},
                )
        return test_results

    def _build_checkpoint(
        self,
        epoch: int,
        epoch_in_progress: bool = False,
        resume_batch: int = 0,
    ) -> dict:
        """组装 checkpoint 字典（续训 + 早停 + 多轮状态 + mid-epoch）。"""
        ckpt = {
            'epoch': epoch,
            'epoch_in_progress': bool(epoch_in_progress),
            'resume_batch': int(resume_batch) if epoch_in_progress else 0,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_f1': self.best_val_f1,
            'best_val_auc': self.best_val_auc,
            'best_val_aupr': self.best_val_aupr,
            'best_threshold_for_test': self.best_threshold_for_test,
            'patience_counter': self.patience_counter,
            'current_round': self.current_round,
            'rounds_without_improve': self.rounds_without_improve,
            'best_round_score': self.best_round_score,
            'training_converged': self.training_converged,
            'epoch_in_round': self.epoch_in_round,
            'global_epoch': self.global_epoch,
            'train_config': copy.deepcopy(self.config),
            'train_config_signature': self.config_signature,
        }
        if self.scaler is not None:
            ckpt['scaler_state_dict'] = self.scaler.state_dict()
        return ckpt

    def save_checkpoint(
        self,
        epoch,
        checkpoint_type='last',
        epoch_in_progress: bool = False,
        resume_batch: int = 0,
    ):
        """
        保存检查点。
        - last: 写入 last_epoch.pth（完整 epoch 结束，或 mid-epoch 可续训）
        - best: 仅当验证指标提升时写入 best_model.pth
        mid-epoch: epoch_in_progress=True，resume_batch=下次从哪一 batch 继续。
        """
        filename = 'best_model.pth' if checkpoint_type == 'best' else 'last_epoch.pth'
        path = os.path.join(self.config['save_dir'], filename)
        ckpt = self._build_checkpoint(
            epoch,
            epoch_in_progress=epoch_in_progress and checkpoint_type == 'last',
            resume_batch=resume_batch if checkpoint_type == 'last' else 0,
        )
        _atomic_torch_save(ckpt, path)
        if checkpoint_type == 'best':
            logger.info("Saved best checkpoint to %s (epoch %d)", path, epoch)
        elif epoch_in_progress:
            logger.info(
                "Saved mid-epoch checkpoint to %s (epoch %d, resume_batch=%d)",
                path, epoch, resume_batch,
            )
        else:
            logger.info(
                "Saved last-epoch checkpoint to %s (epoch %d, resume -> epoch %d)",
                path, epoch, epoch + 1,
            )

    def _checkpoint_signature(self, checkpoint: dict) -> str:
        sig = checkpoint.get("train_config_signature")
        if sig:
            return str(sig)
        train_cfg = checkpoint.get("train_config")
        if isinstance(train_cfg, dict):
            return get_training_config_signature(train_cfg)
        return ""

    def load_checkpoint(self, path):
        """加载模型检查点；返回下一个待训练 epoch（mid-epoch 则返回同一 epoch）。"""
        if not os.path.exists(path):
            logger.warning(f"Checkpoint {path} not found, starting from scratch")
            return 0
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        ckpt_sig = self._checkpoint_signature(checkpoint)
        if ckpt_sig != self.config_signature:
            if self.config.get("allow_resume_config_mismatch", False):
                logger.warning(
                    "Resuming with config mismatch because allow_resume_config_mismatch=True: "
                    "checkpoint=%s current=%s",
                    ckpt_sig or "(missing)",
                    self.config_signature,
                )
            else:
                raise RuntimeError(
                    "Checkpoint config signature mismatch: "
                    f"checkpoint={ckpt_sig or '(missing)'} current={self.config_signature}. "
                    "Use --fresh to restart, or pass --allow-resume-config-mismatch only if "
                    "you intentionally accept mixed training semantics."
                )
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as e:
            err = str(e)
            if "state_dict" in err or "Missing key" in err or "Unexpected key" in err:
                logger.error(
                    "Checkpoint 与当前模型或 torch_geometric 版本不兼容，无法续训（常见原因：旧版 GATConv 使用 "
                    "lin_src/lin_dst，新版使用 lin）。将从头训练。可选：使用 --fresh 忽略 checkpoint；"
                    "或删除 checkpoints/last_epoch.pth 与 best_model.pth；或安装与保存权重时一致的 PyG 版本。"
                )
                logger.error(err)
                return 0
            raise
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint.get('scheduler_state_dict') is not None:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                logger.warning("Failed to restore scheduler state: %s", e)
        if self.scaler is not None and checkpoint.get('scaler_state_dict') is not None:
            try:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            except Exception as e:
                logger.warning("Failed to restore AMP scaler state: %s", e)

        self.best_val_f1 = checkpoint.get('best_val_f1', 0.0)
        self.best_val_auc = checkpoint.get('best_val_auc', 0.0)
        self.best_val_aupr = checkpoint.get('best_val_aupr', 0.0)
        self.best_threshold_for_test = checkpoint.get('best_threshold_for_test', 0.5)
        self.patience_counter = int(checkpoint.get('patience_counter', 0))
        self.current_round = int(checkpoint.get('current_round', 0))
        self.rounds_without_improve = int(checkpoint.get('rounds_without_improve', 0))
        self.best_round_score = float(checkpoint.get('best_round_score', self.best_val_auc))
        self.training_converged = bool(checkpoint.get('training_converged', False))
        self.epoch_in_round = int(checkpoint.get('epoch_in_round', 0))
        self.global_epoch = int(checkpoint.get('global_epoch', checkpoint.get('epoch', 0)))

        epoch_in_progress = bool(checkpoint.get('epoch_in_progress', False))
        resume_batch = int(checkpoint.get('resume_batch', 0) or 0)
        self._epoch_in_progress = epoch_in_progress
        self._resume_batch = resume_batch if epoch_in_progress else 0

        if epoch_in_progress:
            logger.info(
                "Loaded mid-epoch checkpoint from %s (epoch %d, resume_batch=%d)",
                path, checkpoint['epoch'], resume_batch,
            )
            logger.info(
                "Resumed mid-epoch | epoch=%d batch=%d | round=%d | best Val F1=%.4f AUC=%.4f AUPR=%.4f | converged=%s",
                checkpoint['epoch'],
                resume_batch,
                self.current_round,
                self.best_val_f1,
                self.best_val_auc,
                self.best_val_aupr,
                self.training_converged,
            )
            return int(checkpoint['epoch'])

        logger.info("Loaded checkpoint from %s (completed epoch %d)", path, checkpoint['epoch'])
        logger.info(
            "Resumed | next epoch=%d | round=%d | rounds_wo_improve=%d | best Val F1=%.4f AUC=%.4f AUPR=%.4f | patience=%d | converged=%s",
            checkpoint['epoch'] + 1,
            self.current_round,
            self.rounds_without_improve,
            self.best_val_f1,
            self.best_val_auc,
            self.best_val_aupr,
            self.patience_counter,
            self.training_converged,
        )
        return checkpoint['epoch'] + 1

    def _write_progress(self, epoch: int, status: str, extra: dict = None) -> None:
        """写入夜间训练进度 JSON，便于次日续训与排查。"""
        import json
        import datetime as _dt

        progress_path = self.config.get("progress_file") or os.path.join(
            self.config.get("log_dir", "."), "nightly_progress.json"
        )
        payload = {
            "status": status,
            "updated_at": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "train_config_signature": self.config_signature,
            "completed_epoch": int(epoch),
            "next_epoch": int(epoch) + 1,
            "best_val_auc": float(self.best_val_auc),
            "best_val_aupr": float(self.best_val_aupr),
            "best_val_f1": float(self.best_val_f1),
            "best_threshold_for_test": float(self.best_threshold_for_test),
            "patience_counter": int(self.patience_counter),
            "checkpoint_dir": self.config.get("save_dir", ""),
            "last_epoch_path": os.path.join(self.config.get("save_dir", ""), "last_epoch.pth"),
            "best_model_path": os.path.join(self.config.get("save_dir", ""), "best_model.pth"),
        }
        if extra:
            payload.update(extra)
        try:
            os.makedirs(os.path.dirname(os.path.abspath(progress_path)) or ".", exist_ok=True)
            with open(progress_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            logger.info("Progress written to %s (%s)", progress_path, status)
        except Exception as e:
            logger.warning("Failed to write progress file %s: %s", progress_path, e)

    def _score_for_round(self) -> float:
        """轮次提升判定主分数：优先 Val AUC，其次 F1/AUPR。"""
        return float(self.best_val_auc)

    def _run_one_epoch(self, epoch: int, stop_deadline=None):
        """
        执行单个 epoch：训练/验证/存盘。
        返回 (train_metrics, val_metrics, val_ok, improved, interrupted)。
        """
        epoch_start_time = time.time()
        train_loss, train_metrics, interrupted = self.train_epoch(epoch, stop_deadline=stop_deadline)
        if interrupted:
            logger.info(
                "Epoch %d interrupted by schedule; skip validation (mid-epoch checkpoint already saved).",
                epoch,
            )
            return train_metrics, None, False, False, True

        _warn_if_model_nonfinite(self.model, stage=f"after-train-e{epoch}")

        val_ok = True
        try:
            val_loss, val_metrics = self.validate_epoch(epoch)
        except Exception:
            val_ok = False
            logger.exception(
                "Validation failed at epoch %d; will save last_epoch.pth and continue",
                epoch,
            )
            val_loss = float('nan')
            val_metrics = {
                'auc': 0.5,
                'aupr': 0.0,
                'f1_score': 0.0,
                'accuracy': 0.0,
                'best_threshold': self.best_threshold_for_test,
                'p_loss': 0.0,
                'n_loss': 0.0,
            }

        self.scheduler.step()
        epoch_time = time.time() - epoch_start_time
        current_lr = self.scheduler.get_last_lr()[0]
        hard_neg_status = "HardNeg" if self.train_dataset.use_hard_negatives else "RandomNeg"

        logger.info(
            f"Epoch {epoch:3d} | Round {self.current_round} ({self.epoch_in_round + 1}/"
            f"{self.config.get('epochs_per_round', 5)}) | "
            f"Train Loss: {train_loss:.4f} | Train AUC: {train_metrics['auc']:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val AUC: {val_metrics['auc']:.4f} | Val F1: {val_metrics['f1_score']:.4f} | "
            f"Val AUPR: {val_metrics['aupr']:.4f} | LR: {current_lr:.6f} | "
            f"NegType: {hard_neg_status} | Time: {epoch_time:.2f}s"
        )

        f1_eps, auc_eps, aupr_eps = 1e-4, 1e-4, 1e-4
        improved = False
        if val_ok and (
                val_metrics['auc'] > self.best_val_auc + auc_eps or
                val_metrics['aupr'] > self.best_val_aupr + aupr_eps or
                (
                    abs(val_metrics['auc'] - self.best_val_auc) <= auc_eps and
                    abs(val_metrics['aupr'] - self.best_val_aupr) <= aupr_eps and
                    val_metrics['f1_score'] > self.best_val_f1 + f1_eps
                )
        ):
            improved = True
            self.best_val_f1 = val_metrics['f1_score']
            self.best_val_auc = val_metrics['auc']
            self.best_val_aupr = val_metrics['aupr']
            self.best_threshold_for_test = val_metrics['best_threshold']
            self.patience_counter = 0
            self.save_checkpoint(epoch, checkpoint_type='best')
            logger.info(
                f"💾 Val metrics improved! Saved best model at epoch {epoch} | "
                f"F1: {self.best_val_f1:.4f} | AUC: {self.best_val_auc:.4f} | AUPR: {self.best_val_aupr:.4f}"
            )
        elif val_ok:
            self.patience_counter += 1

        self.global_epoch = epoch
        self.save_checkpoint(epoch, checkpoint_type='last', epoch_in_progress=False, resume_batch=0)
        self._write_progress(
            epoch,
            status="epoch_done",
            extra={
                "epoch_in_progress": False,
                "resume_batch": 0,
                "current_round": self.current_round,
                "epoch_in_round": self.epoch_in_round,
                "rounds_without_improve": self.rounds_without_improve,
                "best_round_score": self.best_round_score,
                "train_loss": float(train_loss),
                "train_auc": float(train_metrics.get("auc", 0.0)),
                "val_loss": float(val_loss) if val_loss == val_loss else None,
                "val_auc": float(val_metrics.get("auc", 0.0)),
                "val_f1": float(val_metrics.get("f1_score", 0.0)),
                "val_aupr": float(val_metrics.get("aupr", 0.0)),
            },
        )
        return train_metrics, val_metrics, val_ok, improved, False

    def train(self):
        """
        多轮训练主循环：
          训完一整轮(epochs_per_round) → 自动测试并记录 → 继续下一轮，
          直到连续 round_patience 轮无明显提升，或达到 max_rounds。
        夜间可用 stop_at 到点暂停；次日续训同一状态。
        """
        logger.info("Starting multi-round training (train → test → continue until plateau)...")
        logger.info(f"Configuration: {self.config}")

        start_epoch = 0
        if self.config.get('resume_from_checkpoint'):
            start_epoch = self.load_checkpoint(self.config['resume_from_checkpoint'])

        if self.training_converged:
            logger.info("Training already converged previously. Skip further training.")
            self._write_progress(start_epoch - 1, status="converged")
            return

        stop_at = str(self.config.get("stop_at") or "").strip()
        stop_deadline = None
        if stop_at:
            stop_deadline = _next_stop_deadline(stop_at)
            import datetime as _dt
            logger.info(
                "Nightly stop enabled: stop_at=%s -> deadline=%s",
                stop_at,
                _dt.datetime.fromtimestamp(stop_deadline).strftime("%Y-%m-%d %H:%M:%S"),
            )

        epochs_per_round = max(1, int(self.config.get("epochs_per_round", 5)))
        max_rounds = max(1, int(self.config.get("max_rounds", 40)))
        round_patience = max(1, int(self.config.get("round_patience", 2)))
        min_improve = float(self.config.get("min_round_improve", 1e-4))
        # 总 epoch 上限：与旧 epochs 配置兼容
        max_epochs = int(self.config.get("epochs", epochs_per_round * max_rounds))
        max_epochs = max(max_epochs, epochs_per_round * max_rounds)

        epoch = start_epoch
        last_completed_epoch = start_epoch - 1
        stopped_by_schedule = False

        logger.info(
            "Plan: 1 epoch = 1 round (auto test each epoch) | stop if no improve for %d epochs | max_rounds=%d | start_epoch=%d",
            round_patience, max_rounds, start_epoch,
        )

        while self.current_round < max_rounds and epoch < max_epochs:
            logger.info(
                "===== Start Round %d/%d | epoch_in_round=%d/%d | global_epoch=%d =====",
                self.current_round + 1, max_rounds, self.epoch_in_round, epochs_per_round, epoch,
            )
            round_had_epoch = False
            mid_epoch_paused = False

            while self.epoch_in_round < epochs_per_round and epoch < max_epochs:
                if stop_deadline is not None and time.time() >= stop_deadline:
                    stopped_by_schedule = True
                    logger.info(
                        "⏰ Reached stop_at=%s mid-round %d (epoch %d). Pause for tonight.",
                        stop_at, self.current_round, epoch,
                    )
                    break

                _, _, _, _, interrupted = self._run_one_epoch(epoch, stop_deadline=stop_deadline)
                if interrupted:
                    stopped_by_schedule = True
                    mid_epoch_paused = True
                    logger.info(
                        "⏰ Mid-epoch pause at epoch %d. Progress saved; resume tomorrow from same epoch.",
                        epoch,
                    )
                    break

                last_completed_epoch = epoch
                round_had_epoch = True
                self.epoch_in_round += 1
                epoch += 1

                # 轮内早停：验证长时间无提升，也可提前结束本轮去做测试判定
                if self.patience_counter >= self.config['patience']:
                    logger.info(
                        "Within-round early stop signal at epoch %d (patience=%d). Finish this round and evaluate.",
                        last_completed_epoch, self.config['patience'],
                    )
                    break

                if stop_deadline is not None and time.time() >= stop_deadline:
                    stopped_by_schedule = True
                    logger.info(
                        "⏰ Reached stop_at=%s after epoch %d. Pause for tonight.",
                        stop_at, last_completed_epoch,
                    )
                    break

            if stopped_by_schedule:
                if mid_epoch_paused:
                    logger.info(
                        "Nightly pause mid-epoch. Progress/checkpoint saved; resume tomorrow."
                    )
                    return

                self._write_progress(
                    last_completed_epoch,
                    status="stopped_by_schedule",
                    extra={
                        "stop_at": stop_at,
                        "current_round": self.current_round,
                        "epoch_in_round": self.epoch_in_round,
                        "rounds_without_improve": self.rounds_without_improve,
                        "resume_hint": "Next night will continue the same round",
                    },
                )
                logger.info(
                    "Nightly pause before round complete. Progress saved; resume tomorrow (no round-end test yet)."
                )
                return

            if not round_had_epoch:
                break

            # ===== 每个 epoch（一整轮）结束：自动测试并记录 =====
            logger.info(
                "===== Epoch/Round %d finished. Auto test & record. =====",
                last_completed_epoch,
            )
            test_results = None
            test_every_n_epochs = max(1, int(self.config.get("test_every_n_epochs", 1)))
            should_test_this_round = (
                self.config.get("test_after_each_round", True)
                and ((last_completed_epoch + 1) % test_every_n_epochs == 0)
            )
            if should_test_this_round:
                test_results = self._run_and_record_test(
                    f"round_{self.current_round}_done",
                    last_completed_epoch,
                )
            elif self.config.get("test_after_each_round", True):
                logger.info(
                    "Skip auto test at epoch %d because test_every_n_epochs=%d",
                    last_completed_epoch,
                    test_every_n_epochs,
                )

            # 用本轮后的 best_val 判定是否“明显提升”
            round_score = self._score_for_round()
            improved_round = round_score > self.best_round_score + min_improve
            if improved_round:
                logger.info(
                    "Round %d improved: score %.6f -> %.6f. Continue training.",
                    self.current_round + 1, self.best_round_score, round_score,
                )
                self.best_round_score = round_score
                self.rounds_without_improve = 0
            else:
                self.rounds_without_improve += 1
                logger.info(
                    "Round %d no significant improve (score=%.6f, best=%.6f, streak=%d/%d).",
                    self.current_round + 1, round_score, self.best_round_score,
                    self.rounds_without_improve, round_patience,
                )

            self._write_progress(
                last_completed_epoch,
                status="round_done",
                extra={
                    "current_round": self.current_round,
                    "rounds_without_improve": self.rounds_without_improve,
                    "best_round_score": self.best_round_score,
                    "round_score": round_score,
                    "round_improved": improved_round,
                    "test_metrics": test_results or {},
                },
            )
            # 轮末再存一次，固化 round 状态
            self.save_checkpoint(last_completed_epoch, checkpoint_type='last')

            if self.rounds_without_improve >= round_patience:
                self.training_converged = True
                self.save_checkpoint(last_completed_epoch, checkpoint_type='last')
                self._write_progress(
                    last_completed_epoch,
                    status="converged",
                    extra={
                        "reason": f"no significant improve for {round_patience} consecutive rounds",
                        "best_round_score": self.best_round_score,
                        "current_round": self.current_round,
                    },
                )
                logger.info(
                    "🏁 Converged: no significant improve for %d consecutive epochs. Stop.",
                    round_patience,
                )
                # 收敛后再做一次最终测试记录
                self._run_and_record_test("converged", last_completed_epoch)
                return

            # 进入下一轮
            self.current_round += 1
            self.epoch_in_round = 0
            # 新轮重置轮内 patience，避免上一轮尾部直接触发
            self.patience_counter = 0
            self.save_checkpoint(last_completed_epoch, checkpoint_type='last')

            if stop_deadline is not None and time.time() >= stop_deadline:
                self._write_progress(
                    last_completed_epoch,
                    status="stopped_by_schedule",
                    extra={"stop_at": stop_at, "current_round": self.current_round},
                )
                logger.info("⏰ Stop_at reached after round boundary. Resume next night from new round.")
                return

        # 达到 max_rounds / max_epochs
        self.training_converged = True
        self._write_progress(last_completed_epoch, status="completed")
        self._run_and_record_test("completed", last_completed_epoch)
        logger.info("Training finished (reached max rounds/epochs).")


# ================== 主函数 ==================

def main():
    """主函数入口"""
    print("[DGN-DDI] 解析命令行参数...", flush=True)
    parser = argparse.ArgumentParser(description="DGN-DDI 转导式训练")
    parser.add_argument(
        "--profile",
        choices=["smoke", "full"],
        default="full",
        help="smoke: 子集+20epoch 快验；full: 全数据+100epoch（可用 --fresh 不接 checkpoint）",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="忽略 checkpoints/last_epoch.pth 与 best_model.pth，从头训练",
    )
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="显式开启自动续训（优先 last_epoch.pth，其次 best_model.pth）",
    )
    parser.add_argument(
        "--nightly",
        action="store_true",
        help="夜间分段：自动续训+到点暂停；每整轮结束后自动测试记录，直到性能不再明显提升",
    )
    parser.add_argument(
        "--stop-at",
        type=str,
        default="",
        help="到点暂停时间 HH:MM（如 08:00）；常与 --nightly 合用",
    )
    parser.add_argument(
        "--skip-final-test",
        action="store_true",
        help="关闭轮末/收敛时的自动测试记录",
    )
    parser.add_argument(
        "--epochs-per-round",
        type=int,
        default=0,
        help="一整轮的 epoch 数（默认配置 5；0=用配置默认）",
    )
    parser.add_argument(
        "--round-patience",
        type=int,
        default=0,
        help="连续多少轮无明显提升则停止（默认配置 2；0=用配置默认）",
    )
    parser.add_argument("--loss-fn", choices=["sigmoid", "focal", "hinge", "adaptive"], default="", help="覆盖训练损失")
    parser.add_argument("--focal-alpha", type=float, default=None, help="覆盖 focal alpha")
    parser.add_argument("--focal-gamma", type=float, default=None, help="覆盖 focal gamma")
    parser.add_argument("--hinge-margin", type=float, default=None, help="覆盖 hinge margin")
    parser.add_argument("--sigmoid-label-smoothing", type=float, default=None, help="覆盖 sigmoid 标签平滑")
    parser.add_argument("--hard-neg-ratio", type=float, default=None, help="覆盖硬负样本比例")
    parser.add_argument("--num-candidates-per-pos", type=int, default=0, help="覆盖每个正样本的硬负候选数")
    parser.add_argument("--hard-neg-selection-mode", choices=["per_positive", "global"], default="", help="硬负选择模式")
    parser.add_argument("--hard-neg-pool-strategy", choices=["random_subset", "front_slice"], default="", help="硬负池采样方式")
    parser.add_argument("--hard-neg-max-train-samples", type=int, default=0, help="覆盖硬负池最大样本数")
    parser.add_argument("--test-every-n-epochs", type=int, default=0, help="每隔多少个 epoch 自动测试一次")
    parser.add_argument("--allow-resume-config-mismatch", action="store_true", help="允许在训练配置签名不一致时强行续训")
    args = parser.parse_args()

    config = get_default_config()
    apply_training_profile(config, args.profile)

    # 夜间分段：20:30 启动、08:00 暂停；每 epoch 测试；连续 3 epoch 无提升则收敛停训
    if args.nightly:
        args.resume_latest = True
        if not args.stop_at:
            args.stop_at = "08:00"
        config["progress_file"] = os.path.join(config["log_dir"], "nightly_progress.json")
        config["test_results_dir"] = config["log_dir"]
        config["record_test_results"] = True
        config["test_after_each_round"] = True
        config["epochs_per_round"] = 1
        config["round_patience"] = max(int(config.get("round_patience", 8)), 8)
        config["patience"] = max(int(config.get("patience", 6)), 6)
        config["test_every_n_epochs"] = max(int(config.get("test_every_n_epochs", 5)), 5)
        if not args.skip_final_test:
            config["skip_final_test"] = False
        logger.info(
            "Nightly mode ON: resume + stop_at=%s | test_every_n_epochs=%s | round_patience=%s",
            args.stop_at,
            config["test_every_n_epochs"],
            config["round_patience"],
        )

    if args.epochs_per_round and args.epochs_per_round > 0:
        config["epochs_per_round"] = int(args.epochs_per_round)
    if args.round_patience and args.round_patience > 0:
        config["round_patience"] = int(args.round_patience)
    if args.loss_fn:
        config["loss_fn"] = args.loss_fn
    if args.focal_alpha is not None:
        config["focal_alpha"] = float(args.focal_alpha)
    if args.focal_gamma is not None:
        config["focal_gamma"] = float(args.focal_gamma)
    if args.hinge_margin is not None:
        config["hinge_margin"] = float(args.hinge_margin)
    if args.sigmoid_label_smoothing is not None:
        config["sigmoid_label_smoothing"] = float(args.sigmoid_label_smoothing)
    if args.hard_neg_ratio is not None:
        config["hard_neg_ratio"] = float(args.hard_neg_ratio)
    if args.num_candidates_per_pos and args.num_candidates_per_pos > 0:
        config["num_candidates_per_pos"] = int(args.num_candidates_per_pos)
    if args.hard_neg_selection_mode:
        config["hard_neg_selection_mode"] = args.hard_neg_selection_mode
    if args.hard_neg_pool_strategy:
        config["hard_neg_pool_strategy"] = args.hard_neg_pool_strategy
    if args.hard_neg_max_train_samples and args.hard_neg_max_train_samples > 0:
        config["hard_neg_max_train_samples"] = int(args.hard_neg_max_train_samples)
    if args.test_every_n_epochs and args.test_every_n_epochs > 0:
        config["test_every_n_epochs"] = int(args.test_every_n_epochs)
    if args.allow_resume_config_mismatch:
        config["allow_resume_config_mismatch"] = True

    if args.stop_at:
        _parse_hhmm(args.stop_at)
        config["stop_at"] = args.stop_at
    if args.skip_final_test:
        config["skip_final_test"] = True
        config["record_test_results"] = False
        config["test_after_each_round"] = False

    logger.info(
        "profile=%s | subset_size=%s | epochs=%s | patience=%s | compile_model=%s | stop_at=%s",
        args.profile,
        config.get("subset_size"),
        config["epochs"],
        config["patience"],
        config.get("compile_model", False),
        config.get("stop_at") or "(none)",
    )

    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)

    # 设置随机种子
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['random_seed'])
        torch.cuda.manual_seed_all(config['random_seed'])

    # 仅在显式要求时自动续训，避免历史低质量 checkpoint 污染新实验。
    save_dir = config['save_dir']
    last_epoch_path = os.path.join(save_dir, 'last_epoch.pth')
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    if args.fresh:
        config['resume_from_checkpoint'] = ''
        logger.info("--fresh: 不从 checkpoint 续训")
    elif args.resume_latest and os.path.exists(last_epoch_path):
        config['resume_from_checkpoint'] = last_epoch_path
        logger.info("Resuming training from last_epoch.pth: %s", last_epoch_path)
    elif args.resume_latest and os.path.exists(best_model_path):
        config['resume_from_checkpoint'] = best_model_path
        logger.info("Resuming training from best_model.pth (no last_epoch found): %s", best_model_path)
    else:
        config['resume_from_checkpoint'] = ''
        if args.resume_latest:
            logger.info("No checkpoint found. Starting training from scratch.")
        else:
            logger.info("Auto-resume disabled. Starting training from scratch.")
    if os.path.exists(best_model_path):
        logger.info("Final test will use best_model.pth when training completes")

    hnm = config.get("hard_neg_mode", "off")
    logger.info("hard_neg_mode=%s", hnm)
    if hnm == "subset_epoch" and config.get("use_hard_negative_sampling"):
        start_e = config.get("hard_neg_start_epoch", 3)
        freq = config.get("hard_neg_frequency", 3)
        logger.info(
            "subset_epoch: epoch 0-%d 随机负样本; 从 epoch %d 起每 %d 轮刷新硬负 (epochs %s, ...)",
            start_e - 1,
            start_e,
            freq,
            ", ".join(str(e) for e in range(start_e, start_e + freq * 4, freq)),
        )
        logger.info(
            "subset_epoch: max_train=%s candidates_per_pos=%s hard_ratio=%s",
            config.get("hard_neg_max_train_samples", 10000),
            config.get("num_candidates_per_pos", 5),
            config.get("hard_neg_ratio", 0.7),
        )
    elif hnm == "online_batch":
        logger.info("online_batch: candidates per pos=%s", config.get("online_hard_candidates", 4))
    elif hnm == "off" or not config.get("use_hard_negative_sampling", False):
        logger.info("Using random negative sampling only")

    # 创建训练器并开始训练
    print("[DGN-DDI] 初始化训练器（加载数据 / 模型，请稍候）...", flush=True)
    trainer = DDITrainer(config)
    print("[DGN-DDI] 开始训练循环。", flush=True)
    trainer.train()



if __name__ == "__main__":
    main()