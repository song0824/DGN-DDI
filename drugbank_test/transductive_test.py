import os
import time
import contextlib
import numpy as np
import torch
import logging
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_recall_curve, confusion_matrix,
    roc_curve, precision_score, recall_score, f1_score, matthews_corrcoef
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Any
import json
import argparse

# 导入项目中的其他模块
from data_preprocessing import DDIDataLoader, DrugDataset, DrugDataLoader, merge_ddi_batches
from models import DGN_DDI
from custom_loss import SigmoidLoss, FocalLoss, PairwiseHingeLoss, AdaptiveLoss

# ================== 配置 ==================
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def get_test_config(profile: str = "full"):
    """获取测试配置并验证必要参数。默认对齐 full 训练配置做全量评估。"""
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _drugbank_dir = os.path.join(_script_dir, "drugbank")
    config = {
        # 数据与路径（相对当前脚本目录）
        'test_csv': os.path.join(_drugbank_dir, "fold0", "test.csv"),
        'model_path': os.path.join(_script_dir, "checkpoints", "best_model.pth"),
        'results_dir': os.path.join(_script_dir, "test_results"),

        # 数据处理
        'neg_ent': 3,
        'subset_size': None,

        # 模型超参数（需要与训练时保持一致）
        'hidden_dim': 256,
        'kge_dim': 256,
        'heads_out_feat_params': [64, 64, 64],
        'blocks_params': [4, 4, 4],

        # 测试超参数
        'batch_size': 48,
        'loss_fn': 'sigmoid',
        'focal_alpha': 0.75,
        'focal_gamma': 2.0,
        'focal_balance_by_counts': True,
        'sigmoid_label_smoothing': 0.0,
        'hinge_margin': 1.0,
        'min_recall_for_threshold': 0.0,
        'use_amp': True,

        'warm_bipartite_cache': True,
        'warm_bipartite_max_pairs': None,

        # 其他设置
        'save_predictions': True,
        'save_visualizations': True,
        'threshold': 0.5,
        # 若 checkpoint 含验证集最优阈值，则默认优先使用它。
        'use_checkpoint_threshold': True,
        'random_seed': 42
    }

    if str(profile).lower().strip() == "smoke":
        config['subset_size'] = 2000
        config['batch_size'] = 32
        config['loss_fn'] = 'sigmoid'
        config['heads_out_feat_params'] = [128, 128, 128]
        config['blocks_params'] = [2, 2, 2]

    # 验证必要配置项
    required_keys = ['test_csv', 'model_path', 'results_dir', 'hidden_dim', 'kge_dim']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")

    return config


def _merge_eval_config_from_checkpoint(config: dict, checkpoint: dict) -> dict:
    """用 checkpoint 中的训练配置覆盖关键评估超参，保证训练/测试口径一致。"""
    train_cfg = checkpoint.get("train_config")
    if not isinstance(train_cfg, dict):
        return config

    synced_keys = [
        "hidden_dim",
        "kge_dim",
        "heads_out_feat_params",
        "blocks_params",
        "neg_ent",
        "loss_fn",
        "focal_alpha",
        "focal_gamma",
        "focal_balance_by_counts",
        "sigmoid_label_smoothing",
        "hinge_margin",
        "min_recall_for_threshold",
    ]
    for key in synced_keys:
        if key in train_cfg:
            config[key] = train_cfg[key]
    return config


class DDITester:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.checkpoint = self._load_checkpoint_metadata()

        # 1. 初始化数据加载器
        self.ddi_loader = DDIDataLoader()

        # 2. 加载测试数据
        self.load_test_data()

        # 3. 初始化模型
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

        # 4. 初始化损失函数
        self.loss_fn = self._get_loss_function(self.config['loss_fn']).to(self.device)

        # 5. 加载训练好的模型
        self.load_model()

        # 6. 存储测试结果
        self.test_results = {}
        self.predictions = []
        self.ground_truths = []
        self.scores = []
        self.probabilities = []

        # 记录整体推理时间
        self.total_inference_time = 0

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

    def _load_checkpoint_metadata(self):
        model_path = self.config['model_path']
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint {model_path} not found")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.config = _merge_eval_config_from_checkpoint(self.config, checkpoint)
        return checkpoint

    def load_test_data(self):
        """加载测试数据集，优化数据加载效率"""
        logger.info("Loading test dataset...")
        if not os.path.exists(self.config['test_csv']):
            raise FileNotFoundError(f"Test dataset file {self.config['test_csv']} not found")

        logger.info(f"Reading test data from {self.config['test_csv']}")
        self.test_dataset = DrugDataset(
            self.config['test_csv'],
            self.ddi_loader,
            neg_ent=self.config['neg_ent'],
            shuffle=False
        )

        # 如果指定了子集大小，则使用子集
        subset_size = self.config.get('subset_size')
        if subset_size and subset_size > 0:
            subset_size = min(subset_size, len(self.test_dataset))
            logger.info(f"Using a subset of test data: {subset_size} samples out of {len(self.test_dataset)}")
            # 使用更高效的子集采样方式
            indices = torch.randperm(len(self.test_dataset))[:subset_size]
            self.test_dataset.tri_list = [self.test_dataset.tri_list[i] for i in indices]

        # 根据设备自动调整num_workers
        num_workers = 0 if self.device.type == 'cpu' else min(os.cpu_count(), 4)
        self.test_loader = DrugDataLoader(
            self.test_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=self.device.type == 'cuda'  # 为GPU优化
        )

        logger.info(f"Test data loaded - {len(self.test_dataset)} samples")

        if self.config.get("warm_bipartite_cache", True):
            total, loaded = self.test_dataset.warm_bipartite_cache(
                max_pairs=self.config.get("warm_bipartite_max_pairs")
            )
            logger.info("Bipartite warm [test]: %d unique pairs, %d in memory cache", total, loaded)

    def load_model(self):
        """加载训练好的模型，解决PyTorch 2.6+兼容性问题"""
        model_path = self.config['model_path']
        logger.info(f"Loading model from {model_path}")
        checkpoint = self.checkpoint

        # 加载模型状态
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # 记录模型信息
        if 'epoch' in checkpoint:
            logger.info(f"Model was trained for {checkpoint['epoch']} epochs")
        if 'best_val_auc' in checkpoint:
            logger.info(f"Best validation AUC: {checkpoint['best_val_auc']:.4f}")
        if (
            self.config.get("use_checkpoint_threshold", True)
            and not self.config.get("_threshold_overridden", False)
            and checkpoint.get("best_threshold_for_test") is not None
        ):
            self.config["threshold"] = float(checkpoint["best_threshold_for_test"])
            logger.info("Using threshold from checkpoint: %.4f", self.config["threshold"])

        logger.info("Model loaded successfully!")

    def test(self):
        """主测试函数，优化推理效率"""
        logger.info("Starting model testing...")
        start_time = time.time()

        self.model.eval()
        total_loss = 0
        all_scores, all_labels = [], []
        batch_times = []

        # 自动调整进度条显示
        progress_bar = tqdm(
            self.test_loader,
            desc="Testing",
            leave=True,
            total=len(self.test_loader)
        )

        # 启用CUDA推理优化（如果可用）
        with torch.no_grad():
            if self.device.type == 'cuda':
                torch.backends.cudnn.benchmark = True

            for batch_idx, batch in enumerate(progress_bar):
                batch_start_time = time.time()

                if len(batch) == 3:
                    pos_tri, neg_tri, _ = batch
                else:
                    pos_tri, neg_tri = batch

                # 跳过无效批次
                if pos_tri[0] is None:
                    continue

                try:
                    pos_h, pos_t, pos_r, pos_bg = pos_tri
                    pos_h, pos_t, pos_r = (x.to(self.device, non_blocking=True) for x in [pos_h, pos_t, pos_r])
                    pos_bg = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                              for k, v in pos_bg.items()}

                    neg_h, neg_t, neg_r, neg_bg = neg_tri
                    if neg_h is not None and neg_h.num_graphs > 0:
                        neg_h, neg_t, neg_r = (x.to(self.device, non_blocking=True) for x in [neg_h, neg_t, neg_r])
                        neg_bg = {k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                                  for k, v in neg_bg.items()}
                        merged_tri, n_pos = merge_ddi_batches(
                            (pos_h, pos_t, pos_r, pos_bg),
                            (neg_h, neg_t, neg_r, neg_bg),
                        )
                    else:
                        merged_tri = (pos_h, pos_t, pos_r, pos_bg)
                        n_pos = pos_r.size(0)

                    _ctx = contextlib.nullcontext()
                    if self.config.get("use_amp", True) and self.device.type == "cuda":
                        from torch.amp import autocast
                        _ctx = autocast("cuda", enabled=True)
                    with _ctx:
                        mh, mt, mr, mbg = merged_tri
                        batch_scores = self.model(mh, mt, mr, mbg)
                        p_scores = batch_scores[:n_pos]
                        n_scores = batch_scores[n_pos:] if batch_scores.size(0) > n_pos else torch.tensor([], device=self.device)

                    loss, _, _ = self.loss_fn(p_scores, n_scores)
                    total_loss += loss.item()

                    all_scores.append(p_scores.cpu())
                    all_labels.append(torch.ones_like(p_scores.cpu()))

                    if n_scores.numel() > 0:
                        all_scores.append(n_scores.cpu())
                        all_labels.append(torch.zeros_like(n_scores.cpu()))

                except Exception as e:
                    logger.warning(f"Error processing batch {batch_idx}: {e}")
                    continue

                # 记录时间和进度
                batch_time = time.time() - batch_start_time
                batch_times.append(batch_time)
                progress_bar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Avg_Time': f"{np.mean(batch_times):.3f}s"
                })

            # 关闭CUDA优化
            if self.device.type == 'cuda':
                torch.backends.cudnn.benchmark = False

        # 计算整体指标
        avg_loss = total_loss / len(self.test_loader) if len(self.test_loader) > 0 else 0
        all_scores = torch.cat(all_scores) if all_scores else torch.tensor([])
        all_labels = torch.cat(all_labels) if all_labels else torch.tensor([])

        # 存储结果（all_scores 为 logit/原始分；阈值比较统一使用概率）
        self.scores = all_scores.numpy()
        self.ground_truths = all_labels.numpy()
        probs = torch.sigmoid(all_scores).numpy()
        self.probabilities = probs
        self.predictions = (probs > self.config['threshold']).astype(int)

        # 计算各种指标
        test_metrics = self._calculate_metrics(all_labels.numpy(), all_scores.numpy())
        test_metrics['avg_loss'] = avg_loss
        test_metrics['avg_batch_time'] = np.mean(batch_times) if batch_times else 0
        test_metrics['total_samples'] = len(all_labels)
        test_metrics['total_inference_time'] = time.time() - start_time

        self.test_results = test_metrics

        # 打印结果
        self._print_results(test_metrics)

        self.total_inference_time = test_metrics['total_inference_time']

        return test_metrics

    def _calculate_metrics(self, y_true, y_scores):
        """计算各种评估指标"""
        if len(y_true) == 0:
            logger.warning("No valid samples to calculate metrics")
            return {}

        prob = 1.0 / (1.0 + np.exp(-np.clip(y_scores.astype(np.float64), -50, 50)))
        y_pred = (prob > float(self.config['threshold'])).astype(int)

        metrics = {}

        # 基本指标（单类时降级，避免抛异常）
        if len(np.unique(y_true)) < 2:
            metrics['auc'] = 0.5
            metrics['aupr'] = 0.0
        else:
            metrics['auc'] = roc_auc_score(y_true, y_scores)
            metrics['aupr'] = average_precision_score(y_true, y_scores)
        metrics['accuracy'] = accuracy_score(y_true, y_pred)

        # 精确率、召回率、F1分数
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)

        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        if cm.size == 4:  # 确保矩阵是2x2
            tn, fp, fn, tp = cm.ravel()
        else:
            # 处理只有一类的边缘情况
            tn, fp, fn, tp = (0, 0, 0, 0)
            if len(cm) == 1:
                if np.argmax(cm.shape) == 0:
                    tn, fp = cm[0, 0], 0
                else:
                    tp, fn = cm[0, 0], 0

        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)

        # 特异性（真负率）
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # MCC (Matthews Correlation Coefficient)
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)

        return metrics

    def _print_results(self, metrics):
        """打印测试结果"""
        logger.info("\n" + "=" * 50)
        logger.info("TEST RESULTS")
        logger.info("=" * 50)

        logger.info(f"Total Samples: {metrics.get('total_samples', 0)}")
        logger.info(f"Average Loss: {metrics.get('avg_loss', 0):.4f}")
        logger.info(f"Average Batch Time: {metrics.get('avg_batch_time', 0):.3f}s")
        logger.info(f"Total Inference Time: {metrics.get('total_inference_time', 0):.3f}s")

        logger.info("\nClassification Metrics:")
        logger.info(f"AUC-ROC: {metrics.get('auc', 0):.4f}")
        logger.info(f"AUC-PR:  {metrics.get('aupr', 0):.4f}")
        logger.info(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
        logger.info(f"Precision: {metrics.get('precision', 0):.4f}")
        logger.info(f"Recall: {metrics.get('recall', 0):.4f}")
        logger.info(f"F1-Score: {metrics.get('f1_score', 0):.4f}")
        logger.info(f"Specificity: {metrics.get('specificity', 0):.4f}")
        logger.info(f"MCC: {metrics.get('mcc', 0):.4f}")

        logger.info("\nConfusion Matrix:")
        logger.info(f"True Positives:  {metrics.get('true_positives', 0)}")
        logger.info(f"False Positives: {metrics.get('false_positives', 0)}")
        logger.info(f"True Negatives:  {metrics.get('true_negatives', 0)}")
        logger.info(f"False Negatives: {metrics.get('false_negatives', 0)}")

        logger.info("=" * 50)

    def save_results(self):
        """保存测试结果"""
        if not os.path.exists(self.config['results_dir']):
            os.makedirs(self.config['results_dir'])

        # 保存指标
        metrics_file = os.path.join(self.config['results_dir'], 'test_metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        logger.info(f"Metrics saved to {metrics_file}")

        # 保存预测结果
        if self.config['save_predictions'] and len(self.ground_truths) > 0:
            predictions_df = pd.DataFrame({
                'ground_truth': self.ground_truths,
                'predicted_score': self.scores,
                'predicted_label': self.predictions
            })
            pred_file = os.path.join(self.config['results_dir'], 'predictions.csv')
            predictions_df.to_csv(pred_file, index=False)
            logger.info(f"Predictions saved to {pred_file}")

        # 保存可视化
        if self.config['save_visualizations'] and len(self.ground_truths) > 0:
            self._save_visualizations()

    def _save_visualizations(self):
        """保存可视化图表，不依赖seaborn"""
        try:
            # 创建一个综合报告图
            plt.style.use('default')

            # 1. ROC和PR曲线组合图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # ROC曲线
            fpr, tpr, _ = roc_curve(self.ground_truths, self.scores)
            ax1.plot(fpr, tpr, color='darkorange', lw=2,
                     label=f'ROC curve (AUC = {self.test_results["auc"]:.4f})')
            ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax1.set_xlim([0.0, 1.0])
            ax1.set_ylim([0.0, 1.05])
            ax1.set_xlabel('False Positive Rate')
            ax1.set_ylabel('True Positive Rate')
            ax1.set_title('ROC Curve')
            ax1.legend(loc="lower right")
            ax1.grid(True, alpha=0.3)

            # PR曲线
            precision, recall, _ = precision_recall_curve(self.ground_truths, self.scores)
            ax2.plot(recall, precision, color='red', lw=2,
                     label=f'PR curve (AUC = {self.test_results["aupr"]:.4f})')
            ax2.set_xlim([0.0, 1.0])
            ax2.set_ylim([0.0, 1.05])
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            ax2.set_title('Precision-Recall Curve')
            ax2.legend(loc="lower left")
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            roc_pr_file = os.path.join(self.config['results_dir'], 'roc_pr_curves.png')
            plt.savefig(roc_pr_file, dpi=300, bbox_inches='tight')
            plt.close()

            # 2. 混淆矩阵（不依赖seaborn）
            cm = confusion_matrix(self.ground_truths, self.predictions)
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)

            # 添加标签和标题
            classes = ['No Interaction', 'Interaction']
            ax.set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   xticklabels=classes, yticklabels=classes,
                   title='Confusion Matrix',
                   ylabel='True Label',
                   xlabel='Predicted Label')

            # 添加文本标注
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black")

            cm_file = os.path.join(self.config['results_dir'], 'confusion_matrix.png')
            plt.savefig(cm_file, dpi=300, bbox_inches='tight')
            plt.close()

            # 3. 概率分布（阈值在概率空间）
            plt.figure(figsize=(10, 6))
            plt.hist(self.probabilities[self.ground_truths == 0], bins=50, alpha=0.7,
                     label='No Interaction', color='blue')
            plt.hist(self.probabilities[self.ground_truths == 1], bins=50, alpha=0.7,
                     label='Interaction', color='red')
            plt.axvline(x=self.config['threshold'], color='black', linestyle='--',
                        label=f'Threshold = {self.config["threshold"]}')
            plt.xlabel('Prediction Probability')
            plt.ylabel('Frequency')
            plt.title('Distribution of Prediction Probabilities')
            plt.legend()
            plt.grid(True, alpha=0.3)
            dist_file = os.path.join(self.config['results_dir'], 'score_distribution.png')
            plt.savefig(dist_file, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info("Visualizations saved successfully!")

        except Exception as e:
            logger.error(f"Error saving visualizations: {e}", exc_info=True)

    def detailed_analysis(self):
        """进行详细分析"""
        if len(self.ground_truths) == 0:
            logger.warning("No data available for detailed analysis")
            return

        logger.info("\nPerforming detailed analysis...")

        # 计算不同阈值下的指标
        thresholds = np.arange(0.1, 1.0, 0.1)
        threshold_metrics = []

        for threshold in thresholds:
            y_pred = (self.probabilities > threshold).astype(int)
            precision = precision_score(self.ground_truths, y_pred, zero_division=0)
            recall = recall_score(self.ground_truths, y_pred, zero_division=0)
            f1 = f1_score(self.ground_truths, y_pred, zero_division=0)
            accuracy = accuracy_score(self.ground_truths, y_pred)

            threshold_metrics.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'accuracy': accuracy
            })

        # 保存阈值分析结果
        threshold_df = pd.DataFrame(threshold_metrics)
        threshold_file = os.path.join(self.config['results_dir'], 'threshold_analysis.csv')
        threshold_df.to_csv(threshold_file, index=False)

        # 找到最佳阈值（基于F1分数）
        best_threshold_idx = threshold_df['f1_score'].idxmax()
        best_threshold = threshold_df.iloc[best_threshold_idx]

        logger.info(f"Best threshold (F1): {best_threshold['threshold']:.1f}")
        logger.info(f"  Precision: {best_threshold['precision']:.4f}")
        logger.info(f"  Recall: {best_threshold['recall']:.4f}")
        logger.info(f"  F1-Score: {best_threshold['f1_score']:.4f}")
        logger.info(f"  Accuracy: {best_threshold['accuracy']:.4f}")


# ================== 主函数 ==================

def main():
    """主函数入口，支持命令行参数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Test DDI prediction model')
    parser.add_argument('--profile', choices=['full', 'smoke'], default='full', help='Use full or smoke-style eval defaults')
    parser.add_argument('--config', type=str, help='Path to custom config file (JSON)')
    parser.add_argument('--model-path', type=str, help='Override model path')
    parser.add_argument('--test-csv', type=str, help='Override test CSV path')
    parser.add_argument('--batch-size', type=int, help='Override batch size')
    parser.add_argument('--subset-size', type=int, help='Use a random subset of test data')
    parser.add_argument('--threshold', type=float, help='Override prediction threshold')

    args = parser.parse_args()

    # 加载配置
    config = get_test_config(profile=args.profile)

    # 从JSON文件加载自定义配置
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
            config.update(custom_config)
            logger.info(f"Loaded custom config from {args.config}")

    # 命令行参数覆盖配置
    if args.model_path:
        config['model_path'] = args.model_path
    if args.test_csv:
        config['test_csv'] = args.test_csv
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.subset_size is not None:
        config['subset_size'] = args.subset_size
    if args.threshold is not None:
        config['threshold'] = args.threshold
        config["_threshold_overridden"] = True

    # 创建结果保存目录
    os.makedirs(config['results_dir'], exist_ok=True)

    # 设置随机种子
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['random_seed'])
        torch.cuda.manual_seed_all(config['random_seed'])

    try:
        # 创建测试器并开始测试
        tester = DDITester(config)

        # 执行测试
        test_results = tester.test()

        # 保存结果
        tester.save_results()

        # 详细分析
        tester.detailed_analysis()

        logger.info("\nTesting completed successfully!")
        logger.info(f"Results saved to: {config['results_dir']}")

        return test_results

    except Exception as e:
        logger.error(f"Error during testing: {e}", exc_info=True)
        raise e


if __name__ == "__main__":
    main()
