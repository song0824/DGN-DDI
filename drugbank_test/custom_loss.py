import torch
from torch import nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class SigmoidLoss(nn.Module):
    """
    Sigmoid Loss with enhanced numerical stability and optional adversarial temperature.
    """

    def __init__(self, adv_temperature=None, label_smoothing=0.0):
        super().__init__()
        self.adv_temperature = adv_temperature
        self.label_smoothing = label_smoothing

    def forward(self, p_scores, n_scores):
        """
        Calculate sigmoid loss with numerical stability enhancements.

        Args:
            p_scores: Positive samples scores [batch_size]
            n_scores: Negative samples scores [batch_size]
        """
        # 数值稳定性预处理
        p_scores = torch.nan_to_num(p_scores, nan=0.0, posinf=10.0, neginf=-10.0)
        n_scores = torch.nan_to_num(n_scores, nan=0.0, posinf=10.0, neginf=-10.0)

        # 限制分数范围以避免梯度爆炸
        p_scores = torch.clamp(p_scores, min=-50, max=50)
        n_scores = torch.clamp(n_scores, min=-50, max=50)

        # 如果没有负样本，创建虚拟负样本
        if n_scores.numel() == 0:
            n_scores = -p_scores + torch.randn_like(p_scores) * 0.1
            logger.debug("Created virtual negative samples")

        # 对抗性温度调节（可选）
        if self.adv_temperature and self.adv_temperature > 0:
            try:
                # 计算softmax权重，增加数值稳定性
                n_scores_for_weight = torch.clamp(n_scores, min=-10, max=10)
                weights = F.softmax(self.adv_temperature * n_scores_for_weight, dim=-1).detach()
                # 防止权重过小
                weights = torch.clamp(weights, min=1e-8)
                # 应用权重
                n_scores = weights * n_scores
            except Exception as e:
                logger.warning(f"Error in adversarial weighting: {e}, using unweighted scores")

        # 计算损失，使用更稳定的log-sigmoid
        try:
            # 正样本损失：-log(sigmoid(p_scores)) = log(1 + exp(-p_scores))
            # 使用logsigmoid提供更好的数值稳定性
            p_loss = -F.logsigmoid(p_scores)

            # 负样本损失：-log(1 - sigmoid(n_scores)) = -log(sigmoid(-n_scores)) = log(1 + exp(n_scores))
            n_loss = -F.logsigmoid(-n_scores)

            # 处理无效值
            p_loss = torch.nan_to_num(p_loss, nan=1.0, posinf=10.0, neginf=0.0)
            n_loss = torch.nan_to_num(n_loss, nan=1.0, posinf=10.0, neginf=0.0)

            # 标签平滑（如果启用）
            if self.label_smoothing > 0:
                # 对正样本应用标签平滑
                uniform_loss_p = -torch.log(torch.tensor(0.5, device=p_loss.device))
                p_loss = (1 - self.label_smoothing) * p_loss + self.label_smoothing * uniform_loss_p

                # 对负样本应用标签平滑
                uniform_loss_n = -torch.log(torch.tensor(0.5, device=n_loss.device))
                n_loss = (1 - self.label_smoothing) * n_loss + self.label_smoothing * uniform_loss_n

            # 计算平均损失
            p_loss_mean = p_loss.mean()
            n_loss_mean = n_loss.mean()

            # 加权组合损失
            total_loss = (p_loss_mean + n_loss_mean) / 2

            # 最终数值稳定性检查
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logger.warning("Invalid loss detected, using fallback loss")
                total_loss = torch.tensor(1.0, device=p_scores.device, requires_grad=True)
                p_loss_mean = torch.tensor(0.5, device=p_scores.device)
                n_loss_mean = torch.tensor(0.5, device=p_scores.device)

            return total_loss, p_loss_mean, n_loss_mean

        except Exception as e:
            logger.error(f"Error in loss calculation: {e}")
            # 返回安全的默认值
            device = p_scores.device if p_scores.numel() > 0 else torch.device('cpu')
            total_loss = torch.tensor(1.0, device=device, requires_grad=True)
            p_loss_mean = torch.tensor(0.5, device=device)
            n_loss_mean = torch.tensor(0.5, device=device)
            return total_loss, p_loss_mean, n_loss_mean


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in DDI prediction.
    """

    def __init__(self, alpha=0.5, gamma=2.0, balance_by_counts=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.balance_by_counts = balance_by_counts

    def forward(self, p_scores, n_scores):
        # 数值稳定性处理
        p_scores = torch.nan_to_num(p_scores, nan=0.0, posinf=10.0, neginf=-10.0)
        n_scores = torch.nan_to_num(n_scores, nan=0.0, posinf=10.0, neginf=-10.0)
        p_scores = torch.clamp(p_scores, min=-50, max=50)
        n_scores = torch.clamp(n_scores, min=-50, max=50)

        # 负样本可能为空（例如异常批次/过滤后），给一个可微回退，避免 mean(empty)
        if n_scores.numel() == 0:
            n_scores = -p_scores.detach()

        # 计算sigmoid概率
        p_probs = torch.sigmoid(p_scores)
        n_probs = torch.sigmoid(n_scores)

        # 计算focal loss
        p_loss = -self.alpha * (1 - p_probs) ** self.gamma * torch.log(p_probs + 1e-8)
        n_loss = -(1 - self.alpha) * n_probs ** self.gamma * torch.log(1 - n_probs + 1e-8)
        p_loss = torch.nan_to_num(p_loss, nan=1.0, posinf=10.0, neginf=0.0)
        n_loss = torch.nan_to_num(n_loss, nan=1.0, posinf=10.0, neginf=0.0)

        p_loss_mean = p_loss.mean()
        n_loss_mean = n_loss.mean()
        if self.balance_by_counts:
            n_pos = max(1, int(p_scores.numel()))
            n_neg = max(1, int(n_scores.numel()))
            total_loss = (p_loss_mean * n_pos + n_loss_mean * n_neg) / (n_pos + n_neg)
        else:
            total_loss = p_loss_mean + n_loss_mean

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.warning("Invalid focal loss detected, fallback to safe constants")
            total_loss = torch.tensor(1.0, device=p_scores.device, requires_grad=True)
            p_loss_mean = torch.tensor(0.5, device=p_scores.device)
            n_loss_mean = torch.tensor(0.5, device=p_scores.device)

        return total_loss, p_loss_mean, n_loss_mean


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for better representation learning.
    """

    def __init__(self, margin=1.0, temperature=0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(self, p_scores, n_scores):
        # 数值稳定性处理
        p_scores = torch.clamp(p_scores, min=-10, max=10)
        n_scores = torch.clamp(n_scores, min=-10, max=10)

        # 对比损失：正样本应该有高分，负样本应该有低分
        p_loss = torch.clamp(self.margin - p_scores, min=0).pow(2)
        n_loss = torch.clamp(n_scores - (-self.margin), min=0).pow(2)

        p_loss_mean = p_loss.mean()
        n_loss_mean = n_loss.mean()
        total_loss = p_loss_mean + n_loss_mean

        return total_loss, p_loss_mean, n_loss_mean


class AdaptiveLoss(nn.Module):
    """
    Adaptive loss that combines multiple loss functions based on training progress.
    """

    def __init__(self, base_loss='sigmoid', adaptive_weight=True):
        super().__init__()
        self.sigmoid_loss = SigmoidLoss(label_smoothing=0.1)
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.contrastive_loss = ContrastiveLoss(margin=1.0)
        self.adaptive_weight = adaptive_weight
        self.base_loss = base_loss

        # 可学习的权重参数
        if adaptive_weight:
            self.loss_weights = nn.Parameter(torch.tensor([1.0, 0.3, 0.2]))
        else:
            self.register_buffer('loss_weights', torch.tensor([1.0, 0.3, 0.2]))

    def forward(self, p_scores, n_scores):
        # 计算不同的损失
        sigmoid_loss, sigmoid_p, sigmoid_n = self.sigmoid_loss(p_scores, n_scores)
        focal_loss, focal_p, focal_n = self.focal_loss(p_scores, n_scores)
        contrastive_loss, contr_p, contr_n = self.contrastive_loss(p_scores, n_scores)

        # 归一化权重
        if self.adaptive_weight:
            weights = F.softmax(self.loss_weights, dim=0)
        else:
            weights = self.loss_weights / self.loss_weights.sum()

        # 组合损失
        total_loss = (weights[0] * sigmoid_loss +
                      weights[1] * focal_loss +
                      weights[2] * contrastive_loss)

        # 组合正负样本损失
        p_loss_combined = (weights[0] * sigmoid_p +
                           weights[1] * focal_p +
                           weights[2] * contr_p)

        n_loss_combined = (weights[0] * sigmoid_n +
                           weights[1] * focal_n +
                           weights[2] * contr_n)

        return total_loss, p_loss_combined, n_loss_combined


class PairwiseHingeLoss(nn.Module):
    """
    实现成对铰链损失（Pairwise Hinge Loss），也称为间隔排序损失。
    目标是使正样本的分数比负样本的分数至少高出一个间隔(margin)。
    损失计算公式: Loss = max(0, margin - positive_score + negative_score)
    """

    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin
        # margin值可以根据你的模型输出范围进行调整，2.0是一个比较强的约束

    def forward(self, p_scores, n_scores):
        # 如果没有正样本，则不计算损失
        if p_scores.numel() == 0:
            return torch.tensor(0.0, device=n_scores.device, requires_grad=True), torch.tensor(0.0), torch.tensor(0.0)

        # 如果没有负样本，无法进行成对比较，可以惩罚那些分数不够高的正样本
        if n_scores.numel() == 0:
            p_loss = F.relu(self.margin - p_scores).mean()
            return p_loss, p_loss, torch.tensor(0.0)


        n_per_p = n_scores.size(0) // p_scores.size(0)

        # 处理特殊情况：如果负样本比正样本少
        if n_per_p == 0:
            p_per_n = p_scores.size(0) // n_scores.size(0)
            n_scores_expanded = n_scores.repeat_interleave(p_per_n)
            num_samples = min(p_scores.size(0), n_scores_expanded.size(0))
            p_scores_matched = p_scores[:num_samples]
            n_scores_matched = n_scores_expanded[:num_samples]
        else:  # 正常情况
            p_scores_expanded = p_scores.repeat_interleave(n_per_p)
            num_samples = min(p_scores_expanded.size(0), n_scores.size(0))
            p_scores_matched = p_scores_expanded[:num_samples]
            n_scores_matched = n_scores[:num_samples]

        # 计算Hinge Loss
        loss = F.relu(self.margin - p_scores_matched + n_scores_matched)
        total_loss = loss.mean()

        return total_loss, total_loss, total_loss