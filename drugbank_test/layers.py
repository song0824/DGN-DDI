import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Data
from typing import List, Tuple, Union, Optional
import logging

logger = logging.getLogger(__name__)
try:
    import torch_scatter
except ImportError:
    torch_scatter = None
    logger.info("torch_scatter is not available; using torch.index_add fallback.")


class RobustLayerNorm(nn.Module):
    """鲁棒的LayerNorm，处理单样本和批次normalization"""

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = normalized_shape

        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def _layer_norm(self, x: torch.Tensor) -> torch.Tensor:
        """AMP 下在 float32 中计算 LayerNorm，再转回输入 dtype。"""
        orig_dtype = x.dtype
        if self.elementwise_affine:
            y = F.layer_norm(
                x.float(),
                self.normalized_shape,
                self.weight.float(),
                self.bias.float(),
                self.eps,
            )
        else:
            y = F.layer_norm(x.float(), self.normalized_shape, None, None, self.eps)
        return y.to(orig_dtype)

    def _affine(self, x: torch.Tensor) -> torch.Tensor:
        if not self.elementwise_affine:
            return x
        w = self.weight.to(dtype=x.dtype)
        b = self.bias.to(dtype=x.dtype)
        return x * w + b

    def forward(self, input, batch=None):
        if input.size(0) == 1:
            return self._affine(input)

        if batch is None:
            return self._layer_norm(input)

        unique_batches = torch.unique(batch)
        if len(unique_batches) == 1:
            return self._layer_norm(input)

        output = torch.empty_like(input)
        for b_idx in unique_batches:
            mask = batch == b_idx
            if mask.sum() > 1:
                output[mask] = self._layer_norm(input[mask])
            elif self.elementwise_affine:
                output[mask] = self._affine(input[mask])
            else:
                output[mask] = input[mask]
        return output


class InterGraphAttention(nn.Module):
    """增强的图间注意力，支持双粒度特征和多种二部图格式"""

    def __init__(self, in_dim: int, dropout: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.q = nn.Linear(in_dim, in_dim)
        self.k = nn.Linear(in_dim, in_dim)
        self.v = nn.Linear(in_dim, in_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.norm = RobustLayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / (in_dim ** 0.5)

        # 添加双粒度融合权重
        self.granularity_weight = nn.Parameter(torch.tensor(0.6))  # 原子级权重

    def _safe_attention_computation(self, q, k, v, mask=None, bias=None):
        """安全的注意力计算，处理数值稳定性"""
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 添加位置偏置（二部图信息）
        if bias is not None:
            scores = scores + bias

        # 应用mask
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))

        # 计算注意力权重，处理数值稳定性
        attn = self.softmax(scores)
        attn = torch.where(torch.isnan(attn), torch.zeros_like(attn), attn)
        attn = self.dropout(attn)

        # 应用注意力
        out = torch.matmul(attn, v)
        return out, attn

    def forward(self, h_data: Data, t_data: Data,
                b_graph: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播，支持双粒度特征处理

        Args:
            h_data: 头实体图数据
            t_data: 尾实体图数据
            b_graph: 二部图张量或字典
        """
        h_x, t_x = h_data.x, t_data.x
        h_batch, t_batch = h_data.batch, t_data.batch

        # 鲁棒性检查，处理空图或单节点图
        if h_x.size(0) == 0 or t_x.size(0) == 0:
            return torch.empty(0, self.in_dim, device=h_x.device), \
                torch.empty(0, self.in_dim, device=t_x.device)

        # 将批次中的图数据转换为密集张量
        h_dense, h_mask = to_dense_batch(h_x, h_batch)
        t_dense, t_mask = to_dense_batch(t_x, t_batch)

        # 检查密集张量是否有效，特别是当批次只包含单个图时
        if h_dense.size(0) == 0 or h_dense.size(1) == 0 or t_dense.size(0) == 0 or t_dense.size(1) == 0:
            return torch.empty(0, self.in_dim, device=h_x.device), \
                torch.empty(0, self.in_dim, device=t_x.device)

        try:
            # 转换为密集表示
            h_dense, h_mask = to_dense_batch(h_x, h_batch)
            t_dense, t_mask = to_dense_batch(t_x, t_batch)

            batch_size, max_h_nodes, h_dim = h_dense.shape
            _, max_t_nodes, t_dim = t_dense.shape

            logger.debug(f"Dense shapes - h: {h_dense.shape}, t: {t_dense.shape}")

            # 计算查询、键、值
            q_h = self.q(h_dense)  # [batch_size, max_h_nodes, in_dim]
            k_t = self.k(t_dense)  # [batch_size, max_t_nodes, in_dim]
            v_t = self.v(t_dense)  # [batch_size, max_t_nodes, in_dim]

            q_t = self.q(t_dense)  # 反向注意力
            k_h = self.k(h_dense)
            v_h = self.v(h_dense)

            # 处理二部图信息
            b_graph_processed = None
            if b_graph is not None:
                if isinstance(b_graph, dict):
                    # 如果是字典格式，选择合适的二部图
                    b_graph_processed = b_graph.get('fused', b_graph.get('atom', None))
                else:
                    b_graph_processed = b_graph

                # 调整二部图维度以匹配dense representation
                if b_graph_processed is not None:
                    b_graph_processed = self._adjust_bipartite_graph(
                        b_graph_processed, batch_size, max_h_nodes, max_t_nodes
                    )

            # 构建attention mask
            attn_mask_ht = None
            attn_mask_th = None
            if h_mask is not None and t_mask is not None:
                attn_mask_ht = h_mask.unsqueeze(2) & t_mask.unsqueeze(1)  # [batch, h_nodes, t_nodes]
                attn_mask_th = t_mask.unsqueeze(2) & h_mask.unsqueeze(1)  # [batch, t_nodes, h_nodes]

            # 计算h到t的注意力
            out_h, attn_ht = self._safe_attention_computation(
                q_h, k_t, v_t, mask=attn_mask_ht, bias=b_graph_processed
            )

            # 计算t到h的注意力（反向）
            b_graph_transposed = b_graph_processed.transpose(-2, -1) if b_graph_processed is not None else None
            out_t, attn_th = self._safe_attention_computation(
                q_t, k_h, v_h, mask=attn_mask_th, bias=b_graph_transposed
            )

            # 转换回原始格式
            out_h_flat = out_h[h_mask] if h_mask is not None else out_h.view(-1, out_h.size(-1))
            out_t_flat = out_t[t_mask] if t_mask is not None else out_t.view(-1, out_t.size(-1))

            # 应用layer normalization和残差连接
            out_h_final = self.norm(out_h_flat + h_x, h_batch)
            out_t_final = self.norm(out_t_flat + t_x, t_batch)

            return out_h_final, out_t_final

        except Exception as e:
            logger.error(f"Error in InterGraphAttention: {e}")
            logger.error(f"h_x shape: {h_x.shape}, t_x shape: {t_x.shape}")
            if b_graph is not None:
                logger.error(f"b_graph type: {type(b_graph)}")
            raise e

    def _adjust_bipartite_graph(self, b_graph, batch_size, max_h_nodes, max_t_nodes):
        """调整二部图维度以匹配dense representation"""
        if b_graph is None:
            return None

        try:
            # 处理不同维度的二部图
            if len(b_graph.shape) == 2:  # [h_nodes, t_nodes]
                # 扩展到batch维度
                b_graph = b_graph.unsqueeze(0).expand(batch_size, -1, -1)

            if len(b_graph.shape) == 3:  # [batch_size, h_nodes, t_nodes]
                current_batch, current_h, current_t = b_graph.shape

                # 调整batch维度
                if current_batch != batch_size:
                    if current_batch == 1:
                        b_graph = b_graph.expand(batch_size, -1, -1)
                    else:
                        # 截取或填充
                        if current_batch > batch_size:
                            b_graph = b_graph[:batch_size]
                        else:
                            pad_batch = batch_size - current_batch
                            last_sample = b_graph[-1:].expand(pad_batch, -1, -1)
                            b_graph = torch.cat([b_graph, last_sample], dim=0)

                # 调整空间维度
                if current_h != max_h_nodes or current_t != max_t_nodes:
                    new_b_graph = torch.zeros(batch_size, max_h_nodes, max_t_nodes,
                                              device=b_graph.device, dtype=b_graph.dtype)

                    copy_h = min(current_h, max_h_nodes)
                    copy_t = min(current_t, max_t_nodes)

                    new_b_graph[:, :copy_h, :copy_t] = b_graph[:, :copy_h, :copy_t]

                    # 对扩展区域填充小的正值
                    if max_h_nodes > copy_h:
                        new_b_graph[:, copy_h:, :copy_t] = 0.01
                    if max_t_nodes > copy_t:
                        new_b_graph[:, :copy_h, copy_t:] = 0.01
                        if max_h_nodes > copy_h:
                            new_b_graph[:, copy_h:, copy_t:] = 0.01

                    b_graph = new_b_graph

            # 数值稳定性处理
            b_graph = torch.clamp(b_graph, min=-10, max=10)
            return b_graph

        except Exception as e:
            logger.error(f"Error adjusting bipartite graph: {e}")
            return torch.zeros(batch_size, max_h_nodes, max_t_nodes, device=b_graph.device)


class IntraGraphAttention(nn.Module):
    """增强的图内注意力，支持多头注意力和残差连接"""

    def __init__(self, in_dim: int, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.heads = heads
        self.dim_per_head = in_dim // heads
        self.in_dim = in_dim

        # 多头注意力投影
        self.qkv_proj = nn.Linear(in_dim, 3 * in_dim)
        self.out_proj = nn.Linear(in_dim, in_dim)

        # 归一化和dropout
        self.norm = RobustLayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout)

        # 缩放因子
        self.scale = 1.0 / (self.dim_per_head ** 0.5)

    def forward(self, data: Data) -> torch.Tensor:
        """
        前向传播，处理图内自注意力

        Args:
            data: 图数据对象，包含节点特征和边信息
        """

        x = data.x
        batch = data.batch

        if x.size(0) == 0:
            return x
        if torch.max(torch.bincount(batch)) == 1:
            # 单节点图不应返回全零；保留原始特征并做轻量归一化，避免信息塌缩。
            return self.norm(x, batch)

            # 将批次中的图数据转换为密集张量
        x_dense, mask = to_dense_batch(x, batch)

        try:
            # 投影到查询、键、值
            qkv = self.qkv_proj(x)  # [num_nodes, 3 * in_dim]
            qkv = qkv.view(x.size(0), 3, self.heads, self.dim_per_head)  # [num_nodes, 3, heads, dim_per_head]
            q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # 每个都是 [num_nodes, heads, dim_per_head]

            # 计算注意力分数
            attn_scores = (q * k).sum(dim=-1) * self.scale  # [num_nodes, heads]

            # 处理不同batch的情况
            if batch is not None:
                # 按batch分组计算注意力
                unique_batches = torch.unique(batch)
                attn_weights = torch.zeros_like(attn_scores)

                for b in unique_batches:
                    batch_mask = batch == b
                    batch_nodes = batch_mask.sum()

                    if batch_nodes == 1:
                        # 单节点情况，注意力权重为1
                        attn_weights[batch_mask] = 1.0
                    else:
                        # 多节点情况，计算softmax
                        batch_scores = attn_scores[batch_mask]
                        batch_weights = torch.softmax(batch_scores, dim=0)
                        attn_weights[batch_mask] = batch_weights
            else:
                # 全局softmax
                if x.size(0) == 1:
                    attn_weights = torch.ones_like(attn_scores)
                else:
                    attn_weights = torch.softmax(attn_scores, dim=0)

            # 应用注意力权重
            attn_weights = self.dropout(attn_weights)
            attn_weights = attn_weights.unsqueeze(-1)  # [num_nodes, heads, 1]

            # 加权值向量
            weighted_v = attn_weights * v  # [num_nodes, heads, dim_per_head]

            # 重塑并投影输出
            out = weighted_v.reshape(x.size(0), -1)  # [num_nodes, in_dim]
            out = self.out_proj(out)
            out = self.dropout(out)

            # 残差连接和归一化
            output = self.norm(out + x, batch)

            return output

        except Exception as e:
            logger.error(f"Error in IntraGraphAttention: {e}")
            logger.error(f"Input shape: {x.shape}")
            raise e


class BaseGNNLayer(nn.Module):
    """基础GNN层，支持多层GAT和残差连接"""

    def __init__(self, in_feats: int, out_feats: int, num_layers: int = 2,
                 activation: nn.Module = nn.ReLU(), dropout: float = 0.3,
                 heads: int = 4, norm_type: str = 'layer', use_residual: bool = True):
        super().__init__()
        self.convs = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = dropout
        self.heads = heads
        self.use_residual = use_residual
        self.concat = True

        for i in range(num_layers):
            input_dim = in_feats if i == 0 else out_feats

            # GAT卷积层
            self.convs.append(GATConv(input_dim, out_feats // heads,
                                      heads=heads, concat=self.concat, dropout=dropout))
            self.acts.append(activation)

            # 归一化层
            if norm_type == 'batch':
                self.norms.append(nn.BatchNorm1d(out_feats))
            else:
                self.norms.append(RobustLayerNorm(out_feats))

        # 残差连接的投影层（当输入输出维度不同时）
        if self.use_residual and in_feats != out_feats:
            self.residual_proj = nn.Linear(in_feats, out_feats)
        else:
            self.residual_proj = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 节点特征 [num_nodes, in_feats]
            edge_index: 边索引 [2, num_edges]
            batch: 批次索引 [num_nodes]
        """
        input_x = x

        for i, (conv, act, norm) in enumerate(zip(self.convs, self.acts, self.norms)):
            # 保存残差
            if self.use_residual:
                if i == 0 and self.residual_proj is not None:
                    residual = self.residual_proj(input_x)
                else:
                    residual = x if x.shape == input_x.shape else input_x

            # 卷积操作
            x = conv(x, edge_index)

            # 激活函数
            x = act(x)

            # 归一化
            if isinstance(norm, RobustLayerNorm):
                x = norm(x, batch)
            else:
                x = norm(x)

            # Dropout
            x = F.dropout(x, p=self.dropout, training=self.training)

            # 残差连接
            if self.use_residual and residual.shape == x.shape:
                x = x + residual

        return x


class AtomGNNLayer(BaseGNNLayer):
    """原子级GNN层，专门处理原子特征"""

    def __init__(self, in_feats: int, out_feats: int, num_layers: int = 2):
        super().__init__(in_feats, out_feats, num_layers=num_layers,
                         norm_type='layer', heads=8, dropout=0.2)


class SubstructureGNNLayer(BaseGNNLayer):
    """子结构级GNN层，专门处理子结构特征"""

    def __init__(self, in_feats: int, out_feats: int, num_layers: int = 2):
        super().__init__(in_feats, out_feats, num_layers=num_layers,
                         norm_type='layer', heads=8, dropout=0.2)


def substructure_to_atom_aggregation(atom_num: int, sub_feats: torch.Tensor,
                                     sub_nodes: Union[List[torch.Tensor], torch.Tensor, None]) -> torch.Tensor:
    """
    将子结构特征聚合到原子级别，支持多种输入格式

    Args:
        atom_num: 原子数量
        sub_feats: 子结构特征 [num_substructs, feat_dim]
        sub_nodes: 子结构到原子的映射关系
    """
    device = sub_feats.device
    feat_dim = sub_feats.size(1)

    if sub_nodes is None:
        return torch.zeros(atom_num, feat_dim, device=device)

    try:
        if isinstance(sub_nodes, list):
            if len(sub_nodes) == 0:
                return torch.zeros(atom_num, feat_dim, device=device)

            # 处理列表格式的映射
            atom_indices, sub_indices = [], []
            for sub_idx, nodes in enumerate(sub_nodes):
                if isinstance(nodes, torch.Tensor) and nodes.numel() > 0:
                    atom_indices.append(nodes)
                    sub_indices.append(torch.full_like(nodes, sub_idx, dtype=torch.long))

            if not atom_indices:
                return torch.zeros(atom_num, feat_dim, device=device)

            atom_indices = torch.cat(atom_indices, dim=0)
            sub_indices = torch.cat(sub_indices, dim=0)

            # 使用 scatter 聚合；在缺少 torch_scatter 时退化为 index_add。
            if torch_scatter is not None:
                agg = torch_scatter.scatter_add(sub_feats[sub_indices], atom_indices, dim=0, dim_size=atom_num)
                counts = torch_scatter.scatter_add(
                    torch.ones_like(atom_indices, dtype=sub_feats.dtype),
                    atom_indices,
                    dim=0,
                    dim_size=atom_num,
                ).clamp(min=1).unsqueeze(1)
            else:
                agg = torch.zeros(atom_num, feat_dim, device=device, dtype=sub_feats.dtype)
                agg.index_add_(0, atom_indices, sub_feats[sub_indices])
                counts = torch.zeros(atom_num, device=device, dtype=sub_feats.dtype)
                counts.index_add_(0, atom_indices, torch.ones_like(atom_indices, dtype=sub_feats.dtype))
                counts = counts.clamp(min=1).unsqueeze(1)
            return agg / counts

        elif isinstance(sub_nodes, torch.Tensor):
            if sub_nodes.numel() == 0:
                return torch.zeros(atom_num, feat_dim, device=device)

            if not sub_nodes.is_sparse:
                # 密集矩阵乘法 [atom_num, num_substructs] x [num_substructs, feat_dim]
                if sub_nodes.size(0) != atom_num:
                    logger.warning(f"Mapping matrix size mismatch: {sub_nodes.size(0)} vs {atom_num}")
                    # 调整映射矩阵大小
                    if sub_nodes.size(0) > atom_num:
                        sub_nodes = sub_nodes[:atom_num, :]
                    else:
                        # 扩展映射矩阵
                        pad_size = atom_num - sub_nodes.size(0)
                        padding = torch.zeros(pad_size, sub_nodes.size(1),
                                              device=device, dtype=sub_nodes.dtype)
                        sub_nodes = torch.cat([sub_nodes, padding], dim=0)

                agg = torch.mm(sub_nodes, sub_feats)
                counts = sub_nodes.sum(dim=1, keepdim=True).clamp(min=1e-6)
                return agg / counts
            else:
                # 稀疏矩阵乘法
                return torch.sparse.mm(sub_nodes, sub_feats)
        else:
            raise ValueError(f"Unsupported type for sub_nodes: {type(sub_nodes)}")

    except Exception as e:
        logger.error(f"Error in substructure aggregation: {e}")
        return torch.zeros(atom_num, feat_dim, device=device)


class DualGranularityFusion(nn.Module):
    """
    双粒度特征融合模块，整合原子级和子结构级特征
    实现research paper中的双粒度融合机制
    """

    def __init__(self, atom_dim: int, sub_dim: int, hidden_dim: int,
                 fusion_type: str = 'gated', dropout: float = 0.3):
        super().__init__()
        self.atom_dim = atom_dim
        self.sub_dim = sub_dim
        self.hidden_dim = hidden_dim
        self.fusion_type = fusion_type

        # 特征投影层
        self.atom_proj = nn.Sequential(
            nn.Linear(atom_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.sub_proj = nn.Sequential(
            nn.Linear(sub_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 融合机制
        if fusion_type == 'gated':
            # 门控融合
            self.gate = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.Sigmoid()
            )
        elif fusion_type == 'attention':
            # 注意力融合
            self.attention = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 2),
                nn.Softmax(dim=-1)
            )
        elif fusion_type == 'concat':
            # 拼接融合
            self.fusion_proj = nn.Linear(2 * hidden_dim, hidden_dim)

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, atom_dim),
            nn.Dropout(dropout)
        )

        # 归一化
        self.norm = RobustLayerNorm(atom_dim)

    def forward(self, atom_feats: torch.Tensor, sub_feats: torch.Tensor,
                sub_nodes: Union[torch.Tensor, List, None]) -> torch.Tensor:
        """
        前向传播，融合双粒度特征

        Args:
            atom_feats: 原子级特征 [num_atoms, atom_dim]
            sub_feats: 子结构级特征 [num_substructs, sub_dim]
            sub_nodes: 子结构到原子的映射关系
        """
        try:
            # 聚合子结构特征到原子级别
            agg_sub = substructure_to_atom_aggregation(atom_feats.size(0), sub_feats, sub_nodes)

            # 特征投影
            atom_trans = self.atom_proj(atom_feats)  # [num_atoms, hidden_dim]
            sub_trans = self.sub_proj(agg_sub)  # [num_atoms, hidden_dim]

            # 双粒度特征融合
            if self.fusion_type == 'gated':
                # 门控融合
                combined = torch.cat([atom_trans, sub_trans], dim=-1)  # [num_atoms, 2*hidden_dim]
                gate = self.gate(combined)  # [num_atoms, hidden_dim]
                fused = gate * atom_trans + (1 - gate) * sub_trans

            elif self.fusion_type == 'attention':
                # 注意力融合
                combined = torch.cat([atom_trans, sub_trans], dim=-1)
                attn_weights = self.attention(combined)  # [num_atoms, 2]

                # 加权组合
                stacked_feats = torch.stack([atom_trans, sub_trans], dim=-1)  # [num_atoms, hidden_dim, 2]
                fused = (stacked_feats * attn_weights.unsqueeze(1)).sum(dim=-1)  # [num_atoms, hidden_dim]

            elif self.fusion_type == 'concat':
                # 拼接融合
                combined = torch.cat([atom_trans, sub_trans], dim=-1)
                fused = self.fusion_proj(combined)

            else:
                # 默认加权融合
                fused = 0.6 * atom_trans + 0.4 * sub_trans

            # 输出投影
            output = self.output_proj(fused)

            # 残差连接和归一化
            final_output = self.norm(atom_feats + F.dropout(output, p=0.2, training=self.training))

            return final_output

        except Exception as e:
            logger.error(f"Error in DualGranularityFusion: {e}")
            logger.error(f"atom_feats shape: {atom_feats.shape}, sub_feats shape: {sub_feats.shape}")
            raise e


class CoAttentionLayer(nn.Module):
    """
    协同注意力层，用于药物对的交互建模
    实现头尾实体间的双向注意力机制
    """

    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.dim_per_head = dim // n_heads

        # 多头注意力投影
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        # 归一化和dropout
        self.norm = RobustLayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

        # 缩放因子
        self.scale = 1.0 / (self.dim_per_head ** 0.5)

    def forward(self, head: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        """
        前向传播，计算协同注意力

        Args:
            head: 头实体表示 [batch_size, seq_len, dim]
            tail: 尾实体表示 [batch_size, seq_len, dim]
        """
        try:
            batch_size = head.size(0)

            # 多头投影
            q = self.q_proj(head).view(batch_size, -1, self.n_heads, self.dim_per_head).transpose(1, 2)
            k = self.k_proj(tail).view(batch_size, -1, self.n_heads, self.dim_per_head).transpose(1, 2)
            v = self.v_proj(tail).view(batch_size, -1, self.n_heads, self.dim_per_head).transpose(1, 2)

            # 计算注意力分数
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

            # 处理单样本情况
            if batch_size == 1 or head.size(1) == 1:
                attn = torch.ones_like(scores)
            else:
                attn = self.softmax(scores)

            # Dropout
            attn = self.dropout(attn)

            # 应用注意力
            context = torch.matmul(attn, v)  # [batch_size, n_heads, seq_len, dim_per_head]

            # 重塑输出
            context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
            output = self.out_proj(context)

            # 残差连接和归一化
            final_output = self.norm(output + head)

            return final_output

        except Exception as e:
            logger.error(f"Error in CoAttentionLayer: {e}")
            logger.error(f"head shape: {head.shape}, tail shape: {tail.shape}")
            raise e


class RESCAL(nn.Module):
    """
    RESCAL知识图嵌入模型，用于药物-药物相互作用预测
    实现双线性张量分解进行关系建模 (优化版本)
    """

    def __init__(self, num_relations: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.num_relations = num_relations
        self.embed_dim = embed_dim

        # 关系矩阵参数 [num_relations, embed_dim, embed_dim]
        self.relation_embed = nn.Parameter(torch.randn(num_relations, embed_dim, embed_dim))

        # gain=0.1：缩小初始化幅度，避免 h^T R t 在早期产生过大分数导致 NaN
        nn.init.xavier_uniform_(self.relation_embed, gain=0.1)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # --- 移除了 self.score_proj 线性层 ---

    def forward(self, heads: torch.Tensor, tails: torch.Tensor,
                rels: torch.Tensor, attn: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播，计算三元组得分

        Args:
            heads: 头实体表示 [batch_size, seq_len, embed_dim] or [batch_size, embed_dim]
            tails: 尾实体表示 [batch_size, seq_len, embed_dim] or [batch_size, embed_dim]
            rels: 关系索引 [batch_size]
            attn: 注意力权重 (可选)
        """
        try:
            # 处理多种输入格式
            if len(heads.shape) == 3:  # [batch_size, seq_len, embed_dim]
                if attn is not None:
                    # 使用注意力权重进行池化
                    heads = (attn.unsqueeze(-1) * heads).sum(dim=1)  # [batch_size, embed_dim]
                    tails = (attn.unsqueeze(-1) * tails).sum(dim=1)  # [batch_size, embed_dim]
                else:
                    # 平均池化
                    heads = heads.mean(dim=1)  # [batch_size, embed_dim]
                    tails = tails.mean(dim=1)  # [batch_size, embed_dim]

            # 确保维度正确
            if len(heads.shape) != 2 or heads.size(1) != self.embed_dim:
                raise ValueError(f"Invalid heads shape: {heads.shape}, expected [batch_size, {self.embed_dim}]")
            if len(tails.shape) != 2 or tails.size(1) != self.embed_dim:
                raise ValueError(f"Invalid tails shape: {tails.shape}, expected [batch_size, {self.embed_dim}]")

            # 获取关系矩阵
            rel_mat = self.relation_embed[rels]  # [batch_size, embed_dim, embed_dim]

            # 应用dropout
            heads = self.dropout(heads)
            tails = self.dropout(tails)

            # RESCAL评分计算: h^T * R_r * t
            h_expanded = heads.unsqueeze(1)  # [batch_size, 1, embed_dim]
            t_expanded = tails.unsqueeze(-1)  # [batch_size, embed_dim, 1]

            # h * R_r
            hr = torch.bmm(h_expanded, rel_mat)  # [batch_size, 1, embed_dim]

            # (h * R_r) * t
            score = torch.bmm(hr, t_expanded).squeeze()  # [batch_size]

            # 确保输出是1维的，以处理 batch_size=1 的情况
            if score.dim() == 0:
                score = score.unsqueeze(0)

            # --- 移除了可选的分数投影 score_proj ---

            return score

        except Exception as e:
            logger.error(f"Error in RESCAL: {e}", exc_info=True)
            logger.error(f"heads shape: {heads.shape}, tails shape: {tails.shape}, rels shape: {rels.shape}")
            # 抛出异常以便上层捕获
            raise e