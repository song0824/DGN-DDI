import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_add_pool, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import unbatch
from typing import Tuple, Dict, List, Optional
import logging

from layers import (
    AtomGNNLayer,
    SubstructureGNNLayer,
    DualGranularityFusion,
    CoAttentionLayer,
    RESCAL,
    IntraGraphAttention,
    InterGraphAttention,
    RobustLayerNorm
)

logger = logging.getLogger(__name__)


class DGN_DDI_Block(nn.Module):
    def __init__(self, n_heads: int, head_out_feats: int, hidden_dim: int):
        super().__init__()
        self.intra_att = IntraGraphAttention(n_heads * head_out_feats)
        self.inter_att = InterGraphAttention(n_heads * head_out_feats)
        self.pool = GATConv(2 * n_heads * head_out_feats, head_out_feats, heads=n_heads, concat=True)
        self.norm = RobustLayerNorm(2 * n_heads * head_out_feats)

        self.dual_fusion = DualGranularityFusion(
            atom_dim=2 * n_heads * head_out_feats,
            sub_dim=2 * n_heads * head_out_feats,
            hidden_dim=hidden_dim
        )

    @staticmethod
    def _cat_feats(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if a.size(0) == b.size(0):
            return torch.cat([a, b], dim=-1)
        n = min(a.size(0), b.size(0))
        return torch.cat([a[:n], b[:n]], dim=-1)

    def forward_batch(
        self,
        h_list: List[Dict],
        t_list: List[Dict],
        b_graph: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """批量处理 B 个药物对（一次 intra/inter 注意力），替代逐对 for 循环。"""
        bsz = len(h_list)
        if bsz == 0:
            raise ValueError("forward_batch: empty pair list")
        if bsz == 1:
            return self.forward(h_list[0], t_list[0], b_graph or {})

        h_atom_b = Batch.from_data_list([d["atom"] for d in h_list])
        t_atom_b = Batch.from_data_list([d["atom"] for d in t_list])
        h_sub_b = Batch.from_data_list([d["substruct"] for d in h_list])
        t_sub_b = Batch.from_data_list([d["substruct"] for d in t_list])

        h_intra_a = self.intra_att(h_atom_b)
        t_intra_a = self.intra_att(t_atom_b)
        h_intra_s = self.intra_att(h_sub_b)
        t_intra_s = self.intra_att(t_sub_b)

        fused_bg = None
        sub_bg = None
        if b_graph:
            fused_bg = b_graph.get("fused")
            if fused_bg is None:
                fused_bg = b_graph.get("atom")
            sub_bg = b_graph.get("substruct")

        h_inter_a, t_inter_a = self.inter_att(h_atom_b, t_atom_b, fused_bg)
        h_inter_s, t_inter_s = self.inter_att(h_sub_b, t_sub_b, sub_bg)

        h_ia = unbatch(h_intra_a, h_atom_b.batch)
        t_ia = unbatch(t_intra_a, t_atom_b.batch)
        h_is = unbatch(h_intra_s, h_sub_b.batch)
        t_is = unbatch(t_intra_s, t_sub_b.batch)
        h_ira = unbatch(h_inter_a, h_atom_b.batch)
        t_ira = unbatch(t_inter_a, t_atom_b.batch)
        h_irs = unbatch(h_inter_s, h_sub_b.batch)
        t_irs = unbatch(t_inter_s, t_sub_b.batch)

        h_pool_graphs, t_pool_graphs = [], []
        h_out_list, t_out_list = [], []

        for i in range(bsz):
            h_ff = self.dual_fusion(
                self._cat_feats(h_ia[i], h_ira[i]),
                self._cat_feats(h_is[i], h_irs[i]),
                h_list[i]["atom2substruct"],
            )
            t_ff = self.dual_fusion(
                self._cat_feats(t_ia[i], t_ira[i]),
                self._cat_feats(t_is[i], t_irs[i]),
                t_list[i]["atom2substruct"],
            )
            h_bv = torch.zeros(h_ff.size(0), dtype=torch.long, device=h_ff.device)
            t_bv = torch.zeros(t_ff.size(0), dtype=torch.long, device=t_ff.device)
            h_pool_graphs.append(Data(x=F.elu(self.norm(h_ff, h_bv)), edge_index=h_list[i]["atom"].edge_index))
            t_pool_graphs.append(Data(x=F.elu(self.norm(t_ff, t_bv)), edge_index=t_list[i]["atom"].edge_index))

        h_pool_b = Batch.from_data_list(h_pool_graphs)
        t_pool_b = Batch.from_data_list(t_pool_graphs)
        h_x, h_w = self.pool(h_pool_b.x, h_pool_b.edge_index, return_attention_weights=True)
        t_x, t_w = self.pool(t_pool_b.x, t_pool_b.edge_index, return_attention_weights=True)

        h_xs = unbatch(h_x, h_pool_b.batch)
        t_xs = unbatch(t_x, t_pool_b.batch)
        h_global = global_mean_pool(h_x, h_pool_b.batch)
        t_global = global_mean_pool(t_x, t_pool_b.batch)

        for i in range(bsz):
            n_a = h_xs[i].size(0)
            n_t = t_xs[i].size(0)
            h_out_list.append({
                "atom": Data(
                    x=h_xs[i],
                    edge_index=h_list[i]["atom"].edge_index,
                    batch=torch.zeros(n_a, dtype=torch.long, device=h_x.device),
                    drug_id=h_list[i]["atom"].drug_id if hasattr(h_list[i]["atom"], "drug_id") else None,
                ),
                "substruct": h_list[i]["substruct"],
                "atom2substruct": h_list[i]["atom2substruct"],
            })
            t_out_list.append({
                "atom": Data(
                    x=t_xs[i],
                    edge_index=t_list[i]["atom"].edge_index,
                    batch=torch.zeros(n_t, dtype=torch.long, device=t_x.device),
                    drug_id=t_list[i]["atom"].drug_id if hasattr(t_list[i]["atom"], "drug_id") else None,
                ),
                "substruct": t_list[i]["substruct"],
                "atom2substruct": t_list[i]["atom2substruct"],
            })

        return h_out_list, t_out_list, h_global, t_global, h_w, t_w

    @staticmethod
    def _masked_node_attn(attn, src_mask, dst_mask) -> Optional[torch.Tensor]:
        """将 dense 图间注意力压成源节点重要性 [n_src]。"""
        if attn is None:
            return None
        a = attn[0] if attn.dim() == 3 else attn
        if src_mask is not None and dst_mask is not None:
            sm = src_mask[0] if src_mask.dim() == 2 else src_mask
            dm = dst_mask[0] if dst_mask.dim() == 2 else dst_mask
            a = a[sm][:, dm]
        row = a.sum(dim=-1)
        total = row.sum()
        if float(total) > 0:
            row = row / total
        return row.detach()

    @staticmethod
    def _pool_to_node(weight_tuple, num_nodes: int, device) -> torch.Tensor:
        """GAT return_attention_weights -> 按入边聚合的节点权重。"""
        node = torch.zeros(num_nodes, device=device)
        if not weight_tuple:
            return node
        edge_index, alpha = weight_tuple
        if edge_index is None or alpha is None or alpha.numel() == 0:
            return node
        if alpha.dim() > 1:
            alpha = alpha.mean(dim=-1)
        dst = edge_index[1]
        valid = (dst >= 0) & (dst < num_nodes) & (torch.arange(dst.size(0), device=dst.device) < alpha.size(0))
        if alpha.size(0) != dst.size(0):
            n = min(int(alpha.size(0)), int(dst.size(0)))
            dst = dst[:n]
            alpha = alpha[:n]
            valid = (dst >= 0) & (dst < num_nodes)
        if valid.sum() == 0:
            return node
        node = node.to(dtype=alpha.dtype)
        node.index_add_(0, dst[valid], alpha[valid])
        total = node.sum()
        if float(total) > 0:
            node = node / total
        return node.detach()

    def forward(self, h_data: Dict[str, Data], t_data: Dict[str, Data],
                b_graph: Optional[Dict[str, torch.Tensor]] = None,
                return_explain: bool = False):

        device = h_data['atom'].x.device

        try:
            if return_explain:
                h_intra_atom, h_intra_w_a = self.intra_att(h_data['atom'], return_attn=True)
                t_intra_atom, t_intra_w_a = self.intra_att(t_data['atom'], return_attn=True)
                h_intra_sub, h_intra_w_s = self.intra_att(h_data['substruct'], return_attn=True)
                t_intra_sub, t_intra_w_s = self.intra_att(t_data['substruct'], return_attn=True)
            else:
                h_intra_atom = self.intra_att(h_data['atom'])
                t_intra_atom = self.intra_att(t_data['atom'])
                h_intra_sub = self.intra_att(h_data['substruct'])
                t_intra_sub = self.intra_att(t_data['substruct'])
                h_intra_w_a = t_intra_w_a = h_intra_w_s = t_intra_w_s = None

            fused_b_graph = None
            substruct_b_graph = None
            if b_graph:
                fused_b_graph = b_graph.get('fused', b_graph.get('atom', None))
                substruct_b_graph = b_graph.get('substruct', None)

            if return_explain:
                h_inter_atom, t_inter_atom, attn_ht_a, attn_th_a, h_amask, t_amask = self.inter_att(
                    h_data['atom'], t_data['atom'], fused_b_graph, return_attn=True
                )
                h_inter_sub, t_inter_sub, attn_ht_s, attn_th_s, h_smask, t_smask = self.inter_att(
                    h_data['substruct'], t_data['substruct'], substruct_b_graph, return_attn=True
                )
            else:
                h_inter_atom, t_inter_atom = self.inter_att(h_data['atom'], t_data['atom'], fused_b_graph)
                h_inter_sub, t_inter_sub = self.inter_att(h_data['substruct'], t_data['substruct'], substruct_b_graph)
                attn_ht_a = attn_th_a = attn_ht_s = attn_th_s = None
                h_amask = t_amask = h_smask = t_smask = None

            h_fused_atom = self._cat_feats(h_intra_atom, h_inter_atom)
            t_fused_atom = self._cat_feats(t_intra_atom, t_inter_atom)
            h_fused_sub = self._cat_feats(h_intra_sub, h_inter_sub)
            t_fused_sub = self._cat_feats(t_intra_sub, t_inter_sub)

            h_final_feats = self.dual_fusion(h_fused_atom, h_fused_sub, h_data['atom2substruct'])
            t_final_feats = self.dual_fusion(t_fused_atom, t_fused_sub, t_data['atom2substruct'])

            h_norm = F.elu(self.norm(h_final_feats, h_data['atom'].batch))
            t_norm = F.elu(self.norm(t_final_feats, t_data['atom'].batch))

            h_x, h_weight = self.pool(h_norm, h_data['atom'].edge_index, return_attention_weights=True)
            t_x, t_weight = self.pool(t_norm, t_data['atom'].edge_index, return_attention_weights=True)

            h_data_updated = {
                'atom': Data(
                    x=h_x,
                    edge_index=h_data['atom'].edge_index,
                    batch=h_data['atom'].batch,
                    drug_id=h_data['atom'].drug_id if hasattr(h_data['atom'], 'drug_id') else None
                ),
                'substruct': h_data['substruct'],
                'atom2substruct': h_data['atom2substruct']
            }

            t_data_updated = {
                'atom': Data(
                    x=t_x,
                    edge_index=t_data['atom'].edge_index,
                    batch=t_data['atom'].batch,
                    drug_id=t_data['atom'].drug_id if hasattr(t_data['atom'], 'drug_id') else None
                ),
                'substruct': t_data['substruct'],
                'atom2substruct': t_data['atom2substruct']
            }

            h_global = global_mean_pool(h_x, h_data['atom'].batch)
            t_global = global_mean_pool(t_x, t_data['atom'].batch)

            if return_explain:
                explain = {
                    "intra_h_atom": h_intra_w_a,
                    "intra_t_atom": t_intra_w_a,
                    "intra_h_sub": h_intra_w_s,
                    "intra_t_sub": t_intra_w_s,
                    "inter_h_atom": self._masked_node_attn(attn_ht_a, h_amask, t_amask),
                    "inter_t_atom": self._masked_node_attn(attn_th_a, t_amask, h_amask),
                    "inter_h_sub": self._masked_node_attn(attn_ht_s, h_smask, t_smask),
                    "inter_t_sub": self._masked_node_attn(attn_th_s, t_smask, h_smask),
                    "pool_h_atom": self._pool_to_node(h_weight, h_x.size(0), h_x.device),
                    "pool_t_atom": self._pool_to_node(t_weight, t_x.size(0), t_x.device),
                }
                return h_data_updated, t_data_updated, h_global, t_global, h_weight, t_weight, explain

            return h_data_updated, t_data_updated, h_global, t_global, h_weight, t_weight

        except Exception as e:
            logger.error(f"Error in DGN_DDI_Block forward: {e}")
            raise e


class DGN_DDI(nn.Module):
    def __init__(self, hidden_dim: int, kge_dim: int, rel_total: int,
                 heads_out_feat_params: List[int], blocks_params: List[int],
                 drug_graph_dict, int_to_drug_id, device):
        super().__init__()

        # 设置特征维度（匹配data_preprocessing.py中的特征）
        atom_in_dim = 66  # 原子级特征维度
        sub_in_dim = 86  # 子结构级特征维度 (66原子 + 4拓扑 + 16功能基团)

        # 初始化各层
        self.atom_gnn = AtomGNNLayer(atom_in_dim, hidden_dim)
        self.sub_gnn = SubstructureGNNLayer(sub_in_dim, hidden_dim)
        self.fusion = DualGranularityFusion(hidden_dim, hidden_dim, hidden_dim)

        # 使用鲁棒的LayerNorm
        self.initial_norm = RobustLayerNorm(hidden_dim)
        self.initial_conv = GATConv(hidden_dim, heads_out_feat_params[0],
                                    heads=blocks_params[0], concat=True)

        self.sub_initial_proj = nn.Linear(hidden_dim, blocks_params[0] * heads_out_feat_params[0])

        # DGN-DDI块
        self.blocks = nn.ModuleList([
            DGN_DDI_Block(blocks_params[i], heads_out_feat_params[i], hidden_dim)
            for i in range(len(blocks_params))
        ])

        # 注意力和知识图嵌入
        self.co_attention = CoAttentionLayer(kge_dim)
        self.KGE = RESCAL(rel_total, kge_dim)

        # 可学习的 block 聚合权重：深层 block 表示通常更精炼，允许模型自动调权。
        self.block_weights = nn.Parameter(torch.ones(len(blocks_params)) / len(blocks_params))

        # 训练目标默认假设”正样本分数更高”；避免符号翻转引入额外不确定性。
        self.score_higher_is_better = True
        # 可学习的 logit 缩放因子，初始值1.0；优化器会根据梯度自动调整。
        self.logit_scale = nn.Parameter(torch.tensor(1.0))
        self.ablation_mode = None

        # 存储数据相关信息
        self.drug_graph_dict = drug_graph_dict
        self.int_to_drug_id = int_to_drug_id
        self.device = device

        # 训练加速：子结构/映射常驻 GPU；步内药物编码 LRU（单次 forward 内复用）
        self._gpu_drug_static: Dict[str, Dict] = {}
        self._encode_step_cache: Dict[str, Dict] = {}
        self._preload_drug_static_to_device()

    def set_ablation_mode(self, mode: Optional[str] = None) -> None:
        """推理消融开关。None/full 为完整模型，不改变已加载权重。

        atom_only 是 no_fusion 的别名，不要当成两个独立实验。
        """
        if mode in (None, "", "full"):
            mode = None
        if mode == "atom_only":
            mode = "no_fusion"
        allowed = {None, "no_fusion", "no_inter", "substruct_only"}
        if mode not in allowed:
            raise ValueError(
                f"Unknown ablation mode {mode!r}, expected one of "
                "full/no_fusion/no_inter/atom_only/substruct_only"
            )
        self.ablation_mode = mode
        for module in self.modules():
            if hasattr(module, "ablation_mode"):
                module.ablation_mode = mode
        logger.info("Ablation mode set to %s", mode or "full")

    def _preload_drug_static_to_device(self) -> None:
        """将子结构图与 atom2substruct 预载到 device，避免每次 forward 重复 .to()。"""
        n_drugs = sum(
            1 for k, v in self.drug_graph_dict.items()
            if k != "__metadata__" and isinstance(v, dict)
        )
        logger.info("Preloading static tensors for %d drugs to %s...", n_drugs, self.device)
        for drug_id, data in self.drug_graph_dict.items():
            if drug_id == "__metadata__" or not isinstance(data, dict):
                continue
            if "substruct" not in data or "atom2substruct" not in data:
                continue
            self._gpu_drug_static[drug_id] = {
                "substruct": data["substruct"].to(self.device),
                "atom2substruct": data["atom2substruct"].to(self.device),
            }
        logger.info("Preloaded static drug tensors for %d drugs on %s", len(self._gpu_drug_static), self.device)

    def _copy_encoded_drug(self, drug_data: Dict) -> Dict:
        """浅拷贝编码结果供不同药物对并行使用（block 会替换 Data 对象，不原地改 x）。"""
        atom = drug_data["atom"]
        sub = drug_data["substruct"]
        return {
            "atom": Data(
                x=atom.x,
                edge_index=atom.edge_index,
                batch=atom.batch.clone(),
                drug_id=atom.drug_id if hasattr(atom, "drug_id") else None,
            ),
            "substruct": Data(
                x=sub.x,
                edge_index=sub.edge_index,
                batch=sub.batch.clone(),
                drug_id=sub.drug_id if hasattr(sub, "drug_id") else None,
            ),
            "atom2substruct": drug_data["atom2substruct"],
        }

    def _resolve_drug_id(self, atom_data) -> Optional[str]:
        if not hasattr(atom_data, "drug_id"):
            return None
        drug_id_tensor = atom_data.drug_id
        if isinstance(drug_id_tensor, torch.Tensor):
            drug_id_int = drug_id_tensor.item()
            return self.int_to_drug_id.get(drug_id_int)
        return drug_id_tensor

    def _encode_one_drug(self, atom_data, substruct_data=None) -> Optional[Dict]:
        """对单个药物执行 GNN 初始编码（热路径，跳过冗余 nan 检查）。"""
        drug_id = self._resolve_drug_id(atom_data)
        if drug_id is None:
            return None

        atom_on_device = atom_data.to(self.device)
        static = self._gpu_drug_static.get(drug_id)
        if static is not None:
            drug_data = {
                "atom": atom_on_device,
                "substruct": static["substruct"],
                "atom2substruct": static["atom2substruct"],
            }
        elif drug_id in self.drug_graph_dict:
            stored = self.drug_graph_dict[drug_id]
            drug_data = {
                "atom": atom_on_device,
                "substruct": stored["substruct"].to(self.device),
                "atom2substruct": stored["atom2substruct"].to(self.device),
            }
        elif substruct_data is not None:
            sub_on_device = substruct_data.to(self.device)
            drug_data = {
                "atom": atom_on_device,
                "substruct": sub_on_device,
                "atom2substruct": torch.eye(
                    atom_on_device.x.size(0), sub_on_device.x.size(0), device=self.device
                ),
            }
        else:
            return None

        atom_repr = self.atom_gnn(drug_data["atom"].x, drug_data["atom"].edge_index, drug_data["atom"].batch)
        sub_repr = self.sub_gnn(
            drug_data["substruct"].x, drug_data["substruct"].edge_index, drug_data["substruct"].batch
        )
        fused_repr = self.fusion(atom_repr, sub_repr, drug_data["atom2substruct"])
        fused_norm = self.initial_norm(fused_repr, drug_data["atom"].batch)
        atom_new_x = self.initial_conv(fused_norm, drug_data["atom"].edge_index)
        sub_new_x = self.sub_initial_proj(sub_repr)

        num_atom_nodes = atom_new_x.size(0)
        num_sub_nodes = sub_new_x.size(0)
        return {
            "atom": Data(
                x=atom_new_x,
                edge_index=drug_data["atom"].edge_index,
                batch=torch.zeros(num_atom_nodes, dtype=torch.long, device=self.device),
                drug_id=drug_data["atom"].drug_id if hasattr(drug_data["atom"], "drug_id") else None,
            ),
            "substruct": Data(
                x=sub_new_x,
                edge_index=drug_data["substruct"].edge_index,
                batch=torch.zeros(num_sub_nodes, dtype=torch.long, device=self.device),
                drug_id=drug_data["substruct"].drug_id if hasattr(drug_data["substruct"], "drug_id") else None,
            ),
            "atom2substruct": drug_data["atom2substruct"],
        }

    def _safe_tensor_operation(self, tensor, operation_name):
        """安全的张量操作，添加数值稳定性检查"""
        if tensor is None:
            return tensor

        # 更严格的数值处理
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1e6, neginf=-1e6)

        if torch.isnan(tensor).any():
            logger.warning(f"NaN detected in {operation_name}, replaced with zeros")
            tensor = torch.nan_to_num(tensor, nan=0.0)

        if torch.isinf(tensor).any():
            logger.warning(f"Inf detected in {operation_name}, clamped to safe range")
            tensor = torch.clamp(tensor, min=-1e6, max=1e6)

        return tensor

    def _encode_drug_from_batch(self, drug_batch, additional_data=None):
        """从 Batch 编码药物；未缓存药物用 batched GNN 一次前向。"""
        if drug_batch is None:
            return []

        data_list = drug_batch.to_data_list()
        sub_list = additional_data.to_data_list() if additional_data is not None else None
        drug_ids: List[Optional[str]] = []
        uncached_idx: List[int] = []

        for idx, atom_data in enumerate(data_list):
            drug_id = self._resolve_drug_id(atom_data)
            drug_ids.append(drug_id)
            if drug_id is not None and drug_id not in self._encode_step_cache:
                uncached_idx.append(idx)

        if uncached_idx:
            self._encode_uncached_drugs_batched(uncached_idx, data_list, sub_list)

        drug_data_list = []
        for drug_id in drug_ids:
            if drug_id is None or drug_id not in self._encode_step_cache:
                continue
            drug_data_list.append(self._copy_encoded_drug(self._encode_step_cache[drug_id]))

        if not drug_data_list:
            raise ValueError("No valid drug data found in batch")
        if len(drug_data_list) != len(drug_ids):
            raise ValueError("Some drugs in batch failed to encode")
        return drug_data_list

    def _encode_uncached_drugs_batched(self, indices, data_list, sub_list=None) -> None:
        """对未命中步内缓存的药物做 batched GNN 编码。"""
        atom_graphs = []
        sub_graphs = []
        a2s_list = []
        meta = []

        for idx in indices:
            atom_data = data_list[idx]
            drug_id = self._resolve_drug_id(atom_data)
            if drug_id is None:
                continue
            atom_on = atom_data.to(self.device)
            sub_extra = sub_list[idx] if sub_list is not None and idx < len(sub_list) else None
            static = self._gpu_drug_static.get(drug_id)
            if static is not None:
                sub_g = static["substruct"]
                a2s = static["atom2substruct"]
            elif drug_id in self.drug_graph_dict:
                stored = self.drug_graph_dict[drug_id]
                sub_g = stored["substruct"].to(self.device)
                a2s = stored["atom2substruct"].to(self.device)
            elif sub_extra is not None:
                sub_g = sub_extra.to(self.device)
                a2s = torch.eye(atom_on.x.size(0), sub_g.x.size(0), device=self.device)
            else:
                continue

            atom_graphs.append(atom_on)
            sub_graphs.append(sub_g)
            a2s_list.append(a2s)
            meta.append((drug_id, atom_on, sub_g))

        if not meta:
            return

        atom_batch = Batch.from_data_list(atom_graphs)
        sub_batch = Batch.from_data_list(sub_graphs)

        atom_repr = self.atom_gnn(atom_batch.x, atom_batch.edge_index, atom_batch.batch)
        sub_repr = self.sub_gnn(sub_batch.x, sub_batch.edge_index, sub_batch.batch)
        sub_new_x_all = self.sub_initial_proj(sub_repr)

        atom_repr_split = unbatch(atom_repr, atom_batch.batch)
        sub_repr_split = unbatch(sub_repr, sub_batch.batch)
        sub_new_split = unbatch(sub_new_x_all, sub_batch.batch)

        fused_for_conv = []
        for i, (drug_id, atom_on, sub_g) in enumerate(meta):
            fused = self.fusion(atom_repr_split[i], sub_repr_split[i], a2s_list[i])
            atom_batch_vec = torch.zeros(fused.size(0), dtype=torch.long, device=self.device)
            fused_norm = self.initial_norm(fused, atom_batch_vec)
            fused_for_conv.append(
                Data(
                    x=fused_norm,
                    edge_index=atom_on.edge_index,
                    drug_id=atom_on.drug_id if hasattr(atom_on, "drug_id") else None,
                )
            )

        conv_batch = Batch.from_data_list(fused_for_conv)
        atom_new_all = self.initial_conv(conv_batch.x, conv_batch.edge_index)
        atom_new_split = unbatch(atom_new_all, conv_batch.batch)

        for i, (drug_id, atom_on, sub_g) in enumerate(meta):
            num_atom = atom_new_split[i].size(0)
            num_sub = sub_new_split[i].size(0)
            self._encode_step_cache[drug_id] = {
                "atom": Data(
                    x=atom_new_split[i],
                    edge_index=atom_on.edge_index,
                    batch=torch.zeros(num_atom, dtype=torch.long, device=self.device),
                    drug_id=atom_on.drug_id if hasattr(atom_on, "drug_id") else None,
                ),
                "substruct": Data(
                    x=sub_new_split[i],
                    edge_index=sub_g.edge_index,
                    batch=torch.zeros(num_sub, dtype=torch.long, device=self.device),
                    drug_id=sub_g.drug_id if hasattr(sub_g, "drug_id") else None,
                ),
                "atom2substruct": a2s_list[i],
            }

    def _process_blocks(self, h_data_list, t_data_list, b_graph_dict, return_weights=False):
        """处理 DGN-DDI 块；默认对整批药物对使用 forward_batch。"""
        bsz = len(h_data_list)
        if bsz == 0:
            raise ValueError("empty h_data_list in _process_blocks")

        batched_b = {}
        if b_graph_dict:
            for key, tensor in b_graph_dict.items():
                if tensor is not None and isinstance(tensor, torch.Tensor) and tensor.dim() == 3:
                    batched_b[key] = tensor

        h_list = h_data_list
        t_list = t_data_list
        repr_h_blocks, repr_t_blocks = [], []
        weight_h_list_all, weight_t_list_all = [], []

        for block_idx, block in enumerate(self.blocks):
            if bsz > 1:
                h_list, t_list, r_h, r_t, w_h, w_t = block.forward_batch(h_list, t_list, batched_b)
            else:
                current_b = {}
                if batched_b:
                    for key, tensor in batched_b.items():
                        current_b[key] = tensor[0] if tensor.dim() == 3 else tensor
                h_list[0], t_list[0], r_h, r_t, w_h, w_t = block.forward(h_list[0], t_list[0], current_b)
                if r_h.dim() == 1:
                    r_h = r_h.unsqueeze(0)
                    r_t = r_t.unsqueeze(0)

            repr_h_blocks.append(r_h)
            repr_t_blocks.append(r_t)
            if return_weights:
                weight_h_list_all.append(w_h)
                weight_t_list_all.append(w_t)

        repr_h_all = torch.stack(repr_h_blocks, dim=1)
        repr_t_all = torch.stack(repr_t_blocks, dim=1)

        if return_weights:
            return repr_h_all, repr_t_all, weight_h_list_all, weight_t_list_all
        return repr_h_all, repr_t_all, None, None

    def forward(self, h_data, t_data, rels, b_graph):
        """前向传播，适配data_preprocessing.py的数据格式"""
        self._encode_step_cache.clear()
        try:
            # 处理输入数据格式
            if isinstance(b_graph, dict):
                # 从字典中提取子结构数据
                h_substruct = b_graph.get('h_substruct', None)
                t_substruct = b_graph.get('t_substruct', None)

                # 编码药物数据
                h_data_list = self._encode_drug_from_batch(h_data, h_substruct)
                t_data_list = self._encode_drug_from_batch(t_data, t_substruct)

                # 处理二部图
                b_graph_processed = {
                    key: value for key, value in b_graph.items()
                    if key not in ['h_substruct', 't_substruct'] and value is not None
                }
            else:
                # 兼容旧格式
                h_data_list = self._encode_drug_from_batch(h_data)
                t_data_list = self._encode_drug_from_batch(t_data)
                b_graph_processed = {'fused': b_graph} if b_graph is not None else {}

            # 通过DGN-DDI块处理
            repr_h, repr_t, weight_h_list_all, weight_t_list_all = self._process_blocks(
                h_data_list, t_data_list, b_graph_processed, return_weights=True
            )

            # 双向协同注意力：同时更新 head/tail，减少单向信息偏置。
            repr_h_attended = self.co_attention(repr_h, repr_t)
            repr_t_attended = self.co_attention(repr_t, repr_h)

            # 可学习 block 权重加权聚合（深层 block 可获得更高权重）。
            block_w = torch.softmax(self.block_weights, dim=0).view(1, -1, 1)
            repr_h_agg = (repr_h_attended * block_w).sum(dim=1)
            repr_t_agg = (repr_t_attended * block_w).sum(dim=1)

            # 知识图嵌入计算得分
            try:
                raw_scores = self.KGE(repr_h_agg, repr_t_agg, rels)
            except Exception as e:
                logger.error(f"Error in DGN_DDI forward pass: {e}")
                logger.error(
                    f"Input shapes - repr_h_agg: {repr_h_agg.shape}, repr_t_agg: {repr_t_agg.shape}, r: {rels.shape}")
                raise e
            nan_cnt = int(torch.isnan(raw_scores).sum().item())
            if nan_cnt > 0:
                logger.warning("NaN in raw RESCAL scores: %d/%d — check GNN/RESCAL outputs", nan_cnt, raw_scores.numel())
            if not self.score_higher_is_better:
                raw_scores = -raw_scores
            scale = self.logit_scale.abs() + 1e-6
            scores = raw_scores / scale
            return torch.nan_to_num(scores, nan=0.0, posinf=50.0, neginf=-50.0).clamp(-50.0, 50.0)

        except Exception as e:
            logger.error(f"Error in DGN_DDI forward pass: {e}")
            logger.error(f"Input shapes - h_data: {type(h_data)}, t_data: {type(t_data)}")
            logger.error(f"rels shape: {rels.shape if rels is not None else 'None'}")
            logger.error(f"b_graph type: {type(b_graph)}")
            raise e

    @staticmethod
    def _avg_tensors(tensors: List[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
        valid = [t for t in tensors if t is not None and isinstance(t, torch.Tensor) and t.numel() > 0]
        if not valid:
            return None
        n = min(t.size(0) for t in valid)
        stacked = torch.stack([t[:n].float() for t in valid], dim=0)
        return stacked.mean(dim=0)

    def forward_with_weight(self, h_data, t_data, rels, b_graph):
        """带注意力权重的前向传播，用于可解释性分析。分数计算与 forward 一致。"""
        self._encode_step_cache.clear()
        try:
            if isinstance(b_graph, dict):
                h_substruct = b_graph.get('h_substruct', None)
                t_substruct = b_graph.get('t_substruct', None)
                h_data_list = self._encode_drug_from_batch(h_data, h_substruct)
                t_data_list = self._encode_drug_from_batch(t_data, t_substruct)
                b_graph_processed = {
                    key: value for key, value in b_graph.items()
                    if key not in ['h_substruct', 't_substruct'] and value is not None
                }
            else:
                h_data_list = self._encode_drug_from_batch(h_data)
                t_data_list = self._encode_drug_from_batch(t_data)
                b_graph_processed = {'fused': b_graph} if b_graph is not None else {}

            bsz = len(h_data_list)
            pair_explains: List[Dict] = [dict(
                intra_h_atom=[], intra_t_atom=[], intra_h_sub=[], intra_t_sub=[],
                inter_h_atom=[], inter_t_atom=[], inter_h_sub=[], inter_t_sub=[],
                pool_h_atom=[], pool_t_atom=[],
            ) for _ in range(bsz)]
            repr_h_blocks, repr_t_blocks = [], []

            for block in self.blocks:
                next_h, next_t = [], []
                r_h_list, r_t_list = [], []
                for i in range(bsz):
                    current_b = {}
                    if b_graph_processed:
                        for key, tensor in b_graph_processed.items():
                            if isinstance(tensor, torch.Tensor) and tensor.dim() == 3:
                                current_b[key] = tensor[i] if i < tensor.size(0) else tensor[0]
                            else:
                                current_b[key] = tensor
                    h_i, t_i, r_h, r_t, _, _, expl = block.forward(
                        h_data_list[i], t_data_list[i], current_b, return_explain=True
                    )
                    if r_h.dim() == 1:
                        r_h = r_h.unsqueeze(0)
                        r_t = r_t.unsqueeze(0)
                    next_h.append(h_i)
                    next_t.append(t_i)
                    r_h_list.append(r_h)
                    r_t_list.append(r_t)
                    for k, v in expl.items():
                        if k in pair_explains[i]:
                            pair_explains[i][k].append(v)
                h_data_list, t_data_list = next_h, next_t
                repr_h_blocks.append(torch.cat(r_h_list, dim=0))
                repr_t_blocks.append(torch.cat(r_t_list, dim=0))

            repr_h = torch.stack(repr_h_blocks, dim=1)
            repr_t = torch.stack(repr_t_blocks, dim=1)
            repr_h = self._safe_tensor_operation(repr_h, "repr_h_with_weight")
            repr_t = self._safe_tensor_operation(repr_t, "repr_t_with_weight")

            attentions_h = self.co_attention(repr_h, repr_t)
            attentions_t = self.co_attention(repr_t, repr_h)
            attentions_h = self._safe_tensor_operation(attentions_h, "attentions_h_with_weight")
            attentions_t = self._safe_tensor_operation(attentions_t, "attentions_t_with_weight")

            block_w = torch.softmax(self.block_weights, dim=0)
            block_w_view = block_w.view(1, -1, 1)
            repr_h_agg = (attentions_h * block_w_view).sum(dim=1)
            repr_t_agg = (attentions_t * block_w_view).sum(dim=1)
            repr_h_agg = self._safe_tensor_operation(repr_h_agg, "repr_h_agg_with_weight")
            repr_t_agg = self._safe_tensor_operation(repr_t_agg, "repr_t_agg_with_weight")

            raw_scores = self.KGE(repr_h_agg, repr_t_agg, rels)
            raw_scores = self._safe_tensor_operation(raw_scores, "raw_scores_with_weight")
            if not self.score_higher_is_better:
                raw_scores = -raw_scores
            scale = self.logit_scale.abs() + 1e-6
            scores = self._safe_tensor_operation(raw_scores / scale, "final_scores_with_weight")

            pairs = []
            for i, raw in enumerate(pair_explains):
                pairs.append({
                    "intra_h_atom": self._avg_tensors(raw["intra_h_atom"]),
                    "intra_t_atom": self._avg_tensors(raw["intra_t_atom"]),
                    "intra_h_sub": self._avg_tensors(raw["intra_h_sub"]),
                    "intra_t_sub": self._avg_tensors(raw["intra_t_sub"]),
                    "inter_h_atom": self._avg_tensors(raw["inter_h_atom"]),
                    "inter_t_atom": self._avg_tensors(raw["inter_t_atom"]),
                    "inter_h_sub": self._avg_tensors(raw["inter_h_sub"]),
                    "inter_t_sub": self._avg_tensors(raw["inter_t_sub"]),
                    "pool_h_atom": self._avg_tensors(raw["pool_h_atom"]),
                    "pool_t_atom": self._avg_tensors(raw["pool_t_atom"]),
                    "atom2substruct_h": h_data_list[i]["atom2substruct"],
                    "atom2substruct_t": t_data_list[i]["atom2substruct"],
                })

            explain = {
                "block_weights": block_w.detach(),
                "pairs": pairs,
            }
            return scores, explain

        except Exception as e:
            logger.error(f"Error in forward_with_weight: {e}")
            raise e