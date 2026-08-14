import os
import random
import re
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from collections import defaultdict, OrderedDict
import logging

# 配置日志
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 基于当前脚本所在目录（drugbank_test）的相对路径，便于任意机器运行
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DRUGBANK_DIR = os.path.join(_SCRIPT_DIR, "drugbank")
_BIPARTITE_CACHE_ROOT_DIR = os.path.join(_DRUGBANK_DIR, "bipartite_cache")


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _default_use_bipartite_disk_cache() -> bool:
    """若未显式设置环境变量，且磁盘缓存目录已有 .pt 文件，则默认开启读盘。"""
    if os.getenv("DDI_USE_BIPARTITE_DISK_CACHE") is not None:
        return _env_bool("DDI_USE_BIPARTITE_DISK_CACHE", False)
    cache_dir = _BIPARTITE_CACHE_ROOT_DIR
    if not os.path.isdir(cache_dir):
        return False
    try:
        for root, _, files in os.walk(cache_dir):
            for name in files:
                if name.endswith(".pt"):
                    return True
    except OSError:
        pass
    return False


def _stable_json_dumps(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def get_bipartite_cache_signature(config: dict = None) -> str:
    cfg = config or {
        "SIMILARITY_THRESHOLD": _env_float("DDI_SIMILARITY_THRESHOLD", 0.3),
        "GRANULARITY_WEIGHTS": {"atom": 0.6, "substruct": 0.4},
        "MAX_B_GRAPH_H": _env_int("DDI_MAX_B_GRAPH_H", 128),
        "MAX_B_GRAPH_W": _env_int("DDI_MAX_B_GRAPH_W", 128),
    }
    payload = {
        "similarity_threshold": float(cfg["SIMILARITY_THRESHOLD"]),
        "granularity_weights": cfg["GRANULARITY_WEIGHTS"],
        "max_b_graph_h": int(cfg["MAX_B_GRAPH_H"]),
        "max_b_graph_w": int(cfg["MAX_B_GRAPH_W"]),
    }
    return hashlib.sha1(_stable_json_dumps(payload).encode("utf-8")).hexdigest()[:12]


def get_bipartite_cache_dir(config: dict = None) -> str:
    sig = get_bipartite_cache_signature(config)
    return os.path.join(_BIPARTITE_CACHE_ROOT_DIR, f"v_{sig}")


CONFIG = {
    "DATA_DICT_PATH": os.path.join(_DRUGBANK_DIR, "drug_data_dict.pkl"),
    "DRUG_SMILES_PATH": os.path.join(_DRUGBANK_DIR, "drug_smiles.csv"),
    "DDI_CSV_PATH": os.path.join(_DRUGBANK_DIR, "ddis.csv"),
    "BATCH_SIZE": 32,
    "SIMILARITY_THRESHOLD": _env_float("DDI_SIMILARITY_THRESHOLD", 0.3),
    "GRANULARITY_WEIGHTS": {"atom": 0.6, "substruct": 0.4},
    "MAX_B_GRAPH_H": _env_int("DDI_MAX_B_GRAPH_H", 128),
    "MAX_B_GRAPH_W": _env_int("DDI_MAX_B_GRAPH_W", 128),
    "BIPARTITE_CACHE_DIR": get_bipartite_cache_dir(),
    # 若 bipartite_cache/ 下已有 .pt，默认读盘；否则仅内存 LRU。
    # 强制开关：DDI_USE_BIPARTITE_DISK_CACHE=0|1
    "USE_BIPARTITE_DISK_CACHE": _default_use_bipartite_disk_cache(),
    # 是否将新计算的二部图写回磁盘（训练默认关闭，离线 precompute_bipartite_cache.py 时开）。
    "WRITE_BIPARTITE_DISK_CACHE": _env_bool("DDI_WRITE_BIPARTITE_DISK_CACHE", False),
    "BIPARTITE_MEM_CACHE_MAX": 65536,
    # 训练不需要 RDKit 逐条解析 SMILES；设为 1 可恢复旧行为
    "PARSE_MOL_STRUCTURES_ON_INIT": _env_bool("DDI_PARSE_MOL_STRUCTURES", False),
}


def _normalize_relation_type(r):
    """使 CSV 中 type 可混用 int/float/string 时仍与词表键一致。"""
    if isinstance(r, (float, int)) and not isinstance(r, bool):
        if float(r).is_integer():
            return int(r)
        return r
    return r


def build_global_relation_vocab_from_csv(ddi_csv_path) -> dict:
    """从 ddis.csv 读取全部 `type` 并构建稳定 rel 列表，供全数据划分共用。"""
    df = pd.read_csv(ddi_csv_path, usecols=["type"])
    raw = set(_normalize_relation_type(t) for t in df["type"].unique())
    rel_list = sorted(raw, key=lambda x: (str(type(x)).lower(), str(x)))
    return {
        "rel_to_id": {t: i for i, t in enumerate(rel_list)},
        "id_to_rel": list(rel_list),
    }


def _canonical_drug_pair_key(d1, d2):
    """无向对 (min, max) 以字符串比较 DB ID。"""
    a, b = (d1, d2) if str(d1) <= str(d2) else (d2, d1)
    return a, b, (d1, d2) != (a, b)


def _transpose_b_graph_dict(bdict):
    """当药物顺序与缓存 (min→max) 中 h/t 对调时，对 2D 二部图做转置；子图张量维不变，仅对融合键。"""
    return {
        "atom": bdict["atom"].t() if bdict["atom"] is not None else None,
        "substruct": bdict["substruct"].t() if bdict["substruct"] is not None else None,
        "fused": bdict["fused"].t() if bdict["fused"] is not None else None,
    }


def _filename_safe_drug_id(did) -> str:
    s = str(did)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)[:200]


class BipartiteGraphCache:
    """无向对 (d_min,d_max) 的二部图张量；内存 LRU + 可选落盘 .pt 文件。"""

    def __init__(self, cache_dir, use_disk, write_disk, mem_max, cache_signature: str = ""):
        self.cache_dir = cache_dir
        self.use_disk = use_disk
        self.write_disk = write_disk and use_disk
        self.mem_max = mem_max
        self.cache_signature = cache_signature or get_bipartite_cache_signature()
        self._mem: OrderedDict = OrderedDict()
        if use_disk and cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def _path(self, a, b):
        fn = f"{_filename_safe_drug_id(a)}__{_filename_safe_drug_id(b)}.pt"
        return os.path.join(self.cache_dir, fn)

    def get(self, a, b) -> object:
        a, b, _ = _canonical_drug_pair_key(a, b)
        key = (a, b)
        if key in self._mem:
            self._mem.move_to_end(key)
            return self._mem[key]
        if self.use_disk and self.cache_dir:
            p = self._path(a, b)
            if os.path.isfile(p):
                data = torch.load(p, map_location="cpu", weights_only=False)
                payload = data
                if isinstance(data, dict) and "payload" in data and "cache_signature" in data:
                    if str(data["cache_signature"]) != str(self.cache_signature):
                        logger.warning(
                            "Ignoring stale bipartite cache %s: file signature=%s current=%s",
                            p, data.get("cache_signature"), self.cache_signature,
                        )
                        return None
                    payload = data["payload"]
                self._set_mem(key, payload)
                return payload
        return None

    def set(self, a, b, b_graph_dict) -> None:
        a, b, _ = _canonical_drug_pair_key(a, b)
        key = (a, b)
        self._set_mem(key, b_graph_dict)
        if self.write_disk and self.cache_dir and b_graph_dict is not None:
            p = self._path(a, b)
            try:
                torch.save(
                    {
                        "cache_signature": self.cache_signature,
                        "payload": b_graph_dict,
                    },
                    p,
                )
            except OSError as e:
                logger.warning(f"Failed to write bipartite cache {p}: {e}")

    def _set_mem(self, key, data):
        self._mem[key] = data
        self._mem.move_to_end(key)
        while len(self._mem) > self.mem_max and self.mem_max > 0:
            self._mem.popitem(last=False)

    def warm_memory_from_pairs(self, pairs) -> tuple:
        """将无向药物对批量载入内存 LRU（优先命中磁盘 .pt）。"""
        from tqdm import tqdm

        unique = []
        seen = set()
        for h, t in pairs:
            a, b, _ = _canonical_drug_pair_key(h, t)
            if (a, b) not in seen:
                seen.add((a, b))
                unique.append((a, b))
        loaded = 0
        iterator = unique
        if len(unique) > 100:
            iterator = tqdm(unique, desc="bipartite warm", unit="pair", file=sys.stdout)
        for a, b in iterator:
            if self.get(a, b) is not None:
                loaded += 1
        return len(unique), loaded


def _pad_b_graph_batch_3d(tensor: torch.Tensor, max_h: int, max_w: int, pad_value: float = 0.01) -> torch.Tensor:
    """将 [B, H, W] 二部图 batch 对齐到统一 (max_h, max_w)。"""
    if tensor is None or tensor.numel() == 0:
        return tensor
    b, h, w = tensor.shape
    h = min(h, max_h)
    w = min(w, max_w)
    cropped = tensor[:, :h, :w]
    pad_h = max_h - h
    pad_w = max_w - w
    if pad_h > 0 or pad_w > 0:
        return F.pad(cropped, (0, pad_w, 0, pad_h), value=pad_value)
    return cropped


def _merge_b_graph_batch_tensors(pos_tensor: torch.Tensor, neg_tensor: torch.Tensor) -> torch.Tensor:
    """
    正负 batch 的二部图张量各自按 batch 内 max 做了 pad，合并前需再对齐空间维。
    输入形状均为 [B, H, W]。
    """
    if neg_tensor is None:
        return pos_tensor
    if pos_tensor is None:
        return neg_tensor
    max_h = min(max(pos_tensor.size(1), neg_tensor.size(1)), CONFIG["MAX_B_GRAPH_H"])
    max_w = min(max(pos_tensor.size(2), neg_tensor.size(2)), CONFIG["MAX_B_GRAPH_W"])
    pos_aligned = _pad_b_graph_batch_3d(pos_tensor, max_h, max_w)
    neg_aligned = _pad_b_graph_batch_3d(neg_tensor, max_h, max_w)
    return torch.cat([pos_aligned, neg_aligned], dim=0)


def merge_ddi_batches(pos_tri, neg_tri):
    """
    合并正负样本 collate 结果，供单次 model forward 使用。
    返回 ((h, t, r, bg), n_pos)；无负样本时等价于 (pos_tri, n_pos)。
    """
    pos_h, pos_t, pos_r, pos_bg = pos_tri
    n_pos = int(pos_r.size(0))
    if neg_tri is None or neg_tri[0] is None:
        return pos_tri, n_pos

    neg_h, neg_t, neg_r, neg_bg = neg_tri
    if neg_h is None or not hasattr(neg_h, "num_graphs") or neg_h.num_graphs == 0:
        return pos_tri, n_pos

    combined_h = Batch.from_data_list(pos_h.to_data_list() + neg_h.to_data_list())
    combined_t = Batch.from_data_list(pos_t.to_data_list() + neg_t.to_data_list())
    combined_r = torch.cat([pos_r, neg_r], dim=0)

    combined_bg = {}
    for key, pv in pos_bg.items():
        if pv is None:
            continue
        nv = neg_bg.get(key) if neg_bg else None
        if key in ("h_substruct", "t_substruct"):
            if nv is not None:
                combined_bg[key] = Batch.from_data_list(pv.to_data_list() + nv.to_data_list())
            else:
                combined_bg[key] = pv
        elif isinstance(pv, torch.Tensor) and nv is not None:
            if pv.dim() == 3 and nv.dim() == 3:
                combined_bg[key] = _merge_b_graph_batch_tensors(pv, nv)
            else:
                combined_bg[key] = torch.cat([pv, nv], dim=0)
        else:
            combined_bg[key] = pv

    return (combined_h, combined_t, combined_r, combined_bg), n_pos


"""药物-药物相互作用数据加载器，负责基础数据加载与预处理"""

class DDIDataLoader:
    """药物-药物相互作用数据加载器，负责基础数据加载与预处理"""

    def __init__(self):
        self.drug_graph_dict = self._load_drug_graph_dict()
        self.drug_id_to_int, self.int_to_drug_id = self._extract_id_mappings()
        self.drug_smiles = self._load_drug_smiles()
        self._init_global_relation_vocab()
        bipartite_sig = get_bipartite_cache_signature(CONFIG)
        self.bipartite_cache = BipartiteGraphCache(
            CONFIG.get("BIPARTITE_CACHE_DIR", os.path.join(_DRUGBANK_DIR, "bipartite_cache")),
            CONFIG.get("USE_BIPARTITE_DISK_CACHE", True),
            CONFIG.get("WRITE_BIPARTITE_DISK_CACHE", False),
            int(CONFIG.get("BIPARTITE_MEM_CACHE_MAX", 8192)),
            cache_signature=bipartite_sig,
        )
        logger.info(
            "Bipartite cache config: signature=%s, dir=%s, use_disk=%s, write_disk=%s, mem_max=%s",
            bipartite_sig,
            CONFIG.get("BIPARTITE_CACHE_DIR", ""),
            bool(CONFIG.get("USE_BIPARTITE_DISK_CACHE", False)),
            bool(CONFIG.get("WRITE_BIPARTITE_DISK_CACHE", False)),
            int(CONFIG.get("BIPARTITE_MEM_CACHE_MAX", 8192)),
        )
        self.all_pos_tup = []  # 从外部CSV加载，这里初始化空列表
        if CONFIG.get("PARSE_MOL_STRUCTURES_ON_INIT", False):
            self.drug_to_mol_graph = self._parse_mol_structures()
        else:
            self.drug_to_mol_graph = {}
            logger.info("Skipped RDKit mol parsing on init (DDI_PARSE_MOL_STRUCTURES=0)")
        self._build_relation_structures()  # 初始构建空结构，后续会更新
        logger.info("DDIDataLoader initialization completed")

    def _load_drug_graph_dict(self):
        """加载药物图数据字典（包含原子级和子结构级双粒度信息）"""
        logger.info("Loading drug graph data from %s ...", CONFIG["DATA_DICT_PATH"])
        try:
            with open(CONFIG["DATA_DICT_PATH"], 'rb') as f:
                logger.info("Pickle file opened, deserializing (large file may take 1-2 min)...")
                data = pd.read_pickle(f)

            # 验证元数据完整性
            if '__metadata__' in data:
                metadata = data['__metadata__']
                logger.info(f"Metadata: Processed {metadata.get('total_drugs', 'N/A')} drugs "
                            f"(Success: {metadata.get('successful_count', 0)}, "
                            f"Failed: {metadata.get('failed_count', 0)})")

            # 过滤无效药物数据
            valid_drugs = [k for k in data.keys() if k != '__metadata__'
                           and isinstance(data[k], dict)
                           and all(key in data[k] for key in ['atom', 'substruct', 'atom2substruct'])]
            logger.info(f"Loaded {len(valid_drugs)} valid drugs with dual-granularity features")
            return data
        except Exception as e:
            logger.error(f"Failed to load drug graph data: {str(e)}")
            raise

    def _extract_id_mappings(self):
        """提取药物ID与整数索引的映射关系"""
        logger.info("Creating drug ID mappings...")
        try:
            if '__metadata__' in self.drug_graph_dict and 'drug_id_to_int' in self.drug_graph_dict['__metadata__']:
                drug_id_to_int = self.drug_graph_dict['__metadata__']['drug_id_to_int']
                int_to_drug_id = {v: k for k, v in drug_id_to_int.items()}
                logger.info(f"Loaded existing mappings with {len(drug_id_to_int)} entries")
            else:
                valid_drug_ids = [k for k in self.drug_graph_dict.keys() if k != '__metadata__']
                drug_id_to_int = {drug_id: idx for idx, drug_id in enumerate(valid_drug_ids)}
                int_to_drug_id = {v: k for k, v in drug_id_to_int.items()}
                logger.info(f"Created new mappings for {len(valid_drug_ids)} drugs")
            return drug_id_to_int, int_to_drug_id
        except Exception as e:
            logger.error(f"Failed to create drug ID mappings: {str(e)}")
            raise

    def _init_global_relation_vocab(self):
        """自 ddis.csv 构建全体关系 type 的固定整型 id，不随各 split 覆盖。"""
        ddi_path = CONFIG["DDI_CSV_PATH"]
        if not os.path.isfile(ddi_path):
            logger.error(f"DDI CSV for relation vocab not found: {ddi_path}; rel_to_id 为空。")
            self.rel_to_id = {}
            self.id_to_rel = []
            self.rel_total = 0
            return
        try:
            vocab = build_global_relation_vocab_from_csv(ddi_path)
            self.rel_to_id = vocab["rel_to_id"]
            self.id_to_rel = vocab["id_to_rel"]
            self.rel_total = len(self.id_to_rel)
            logger.info(
                f"Global relation vocabulary from {ddi_path}: rel_total={self.rel_total} unique types"
            )
        except Exception as e:
            logger.error(f"Failed to build global relation vocabulary: {e}")
            self.rel_to_id, self.id_to_rel, self.rel_total = {}, [], 0

    def _load_drug_smiles(self):
        """加载药物SMILES字符串数据"""
        logger.info("Loading drug SMILES data...")
        try:
            df = pd.read_csv(CONFIG["DRUG_SMILES_PATH"])
            if not {'drug_id', 'smiles'}.issubset(df.columns):
                raise ValueError("SMILES file must contain 'drug_id' and 'smiles' columns")
            logger.info(f"Loaded SMILES data for {len(df)} drugs")
            return df
        except Exception as e:
            logger.error(f"Failed to load SMILES data: {str(e)}")
            raise

    def _load_and_filter_ddis_from_csv(self, csv_path, max_rows=None):
        """从指定CSV文件加载并过滤有效的药物-药物相互作用数据"""
        logger.info(f"Loading DDI interaction data from {csv_path}...")
        try:
            read_kw = {"nrows": int(max_rows)} if max_rows and int(max_rows) > 0 else {}
            df = pd.read_csv(csv_path, **read_kw)
            logger.info("Read %d rows from CSV, filtering valid drug pairs...", len(df))
            if not {'d1', 'd2', 'type'}.issubset(df.columns):
                raise ValueError("DDI file must contain 'd1', 'd2', and 'type' columns")

            # 过滤掉不存在的药物对
            valid_ddis = []
            for h, t, r in zip(df['d1'], df['d2'], df['type']):
                r = _normalize_relation_type(r)
                if h in self.drug_graph_dict and t in self.drug_graph_dict:
                    valid_ddis.append((h, t, r))

            logger.info(f"Filtered DDI data from {csv_path}: {len(valid_ddis)}/{len(df)} valid interactions")
            return valid_ddis
        except Exception as e:
            logger.error(f"Failed to load DDI data from {csv_path}: {str(e)}")
            raise

    def _parse_mol_structures(self):
        """解析药物分子结构（用于辅助验证）"""
        drug_to_mol = {}
        for _, row in self.drug_smiles.iterrows():
            drug_id = row['drug_id']
            if drug_id in self.drug_id_to_int:
                try:
                    from rdkit import Chem
                    mol = Chem.MolFromSmiles(str(row['smiles']).strip())
                    if mol:
                        drug_to_mol[drug_id] = mol
                except Exception as e:
                    logger.warning(f"Failed to parse SMILES for drug {drug_id}: {str(e)}")
        logger.info(f"Successfully parsed molecular structures for {len(drug_to_mol)} drugs")
        return drug_to_mol

    def _build_relation_structures(self, ddi_triples=None):
        """构建关系统计结构，用于负样本生成"""
        self.ALL_TRUE_H_WITH_TR = defaultdict(set)  # (t, r) -> {h1, h2, ...}
        self.ALL_TRUE_T_WITH_HR = defaultdict(set)  # (h, r) -> {t1, t2, ...}
        self.FREQ_REL = defaultdict(int)  # 关系出现频率
        self.ALL_H_WITH_R = defaultdict(set)  # r -> {h1, h2, ...}
        self.ALL_T_WITH_R = defaultdict(set)  # r -> {t1, t2, ...}

        # 使用提供的三元组列表或已加载的全部正样本
        triples = ddi_triples if ddi_triples is not None else self.all_pos_tup
        for h, t, r in triples:
            r = _normalize_relation_type(r)
            self.ALL_TRUE_H_WITH_TR[(t, r)].add(h)
            self.ALL_TRUE_T_WITH_HR[(h, r)].add(t)
            self.FREQ_REL[r] += 1
            self.ALL_H_WITH_R[r].add(h)
            self.ALL_T_WITH_R[r].add(t)

        # 转换为numpy数组提高查询效率
        self.ALL_TRUE_H_WITH_TR = {k: np.array(list(v)) for k, v in self.ALL_TRUE_H_WITH_TR.items()}
        self.ALL_TRUE_T_WITH_HR = {k: np.array(list(v)) for k, v in self.ALL_TRUE_T_WITH_HR.items()}
        self.ALL_H_WITH_R = {r: np.array(list(v)) for r, v in self.ALL_H_WITH_R.items()}
        self.ALL_T_WITH_R = {r: np.array(list(v)) for r, v in self.ALL_T_WITH_R.items()}

        # 计算关系频率统计（用于负样本生成策略）
        self.ALL_HEAD_PER_TAIL = {}  # 每个尾实体平均头实体数
        self.ALL_TAIL_PER_HEAD = {}  # 每个头实体平均尾实体数
        for r in self.FREQ_REL:
            num_tails = len(self.ALL_T_WITH_R.get(r, []))
            num_heads = len(self.ALL_H_WITH_R.get(r, []))
            self.ALL_HEAD_PER_TAIL[r] = self.FREQ_REL[r] / num_tails if num_tails > 0 else 0
            self.ALL_TAIL_PER_HEAD[r] = self.FREQ_REL[r] / num_heads if num_heads > 0 else 0

        logger.info(f"Built relation structures for {len(self.FREQ_REL)} unique DDI types")


"""药物相互作用预测数据集，支持双粒度特征和二部图构建"""


class DrugDataset(Dataset):
    """药物相互作用预测数据集，支持双粒度特征和二部图构建"""

    def __init__(self, csv_path, data_loader, neg_ent=1, disjoint_split=True, shuffle=True, return_pos_triples=False,
                 hard_neg_ratio=0.7, csv_max_rows=None):
        self.data_loader = data_loader
        self.neg_ent = neg_ent  # 每个正样本生成的负样本数
        self.return_pos_triples = return_pos_triples
        self.hard_neg_ratio = hard_neg_ratio
        self.tri_list = self.data_loader._load_and_filter_ddis_from_csv(csv_path, max_rows=csv_max_rows)
        self._drug_data_cache = {}
        # 关系统计按数据集维持，避免 train/val/test 互相覆盖导致采样偏移。
        self._build_local_relation_structures(self.tri_list)

        # 确定数据集包含的药物ID
        if disjoint_split and self.tri_list:
            d1, d2, _ = zip(*self.tri_list)
            self.drug_ids = np.array(list(set(d1) | set(d2)))
        else:
            self.drug_ids = np.array([k for k in data_loader.drug_graph_dict.keys() if k != '__metadata__'])

        # 过滤存在于药物图字典中的药物ID
        self.drug_ids = np.array([id for id in self.drug_ids if id in data_loader.drug_graph_dict])

        # 打乱数据
        if shuffle and self.tri_list:
            random.shuffle(self.tri_list)

        logger.info(f"Dataset initialized with {len(self.tri_list)} samples "
                    f"({len(self.drug_ids)} unique drugs)")

    def __len__(self):
        return len(self.tri_list)

    def __getitem__(self, idx):
        return self.tri_list[idx]

    def _build_local_relation_structures(self, ddi_triples=None):
        """构建仅属于当前数据集的关系统计结构。"""
        self.ALL_TRUE_H_WITH_TR = defaultdict(set)  # (t, r) -> {h1, h2, ...}
        self.ALL_TRUE_T_WITH_HR = defaultdict(set)  # (h, r) -> {t1, t2, ...}
        self.FREQ_REL = defaultdict(int)  # 关系出现频率
        self.ALL_H_WITH_R = defaultdict(set)  # r -> {h1, h2, ...}
        self.ALL_T_WITH_R = defaultdict(set)  # r -> {t1, t2, ...}

        triples = ddi_triples if ddi_triples is not None else self.tri_list
        for h, t, r in triples:
            r = _normalize_relation_type(r)
            self.ALL_TRUE_H_WITH_TR[(t, r)].add(h)
            self.ALL_TRUE_T_WITH_HR[(h, r)].add(t)
            self.FREQ_REL[r] += 1
            self.ALL_H_WITH_R[r].add(h)
            self.ALL_T_WITH_R[r].add(t)

        self.ALL_TRUE_H_WITH_TR = {k: np.array(list(v)) for k, v in self.ALL_TRUE_H_WITH_TR.items()}
        self.ALL_TRUE_T_WITH_HR = {k: np.array(list(v)) for k, v in self.ALL_TRUE_T_WITH_HR.items()}
        self.ALL_H_WITH_R = {r: np.array(list(v)) for r, v in self.ALL_H_WITH_R.items()}
        self.ALL_T_WITH_R = {r: np.array(list(v)) for r, v in self.ALL_T_WITH_R.items()}

        self.ALL_HEAD_PER_TAIL = {}
        self.ALL_TAIL_PER_HEAD = {}
        for r in self.FREQ_REL:
            num_tails = len(self.ALL_T_WITH_R.get(r, []))
            num_heads = len(self.ALL_H_WITH_R.get(r, []))
            self.ALL_HEAD_PER_TAIL[r] = self.FREQ_REL[r] / num_tails if num_tails > 0 else 0
            self.ALL_TAIL_PER_HEAD[r] = self.FREQ_REL[r] / num_heads if num_heads > 0 else 0

    def _get_drug_data(self, drug_id):
        """安全获取药物的双粒度特征数据并验证完整性（带实例级缓存）。"""
        if not hasattr(self, "_drug_data_cache") or self._drug_data_cache is None:
            self._drug_data_cache = {}
        if drug_id in self._drug_data_cache:
            return self._drug_data_cache[drug_id]

        if drug_id not in self.data_loader.drug_graph_dict:
            logger.error(f"Drug {drug_id} not found in dataset")
            return None

        data = self.data_loader.drug_graph_dict[drug_id]

        # 验证双粒度特征完整性
        required_keys = ['atom', 'substruct', 'atom2substruct']
        if not all(k in data for k in required_keys):
            logger.error(f"Drug {drug_id} missing features: {[k for k in required_keys if k not in data]}")
            return None

        # 验证原子级特征
        atom_data = data['atom']
        if not hasattr(atom_data, 'x') or atom_data.x is None or atom_data.x.numel() == 0 or \
                torch.isnan(atom_data.x).any() or torch.isinf(atom_data.x).any():
            logger.error(f"Drug {drug_id} has invalid atom features (empty, NaN, or Inf)")
            return None
        if not hasattr(atom_data, 'edge_index') or atom_data.edge_index is None:
            logger.error(f"Drug {drug_id} has invalid atom edge index")
            return None

        # 验证子结构级特征
        substruct_data = data['substruct']
        if not hasattr(substruct_data, 'x') or substruct_data.x is None or substruct_data.x.numel() == 0 or \
                torch.isnan(substruct_data.x).any() or torch.isinf(substruct_data.x).any():
            logger.error(f"Drug {drug_id} has invalid substruct features (empty, NaN, or Inf)")
            return None
        if not hasattr(substruct_data, 'edge_index') or substruct_data.edge_index is None:
            logger.error(f"Drug {drug_id} has invalid substruct edge index")
            return None

        self._drug_data_cache[drug_id] = data
        return data

    def _calculate_feature_similarity(self, features1, features2):
        """计算两组特征之间的相似性矩阵"""
        features1 = features1.cpu()
        features2 = features2.cpu()

        # 归一化特征
        features1_norm = F.normalize(features1, p=2, dim=1)
        features2_norm = F.normalize(features2, p=2, dim=1)

        # 计算余弦相似度
        similarity = torch.mm(features1_norm, features2_norm.t())

        # 应用阈值增强相关性
        similarity = (similarity + 1) / 2  # 归一化到0-1范围
        similarity = torch.where(similarity > CONFIG["SIMILARITY_THRESHOLD"],
                                 similarity,
                                 torch.tensor(0.01, device=similarity.device))
        return similarity

    def _create_granular_b_graph(self, h_data, t_data, granularity):
        """创建特定粒度的二部图（原子级或子结构级）"""
        try:
            # 获取指定粒度的特征
            h_features = h_data[granularity].x
            t_features = t_data[granularity].x

            # 获取实际节点数
            h_nodes = h_features.size(0)
            t_nodes = t_features.size(0)

            if h_nodes <= 0 or t_nodes <= 0:
                raise ValueError(f"Invalid node counts for {granularity} level: h={h_nodes}, t={t_nodes}")

            # 基于特征相似性创建二部图
            b_graph = self._calculate_feature_similarity(h_features, t_features)

            # 数值稳定性处理
            return torch.clamp(b_graph, min=1e-7, max=1.0)
        except Exception as e:
            logger.error(f"Failed to create {granularity} level bipartite graph: {str(e)}")
            return None

    def _create_dual_granularity_b_graph(self, h_data, t_data, h_id=None, t_id=None):
        """创建双粒度融合的二部图；可缓存无向对 (d_min,d_max) 的结果。"""
        cache = getattr(self.data_loader, "bipartite_cache", None)
        if h_id is not None and t_id is not None and cache is not None:
            a, b, need_transpose = _canonical_drug_pair_key(h_id, t_id)
            cached = cache.get(a, b)
            if cached is not None:
                d = {k: cached[k] for k in ("atom", "substruct", "fused") if k in cached}
                if need_transpose and len(d) == 3:
                    d = _transpose_b_graph_dict(d)
                return d
        # 生成原子级二部图
        atom_graph = self._create_granular_b_graph(h_data, t_data, "atom")
        if atom_graph is None:
            return None

        # 生成子结构级二部图
        substruct_graph = self._create_granular_b_graph(h_data, t_data, "substruct")
        if substruct_graph is None:
            return None

        # 获取原子-子结构映射矩阵
        h_a2s = h_data["atom2substruct"]  # [num_atoms_h, num_substructs_h]
        t_a2s = t_data["atom2substruct"]  # [num_atoms_t, num_substructs_t]

        target_device = substruct_graph.device
        h_a2s = h_a2s.to(target_device)
        t_a2s = t_a2s.to(target_device)

        # 子结构级到原子级的投影
        try:
            projected_substruct_graph = torch.mm(torch.mm(h_a2s, substruct_graph), t_a2s.t())
        except RuntimeError as e:
            logger.error(f"Matrix multiplication for substructure projection failed: {e}")
            logger.error(
                f"Shapes: h_a2s: {h_a2s.shape}, substruct_graph: {substruct_graph.shape}, t_a2s.t(): {t_a2s.t().shape}")
            return None

        # 规范化投影后的子结构二部图
        if projected_substruct_graph.numel() > 0:
            projected_substruct_graph = F.normalize(projected_substruct_graph, p=1, dim=1)

        # 双粒度融合（确保形状匹配）
        if atom_graph.shape != projected_substruct_graph.shape:
            min_h = min(atom_graph.shape[0], projected_substruct_graph.shape[0])
            min_w = min(atom_graph.shape[1], projected_substruct_graph.shape[1])
            atom_graph = atom_graph[:min_h, :min_w]
            projected_substruct_graph = projected_substruct_graph[:min_h, :min_w]

        fused_graph = (CONFIG["GRANULARITY_WEIGHTS"]["atom"] * atom_graph +
                       CONFIG["GRANULARITY_WEIGHTS"]["substruct"] * projected_substruct_graph)

        result = {
            "atom": atom_graph,
            "substruct": substruct_graph,
            "fused": fused_graph,
        }

        if h_id is not None and t_id is not None and cache is not None:
            a, b, tr = _canonical_drug_pair_key(h_id, t_id)
            if tr:
                to_store = _transpose_b_graph_dict(result)
            else:
                to_store = {k: v.clone().detach() for k, v in result.items()}
            cache.set(a, b, to_store)

        return result

    def _create_b_graph_batch(self, b_graphs_dict, granularity):
        """为特定粒度创建二部图批次"""
        if not b_graphs_dict or granularity not in b_graphs_dict[0] or b_graphs_dict[0][granularity] is None:
            return None

        b_graphs = [bg[granularity] for bg in b_graphs_dict if bg[granularity] is not None]

        if not b_graphs:
            return None

        # 计算批次内最大尺寸
        max_h = max(bg.size(0) for bg in b_graphs)
        max_w = max(bg.size(1) for bg in b_graphs)

        # 限制最大尺寸（防止内存溢出）
        max_h = min(max_h, CONFIG["MAX_B_GRAPH_H"])
        max_w = min(max_w, CONFIG["MAX_B_GRAPH_W"])

        # 填充并堆叠
        padded_graphs = []
        for bg in b_graphs:
            if bg.size(0) > max_h:
                bg = bg[:max_h, :]
            if bg.size(1) > max_w:
                bg = bg[:, :max_w]

            pad_h = max_h - bg.size(0)
            pad_w = max_w - bg.size(1)
            padded = F.pad(bg, (0, pad_w, 0, pad_h), value=0.01)

            padded_graphs.append(padded)

        return torch.stack(padded_graphs, dim=0)

    def _create_drug_batch(self, drug_data_list, granularity='atom'):
        """创建药物特征批次（支持原子级/子结构级双粒度）"""
        if not drug_data_list:
            return None

        data_list = []
        for data in drug_data_list:
            graph_data = data[granularity].clone()
            graph_data = graph_data.to('cpu')
            if torch.isnan(graph_data.x).any() or torch.isinf(graph_data.x).any():
                graph_data.x = torch.nan_to_num(graph_data.x, nan=0.0, posinf=1e6, neginf=-1e6)
            data_list.append(graph_data)

        try:
            batch = Batch.from_data_list(data_list)
            return batch
        except Exception as e:
            logger.error(f"Failed to create {granularity} batch: {str(e)}")
            return None

    def _get_relation_id(self, relation):
        """将关系类型转换为整数ID（与全局 ddis 词表一致）。"""
        r = _normalize_relation_type(relation)
        rid = self.data_loader.rel_to_id.get(r)
        if rid is not None:
            return rid
        if self.data_loader.rel_total == 0:
            return 0
        logger.warning(f"Unknown relation type: {relation} (normalized {r!r}), using 0")
        return 0

    def _corrupt_ent(self, other_ent, rel, ent_dict, n=1):
        """生成负样本实体（替换头/尾实体）"""
        positive_set = set(ent_dict.get((other_ent, rel), []))
        candidates = [d for d in self.drug_ids if d not in positive_set]

        if not candidates:
            logger.warning(f"No valid candidates for ({other_ent}, {rel}), using random")
            candidates = self.drug_ids

        return np.random.choice(candidates, size=min(n, len(candidates)), replace=False)

    def _corrupt_head(self, tail, rel, n=1):
        """生成头实体负样本"""
        return self._corrupt_ent(tail, rel, self.ALL_TRUE_H_WITH_TR, n)

    def _corrupt_tail(self, head, rel, n=1):
        """生成尾实体负样本"""
        return self._corrupt_ent(head, rel, self.ALL_TRUE_T_WITH_HR, n)

    def _generate_neg_samples(self, h, t, r):
        """根据关系统计生成负样本"""
        if r not in self.ALL_TAIL_PER_HEAD or r not in self.ALL_HEAD_PER_TAIL:
            neg_head_count = self.neg_ent // 2
            neg_tail_count = self.neg_ent - neg_head_count
        else:
            prob = self.ALL_TAIL_PER_HEAD[r] / (
                    self.ALL_TAIL_PER_HEAD[r] + self.ALL_HEAD_PER_TAIL[r]
            )
            neg_head_count = sum(random.random() < prob for _ in range(self.neg_ent))
            neg_tail_count = self.neg_ent - neg_head_count

        return (
            self._corrupt_head(t, r, neg_head_count),
            self._corrupt_tail(h, r, neg_tail_count)
        )

    def _neg_sample_heads_tails(self, h, t, r):
        return self._generate_neg_samples(h, t, r)

    def collate_fn(self, batch):
        """批处理函数，构建正负样本批次并确保双粒度二部图一致性"""
        # 过滤掉无法获取有效数据的样本
        valid_triples = []
        for h, t, r in batch:
            try:
                h_data = self._get_drug_data(h)
                t_data = self._get_drug_data(t)
                if h_data is not None and t_data is not None:
                    valid_triples.append((h, t, r, h_data, t_data))
                else:
                    logger.warning(f"Skipping invalid sample: ({h}, {t}, {r})")
            except Exception as e:
                logger.warning(f"Skipping sample due to error: ({h}, {t}, {r}) - {e}")
                continue

        if not valid_triples:
            logger.warning("Empty batch after filtering invalid samples.")
            return (None, None, None, None), (None, None, None, None), None

        pos_h_data = []
        pos_t_data = []
        pos_rels = []
        pos_b_graphs = []
        pos_meta = [] if self.return_pos_triples else None

        neg_h_data = []
        neg_t_data = []
        neg_rels = []
        neg_b_graphs = []

        for h, t, r, h_data, t_data in valid_triples:
            try:
                # 正样本
                b_graph = self._create_dual_granularity_b_graph(h_data, t_data, h_id=h, t_id=t)
                if b_graph is None:
                    continue
                pos_h_data.append(h_data)
                pos_t_data.append(t_data)
                pos_rels.append(r)
                pos_b_graphs.append(b_graph)
                if self.return_pos_triples:
                    pos_meta.append((h, t, r))

                # 负样本
                neg_heads, neg_tails = self._neg_sample_heads_tails(h, t, r)

                for neg_h in neg_heads:
                    neg_h_data_dict = self._get_drug_data(neg_h)
                    if neg_h_data_dict is not None:
                        neg_b = self._create_dual_granularity_b_graph(
                            neg_h_data_dict, t_data, h_id=neg_h, t_id=t
                        )
                        if neg_b is not None:
                            neg_h_data.append(neg_h_data_dict)
                            neg_t_data.append(t_data)
                            neg_rels.append(r)
                            neg_b_graphs.append(neg_b)

                for neg_t in neg_tails:
                    neg_t_data_dict = self._get_drug_data(neg_t)
                    if neg_t_data_dict is not None:
                        neg_b = self._create_dual_granularity_b_graph(
                            h_data, neg_t_data_dict, h_id=h, t_id=neg_t
                        )
                        if neg_b is not None:
                            neg_h_data.append(h_data)
                            neg_t_data.append(neg_t_data_dict)
                            neg_rels.append(r)
                            neg_b_graphs.append(neg_b)
            except Exception as e:
                logger.warning(f"Failed to process triple ({h}, {t}, {r}) in collate_fn: {e}")
                continue

        # 处理空批次情况
        if not pos_h_data:
            return (None, None, None, None), (None, None, None, None), None

        # 构建正样本批次
        pos_h_batch = self._create_drug_batch(pos_h_data, 'atom')
        pos_t_batch = self._create_drug_batch(pos_t_data, 'atom')
        pos_h_substruct = self._create_drug_batch(pos_h_data, 'substruct')
        pos_t_substruct = self._create_drug_batch(pos_t_data, 'substruct')
        pos_rels_tensor = torch.tensor([self._get_relation_id(r) for r in pos_rels], dtype=torch.long)
        pos_b_graphs_atom = self._create_b_graph_batch(pos_b_graphs, 'atom')
        pos_b_graphs_substruct = self._create_b_graph_batch(pos_b_graphs, 'substruct')
        pos_b_graphs_fused = self._create_b_graph_batch(pos_b_graphs, 'fused')

        pos_b_graphs_tensor = {
            'atom': pos_b_graphs_atom,
            'substruct': pos_b_graphs_substruct,
            'fused': pos_b_graphs_fused,
            'h_substruct': pos_h_substruct,
            't_substruct': pos_t_substruct
        }

        # 构建负样本批次
        neg_h_batch = self._create_drug_batch(neg_h_data, 'atom')
        neg_t_batch = self._create_drug_batch(neg_t_data, 'atom')
        neg_h_substruct = self._create_drug_batch(neg_h_data, 'substruct')
        neg_t_substruct = self._create_drug_batch(neg_t_data, 'substruct')
        neg_rels_tensor = torch.tensor([self._get_relation_id(r) for r in neg_rels], dtype=torch.long)
        neg_b_graphs_atom = self._create_b_graph_batch(neg_b_graphs, 'atom')
        neg_b_graphs_substruct = self._create_b_graph_batch(neg_b_graphs, 'substruct')
        neg_b_graphs_fused = self._create_b_graph_batch(neg_b_graphs, 'fused')

        neg_b_graphs_tensor = {
            'atom': neg_b_graphs_atom,
            'substruct': neg_b_graphs_substruct,
            'fused': neg_b_graphs_fused,
            'h_substruct': neg_h_substruct,
            't_substruct': neg_t_substruct
        }

        return (pos_h_batch, pos_t_batch, pos_rels_tensor, pos_b_graphs_tensor), \
            (neg_h_batch, neg_t_batch, neg_rels_tensor, neg_b_graphs_tensor), pos_meta

    def warm_bipartite_cache(self, max_pairs=None) -> tuple:
        """训练前将当前 tri_list 涉及的药物对预热进二部图内存缓存。"""
        cache = getattr(self.data_loader, "bipartite_cache", None)
        if cache is None:
            return 0, 0
        pairs = [(h, t) for h, t, _ in self.tri_list]
        if max_pairs is not None and max_pairs > 0 and len(pairs) > max_pairs:
            seen = set()
            limited = []
            for h, t in pairs:
                a, b, _ = _canonical_drug_pair_key(h, t)
                if (a, b) in seen:
                    continue
                seen.add((a, b))
                limited.append((h, t))
                if len(limited) >= max_pairs:
                    break
            pairs = limited
        return cache.warm_memory_from_pairs(pairs)

    def collate_positives_only(self, batch):
        """
        仅组正样本 batch（无负样本），供 online 硬负候选一次前向打分。
        batch: List[(h, t, r), ...]
        """
        if not batch:
            return None
        valid_triples = []
        for h, t, r in batch:
            try:
                h_data = self._get_drug_data(h)
                t_data = self._get_drug_data(t)
                if h_data is not None and t_data is not None:
                    valid_triples.append((h, t, r, h_data, t_data))
            except Exception:
                continue
        if not valid_triples:
            return None
        pos_h_data, pos_t_data, pos_rels, pos_b_graphs = [], [], [], []
        for h, t, r, h_data, t_data in valid_triples:
            b_graph = self._create_dual_granularity_b_graph(h_data, t_data, h_id=h, t_id=t)
            if b_graph is None:
                continue
            pos_h_data.append(h_data)
            pos_t_data.append(t_data)
            pos_rels.append(r)
            pos_b_graphs.append(b_graph)
        if not pos_h_data:
            return None
        pos_h_batch = self._create_drug_batch(pos_h_data, "atom")
        pos_t_batch = self._create_drug_batch(pos_t_data, "atom")
        pos_h_substruct = self._create_drug_batch(pos_h_data, "substruct")
        pos_t_substruct = self._create_drug_batch(pos_t_data, "substruct")
        pos_rels_tensor = torch.tensor([self._get_relation_id(r) for r in pos_rels], dtype=torch.long)
        pos_b_graphs_atom = self._create_b_graph_batch(pos_b_graphs, "atom")
        pos_b_graphs_substruct = self._create_b_graph_batch(pos_b_graphs, "substruct")
        pos_b_graphs_fused = self._create_b_graph_batch(pos_b_graphs, "fused")
        pos_b_graphs_tensor = {
            "atom": pos_b_graphs_atom,
            "substruct": pos_b_graphs_substruct,
            "fused": pos_b_graphs_fused,
            "h_substruct": pos_h_substruct,
            "t_substruct": pos_t_substruct,
        }
        return (pos_h_batch, pos_t_batch, pos_rels_tensor, pos_b_graphs_tensor)


"""优化的药物数据加载器，处理批次加载与多进程问题"""


class DrugDataLoader(DataLoader):
    """优化的药物数据加载器，处理批次加载与多进程问题"""

    def __init__(self, dataset, **kwargs):
        if 'num_workers' in kwargs and kwargs['num_workers'] > 0:
            logger.warning("Setting num_workers=0 to avoid graph data serialization issues")
            kwargs['num_workers'] = 0

        if 'batch_size' not in kwargs:
            kwargs['batch_size'] = CONFIG["BATCH_SIZE"]

        super().__init__(dataset, collate_fn=dataset.collate_fn, **kwargs)
        logger.info(f"DataLoader initialized with batch_size={kwargs['batch_size']}")


# """测试数据预处理 pipeline 完整性"""
#
# def test_data_pipeline():
#     """测试数据预处理 pipeline 完整性"""
#     logger.info("\n===== Starting Data Pipeline Test =====")
#     try:
#         ddi_loader = DDIDataLoader()
#         # 使用测试CSV路径
#         train_dataset = DrugDataset("D:/py.test/DGN-DDI/drugbank_test/drugbank/fold0/train.csv", ddi_loader, neg_ent=1)
#         test_dataset = DrugDataset("D:/py.test/DGN-DDI/drugbank_test/drugbank/fold0/test.csv", ddi_loader, neg_ent=1, shuffle=False)
#         data_loader = DrugDataLoader(train_dataset, batch_size=4)
#
#         for i, (pos_tri, neg_tri) in enumerate(data_loader):
#             if i >= 3:
#                 break
#             if pos_tri is None or pos_tri[0] is None:
#                 logger.warning(f"Batch {i + 1} has no valid positive samples.")
#                 continue
#
#             pos_h, pos_t, pos_r, pos_bg = pos_tri
#             neg_h, neg_t, neg_r, neg_bg = neg_tri
#
#             logger.info(f"\nBatch {i + 1} Positive Samples:")
#             logger.info(f"  Head atom graphs: {pos_h.num_graphs} (nodes: {pos_h.num_nodes})")
#             logger.info(
#                 f"  Head substruct graphs: {pos_bg['h_substruct'].num_graphs} (nodes: {pos_bg['h_substruct'].num_nodes})")
#             logger.info(f"  Relations shape: {pos_r.shape}")
#             logger.info(f"  Atom bipartite graphs shape: {pos_bg['atom'].shape}")
#             logger.info(f"  Substruct bipartite graphs shape: {pos_bg['substruct'].shape}")
#             logger.info(f"  Fused bipartite graphs shape: {pos_bg['fused'].shape}")
#
#             if neg_h is not None:
#                 logger.info(f"Batch {i + 1} Negative Samples:")
#                 logger.info(f"  Head atom graphs: {neg_h.num_graphs}")
#
#         logger.info("===== Data Pipeline Test Completed Successfully =====")
#
#     except Exception as e:
#         logger.error(f"Data Pipeline Test Failed: {str(e)}", exc_info=True)
#
#
# if __name__ == "__main__":
#     test_data_pipeline()