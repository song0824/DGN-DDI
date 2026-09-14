"""DGN-DDI 可解释性：注意力热力图、子结构贡献、官能团对齐。

默认读取 paper_lock 中冻结的 best_model.pth，不覆盖训练权重。
"""
import sys
import os

# 必须在重依赖 import 之前打印，否则控制台会长时间空白。
print("[DGN-DDI] 可解释性脚本已启动。", flush=True)
print("[DGN-DDI] 正在加载 PyTorch / PyG / RDKit（首次可能需 1–3 分钟，不是卡死）...", flush=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import argparse
import json
import logging
import traceback
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

print("[DGN-DDI] 基础库已加载，继续导入项目模块...", flush=True)

from data_preprocessing import DDIDataLoader, DrugDataset
from models import DGN_DDI
from transductive_test import _merge_eval_config_from_checkpoint, get_test_config

print("[DGN-DDI] 项目模块已加载。", flush=True)

logger = logging.getLogger(__name__)


def _setup_console_logging() -> None:
    """保证 INFO 日志打到 stdout，避免被缓冲或只进 stderr。"""
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    has_stream = any(isinstance(h, logging.StreamHandler) for h in root.handlers)
    if not has_stream:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(fmt)
        root.addHandler(handler)
    for handler in root.handlers:
        handler.setFormatter(fmt)
        try:
            handler.flush()
        except Exception:
            pass

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 开题案例 + 若干常见抗凝/CYP 相关对；不在数据中则自动跳过。
DEFAULT_CASES = [
    ("DB00945", "DB00682", "aspirin_warfarin"),
    ("DB00682", "DB00945", "warfarin_aspirin"),
    ("DB01050", "DB00641", "ibuprofen_simvastatin"),
    ("DB00316", "DB00682", "acetaminophen_warfarin"),
    ("DB00758", "DB00682", "clopidogrel_warfarin"),
    ("DB01076", "DB00682", "atorvastatin_warfarin"),
    ("DB00213", "DB00682", "pantoprazole_warfarin"),
    ("DB00999", "DB00682", "hydrochlorothiazide_warfarin"),
    ("DB00682", "DB00338", "warfarin_omeprazole"),
    ("DB00338", "DB00682", "omeprazole_warfarin"),
    ("DB00381", "DB00641", "amlodipine_simvastatin"),
    ("DB00641", "DB00381", "simvastatin_amlodipine"),
    ("DB00641", "DB01118", "simvastatin_amiodarone"),
    ("DB01118", "DB00641", "amiodarone_simvastatin"),
]

# 开题创新点3：酸性中心 / 疏水环 / 杂环等药效团，避免把酯/羧酸误标成 Ketone。
# 同一名称可对应多条 SMARTS（Kekulé / 芳香 / 4-羟基香豆素互变异构）。
PHARMACOPHORE_SMARTS = {
    "羧酸(酸性中心)": "C(=O)[O;H,-]",
    "酯": "[#6]C(=O)O[#6]",
    "酰胺": "C(=O)[NH2,NH1,N]",
    "酚羟基": "c[OX2H]",
    "醇羟基": "[CX4][OX2H]",
    "酮": "[#6][CX3](=O)[#6]",
    "芳环(疏水)": "c1ccccc1",
    "杂环": "[r5,r6;!#6]",
    "卤素": "[F,Cl,Br,I]",
    "磺酰胺": "S(=O)(=O)N",
    "香豆素内酯": [
        "O=C1Oc2ccccc2C=C1",                 # 2H-chromen-2-one (Kekulé)
        "O=C1Oc2ccccc2C(O)=C1",              # 4-hydroxycoumarin enol
        "O=c1oc2ccccc2cc1",                  # aromatic coumarin
        "O=c1oc2ccccc2c(O)c1",               # aromatic 4-hydroxycoumarin
        "[#8]=[#6]1[#8]c2ccccc2[#6]=[#6]1",  # generic fused 2-pyrone
    ],
}

# 图上只用英文，避免 DejaVu Sans 缺汉字。JSON/CSV 仍保留中文键，方便写论文。
PHARMACOPHORE_LABEL_EN = {
    "羧酸(酸性中心)": "Carboxylic acid",
    "酯": "Ester",
    "酰胺": "Amide",
    "酚羟基": "Phenolic OH",
    "醇羟基": "Alcoholic OH",
    "酮": "Ketone",
    "芳环(疏水)": "Aromatic ring",
    "杂环": "Heterocycle",
    "卤素": "Halogen",
    "磺酰胺": "Sulfonamide",
    "香豆素内酯": "Coumarin lactone",
}

# 已知临床机制（文献共识，供“注意力 vs 药理”对照，不替代说明书）。
KNOWN_MECHANISMS = {
    frozenset({"DB00945", "DB00682"}): {
        "pair_name": "阿司匹林–华法林",
        "clinical_effect": "出血风险增强",
        "mechanism": (
            "阿司匹林不可逆抑制血小板 COX-1，并可能竞争血浆蛋白结合；"
            "华法林抑制 VKOR，阻断维生素K循环导致凝血因子减少。"
        ),
        "expected_groups": {
            "DB00945": ["羧酸(酸性中心)", "酯", "芳环(疏水)"],
            "DB00682": ["香豆素内酯", "酚羟基", "芳环(疏水)"],
        },
        "expected_sites": {
            "DB00945": "羧基与乙酰酯（抗血小板活性相关）",
            "DB00682": "4-羟基香豆素核 / 内酯羰基（抗凝活性相关）",
        },
    },
    frozenset({"DB00316", "DB00682"}): {
        "pair_name": "对乙酰氨基酚–华法林",
        "clinical_effect": "高剂量下可能增强抗凝、升高INR",
        "mechanism": "对乙酰氨基酚代谢可能干扰维生素K循环；华法林仍作用于VKOR。",
        "expected_groups": {
            "DB00316": ["酚羟基", "酰胺", "芳环(疏水)"],
            "DB00682": ["香豆素内酯", "酚羟基"],
        },
        "expected_sites": {
            "DB00316": "酚羟基与乙酰胺",
            "DB00682": "4-羟基香豆素核",
        },
    },
    frozenset({"DB00758", "DB00682"}): {
        "pair_name": "氯吡格雷–华法林",
        "clinical_effect": "抗血小板+抗凝，出血风险叠加",
        "mechanism": "氯吡格雷经CYP2C19活化后抑制P2Y12；华法林抑制VKOR，二者作用通路不同但终点都影响止血。",
        "expected_groups": {
            "DB00758": ["酯", "卤素", "杂环"],
            "DB00682": ["香豆素内酯", "酚羟基"],
        },
        "expected_sites": {
            "DB00758": "噻吩并吡啶杂环与甲酯",
            "DB00682": "4-羟基香豆素核",
        },
    },
    frozenset({"DB01076", "DB00682"}): {
        "pair_name": "阿托伐他汀–华法林",
        "clinical_effect": "可能升高华法林抗凝强度",
        "mechanism": "竞争CYP3A4/蛋白结合，改变华法林游离浓度；华法林药效团仍是香豆素核。",
        "expected_groups": {
            "DB01076": ["羧酸(酸性中心)", "芳环(疏水)", "杂环"],
            "DB00682": ["香豆素内酯", "酚羟基"],
        },
        "expected_sites": {
            "DB01076": "戊酸侧链与芳环/吡咯",
            "DB00682": "4-羟基香豆素核",
        },
    },
    frozenset({"DB00213", "DB00682"}): {
        "pair_name": "泮托拉唑–华法林",
        "clinical_effect": "PPI可能轻度影响华法林代谢",
        "mechanism": "泮托拉唑经CYP2C19/CYP3A4代谢，可能改变S-华法林代谢；华法林仍作用于VKOR。",
        "expected_groups": {
            "DB00213": ["杂环", "芳环(疏水)"],
            "DB00682": ["香豆素内酯", "酚羟基"],
        },
        "expected_sites": {
            "DB00213": "苯并咪唑与吡啶",
            "DB00682": "4-羟基香豆素核",
        },
    },
    frozenset({"DB01118", "DB00682"}): {
        "pair_name": "胺碘酮–华法林",
        "clinical_effect": "抗凝增强、INR升高",
        "mechanism": "胺碘酮抑制CYP2C9等，减少S-华法林清除；华法林仍作用于VKOR。",
        "expected_groups": {
            "DB01118": ["卤素", "酮", "芳环(疏水)"],
            "DB00682": ["香豆素内酯", "酚羟基"],
        },
        "expected_sites": {
            "DB01118": "碘代苯甲醚与酮",
            "DB00682": "4-羟基香豆素核",
        },
    },
    frozenset({"DB00472", "DB00682"}): {
        "pair_name": "氟西汀–华法林",
        "clinical_effect": "可能增强抗凝",
        "mechanism": "氟西汀抑制CYP2C9/CYP2D6，可能升高华法林浓度。",
        "expected_groups": {
            "DB00472": ["芳环(疏水)", "卤素"],
            "DB00682": ["香豆素内酯", "酚羟基"],
        },
        "expected_sites": {
            "DB00472": "三氟甲基苯氧侧链",
            "DB00682": "4-羟基香豆素核",
        },
    },
    frozenset({"DB00338", "DB00682"}): {
        "pair_name": "奥美拉唑–华法林",
        "clinical_effect": "PPI可能轻度影响华法林代谢",
        "mechanism": "奥美拉唑经CYP2C19代谢，可能改变S-华法林代谢。",
        "expected_groups": {
            "DB00338": ["杂环", "芳环(疏水)"],
            "DB00682": ["香豆素内酯", "酚羟基"],
        },
        "expected_sites": {
            "DB00338": "苯并咪唑与吡啶",
            "DB00682": "4-羟基香豆素核",
        },
    },
    frozenset({"DB00537", "DB00682"}): {
        "pair_name": "环丙沙星–华法林",
        "clinical_effect": "抗凝增强",
        "mechanism": "喹诺酮可能抑制华法林代谢或改变肠道菌群维生素K。",
        "expected_groups": {
            "DB00537": ["卤素", "杂环", "羧酸(酸性中心)"],
            "DB00682": ["香豆素内酯", "酚羟基"],
        },
        "expected_sites": {
            "DB00537": "氟喹诺酮核与羧酸",
            "DB00682": "4-羟基香豆素核",
        },
    },
    frozenset({"DB00641", "DB00682"}): {
        "pair_name": "辛伐他汀–华法林",
        "clinical_effect": "可能升高抗凝强度",
        "mechanism": "竞争CYP3A4/蛋白结合；华法林药效团仍是香豆素核。",
        "expected_groups": {
            "DB00641": ["醇羟基", "酯"],
            "DB00682": ["香豆素内酯", "酚羟基"],
        },
        "expected_sites": {
            "DB00641": "内酯/开环酸侧链",
            "DB00682": "4-羟基香豆素核",
        },
    },
    frozenset({"DB00641", "DB01118"}): {
        "pair_name": "辛伐他汀–胺碘酮",
        "clinical_effect": "他汀暴露升高、肌病风险",
        "mechanism": "胺碘酮抑制CYP3A4，减少辛伐他汀代谢。",
        "expected_groups": {
            "DB00641": ["醇羟基", "酯"],
            "DB01118": ["卤素", "酮", "芳环(疏水)"],
        },
        "expected_sites": {
            "DB00641": "内酯侧链",
            "DB01118": "碘代芳酮",
        },
    },
    frozenset({"DB00381", "DB00641"}): {
        "pair_name": "氨氯地平–辛伐他汀",
        "clinical_effect": "他汀暴露升高",
        "mechanism": "氨氯地平抑制CYP3A4介导的辛伐他汀代谢。",
        "expected_groups": {
            "DB00381": ["酯", "卤素", "杂环"],
            "DB00641": ["醇羟基", "酯"],
        },
        "expected_sites": {
            "DB00381": "二氢吡啶酯",
            "DB00641": "内酯侧链",
        },
    },
}


def _to_numpy(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _normalize(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if arr is None or arr.size == 0:
        return arr
    a = np.asarray(arr, dtype=np.float64)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo < 1e-8:
        return np.full_like(a, 0.5)
    return (a - lo) / (hi - lo)


def _atom_scores(pair: Dict, side: str, keys: List[str]) -> np.ndarray:
    parts = [_normalize(_to_numpy(pair.get(k))) for k in keys]
    parts = [p for p in parts if p is not None and p.size > 0]
    if not parts:
        return np.zeros(0, dtype=np.float64)
    n = min(p.size for p in parts)
    stacked = np.stack([p[:n] for p in parts], axis=0)
    return _normalize(stacked.mean(axis=0))


def _combine_atom_scores(pair: Dict, side: str) -> np.ndarray:
    return _atom_scores(pair, side, [f"intra_{side}_atom", f"inter_{side}_atom", f"pool_{side}_atom"])


def _intra_atom_scores(pair: Dict, side: str) -> np.ndarray:
    return _atom_scores(pair, side, [f"intra_{side}_atom"])


def _inter_atom_scores(pair: Dict, side: str) -> np.ndarray:
    return _atom_scores(pair, side, [f"inter_{side}_atom"])


def _substruct_from_atoms(atom_scores: np.ndarray, a2s) -> np.ndarray:
    mat = _to_numpy(a2s)
    if mat is None or atom_scores.size == 0:
        return np.zeros(0, dtype=np.float64)
    n = min(atom_scores.size, mat.shape[0])
    contrib = mat[:n].T @ atom_scores[:n]
    counts = mat[:n].sum(axis=0).clip(min=1e-6)
    return contrib / counts


def _iter_smarts(smarts) -> List[str]:
    if isinstance(smarts, (list, tuple)):
        return [str(s) for s in smarts]
    return [str(smarts)]


def _fg_atom_hits(mol: Chem.Mol) -> Dict[str, List[int]]:
    hits = {}
    if mol is None:
        return hits
    for name, smarts in PHARMACOPHORE_SMARTS.items():
        atoms = set()
        for pattern in _iter_smarts(smarts):
            q = Chem.MolFromSmarts(pattern)
            if q is None:
                continue
            for match in mol.GetSubstructMatches(q):
                atoms.update(int(i) for i in match)
        if atoms:
            hits[name] = sorted(atoms)
    return hits


def _lookup_mechanism(h: str, t: str) -> Optional[Dict]:
    return KNOWN_MECHANISMS.get(frozenset({h, t}))


def _align_mechanism(
    drug_id: str,
    fg_rows: List[Dict],
    mech: Optional[Dict],
    top_k: int = 3,
) -> Dict:
    expected = list((mech or {}).get("expected_groups", {}).get(drug_id, []))
    by_mean = [r["functional_group"] for r in fg_rows[:top_k]]
    by_max = [
        r["functional_group"]
        for r in sorted(fg_rows, key=lambda r: r["max_attention"], reverse=True)[:top_k]
    ]
    # 大骨架（香豆素核）按均值会被稀释；max 与 mean 的 Top-k 并集更公允。
    predicted = list(dict.fromkeys(by_mean + by_max))
    hit = [g for g in expected if g in predicted]
    miss = [g for g in expected if g not in predicted]
    extra = [g for g in by_mean if g not in expected]
    return {
        "expected_groups": expected,
        "top_predicted_groups": by_mean,
        "top_predicted_groups_by_max": by_max,
        "hit_groups": hit,
        "miss_groups": miss,
        "extra_groups": extra,
        "alignment_recall": (len(hit) / len(expected)) if expected else None,
        "expected_site": (mech or {}).get("expected_sites", {}).get(drug_id),
    }


def _granularity_compare(atom_scores: np.ndarray, sub_scores: Optional[np.ndarray]) -> Dict:
    atom_std = float(np.std(atom_scores)) if atom_scores.size else 0.0
    sub_std = float(np.std(sub_scores)) if sub_scores is not None and sub_scores.size else 0.0
    if atom_std > sub_std + 1e-6:
        dominant = "atom"
    elif sub_std > atom_std + 1e-6:
        dominant = "substructure"
    else:
        dominant = "balanced"
    return {
        "atom_attention_std": atom_std,
        "substructure_attention_std": sub_std,
        "dominant_granularity": dominant,
        "note": "标准差更大表示该粒度上的注意力更集中，用于开题的跨粒度差异分析。",
    }


def _fg_scores(atom_scores: np.ndarray, fg_hits: Dict[str, List[int]]) -> List[Dict]:
    rows = []
    for name, idxs in fg_hits.items():
        valid = [i for i in idxs if i < atom_scores.size]
        if not valid:
            continue
        rows.append({
            "functional_group": name,
            "n_atoms": len(valid),
            "mean_attention": float(np.mean(atom_scores[valid])),
            "max_attention": float(np.max(atom_scores[valid])),
            "atom_indices": valid,
        })
    rows.sort(key=lambda r: r["mean_attention"], reverse=True)
    return rows


def _score_to_color(score: float) -> Tuple[float, float, float]:
    # 低=浅黄，高=红
    s = float(np.clip(score, 0.0, 1.0))
    return (1.0, 1.0 - 0.75 * s, 0.15 * (1.0 - s))


def _draw_heatmap(mol: Chem.Mol, atom_scores: np.ndarray, out_png: str, title: str) -> None:
    if mol is None:
        return
    rdDepictor.Compute2DCoords(mol)
    n = mol.GetNumAtoms()
    scores = np.zeros(n, dtype=np.float64)
    if atom_scores.size:
        m = min(n, atom_scores.size)
        scores[:m] = atom_scores[:m]
    highlight = {i: _score_to_color(scores[i]) for i in range(n)}
    drawer = rdMolDraw2D.MolDraw2DCairo(720, 480)
    drawer.drawOptions().addAtomIndices = True
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(range(n)),
        highlightAtomColors=highlight,
        legend=title,
    )
    drawer.FinishDrawing()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    with open(out_png, "wb") as f:
        f.write(drawer.GetDrawingText())


def _en_label(name: str) -> str:
    return PHARMACOPHORE_LABEL_EN.get(name, name)


def _bar_chart(labels: List[str], values: List[float], out_png: str, title: str, xlabel: str) -> None:
    if not labels:
        return
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(8, max(3.0, 0.35 * len(labels) + 1.5)))
    y = np.arange(len(labels))
    ax.barh(y, values, color="#c0392b", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xlim(0.0, 1.05)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _load_smiles_map(ddi_loader: DDIDataLoader) -> Dict[str, str]:
    smiles = {}
    for drug_id, data in ddi_loader.drug_graph_dict.items():
        if drug_id == "__metadata__" or not isinstance(data, dict):
            continue
        if data.get("smiles"):
            smiles[drug_id] = str(data["smiles"])
    if ddi_loader.drug_smiles is not None:
        for _, row in ddi_loader.drug_smiles.iterrows():
            smiles.setdefault(str(row["drug_id"]), str(row["smiles"]))
    return smiles


def _find_pair_relation(csvs: List[str], h: str, t: str) -> Optional[object]:
    for path in csvs:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        exact = df[(df["d1"] == h) & (df["d2"] == t)]
        if len(exact):
            return exact.iloc[0]["type"]
        rev = df[(df["d1"] == t) & (df["d2"] == h)]
        if len(rev):
            return rev.iloc[0]["type"]
    return None


def _pair_split_membership(csvs: List[str], h: str, t: str) -> Dict[str, object]:
    """标出 (h,t) / (t,h) 分别落在 train/val/test 的哪些划分。"""
    names = []
    for path in csvs:
        base = os.path.basename(path).replace(".csv", "")
        names.append((base, path))
    hits = {"forward": [], "reverse": []}
    for split_name, path in names:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, usecols=["d1", "d2"])
        if ((df["d1"] == h) & (df["d2"] == t)).any():
            hits["forward"].append(split_name)
        if ((df["d1"] == t) & (df["d2"] == h)).any():
            hits["reverse"].append(split_name)
    all_splits = sorted(set(hits["forward"] + hits["reverse"]))
    return {
        "forward_splits": hits["forward"],
        "reverse_splits": hits["reverse"],
        "splits": all_splits,
        "in_test": "test" in all_splits,
        "in_train": "train" in all_splits,
        "in_val": "val" in all_splits,
    }


def _pick_fallback_cases(csvs: List[str], n: int = 6) -> List[Tuple[str, str, str]]:
    rows = []
    for path in csvs:
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        rows.append(df[["d1", "d2", "type"]])
    if not rows:
        return []
    df = pd.concat(rows, ignore_index=True).drop_duplicates(subset=["d1", "d2"])
    picked = df.head(n)
    cases = []
    for _, r in picked.iterrows():
        cases.append((str(r["d1"]), str(r["d2"]), f"{r['d1']}_{r['d2']}"))
    return cases


def _build_model(config, ddi_loader, device):
    ckpt = torch.load(config["model_path"], map_location=device, weights_only=False)
    config = _merge_eval_config_from_checkpoint(config, ckpt)
    rel_n = max(1, int(getattr(ddi_loader, "rel_total", 0) or 0))
    model = DGN_DDI(
        hidden_dim=config["hidden_dim"],
        kge_dim=config["kge_dim"],
        rel_total=rel_n,
        heads_out_feat_params=config["heads_out_feat_params"],
        blocks_params=config["blocks_params"],
        drug_graph_dict=ddi_loader.drug_graph_dict,
        int_to_drug_id=ddi_loader.int_to_drug_id,
        device=device,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config, ckpt


def _score_logit(dataset: DrugDataset, model: DGN_DDI, device: torch.device, h: str, t: str, rel):
    packed = dataset.collate_positives_only([(h, t, rel)])
    if packed is None or packed[0] is None:
        return None
    pos_h, pos_t, pos_r, pos_bg = packed
    pos_h = pos_h.to(device)
    pos_t = pos_t.to(device)
    pos_r = pos_r.to(device)
    pos_bg = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in pos_bg.items()}
    with torch.no_grad():
        scores = model(pos_h, pos_t, pos_r, pos_bg)
    return float(scores[0].item())


def explain_pair(
    dataset: DrugDataset,
    model: DGN_DDI,
    device: torch.device,
    h: str,
    t: str,
    rel,
    smiles_map: Dict[str, str],
    out_dir: str,
    case_name: str,
    csvs: Optional[List[str]] = None,
) -> Optional[Dict]:
    packed = dataset.collate_positives_only([(h, t, rel)])
    if packed is None or packed[0] is None:
        logger.warning("Skip %s-%s: cannot build batch", h, t)
        return None
    pos_h, pos_t, pos_r, pos_bg = packed
    pos_h = pos_h.to(device)
    pos_t = pos_t.to(device)
    pos_r = pos_r.to(device)
    pos_bg = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in pos_bg.items()}

    with torch.no_grad():
        scores, explain = model.forward_with_weight(pos_h, pos_t, pos_r, pos_bg)
    score = float(scores[0].item()) if scores.numel() else float("nan")
    prob = float(1.0 / (1.0 + np.exp(-np.clip(score, -50, 50))))
    pair = explain["pairs"][0]
    h_intra = _intra_atom_scores(pair, "h")
    t_intra = _intra_atom_scores(pair, "t")
    h_inter = _inter_atom_scores(pair, "h")
    t_inter = _inter_atom_scores(pair, "t")
    h_atom = _combine_atom_scores(pair, "h")
    t_atom = _combine_atom_scores(pair, "t")
    h_sub = _normalize(_substruct_from_atoms(h_atom, pair.get("atom2substruct_h")))
    t_sub = _normalize(_substruct_from_atoms(t_atom, pair.get("atom2substruct_t")))

    h_mol = Chem.MolFromSmiles(smiles_map.get(h, "")) if smiles_map.get(h) else None
    t_mol = Chem.MolFromSmiles(smiles_map.get(t, "")) if smiles_map.get(t) else None
    h_fg = _fg_scores(h_atom, _fg_atom_hits(h_mol))
    t_fg = _fg_scores(t_atom, _fg_atom_hits(t_mol))
    mech = _lookup_mechanism(h, t)
    h_align = _align_mechanism(h, h_fg, mech)
    t_align = _align_mechanism(t, t_fg, mech)

    case_dir = os.path.join(out_dir, case_name)
    os.makedirs(case_dir, exist_ok=True)
    # 开题：图内=药效团，图间=接触位点，合成图仅作对照。图注用英文以免缺字。
    _draw_heatmap(h_mol, h_intra, os.path.join(case_dir, f"{h}_intra_heatmap.png"), f"{h} IntraGraph (pharmacophore)")
    _draw_heatmap(t_mol, t_intra, os.path.join(case_dir, f"{t}_intra_heatmap.png"), f"{t} IntraGraph (pharmacophore)")
    _draw_heatmap(h_mol, h_inter, os.path.join(case_dir, f"{h}_inter_heatmap.png"), f"{h} InterGraph (contact site)")
    _draw_heatmap(t_mol, t_inter, os.path.join(case_dir, f"{t}_inter_heatmap.png"), f"{t} InterGraph (contact site)")
    _draw_heatmap(h_mol, h_atom, os.path.join(case_dir, f"{h}_atom_heatmap.png"), f"{h} combined attention")
    _draw_heatmap(t_mol, t_atom, os.path.join(case_dir, f"{t}_atom_heatmap.png"), f"{t} combined attention")

    if h_sub is not None and h_sub.size:
        labels = [f"sub-{i}" for i in range(h_sub.size)]
        _bar_chart(labels, [float(v) for v in h_sub], os.path.join(case_dir, f"{h}_substruct.png"),
                   f"{h} substructure contribution", "Normalized attention")
    if t_sub is not None and t_sub.size:
        labels = [f"sub-{i}" for i in range(t_sub.size)]
        _bar_chart(labels, [float(v) for v in t_sub], os.path.join(case_dir, f"{t}_substruct.png"),
                   f"{t} substructure contribution", "Normalized attention")

    if h_fg:
        _bar_chart(
            [_en_label(r["functional_group"]) for r in h_fg],
            [r["mean_attention"] for r in h_fg],
            os.path.join(case_dir, f"{h}_functional_groups.png"),
            f"{h} pharmacophore attention",
            "Mean atom attention",
        )
    if t_fg:
        _bar_chart(
            [_en_label(r["functional_group"]) for r in t_fg],
            [r["mean_attention"] for r in t_fg],
            os.path.join(case_dir, f"{t}_functional_groups.png"),
            f"{t} pharmacophore attention",
            "Mean atom attention",
        )

    block_w = _to_numpy(explain.get("block_weights"))
    recalls = [x for x in (h_align.get("alignment_recall"), t_align.get("alignment_recall")) if x is not None]
    split_info = _pair_split_membership(csvs or [], h, t) if csvs else {}
    reverse_logit = _score_logit(dataset, model, device, t, h, rel)
    record = {
        "case_name": case_name,
        "head": h,
        "tail": t,
        "relation": str(rel),
        "logit": score,
        "reverse_logit": reverse_logit,
        "probability": prob,
        "prediction_positive": bool(score > 0),
        "split": split_info,
        "block_weights": [float(x) for x in block_w.reshape(-1)] if block_w is not None else [],
        "head_top_atoms": np.argsort(-h_atom)[:8].tolist() if h_atom.size else [],
        "tail_top_atoms": np.argsort(-t_atom)[:8].tolist() if t_atom.size else [],
        "head_pharmacophores": h_fg,
        "tail_pharmacophores": t_fg,
        "head_granularity": _granularity_compare(h_atom, h_sub),
        "tail_granularity": _granularity_compare(t_atom, t_sub),
        "known_mechanism": mech,
        "head_mechanism_alignment": h_align,
        "tail_mechanism_alignment": t_align,
        "pair_alignment_recall": (sum(recalls) / len(recalls)) if recalls else None,
        "note": (
            "IntraGraph 对应开题“药物内部核心药效团”；"
            "InterGraph 对应“两药接触关键位点”。"
            "alignment_recall 是已知机制基团出现在注意力 Top-3 的比例，不是预测准确率。"
            "logit 方向随 (head,tail) 有向关系变化，必须同时报两个方向。"
            "论文插图只用 prediction_positive=True 且机制已知的对；阿司匹林–华法林两个方向都要展示。"
        ),
    }
    with open(os.path.join(case_dir, "case_summary.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    logger.info(
        "Wrote case %s | logit=%.4f prob=%.4f align=%.2f",
        case_name, score, prob, record["pair_alignment_recall"] or -1.0,
    )
    return record


def main():
    _setup_console_logging()
    parser = argparse.ArgumentParser(description="DGN-DDI interpretability (paper_lock weights)")
    parser.add_argument(
        "--model-path",
        default=os.path.join(_SCRIPT_DIR, "checkpoints", "paper_lock", "best_model.pth"),
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(_SCRIPT_DIR, "interpret_results"),
    )
    parser.add_argument(
        "--pairs",
        default="",
        help="Extra pairs as h,t,rel;h2,t2,rel2  (rel optional)",
    )
    parser.add_argument("--max-fallback", type=int, default=6)
    args = parser.parse_args()

    config = get_test_config(profile="full")
    config["model_path"] = args.model_path
    if not os.path.exists(config["model_path"]):
        fallback = os.path.join(_SCRIPT_DIR, "checkpoints", "best_model.pth")
        logger.warning("paper_lock model missing, fallback to %s", fallback)
        config["model_path"] = fallback

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DGN-DDI] 使用设备: {device}", flush=True)
    print("[DGN-DDI] 正在加载药物图 / SMILES（pkl 较大，可能再等 1–2 分钟）...", flush=True)
    ddi_loader = DDIDataLoader()
    print("[DGN-DDI] 正在构建测试 Dataset...", flush=True)
    dataset = DrugDataset(config["test_csv"], ddi_loader, neg_ent=0, shuffle=False)
    print("[DGN-DDI] 正在加载模型权重并预载药物张量到 GPU（这一步最久）...", flush=True)
    model, config, _ckpt = _build_model(config, ddi_loader, device)
    print("[DGN-DDI] 模型已就绪，开始选案例并画图。", flush=True)
    smiles_map = _load_smiles_map(ddi_loader)

    bank = os.path.join(_SCRIPT_DIR, "drugbank")
    csvs = [
        os.path.join(bank, "fold0", "train.csv"),
        os.path.join(bank, "fold0", "val.csv"),
        os.path.join(bank, "fold0", "test.csv"),
    ]

    cases: List[Tuple[str, str, str, object]] = []
    for h, t, name in DEFAULT_CASES:
        if h not in ddi_loader.drug_graph_dict or t not in ddi_loader.drug_graph_dict:
            continue
        rel = _find_pair_relation(csvs, h, t)
        if rel is None:
            continue
        cases.append((h, t, name, rel))

    if args.pairs.strip():
        for item in args.pairs.split(";"):
            parts = [p.strip() for p in item.split(",") if p.strip()]
            if len(parts) < 2:
                continue
            h, t = parts[0], parts[1]
            rel = parts[2] if len(parts) > 2 else _find_pair_relation(csvs, h, t)
            if rel is None:
                logger.warning("No relation found for %s-%s, skip", h, t)
                continue
            cases.append((h, t, f"{h}_{t}", rel))

    if len(cases) < 3:
        for h, t, name in _pick_fallback_cases(csvs, n=args.max_fallback):
            rel = _find_pair_relation(csvs, h, t)
            if rel is None:
                continue
            cases.append((h, t, name, rel))

    seen = set()
    unique_cases = []
    for h, t, name, rel in cases:
        key = (h, t, str(rel))
        if key in seen:
            continue
        seen.add(key)
        unique_cases.append((h, t, name, rel))

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[DGN-DDI] 共 {len(unique_cases)} 个案例，输出目录: {args.out_dir}", flush=True)
    summaries = []
    for idx, (h, t, name, rel) in enumerate(unique_cases, start=1):
        print(f"[DGN-DDI] ({idx}/{len(unique_cases)}) 解释 {h} - {t} ({name}) ...", flush=True)
        rec = explain_pair(dataset, model, device, h, t, rel, smiles_map, args.out_dir, name, csvs=csvs)
        if rec:
            summaries.append(rec)

    index_path = os.path.join(args.out_dir, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({"n_cases": len(summaries), "cases": summaries}, f, ensure_ascii=False, indent=2)

    align_rows = []
    for rec in summaries:
        mech = rec.get("known_mechanism") or {}
        align_rows.append({
            "case_name": rec.get("case_name"),
            "pair_name": mech.get("pair_name", ""),
            "head": rec.get("head"),
            "tail": rec.get("tail"),
            "logit": rec.get("logit"),
            "reverse_logit": rec.get("reverse_logit"),
            "prediction_positive": rec.get("prediction_positive"),
            "in_test": (rec.get("split") or {}).get("in_test"),
            "in_train": (rec.get("split") or {}).get("in_train"),
            "splits": "|".join((rec.get("split") or {}).get("splits") or []),
            "clinical_effect": mech.get("clinical_effect", ""),
            "mechanism": mech.get("mechanism", ""),
            "head_top_groups": "|".join(rec.get("head_mechanism_alignment", {}).get("top_predicted_groups", [])),
            "tail_top_groups": "|".join(rec.get("tail_mechanism_alignment", {}).get("top_predicted_groups", [])),
            "head_hit": "|".join(rec.get("head_mechanism_alignment", {}).get("hit_groups", [])),
            "tail_hit": "|".join(rec.get("tail_mechanism_alignment", {}).get("hit_groups", [])),
            "head_miss": "|".join(rec.get("head_mechanism_alignment", {}).get("miss_groups", [])),
            "tail_miss": "|".join(rec.get("tail_mechanism_alignment", {}).get("miss_groups", [])),
            "pair_alignment_recall": rec.get("pair_alignment_recall"),
            "head_dominant_granularity": rec.get("head_granularity", {}).get("dominant_granularity"),
            "tail_dominant_granularity": rec.get("tail_granularity", {}).get("dominant_granularity"),
        })
    csv_path = os.path.join(args.out_dir, "mechanism_alignment.csv")
    pd.DataFrame(align_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    recalls = [r["pair_alignment_recall"] for r in align_rows if r.get("pair_alignment_recall") is not None]
    summary_path = os.path.join(args.out_dir, "innovation3_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_cases": len(summaries),
            "n_with_known_mechanism": sum(1 for r in align_rows if r.get("pair_name")),
            "mean_alignment_recall": (sum(recalls) / len(recalls)) if recalls else None,
            "howto_write_thesis": [
                "用 IntraGraph 热力图写“药物内部核心药效团”",
                "用 InterGraph 热力图写“两药接触关键位点”",
                "用 mechanism_alignment.csv 写“注意力基团 vs 已知机制”对照表",
                "alignment_recall 只说明基团对上了，不代表该方向预测为正样本",
                "每个案例必须同时看两个方向的 logit，禁止只挑 logit>0",
                "阿司匹林–华法林请同时看 aspirin_warfarin 与 warfarin_aspirin 两个方向",
                "论文插图只用 prediction_positive=True 且机制已知的对，并标明 train/val/test",
                "论文插图请用英文标签的 PNG（已避免中文字体缺字）",
            ],
        }, f, ensure_ascii=False, indent=2)
    print(f"[DGN-DDI] 机制对照表: {csv_path}", flush=True)
    print(f"[DGN-DDI] 创新点3摘要: {summary_path}", flush=True)
    logger.info("Interpretability done: %d cases -> %s", len(summaries), args.out_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[DGN-DDI] 运行失败，完整报错如下：", flush=True)
        traceback.print_exc()
        sys.exit(1)
