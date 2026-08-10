import pandas as pd
import torch
import torch.nn.functional as F
import pickle
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import BRICS
from torch_geometric.data import Data


def atom_feature_to_66dim(atom: Chem.Atom) -> torch.Tensor:
    features = []
    atomic_num = min(atom.GetAtomicNum(), 53)
    features.append(F.one_hot(torch.tensor(atomic_num - 1), num_classes=53).float())
    features.append(torch.tensor([atom.GetDegree() / 10.0]))
    features.append(torch.tensor([atom.GetImplicitValence() / 10.0]))
    features.append(torch.tensor([atom.GetFormalCharge() / 5.0]))
    features.append(torch.tensor([atom.GetNumRadicalElectrons() / 5.0]))
    hybrid = min(int(atom.GetHybridization()), 6)
    features.append(F.one_hot(torch.tensor(hybrid), num_classes=7).float())
    features.append(torch.tensor([int(atom.GetIsAromatic())]))
    features.append(torch.tensor([atom.GetTotalNumHs() / 10.0]))
    feat = torch.cat(features)
    assert feat.shape[0] == 66
    return feat



def smiles_to_data(smiles: str, label: int = 1, drug_id: str = None, drug_id_to_int=None) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # 检查分子是否包含原子，防止生成空的图
    if mol.GetNumAtoms() == 0:
        return None

    # 提取原子特征
    try:
        x = torch.stack([atom_feature_to_66dim(atom) for atom in mol.GetAtoms()])
    except Exception as e:
        # 捕获特征提取过程中可能发生的错误
        print(f"Error extracting features for drug {drug_id}: {e}")
        return None

    # 增加对NaN和Inf的检查
    if torch.isnan(x).any() or torch.isinf(x).any():
        print(f"Found NaN or Inf in atom features for drug {drug_id}")
        return None

    edge_index = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]
    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # 确保edge_index不是空的，否则后续操作可能会失败
    if edge_index.size(1) == 0 and x.size(0) > 1:
        # 如果一个图有多个节点但没有边，这可能是个问题，但模型可能能处理
        pass

    y = torch.full((x.size(0),), label, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)

    if drug_id is not None and drug_id_to_int is not None:
        data.drug_id = torch.tensor([drug_id_to_int[drug_id]], dtype=torch.long)

    return data


# 定义常见功能基团的SMARTS模式
FUNCTIONAL_GROUPS_SMARTS = {
    "Acid": "C(=O)[O;H,-]",  # 羧酸
    "Alcohol": "[OX2H]",  # 醇
    "Aldehyde": "[CX3H1](=O)",  # 醛
    "Amide": "C(=O)[NH2,NH1,N]",  # 酰胺
    "Amine": "[NX3;H2,H1,H0;!$(NC=O)]",  # 胺（伯、仲、叔，不包括酰胺）
    "AromaticAmine": "c[NX3;H2,H1,H0;!$(NC=O)]",  # 芳香胺
    "Ester": "C(=O)O[CX4]",  # 酯
    "Ether": "[OX2]([CX4])[CX4]",  # 醚
    "Ketone": "[CX3](=O)[#6]",  # 酮
    "Nitrile": "[NX1]#[CX2]",  # 腈
    "Phenol": "c[OX2H]",  # 苯酚
    "Thiol": "[SX2H]",  # 硫醇
    "Sulfonamide": "S(=O)(=O)N",  # 磺酰胺
    "Halogen": "[F,Cl,Br,I]",  # 卤素
    "Nitro": "N(=O)[O,-]"  # 硝基
}

# 预编译SMARTS模式以提高效率
PRECOMPUTED_FGROUPS = {name: Chem.MolFromSmarts(smarts) for name, smarts in FUNCTIONAL_GROUPS_SMARTS.items()}


def get_functional_group_features(mol: Chem.Mol, atom_indices: tuple) -> torch.Tensor:
    """为给定的子结构（由其原子索引定义）提取功能基团相关特征"""
    fg_feature_dim = len(FUNCTIONAL_GROUPS_SMARTS) + 1  # +1 用于存储独特功能基团数量

    try:
        sub_mol = Chem.MolFragmentToMol(mol, list(atom_indices), kekulize=True)
    except Exception:
        try:
            sub_mol_smiles = Chem.MolToSmiles(Chem.MolFragmentToMol(mol, list(atom_indices), kekulize=True),
                                              kekuleSmiles=True)
            sub_mol = Chem.MolFromSmiles(sub_mol_smiles)
        except Exception:
            sub_mol = None

    if sub_mol is None:
        return torch.zeros(fg_feature_dim)

    # 功能基团独热编码特征
    fg_present = [0] * len(FUNCTIONAL_GROUPS_SMARTS)
    unique_fg_count = 0

    for i, (fg_name, fg_mol) in enumerate(PRECOMPUTED_FGROUPS.items()):
        if sub_mol.HasSubstructMatch(fg_mol):
            fg_present[i] = 1
            unique_fg_count += 1

    fg_feats = torch.tensor(fg_present, dtype=torch.float)

    # 独有功能基团数量 (归一化)
    num_unique_fg_feat = torch.tensor([unique_fg_count / len(FUNCTIONAL_GROUPS_SMARTS)])

    return torch.cat([fg_feats, num_unique_fg_feat])


def extract_substructure_data(smiles: str, drug_id: str = None, mol=None, drug_id_to_int=None):
    if mol is None:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None, 0

    num_atoms = mol.GetNumAtoms()

    try:
        broken_mol = BRICS.BreakBRICSBonds(mol)
        frags_rdkit = list(Chem.GetMolFrags(broken_mol, asMols=False, sanitizeFrags=False))
        frags = [tuple(sorted(f)) for f in frags_rdkit]
    except Exception:
        frags = []

    if len(frags) == 0:
        frags = [tuple(range(num_atoms))]

    num_substructs = len(frags)

    atom2substruct = torch.zeros((num_atoms, num_substructs), dtype=torch.float)
    for sub_idx, atom_indices in enumerate(frags):
        for atom_idx in atom_indices:
            if atom_idx < num_atoms:
                atom2substruct[atom_idx, sub_idx] = 1.0

    substruct_feats = []
    expected_fg_dim = len(FUNCTIONAL_GROUPS_SMARTS) + 1
    expected_topo_dim = 4
    base_atom_feat_dim = 66
    total_expected_substruct_feat_dim = base_atom_feat_dim + expected_topo_dim + expected_fg_dim

    for atom_indices in frags:
        atom_feats_in_sub = [atom_feature_to_66dim(mol.GetAtomWithIdx(idx)) for idx in atom_indices if idx < num_atoms]
        mean_atom_feat = torch.stack(atom_feats_in_sub).mean(dim=0) if atom_feats_in_sub else torch.zeros(
            base_atom_feat_dim)

        sub_mol_for_topo_and_fg = None
        try:
            sub_mol_for_topo_and_fg = Chem.MolFragmentToMol(mol, list(atom_indices), kekulize=True)
            if sub_mol_for_topo_and_fg:
                Chem.SanitizeMol(sub_mol_for_topo_and_fg)
        except Exception:
            sub_mol_for_topo_and_fg = None

        if sub_mol_for_topo_and_fg is None:
            num_bonds_in_sub = 0
            num_rings_in_sub = 0
            is_aromatic_sub = 0
            fg_feats_tensor = torch.zeros(expected_fg_dim)
        else:
            num_bonds_in_sub = sub_mol_for_topo_and_fg.GetNumBonds()
            num_rings_in_sub = sub_mol_for_topo_and_fg.GetNumRings()
            is_aromatic_sub = int(any(atom.GetIsAromatic() for atom in sub_mol_for_topo_and_fg.GetAtoms()))
            fg_feats_tensor = get_functional_group_features(mol, atom_indices)

        topo_feats = torch.tensor([
            len(atom_indices) / 50.0,
            num_bonds_in_sub / 50.0,
            num_rings_in_sub / 10.0,
            float(is_aromatic_sub)
        ])

        combined_feat = torch.cat([mean_atom_feat, topo_feats, fg_feats_tensor])
        if combined_feat.shape[0] != total_expected_substruct_feat_dim:
            combined_feat = torch.zeros(total_expected_substruct_feat_dim)

        substruct_feats.append(combined_feat)

    x_substruct = torch.stack(substruct_feats) if substruct_feats else torch.empty(
        (0, total_expected_substruct_feat_dim), dtype=torch.float)

    atom_to_substruct_map = {}
    for sub_idx, atom_indices in enumerate(frags):
        for atom_idx in atom_indices:
            atom_to_substruct_map.setdefault(atom_idx, set()).add(sub_idx)

    edge_index_atom = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i < num_atoms and j < num_atoms:
            edge_index_atom.append((i, j))
            edge_index_atom.append((j, i))

    substruct_edge_set = set()
    for (i, j) in edge_index_atom:
        subs_i = atom_to_substruct_map.get(i, set())
        subs_j = atom_to_substruct_map.get(j, set())
        for si in subs_i:
            for sj in subs_j:
                if si != sj:
                    substruct_edge_set.add(tuple(sorted((si, sj))))

    if len(substruct_edge_set) == 0:
        edge_index_substruct = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index_substruct = torch.tensor(list(substruct_edge_set), dtype=torch.long).t().contiguous()

    y_substruct = torch.full((x_substruct.size(0),), 2, dtype=torch.long)
    substruct_data = Data(x=x_substruct, edge_index=edge_index_substruct, y=y_substruct)

    if drug_id is not None and drug_id_to_int is not None:
        substruct_data.drug_id = torch.tensor([drug_id_to_int[drug_id]], dtype=torch.long)

    return substruct_data, atom2substruct, num_substructs


def summarize_graph_stats(drug_graph_dict):
    """统计并打印图数据的统计信息"""
    # 排除元数据进行统计
    drug_data = {k: v for k, v in drug_graph_dict.items() if k != '__metadata__'}
    total_drugs = len(drug_data)
    if total_drugs == 0:
        print("No drugs processed for statistics.")
        return

    total_atoms, total_atom_edges = 0, 0
    total_substructs, total_substruct_edges = 0, 0

    for drug_id, data in drug_data.items():
        atom_x = data['atom'].x
        atom_edge_index = data['atom'].edge_index
        substruct_x = data['substruct'].x
        substruct_edge_index = data['substruct'].edge_index

        total_atoms += atom_x.size(0)
        if atom_edge_index.dim() == 2 and atom_edge_index.size(0) == 2:
            total_atom_edges += atom_edge_index.size(1) // 2

        total_substructs += substruct_x.size(0)
        if substruct_edge_index.dim() == 2 and substruct_edge_index.size(0) == 2:
            total_substruct_edges += substruct_edge_index.size(1) // 2

    print(f"\n=== Graph Statistics ===")
    print(f"Total drugs processed: {total_drugs}")
    print(f"Average atoms per drug: {total_atoms / total_drugs:.2f}")
    print(f"Average atom edges per drug: {total_atom_edges / total_drugs:.2f}")
    print(f"Average substructures per drug: {total_substructs / total_drugs:.2f}")
    print(f"Average substructure edges per drug: {total_substruct_edges / total_drugs:.2f}")

    # 验证drug_id是否正确添加
    if total_drugs > 0:
        sample_drug_id = next(iter(drug_data.keys()))
        sample_data = drug_data[sample_drug_id]

        atom_has_drug_id = hasattr(sample_data['atom'], 'drug_id')
        substruct_has_drug_id = hasattr(sample_data['substruct'], 'drug_id')

        print(f"\n=== Drug ID Integration Status ===")
        print(f"Atom graphs contain drug_id: {atom_has_drug_id}")
        print(f"Substructure graphs contain drug_id: {substruct_has_drug_id}")

        if atom_has_drug_id:
            print(f"Sample atom graph drug_id: {sample_data['atom'].drug_id}")
        if substruct_has_drug_id:
            print(f"Sample substruct graph drug_id: {sample_data['substruct'].drug_id}")


def validate_data_integrity(drug_graph_dict):
    """验证生成的数据完整性"""
    print(f"\n=== Data Integrity Validation ===")

    # 排除元数据进行验证
    drug_data = {k: v for k, v in drug_graph_dict.items() if k != '__metadata__'}
    issues_found = []

    for drug_id, data in drug_data.items():
        # 检查必要的键是否存在
        required_keys = ['atom', 'substruct', 'atom2substruct']
        for key in required_keys:
            if key not in data:
                issues_found.append(f"Missing key '{key}' for drug {drug_id}")

        # 检查Data对象的基本属性
        atom_data = data['atom']
        substruct_data = data['substruct']

        if not hasattr(atom_data, 'x') or not hasattr(atom_data, 'edge_index'):
            issues_found.append(f"Atom data missing basic attributes for drug {drug_id}")

        if not hasattr(substruct_data, 'x') or not hasattr(substruct_data, 'edge_index'):
            issues_found.append(f"Substruct data missing basic attributes for drug {drug_id}")

        # 检查映射矩阵维度
        atom2substruct = data['atom2substruct']
        if atom2substruct.size(0) != atom_data.x.size(0):
            issues_found.append(f"Atom2substruct matrix atom dimension mismatch for drug {drug_id}")

        if atom2substruct.size(1) != substruct_data.x.size(0):
            issues_found.append(f"Atom2substruct matrix substruct dimension mismatch for drug {drug_id}")

    if issues_found:
        print("Issues found:")
        for issue in issues_found[:10]:  # 只显示前10个问题
            print(f"  - {issue}")
        if len(issues_found) > 10:
            print(f"  ... and {len(issues_found) - 10} more issues")
    else:
        print("✓ All data integrity checks passed!")


def main(input_csv=None, output_pkl=None):
    import os
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _drugbank_dir = os.path.join(_script_dir, "drugbank")
    if input_csv is None:
        input_csv = os.path.join(_drugbank_dir, "drug_smiles.csv")
    if output_pkl is None:
        output_pkl = os.path.join(_drugbank_dir, "drug_data_dict.pkl")

    print("Loading drug SMILES data...")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} drugs from {input_csv}")

    # 创建药物ID映射
    drug_id_to_int = {did: idx for idx, did in enumerate(df['drug_id'].unique())}
    int_to_drug_id = {v: k for k, v in drug_id_to_int.items()}  # 反向映射

    # 初始化药物图字典，包含元数据键
    drug_graph_dict = {
        '__metadata__': {
            'drug_id_to_int': drug_id_to_int,
            'int_to_drug_id': int_to_drug_id,
            'total_drugs': len(df['drug_id'].unique()),
            'processed_time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    }

    failed_drugs = []

    print(f"\nProcessing {len(df)} drugs...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing drugs"):
        drug_id, smiles = row['drug_id'], row['smiles']

        atom_data = smiles_to_data(smiles, label=1, drug_id=drug_id, drug_id_to_int=drug_id_to_int)
        if atom_data is None:
            failed_drugs.append((drug_id, "SMILES parsing failed or invalid data"))
            continue

        mol_for_substruct = Chem.MolFromSmiles(smiles)
        if mol_for_substruct is None:
            failed_drugs.append((drug_id, "Molecule parsing for substructure failed"))
            continue

        substruct_data, atom2substruct, num_subs = extract_substructure_data(
            smiles, drug_id=drug_id, mol=mol_for_substruct, drug_id_to_int=drug_id_to_int
        )
        if substruct_data is None:
            failed_drugs.append((drug_id, "Substructure extraction failed"))
            continue

        # 将药物数据添加到字典
        drug_graph_dict[drug_id] = {
            'atom': atom_data,
            'substruct': substruct_data,
            'atom2substruct': atom2substruct,
            'smiles': smiles  # 额外存储SMILES字符串，方便后续参考
        }


    drug_graph_dict['__metadata__']['successful_count'] = len(drug_graph_dict) - 1  # 减去元数据条目
    drug_graph_dict['__metadata__']['failed_count'] = len(failed_drugs)

    print(f"\nSaving results to {output_pkl}...")
    with open(output_pkl, 'wb') as f:
        pickle.dump(drug_graph_dict, f)

    print(f"\nSuccessfully processed: {len(drug_graph_dict) - 1} drugs")  # 减去元数据条目
    print(f"Failed: {len(failed_drugs)} drugs")

    # 生成统计信息
    summarize_graph_stats(drug_graph_dict)
    # 验证数据完整性
    validate_data_integrity(drug_graph_dict)


if __name__ == "__main__":
    main()
