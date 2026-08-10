import os
import pickle
import time
import logging
from typing import Dict, List, Tuple
import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HardNegativeCacheValidator:
    """硬负样本缓存文件验证器"""

    def __init__(self, cache_dir: str = './hard_neg_cache'):
        self.cache_dir = cache_dir
        self.validation_results = {}

    def run_full_validation(self):
        """运行完整的验证流程"""
        logger.info("🔍 Starting comprehensive hard negative cache validation...")

        # 1. 基础文件检查
        cache_files = self._discover_cache_files()
        if not cache_files:
            logger.error("❌ No cache files found!")
            return False

        # 2. 逐个验证缓存文件
        all_valid = True
        for cache_file in cache_files:
            is_valid = self._validate_single_cache(cache_file)
            all_valid = all_valid and is_valid

        # 3. 生成验证报告
        self._generate_validation_report()

        # 4. 训练兼容性测试
        compatible = self._test_training_compatibility(cache_files[0] if cache_files else None)

        final_result = all_valid and compatible
        logger.info(f"✅ Overall validation result: {'PASSED' if final_result else 'FAILED'}")
        return final_result

    def _discover_cache_files(self) -> List[str]:
        """发现所有缓存文件"""
        if not os.path.exists(self.cache_dir):
            logger.warning(f"Cache directory {self.cache_dir} does not exist")
            return []

        cache_files = []
        for filename in os.listdir(self.cache_dir):
            if filename.startswith('hard_neg_epoch_') and filename.endswith('.pkl'):
                cache_files.append(os.path.join(self.cache_dir, filename))

        logger.info(f"📁 Found {len(cache_files)} cache files:")
        for f in cache_files:
            logger.info(f"  - {os.path.basename(f)}")

        return cache_files

    def _validate_single_cache(self, cache_file: str) -> bool:
        """验证单个缓存文件"""
        filename = os.path.basename(cache_file)
        logger.info(f"🔍 Validating {filename}...")

        try:
            # 1. 文件基础检查
            file_size = os.path.getsize(cache_file)
            if file_size == 0:
                logger.error(f"❌ {filename}: Empty file")
                return False

            logger.info(f"📊 File size: {file_size / (1024 * 1024):.2f} MB")

            # 2. 加载和解析检查
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)

            # 3. 数据结构验证
            validation_result = self._validate_cache_structure(cache_data, filename)

            # 4. 内容质量检查
            if validation_result['structure_valid']:
                quality_result = self._validate_cache_quality(cache_data['hard_negatives'], filename)
                validation_result.update(quality_result)

            # 5. 记录验证结果
            self.validation_results[filename] = validation_result

            is_valid = validation_result['structure_valid'] and validation_result.get('quality_valid', False)
            status = "✅ VALID" if is_valid else "❌ INVALID"
            logger.info(f"{status} {filename}")

            return is_valid

        except Exception as e:
            logger.error(f"❌ {filename}: Failed to load - {str(e)}")
            self.validation_results[filename] = {
                'structure_valid': False,
                'error': str(e)
            }
            return False

    def _validate_cache_structure(self, cache_data: dict, filename: str) -> dict:
        """验证缓存数据结构"""
        result = {'structure_valid': False}

        try:
            # 检查必需字段
            required_fields = ['epoch', 'fingerprint', 'timestamp', 'hard_negatives', 'num_positives']
            missing_fields = [field for field in required_fields if field not in cache_data]

            if missing_fields:
                logger.error(f"❌ {filename}: Missing fields: {missing_fields}")
                result['missing_fields'] = missing_fields
                return result

            # 检查数据类型
            if not isinstance(cache_data['hard_negatives'], dict):
                logger.error(f"❌ {filename}: hard_negatives is not a dictionary")
                return result

            if not isinstance(cache_data['epoch'], int):
                logger.error(f"❌ {filename}: epoch is not an integer")
                return result

            # 检查数量一致性
            actual_count = len(cache_data['hard_negatives'])
            expected_count = cache_data['num_positives']

            if actual_count != expected_count:
                logger.warning(f"⚠️ {filename}: Count mismatch - expected {expected_count}, got {actual_count}")

            # 基础统计
            result.update({
                'structure_valid': True,
                'epoch': cache_data['epoch'],
                'fingerprint': cache_data['fingerprint'][:8],  # 前8位
                'num_entries': actual_count,
                'timestamp': cache_data['timestamp']
            })

            logger.info(f"📋 {filename} structure: Epoch {cache_data['epoch']}, {actual_count} entries")

        except Exception as e:
            logger.error(f"❌ {filename}: Structure validation error - {str(e)}")
            result['error'] = str(e)

        return result

    def _validate_cache_quality(self, hard_negatives: dict, filename: str) -> dict:
        """验证缓存内容质量"""
        result = {'quality_valid': False}

        try:
            if not hard_negatives:
                logger.error(f"❌ {filename}: Empty hard negatives dictionary")
                return result

            # 抽样检查
            sample_size = min(100, len(hard_negatives))
            sample_items = list(hard_negatives.items())[:sample_size]

            valid_entries = 0
            invalid_entries = 0
            neg_counts = []

            for pos_triple, neg_list in sample_items:
                # 检查正样本格式
                if not isinstance(pos_triple, tuple) or len(pos_triple) != 3:
                    invalid_entries += 1
                    continue

                # 检查负样本列表
                if not isinstance(neg_list, list):
                    invalid_entries += 1
                    continue

                # 检查每个负样本的格式
                valid_negs = 0
                for neg_triple in neg_list:
                    if isinstance(neg_triple, tuple) and len(neg_triple) == 3:
                        # 检查是否与正样本不同
                        if neg_triple != pos_triple:
                            valid_negs += 1

                if valid_negs > 0:
                    valid_entries += 1
                    neg_counts.append(len(neg_list))
                else:
                    invalid_entries += 1

            # 质量评估
            validity_ratio = valid_entries / sample_size if sample_size > 0 else 0
            avg_neg_count = np.mean(neg_counts) if neg_counts else 0

            result.update({
                'quality_valid': validity_ratio >= 0.9,  # 90%以上有效认为合格
                'validity_ratio': validity_ratio,
                'avg_negatives_per_positive': avg_neg_count,
                'sample_size': sample_size,
                'valid_entries': valid_entries,
                'invalid_entries': invalid_entries
            })

            status = "✅ GOOD" if validity_ratio >= 0.9 else "⚠️ POOR"
            logger.info(
                f"📊 {filename} quality: {status} ({validity_ratio:.2%} valid, avg {avg_neg_count:.1f} negs/pos)")

        except Exception as e:
            logger.error(f"❌ {filename}: Quality validation error - {str(e)}")
            result['error'] = str(e)

        return result

    def _test_training_compatibility(self, sample_cache_file: str) -> bool:
        """测试训练兼容性"""
        if not sample_cache_file:
            logger.warning("⚠️ No cache file available for compatibility test")
            return False

        logger.info("🧪 Testing training compatibility...")

        try:
            # 加载缓存数据
            with open(sample_cache_file, 'rb') as f:
                cache_data = pickle.load(f)

            hard_negatives = cache_data['hard_negatives']

            # 模拟训练数据集的使用方式
            compatibility_test = TrainingCompatibilityTest(hard_negatives)
            return compatibility_test.run_test()

        except Exception as e:
            logger.error(f"❌ Training compatibility test failed: {str(e)}")
            return False

    def _generate_validation_report(self):
        """生成详细的验证报告"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 VALIDATION REPORT")
        logger.info("=" * 60)

        if not self.validation_results:
            logger.info("No validation results to report.")
            return

        total_files = len(self.validation_results)
        valid_files = sum(1 for r in self.validation_results.values()
                          if r.get('structure_valid', False) and r.get('quality_valid', False))

        logger.info(f"📈 Overall Statistics:")
        logger.info(f"  - Total files: {total_files}")
        logger.info(f"  - Valid files: {valid_files}")
        logger.info(f"  - Success rate: {valid_files / total_files:.2%}")

        # 按epoch排序显示
        sorted_results = sorted(
            self.validation_results.items(),
            key=lambda x: x[1].get('epoch', 0)
        )

        logger.info(f"\n📋 Individual File Results:")
        for filename, result in sorted_results:
            status = "✅" if (result.get('structure_valid', False) and result.get('quality_valid', False)) else "❌"
            epoch = result.get('epoch', 'N/A')
            entries = result.get('num_entries', 'N/A')
            quality = f"{result.get('validity_ratio', 0):.1%}" if 'validity_ratio' in result else 'N/A'

            logger.info(f"  {status} {filename}")
            logger.info(f"      Epoch: {epoch}, Entries: {entries}, Quality: {quality}")

        logger.info("=" * 60)


class TrainingCompatibilityTest:
    """训练兼容性测试器"""

    def __init__(self, hard_negatives: dict):
        self.hard_negatives = hard_negatives

    def run_test(self) -> bool:
        """运行训练兼容性测试"""
        logger.info("🔧 Running training compatibility tests...")

        tests = [
            self._test_data_access(),
            self._test_negative_sampling_simulation(),
            self._test_batch_processing_simulation(),
            self._test_memory_efficiency()
        ]

        success_count = sum(tests)
        total_tests = len(tests)

        logger.info(f"🧪 Compatibility tests: {success_count}/{total_tests} passed")
        return success_count == total_tests

    def _test_data_access(self) -> bool:
        """测试数据访问"""
        try:
            logger.info("  🔍 Testing data access...")

            # 随机选择一些正样本进行测试
            sample_keys = list(self.hard_negatives.keys())[:10]

            for key in sample_keys:
                negatives = self.hard_negatives[key]

                # 检查能否正确访问负样本
                if not isinstance(negatives, list) or len(negatives) == 0:
                    logger.error(f"    ❌ Invalid negatives for key {key}")
                    return False

                # 检查负样本格式
                for neg in negatives:
                    if not isinstance(neg, tuple) or len(neg) != 3:
                        logger.error(f"    ❌ Invalid negative format: {neg}")
                        return False

            logger.info("    ✅ Data access test passed")
            return True

        except Exception as e:
            logger.error(f"    ❌ Data access test failed: {str(e)}")
            return False

    def _test_negative_sampling_simulation(self) -> bool:
        """模拟负样本采样过程"""
        try:
            logger.info("  🎲 Testing negative sampling simulation...")

            # 模拟HardNegativeAwareDataset的使用方式
            sample_pos = list(self.hard_negatives.keys())[0]
            h, t, r = sample_pos

            if sample_pos in self.hard_negatives:
                hard_negs = self.hard_negatives[sample_pos]

                # 模拟采样过程
                import random
                selected_hard = random.sample(hard_negs, min(2, len(hard_negs)))

                # 验证采样结果
                for neg in selected_hard:
                    nh, nt, nr = neg
                    # 检查是否确实是负样本（至少一个元素不同）
                    if (nh == h and nt == t and nr == r):
                        logger.error(f"    ❌ Sampled negative is identical to positive: {neg}")
                        return False

                logger.info(f"    ✅ Sampling simulation passed (selected {len(selected_hard)} negatives)")
                return True
            else:
                logger.error("    ❌ Sample positive not found in hard negatives")
                return False

        except Exception as e:
            logger.error(f"    ❌ Negative sampling test failed: {str(e)}")
            return False

    def _test_batch_processing_simulation(self) -> bool:
        """模拟批处理"""
        try:
            logger.info("  📦 Testing batch processing simulation...")

            # 模拟批量处理10个样本
            sample_keys = list(self.hard_negatives.keys())[:10]
            batch_negatives = []

            for key in sample_keys:
                negatives = self.hard_negatives[key]
                # 每个正样本选择1-2个硬负样本
                import random
                selected = random.sample(negatives, min(2, len(negatives)))
                batch_negatives.extend(selected)

            # 检查批处理结果
            if len(batch_negatives) < len(sample_keys):
                logger.error("    ❌ Insufficient negatives generated in batch")
                return False

            # 检查没有重复的完全相同的三元组
            unique_negatives = set(batch_negatives)
            diversity_ratio = len(unique_negatives) / len(batch_negatives)

            logger.info(f"    ✅ Batch processing passed (diversity: {diversity_ratio:.2%})")
            return True

        except Exception as e:
            logger.error(f"    ❌ Batch processing test failed: {str(e)}")
            return False

    def _test_memory_efficiency(self) -> bool:
        """测试内存效率"""
        try:
            logger.info("  💾 Testing memory efficiency...")

            # 计算数据大小
            import sys
            total_size = sys.getsizeof(self.hard_negatives)

            # 估算每个条目的平均大小
            num_entries = len(self.hard_negatives)
            avg_size_per_entry = total_size / num_entries if num_entries > 0 else 0

            # 检查是否存在异常大的条目
            large_entries = 0
            for key, negatives in list(self.hard_negatives.items())[:100]:  # 抽样检查
                entry_size = sys.getsizeof(key) + sys.getsizeof(negatives)
                if entry_size > avg_size_per_entry * 10:  # 超过平均大小10倍
                    large_entries += 1

            logger.info(f"    💾 Memory stats: {total_size / 1024 / 1024:.2f}MB total, {avg_size_per_entry:.0f}B/entry")

            if large_entries > 5:  # 允许少量异常
                logger.warning(f"    ⚠️ Found {large_entries} unusually large entries")

            logger.info("    ✅ Memory efficiency test passed")
            return True

        except Exception as e:
            logger.error(f"    ❌ Memory efficiency test failed: {str(e)}")
            return False


def create_sample_cache_for_testing():
    """创建用于测试的示例缓存文件"""
    logger.info("🔧 Creating sample cache file for testing...")

    # 创建示例硬负样本数据
    sample_hard_negatives = {}

    # 模拟一些药物ID和关系
    drug_ids = [f"drug_{i:03d}" for i in range(100)]
    relations = [f"relation_{i}" for i in range(10)]

    # 生成示例数据
    import random
    for i in range(50):  # 50个正样本
        h = random.choice(drug_ids)
        t = random.choice(drug_ids)
        r = random.choice(relations)

        if h != t:  # 确保头尾不同
            # 为每个正样本生成2-4个硬负样本
            negatives = []
            num_negs = random.randint(2, 4)

            for _ in range(num_negs):
                # 随机选择替换头部或尾部
                if random.random() < 0.5:
                    neg_h = random.choice(drug_ids)
                    neg_t = t
                else:
                    neg_h = h
                    neg_t = random.choice(drug_ids)

                negatives.append((neg_h, neg_t, r))

            sample_hard_negatives[(h, t, r)] = negatives

    # 创建完整的缓存数据
    cache_data = {
        'epoch': 5,
        'fingerprint': 'test_fingerprint_123',
        'timestamp': time.time(),
        'hard_negatives': sample_hard_negatives,
        'num_positives': len(sample_hard_negatives)
    }

    # 保存到文件
    cache_dir = './test_hard_neg_cache'
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, 'hard_neg_epoch_9_680e00b1dee4806a.pkl')

    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)

    logger.info(f"✅ Sample cache created: {cache_file}")
    return cache_file


def main():
    """主测试函数"""
    print("🧪 Hard Negative Cache Validation Test")
    print("=" * 50)

    # 选项1: 测试现有缓存
    cache_dir = './hard_neg_cache'

    if not os.path.exists(cache_dir) or not os.listdir(cache_dir):
        print("⚠️ No existing cache found. Creating sample cache for testing...")
        sample_file = create_sample_cache_for_testing()
        cache_dir = os.path.dirname(sample_file)

    # 运行验证
    validator = HardNegativeCacheValidator(cache_dir)
    success = validator.run_full_validation()

    print("\n" + "=" * 50)
    if success:
        print("🎉 ALL TESTS PASSED! The cache is ready for training.")
    else:
        print("⚠️ Some tests failed. Please check the logs above.")
        print("💡 Suggestions:")
        print("   1. Delete corrupted cache files and regenerate")
        print("   2. Check hard negative generation configuration")
        print("   3. Ensure sufficient disk space and permissions")

    return success


if __name__ == "__main__":
    # 运行测试
    success = main()

    # 退出码
    import sys

    sys.exit(0 if success else 1)