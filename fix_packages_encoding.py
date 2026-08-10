# 将 dgn_ddi_packages.txt (UTF-16 + 空格) 转为 conda 可读的 UTF-8 列表
import os
path = os.path.join(os.path.dirname(__file__), 'dgn_ddi_packages.txt')
out_path = os.path.join(os.path.dirname(__file__), 'dgn_ddi_packages_conda.txt')

try:
    with open(path, 'r', encoding='utf-16') as f:
        lines = f.readlines()
except Exception:
    with open(path, 'r', encoding='utf-16-le') as f:
        lines = f.readlines()

out = []
for line in lines:
    s = line.strip().replace(' ', '')
    if not s or s.startswith('#'):
        continue
    out.append(s)

with open(out_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print('Written', len(out), 'packages to', out_path)
