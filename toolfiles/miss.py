import os
import shutil

# 定义文件夹路径
testdata_dir = '/hy-tmp/DATA/test_sample0'
inp_outputs_dir = '/hy-tmp/outputs'
missing_files_dir = '/hy-tmp/inp_miss'

# 确保目标文件夹存在
os.makedirs(missing_files_dir, exist_ok=True)

# 获取testdata中的所有id
testdata_files = os.listdir(testdata_dir)
testdata_ids = set(f.rsplit('_' , 1)[0] for f in testdata_files if f.endswith('mask.png'))

# 获取inp_outputs中的所有id
inp_outputs_files = os.listdir(inp_outputs_dir)
inp_outputs_ids = set(f.rsplit('_' , 1)[0] for f in inp_outputs_files if f.endswith('mask.png'))

# 找出inp_outputs中缺少的id
missing_ids = testdata_ids - inp_outputs_ids

# 复制缺少的id文件到目标文件夹
for missing_id in missing_ids:
    id_file = os.path.join(testdata_dir, f'{missing_id}.jpg')
    id_mask_file = os.path.join(testdata_dir, f'{missing_id}_mask.png')
    if os.path.exists(id_file):
        shutil.copy(id_file, missing_files_dir)
    if os.path.exists(id_mask_file):
        shutil.copy(id_mask_file, missing_files_dir)

print(f'共找到{len(missing_ids)}个缺失的id文件并复制到{missing_files_dir}。')