import os
import shutil

# 定义路径
source_folder = '/hy-tmp/DATA/sample3'  # 替换为源文件夹的路径
destination_folder = '/hy-tmp/DATA/sample3_filtered'  # 替换为目标文件夹的路径
txt_file_path = '/hy-tmp/DATA/mask_files_record3.txt'  # 替换为记录mask文件名的txt文件路径

# 确保目标文件夹存在
os.makedirs(destination_folder, exist_ok=True)

# 从txt文件中读取img_id列表
with open(txt_file_path, 'r') as f:
    img_ids = [line.strip().split('_mask.png')[0] for line in f]

# 遍历源文件夹中的所有文件，移动包含img_id的文件
for img_id in img_ids:
    # 定义相关文件的模式
    patterns = [
        f"{img_id}.jpg",
        f"{img_id}_mask.png",
        #f"{img_id}_ori_123.png",
        #f"{img_id}_ori_321.png",
        #f"{img_id}_ori_777.png",
        #f"{img_id}_removed_123.png",
        #f"{img_id}_removed_321.png",
        #f"{img_id}_removed_777.png"
    ]
    
    # 移动匹配模式的文件
    for pattern in patterns:
        source_file = os.path.join(source_folder, pattern)
        if os.path.exists(source_file):
            destination_file = os.path.join(destination_folder, pattern)
            shutil.move(source_file, destination_file)

print("文件移动完成")
