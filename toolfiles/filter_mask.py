import os
import cv2
import numpy as np

# 定义文件夹路径和输出txt文件路径
folder_path = '/hy-tmp/DATA/sample3'  # 替换为mask文件夹的路径
output_file = '/hy-tmp/DATA/mask_files_record3.txt'  # 输出txt文件的路径
mask_files = [filename for filename in os.listdir(folder_path) if filename.endswith("_mask.png")]
count = 0
# 遍历文件夹中的所有mask文件
with open(output_file, 'w') as f:
    for filename in mask_files:
        # 读取mask文件
        file_path = os.path.join(folder_path, filename)
        mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        
        # 计算mask区域的比例
        total_pixels = mask.size
        mask_pixels = np.sum(mask > 0)
        mask_ratio = mask_pixels / total_pixels
        
        # 判断mask区域是否小于5%或大于90%
        if mask_ratio < 0.05 or mask_ratio > 0.90:
            count += 1
            # 记录文件名到txt文件中
            f.write(filename + '\n')

print(f"共筛选出{count}个，结果已保存到", output_file)
