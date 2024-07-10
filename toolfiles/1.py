import os

# 定义文件夹路径和输出txt文件路径
folder_path = '/hy-tmp/DATA/sample'  # 替换为mask文件夹的路径
output_file = '/hy-tmp/DATA/sample.txt'  # 输出txt文件的路径
img_suffix = '.png'
mask_files = [filename.rsplit('_mask', 1)[0] + img_suffix for filename in os.listdir(folder_path) if filename.endswith("_mask.png")]
count = 0
# 遍历文件夹中的所有mask文件
with open(output_file, 'w') as f:
    for filename in mask_files:
        count += 1
        # 记录文件名到txt文件中
        f.write('/hy-tmp/DATA/test-masks/' + filename + '\n')

print(f"共筛选出{count}个，结果已保存到", output_file)
