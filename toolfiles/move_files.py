import os
import shutil

# 源文件夹路径
source_folder = '/hy-tmp/outputs'
# 目标文件夹路径
destination_folder = '/hy-tmp/Myoutputs'

# 确保目标文件夹存在，如果不存在则创建
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder):
    # 检查文件是否以'_removed.png'结尾
    if filename.endswith('_removed.png'):
        # 构造完整的源文件路径和目标文件路径
        source_file = os.path.join(source_folder, filename)
        destination_file = os.path.join(destination_folder, filename)
        # 移动文件到目标文件夹
        shutil.move(source_file, destination_file)
        print(f'Moved: {filename}')

print('All files have been moved.')
