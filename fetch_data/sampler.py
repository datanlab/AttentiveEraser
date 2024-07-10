import os
import random

test_files_path           = '/hy-tmp/DATA/test-masks/'
list_of_random_test_files = '/hy-tmp/DATA/sample3.txt'
file_path = '/hy-tmp/DATA/sample1.txt'
# 初始化一个空列表来存储文件中的行
lines = []

# 使用 with 语句打开文件，确保文件正确关闭
with open(file_path, 'r') as file:
    # 遍历文件中的每一行
    for line in file:
        # 使用 strip() 方法去除每行末尾的换行符或空白字符
        # 然后将处理后的行添加到列表中
        lines.append(line.strip())

test_files = [
    test_files_path + image for image in os.listdir(test_files_path)
]

test_files = [x for x in test_files if x not in lines]

print(f'Sampling 8000 images out of {len(test_files)} images in {test_files_path}' + \
      f'and put their paths to {list_of_random_test_files}')
print('Our training procedure will pick best checkpoints according to metrics, computed on these images.')

random.shuffle(test_files)
test_files_random = test_files[0:1000]
with open(list_of_random_test_files, 'w') as fw:
    for filename in test_files_random:
        fw.write(filename+'\n')
print('...done')

