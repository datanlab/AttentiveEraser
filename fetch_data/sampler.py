import os
import random

test_files_path           = './DATA/test-masks/'
list_of_random_test_files = './DATA/sample.txt'
file_path = './DATA/sample.txt'
lines = []

with open(file_path, 'r') as file:
    for line in file:
        lines.append(line.strip())

test_files = [
    test_files_path + image for image in os.listdir(test_files_path)
]

test_files = [x for x in test_files if x not in lines]

print(f'Sampling 10000 images out of {len(test_files)} images in {test_files_path}' + \
      f'and put their paths to {list_of_random_test_files}')
print('Our training procedure will pick best checkpoints according to metrics, computed on these images.')

random.shuffle(test_files)
test_files_random = test_files[0:10000]
with open(list_of_random_test_files, 'w') as fw:
    for filename in test_files_random:
        fw.write(filename+'\n')
print('...done')

