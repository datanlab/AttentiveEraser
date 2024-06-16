import os
import random

test_files_path           = '/hy-tmp/DATA/test-masks/'
list_of_random_test_files = '/hy-tmp/DATA/test_random_files_100.txt'

test_files = [
    test_files_path + image for image in os.listdir(test_files_path)
]

print(f'Sampling 100 images out of {len(test_files)} images in {test_files_path}' + \
      f'and put their paths to {list_of_random_test_files}')
print('Our training procedure will pick best checkpoints according to metrics, computed on these images.')

random.shuffle(test_files)
test_files_random = test_files[0:100]
with open(list_of_random_test_files, 'w') as fw:
    for filename in test_files_random:
        fw.write(filename+'\n')
print('...done')

