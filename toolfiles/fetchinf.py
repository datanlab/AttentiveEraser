import csv

def read_test_random_files(file_path):
    mask_paths = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            mask_path = line.strip().rsplit('/', 1)[1]
            mask_paths.append(mask_path)
    return mask_paths

def read_class_descriptions(file_path):
    label_map = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            labelid, labelname = row
            label_map[labelid] = labelname
    return label_map

def read_detailed_csv(file_path, mask_paths, label_map):
    results = []
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        header = next(reader)  # Skip header if there is one
        for row in reader:
            mask_path, image_id, labelid, box_id, box_x_min, box_x_max, box_y_min, box_y_max, predicted_iou, clicks = row
            if mask_path in mask_paths:
                labelname = label_map.get(labelid, "Unknown")
                results.append([mask_path, labelname, box_x_min, box_x_max, box_y_min, box_y_max])
    return results

def write_to_csv(file_path, data):
    with open(file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['MaskPath', 'LabelName', 'BoxXMin', 'BoxXMax', 'BoxYMin', 'BoxYMax'])
        writer.writerows(data)

# 文件路径
test_random_files_path = '/hy-tmp/DATA/test_random_files.txt'
class_descriptions_path = '/hy-tmp/DATA/class-descriptions-boxable.csv'
detailed_csv_path = '/hy-tmp/DATA/test-annotations-object-segmentation.csv'  # 替换为你的CSV文件路径
output_csv_path = '/hy-tmp/DATA/fetch_output.csv'

# 读取文件内容
mask_paths = read_test_random_files(test_random_files_path)
label_map = read_class_descriptions(class_descriptions_path)
detailed_data = read_detailed_csv(detailed_csv_path, mask_paths, label_map)

# 写入新的CSV文件
write_to_csv(output_csv_path, detailed_data)

print(f"Data has been written to {output_csv_path}")