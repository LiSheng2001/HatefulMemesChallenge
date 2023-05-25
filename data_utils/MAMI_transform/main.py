'''
将MAMI数据集的格式转换成类似HatefulMemes的格式
具体事情包括：
- 切分验证集
- 将数据导出为jsonl格式
- 将图片转为png格式并使用对应ID编码
'''
import pandas as pd
import json
import os
import glob
from PIL import Image
from tqdm import tqdm

base_dir = "/root/autodl-tmp"
base_ouput_dir = "/root/autodl-tmp/MAMI"
os.makedirs(base_ouput_dir, exist_ok=True)

# 加载数据集
train_dataset = pd.read_csv(os.path.join(base_dir, "TRAINING/training.csv"), sep="\t")
# 加载测试数据集
test_dataset = pd.read_csv(os.path.join(base_dir, "test/Test.csv"), sep="\t")
# 加载测试数据集的label
test_label = pd.read_table(os.path.join(base_dir, "test_labels.txt"), header=None,
                            names=['file_name', 'misogynous', 'shaming', 'stereotype', 'objectification', 'violence'])
# 合并测试集的特征和label
test_dataset = pd.merge(test_dataset, test_label, on="file_name")

# 处理图片路径
train_dataset["img"] = "img/" + train_dataset["file_name"]
test_dataset["img"] = "img/" + test_dataset["file_name"]

# 切分验证集
p_train_dataset = train_dataset.sample(frac=0.9,random_state=2,axis=0) 
validation_dataset = train_dataset[~train_dataset.index.isin(p_train_dataset.index)]

# 将id补到5位数使得jsonl文件更好看
def pad_to_five_digits(number):
    number_str = str(number)
    if len(number_str) < 5:
        padding = "0" * (5 - len(number_str))
        padded_number = padding + number_str
        return padded_number
    else:
        return number_str

# 开始导出数据集为jsonl
def extract_line(anno_line):
    data = {
        "id": pad_to_five_digits(anno_line.file_name.replace(".jpg", "")), 
        "img": "img/" + pad_to_five_digits(anno_line.file_name.replace(".jpg", "")) + ".png", 
        "label": anno_line.misogynous, 
        "text": anno_line["Text Transcription"]
    }
    return data


with open(os.path.join(base_ouput_dir, "train.jsonl"), 'w') as f:
    for idx, anno_line in p_train_dataset.iterrows():
        # 模仿HatefulMemes的处理，只要id, img, label和text列
        data_line = extract_line(anno_line)
        f.write(f"{json.dumps(data_line)}\n")

with open(os.path.join(base_ouput_dir, "dev.jsonl"), 'w') as f:
    for idx, anno_line in validation_dataset.iterrows():
        # 模仿HatefulMemes的处理，只要id, img, label和text列
        data_line = extract_line(anno_line)
        f.write(f"{json.dumps(data_line)}\n")

with open(os.path.join(base_ouput_dir, "test.jsonl"), 'w') as f:
    for idx, anno_line in test_dataset.iterrows():
        # 模仿HatefulMemes的处理，只要id, img, label和text列
        data_line = extract_line(anno_line)
        f.write(f"{json.dumps(data_line)}\n")

print("--------------导出jsonl数据成功--------------------")


# 将训练集和测试集数据都搬到base_ouput_dir的img目录中
train_dir = os.path.join(base_dir, "TRAINING/")
train_imgs = glob.glob(os.path.join(train_dir, "*.jpg"))
test_dir = os.path.join(base_dir, "test/")
test_imgs = glob.glob(os.path.join(test_dir, "*.jpg"))
# 拼接起来
imgs = train_imgs + test_imgs

# 开始处理
image_output_dir = os.path.join(base_ouput_dir, "img/")
os.makedirs(image_output_dir, exist_ok=True)


with tqdm(total=len(imgs)) as pbar:
    pbar.set_description('Processing')
    for i, img_path in enumerate(imgs):
        # 打开并转成RGB格式
        img = Image.open(img_path).convert("RGB")
        # 处理图片名词
        filename = os.path.basename(img_path)
        # 处理后缀并填充到5位id
        basename = pad_to_five_digits(filename.replace(".jpg", ""))
        # 保存到对应目录
        img.save(os.path.join(image_output_dir, basename + ".png"))
        # 报告完成情况
        pbar.update(1)

print("-------------------数据集迁移完成！---------------------")

