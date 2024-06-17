import random
import csv

with open('D:\\Fine-tuning\\googledata\\SST2\\train.tsv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
    # 从数据中随机抽样
    random.seed(48)
    sample = random.sample(data, 50000)
    # 将结果写入新文件
    with open('D:\\Fine-tuning\\googledata\\sample\\SST5K.tsv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(sample)