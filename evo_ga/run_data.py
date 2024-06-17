import random
from random import randint

oldf = open('D:\\Fine-tuning\\dataset\\MNLI_5K\\dev_mismatched.tsv', 'r', encoding='UTF-8')
newf = open('D:\\Fine-tuning\\dataset\\MNLI_5K_new\\dev_mismatched.tsv', 'w', encoding='UTF-8')
n = 0
# sample(x,y)函数的作用是从序列x中，随机选择y个不重复的元素
resultList = random.sample(range(0, 9516), 2500)

lines = oldf.readlines()
for i in resultList:
    newf.write(lines[i])

oldf.close()
newf.close()