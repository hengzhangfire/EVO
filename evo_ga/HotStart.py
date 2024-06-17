import numpy as np
# 专家知识1：BERT的最后一层能提供最佳性能
def code(individual):
    CodeIndividual = []
    for i in range(len(individual)):
        if individual[i] < 1e-5:
            CodeIndividual.append(individual[i]*1e6)
        else:
            CodeIndividual.append(individual[i]*1e5+10)
    return CodeIndividual

# 	《How to Fine-Tune BERT for Text Classification?》
# 较低的学习率（如2e-5）是使BERT克服灾难性遗忘问题的必要条件。在4e-4的积极学习率下，训练集无法收敛。
individual=[]
individual1 = [2e-5, 2e-5, 2e-5, 2e-5, 2e-5, 2e-5, 2e-5, 2e-5, 2e-5, 2e-5, 2e-5, 2e-5]
individual1 = code(individual1)

#  逐层降低层速率，将基本学习率设置为ηL，并使用ηk−1=ξ·ηk，其中ξ是衰减因子，小于或等于1。且将较低的学习速率分配给较低的层对于微调BERT是有效的，并且适当的设置为ξ=0.95和lr=2.0e-5。
individual2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2e-5]
for i in range(11):
    j = 10-i
    individual2[j] = individual2[j+1]*0.95
individual2=code(individual2)


# BERT的最后一层（Layer11）能提供最佳性能
#individual3 = [2e-5, 2e-5, 2e-5, 2e-5, 2e-5, 2e-5, 2e-5, 2e-5, 2e-5, 2e-5, 2e-5, 0]
#individual3=code(individual3)


# 《Investigating Transferability in Pretrained Language Models》
# 早期层（前四层）提供了最多的QNLI增益，但中间层为CoLA和SST-2带来了额外的提升。
# QNLI
individual3 = [0, 0, 0, 0, 2e-5, 2e-5, 2e-5, 2e-5, 2e-5, 2e-5, 2e-5, 2e-5]
individual3=code(individual3)
# CoLA、SST-2
individual4 = [2e-5, 2e-5, 2e-5, 2e-5, 0, 0, 0, 0, 2e-5, 2e-5, 2e-5, 2e-5]
individual4 = code(individual4)






# 《Universal Language Model Fine-tuning for Text Classification》

# 1.	我们根据经验发现，首先通过仅微调最后一层并使用ηL来选择最后一层的学习率η−1=ηl/2.6作为下层的学习率
# 原来的
lr = 2e-5
individual5 = [0, 0, 0, 0, lr, lr, lr, lr, lr*2.6, lr*2.6, lr*2.6, lr*2.6]
individual5 = code(individual5)



## 《FreeLB: Enhanced adversarial training for natural language understanding》
# 除了MRPC，我们发现使用5×10−6的学习率给出更好的结果
# 下方已经包括
# 《Better fine-tuning by reducing representational collapse》tter fine-tuning by reducing representational collapse》
# MNLI\QNLI\QQP\SST-2\
individual6 = [5e-6, 5e-6, 5e-6, 5e-6, 5e-6, 5e-6, 5e-6, 5e-6, 5e-6, 5e-6, 5e-6, 5e-6]
individual6 = code(individual6)
## RTE\ MRPC\ CoLA
individual7 = [1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5]
individual7 = code(individual7)





# 最终使用的
individual.append(individual1)
individual.append(individual2)
individual.append(individual3)
individual.append(individual4)
individual.append(individual5)
individual.append(individual6)
individual.append(individual7)


