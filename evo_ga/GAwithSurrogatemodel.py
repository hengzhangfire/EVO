import matplotlib.pyplot as plt
from run import *
from deap import base, creator, tools
from run_nosave import glue_evaluation
from run_nosave import glue_evaluation1
from run_nosave import glue_evaluation2
import random
from bayes_opt import BayesianOptimization
import numpy as np
from HotStart import individual
import os
import time


def uncode(layer):#解码
    if layer < 10:
        if layer < 1:
            return 0
        else:
            return layer*1e-6
    else:
        layer = layer-10
        return layer * 1e-5

def code(individual):#编码 跑热启动需要用到的
    CodeIndividual = []
    for i in range(len(individual)):
        if individual[i] < 1e-5:
            CodeIndividual.append(individual[i]*1e6)
        else:
            CodeIndividual.append(individual[i]*1e5+10)
    return CodeIndividual

def classicalGA():
    # 固定随机种子
    random.seed(42)
    # 定义问题
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    # 创建Individual类，继承list，求最大值
    creator.create('Individual', list, fitness=creator.FitnessMax)

    # 定义个体
    # 基因长度
    # 正式的
    gen_size = 12
    lb = [0] * gen_size #编码下界
    up = [20] * gen_size


    # 在上界与下界用均匀分布生成实数向量
    def uniform(lb, up):
        return [random.uniform(a, b) for a, b in zip(lb, up)]

    # 生成个体
    toolbox = base.Toolbox()
    # 注册一个Binary的alias，指向scipy.stats中的bernoulli.rvs，概率为0.5
    toolbox.register('Attr_float', uniform, lb, up)  # 等概率取0和1
    # 注册用tools.initRepeat生成长度为GENE_LENGTH的Individual、等概率取0和1、个体的基因长度
    toolbox.register('Individual', tools.initIterate, creator.Individual, toolbox.Attr_float)

    # 注册种群
    toolbox.register('Population', tools.initRepeat, list, toolbox.Individual)

    # 测试用的个体
    IndForTest= [
                [11,0,0,0, 0,0,0,0, 0,0,0,0],
                 [11,11,0,0, 0,0,0,0, 0,0,0,0],
                 [11,11,11,0, 0,0,0,0, 0,0,0,0],
                 [11,11,11,11,  0,0,0,0, 0,0,0,0],
                 [11,11,11,11, 11,0,0,0, 0,0,0,0],
                 [11,11,11,11, 11,11,0,0, 0,0,0,0],
                 [11,11,11,11, 11,11,11,0, 0,0,0,0],
                 [11,11,11,11, 11,11,11,11, 0,0,0,0],
                 [11,11,11,11, 11,11,11,11, 11,0,0,0],
                 [11,11,11,11, 11,11,11,11, 11,11,0,0],
                 [11,11,11,11, 11,11,11,11, 11,11,11,0],
                 [11,11,11,11, 11,11,11,11, 11,11,11,11]
                 ]
    #测试结果写入的文件
    path1 = 'D:\\AutoFT\\SST_Time\\test.txt'
    file1 = open(path1, 'w')

    # 定义评价函数
    def glue_eval(individual):
        individual_lr=[]
        for i in range(len(individual)):
            individual_lr.append(uncode(individual[i]))
        result = glue_evaluation(individual_lr)
        for key, value in result.items(): # 跑其他任务需要改一下
            return value * 100,

    def glue_eval2(layer0, layer1, layer2, layer3, layer4, layer5, layer6, layer7, layer8, layer9, layer10, layer11):
        layer_lr = []
        layer_lr.append(uncode(layer0))
        layer_lr.append(uncode(layer1))
        layer_lr.append(uncode(layer2))
        layer_lr.append(uncode(layer3))
        layer_lr.append(uncode(layer4))
        layer_lr.append(uncode(layer5))
        layer_lr.append(uncode(layer6))
        layer_lr.append(uncode(layer7))
        layer_lr.append(uncode(layer8))
        layer_lr.append(uncode(layer9))
        layer_lr.append(uncode(layer10))
        layer_lr.append(uncode(layer11))
        result = glue_evaluation(layer_lr)
        for key, value in result.items():
            return value*100,
    # 注册GPR
    pbounds = {'layer0': (0, 20), 'layer1': (0, 20), 'layer2': (0, 20), 'layer3': (0, 20), 'layer4': (0, 20), 'layer5': (0, 20),'layer6': (0, 20), 'layer7': (0, 20), 'layer8': (0, 20), 'layer9': (0, 20), 'layer10': (0, 20), 'layer11': (0, 20)}
    optimizer = BayesianOptimization(
        f=glue_eval2,
        pbounds=pbounds,
        random_state=1,
    )

    # 注册评价函数
    toolbox.register('evaluate', glue_eval)
    # 注册Tournsize为2的锦标赛选择
    toolbox.register('select', tools.selTournament, tournsize=2)
    toolbox.register('mate', tools.cxSimulatedBinaryBounded, eta=20, low=lb, up=up)
    toolbox.register('mutate', tools.mutPolynomialBounded, eta=20, low=lb, up=up, indpb=0.5)

    # 注册计算过程中需要记录的数据 也是数据记录
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)  # 计算适应度的函数
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # 注册记录
    logbook = tools.Logbook()
    logbook.header = 'gen', 'avg', 'std', 'min', 'max'

    # 遗传算法参数
    pop_size = 40
    N_GEN = 20
    # 较高的变异率和交叉率似乎更有好处
    cxpb = 0.8  # 交叉概率
    mutpb = 0.8  # 突变概率
    hotpop=toolbox.Population(n=7)
    for i in range(7):
        hotpop[i][:] = individual[i]

    #lyx 测试先验知识有无用
    #有先验知识版本
    rndpop = toolbox.Population(n=(pop_size-9))
    pop = hotpop + rndpop
    #无先验知识版本
    #rndpop = toolbox.Population(n=pop_size)
    #pop =rndpop

    # 评价族群
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        optimizer.register(ind, fit[0]) # 第一代的数据喂进去了
    # 喂完第一批数据进行第一次拟合
    optimizer.FitGP()
    # 根据族群适应度，编译出stats记录
    record = stats.compile(pop)
    logbook.record(gen=0, **record)  # 第0代
    record_bestfit = []
    # 遗传算法迭代
    for gen in range(1, N_GEN + 1):
        # 育种选择
        selectTour = toolbox.select(pop, pop_size)
        # 复制
        selectInd = list(map(toolbox.clone, selectTour))
        # 变异操作
        # 交叉
        for child1, child2 in zip(selectInd[::2], selectInd[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # 变异
        for ind in selectInd:
            if random.random() < mutpb:
                toolbox.mutate(ind)
                del ind.fitness.values

        # 对于被改变的个体，重新计算其适应度
        # 对于被改变的个体 = 新的没有评估过的个体
        # 需要重新计算适应度
        # 但是这里我们先用代理评估
        invalid_ind = [ind for ind in selectInd if (not ind.fitness.valid) and (not ind in pop)]
        predict_fitness = []
        # GPR效果测试
        test_fitness = []
        for each_individual in IndForTest:
            test_individual = np.array(each_individual)
            test_individual = test_individual.reshape(1,-1)
            value = optimizer.predicwithGP(test_individual)
            test_fitness.append(value[0])
        file1.write(str(test_fitness))
        file1.write('\r\n')
        # 个体预测
        for new_individual in invalid_ind:
            new_individual = np.array(new_individual)
            new_individual = new_individual.reshape(1, -1)
            value = optimizer.predicwithGP(new_individual) # 得到该个体的预估值
            predict_fitness.append(value[0]) # 合并所有新个体的预估值
        # 所有的个体一起排序
        WorstInd = tools.selWorst(pop, 1)[0]         # 以最低的真实值为标准筛选出有潜力的个体
        Standard_score = WorstInd.fitness.values[0]
        chosen_pop = []
        for i in range(len(invalid_ind)): #对于每一个新个体
             # ind.fitness.values = fit 不能用这个赋值，会带来混乱，整个函数要对这个属性的赋值一值保持真实
             if predict_fitness[i] > Standard_score:
                 chosen_pop.append(invalid_ind[i])
        if len(chosen_pop) < 5:
            chosen_pop = invalid_ind
        # test
        suggest = optimizer.my_suggest()
        suggest_ind = toolbox.Population(n=1)
        # 转换一下二者格式
        i = 0
        for key, value in suggest.items():
            suggest_ind[0][i]=value
            i = i+1
        # 需要将sugget
        # 所选的个体进行真实的评估
        # 每一个真实评估的个体都需要喂进GPR中用以更新代理模型
        new_evaluate_ind = chosen_pop+suggest_ind
        #    [ind for ind in chosen_pop if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, new_evaluate_ind)
        for ind, fit in zip(new_evaluate_ind, fitnesses):
            ind.fitness.values = fit
            # 异常处理，避免可能出现的任何重复的点，我们用try来处理
            try:
                optimizer.register(ind, fit[0])
            except:
                print('same point')
        # 产生的真实评估用于更新代理模型
        optimizer.FitGP()
        # 环境选择
        # 精英策略
        combinedPop = pop + new_evaluate_ind
        pop = tools.selBest(combinedPop, pop_size)
        record = stats.compile(pop)
        logbook.record(gen=gen, **record)
        # 记录每一代的最佳值
        bestInd = tools.selBest(pop, 1)[0]
        bestFit = bestInd.fitness.values[0]
        record_bestfit.append(bestFit)
    # 迭代结束
    # 打印迭代过程
    print(logbook)
    # 输出结果
    bestInd = tools.selBest(pop, 1)[0]
    bestFit = bestInd.fitness.values[0]
    print('最优解为：', str(bestInd))
    print('对应的函数最大值为:', str(bestFit))
    # 对迭代过程可视化 并保存图
    gen = logbook.select('gen')
    max_fitness = logbook.select('max')
    avg_fitness = logbook.select('avg')

    # 创建一个txt文件保存结果
    path = 'D:\\AutoFT\\SST_Time\\res.txt'
    file = open(path, 'w')
    file.write(str(bestInd))


    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot()
    plt.plot(gen, max_fitness, 'r-', label='MAX_FITNESS')
    plt.plot(gen, avg_fitness, 'b-', label='AVG_FITNESS')
    plt.xlabel('gen')
    plt.ylabel('fit')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('D:\\AutoFT\\SST_Time\\fig' + '.png')
    file.write(str(logbook))
    file.close()
    file1.close()

def classicalGA1():
    # 固定随机种子
    random.seed(36)
    # 定义问题
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    # 创建Individual类，继承list，求最大值
    creator.create('Individual', list, fitness=creator.FitnessMax)

    # 定义个体
    # 基因长度
    # 正式的
    gen_size = 12
    lb = [0] * gen_size #编码下界
    up = [20] * gen_size


    # 在上界与下界用均匀分布生成实数向量
    def uniform(lb, up):
        return [random.uniform(a, b) for a, b in zip(lb, up)]

    # 生成个体
    toolbox = base.Toolbox()
    # 注册一个Binary的alias，指向scipy.stats中的bernoulli.rvs，概率为0.5
    toolbox.register('Attr_float', uniform, lb, up)  # 等概率取0和1
    # 注册用tools.initRepeat生成长度为GENE_LENGTH的Individual、等概率取0和1、个体的基因长度
    toolbox.register('Individual', tools.initIterate, creator.Individual, toolbox.Attr_float)

    # 注册种群
    toolbox.register('Population', tools.initRepeat, list, toolbox.Individual)

    # 测试用的个体
    IndForTest= [
                [11,0,0,0, 0,0,0,0, 0,0,0,0],
                 [11,11,0,0, 0,0,0,0, 0,0,0,0],
                 [11,11,11,0, 0,0,0,0, 0,0,0,0],
                 [11,11,11,11,  0,0,0,0, 0,0,0,0],
                 [11,11,11,11, 11,0,0,0, 0,0,0,0],
                 [11,11,11,11, 11,11,0,0, 0,0,0,0],
                 [11,11,11,11, 11,11,11,0, 0,0,0,0],
                 [11,11,11,11, 11,11,11,11, 0,0,0,0],
                 [11,11,11,11, 11,11,11,11, 11,0,0,0],
                 [11,11,11,11, 11,11,11,11, 11,11,0,0],
                 [11,11,11,11, 11,11,11,11, 11,11,11,0],
                 [11,11,11,11, 11,11,11,11, 11,11,11,11]
                 ]
    #测试结果写入的文件
    path1 = 'D:\\AutoFT\\0119test\\QQP36\\test.txt'
    file1 = open(path1, 'w')

    # 定义评价函数
    def glue_eval(individual):
        individual_lr=[]
        for i in range(len(individual)):
            individual_lr.append(uncode(individual[i]))
        result = glue_evaluation1(individual_lr)
        for key, value in result.items(): # 跑其他任务需要改一下
            return value * 100,

    def glue_eval2(layer0, layer1, layer2, layer3, layer4, layer5, layer6, layer7, layer8, layer9, layer10, layer11):
        layer_lr = []
        layer_lr.append(uncode(layer0))
        layer_lr.append(uncode(layer1))
        layer_lr.append(uncode(layer2))
        layer_lr.append(uncode(layer3))
        layer_lr.append(uncode(layer4))
        layer_lr.append(uncode(layer5))
        layer_lr.append(uncode(layer6))
        layer_lr.append(uncode(layer7))
        layer_lr.append(uncode(layer8))
        layer_lr.append(uncode(layer9))
        layer_lr.append(uncode(layer10))
        layer_lr.append(uncode(layer11))
        result = glue_evaluation1(layer_lr)
        for key, value in result.items():
            return value*100,
    # 注册GPR
    pbounds = {'layer0': (0, 20), 'layer1': (0, 20), 'layer2': (0, 20), 'layer3': (0, 20), 'layer4': (0, 20), 'layer5': (0, 20),'layer6': (0, 20), 'layer7': (0, 20), 'layer8': (0, 20), 'layer9': (0, 20), 'layer10': (0, 20), 'layer11': (0, 20)}
    optimizer = BayesianOptimization(
        f=glue_eval2,
        pbounds=pbounds,
        random_state=1,
    )

    # 注册评价函数
    toolbox.register('evaluate', glue_eval)
    # 注册Tournsize为2的锦标赛选择
    toolbox.register('select', tools.selTournament, tournsize=2)
    toolbox.register('mate', tools.cxSimulatedBinaryBounded, eta=20, low=lb, up=up)
    toolbox.register('mutate', tools.mutPolynomialBounded, eta=20, low=lb, up=up, indpb=0.5)

    # 注册计算过程中需要记录的数据 也是数据记录
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)  # 计算适应度的函数
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # 注册记录
    logbook = tools.Logbook()
    logbook.header = 'gen', 'avg', 'std', 'min', 'max'

    # 遗传算法参数
    pop_size = 40
    N_GEN = 20
    # 较高的变异率和交叉率似乎更有好处
    cxpb = 0.8  # 交叉概率
    mutpb = 0.8  # 突变概率
    hotpop=toolbox.Population(n=7)
    for i in range(7):
        hotpop[i][:] = individual[i]

    #lyx 测试先验知识有无用
    #有先验知识版本
    rndpop = toolbox.Population(n=(pop_size-9))
    pop = hotpop + rndpop
    #无先验知识版本
    #rndpop = toolbox.Population(n=pop_size)
    #pop =rndpop

    # 评价族群
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        optimizer.register(ind, fit[0]) # 第一代的数据喂进去了
    # 喂完第一批数据进行第一次拟合
    optimizer.FitGP()
    # 根据族群适应度，编译出stats记录
    record = stats.compile(pop)
    logbook.record(gen=0, **record)  # 第0代
    record_bestfit = []
    # 遗传算法迭代
    for gen in range(1, N_GEN + 1):
        # 育种选择
        selectTour = toolbox.select(pop, pop_size)
        # 复制
        selectInd = list(map(toolbox.clone, selectTour))
        # 变异操作
        # 交叉
        for child1, child2 in zip(selectInd[::2], selectInd[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # 变异
        for ind in selectInd:
            if random.random() < mutpb:
                toolbox.mutate(ind)
                del ind.fitness.values

        # 对于被改变的个体，重新计算其适应度
        # 对于被改变的个体 = 新的没有评估过的个体
        # 需要重新计算适应度
        # 但是这里我们先用代理评估
        invalid_ind = [ind for ind in selectInd if (not ind.fitness.valid) and (not ind in pop)]
        predict_fitness = []
        # GPR效果测试
        test_fitness = []
        for each_individual in IndForTest:
            test_individual = np.array(each_individual)
            test_individual = test_individual.reshape(1,-1)
            value = optimizer.predicwithGP(test_individual)
            test_fitness.append(value[0])
        file1.write(str(test_fitness))
        file1.write('\r\n')
        # 个体预测
        for new_individual in invalid_ind:
            new_individual = np.array(new_individual)
            new_individual = new_individual.reshape(1, -1)
            value = optimizer.predicwithGP(new_individual) # 得到该个体的预估值
            predict_fitness.append(value[0]) # 合并所有新个体的预估值
        # 所有的个体一起排序
        WorstInd = tools.selWorst(pop, 1)[0]         # 以最低的真实值为标准筛选出有潜力的个体
        Standard_score = WorstInd.fitness.values[0]
        chosen_pop = []
        for i in range(len(invalid_ind)): #对于每一个新个体
             # ind.fitness.values = fit 不能用这个赋值，会带来混乱，整个函数要对这个属性的赋值一值保持真实
             if predict_fitness[i] > Standard_score:
                 chosen_pop.append(invalid_ind[i])
        if len(chosen_pop) < 5:
            chosen_pop = invalid_ind
        # test
        suggest = optimizer.my_suggest()
        suggest_ind = toolbox.Population(n=1)
        # 转换一下二者格式
        i = 0
        for key, value in suggest.items():
            suggest_ind[0][i]=value
            i = i+1
        # 需要将sugget
        # 所选的个体进行真实的评估
        # 每一个真实评估的个体都需要喂进GPR中用以更新代理模型
        new_evaluate_ind = chosen_pop+suggest_ind
        #    [ind for ind in chosen_pop if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, new_evaluate_ind)
        for ind, fit in zip(new_evaluate_ind, fitnesses):
            ind.fitness.values = fit
            # 异常处理，避免可能出现的任何重复的点，我们用try来处理
            try:
                optimizer.register(ind, fit[0])
            except:
                print('same point')
        # 产生的真实评估用于更新代理模型
        optimizer.FitGP()
        # 环境选择
        # 精英策略
        combinedPop = pop + new_evaluate_ind
        pop = tools.selBest(combinedPop, pop_size)
        record = stats.compile(pop)
        logbook.record(gen=gen, **record)
        # 记录每一代的最佳值
        bestInd = tools.selBest(pop, 1)[0]
        bestFit = bestInd.fitness.values[0]
        record_bestfit.append(bestFit)
    # 迭代结束
    # 打印迭代过程
    print(logbook)
    # 输出结果
    bestInd = tools.selBest(pop, 1)[0]
    bestFit = bestInd.fitness.values[0]
    print('最优解为：', str(bestInd))
    print('对应的函数最大值为:', str(bestFit))
    # 对迭代过程可视化 并保存图
    gen = logbook.select('gen')
    max_fitness = logbook.select('max')
    avg_fitness = logbook.select('avg')

    # 创建一个txt文件保存结果
    path = 'D:\\AutoFT\\0119test\\QQP36\\res.txt'
    file = open(path, 'w')
    file.write(str(bestInd))


    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot()
    plt.plot(gen, max_fitness, 'r-', label='MAX_FITNESS')
    plt.plot(gen, avg_fitness, 'b-', label='AVG_FITNESS')
    plt.xlabel('gen')
    plt.ylabel('fit')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('D:\\AutoFT\\0119test\\QQP36\\fig' + '.png')
    file.write(str(logbook))
    file.close()
    file1.close()

def classicalGA2():
    # 固定随机种子
    random.seed(36)
    # 定义问题
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    # 创建Individual类，继承list，求最大值
    creator.create('Individual', list, fitness=creator.FitnessMax)

    # 定义个体
    # 基因长度
    # 正式的
    gen_size = 12
    lb = [0] * gen_size #编码下界
    up = [20] * gen_size


    # 在上界与下界用均匀分布生成实数向量
    def uniform(lb, up):
        return [random.uniform(a, b) for a, b in zip(lb, up)]

    # 生成个体
    toolbox = base.Toolbox()
    # 注册一个Binary的alias，指向scipy.stats中的bernoulli.rvs，概率为0.5
    toolbox.register('Attr_float', uniform, lb, up)  # 等概率取0和1
    # 注册用tools.initRepeat生成长度为GENE_LENGTH的Individual、等概率取0和1、个体的基因长度
    toolbox.register('Individual', tools.initIterate, creator.Individual, toolbox.Attr_float)

    # 注册种群
    toolbox.register('Population', tools.initRepeat, list, toolbox.Individual)

    # 测试用的个体
    IndForTest= [
                [11,0,0,0, 0,0,0,0, 0,0,0,0],
                 [11,11,0,0, 0,0,0,0, 0,0,0,0],
                 [11,11,11,0, 0,0,0,0, 0,0,0,0],
                 [11,11,11,11,  0,0,0,0, 0,0,0,0],
                 [11,11,11,11, 11,0,0,0, 0,0,0,0],
                 [11,11,11,11, 11,11,0,0, 0,0,0,0],
                 [11,11,11,11, 11,11,11,0, 0,0,0,0],
                 [11,11,11,11, 11,11,11,11, 0,0,0,0],
                 [11,11,11,11, 11,11,11,11, 11,0,0,0],
                 [11,11,11,11, 11,11,11,11, 11,11,0,0],
                 [11,11,11,11, 11,11,11,11, 11,11,11,0],
                 [11,11,11,11, 11,11,11,11, 11,11,11,11]
                 ]
    #测试结果写入的文件
    path1 = 'D:\\AutoFT\\0119test\\CoLA36\\test.txt'
    file1 = open(path1, 'w')

    # 定义评价函数
    def glue_eval(individual):
        individual_lr=[]
        for i in range(len(individual)):
            individual_lr.append(uncode(individual[i]))
        result = glue_evaluation2(individual_lr)
        for key, value in result.items(): # 跑其他任务需要改一下
            return value * 100,

    def glue_eval2(layer0, layer1, layer2, layer3, layer4, layer5, layer6, layer7, layer8, layer9, layer10, layer11):
        layer_lr = []
        layer_lr.append(uncode(layer0))
        layer_lr.append(uncode(layer1))
        layer_lr.append(uncode(layer2))
        layer_lr.append(uncode(layer3))
        layer_lr.append(uncode(layer4))
        layer_lr.append(uncode(layer5))
        layer_lr.append(uncode(layer6))
        layer_lr.append(uncode(layer7))
        layer_lr.append(uncode(layer8))
        layer_lr.append(uncode(layer9))
        layer_lr.append(uncode(layer10))
        layer_lr.append(uncode(layer11))
        result = glue_evaluation2(layer_lr)
        for key, value in result.items():
            return value*100,
    # 注册GPR
    pbounds = {'layer0': (0, 20), 'layer1': (0, 20), 'layer2': (0, 20), 'layer3': (0, 20), 'layer4': (0, 20), 'layer5': (0, 20),'layer6': (0, 20), 'layer7': (0, 20), 'layer8': (0, 20), 'layer9': (0, 20), 'layer10': (0, 20), 'layer11': (0, 20)}
    optimizer = BayesianOptimization(
        f=glue_eval2,
        pbounds=pbounds,
        random_state=1,
    )

    # 注册评价函数
    toolbox.register('evaluate', glue_eval)
    # 注册Tournsize为2的锦标赛选择
    toolbox.register('select', tools.selTournament, tournsize=2)
    toolbox.register('mate', tools.cxSimulatedBinaryBounded, eta=20, low=lb, up=up)
    toolbox.register('mutate', tools.mutPolynomialBounded, eta=20, low=lb, up=up, indpb=0.5)

    # 注册计算过程中需要记录的数据 也是数据记录
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)  # 计算适应度的函数
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # 注册记录
    logbook = tools.Logbook()
    logbook.header = 'gen', 'avg', 'std', 'min', 'max'

    # 遗传算法参数
    pop_size = 40
    N_GEN = 20
    # 较高的变异率和交叉率似乎更有好处
    cxpb = 0.8  # 交叉概率
    mutpb = 0.8  # 突变概率
    hotpop=toolbox.Population(n=7)
    for i in range(7):
        hotpop[i][:] = individual[i]

    #lyx 测试先验知识有无用
    #有先验知识版本
    rndpop = toolbox.Population(n=(pop_size-9))
    pop = hotpop + rndpop
    #无先验知识版本
    #rndpop = toolbox.Population(n=pop_size)
    #pop =rndpop

    # 评价族群
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        optimizer.register(ind, fit[0]) # 第一代的数据喂进去了
    # 喂完第一批数据进行第一次拟合
    optimizer.FitGP()
    # 根据族群适应度，编译出stats记录
    record = stats.compile(pop)
    logbook.record(gen=0, **record)  # 第0代
    record_bestfit = []
    # 遗传算法迭代
    for gen in range(1, N_GEN + 1):
        # 育种选择
        selectTour = toolbox.select(pop, pop_size)
        # 复制
        selectInd = list(map(toolbox.clone, selectTour))
        # 变异操作
        # 交叉
        for child1, child2 in zip(selectInd[::2], selectInd[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # 变异
        for ind in selectInd:
            if random.random() < mutpb:
                toolbox.mutate(ind)
                del ind.fitness.values

        # 对于被改变的个体，重新计算其适应度
        # 对于被改变的个体 = 新的没有评估过的个体
        # 需要重新计算适应度
        # 但是这里我们先用代理评估
        invalid_ind = [ind for ind in selectInd if (not ind.fitness.valid) and (not ind in pop)]
        predict_fitness = []
        # GPR效果测试
        test_fitness = []
        for each_individual in IndForTest:
            test_individual = np.array(each_individual)
            test_individual = test_individual.reshape(1,-1)
            value = optimizer.predicwithGP(test_individual)
            test_fitness.append(value[0])
        file1.write(str(test_fitness))
        file1.write('\r\n')
        # 个体预测
        for new_individual in invalid_ind:
            new_individual = np.array(new_individual)
            new_individual = new_individual.reshape(1, -1)
            value = optimizer.predicwithGP(new_individual) # 得到该个体的预估值
            predict_fitness.append(value[0]) # 合并所有新个体的预估值
        # 所有的个体一起排序
        WorstInd = tools.selWorst(pop, 1)[0]         # 以最低的真实值为标准筛选出有潜力的个体
        Standard_score = WorstInd.fitness.values[0]
        chosen_pop = []
        for i in range(len(invalid_ind)): #对于每一个新个体
             # ind.fitness.values = fit 不能用这个赋值，会带来混乱，整个函数要对这个属性的赋值一值保持真实
             if predict_fitness[i] > Standard_score:
                 chosen_pop.append(invalid_ind[i])
        if len(chosen_pop) < 5:
            chosen_pop = invalid_ind
        # test
        suggest = optimizer.my_suggest()
        suggest_ind = toolbox.Population(n=1)
        # 转换一下二者格式
        i = 0
        for key, value in suggest.items():
            suggest_ind[0][i]=value
            i = i+1
        # 需要将sugget
        # 所选的个体进行真实的评估
        # 每一个真实评估的个体都需要喂进GPR中用以更新代理模型
        new_evaluate_ind = chosen_pop+suggest_ind
        #    [ind for ind in chosen_pop if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, new_evaluate_ind)
        for ind, fit in zip(new_evaluate_ind, fitnesses):
            ind.fitness.values = fit
            # 异常处理，避免可能出现的任何重复的点，我们用try来处理
            try:
                optimizer.register(ind, fit[0])
            except:
                print('same point')
        # 产生的真实评估用于更新代理模型
        optimizer.FitGP()
        # 环境选择
        # 精英策略
        combinedPop = pop + new_evaluate_ind
        pop = tools.selBest(combinedPop, pop_size)
        record = stats.compile(pop)
        logbook.record(gen=gen, **record)
        # 记录每一代的最佳值
        bestInd = tools.selBest(pop, 1)[0]
        bestFit = bestInd.fitness.values[0]
        record_bestfit.append(bestFit)
    # 迭代结束
    # 打印迭代过程
    print(logbook)
    # 输出结果
    bestInd = tools.selBest(pop, 1)[0]
    bestFit = bestInd.fitness.values[0]
    print('最优解为：', str(bestInd))
    print('对应的函数最大值为:', str(bestFit))
    # 对迭代过程可视化 并保存图
    gen = logbook.select('gen')
    max_fitness = logbook.select('max')
    avg_fitness = logbook.select('avg')

    # 创建一个txt文件保存结果
    path = 'D:\\AutoFT\\0119test\\CoLA36\\res.txt'
    file = open(path, 'w')
    file.write(str(bestInd))


    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot()
    plt.plot(gen, max_fitness, 'r-', label='MAX_FITNESS')
    plt.plot(gen, avg_fitness, 'b-', label='AVG_FITNESS')
    plt.xlabel('gen')
    plt.ylabel('fit')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('D:\\AutoFT\\0119test\\CoLA36\\fig' + '.png')
    file.write(str(logbook))
    file.close()
    file1.close()



if __name__ == '__main__':
    start_time = time.time()  # 程序开始时间
    #classicalGA2()
    #classicalGA1()
    classicalGA()
    end_time = time.time()  # 程序结束时间
    run_time = end_time - start_time  # 程序的运行时间，单位为秒
    print(run_time)
