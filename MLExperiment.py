import math

import torch
from torch import tensor
from matplotlib import pyplot as plt
import random
import scipy
import time


# calculates least squared loss
def leastSquaredLoss(instances, labels, weights, t=1):
    return (sum(((1 / (1 + torch.exp(-t*(torch.inner(instances, weights))))) - labels)**2))/len(instances)
   
# calculated cross entropy loss
def crossEntropyLoss(instances, labels, weights):
    return (-1*sum( labels * (torch.log(1 / (1 + torch.exp(-1*(torch.inner(instances, weights)+.0001))))) + (1 - labels) *torch.log( 1 - (1 / (1 + torch.exp(-1*(torch.inner(instances, weights) +.0001))))) ))/len(instances)

# calculates softmax loss
def softmaxLoss(instances, labels, weights):
    return (sum(torch.log( 1 + torch.exp(-labels*((torch.inner(instances, weights)))) )))/len(instances)

# non linear step fuction for classification
def step(num) :
    if num >= 0.5:
        return 1
    else:
        return 0

# This will see if line learned successful classifies all data correctly
def isClassified(instances,labels,weights,loss):
    if (loss == 3) :
        for i in range(len(instances)):
            if (labels[i] * torch.inner(instances[i],weights)) <= 0:
                return True
    else: 
        for i in range(len(instances)):
            if step((1 / (1 + torch.exp(-1*(torch.inner(instances[i], weights)))))) != labels[i]:
                return True
            elif step((1 / (1 + torch.exp(-1*(torch.inner(instances[i], weights)))))) != labels[i]:
                return True
    return False

# counts the number of datapoints incorrectly classified
def numberOfIncorrect(instances,labels,weights,loss):
    count = 0
    if (loss == 3) :
        for i in range(len(instances)):
            if (labels[i] * torch.inner(instances[i],weights)) <= 0:
                count+=1
    else: 
        for i in range(len(instances)):
            if step((1 / (1 + torch.exp(-1*(torch.inner(instances[i], weights)))))) != labels[i]:
                count+=1
            elif step((1 / (1 + torch.exp(-1*(torch.inner(instances[i], weights)))))) != labels[i]:
                count+=1
    return count


''' gradient decent algoritm will only run a max of 3000 iterations. loss=1 is LSL , loss=2 is CEL, loss=3 is SML
 returns wieghts, amount of time , # of iterations and # incorrect predictions '''
def gradentDecent(instances,testInstances,labels,testLabels,loss=1,step_size=0.1):
    startTime = time.perf_counter()
    weights = tensor([1] * len(instances[0]),dtype=torch.float32,requires_grad=True)
    instances = tensor(instances,dtype=torch.float32,requires_grad=True)
    labels = tensor(labels,dtype=torch.float32,requires_grad=True)
    testInstances = tensor(testInstances,dtype=torch.float32,requires_grad=True)
    testLabels = tensor(testLabels,dtype=torch.float32,requires_grad=True)
    w = []
    iterations = 0
    while isClassified(instances,labels,weights,loss):
        iterations += 1
        if (loss==1):
            loss_value = leastSquaredLoss(instances, labels, weights)
        elif (loss==2):
            loss_value = crossEntropyLoss(instances, labels, weights)
        elif(loss==3):
            loss_value = softmaxLoss(instances, labels, weights)
        loss_value.backward()
        # tell PyTorch not to keep gradients in here
        with torch.no_grad():
            weights -= step_size * weights.grad
        w.append(loss_value)
        weights.grad.zero_()
        if (iterations == 3000):
            break
    x_range = torch.linspace(0, len(w), len(w))
    w = tensor(w,dtype=torch.float32)
    # plt.plot(x_range, w.detach().numpy())
    # plt.show()
    endTime = time.perf_counter()
    incorrect = numberOfIncorrect(testInstances, testLabels, weights, loss)
    return weights, iterations, endTime-startTime, incorrect

# perceptron algorithm using a non linear step function
def perceptronLearningAlgorithmBasic(instances, testInstances, labels,testLabels):
    startTime = time.perf_counter()
    instances = tensor(instances,dtype=torch.float32)
    labels = tensor(labels,dtype=torch.float32)
    weights = tensor([0] * len(instances[0]),dtype=torch.float32)
    testInstances = tensor(testInstances,dtype=torch.float32,requires_grad=True)
    testLabels = tensor(testLabels,dtype=torch.float32,requires_grad=True)
    numOfMisclassifiedData = 10
    iterations = 0
    while numOfMisclassifiedData != 0:
        iterations+=1
        numOfMisclassifiedData = 0
        for i in range(len(instances)):
            prediction = 0
            if (torch.inner(instances[i],weights)>=0):
                prediction = 1
            else:
                prediction = -1
            if (prediction!=labels[i]):
                numOfMisclassifiedData+=1
                weights = weights + labels[i] * instances[i]
        if (iterations == 3000):
            break
    endTime = time.perf_counter()
    incorrect = numberOfIncorrect(testInstances, testLabels, weights,3)
    return weights, iterations, endTime-startTime, incorrect

# perceptron algorithm using a linear algebra
def perceptronLearningAlgorithm(instances,testInstances, labels, testLabels):
    startTime = time.perf_counter()
    instances = tensor(instances,dtype=torch.float32)
    labels = tensor(labels,dtype=torch.float32)
    weights = tensor([0] * len(instances[0]),dtype=torch.float32)
    testInstances = tensor(testInstances,dtype=torch.float32,requires_grad=True)
    testLabels = tensor(testLabels,dtype=torch.float32,requires_grad=True)
    iterations = 0
    while isClassified(instances,labels,weights,3):
        iterations +=1
        for i in range(len(instances)):
            if (labels[i] * torch.inner(instances[i],weights)<=0):
                weights += labels[i] * instances[i]
        if (iterations == 3000):
            break
    endTime = time.perf_counter()
    incorrect = numberOfIncorrect(testInstances, testLabels, weights,3)
    return weights, iterations, endTime-startTime, incorrect

# linear program to solve classification problem
def linearProgram(instances, testInstances, labels, testLabels):
    testInstances = tensor(testInstances,dtype=torch.float32,requires_grad=True)
    testLabels = tensor(testLabels,dtype=torch.float32,requires_grad=True)
    startTime = time.perf_counter()
    c = [0] * len(instances[0])
    A=[]
    b=[]
    for i in range(len(instances)):
        temp=[]
        for j in range(len(instances[i])):
            temp.append(-instances[i][j] * labels[i])
        A.append(temp)
    for label in labels:
        b.append(-1)
    res = scipy.optimize.linprog(c, A_ub=A,b_ub=b, bounds=(None,None))
    endTime = time.perf_counter()
    weights = tensor(res.x,dtype=torch.float32,requires_grad=True)
    incorrect = numberOfIncorrect(testInstances, testLabels, weights,3)
    return res.x, endTime-startTime, incorrect

# increases dementionality of weights to add a bias
def addBias(instances, weights):
    weights.append(random.random())
    for i in range(len(instances)):
        instances[i].append(1)
    return instances, weights

# generates random points of data. Classifies them to be linearly separable
def generateData(dim=2, size=100):
    hitFirst = True
    hitSecond = True
    while hitFirst or hitSecond:
        weights = []
        hitFirst = True
        hitSecond = True
        for i in range(dim+1):
            weights.append((random.random()-0.5) * 2)
        
        instances = []
        test_instances = []
        test_labels = []
        labels = []
        weightsTensor = tensor(weights,dtype=torch.float32,requires_grad=True)
        for i in range(size):
            instance = []
            for j in range(dim):
                instance.append(random.random()*5)
            instance.append(1)
            instances.append(instance)
            instanceTensor = tensor(instance,dtype=torch.float32,requires_grad=True)
            if (torch.inner(weightsTensor,instanceTensor)>=0):
                labels.append(1)
                hitFirst = False
            else:
                labels.append(0)
                hitSecond = False
                
        for i in range(size):
            instance = []
            for j in range(dim):
                instance.append(random.random()*5)
            instance.append(1)
            test_instances.append(instance)
            instanceTensor = tensor(instance,dtype=torch.float32,requires_grad=True)
            if (torch.inner(weightsTensor,instanceTensor)>=0):
                test_labels.append(1)
                hitFirst = False
            else:
                test_labels.append(0)
                hitSecond = False
                
    return instances,labels, test_instances,test_labels

# flips labels from 0 to -1
def changeLabelsToNegativeOne(labels):
    for i in range(len(labels)):
        if (labels[i]==0):
            labels[i] = -1
# not used
# def isLabelsZeroToOne(labels):
#     for i in labels:
#         if i == 0:
#             return True
#     return False

#not used
# def wrongPredictions(weights, test_instances, test_labels):
#     total = 0
#     if (isLabelsZeroToOne(test_labels) == False) :
#         for i in range(len(test_instances)):
#             if (test_labels[i] * torch.inner(test_instances[i],weights)) <= 0:
#                 total+=1
#     else: 
#         for i in range(len(test_instances)):
#             if step((1 / (1 + torch.exp(-1*(torch.inner(test_instances[i], weights)))))) != test_labels[i]:
#                 total+=1
#             elif step((1 / (1 + torch.exp(-1*(torch.inner(test_instances[i], weights)))))) != test_labels[i]:
#                 total+=1
#     return total

# polts the data with color coding. also shows line learned
def plotData(instances, labels, weights, name="scatter-simple.png"):
    print(weights)
    for i in range(len(instances)):
        color = [[0,1,0]]
        if labels[i] == 1:
            color = [[0,0,1]]
        plt.scatter(instances[i][0],instances[i][1],c=color)
    x1 = 1.0
    y1 = float((-weights[0]/weights[1]) - weights[2]/weights[1])
    x2 = 2.0
    y2 = float((-weights[0]/weights[1])*2 - weights[2]/weights[1])
    plt.axline((x1,y1),(x2,y2),linewidth=2, color='r')
    # plt.savefig(name, bbox_inches="tight")
    plt.title(name)
    plt.savefig(name)
    plt.clf()

# not used
# def addToTotal(results,totalResults,linearProgramResults,LPTotalResults):
#     for algo in results:
#         for i in range(len(algo)):
#             totalResults[i]+= algo[i]

#     for i in range(len(linearProgramResults[0])):
#         LPTotalResults[i]+= linearProgramResults[0][i]

# run experiment
instances,labels, test_instances, test_lables = generateData(size=100)
LSLResults = []
CELResults = []
SMLResults = []
PBResluts = []
PResults = []
linearProgramResults = []
# run each algorithm and plot the classified data points along with the linear seprabale line learned
LSLResults.append(gradentDecent(instances,test_instances,labels,test_lables)) 
plotData(instances,labels,LSLResults[0][0],name="Least_squared_loss.png")
plotData(test_instances,test_lables,LSLResults[0][0],name="Least_squared_loss_test.png")
CELResults.append(gradentDecent(instances,test_instances,labels, test_lables,loss=2)) 
plotData(instances,labels,CELResults[0][0],name="Cross_entropy_loss.png")
plotData(test_instances,test_lables,CELResults[0][0],name="Cross_entropy_loss_test.png")
changeLabelsToNegativeOne(labels)
changeLabelsToNegativeOne(test_lables)
SMLResults.append(gradentDecent(instances,test_instances,labels,test_lables,loss=3)) 
plotData(instances,labels,SMLResults[0][0],name="Softmax_loss.png")
plotData(test_instances,test_lables,SMLResults[0][0],name="Softmax_loss_test.png")
PBResluts.append(perceptronLearningAlgorithmBasic(instances,test_instances,labels,test_lables))
plotData(instances,labels,PBResluts[0][0],name="Basic_perceptron.png")
plotData(test_instances,test_lables,PBResluts[0][0],name="Basic_perceptron_test.png")
PResults.append(perceptronLearningAlgorithmBasic(instances,test_instances, labels,test_lables))
plotData(instances,labels,PResults[0][0],name="Perceptron.png")
plotData(test_instances,test_lables,PResults[0][0],name="Perceptron_test.png")
linearProgramResults.append(linearProgram(instances, test_instances, labels,test_lables))
plotData(instances,labels,linearProgramResults[0][0],name="LinearProgram.png")
plotData(test_instances,test_lables,linearProgramResults[0][0],name="LinearProgram_test.png")


# create a file to save results
wfile = open("lab3results.txt", mode='w')
orderOfAlgoritms = ["Least_squared","Cross_Entropy","Softmax","Basic_Perceptron","Perceptron","Linear_Program"]
stringToWrite = ""
count = 0

# run each algorithm 99 times and add results to respective list
for i in range(99):
    instances,labels, test_instances, test_lables = generateData(size=100)
    LSLResults.append(gradentDecent(instances,test_instances,labels,test_lables)) 
    CELResults.append(gradentDecent(instances,test_instances,labels, test_lables,loss=2))
    changeLabelsToNegativeOne(labels)
    changeLabelsToNegativeOne(test_lables)
    SMLResults.append(gradentDecent(instances,test_instances,labels,test_lables,loss=3)) 
    PBResluts.append(perceptronLearningAlgorithmBasic(instances,test_instances,labels,test_lables))
    PResults.append(perceptronLearningAlgorithmBasic(instances,test_instances, labels,test_lables))
    linearProgramResults.append(linearProgram(instances, test_instances, labels,test_lables))
    print(i)
    

# write out the results for each algorthm (computes average) to file
results = [0,0,0]
for weights, iterations, timeTook, incorrect in LSLResults:
    results[0] += iterations
    results[1] += timeTook
    results[2] += incorrect
stringToWrite += f"The least squared loss (GD) on average got {results[2]/100} incorrect. It took {results[0]//100} iterations and {results[1]//100} time to train\n"


results = [0,0,0]
for weights, iterations, timeTook, incorrect in CELResults:
    results[0] += iterations
    results[1] += timeTook
    results[2] += incorrect
stringToWrite += f"The cross entropy loss (GD) on average got {results[2]/100} incorrect. It took {results[0]//100} iterations and {results[1]//100} time to train\n"


results = [0,0,0]
for weights, iterations, timeTook, incorrect in SMLResults:
    results[0] += iterations
    results[1] += timeTook
    results[2] += incorrect
stringToWrite += f"The soft max loss (GD) on average got {results[2]/100} incorrect. It took {results[0]//100} iterations and {results[1]//100} time to train\n"


results = [0,0,0]
for weights, iterations, timeTook, incorrect in PBResluts:
    results[0] += iterations
    results[1] += timeTook
    results[2] += incorrect
stringToWrite += f"The basic perceptron algorithm on average got {results[2]/100} incorrect. It took {results[0]//100} iterations and {results[1]//100} time to train\n"


results = [0,0,0]
for weights, iterations, timeTook, incorrect in PResults:
    results[0] += iterations
    results[1] += timeTook
    results[2] += incorrect
stringToWrite += f"The perceptron algorithm on average got {results[2]/100} incorrect. It took {results[0]//100} iterations and {results[1]//100} time to train\n"


results = [0,0]
for weights, timeTook, incorrect in linearProgramResults:
    results[0] += timeTook
    results[1] += incorrect
stringToWrite += f"The linear programming algorithm on average got {results[1]/100} incorrect. It took {results[0]//100} time to train\n"
wfile.write(stringToWrite)
wfile.close()



