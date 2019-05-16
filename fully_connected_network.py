# coding: utf-8

# coded by wzw / 2019/5/14
import numpy as np
import os
import struct


# 定义函数读取数据集
def loadData(path, kind="train"):
    images_path = os.path.join(path, kind + "-images.idx3-ubyte")
    labels_path = os.path.join(path, kind + "-labels.idx1-ubyte")
    # 读取标签
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    # 读取图片数据     
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
    # 读取的标签转程one_hot
    ls = []
    for i in range(labels.size):
        temp = np.zeros(10)
        temp[labels[i]] = 1
        ls.append(temp)
    labels = np.array(ls)
    return images/255000, labels


class myNetwork:
    # 传入层数, 
    def __init__(self, numOfLayers, numNeurons_preLayer, learningrate):
        self.numOfLayers = numOfLayers
        self.numNeurons_preLayer = numNeurons_preLayer
        self.learningrate = learningrate
        self.weight = []
        for i in range(numOfLayers):
            self.weight.append(np.random.normal(0.0, 0.1, (self.numNeurons_preLayer[i+1], self.numNeurons_preLayer[i])))
#             self.activation_function1 = lambda x : 1/(1 + np.exp(-x))
#             self.activation_function2 = lambda x : np.maximum(x, 0.0)

    def activation_function(self, x, kind='relu'):
        if kind == 'sigmoid':
            return 1/(1+np.exp(-x))
        if kind == 'relu':
            return np.maximum(x, 0.0)
    
    def forward(self, input_data, targets):
        input_data = np.array(input_data, ndmin=2).T
        targets = np.array(targets, ndmin=2).T
        self.outputs = []
        self.outputs.append(input_data)
        self.z = []
        self.z.append(input_data)
        
    
        for i in range(self.numOfLayers):
            temp_inputs = np.dot(self.weight[i],input_data)
            self.z.append(temp_inputs)
            if i != (self.numOfLayers - 1):
                temp_outputs = self.activation_function(temp_inputs)
            else:
                temp_outputs = self.activation_function(temp_inputs,kind= 'relu')
            input_data = temp_outputs
            self.outputs.append(temp_outputs)
                    
        #计算误差
        self.errors = []
        for i in range(self.numOfLayers):
            if i == 0:
                self.errors.append( self.outputs[-1] - targets)
            else:
                self.errors.append(np.dot((self.weight[self.numOfLayers-i]).T, self.errors[i-1]))

        return list(input_data).index(max(list(input_data))) == list(targets).index(1)
    

    def backforward(self, targets):
        grad = []
        pre = 2 * self.errors[0] * self.relu_back(self.z[-1])
        for i in range(self.numOfLayers):
#             self.weight[self.numOfLayers-i-1] -= self.learningrate * np.dot((self.errors[i]  * self.outputs[-1-i] * (1.0 - self.outputs[-1-i])), np.transpose(self.outputs[-1-i-1])) 
            grad.append(np.dot(pre, self.outputs[-1-i-1].T))
            pre = self.relu_back(self.z[-1-i-1]) * np.dot(self.weight[-1-i].T, pre)

            
        for i in range(self.numOfLayers):
            self.weight[self.numOfLayers-i-1] -= self.learningrate * grad[i]
    
    def sigmoid_back(self, z):
        return np.exp(z)/((1+np.exp(-z))*(1+np.exp(-z)))
                           
    def relu_back(self, z):
        z[ z<0 ] = 0
        z[ z>0 ] = 1
        return z

        
    def test(self, test_input, test_labels):
        inputs = np.array(test_input, ndmin=2).T
        for i in range(self.numOfLayers):
            temp_inputs = np.dot(self.weight[i], inputs)
            temp_outputs = self.activation_function(temp_inputs)
            inputs = temp_outputs
        return list(inputs).index(max(list(inputs))) == list(test_labels).index(1)
    


learning_rate = 0.1
images_data, labels = loadData("C:\\Users\\Awei\\Desktop\\实现全连接网络识别MNIST\MNIST_data", kind='train')
test_images_data, test_labels = loadData("C:\\Users\\Awei\\Desktop\\实现全连接网络识别MNIST\MNIST_data", kind='t10k')

ls = [784, 32, 32, 32, 16, 10]

n = myNetwork(5, ls, 0.6)
for i in range(30):
    if i/6 == 0:
        n.learningrate = n.learningrate/2
    count1 = 0
    for j in range(len(images_data)):
        tag = n.forward(images_data[j], labels[j])
        n.backforward(labels[j])
        if tag == True:
            count1 += 1
    count2=0
    for k in range(len(test_images_data)):
        if n.test(test_images_data[k], test_labels[k]):
            count2 += 1
    print("epoch"+ str(i) +" Train acc:" + str(count1/60000) + " Train acc:" + str(count2/10000))

