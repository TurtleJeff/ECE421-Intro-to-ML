"""
Oct/2019

@author: Yilun Wan, Da Hao
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def loadData():
    with np.load('notMNIST.npz') as data :
        Data, Target = data ['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

    
def MSE(W, b, x, y, reg):
    return np.square(np.matmul(x, W) + b - y).mean() + np.square(W).sum() * reg / 2


def gradMSE(W, b, x, y, reg):
    grad_W = np.matmul(np.matmul(x, W) + b - y, x) * 2 / len(y) + reg*W
    grad_b = (np.matmul(x, W) + b - y).sum() * 2/ len(y)
    return grad_W, grad_b


def grad_descent(W, b, x, y, alpha, epochs, reg, error_tol, validData, validTarget, testData, testTarget, lossType = "MSE"):
    if lossType == "CE":
        train_loss = [crossEntropyLoss(W, b, x, y, reg)]
        valid_loss = [crossEntropyLoss(W, b, validData, validTarget, reg)]
        test_loss = [crossEntropyLoss(W, b, testData, testTarget, reg)]
        
        train_acc = [calculate_accuracy(np.matmul(x, W)+b, y)]
        valid_acc = [calculate_accuracy(np.matmul(validData, W)+b, validTarget)]
        test_acc = [calculate_accuracy(np.matmul(testData, W)+b, testTarget)]
        
        W_best = W
        b_best = b
        for i in range(epochs):
            gradCE_W, gradCE_b = gradCE(W, b, x, y, reg)
            W_new = W - gradCE_W * alpha
            b_new = b - gradCE_b * alpha
            
            train_loss.append(crossEntropyLoss(W_new, b_new, x, y, reg))
            valid_loss.append(crossEntropyLoss(W_new, b_new, validData, validTarget, reg))
            test_loss.append(crossEntropyLoss(W_new, b_new, testData, testTarget, reg))
            
            train_acc.append(calculate_accuracy(np.matmul(x, W)+b, y))
            valid_acc.append(calculate_accuracy(np.matmul(validData, W)+b, validTarget))
            test_acc.append(calculate_accuracy(np.matmul(testData, W)+b, testTarget))
            if np.linalg.norm(W_new - W) < error_tol:
                print("stop at iteration: %d" %(i + 1))
                break
            if crossEntropyLoss(W_new, b_new, validData, validTarget, reg) < crossEntropyLoss(W, b, validData, validTarget, reg):
                W_best = W_new
                b_best = b_new
            W = W_new
            b = b_new
        # MSE
    elif lossType == "MSE": 
        train_loss = [MSE(W, b, x, y, reg)]
        valid_loss = [MSE(W, b, validData, validTarget, reg)]
        test_loss = [MSE(W, b, testData, testTarget, reg)]
        
        train_acc = [calculate_accuracy(np.matmul(x, W)+b, y)]
        valid_acc = [calculate_accuracy(np.matmul(validData, W)+b, validTarget)]
        test_acc = [calculate_accuracy(np.matmul(testData, W)+b, testTarget)]
        
        W_best = W
        b_best = b
        for i in range(epochs):
            gradMSE_W, gradMSE_b = gradMSE(W, b, x, y, reg)
            W_new = W - gradMSE_W * alpha
            b_new = b - gradMSE_b * alpha
            
            train_loss.append(MSE(W_new, b_new, x, y, reg))
            valid_loss.append(MSE(W_new, b_new, validData, validTarget, reg)) 
            test_loss.append(MSE(W_new, b_new, validData, validTarget, reg))
            
            train_acc.append(calculate_accuracy(np.matmul(x, W)+b, y))
            valid_acc.append(calculate_accuracy(np.matmul(validData, W)+b, validTarget))
            test_acc.append(calculate_accuracy(np.matmul(testData, W)+b, testTarget))           
            if np.linalg.norm(W_new - W) < error_tol:
                print("stop at iteration: %d" %(i + 1))
                break
            if MSE(W_new, b_new, validData, validTarget, reg) < MSE(W, b, validData, validTarget, reg):
                W_best = W_new
                b_best = b_new
            W = W_new
            b = b_new
    return W_best, b_best, train_loss, valid_loss, test_loss, train_acc, valid_acc, test_acc

    
def crossEntropyLoss(W, b, x, y, reg):
    error_1 = - np.matmul(np.log(1 / (1 + np.exp( - np.matmul(x, W) - b))), y)
    error_0 = - np.matmul(np.log(1 - 1 / (1 + np.exp( - np.matmul(x, W) - b))), 1 - y)
    error_w = np.square(W).sum() * reg / 2
    return (error_1 + error_0)/len(y) + error_w


def gradCE(W, b, x, y, reg):
    z = np.matmul(x, W) + b
    gradCE_W = np.matmul(1/(1+np.exp(-z)) - y, x)/len(y) + 2*reg*W    
    gradCE_b = (1/(1+np.exp(-z)) - y).sum()/len(y)
    return gradCE_W, gradCE_b


def buildGraph(loss="MSE", reg = 1.0, beta1 = 0.9, beta2 = 0.999, epsilon=1e-08):
# 	#Initialize weight and bias tensors
    tf.set_random_seed(421)
    W = tf.Variable(tf.truncated_normal(shape=[784, 1], stddev=0.5), name="weights")
    b = tf.Variable(0.0, name="biases")
    x = tf.placeholder(tf.float32, [None, 784], name="input_x")
    y_target = tf.placeholder(tf.float32, [None, 1], name="target_y")
    
    if loss == "MSE":
        y_predicted = tf.matmul(x,W) + b
        MSE_loss = tf.losses.mean_squared_error(y_target, y_predicted)
        reg_loss = tf.nn.l2_loss(W, name = "reg_error")
        total_loss = MSE_loss + reg*reg_loss
        optimizer = tf.train.GradientDescentOptimizer(0.001)
        train = optimizer.minimize(total_loss)
        
    elif loss == "CE":
        y_predicted = tf.matmul(x,W) + b
        CE_loss = tf.losses.sigmoid_cross_entropy(y_target, y_predicted)
        reg_loss = tf.nn.l2_loss(W, name = "reg_error")
        total_loss = CE_loss + reg*reg_loss
        optimizer = tf.train.GradientDescentOptimizer(0.001)
        train = optimizer.minimize(total_loss)
        
    elif loss == "ADAM_CE":
        y_predicted = tf.matmul(x,W) + b
        CE_loss = tf.losses.sigmoid_cross_entropy(y_target, y_predicted)
        reg_loss = tf.nn.l2_loss(W, name = "reg_error")
        total_loss = CE_loss + reg*reg_loss
        optimizer = tf.train.AdamOptimizer(0.001, beta1, beta2, epsilon)
        train = optimizer.minimize(total_loss)
        
    elif loss == "ADAM_MSE":
        y_predicted = tf.matmul(x,W) + b
        CE_loss = tf.losses.mean_squared_error(y_target, y_predicted)
        reg_loss = tf.nn.l2_loss(W, name = "reg_error")
        total_loss = CE_loss + reg*reg_loss
        optimizer = tf.train.AdamOptimizer(0.001, beta1, beta2, epsilon)
        train = optimizer.minimize(total_loss)
        
    return W, b, x, y_target, y_predicted, total_loss, train, reg


def calculate_accuracy(predicted_value, target_value):
    num_right = 0
    num_wrong = 0
    for j in range(len(target_value)):   
        if predicted_value[j] <= 0.5:
            predicted_value[j] = 0
        else:
            predicted_value[j] = 1    
        if predicted_value[j] == target_value[j]:
            num_right += 1
        else:
            num_wrong += 1
    return num_right/(num_right+num_wrong)

#load the data    
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()

#Part 1 MSE--------------------------------------------------------------------

#Shape the Data
trainData = np.reshape(trainData,(3500, 784))
trainTarget = np.reshape(trainTarget, (3500))
validData = np.reshape(validData, (100, 784))
validTarget = np.reshape(validTarget, (100))
testData = np.reshape(testData,(145, 784))
testTarget = np.reshape(testTarget, (145))

#Training Parameters
alpha_q_three = [0.005, 0.001, 0.0001]
reg_q_three = 0
alpha_q_four = 0.005
reg_q_four = [0.001, 0.1, 0.5]
error_tol=10e-7
epochs = 5000

#info collectors for loss and accuracy
train_loss = np.empty([3, epochs + 1])
valid_loss = np.empty([3, epochs + 1])
test_loss = np.empty([3, epochs + 1])
train_acc = np.empty([3, epochs + 1])
valid_acc = np.empty([3, epochs + 1])
test_acc = np.empty([3, epochs + 1])

#training using different learning rate
for i in range(3):
    W = np.zeros(28*28)
    b = 0
    w_trained, b_trained, train_loss[i,:], valid_loss[i,:], test_loss[i,:], train_acc[i,:], valid_acc[i,:], test_acc[i,:]  = grad_descent(W, b, trainData, trainTarget, alpha_q_three[i], epochs, reg_q_three, error_tol, validData, validTarget, testData, testTarget)

#plot the losses
fig, pic_part_three_one = plt.subplots()
for i in range(3):
    plt.plot(train_loss[i, :], label = "leanring rate = %.4f" %alpha_q_three[i])
pic_part_three_one.legend()
pic_part_three_one.set_xlabel("Iterations")
plt.title("train loss (MSE)")
fig, pic_part_three_one = plt.subplots()
for i in range(3):
    plt.plot(valid_loss[i, :], label = "leanring rate = %.4f" %alpha_q_three[i])
pic_part_three_one.legend()
pic_part_three_one.set_xlabel("Iterations")
plt.title("valid loss (MSE)")
fig, pic_part_three_one = plt.subplots()
for i in range(3):
    plt.plot(test_loss[i, :], label = "leanring rate = %.4f" %alpha_q_three[i])
pic_part_three_one.legend()
pic_part_three_one.set_xlabel("Iterations")
plt.title("test loss (MSE)")
  
#plot the accuracies
fig, pic_part_three_one = plt.subplots()
for i in range(3):
    plt.plot(train_acc[i, :], label = "leanring rate = %.4f" %alpha_q_three[i])
pic_part_three_one.legend()
pic_part_three_one.set_xlabel("Iterations")
plt.title(" train accuracies (MSE)")
fig, pic_part_three_one = plt.subplots()
for i in range(3):
    plt.plot(valid_acc[i, :], label = "leanring rate = %.4f" %alpha_q_three[i])
pic_part_three_one.legend()
pic_part_three_one.set_xlabel("Iterations")
plt.title(" valid accuracies (MSE)")
fig, pic_part_three_one = plt.subplots()
for i in range(3):
    plt.plot(test_acc[i, :], label = "leanring rate = %.4f" %alpha_q_three[i])
pic_part_three_one.legend()
pic_part_three_one.set_xlabel("Iterations")
plt.title(" test accuracies (MSE)")

#training using different regulation
for i in range(3):
    W = np.zeros(28*28)
    b = 0
    w_trained, b_trained, train_loss[i,:], valid_loss[i,:],  test_loss[i,:], train_acc[i,:], valid_acc[i,:], test_acc[i,:] = grad_descent(W, b, trainData, trainTarget, alpha_q_four, epochs, reg_q_four[i], error_tol, validData, validTarget, testData, testTarget)

#plot the losses
fig, pic_part_three_one = plt.subplots()
for i in range(3):
    plt.plot(train_loss[i, :], label = "lambda = %.3f" %reg_q_four[i])
pic_part_three_one.legend()
pic_part_three_one.set_xlabel("Iterations")
plt.title("train loss (MSE)")
fig, pic_part_three_one = plt.subplots()
for i in range(3):
    plt.plot(valid_loss[i, :], label = "lambda = %.3f" %reg_q_four[i])
pic_part_three_one.legend()
pic_part_three_one.set_xlabel("Iterations")
plt.title("valid loss (MSE)")
fig, pic_part_three_one = plt.subplots()
for i in range(3):
    plt.plot(test_loss[i, :], label = "lambda = %.3f" %reg_q_four[i])
pic_part_three_one.legend()
pic_part_three_one.set_xlabel("Iterations")
plt.title("test loss (MSE)")
  
#plot the accuracies
fig, pic_part_three_one = plt.subplots()
for i in range(3):
    plt.plot(train_acc[i, :], label = "lambda = %.3f" %reg_q_four[i])
pic_part_three_one.legend()
pic_part_three_one.set_xlabel("Iterations")
plt.title(" train accuracies (MSE)")
fig, pic_part_three_one = plt.subplots()
for i in range(3):
    plt.plot(valid_acc[i, :], label = "lambda = %.3f" %reg_q_four[i])
pic_part_three_one.legend()
pic_part_three_one.set_xlabel("Iterations")
plt.title(" valid accuracies (MSE)")
fig, pic_part_three_one = plt.subplots()
for i in range(3):
    plt.plot(test_acc[i, :], label = "lambda = %.3f" %reg_q_four[i])
pic_part_three_one.legend()
pic_part_three_one.set_xlabel("Iterations")
plt.title(" test accuracies (MSE)")

#Part 1 End

# =============================================================================
# #Part 2 CE---------------------------------------------------------------------
# 
# #Shape the traning data
# trainData = np.reshape(trainData,(3500, 784))
# trainTarget = np.reshape(trainTarget, (3500))
# validData = np.reshape(validData, (100, 784))
# validTarget = np.reshape(validTarget, (100))
# testData = np.reshape(testData,(145, 784))
# testTarget = np.reshape(testTarget, (145))
# 
# #Training parameters
# epochs = 5000
# alpha = 0.005
# reg = 0.1
# error_tol=10e-7
# 
# #Variables to collect info
# train_loss = np.empty([3, epochs + 1])
# valid_loss = np.empty([3, epochs + 1])
# test_loss = np.empty([3, epochs + 1])
# train_acc = np.empty([3, epochs + 1])
# valid_acc = np.empty([3, epochs + 1])
# test_acc = np.empty([3, epochs + 1])
# 
# #Start training
# W = np.zeros(28*28)
# b = 0
# w_trained, b_trained, train_loss[0,:], valid_loss[0,:], test_loss[0,:], train_acc[0,:], valid_acc[0,:], test_acc[0,:] = grad_descent(W, b, trainData, trainTarget, alpha, epochs, reg, error_tol, validData, validTarget, testData, testTarget, "CE")
# 
# #plot the losses
# fig, pic_part_three_one = plt.subplots()
# plt.plot(train_loss[0, :], label = "train loss")
# plt.plot(valid_loss[0, :], label = "valid loss")
# plt.plot(test_loss[0, :], label = "test loss")
# pic_part_three_one.legend()
# pic_part_three_one.set_xlabel("Iterations")
# plt.title("Losses (CE)")
# 
# #plot the accuracies
# fig, pic_part_three_one = plt.subplots()
# plt.plot(train_acc[0, :], label = "train accuracy")
# plt.plot(valid_acc[0, :], label = "valid accuracy")
# plt.plot(test_acc[0, :], label = "test accuracy")
# pic_part_three_one.legend()
# pic_part_three_one.set_xlabel("Iterations")
# plt.title("Accuracies (CE)")
# 
# #print the final accuracies
# print("fianl training accuracy is %.6f" %train_accuracy_array[0,epochs-1])
# print("fianl validation accuracy is %.6f" %valid_accuracy_array[0,epochs-1])
# print("fianl testing accuracy is %.6f" %test_accuracy_array[0,epochs-1])
# 
# #Part 2 End
# =============================================================================
  
# =============================================================================
# #Part 3 SGD--------------------------------------------------------------------
# 
# #Shape the traning data
# trainData = np.reshape(trainData,(3500, 784))
# validData = np.reshape(validData, (100, 784))
# testData = np.reshape(testData,(145, 784))
# trainTarget = np.reshape(trainTarget, (3500,1))
# validTarget = np.reshape(validTarget, (100,1))
# testTarget = np.reshape(testTarget, (145,1))
# 
# #adam parameters
# beta1=0.99 #0.95, 0.99
# beta2=0.9999 #0.99, 0.9999
# epsilon=1e-4 #1e-9, 1e-4
# 
# #training setting
# batch_size = 100
# epochs = 700
# batch_number = int(len(trainTarget)/batch_size)
# loss_type = "ADAM_MSE" #set the loss function, either "ADAM_CE" or "ADAM_MSE"
# 
# #Set the var to collect the loss and accuracy
# train_loss_array = np.empty([3, epochs])
# valid_loss_array = np.empty([3, epochs])
# test_loss_array = np.empty([3, epochs])
# train_accuracy_array = np.empty([3, epochs])
# valid_accuracy_array = np.empty([3, epochs])
# test_accuracy_array = np.empty([3, epochs])
# 
# W, b, x, y_target, y_predicted, total_loss, train, reg = buildGraph(loss_type, reg = 0.0) #change adam parameters here
# init = tf.global_variables_initializer()
# 
# with tf.Session() as sess:
#     sess.run(init)
#     for epo in range(epochs):
#         for i in range(batch_number):
#             batch_trainData = trainData[batch_size*i:batch_size*(i+1), :]
#             batch_trainTarget = trainTarget[batch_size*i:batch_size*(i+1), :]
#             _,tn_loss, temp_predicted_value = sess.run([train, total_loss, y_predicted], feed_dict={x: batch_trainData, y_target: batch_trainTarget})
#         #get the losses    
#         train_loss_array[0,epo] = sess.run(total_loss, feed_dict={x: trainData, y_target: trainTarget})
#         valid_loss_array[0,epo] = sess.run(total_loss, feed_dict={x: validData, y_target: validTarget})
#         test_loss_array[0,epo] = sess.run(total_loss, feed_dict={x: testData, y_target: testTarget})
#         #get the accuracy 
#         temp_predicted_value = sess.run(y_predicted, feed_dict={x: trainData, y_target: trainTarget})
#         train_accuracy_array[0,epo] = calculate_accuracy(temp_predicted_value, trainTarget)
#         temp_predicted_value = sess.run(y_predicted, feed_dict={x: validData, y_target: validTarget})
#         valid_accuracy_array[0,epo] = calculate_accuracy(temp_predicted_value, validTarget)
#         temp_predicted_value = sess.run(y_predicted, feed_dict={x: testData, y_target: testTarget})
#         test_accuracy_array[0,epo] = calculate_accuracy(temp_predicted_value, testTarget)
#         #shuffle the train set
#         np.random.seed(421+epo)
#         np.random.shuffle(trainData)
#         np.random.seed(421+epo)
#         np.random.shuffle(trainTarget)        
# 
# #plot the losses
# fig, pic_part_three_one = plt.subplots()
# plt.plot(train_loss_array[0,:], label = "train loss")
# plt.plot(valid_loss_array[0,:], label = "valid loss")
# plt.plot(test_loss_array[0,:], label = "test loss")
# pic_part_three_one.legend()
# pic_part_three_one.set_xlabel("Iterations")
# plt.title("Losses (%s) Batch Size = %d" %(loss_type, batch_size))
# 
# #plot the accuracies
# fig, pic_part_three_one = plt.subplots()
# plt.plot(train_accuracy_array[0,:], label = "train accuracy")
# plt.plot(valid_accuracy_array[0,:], label = "valid accuracy")
# plt.plot(test_accuracy_array[0,:], label = "test accuracy")
# pic_part_three_one.legend()
# pic_part_three_one.set_xlabel("Iterations")
# plt.title("Accuracies (%s) Batch Size = %d" %(loss_type, batch_size))
# 
# #print the final accuracies
# print("fianl training accuracy is %.6f" %train_accuracy_array[0,epochs-1])
# print("fianl validation accuracy is %.6f" %valid_accuracy_array[0,epochs-1])
# print("fianl testing accuracy is %.6f" %test_accuracy_array[0,epochs-1])
#
# #Part 3 End
# =============================================================================
