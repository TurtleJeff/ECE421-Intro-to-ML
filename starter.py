import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
#import the data
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
    

def test_MSE():
    W = np.array([2, 0, 1])
    x = np.array([[1,2,1], [1,1,3], [1,1,4], [4, 1, 1]])
    b = 1
    y = np.array([2, 7, 5, 9])
    reg = 0.1
    print(MSE(W, b, x, y, reg))
    return()


def gradMSE(W, b, x, y, reg):
    grad_W = np.matmul(np.matmul(x, W) + b - y, x) * 2 / len(y) + reg*W
    grad_b = (np.matmul(x, W) + b - y).sum() * 2/ len(y)
    return grad_W, grad_b


def test_gradMSE():
    W = np.array([2, 0, 1])
    x = np.array([[1,2,1], [1,1,3], [1,1,4], [4, 1, 1]])
    b = 1
    y = np.array([2, 7, 5, 9])
    reg = 0.1
    print(gradMSE(W, b, x, y, reg))
    return()
    

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
        W_best = W
        b_best = b
        for i in range(epochs):
            gradMSE_W, gradMSE_b = gradMSE(W, b, x, y, reg)
            W_new = W - gradMSE_W * alpha
            b_new = b - gradMSE_b * alpha
            train_loss.append(MSE(W_new, b_new, x, y, reg))
            valid_loss.append(MSE(W_new, b_new, validData, validTarget, reg))       
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


def test_crossEntropyLoss():
    W = np.array([2, 0, 1])
    x = np.array([[3,2,-2], [1,1,-3], [-2,1,4], [4, 1, -7.5]])
    b = 0
    y = np.array([1, 1, 0, 1])
    reg = 0.1
    print(crossEntropyLoss(W, b, x, y, reg))
    return()
    
    
def gradCE(W, b, x, y, reg):
    z = np.matmul(x, W) + b
    #gradCE_W = np.matmul(- y/(1+np.exp(z)) + (1-y)/(1+np.exp(-z)),x)/len(y) + 2*reg*W
    gradCE_W = np.matmul(1/(1+np.exp(-z)) - y, x)/len(y) + 2*reg*W
    #gradCE_b = (-y/(1+np.exp(np.matmul(x, W) + b)) + (1-y)/(1+np.exp(-np.matmul(x, W) - b))).sum()/len(y)
    gradCE_b = (1/(1+np.exp(-z)) - y).sum()/len(y)
    return gradCE_W, gradCE_b


def test_gradCE():
    W = np.array([2.3, 0, 1])
    x = np.array([[1,2,-1], [1.4,1,-3], [-2,1,4], [4, 1, -7.5]])
    b = 0.1
    y = np.array([0, 1, 1, 1])
    reg = 0.1
    print(gradCE(W, b, x, y, reg))
    return


def buildGraph(loss="MSE", beta1 = 0.9, beta2 = 0.999, epsilon=1e-08):
# 	#Initialize weight and bias tensors
    tf.set_random_seed(421)
    W = tf.Variable(tf.truncated_normal(shape=[784, 1], stddev=0.5), name="weights")
    b = tf.Variable(0.0, name="biases")
    x = tf.placeholder(tf.float32, [None, 784], name="input_x")
    y_target = tf.placeholder(tf.float32, [None, 1], name="target_y")
    reg = tf.placeholder(tf.float32, name = "reg_parameter") 
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
        total_loss = CE_loss + 0*reg_loss
        optimizer = tf.train.AdamOptimizer(0.001, beta1, beta2, epsilon)
        train = optimizer.minimize(total_loss)
    elif loss == "ADAM_MSE":
        y_predicted = tf.matmul(x,W) + b
        CE_loss = tf.losses.mean_squared_error(y_target, y_predicted)
        reg_loss = tf.nn.l2_loss(W, name = "reg_error")
        total_loss = CE_loss + 0*reg_loss
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

trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
trainData = np.reshape(trainData,(3500, 784))
trainTarget = np.reshape(trainTarget, (3500))
validData = np.reshape(validData, (100, 784))
validTarget = np.reshape(validTarget, (100))
testData = np.reshape(testData,(145, 784))
testTarget = np.reshape(testTarget, (145))


epochs = 5000
train_loss = np.empty([3, epochs + 1])
valid_loss = np.empty([3, epochs + 1])
test_loss = np.empty([3, epochs + 1])

train_acc = np.empty([3, epochs + 1])
valid_acc = np.empty([3, epochs + 1])
test_acc = np.empty([3, epochs + 1])

# Part 1
alpha_q_three = [0.005, 0.001, 0.0001]
reg_q_three = 0
alpha_q_four = 0.005
reg_q_four = [0.001, 0.1, 0.5]
error_tol=10e-7

# =============================================================================
# y = trainTarget
# x = trainData
# W = np.zeros(28*28)
# b = 0
# reg = 0
# kk = np.square(np.matmul(x, W) + b - y).mean() + np.square(W).sum() * reg / 2
# print(kk)
# =============================================================================
# Part 1 Question 3
for i in range(3):
    W = np.zeros(28*28)
    b = 0
    w_trained, b_trained, train_loss[i,:], valid_loss[i,:] = grad_descent(W, b, trainData, trainTarget, alpha_q_three[i], epochs, reg_q_three, error_tol, validData, validTarget)
    #testing
    predicted_value = np.matmul(testData, w_trained) + b_trained
    num_right = 0
    num_wrong = 0
    for j in range(len(testData)):   
        if predicted_value[j] <= 0.5:
            predicted_value[j] = 0
        else:
            predicted_value[j] = 1    
        if predicted_value[j] == testTarget[j]:
            num_right += 1
        else:
            num_wrong += 1
    print("testing result is %d right and %d wrong with learning rate = %.4f" %(num_right, num_wrong, alpha_q_three[i]))

fig, pic_vl_three = plt.subplots()
for i in range(3):
    pic_vl_three.plot(valid_loss[i, :], label = "alpha = %.4f" % alpha_q_three[i])
pic_vl_three.legend()
plt.title("Valid Loss (Q3)")

fig, pic_tl_three = plt.subplots()
for i in range(3):
    pic_tl_three.plot(train_loss[i, :], label = "alpha = %.4f" % alpha_q_three[i])
pic_tl_three.legend()
plt.title("Train Loss (Q3)")

        
# Part 1 Question 4
for i in range(3):
    W = np.zeros(28*28)
    b = 0
    w_trained, b_trained, train_loss[i,:], valid_loss[i,:] = grad_descent(W, b, trainData, trainTarget, alpha_q_four, epochs, reg_q_four[i], error_tol, validData, validTarget)
    #testing
    predicted_value = np.matmul(testData, w_trained) + b_trained
    num_right = 0
    num_wrong = 0
    for j in range(len(testData)):   
        if predicted_value[j] <= 0.5:
            predicted_value[j] = 0
        else:
            predicted_value[j] = 1    
        if predicted_value[j] == testTarget[j]:
            num_right += 1
        else:
            num_wrong += 1
    print("testing result is %d right and %d wrong with regularization parameter = %.4f" %(num_right, num_wrong, reg_q_four[i]))

fig, pic_vl_four = plt.subplots()
for i in range(3):
    pic_vl_four.plot(valid_loss[i, :], label = "regularization parameter = %.4f" % reg_q_four[i])
pic_vl_four.legend()
plt.title("Valid Loss (Q4)")

fig, pic_tl_four = plt.subplots()
for i in range(3):
    pic_tl_four.plot(train_loss[i, :], label = "regularization parameter = %.4f" % reg_q_four[i])
pic_tl_four.legend()
plt.title("Train Loss (Q4)")
# Part 1 End

# =============================================================================
# #Part 2
# alpha = 0.005
# reg = 0.1
# error_tol=10e-7
# 
# W = np.zeros(28*28)
# b = 0
# w_trained, b_trained, train_loss[0,:], valid_loss[0,:], test_loss[0,:], train_acc[0,:], valid_acc[0,:], test_acc[0,:] = grad_descent(W, b, trainData, trainTarget, alpha, epochs, reg, error_tol, validData, validTarget, testData, testTarget, "CE")
# 
# fig, pic_part_three_one = plt.subplots()
# plt.plot(train_loss[0, :], label = "train loss")
# plt.plot(valid_loss[0, :], label = "valid loss")
# plt.plot(test_loss[0, :], label = "test loss")
# pic_part_three_one.legend()
# pic_part_three_one.set_xlabel("Iterations")
# plt.title("Losses (CE)")
# 
# fig, pic_part_three_one = plt.subplots()
# plt.plot(train_acc[0, :], label = "train accuracy")
# plt.plot(valid_acc[0, :], label = "valid accuracy")
# plt.plot(test_acc[0, :], label = "test accuracy")
# pic_part_three_one.legend()
# pic_part_three_one.set_xlabel("Iterations")
# plt.title("Accuracies (CE)")
# =============================================================================

#Part 3 Preperations
trainTarget = np.reshape(trainTarget, (3500,1))
validTarget = np.reshape(validTarget, (100,1))
testTarget = np.reshape(testTarget, (145,1))

# =============================================================================
# #Part 3 Question 2,5
# batch_size = 1750
# epochs = 700
# batch_number = int(len(trainTarget)/batch_size)
# loss_type = "ADAM_MSE"
# 
# train_loss_array = np.empty([3, epochs])
# valid_loss_array = np.empty([3, epochs])
# test_loss_array = np.empty([3, epochs])
# train_accuracy_array = np.empty([3, epochs])
# valid_accuracy_array = np.empty([3, epochs])
# test_accuracy_array = np.empty([3, epochs])
# 
# W, b, x, y_target, y_predicted, total_loss, train = buildGraph(loss_type)
# init = tf.global_variables_initializer()
# 
# with tf.Session() as sess:
#     sess.run(init)
#     k = 0
#     for epoch in range(epochs):
#         for i in range(batch_number):
#             batch_trainData = trainData[batch_size*i:batch_size*(i+1), :]
#             batch_trainTarget = trainTarget[batch_size*i:batch_size*(i+1), :]
#             _,tn_loss = sess.run([train, total_loss], feed_dict={x: batch_trainData, y_target: batch_trainTarget})
#         
#         train_loss_array[0,k] = sess.run(total_loss, feed_dict={x: trainData, y_target: trainTarget})
#         valid_loss_array[0,k] = sess.run(total_loss, feed_dict={x: validData, y_target: validTarget})
#         test_loss_array[0,k] = sess.run(total_loss, feed_dict={x: testData, y_target: testTarget})
#         
#         temp_predicted_value = sess.run(y_predicted, feed_dict={x: trainData, y_target: trainTarget})
#         train_accuracy_array[0,k] = calculate_accuracy(temp_predicted_value, trainTarget)
#         temp_predicted_value = sess.run(y_predicted, feed_dict={x: validData, y_target: validTarget})
#         valid_accuracy_array[0,k] = calculate_accuracy(temp_predicted_value, validTarget)
#         temp_predicted_value = sess.run(y_predicted, feed_dict={x: testData, y_target: testTarget})
#         test_accuracy_array[0,k] = calculate_accuracy(temp_predicted_value, testTarget)            
#         k +=1
#     np.random.shuffle(trainTarget)
#     state = np.random.get_state()
#     np.random.set_state(state)
#     np.random.shuffle(trainData)
# 
# fig, pic_part_three_one = plt.subplots()
# plt.plot(train_loss_array[0,:], label = "train loss")
# plt.plot(valid_loss_array[0,:], label = "valid loss")
# plt.plot(test_loss_array[0,:], label = "test loss")
# pic_part_three_one.legend()
# pic_part_three_one.set_xlabel("Iterations")
# plt.title("Losses with batch size %d (ADAM_MSE) (P3Q3)" %batch_size)
# 
# fig, pic_part_three_two = plt.subplots()
# plt.plot(train_accuracy_array[0,:], label = "train accuracy")
# plt.plot(valid_accuracy_array[0,:], label = "valid accuracy")
# plt.plot(test_accuracy_array[0,:], label = "test accuracy")
# pic_part_three_two.legend()
# pic_part_three_two.set_xlabel("Iterations")
# plt.title("Accuracies with batch size %d (ADAM_MSE) (P3Q3)" %batch_size)
# 
# print(train_accuracy_array[0,699])
# print(valid_accuracy_array[0,699])
# print(test_accuracy_array[0,699])
# #Part 3 Question 2,5 End
# =============================================================================

# =============================================================================
# #Part 3 Question 3,5
# batch_size = np.array([100, 700, 1750])
# epochs = 500
# 
# train_loss_array = np.empty([3, epochs])
# valid_loss_array = np.empty([3, epochs])
# test_loss_array = np.empty([3, epochs])
# train_accuracy_array = np.empty([3, epochs])
# valid_accuracy_array = np.empty([3, epochs])
# test_accuracy_array = np.empty([3, epochs])
# 
# batch_number = len(trainTarget)/batch_size
# loss_type = "MSE"
# W, b, x, y_target, y_predicted, total_loss, train = buildGraph(loss_type)
# init = tf.global_variables_initializer()
# 
# with tf.Session() as sess:
#     for batch_index in range(3):
#         sess.run(init)
#         k = 0
#         for epoch in range(epochs):
#             for i in range(int(batch_number[batch_index])):
#                 batch_trainData = trainData[batch_size[batch_index]*i:batch_size[batch_index]*(i+1), :]
#                 batch_trainTarget = trainTarget[batch_size[batch_index]*i:batch_size[batch_index]*(i+1), :]
#                 _,tn_loss = sess.run([train, total_loss], feed_dict={x: batch_trainData, y_target: batch_trainTarget})
#         
#             train_loss_array[batch_index,k] = sess.run(total_loss, feed_dict={x: trainData, y_target: trainTarget})
#             valid_loss_array[batch_index,k] = sess.run(total_loss, feed_dict={x: validData, y_target: validTarget})
#             test_loss_array[batch_index,k] = sess.run(total_loss, feed_dict={x: testData, y_target: testTarget})
#         
#             temp_predicted_value = sess.run(y_predicted, feed_dict={x: trainData, y_target: trainTarget})
#             train_accuracy_array[batch_index,k] = calculate_accuracy(temp_predicted_value, trainTarget)
#             temp_predicted_value = sess.run(y_predicted, feed_dict={x: validData, y_target: validTarget})
#             valid_accuracy_array[batch_index,k] = calculate_accuracy(temp_predicted_value, validTarget)
#             temp_predicted_value = sess.run(y_predicted, feed_dict={x: testData, y_target: testTarget})
#             test_accuracy_array[batch_index,k] = calculate_accuracy(temp_predicted_value, testTarget)            
#             k +=1
#         np.random.shuffle(trainTarget)
#         state = np.random.get_state()
#         np.random.set_state(state)
#         np.random.shuffle(trainData)
# 
# for i in range(3):
#     fig, pic_part_three_one = plt.subplots()
#     plt.plot(train_loss_array[i,:], label = "train loss")
#     plt.plot(valid_loss_array[i,:], label = "valid loss")
#     plt.plot(test_loss_array[i,:], label = "test loss")
#     pic_part_three_one.legend()
#     plt.title("Losses with batch size %d (P3Q3)" %(batch_size[i]))
# 
#     fig, pic_part_three_one = plt.subplots()
#     plt.plot(train_accuracy_array[i,:], label = "train accuracy")
#     plt.plot(valid_accuracy_array[i,:], label = "valid accuracy")
#     plt.plot(test_accuracy_array[i,:], label = "test accuracy")
#     pic_part_three_one.legend()
#     plt.title("Accuracies with batch size %d (P3Q3)" %(batch_size[i]))
# #Part 3 Question 3,5 End   
# =============================================================================

# =============================================================================
# #Part 3 Question 4
# beta1=0.99 #0.95, 0.99
# beta2=0.9999 #0.99, 0.9999
# epsilon=1e-4 #1e-9, 1e-4
# 
# batch_size = 100
# epochs = 700
# regulation = 0
# batch_number = int(len(trainTarget)/batch_size)
# loss_type = "ADAM_CE"
# 
# train_loss_array = np.empty([3, epochs])
# valid_loss_array = np.empty([3, epochs])
# test_loss_array = np.empty([3, epochs])
# train_accuracy_array = np.empty([3, epochs])
# valid_accuracy_array = np.empty([3, epochs])
# test_accuracy_array = np.empty([3, epochs])
# 
# W, b, x, y_target, y_predicted, total_loss, train, reg = buildGraph(loss_type)
# init = tf.global_variables_initializer()
# 
# with tf.Session() as sess:
#     sess.run(init)
#     for epo in range(epochs):
#         for i in range(batch_number):
#             batch_trainData = trainData[batch_size*i:batch_size*(i+1), :]
#             batch_trainTarget = trainTarget[batch_size*i:batch_size*(i+1), :]
#             _,tn_loss, temp_predicted_value = sess.run([train, total_loss, y_predicted], feed_dict={x: batch_trainData, y_target: batch_trainTarget, reg: regulation})
#         train_accuracy = calculate_accuracy(temp_predicted_value, batch_trainTarget)
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
# # =============================================================================
# #         np.random.shuffle(trainData)
# #         state = np.random.get_state()
# #         np.random.set_state(state)
# #         np.random.shuffle(trainTarget)
# # =============================================================================
# 
# 
# fig, pic_part_three_one = plt.subplots()
# plt.plot(train_loss_array[0,:], label = "train loss")
# plt.plot(valid_loss_array[0,:], label = "valid loss")
# plt.plot(test_loss_array[0,:], label = "test loss")
# pic_part_three_one.legend()
# pic_part_three_one.set_xlabel("Iterations")
# plt.title("Losses (ADAM_CE) Batch Size = %.d beta2=0.9999" %batch_size)
# 
# fig, pic_part_three_one = plt.subplots()
# plt.plot(train_accuracy_array[0,:], label = "train accuracy")
# plt.plot(valid_accuracy_array[0,:], label = "valid accuracy")
# plt.plot(test_accuracy_array[0,:], label = "test accuracy")
# pic_part_three_one.legend()
# pic_part_three_one.set_xlabel("Iterations")
# plt.title("Accuracies (ADAM_CE) Batch Size = %.d beta2=0.9999" %batch_size)
# 
# print(train_accuracy_array[0,epochs-1])
# print(valid_accuracy_array[0,epochs-1])
# print(test_accuracy_array[0,epochs-1])
# #Part 3 Question 4 End
# =============================================================================
