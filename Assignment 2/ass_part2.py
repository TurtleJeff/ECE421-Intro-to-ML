import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #only shows the errors

# Load the data
def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest

#Implementation of Neural Network using Tensorflow
def CNN(reg = 0, p = 1):
    tf.reset_default_graph() #print(tensor.get_shape().as_list())
    
    #Trainging Model
    Wf = tf.get_variable('Wf', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer())
    bf = tf.get_variable('bf', shape=(32), initializer=tf.contrib.layers.xavier_initializer())
    Wh = tf.get_variable('Wh', shape=(14*14*32,784), initializer=tf.contrib.layers.xavier_initializer())
    bh = tf.get_variable('bh', shape=(784), initializer=tf.contrib.layers.xavier_initializer())
    Wo = tf.get_variable('Wo', shape=(784, 10), initializer=tf.contrib.layers.xavier_initializer())
    bo = tf.get_variable('bo', shape=(10), initializer=tf.contrib.layers.xavier_initializer())
    
    #Training Data Set
    x = tf.placeholder(tf.float32, [None, 28, 28, 1], name="input_x")
    y = tf.placeholder(tf.float32, [None, 10], name="target_y")
    
    #2. Convolution Layer
    conv_layer = tf.nn.conv2d(input = x, filter = Wf, strides=[1, 1, 1, 1], padding='SAME')#'Same' keeps the output shape as the input by adding 0s
    conv_layer = tf.nn.bias_add(conv_layer,bf)
    #3. ReLU Activation
    conv_layer = tf.nn.relu(conv_layer)
    #4. Batch Normalization(simple batch normalization)
    batchMean, batchVar = tf.nn.moments(conv_layer,[0]) #range(len(shape)-1)
    nor_layer = tf.nn.batch_normalization(conv_layer,batchMean,batchVar,offset = None, scale = None,variance_epsilon=1e-3)
    #5. Max Pooling Layer
    maxpool_layer = tf.nn.max_pool(nor_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #6. Flatten layer
    flatten_layer = tf.reshape(maxpool_layer, [-1, Wh.get_shape().as_list()[0]])
    #7. Fully Connected Layer 1
    FC1_layer = tf.add(tf.matmul(flatten_layer, Wh), bh)
    #Optional Dropout
    FC1_layer = tf.nn.dropout(FC1_layer, p)
    #8. ReLU Activation
    relu_layer = tf.nn.relu(FC1_layer)
    #9. Fully Connected Layer 2
    FC2_layer = tf.add(tf.matmul(relu_layer, Wo), bo)
    #10. Softmax Output
    output = tf.nn.softmax(FC2_layer)
    #Optional Regularization
    reg_loss = tf.nn.l2_loss(Wh) + tf.nn.l2_loss(Wo)
    #11. Cross Entropy Loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))+reg*reg_loss
    
    train = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
    
    #Accuracy Calculation
    compared_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(compared_prediction, tf.float32))
    
    return Wf, bf, Wh, bh, Wo, bo, x, y, train, loss, accuracy

trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
newtrain, newvalid, newtest = convertOneHot(trainTarget, validTarget, testTarget)

#Reshape the Data
trainData = trainData.reshape(-1, 28, 28, 1)
testData = testData.reshape(-1, 28, 28, 1)
validData = validData.reshape(-1, 28, 28, 1)

batch_size = 32
epochs = 2

weight_decay_coeff = [0.01, 0.1, 0.5]
dropout_prop = [0.9, 0.75, 0.5]

#Arrays to collect loss and accuracy
train_loss_array = np.empty([3, epochs])
valid_loss_array = np.empty([3, epochs])
test_loss_array = np.empty([3, epochs])
train_acc_array = np.empty([3, epochs])
valid_acc_array = np.empty([3, epochs])
test_acc_array = np.empty([3, epochs])


for k in range(3):
    reg = 0
    p = dropout_prop[k]
    Wf, bf, Wh, bh, Wo, bo, x, y, train, loss, accuracy = CNN(reg, p)

    with tf.Session() as sess:
    
        init = tf.global_variables_initializer()
        sess.run(init)
    
        for epo_num in range(epochs):
            for i in range(len(trainData)//batch_size):
                batch_trainData = trainData[batch_size*i:batch_size*(i+1), :]
                batch_trainTarget = newtrain[batch_size*i:batch_size*(i+1), :]
                opt = sess.run(train, feed_dict={x: batch_trainData, y: batch_trainTarget})
            
            train_loss_value, train_acc_value = sess.run([loss, accuracy], feed_dict={x: trainData, y: newtrain})
            valid_loss_value, valid_acc_value = sess.run([loss, accuracy], feed_dict={x: validData, y: newvalid})
            test_loss_value, test_acc_value = sess.run([loss, accuracy], feed_dict={x: testData, y: newtest})
        
            train_loss_array[k,epo_num] = (train_loss_value)
            train_acc_array[k,epo_num] = (train_acc_value)
            valid_loss_array[k,epo_num] = (valid_loss_value)
            valid_acc_array[k,epo_num] = (valid_acc_value)
            test_loss_array[k,epo_num] = (test_loss_value)
            test_acc_array[k,epo_num] = (test_acc_value)
            #Shuffle the Data
            np.random.seed(421+epo_num)
            np.random.shuffle(trainData)
            np.random.seed(421+epo_num)
            np.random.shuffle(newtrain) 
        
            print("Iteration: %d, Loss: %.4f Acc: %.4f" %(epo_num, train_loss_value, train_acc_value))
        print("training completed")

# =============================================================================
# fig, pic_loss = plt.subplots()
# plt.plot(train_loss_array[0,:], label = "train loss")
# plt.plot(valid_loss_array[0,:], label = "valid loss")
# plt.plot(test_loss_array[0,:], label = "test loss")
# pic_loss.legend()
# pic_loss.set_xlabel("Iterations")
# plt.title("Losses")
# 
# fig, pic_acc = plt.subplots()
# plt.plot(train_acc_array[0,:], label = "train accuracy")
# plt.plot(valid_acc_array[0,:], label = "valid accuracy")
# plt.plot(test_acc_array[0,:], label = "test accuracy")
# pic_acc.legend()
# pic_acc.set_xlabel("Iterations")
# plt.title("Accuracies")
# =============================================================================

# =============================================================================
# fig, pic_acc = plt.subplots()
# plt.plot(train_acc_array[0,:], label = "p: 0.95")
# plt.plot(train_acc_array[1,:], label = "p: 0.75")
# plt.plot(train_acc_array[2,:], label = "p: 0.5")
# pic_acc.legend()
# pic_acc.set_xlabel("Iterations")
# plt.title("Train Accuracies")
# 
# fig, pic_acc = plt.subplots()
# plt.plot(valid_acc_array[0,:], label = "p: 0.95")
# plt.plot(valid_acc_array[1,:], label = "p: 0.75")
# plt.plot(valid_acc_array[2,:], label = "p: 0.5 ")
# pic_acc.legend()
# pic_acc.set_xlabel("Iterations")
# plt.title("Valid Accuracies")
# 
# fig, pic_acc = plt.subplots()
# plt.plot(test_acc_array[0,:], label = "p: 0.95")
# plt.plot(test_acc_array[1,:], label = "p: 0.75")
# plt.plot(test_acc_array[2,:], label = "p: 0.5")
# pic_acc.legend()
# pic_acc.set_xlabel("Iterations")
# plt.title("Test Accuracies")
# =============================================================================



    
    
    
    
    
    
    
    
    
    
    
    
    
