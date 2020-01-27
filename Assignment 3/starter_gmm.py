import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp

# Loading data
data = np.load('data100D.npy')
#data = np.load('data2D.npy')
[num_pts, dim] = np.shape(data)

# For Validation set
is_valid = True #True for part 2 and part 3
if is_valid:
  valid_batch = int(num_pts / 3.0)
  np.random.seed(45689)
  rnd_idx = np.arange(num_pts)
  np.random.shuffle(rnd_idx)
  val_data = data[rnd_idx[:valid_batch]]
  data = data[rnd_idx[valid_batch:]]

def distanceFunc(X, MU):
    X = tf.expand_dims(X, 1)
    MU = tf.expand_dims(MU, 0)
    pair_dist = tf.reduce_sum(tf.square(tf.subtract(X, MU)), 2)
    return pair_dist

def log_GaussPDF(X, mu, sigma):
    part_a = -0.5*dim*tf.transpose(tf.log(2*np.pi*tf.square(sigma)))
    part_b = tf.divide(distanceFunc(X, mu), -2*tf.transpose(tf.square(sigma)))
    return part_a + part_b  

def log_posterior(log_PDF, log_pi):
    return tf.add(log_PDF, tf.transpose(log_pi)) - hlp.reduce_logsumexp(tf.add(log_PDF, tf.transpose(log_pi)), 1, keep_dims=True)

def GMM(K):
    # define the model parameters
    MU = tf.Variable(tf.truncated_normal(shape=[K, dim]), name="MU", )
    
    sigma = tf.Variable(tf.truncated_normal(shape=[K, 1]), name="sigma")
    sigma = tf.exp(sigma)
    
    log_pi = tf.Variable(tf.truncated_normal(shape=[K, 1]), name="pi")
    log_pi = hlp.logsoftmax(log_pi)
    
    # input data
    X = tf.placeholder(tf.float32, [None, dim], name = 'data');
    
    # call the log_PDF
    log_PDF = log_GaussPDF(X, MU, sigma)
    
    # find the most possible cluster for each point 
    log_post = log_posterior(log_PDF, log_pi)
    assigned = tf.argmax(log_post, 1)
    
    # define the loss function #print(loss.get_shape().as_list())
    loss = tf.add(log_PDF, tf.transpose(log_pi))
    loss = hlp.reduce_logsumexp(loss)
    loss = -1*tf.reduce_sum(loss)
    
    # train the model
    train = tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9, beta2=0.99, epsilon=1e-5).minimize(loss)
    
    return MU, sigma, log_pi, X, assigned, loss, train


################################################################################################################
#start the training

#K_array = [1, 2, 3, 4, 5] #Part 2
K_array = [5, 10, 15, 20, 30] #Part 3
K = 10#K_array[1];
updates = 500

train_loss_array = np.empty([updates])
valid_loss_array = np.empty([updates])

MU, sigma, log_pi, X, assigned, loss, train = GMM(K);

with tf.Session() as sess:
    
    init = tf.global_variables_initializer()
    sess.run(init)
    
    for i in range(updates):
        opt = sess.run(train, feed_dict={X: data})
        
        train_loss_array[i] = sess.run(loss, feed_dict={X: data})
        valid_loss_array[i] = sess.run(loss, feed_dict={X: val_data})
        
    means = sess.run(MU, feed_dict={X: data})
    
    sd = sess.run(sigma, feed_dict={X: data})
    var = np.square(sd)
    
    pi = sess.run(log_pi, feed_dict={X: data})
    pi = np.exp(pi)
    
    cluster_index = sess.run(assigned, feed_dict={X: data})


############################################################################################################    

# =============================================================================
# #plot the point map
# colors = ["b", "y", "r", "g", "m"]
# data_colors = []
# for i in range(0, data.shape[0]):
#     data_colors.append(colors[cluster_index[i]])
# plt.figure(1, (24,18))
# plt.title("Data Points, K = {}".format(K), fontsize=34)
# plt.scatter(data[:,0], data[:,1], c = data_colors) 
# plt.ylabel("y", fontsize=24)
# plt.xlabel("x", fontsize=24)
# =============================================================================

#plot the losses
plt.figure(2, (24,18))
plt.title("Losses, K = {}".format(K), fontsize=34)
train_loss_array = train_loss_array/(2.0*num_pts/3.0)
valid_loss_array = valid_loss_array/(num_pts/3.0)
plt.plot(train_loss_array, label = "train loss")
plt.plot(valid_loss_array, label = "valid loss")
plt.ylabel("loss", fontsize=24)
plt.xlabel("updates", fontsize=24)      
plt.legend(fontsize=24)


# =============================================================================
# #plot the loss for part 1
# plt.figure(2, (24,18))
# plt.title("Loss", fontsize=34)
# plt.plot(train_loss_array)#label = "train loss"
# plt.ylabel("train loss", fontsize=24)
# plt.xlabel("updates", fontsize=24)   
# =============================================================================
    