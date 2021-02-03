from __future__ import print_function, division
import os
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import datetime

# GPU settings
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# get the base graph and generator matrix
code_PCM0 = np.loadtxt("./BaseGraph/BaseGraph2_Set0.txt", int, delimiter='	')
code_PCM1 = np.loadtxt("./BaseGraph/BaseGraph2_Set1.txt", int, delimiter='	')
code_PCM2 = np.loadtxt("./BaseGraph/BaseGraph2_Set2.txt", int, delimiter='	')
code_GM_16 = np.loadtxt("./BaseGraph_GM/LDPC_GM_BG2_16.txt", int, delimiter=',')
code_GM_3 = np.loadtxt("./BaseGraph_GM/LDPC_GM_BG2_3.txt", int, delimiter=',')
code_GM_10 = np.loadtxt("./BaseGraph_GM/LDPC_GM_BG2_10.txt", int, delimiter=',')

code_PCM = code_PCM0.copy()
Ldpc_PCM = [code_PCM0, code_PCM1, code_PCM2]# three LDPC codes with different code lengths
Ldpc_GM = [code_GM_16, code_GM_3, code_GM_10]
Z_array = np.array([16, 3, 10])

N = 52
m = 42
code_n = N
code_k = N - m
for i in range(0, code_PCM.shape[0]):
    for j in range(0, code_PCM.shape[1]):
        if (code_PCM[i, j] == -1):
            code_PCM[i, j] = 0
        else:
            code_PCM[i, j] = 1

# network hyper-parameters
iters_max = 5     # number of iterations
sum_edge_c = np.sum(code_PCM, axis=1)
sum_edge_v = np.sum(code_PCM, axis=0)
sum_edge = np.sum(sum_edge_v)
neurons_per_even_layer = neurons_per_odd_layer = np.sum(sum_edge_v)
input_output_layer_size = N

# init the AWGN #
code_rate = 1.0 * (N - m) / (N-2)
# train SNR
SNR_Matrix = np.array([[9.0,6.05,4.1,2.95,2.25,1.8,1.55,1.3,1.15,1.05,0.94,0.85,0.83,0.81,0.8,0.8,0.8,0.75,0.75,0.7,0.7,0.7,0.7,0.7,0.7],
                       [9.1,6.2,4.6,3.7,3.2,3.0,2.8,2.7,2.6,2.55,2.5,2.45,2.4,2.4,2.4,2.35,2.35,2.3,2.3,2.3,2.25,2.25,2.25,2.25,2.25],
                       [9,6.05,4.1,3,2.4,2,1.7,1.5,1.4,1.4,1.3,1.3,1.2,1.2,1.2,1.2,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1,1]])
SNR_lin = 10.0 ** (SNR_Matrix / 10.0)
SNR_sigma = np.sqrt(1.0 / (2.0 * SNR_lin * code_rate))
# ramdom seed
word_seed = 2042
noise_seed = 1074
wordRandom = np.random.RandomState(word_seed)  # word seed
noiseRandom = np.random.RandomState(noise_seed)  # noise seed

# train settings
single_damping = False
learning_rate = 0.001
train_on_zero_word = True
numOfWordSim_train = 30
batch_size = numOfWordSim_train
num_of_batch = 10000

#get train samples
def create_mix_epoch(scaling_factor, wordRandom, noiseRandom, numOfWordSim, code_n, code_k, Z, code_GM, is_zeros_word):
    X = np.zeros([1, code_n * Z], dtype=np.float32)
    Y = np.zeros([1, code_n * Z], dtype=np.int64)

    # build set for epoch
    for sf_i in scaling_factor:
        if is_zeros_word:
            infoWord_i = 0 * wordRandom.randint(0, 2, size=(numOfWordSim, code_k * Z))
        else:
            infoWord_i = wordRandom.randint(0, 2, size=(numOfWordSim, code_k * Z))

        Y_i = np.dot(infoWord_i, code_GM) % 2
        X_p_i = noiseRandom.normal(0.0, 1.0, Y_i.shape) * sf_i + (-1) ** (1 - Y_i)  # pay attention to this 1->1 0->-1
        x_llr_i = 2 * X_p_i / ((sf_i) ** 2)  # defined as p1/p0
        X = np.vstack((X, x_llr_i))
        Y = np.vstack((Y, Y_i))
    X = X[1:]
    Y = Y[1:]
    X = np.reshape(X, [batch_size, code_n, Z])
    return X, Y

# calculate ber and fer
def calc_ber_fer(snr_db, Y_test_pred, Y_test, numOfWordSim):
    ber_test = np.zeros(snr_db.shape[0])
    fer_test = np.zeros(snr_db.shape[0])
    for i in range(0, snr_db.shape[0]):
        Y_test_pred_i = Y_test_pred[i * numOfWordSim:(i + 1) * numOfWordSim, :]
        Y_test_i = Y_test[i * numOfWordSim:(i + 1) * numOfWordSim, :]
        ber_test[i] = np.abs(((Y_test_pred_i > 0) - Y_test_i)).sum() / (Y_test_i.shape[0] * Y_test_i.shape[1])
        fer_test[i] = (np.abs(((Y_test_pred_i > 0) - Y_test_i)).sum(axis=1) > 0).sum() * 1.0 / Y_test_i.shape[0]
    return ber_test, fer_test


############################     init the connecting matrix between network layers   #################################
Lift_Matrix1 = []
Lift_Matrix2 = []
W_odd2even = np.zeros((sum_edge, sum_edge), dtype=np.float32)
W_skipconn2even = np.zeros((N, sum_edge), dtype=np.float32)
W_even2odd = np.zeros((sum_edge, sum_edge), dtype=np.float32)
W_output = np.zeros((sum_edge, N), dtype=np.float32)

# init lifting matrix for cyclic shift
for t in range(0, 3, 1):
    Lift_M1 = np.zeros((neurons_per_odd_layer * Z_array[t], neurons_per_odd_layer * Z_array[t]), np.float32)
    Lift_M2 = np.zeros((neurons_per_odd_layer * Z_array[t], neurons_per_odd_layer * Z_array[t]), np.float32)
    code_PCM1 = Ldpc_PCM[t]
    k = 0
    for j in range(0, code_PCM1.shape[1]):
        for i in range(0, code_PCM1.shape[0]):
            if (code_PCM1[i, j] != -1):
                Lift_num = code_PCM1[i, j] % Z_array[t]
                for h in range(0, Z_array[t], 1):
                    Lift_M1[k * Z_array[t] + h, k * Z_array[t] + (h + Lift_num) % Z_array[t]] = 1
                k = k + 1
    k = 0
    for i in range(0, code_PCM1.shape[0]):
        for j in range(0, code_PCM1.shape[1]):
            if (code_PCM1[i, j] != -1):
                Lift_num = code_PCM1[i, j] % Z_array[t]
                for h in range(0, Z_array[t], 1):
                    Lift_M2[k * Z_array[t] + h, k * Z_array[t] + (h + Lift_num) % Z_array[t]] = 1
                k = k + 1
    Lift_Matrix1.append(Lift_M1)
    Lift_Matrix2.append(Lift_M2)

# init W_odd2even  variable node updating
k = 0
vec_tmp = np.zeros((sum_edge), dtype=np.float32)  # even layer index read with column
for j in range(0, code_PCM.shape[1], 1):  # run over the columns
    for i in range(0, code_PCM.shape[0], 1):  # break after the first one
        if (code_PCM[i, j] == 1):  # finding the first one is ok
            num_of_conn = int(np.sum(code_PCM[:, j]))  # get the number of connection of the variable node
            idx = np.argwhere(code_PCM[:, j] == 1)  # get the indexes
            for l in range(0, num_of_conn, 1):  # adding num_of_conn columns to W
                vec_tmp = np.zeros((sum_edge), dtype=np.float32)
                for r in range(0, code_PCM.shape[0], 1):  # adding one to the right place
                    if (code_PCM[r, j] == 1 and idx[l][0] != r):
                        idx_row = np.cumsum(code_PCM[r, 0:j + 1])[-1] - 1
                        odd_layer_node_count = 0
                        if r > 0:
                            odd_layer_node_count = np.cumsum(sum_edge_c[0:r])[-1]
                        vec_tmp[idx_row + odd_layer_node_count] = 1  # offset index adding
                W_odd2even[:, k] = vec_tmp.transpose()
                k += 1
            break

# init W_even2odd  parity check node updating
k = 0
for j in range(0, code_PCM.shape[1], 1):
    for i in range(0, code_PCM.shape[0], 1):
        if (code_PCM[i, j] == 1):
            idx_row = np.cumsum(code_PCM[i, 0:j + 1])[-1] - 1
            idx_col = np.cumsum(code_PCM[0: i + 1, j])[-1] - 1
            odd_layer_node_count_1 = 0
            odd_layer_node_count_2 = np.cumsum(sum_edge_c[0:i + 1])[-1]
            if i > 0:
                odd_layer_node_count_1 = np.cumsum(sum_edge_c[0:i])[-1]
            W_even2odd[k, odd_layer_node_count_1:odd_layer_node_count_2] = 1.0
            W_even2odd[k, odd_layer_node_count_1 + idx_row] = 0.0
            k += 1  # k is counted in column direction

# init W_output odd to output
k = 0
for j in range(0, code_PCM.shape[1], 1):
    for i in range(0, code_PCM.shape[0], 1):
        if (code_PCM[i, j] == 1):
            idx_row = np.cumsum(code_PCM[i, 0:j + 1])[-1] - 1
            idx_col = np.cumsum(code_PCM[0: i + 1, j])[-1] - 1
            odd_layer_node_count = 0
            if i > 0:
                odd_layer_node_count = np.cumsum(sum_edge_c[0:i])[-1]
            W_output[odd_layer_node_count + idx_row, k] = 1.0
    k += 1

# init W_skipconn2even  channel input
k = 0
for j in range(0, code_PCM.shape[1], 1):
    for i in range(0, code_PCM.shape[0], 1):
        if (code_PCM[i, j] == 1):
            W_skipconn2even[j, k] = 1.0
            k += 1


##############################  bulid four neural networks(Z = 16,3, 10, 6) ############################
net_dict = {}
# init the learnable network parameters
Weights_Var = np.ones(sum_edge, dtype=np.float32)
Biases_Var = -0.5 * np.ones(sum_edge, dtype=np.float32)
for i in range(0, iters_max, 1):
    net_dict["Weights_Var{0}".format(i)] = tf.Variable(Weights_Var.copy(), name="Weights_Var".format(i))
    net_dict["Biases_Var{0}".format(i)] = tf.Variable(Biases_Var.copy(), name="Biases_Var".format(i))
    if single_damping:
        damping_factor = 0.5 * np.ones(1, dtype=np.float32)
        net_dict["damping_factor{0}".format(i)] = tf.Variable(damping_factor.copy(), name="damping_factor".format(i))
    else:
        damping_factor = 0.5 * np.ones(sum_edge, dtype=np.float32)
        net_dict["damping_factor{0}".format(i)] = tf.Variable(damping_factor.copy(), name="damping_factor".format(i))

# the decoding neural network of Z=16
Z = 16
xa = tf.placeholder(tf.float32, shape=[batch_size, N, Z], name='xa')
ya = tf.placeholder(tf.float32, shape=[batch_size, N * Z], name='ya')
xa_input = tf.transpose(xa, [0, 2, 1])
net_dict["LLRa{0}".format(0)] = tf.zeros((batch_size, Z, sum_edge), dtype=tf.float32)
net_dict["infoM_lastlayera{0}".format(0)] = tf.zeros((batch_size, Z, sum_edge), dtype=tf.float32)
for i in range(0, iters_max, 1):
    #variable node update
    x0 = tf.matmul(xa_input, W_skipconn2even)
    x1 = tf.matmul(net_dict["LLRa{0}".format(i)], W_odd2even)
    x2_3 = tf.add(x0, x1)
    x2 = tf.add(tf.multiply(x2_3, 1 - net_dict["damping_factor{0}".format(i)]),
                tf.multiply(net_dict["infoM_lastlayera{0}".format(i)], net_dict["damping_factor{0}".format(i)]))
    net_dict["infoM_lastlayera{0}".format(i + 1)] = x2
    x2 = tf.transpose(x2, [0, 2, 1])
    x2 = tf.reshape(x2, [batch_size, neurons_per_odd_layer * Z])
    x2 = tf.matmul(x2, Lift_Matrix1[0].transpose())
    x2 = tf.reshape(x2, [batch_size, neurons_per_odd_layer, Z])
    x2 = tf.transpose(x2, [0, 2, 1])
    x_tile = tf.tile(x2, multiples=[1, 1, neurons_per_odd_layer])
    W_input_reshape = tf.reshape(W_even2odd.transpose(), [-1])
    #check node update
    x_tile_mul = tf.multiply(x_tile, W_input_reshape)
    x2_1 = tf.reshape(x_tile_mul, [batch_size, Z, neurons_per_odd_layer, neurons_per_odd_layer])
    x2_abs = tf.add(tf.abs(x2_1), 10000 * (1 - tf.to_float(tf.abs(x2_1) > 0)))
    x3 = tf.reduce_min(x2_abs, axis=3)
    x2_2 = -x2_1
    x4 = tf.add(tf.zeros((batch_size, Z, neurons_per_odd_layer, neurons_per_odd_layer)), 1 - 2 * tf.to_float(x2_2 < 0))
    x4_prod = -tf.reduce_prod(x4, axis=3)
    x_output_0 = tf.multiply(x3, tf.sign(x4_prod))
    x_output_0 = tf.transpose(x_output_0, [0, 2, 1])
    x_output_0 = tf.reshape(x_output_0, [batch_size, Z * neurons_per_odd_layer])
    x_output_0 = tf.matmul(x_output_0, Lift_Matrix2[0])
    x_output_0 = tf.reshape(x_output_0, [batch_size, neurons_per_odd_layer, Z])
    x_output_0 = tf.transpose(x_output_0, [0, 2, 1])
    # add learnable parameters
    x_output_1 = tf.add(tf.multiply(tf.abs(x_output_0),net_dict["Weights_Var{0}".format(i)]), net_dict["Biases_Var{0}".format(i)])
    x_output_1 = tf.multiply(x_output_1, tf.to_float(x_output_1 > 0))
    net_dict["LLRa{0}".format(i+1)] = tf.multiply(x_output_1, tf.sign(x_output_0)) # update the LLR
    # output
    y_output_2 = tf.matmul(net_dict["LLRa{0}".format(i+1)], W_output)
    y_output_3 = tf.transpose(y_output_2, [0, 2, 1])
    y_output_4 = tf.add(xa, y_output_3)
    net_dict["ya_output{0}".format(i)] = tf.reshape(y_output_4, [batch_size, N * Z], name='ya_output'.format(i))
    # calculate loss
    net_dict["lossa{0}".format(i)] = 1.0 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=ya,
                                                            logits=net_dict["ya_output{0}".format(i)]), name='lossa'.format(i))
    # AdamOptimizer
    net_dict["train_stepa{0}".format(i)] = tf.train.AdamOptimizer(learning_rate=
                                                                learning_rate).minimize(net_dict["lossa{0}".format(i)],
                                var_list = [net_dict["Weights_Var{0}".format(i)], net_dict["Biases_Var{0}".format(i)], net_dict["damping_factor{0}".format(i)]])


##################################  Train  ####################################
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
for iter in range(0, iters_max, 1):
    for i in range(0, num_of_batch, 1):
            Z = 16
            SNR_set = np.array([SNR_sigma[0, iter]])
            training_received_data, training_coded_bits = create_mix_epoch(SNR_set, wordRandom, noiseRandom, numOfWordSim_train,
                                                                           code_n, code_k, Z,
                                                                           Ldpc_GM[0],
                                                                           train_on_zero_word)
            training_labels_for_mse = training_coded_bits
            y_pred, train_loss, _ = sess.run(fetches=[net_dict["ya_output{0}".format(iter)], net_dict["lossa{0}".format(iter)],
                                                      net_dict["train_stepa{0}".format(iter)]],
                                             feed_dict={xa: training_received_data, ya: training_labels_for_mse})
            if i % 200 == 0:
                print('iteration: [{0}/{1}]\t'
                      'epoch: [{2}/{3}]\t'
                      'loss: {4}\t'.format(
                    iter + 1, iters_max, i, num_of_batch, train_loss))




    ##################################  save weights and biases  ####################################
    a, b = sess.run(fetches=[net_dict["Weights_Var{0}".format(iter)], net_dict["Biases_Var{0}".format(iter)]])
    np.savetxt('./Weights_Var/Weights_Var{0}.txt'.format(iter), a, fmt='%s', delimiter=',')
    np.savetxt('./Biases_Var/Biases_Var{0}.txt'.format(iter), b, fmt='%s', delimiter=',')
    c = sess.run(fetches=[net_dict["damping_factor{0}".format(iter)]])
    np.savetxt('./damping_factor/damping_factor{0}.txt'.format(iter), c, fmt='%s', delimiter='\n')

