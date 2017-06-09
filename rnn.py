import csv
import numpy as np
import tensorflow as tf

dataList = list()
train_dataset = 'dataset_63642samples_with_noise.csv'
with open(train_dataset, 'rb') as csvfile:
    creader = csv.reader(csvfile)
    for row in creader:
        dataList.append(row)
trainArray = np.asarray(dataList, dtype=float)
dDev = np.std(trainArray, 1)
dMean = np.mean(trainArray, 1)
for i in range(trainArray.shape[1]):
    trainArray[:,i] = (trainArray[:,i] - dMean)/dDev
noisetarget = np.append(np.asarray([[0],[0]], dtype=float), trainArray[4:, :-1], axis=1)
tInput = np.append(trainArray[:4,:], noisetarget, axis=0)
trainArray = np.append(tInput, trainArray[4:,:], axis=0)
dataList = list()
valid_dataset = 'dataset_63642samples_with_noise.csv'
with open(valid_dataset, 'rb') as csvfile:
    creader = csv.reader(csvfile)
    for row in creader:
        dataList.append(row)
validArray = np.asarray(dataList, dtype=float)
dDev = np.std(validArray, 1)
dMean = np.mean(validArray, 1)
for i in range(validArray.shape[1]):
    validArray[:,i] = (validArray[:,i] - dMean)/dDev

valid_size = 1000
test_size = 4000
valid_set = validArray[:,:valid_size]
test_set = validArray[:,valid_size : 5 * valid_size]
train_set = trainArray 
train_size = train_set.shape[1]    

batch_size=30
num_unrollings=1
input_size = 4
output_size = 2
train_input_size=input_size+output_size
    
class GenerateBatchData(object):
    def __init__(self, data_set, batch_size, num_step, input_size):
        self._data_set = data_set
        self._batch_size = batch_size
        self._data_size = len(self._data_set[1,:])
        self._sess_size = self._data_size //self._batch_size
        self._cursor = 0 
        self._input_size = input_size
        self._num_step = num_step
    def _next_batch(self):
        batch = np.ndarray([self._batch_size, self._input_size], dtype=float)
        target = np.ndarray([self._batch_size, output_size], dtype=float)
        for i in range(self._batch_size):
            batch[i, :] = self._data_set[:self._input_size, self._cursor + i * self._sess_size]
            target[i, :] = self._data_set[self._input_size:, self._cursor + i * self._sess_size]
        self._cursor = (self._cursor+1)%self._sess_size
        return batch, target
    def next(self):
        batches = list()
        targets = list()
        for i in range(self._num_step):
            batch, target = self._next_batch()
            batches.append(batch)
            targets.append(target)
        return batches, targets
train_gen = GenerateBatchData(train_set, batch_size, num_unrollings,
        train_input_size)
valid_gen = GenerateBatchData(valid_set, 1, 1, input_size)
test_gen = GenerateBatchData(test_set, 1, 1, input_size)
hidden1_size = 64
hidden2_size = 64
hidden3_size = 64
hidden4_size = 64
dropoutRate = 0.3
graph=tf.Graph()
with graph.as_default():

    w0 = tf.Variable(tf.truncated_normal([train_input_size, hidden1_size]))
    b0 = tf.Variable(tf.zeros([hidden1_size]))

    w1 = tf.Variable(tf.truncated_normal([hidden1_size, hidden2_size]))
    b1 = tf.Variable(tf.zeros([hidden2_size]))

    w2 = tf.Variable(tf.truncated_normal([hidden2_size, hidden3_size]))
    b2 = tf.Variable(tf.zeros([hidden3_size]))

    w3 = tf.Variable(tf.truncated_normal([hidden3_size, hidden4_size]))
    b3 = tf.Variable(tf.zeros([hidden4_size]))
    w4 = tf.Variable(tf.truncated_normal([hidden4_size, output_size]))
    b4 = tf.Variable(tf.zeros([output_size]))

    def train_feed_model(x):
        hidden1_layer = tf.nn.relu(tf.matmul(x, w0) + b0)
        hidden2_layer = tf.nn.dropout(tf.nn.relu(tf.matmul(hidden1_layer, w1) +
                    b1), dropoutRate)
        hidden3_layer = tf.nn.dropout(tf.nn.relu(tf.matmul(hidden2_layer, w2) +
                    b2), dropoutRate)
        hidden4_layer = tf.nn.dropout(tf.nn.relu(tf.matmul(hidden3_layer, w3) +
                    b3), dropoutRate)
        predictions = tf.matmul(hidden4_layer,w4) + b4
        return predictions
    def feed_model(x):
        hidden1_layer = tf.nn.relu(tf.matmul(x, w0) + b0)
        hidden2_layer = tf.nn.relu(tf.matmul(hidden1_layer, w1) + b1)
        hidden3_layer = tf.nn.relu(tf.matmul(hidden2_layer, w2) + b2)
        hidden4_layer = tf.nn.relu(tf.matmul(hidden3_layer, w3) + b3)
        predictions = tf.matmul(hidden4_layer,w4) + b4
        return predictions
    
    train_inputs = list()
    train_targets = list()
    outputs=list()
    for _ in range(num_unrollings):
        train_inputs.append(tf.placeholder(tf.float32, shape=(batch_size,
                        train_input_size)))
        train_targets.append(tf.placeholder(tf.float32,
                    shape=(batch_size,output_size)))
    for p in train_inputs:
        predictions = train_feed_model(p)
        loss = tf.losses.mean_squared_error(tf.concat(train_targets,0),
                predictions)
    
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        2.0, global_step, 500, 0.6)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
    optimizer = optimizer.apply_gradients(
        zip(gradients, v), global_step=global_step)


    valid_input = tf.placeholder(tf.float32, shape=(1,input_size))
    valid_target = target = tf.Variable(tf.zeros([1, output_size]))
    reset_state = tf.group(target.assign(tf.Variable(tf.zeros([1,
                        output_size]))),
        valid_target.assign(tf.Variable(tf.zeros([1, output_size]))))
    with tf.control_dependencies([valid_target.assign(target)]):
        target = feed_model(tf.concat([valid_input, valid_target],1))

num_steps = train_size//batch_size
summary_frequency = 100

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    mean_loss = 0
    for step in range(num_steps):
        batches, labels = train_gen.next()
        feed_dict = dict()
        for i in range(num_unrollings):
            feed_dict[train_inputs[i]] = batches[i]
#            print(batches[i])
            feed_dict[train_targets[i]] = labels[i]
        _, l, p, lr = session.run(
            [optimizer, loss, predictions, learning_rate], feed_dict=feed_dict)
        mean_loss += l
        if step % summary_frequency == 0:
            if step > 0:
                mean_loss = mean_loss / summary_frequency
      # The mean loss is an estimate of the loss over the last few batches.
            print(
                'Average loss at step %d: %f learning rate: %f' % (step,
                    mean_loss, lr))
            mean_loss = 0
            reset_state.run() 
            valid_loss = 0;
            test_size = valid_size 
            for _ in range(test_size):
                vb,vl = valid_gen.next()
                predict = target.eval({valid_input: vb[0]})
                valid_loss = valid_loss + ((predict - vl[0])**2).mean()
            print('Validation mean loss: %.2f' % float(valid_loss / test_size))
    reset_state.run() 
    valid_loss = 0;
    test_size = valid_size * 4 
    display = test_size * 0.8
    for i in range(test_size):
        vb,vl = test_gen.next()
        predict = target.eval({valid_input: vb[0]})
        valid_loss = valid_loss + ((predict - vl[0])**2).mean()
        if i > display and i < display + 10:
            #print("positions:", vl[0]*dDev[4:]+dMean[4:])
            #print("predictions:", predict*dDev[4:]+dMean[4:])
            print("positions:", vl[0])
            print("predictions:", predict)
    print('Testing mean loss: %.2f' % float(valid_loss / test_size))
