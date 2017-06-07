import csv
import numpy as np
import tensorflow as tf

dataList = list()
train_dataset = 'dataset_10518samples.csv'
with open(input_dataset, 'rb') as csvfile:
    creader = csv.reader(csvfile)
    for row in creader:
        dataList.append(row)
trainArray = np.asarray(dataList, dtype=float)
dDev = np.std(trainArray, 1)
dMean = np.mean(trainArray, 1)
for i in range(trainArray.shape[1]):
    trainArray[:,i] = (trainArray[:,i] - dMean)/dDev

    
dataList = list()
valid_dataset = 'dataset_10518samples.csv'
with open(valid_dataset, 'rb') as csvfile:
    creader = csv.reader(csvfile)
    for row in creader:
        dataList.append(row)
validArray = np.asarray(dataList, dtype=float)
dDev = np.std(validArray, 1)
dMean = np.mean(validArray, 1)
for i in range(validArray.shape[1]):
    validArray[:,i] = (validArray[:,i] - dMean)/dDev

valid_set = validArray
train_set = trainArray 

batch_size=1
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
        batch = np.ndarray([self._batch_size, input_size], dtype=float)
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
hidden1_size = 16
hidden2_size = 32
hidden3_size = 64
graph=tf.Graph()
with graph.as_default():

    w0 = tf.Variable(tf.truncated_normal([train_input_size, hidden1_size]))
    b0 = tf.Variable(tf.zeros([hidden1_size]))

    w1 = tf.Variable(tf.truncated_normal([hidden1_size, hidden2_size]))
    b1 = tf.Variable(tf.zeros([hidden2_size]))

    w2 = tf.Variable(tf.truncated_normal([hidden2, hidden3_size]))
    b2 = tf.Variable(tf.zeros([hidden3_size]))

    w3 = tf.Variable(tf.truncated_normal([hidden3_size, output_size]))
    b3 = tf.Variable(tf.zeros([output_size]))

    def feed_model(x):
        hidden1_layer = tf.nn.relu(tf.matmul(x, w0) + b0)
        hidden2_layer = tf.nn.relu(tf.matmul(hidden1_layer, w1) + b1)
        hidden3_layer = tf.nn.relu(tf.matmul(hidden2_layer, w2) + b2)
        predictions = tf.matmul(hidden3_layer,w3) + b3
        return predictions
    
    train_inputs = list()
    train_targets = list()
    outputs=list()
    for _ in range(num_unrollings):
        train_inputs.append(tf.placeholder(tf.float32, shape=(batch_size,
                        train_input_size)))
        train_targets.append(tf.placeholder(tf.float32,
                    shape=(batch_size,output_size)))
                1.0), trainable=False) 
    for p in train_inputs:
        predictions = feed_model(train_inputs)
        loss = tf.losses.mean_squared_error(tf.concat(train_targets,0),
                predictions)+(tf.nn.l2_loss(w0)+ tf.nn.l2_loss(w1)+
                    tf.nn.l2_loss(w2)) * 0.0001
    
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        0.009, global_step, 1000, 0.8)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
    optimizer = optimizer.apply_gradients(
        zip(gradients, v), global_step=global_step)


    valid_input = tf.placeholder(tf.float32, shape=(1,input_size))
    init_valid_output = tf.Variable(tf.zeros([1, lstm_size]))
    init_valid_state = init_state
    reset_state = tf.group(
        init_valid_output.assign(tf.zeros([1, lstm_size])),
        init_valid_state.assign(init_state))
    valid_output, valid_state = lstm_cell(
        valid_input, init_valid_output, init_valid_state)
    with tf.control_dependencies([init_valid_output.assign(valid_output),
                                init_valid_state.assign(valid_state)]):
        sample_prediction = feed_model(valid_output)

number_reset = 10159
num_steps = 20300
summary_frequency = 500

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    mean_loss = 0
    for step in range(num_steps):
        if(step == number_reset):
            reset_train.run()
        batches, labels = train_gen.next()
        feed_dict = dict()
        for i in range(num_unrollings):
            feed_dict[train_inputs[i]] = batches[i]
#            print(batches[i])
            feed_dict[train_targets[i]] = labels[i]
        _, l, p = session.run(
            [optimizer, loss, predictions], feed_dict=feed_dict)
        #_, l, p, lr = session.run(
        #    [optimizer, loss, predictions, learning_rate], feed_dict=feed_dict)
        mean_loss += l
        if step % summary_frequency == 0:
            if step > 0:
                mean_loss = mean_loss / summary_frequency
      # The mean loss is an estimate of the loss over the last few batches.
            print(
                'Average loss at step %d: %f learning rate: %f' % (step,
                    mean_loss, 0.01))
            mean_loss = 0
    #        print("predictions:", p)
    #        print("targets:", labels)
    reset_state.run() 
    valid_loss = 0;
    test_size = valid_size
    for i in range(test_size):
        vb,vl = valid_gen.next()
        predict = sample_prediction.eval({valid_input: vb[0]})
        valid_loss = valid_loss + ((predict-vl[0])**2).mean()
        if i> (test_size - 10):
            print(i,'='*80)
            print("frequencies:", vb[0])
            print("positions:", vl[0])
            print("predictions:",predict)
    print('Validation mean loss: %.2f' % float(valid_loss / test_size))

