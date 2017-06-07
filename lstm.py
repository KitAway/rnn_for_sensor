import csv
import numpy as np
import tensorflow as tf

dataList = list()
with open('dataset_63642samples_with_noise.csv', 'rb') as csvfile:
    creader = csv.reader(csvfile)
    for row in creader:
        dataList.append(row)
dArray = np.asarray(dataList, dtype=float)
dDev = np.std(dArray, 1)
dMean = np.mean(dArray, 1)
for i in range(dArray.shape[1]):
    dArray[:,i] = (dArray[:,i] - dMean)/dDev

valid_size = 1000
valid_set = dArray[:,:valid_size]
train_set = dArray[:,valid_size:] 

lstm_size = 64
batch_size=50
num_unrollings=1
input_size = 4
output_size = 2
    
class GenerateBatchData(object):
    def __init__(self, data_set, batch_size, num_step):
        self._data_set = data_set
        self._batch_size = batch_size
        self._data_size = len(self._data_set[1,:])
        self._sess_size = self._data_size //self._batch_size
        self._cursor = 0 
        self._num_step = num_step
    def _next_batch(self):
        batch = np.ndarray([self._batch_size, input_size], dtype=float)
        target = np.ndarray([self._batch_size, output_size], dtype=float)
        for i in range(self._batch_size):
            batch[i, :] = self._data_set[:input_size, self._cursor + i * self._sess_size]
            target[i, :] = self._data_set[input_size:, self._cursor + i * self._sess_size]
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
train_gen = GenerateBatchData(train_set, batch_size, num_unrollings)
valid_gen = GenerateBatchData(valid_set, 1, 1)
batch, label = train_gen.next()
#print("batch:", batch)
#print("label:", label)            
extent_size = 32
hidden_size = 32
graph=tf.Graph()
with graph.as_default():

    w0 = tf.Variable(tf.truncated_normal([input_size, extent_size]))
    b0 = tf.Variable(tf.zeros([extent_size]))

    lx = tf.Variable(tf.truncated_normal([extent_size, 4 * lstm_size], -0.1, 0.1))
    lm = tf.Variable(tf.truncated_normal([lstm_size, 4 * lstm_size], -0.1, 0.1))
    lb = tf.Variable(tf.zeros([1, 4 * lstm_size]))
    
    def train_lstm_cell(i,o,state):
        """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
        Note that in this formulation, we omit the various connections between the
        previous state and the gates."""
        ie = tf.nn.relu(tf.nn.dropout(tf.matmul(i, w0)+b0,0.5))
        _matmul=tf.nn.dropout(tf.matmul(ie, lx),0.5) + tf.matmul(o, lm) + lb
        input_gate, forget_gate, update, output_gate = tf.split(_matmul, 4, 1)
        state = tf.sigmoid(forget_gate) * state + tf.sigmoid(input_gate) * tf.tanh(update)
        return tf.sigmoid(output_gate) * tf.tanh(state), state
    def lstm_cell(i,o,state):
        """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
        Note that in this formulation, we omit the various connections between the
        previous state and the gates."""
        ie = tf.nn.relu(tf.matmul(i, w0)+b0)
        _matmul=tf.matmul(ie, lx) + tf.matmul(o, lm) + lb
        input_gate, forget_gate, update, output_gate = tf.split(_matmul, 4, 1)
        state = tf.sigmoid(forget_gate) * state + tf.sigmoid(input_gate) * tf.tanh(update)
        return tf.sigmoid(output_gate) * tf.tanh(state), state
    
    w1 = tf.Variable(tf.truncated_normal([lstm_size, hidden_size]), 
            dtype=tf.float32)
    b1 = tf.Variable(tf.zeros([hidden_size]), dtype=tf.float32)
    w2 = tf.Variable(tf.truncated_normal([hidden_size, output_size]), 
            dtype=tf.float32)
    b2 = tf.Variable(tf.zeros([output_size]), dtype=tf.float32)

    def train_feed_model(output):
        hidden_layer = tf.nn.relu(tf.nn.dropout(tf.matmul(output, w1) +
                    b1,0.5))
        predictions = tf.matmul(hidden_layer,w2) + b2
        return predictions
    def feed_model(output):
        hidden_layer = tf.nn.relu(tf.matmul(output, w1) + b1)
        predictions = tf.matmul(hidden_layer,w2) + b2
        return predictions
    
    train_inputs = list()
    train_targets = list()
    outputs=list()
    for _ in range(num_unrollings):
        train_inputs.append(tf.placeholder(tf.float32, shape=(batch_size,
                        input_size)))
        train_targets.append(tf.placeholder(tf.float32,
                    shape=(batch_size,output_size)))
    init_state = tf.Variable(tf.random_uniform([batch_size, lstm_size], -1.0,
                1.0), trainable=False) 
    init_output = tf.Variable(tf.zeros([batch_size, lstm_size]), trainable=False) 
    output = init_output
    state = init_state
    reset_train = tf.group(state.assign(init_state))
    for p in train_inputs:
        output, state = train_lstm_cell(p,output, state)
        outputs.append(output)
    with tf.control_dependencies([init_state.assign(state),
                init_output.assign(output)]):
        outCat = tf.concat(outputs,0)
        predictions = feed_model(outCat)
        loss = tf.losses.mean_squared_error(tf.concat(train_targets,0),
                predictions)
    
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        0.5, global_step, 300, 0.9)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
    optimizer = optimizer.apply_gradients(
        zip(gradients, v), global_step=global_step)


    init_state_single = tf.Variable(tf.random_uniform([1, lstm_size], -1.0,
                1.0), trainable=False) 
    valid_input = tf.placeholder(tf.float32, shape=(1,input_size))
    init_valid_output = tf.Variable(tf.zeros([1, lstm_size]))
    init_valid_state = init_state_single
    reset_state = tf.group(
        init_valid_output.assign(tf.zeros([1, lstm_size])),
        init_valid_state.assign(init_state_single))
    valid_output, valid_state = lstm_cell(
        valid_input, init_valid_output, init_valid_state)
    with tf.control_dependencies([init_valid_output.assign(valid_output),
                                init_valid_state.assign(valid_state)]):
        sample_prediction = feed_model(valid_output)

num_steps = 1201
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
