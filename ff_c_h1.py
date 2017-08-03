#!/bin/python
import csv
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

batch_size = 35
capaSen = 4
output_size = 2
trainCSV = "trainset_tnoise_105101.csv"
testCSV = "testset_tnoise_393.csv"


def main(capa_only):
    if capa_only:
        input_size = capaSen
    else:
        input_size = 20
    dataList = list()
    with open(trainCSV, 'rb') as csvfile:
        creader = csv.reader(csvfile)
        for row in creader:
            dataList.append(row)
    dArray = np.asarray(dataList, dtype=float)
    if capa_only:
        dArray = np.delete(dArray, np.s_[capaSen:-output_size], axis = 0)
    dDev = np.std(dArray, 1)
    dMean = np.mean(dArray, 1)

    print("Mean value is %.10f,%.10f,%.10f,%.10f,%.10f,%.10f"%tuple(dMean.tolist()))
    print("Stdev value is %.10f,%.10f,%.10f,%.10f,%.10f,%.10f"%tuple(dDev.tolist()))
    for i in range(dArray.shape[1]):
        dArray[:,i] = (dArray[:,i] - dMean)/dDev

    testList=list()
    with open(testCSV, 'rb') as csvfile:
        creader = csv.reader(csvfile)
        for row in creader:
            testList.append(row)

    test_data = np.asarray(testList, dtype=float)
    if capa_only:
        test_data = np.delete(test_data, np.s_[capaSen:-output_size], axis = 0)
    target_list=test_data[input_size:,:].copy()
    output_list=np.zeros(target_list.shape)
    for i in range(test_data.shape[1]):
        test_data[:,i] = (test_data[:,i] - dMean)/dDev
#        print(test_data[:,i])
#    return
    ################################
    # take validation array from dArray
    valid_size = 400
    valid_set = dArray[:,5000:(5000+valid_size)]
    ################################
    test_set = test_data
    # consider all test size
    test_size=test_set.shape[1]
    #################################
    train_set = dArray
    train_size = train_set.shape[1]
    #################################


        
    class GenerateBatchData(object):
        def __init__(self, data_set, batch_size):
            self._data_set = data_set
            self._batch_size = batch_size
            self._data_size = len(self._data_set[1,:])
            self._sess_size = self._data_size //self._batch_size
            self._cursor = 0 
        def _next_batch(self):
            batch = np.ndarray([self._batch_size, input_size], dtype=float)
            target = np.ndarray([self._batch_size, output_size], dtype=float)
            for i in range(self._batch_size):
                batch[i, :] = self._data_set[:input_size, self._cursor + i * self._sess_size]
                target[i, :] = self._data_set[input_size:, self._cursor + i * self._sess_size]
            self._cursor = (self._cursor+1)%self._sess_size
            return batch, target
        def next(self):
            batch, target = self._next_batch()
            return batch, target
        def reset(self):
            self._cursor = 0

    train_gen = GenerateBatchData(train_set, batch_size)
    valid_gen = GenerateBatchData(valid_set, 1)
    test_gen = GenerateBatchData(test_set, 1)
    batch, label = train_gen.next()

    hidden_size = 120
    keepRate = 1.0

    graph=tf.Graph()
    with graph.as_default():

        w1 = tf.Variable(tf.truncated_normal([input_size, hidden_size]))
        b1 = tf.Variable(tf.zeros([1, hidden_size]))
        w2 = tf.Variable(tf.truncated_normal([hidden_size, output_size], stddev =
                    0.1))
        b2 = tf.Variable(tf.zeros([1, output_size]))

        def train_cell(i):
            hidden_layer = tf.nn.dropout(tf.nn.relu(tf.matmul(i, w1)+b1),keepRate)
            output_layer = tf.matmul(hidden_layer, w2)+b2
            return output_layer
        def ff_cell(i):
            hidden_layer = tf.nn.relu(tf.matmul(i, w1)+b1)
            output_layer = tf.matmul(hidden_layer, w2)+b2
            return output_layer
            

        train_inputs = tf.placeholder(tf.float32, shape=(batch_size,input_size))
        train_targets =tf.placeholder(tf.float32, shape=(batch_size,output_size))
        predictions = train_cell(train_inputs)

            # absolute error
        loss = tf.losses.mean_squared_error(train_targets, predictions)
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(
            0.5, global_step,3500, 0.5)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        gradients, v = zip(*optimizer.compute_gradients(loss))
        gradients, _ = tf.clip_by_global_norm(gradients, 1.75)
        optimizer = optimizer.apply_gradients(
            zip(gradients, v), global_step=global_step)

        valid_input = tf.placeholder(tf.float32, shape=(1,input_size))
        sample_prediction= ff_cell(valid_input)

    num_repeat = 20
    num_steps = train_size//batch_size
    summary_frequency = 2000

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print 'Initialized'
        for _ in range(num_repeat):
            mean_loss = 0
            print '*'*30, 'repeating...','*'*30
            for step in range(num_steps):
                batches, labels = train_gen.next()
                feed_dict=dict()
                feed_dict[train_inputs] = batches
                feed_dict[train_targets] = labels
                _, l, p, lr = session.run(
                    [optimizer, loss, predictions, learning_rate], feed_dict=feed_dict)
                mean_loss += l
                if step % summary_frequency == 0:
                    if step > 0:
                        mean_loss = mean_loss / summary_frequency
              # The mean loss is an estimate of the loss over the last few batches.
                    print 'Average loss at step %d: %f learning rate: %f' % (step,
                            mean_loss, lr)
                    mean_loss = 0
            #        print("predictions:", p)
            #        print("targets:", labels)
                    valid_loss = 0;
                    #test_size = valid_size 
                    for _ in range(valid_size):
                        vb,vl = valid_gen.next()
                        predict = sample_prediction.eval({valid_input: vb})
                        predictPos = predict * dDev[input_size:]
                        targetPos = vl * dDev[input_size:]
                        valid_loss = valid_loss + ((predictPos - targetPos)**2).mean()
                    print 'Validation mean loss: %.4f' % float(valid_loss / test_size)

        test_loss = 0;
        display = test_size * 0.8

        result_list = list()
        for i in range(test_size):
            vb,vl = test_gen.next()
            predict = sample_prediction.eval({valid_input: vb})
            predictPos = predict * dDev[input_size:] + dMean[input_size:]
            targetPos = vl * dDev[input_size:] + dMean[input_size:]
            test_loss = test_loss + ((predictPos - targetPos)**2).mean()
            output_list[:,i] =predictPos
            result_list.append(predictPos)
            if i > display and i < display + 10:
    # put predictions in an array
                print '='*80
                print "positions:", targetPos 
                print "predictions:", predictPos 
        print 'Testing mean loss: %.4f' % float(test_loss / test_size) 
    #    fig=plt.figure()
    #    plt.plot(output_list[0,:], output_list[1,:],'*-')
    #    plt.plot(target_list[0,:], target_list[1,:],'k-')
    #    fig.savefig('lstm_size%d-infrasize%d-repeating%d-dropRate%d'%(lstm_size,infrared_size,num_repeat,
    #                100-100 * keepRate))
        #plt.show()
        with open("result.csv",'wb') as resultFile:
            wr = csv.writer(resultFile, delimiter = ',')
            for l in result_list:
               wr.writerow(list(l))
        with open("weight_0.csv",'wb') as resultFile:
            wr = csv.writer(resultFile, delimiter = ',')
            lt = w1.eval().tolist()
            for l in lt:
               wr.writerow(list(l))
        with open("bias_0.csv",'wb') as resultFile:
            wr = csv.writer(resultFile, delimiter = ',')
            lt = b1.eval().tolist()
            for l in lt:
               wr.writerow(list(l))
        with open("weight_1.csv",'wb') as resultFile:
            wr = csv.writer(resultFile, delimiter = ',')
            lt = w2.eval().tolist()
            for l in lt:
               wr.writerow(list(l))
        with open("bias_1.csv",'wb') as resultFile:
            wr = csv.writer(resultFile, delimiter = ',')
            lt = b2.eval().tolist()
            for l in lt:
               wr.writerow(list(l))
if __name__=='__main__':
    main(True)
