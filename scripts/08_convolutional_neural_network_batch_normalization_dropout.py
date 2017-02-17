import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# Batch normalization
def batchnorm_layer(Ylogits, is_test, offset, iteration, convolutional=False):
    """Batch normalization layer for a neural network.  A batch normalization layer normalizes the
    current batch of inputs by subtracting the mean and dividing by the variance.  For test sets,
    it keeps an exponential moving average of the mean and variance.

    Parameters
    ----------
    Ylogits : tensor
        The output of the previous layer
    is_test : bool
        `True` if this is test data, `False` for training
    offset : tf.Variable
        The bias term
    iteration : int
        The current iteration of the nueral network
    convolutional : bool
        `True` if the previous layer (Ylogits) was a convolutional layer

    Returns
    -------
    The batch normalized outputs and the updated moving averages of mean and variance
    """
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.9999, iteration)
    if convolutional: # average across batch, width, height
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, variance_epsilon=1e-5)

    return Ybn, update_moving_averages


# Load data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Layers
K = 6
L = 12
M = 24
N = 200

# Initialize variables
learning_rate = tf.placeholder(tf.float32)
pkeep = tf.placeholder(tf.float32)

# Batch normalization
is_test = tf.placeholder(tf.bool)
iteration = tf.placeholder(tf.int32)

# Convolutional layers
X = tf.placeholder(tf.float32, [None, 784])
W1 = tf.Variable(tf.truncated_normal([5, 5, 1, K], stddev=0.1))
B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]))
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]))

# Fully connected layers
W4 = tf.Variable(tf.truncated_normal([7 * 7 * M, N], stddev=0.1))
B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]))
W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
B5 = tf.Variable(tf.constant(0.1, tf.float32, [10]))

# Neural net layers
X_reshaped = tf.reshape(X, shape=[-1, 28, 28, 1])
Y1_logits = tf.nn.conv2d(X_reshaped, W1, strides=[1, 1, 1, 1], padding="SAME")
Y1bn, exp_moving_averages_1 = batchnorm_layer(Y1_logits, is_test, B1, iteration, convolutional=True)
Y1 = tf.nn.dropout(tf.nn.relu(Y1bn), pkeep)
Y2_logits = tf.nn.conv2d(Y1, W2, strides=[1, 2, 2, 1], padding="SAME")
Y2bn, exp_moving_averages_2 = batchnorm_layer(Y2_logits, is_test, B2, iteration, convolutional=True)
Y2 = tf.nn.dropout(tf.nn.relu(Y2bn), pkeep)
Y3_logits = tf.nn.conv2d(Y2, W3, strides=[1, 2, 2, 1], padding="SAME")
Y3bn, exp_moving_averages_3 = batchnorm_layer(Y3_logits, is_test, B3, iteration, convolutional=True)
Y3 = tf.nn.dropout(tf.nn.relu(Y3bn), pkeep)
Y4_logits = tf.matmul(tf.reshape(Y3, shape=[-1, 7 * 7 * M]), W4)
Y4bn, exp_moving_averages_4 = batchnorm_layer(Y4_logits, is_test, B4, iteration)
Y4 = tf.nn.dropout(tf.nn.relu(Y4bn), pkeep)
Y_logits = tf.matmul(Y4, W5) + B5
Y = tf.nn.softmax(Y_logits)
Y_ = tf.placeholder(tf.float32, [None, 10])

exp_moving_averages = tf.group(exp_moving_averages_1, exp_moving_averages_2,
                               exp_moving_averages_3, exp_moving_averages_4)

# Performance metrics
cross_entropy = -tf.reduce_sum(Y_ * tf.log(tf.clip_by_value(Y, 1e-10, 1.0)))
is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Optimizer setup
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_step = optimizer.minimize(cross_entropy)

# Start tf session
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Train the model
training_accuracies = []
training_cross_entropies = []
test_accuracies = []
test_cross_entropies = []
learning_rate_start = 0.003
decay_step_rate = 250
steps = 10001

for i in range(steps):
    # Load batch of images and correct answers
    batch_X, batch_Y = mnist.train.next_batch(100)

    # Decay the learning rate as more steps are completed to stabilize learning
    train_data = {
        X: batch_X,
        Y_: batch_Y,
        learning_rate: learning_rate_start * 0.95 ** (i / decay_step_rate),
        pkeep: 0.75,
        is_test: False,
        iteration: i
    }

    # Train
    sess.run(train_step, feed_dict=train_data)

    if i % 10 == 0:
        # Check accuracy
        training_accuracy, training_cross_ent = sess.run([accuracy, cross_entropy],
                                                         feed_dict=train_data)
        training_accuracies.append(training_accuracy)
        training_cross_entropies.append(training_cross_ent)

        # Check accuracy on test data
        test_data={
            X: mnist.test.images,
            Y_: mnist.test.labels,
            learning_rate: learning_rate_start * 0.95 ** (i / decay_step_rate),
            pkeep: 1.00,
            is_test: True,
            iteration: i
        }
        test_accuracy, test_cross_ent = sess.run([accuracy, cross_entropy], feed_dict=test_data)
        test_accuracies.append(test_accuracy)
        test_cross_entropies.append(test_cross_ent / 100)

        if i % 1000 == 0:
            print "#### At step {} ####".format(i)
            print "Training accuracy = {}\nTraining cross entropy = {}\n".format(
                training_accuracy, training_cross_ent
            )
            print "Testing accuracy = {}\nTesting cross entropy = {}\n".format(
                test_accuracy, test_cross_ent / 100
            )

    sess.run(exp_moving_averages, feed_dict=train_data)

print "Max test accuracy achieved = {:.2f}%".format(max(test_accuracies) * 100) # 99.14%

# Plot the performance
plt.figure(figsize=(15, 8))
plt.subplot(121)
plt.plot(range(0, 10001, 10), training_accuracies)
plt.plot(range(0, 10001, 10), test_accuracies)
plt.ylim(0.94, 1.00)
plt.title("Accuracy")
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.legend(["Training Accuracy", "Test Accuracy"], loc="lower right")
plt.subplot(122)
plt.plot(range(0, 10001, 10), training_cross_entropies)
plt.plot(range(0, 10001, 10), test_cross_entropies)
plt.ylim(0, 20)
plt.title("Cross Entropy Loss")
plt.xlabel("Iterations")
plt.ylabel("Cross Entropy")
plt.legend(["Training Loss", "Test Loss"], loc="upper right")
plt.savefig("output_images/08_convolutional_neural_network_batch_normalization_dropout.png")
