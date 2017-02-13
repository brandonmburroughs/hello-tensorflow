import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# Load data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Layers
K = 200
L = 100
M = 60
N = 30

# Initialize variables
learning_rate = tf.placeholder(tf.float32)
pkeep = tf.placeholder(tf.float32)

X = tf.placeholder(tf.float32, [None, 784])
W1 = tf.Variable(tf.truncated_normal([784, K], stddev=0.1))
B1 = tf.Variable(tf.zeros([K]))
W2 = tf.Variable(tf.truncated_normal([K, L], stddev=0.1))
B2 = tf.Variable(tf.zeros([L]))
W3 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
B3 = tf.Variable(tf.zeros([M]))
W4 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
B4 = tf.Variable(tf.zeros([N]))
W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
B5 = tf.Variable(tf.zeros([10]))

# Neural net layers
Y1 = tf.nn.dropout(tf.nn.relu(tf.matmul(X, W1) + B1), pkeep)
Y2 = tf.nn.dropout(tf.nn.relu(tf.matmul(Y1, W2) + B2), pkeep)
Y3 = tf.nn.dropout(tf.nn.relu(tf.matmul(Y2, W3) + B3), pkeep)
Y4 = tf.nn.dropout(tf.nn.relu(tf.matmul(Y3, W4) + B4), pkeep)
Y = tf.nn.softmax(tf.matmul(Y4, W5) + B5)
Y_ = tf.placeholder(tf.float32, [None, 10])

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
        pkeep: 0.75
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
            pkeep: 1.0
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

print "Max test accuracy achieved = {:.2f}%".format(max(test_accuracies) * 100) # 98.08%

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
plt.savefig("output_images/04_deep_network_dropout.png")
