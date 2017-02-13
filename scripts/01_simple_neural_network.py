import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# Load data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Initialize variables
X = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Single neural net layer
Y = tf.nn.softmax(tf.matmul(X, W) + b)
Y_ = tf.placeholder(tf.float32, [None, 10])

# Performance metrics
cross_entropy = -tf.reduce_sum(Y_ * tf.log(tf.clip_by_value(Y, 1e-10, 1.0)))
is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Optimizer setup
optimizer = tf.train.GradientDescentOptimizer(0.003)
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
steps = 10001

for i in range(steps):
    # Load batch of images and correct answers
    batch_X, batch_Y = mnist.train.next_batch(100)
    train_data = {
        X: batch_X,
        Y_: batch_Y
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
            Y_: mnist.test.labels
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

print "Max test accuracy achieved = {:.2f}%".format(max(test_accuracies) * 100) # 92.70%

# Plot the performance
plt.figure(figsize=(15, 8))
plt.subplot(121)
plt.plot(range(0, 10001, 10), training_accuracies)
plt.plot(range(0, 10001, 10), test_accuracies)
plt.ylim(0.85, 1.00)
plt.title("Accuracy")
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.legend(["Training Accuracy", "Test Accuracy"], loc="lower right")
plt.subplot(122)
plt.plot(range(0, 10001, 10), training_cross_entropies)
plt.plot(range(0, 10001, 10), test_cross_entropies)
plt.ylim(0, 60)
plt.title("Cross Entropy Loss")
plt.xlabel("Iterations")
plt.ylabel("Cross Entropy")
plt.legend(["Training Loss", "Test Loss"], loc="upper right")
plt.savefig("output_images/01_simple_neural_network.png")
