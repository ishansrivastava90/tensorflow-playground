import tensorflow as tf

# Model Params
w = tf.Variable([2.3], dtype=tf.float32)
b = tf.Variable([-0.4], dtype=tf.float32)

#print(w)

# input
x = tf.placeholder(tf.float32);

# model definition
linear_model = w * x + b;

# target
y = tf.placeholder(tf.float32);

# loss definition ( SSE )
loss = tf.reduce_sum(tf.square(y - linear_model));

# loss optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01);
train = optimizer.minimize(loss);

# training data
x_train = [1, 2, 3, 4] #[1.3, 2, -0.4, -8.7, 3.21, -8.3, 8.4, 0.9, -82.40, 0.32];
y_train = [0, -1, -2, -3] #[0.32, 3.4, -2.3, 54.3, -24.1, 1.2, 9.2, 0.02, -0.52, -12.5];

# do training
init = tf.global_variables_initializer();
sess = tf.Session();
sess.run(init);

for i in range(100):
	sess.run(train, {x: x_train, y: y_train});
	
# evaluation
curr_w, curr_b, curr_loss = sess.run([w, b, loss], {x: x_train, y: y_train});
print("w: %s b: %s loss: %s"%(curr_w, curr_b, curr_loss));



