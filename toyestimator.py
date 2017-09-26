import tensorflow as tf
import numpy as np   # to load data

# feature definitions
feature_cols = [tf.feature_column.numeric_column("x", shape=[1])]

# use an appropriate estimator
estimator = tf.estimator.DNNRegressor(feature_columns=feature_cols, hidden_units=[1024, 512, 256]);


# TensorFlow provides many helper methods to read and set up data sets.
# Here we use two data sets: one for training and one for evaluation
# We have to tell the function how many batches
# of data (num_epochs) we want and how big each batch should be.
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])

input_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_inp_fn = tf.estimator.inputs.numpy_input_fn({"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_inp_fn = tf.estimator.inputs.numpy_input_fn({"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

# train for 1000 steps
estimator.train(input_fn=input_fn, steps=1000)

# evaluation
train_metrics = estimator.evaluate(input_fn=train_inp_fn)
eval_metrics = estimator.evaluate(input_fn=eval_inp_fn)

print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)