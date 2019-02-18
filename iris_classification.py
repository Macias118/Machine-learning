import os
import tensorflow as tf
import csv

train_url = 'https://gist.githubusercontent.com/' \
    'curran/a08a1080b88344b0c8a7/raw/' \
    'd546eaee765268bf2f487608c537c05e22e4b221/iris.csv'

def loss(model, x, y):
  y_ = model(x)
  return tf.losses.sparse_softmax_cross_entropy(
    labels=y,
    logits=y_)

def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

def get_data_from_csv_file(filepath):
    with open(filepath) as file:
        reader = csv.reader(file)
        return [line for line in reader]

def get_trained_model():
    tf.enable_eager_execution()

    # Get training data from url to file
    # train_dataset_fp - path to file
    train_dataset_fp = tf.keras.utils.get_file(
        fname=os.path.basename(train_url),
        origin=train_url)

    # Set 3 classes of iris
    # Get data from file to variable named `train_data`
    class_names = ['setosa', 'versicolor', 'virginica']
    train_data = get_data_from_csv_file(train_dataset_fp)

    # Get 4 column_names from first element
    # ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    column_names = train_data[0]
    train_data.pop(0)

    # Change target values from iris names to numbers [0,1,2]
    for line in train_data:
        line[-1] = class_names.index(line[-1])

    # Seperate training data from target values
    features = []
    target_values = []
    for line in train_data:
        features.append(line[:-1])
        target_values.append(line[-1])

    # Convert features to float
    features = [[float(f) for f in line] for line in features]

    # Create tensors from training data
    features = [tf.constant(i) for i in features]

    # Create tensor object representing list of features
    features = tf.stack(list(features))

    # Build model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # input shape required
        tf.keras.layers.Dense(10, activation=tf.nn.relu),
        tf.keras.layers.Dense(3)
    ])

    # Make prediction and convert logits
    # to a probability for each class
    predictions = model(features)
    predictions = tf.nn.softmax(predictions)

    # Setup the optimizer
    # and the global_step counter
    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate=0.01)
    global_step = tf.Variable(0)

    from tensorflow import contrib
    tfe = contrib.eager

    # Keep results for plotting
    train_loss_results = []
    train_accuracy_results = []

    num_epochs = 500

    for epoch in range(num_epochs):
        epoch_loss_avg = tfe.metrics.Mean()
        epoch_accuracy = tfe.metrics.Accuracy()

        loss_value, grads = grad(model, features, target_values)
        optimizer.apply_gradients(
                zip(grads, model.trainable_variables),
                global_step)
        epoch_loss_avg(loss_value)
        epoch_accuracy(tf.argmax(
                model(features),
                axis=1,
                output_type=tf.int32),
            target_values)
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        if epoch % 50 == 0:
            print("Epoch {:03d} / {}: Loss: {:.3f}, Accuracy: {:.3%}".format(
                epoch,
                num_epochs,
                epoch_loss_avg.result(),
                epoch_accuracy.result()))

    return model

def test(model, class_names, *predict_dataset):
    predictions = model(tf.convert_to_tensor(predict_dataset))

    for i, logits in enumerate(predictions):
        class_idx = tf.argmax(logits).numpy()
        p = tf.nn.softmax(logits)[class_idx]
        name = class_names[class_idx]
        print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))

if __name__ == '__main__':
    model = get_trained_model()
    class_names = ['setosa', 'versicolor', 'virginica']
    test(
        model,
        class_names,
        [5.1, 3.3, 1.7, 0.5,],
        [5.9, 3.0, 4.2, 1.5,],
        [6.9, 3.1, 5.4, 2.1],
        [7.2, 6. , 5. , 2.1])
