import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Input
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

tf.get_logger().setLevel(logging.ERROR)
keras.utils.set_random_seed(17)

EPOCHS = 100
BATCH_SIZE = 16
DROPOUT_VAL = 0.3
WINDOW_SIZE = 10

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

x_train = train_data['x'].values
X, y = [], []
for i in range(len(x_train) - WINDOW_SIZE):
    X.append(x_train[i:i + WINDOW_SIZE])
    y.append(x_train[i + WINDOW_SIZE])
X, y = np.array(X), np.array(y)

train_size = int(0.8 * len(X))
X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:], y[train_size:]

inputs = Input(shape=(WINDOW_SIZE, 1))
x = LSTM(64, return_sequences=True)(inputs)
x = Dropout(DROPOUT_VAL)(x)
x = LSTM(32)(x)
x = Dense(16, activation="relu", kernel_initializer="glorot_uniform")(x)
output = Dense(1, activation="linear", kernel_initializer="glorot_uniform")(x)

model = keras.Model(inputs, output)

model.compile(
    loss='mse',
    optimizer='adam',
    metrics=['mae']
)

model.summary()

history = model.fit(
    X_train.reshape(-1, WINDOW_SIZE, 1),
    y_train,
    validation_data=(X_val.reshape(-1, WINDOW_SIZE, 1), y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
    shuffle=True
)

print("\nИстория обучения (первые 4 эпохи):")
for epoch in range(min(4, EPOCHS)):
    print(f"Epoch {epoch+1}/{EPOCHS} - Train MAE: {history.history['mae'][epoch]:.4f} - Train Loss: {history.history['loss'][epoch]:.4f} - Val MAE: {history.history['val_mae'][epoch]:.4f} - Val Loss: {history.history['val_loss'][epoch]:.4f}")

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['train', 'validation'])
plt.savefig('loss_plot.png')
plt.close()

plt.plot(history.history['mae'], label='mae')
plt.plot(history.history['val_mae'], label='val_mae')
plt.title('Mean Absolute Error')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend(['train', 'validation'])
plt.savefig('mae_plot.png')
plt.close()

last_sequence = x_train[-WINDOW_SIZE:]
predictions = []

current_sequence = last_sequence.copy()
for _ in range(len(test_data)):
    input_seq = current_sequence.reshape(1, WINDOW_SIZE, 1)
    pred = model.predict(input_seq, verbose=0)[0][0]
    predictions.append(pred)
    current_sequence = np.append(current_sequence[1:], pred)

pred_df = pd.DataFrame({'x': predictions})
pred_df.to_csv('output.csv', index=False)
print("\nПредсказания сохранены в pred.csv")

val_metrics = model.evaluate(X_val.reshape(-1, WINDOW_SIZE, 1), y_val, verbose=2)
print(f"\nВалидация: MAE = {val_metrics[1]:.4f}, Loss = {val_metrics[0]:.4f}")