import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class CNNModel(Model):
    def __init__(self):
        super(MyCNNModel, self).__init__()
        self.conv1 = Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))
        self.pool1 = MaxPooling2D(pool_size=(2, 2))
        self.conv2 = Conv2D(64, (3, 3), activation='relu')
        self.pool2 = MaxPooling2D(pool_size=(2, 2))
        self.flatten = Flatten()
        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
        'mnist/train',
        target_size=(28, 28),
        color_mode='grayscale',
        batch_size=32,
        class_mode='categorical')

test_generator = datagen.flow_from_directory(
    'mnist/test',
    target_size = (28, 28),
    color_mode = 'grayscale',
    batch_size = 32,
    class_mode = 'categorical'
)

model = MyCNNModel()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_generator, epochs=10)
