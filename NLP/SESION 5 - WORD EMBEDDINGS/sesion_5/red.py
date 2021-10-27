from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop

# Nuestro feature map de entrada es de 150x150x3, con 150x150 píxels y 3 canales (RGB)
img_input = layers.Input(shape=(150, 150, 3))

# La primera convolución extrae 16 filtros de 3x3, relu va diréctamente embebido en esta capa
# La convolución va seguida, en este caso, de una capa de max-pooling con ventanas de 2x2
x = layers.Conv2D(16, 3, activation='relu')(img_input)
x = layers.MaxPooling2D(2)(x)

# La segunda convolución extrae 32 filtros de 3x3
# La convolución va seguida de una capa de max-pooling con ventanas de 2x2
x = layers.Conv2D(32, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

# La segunda convolución extrae 64 filtros de 3x3
# La convolución va seguida de una capa de max-pooling con ventanas de 2x2
x = layers.Convolution2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

# Vectorizamos el feature-map de salida mediante la capa flatten
x = layers.Flatten()(x)

# Creamos una capa fully-connected con función de activación ReLU y 512 neuronas
x = layers.Dense(512, activation='relu')(x)

# TODO 2. Añadir una capa de dropout con ratio de drop del 0.5

# Creamos la capa de salida con un sólo nodo y función de activación sigmoide
output = layers.Dense(1, activation='sigmoid')(x)

# Configuramos y compilamos el modelo
model = Model(img_input, output)
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])