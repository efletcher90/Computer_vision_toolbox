from keras.losses import binary_crossentropy, dice
from keras.layers import Conv2D, Input, Rescaling, Conv2DTranspose, MaxPooling2D, Dropout, concatenate
from keras.models import Model
from keras.optimizers import Adam

class DoubleConvBlock(Model):
    def __init__(self, filters, dropout_rate, pool_size):
        super().__init__()
        self.convolution_1 = Conv2D(filters, kernel_size=(3,3), activation="leaky", kernel_initializer="he_normal", padding='same')
        self.dropout = Dropout(dropout_rate)
        self.convolution_2 = Conv2D(filters, kernel_size=(3,3), activation="leaky", kernel_initializer="he_normal", padding='same')

    def forward(self, inputs, training=False):
        x = self.convolution_1(inputs)
        x = self.dropout(x, training=training)
        x = self.convolution_2(x)

        return x

class DownSampleBlock(Model):
    def __init__(self, filters, dropout_rate, pool_size):
        super().__init__()
        self.enc_1 = DoubleConvBlock(filters, dropout_rate, pool_size)
        self.pool = MaxPooling2D(pool_size=(2,2))

    def forward(self, inputs, training=False):
        skip = self.enc_1(inputs, training=training)
        pooled = self.pool(skip)

        return skip, pooled