from keras.layers import Conv2D, Layer, Input, Rescaling, Conv2DTranspose, MaxPooling2D, Dropout, Concatenate, LeakyReLU
from keras.models import Model
from keras.optimizers import Adam

from config import L_RATE

class DoubleConvBlock(Layer):
    def __init__(self, filters, dropout_rate):
        super().__init__()
        self.convolution_1 = Conv2D(filters, kernel_size=(3,3), kernel_initializer="he_normal", padding='same')
        self.leakyReLU_1 = LeakyReLU(negative_slope=0.1)
        self.dropout = Dropout(dropout_rate)
        self.convolution_2 = Conv2D(filters, kernel_size=(3,3), kernel_initializer="he_normal", padding='same')
        self.leakyReLU_2 = LeakyReLU(negative_slope=0.1)

    def call(self, inputs, training=False):
        x = self.convolution_1(inputs)
        x = self.leakyReLU_1(x)
        x = self.dropout(x, training=training)
        x = self.convolution_2(x)
        x = self.leakyReLU_2(x)

        return x

class DownSampleBlock(Layer):
    def __init__(self, filters, dropout_rate):
        super().__init__()
        self.enc_1 = DoubleConvBlock(filters, dropout_rate)
        self.pool = MaxPooling2D(pool_size=(2,2))

    def call(self, inputs, training=False):
        encoded_data = self.enc_1(inputs, training=training)
        pooled_data = self.pool(encoded_data)

        return encoded_data, pooled_data

class UpSampleBlock(Layer):
    def __init__(self, transpose_filters, conv_filters, dropout_rate, ):
        super().__init__()
        self.transpose = Conv2DTranspose(transpose_filters, kernel_size=(2, 2), strides=(2, 2), padding='same')
        self.concat = Concatenate(axis=3)
        self.decode = DoubleConvBlock(conv_filters, dropout_rate)

    def call(self, inputs, skip, training=False):
        x = self.transpose(inputs)
        x = self.concat([x, skip])
        x = self.decode(x, training=training)

        return x

class UNet(Model):
    def __init__(self, img_channels=1):
        super().__init__()
        self.rescale = Rescaling(1.0 / 255)

        self.encoder_conv1 = DownSampleBlock(64,0.1)
        self.encoder_conv2 = DownSampleBlock(128, 0.1)
        self.encoder_conv3 = DownSampleBlock(256, 0.2)
        self.encoder_conv4 = DownSampleBlock(512, 0.2)

        self.bottle_neck = DoubleConvBlock(1024, 0.3)

        self.decoder_conv1 = UpSampleBlock(transpose_filters=512, conv_filters=512, dropout_rate=0.2)
        self.decoder_conv2 = UpSampleBlock(transpose_filters=256, conv_filters=256, dropout_rate=0.2)
        self.decoder_conv3 = UpSampleBlock(transpose_filters=128, conv_filters=128, dropout_rate=0.1)
        self.decoder_conv4 = UpSampleBlock(transpose_filters=64, conv_filters=64, dropout_rate=0.1)

        self.output_conv = Conv2D(1, (1, 1), activation="sigmoid")

    def call(self, inputs, training=False):
        x = self.rescale(inputs)

        skip1, x = self.encoder_conv1(x, training=training)
        skip2, x = self.encoder_conv2(x, training=training)
        skip3, x = self.encoder_conv3(x, training=training)
        skip4, x = self.encoder_conv4(x, training=training)

        x = self.bottle_neck(x, training=training)

        x = self.decoder_conv1(x, skip4, training=training)
        x = self.decoder_conv2(x, skip3, training=training)
        x = self.decoder_conv3(x, skip2, training=training)
        x = self.decoder_conv4(x, skip1, training=training)

        return self.output_conv(x)

def build_unet(img_height, img_width, img_channels):
    """
    Usage:
        model = build_unet(1024, 1024, 1)
        model.compile(optimizer=Adam(learning_rate=...), loss=bce_and_dice_lossf, metrics=[dice])
    """
    model = UNet(img_channels=img_channels)
    inputs = Input((img_height, img_width, img_channels))
    outputs = model(inputs)

    return Model(inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    test_model = build_unet(256, 256, 1)
    test_model.compile(optimizer=Adam(learning_rate=L_RATE))
    test_model.summary()

    ### still need to sort out the loss functions
    ### and training



