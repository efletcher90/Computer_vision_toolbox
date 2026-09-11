from UNet_architectures.unet_pytorch import UNet
from Data_utilities.unet_pytorch_training import TrainingSetGenerator
import unet_config as u_conf
from unet_config import L_RATE, LOSS_F





if '__name__' == '__main__':
    Controller()


    def build_unet(self):
        model = UNet(input_channels=1, num_classes=1)
        loss_f = u_conf.LOSS_F
        optimiser = torch.optim.Adam(model.parameters(), lr=u_conf.L_RATE)