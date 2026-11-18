from PIL import Image
from numpy import asarray

class OctThicknessMapping:
    def __init__(self)


    def get_pixel_thickness_values(self, ):
         = []


        for ez_img, rpe_img in ez_imgs, rpe_imgs:

            ez_img = Image.open('Sample1.png')
            rpe_img = Image.open('Sample2.png')
            res = asarray(img)
            print(res)



# get the hex pixel values from the two oct scans provided from Justis/Cleveland - the EZ->RPE and RPE->BM map pngs
# def get_hex_pixel_values(self):
