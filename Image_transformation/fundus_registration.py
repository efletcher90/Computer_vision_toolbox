import matplotlib.pyplot as plt
import fundus_image_toolbox as fit

config = fit.get_registration_config()
config['device'] = 'cpu'

mp_png = r'C:\Users\EdFletcher\OneDrive - AAVantgarde\CARINA\Example Cleveland MP OCT data\sandbox\stillPicture.png'
oct_tif = r'C:\Users\EdFletcher\OneDrive - AAVantgarde\CARINA\Example Cleveland MP OCT data\sandbox\20101-01_Baseline_OS_SLO.tif'

model, matcher = fit.load_registration_model(config)

moving_image_aligned = fit.register(
    mp_png,
    oct_tif,
    show=True,
    show_mapping=True,
    config=config,
    model=model,
    matcher=matcher
)

print("Size of the moving and aligned images:", plt.imread(oct_tif).shape, moving_image_aligned.shape)
print("Use at least 512px images, as this is the internally used size. As can be seen from the above, the final output image keeps the original size.")