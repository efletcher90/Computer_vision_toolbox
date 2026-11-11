import pingouin as pg
import pandas as pd
import matplotlib.pyplot as plt

AAVB_DRKGRN, AAVB_LITGRN = ["#00443E","#BBFFAF"]

data_dict = {'gold_standard': [35, 36, 36.4, 36.6, 36.8, 37, 37.1, 37.3, 37.4, 37.6,
                         38, 38.1, 38.2, 38.5, 38.6, 38.7, 39, 39.1, 39.2, 39.3],

        'new': [32.2, 33.8, 34.0,35.6, 34.0, 34.9, 35.5, 35.3, 34.6, 35.1, 39.8,
                39.2, 40.2, 40.2, 40.4, 40.2, 40.6, 40.7, 41.2, 41.0]
        }

df = pd.DataFrame(data_dict)

ax = pg.plot_blandaltman(x=df['new'], y=df['gold_standard'], color='darkgreen', edgecolor=AAVB_DRKGRN, alpha=0.5, annotate=False)
ax.set_xlabel('Mean of paired measurements (mm$^2$)')
ax.set_ylabel('Delta of paired measurements (mm$^2$)')

zero, bias, upper, lower = ax.lines

zero.set_color('black')
zero.set_linestyle(':')

bias.set_color('darkgreen')
upper.set_color('green')
lower.set_color('green')

zero.set_label(f'Zero mean')
bias.set_label(f'Mean bias = {bias.get_ydata()[0]:.2f} mm$^2$')

# used the upper CI line here for the legend, could use the other if needed. Only one dashed line needed
upper.set_label(f'95% LoA range = +{upper.get_ydata()[0]:.2f} to {lower.get_ydata()[0]:.2f} mm$^2$')


ax.legend(loc='lower right', fontsize=9, frameon=False, framealpha=0.25, labelspacing=0.2, borderpad=0.3)

bias_ci, upper_ci, lower_ci = ax.patches

bias_ci.set_facecolor(AAVB_DRKGRN)
upper_ci.set_facecolor(AAVB_LITGRN)
lower_ci.set_facecolor(AAVB_LITGRN)

plt.show()
