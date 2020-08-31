import colorsys
import matplotlib.pyplot as plt
import seaborn as sns

N = 5
HSV_tuples = [(x*1.0/N, 0.75, 0.75) for x in range(N)]
RGB_tuples = map(lambda x: colorsys.hsv_to_rgb(*x), HSV_tuples)

# for i in RGB_tuples:
#     sns.palplot(sns.light_palette(color=i, n_colors=4, reverse=False, input='RGB'))
#     plt.show()

print(sns.hls_palette(n_colors=N, h=0.2, l=0.5, s=0.5))

# for i in range(N):
#     sns.palplot(sns.light_palette(color=sns.hls_palette(n_colors=N, h=0.2, l=0.5, s=0.5)[i], 
#         n_colors=4, reverse=False, input='RGB'))
#     plt.show()