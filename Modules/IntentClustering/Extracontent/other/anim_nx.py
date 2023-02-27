# @Author: shounak.ray
# @Date:   2022-06-29T12:53:01-07:00
# @Last modified by:   shounak.ray
# @Last modified time: 2022-06-30T00:21:33-07:00

import networkx as nx
import matplotlib.pyplot as plt
from celluloid import Camera
import glob
import imageio.v2 as imageio


def this_graphs(i):
    G = nx.grid_2d_graph(i, i)
    nx.draw_networkx_nodes(G, pos=nx.spring_layout(G), node_size=20)


fig, ax = plt.subplots(figsize=(10, 10))
camera = Camera(fig)

for i in range(20):
    this_graphs(i)
    camera.snap()

this_graphs(10)
plt.savefig('some.png')

animation = camera.animate(interval=100)
animation.save('test.mp4', writer='ffmpeg')

# Just animate
nums = sorted([int(''.join(filter(str.isdigit, s))) for s in glob.glob('pictures/*.png')])
img_paths = [f'pictures/neurons-{n}.png' for n in nums]
ims = (imageio.imread(f) for f in img_paths)
imageio.mimwrite('file.mp4', list(ims), fps=60)
