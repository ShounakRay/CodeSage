# @Author: shounak.ray
# @Date:   2022-06-28T09:44:02-07:00
# @Last modified by:   shounak.ray
# @Last modified time: 2022-06-30T11:50:00-07:00

import sys
print(sys.version_info)

import matplotlib.pyplot as plt
import math
import numpy as np
import networkx as nx
import pandas as pd
from sklearn import datasets
from sklearn.datasets import make_blobs
from scipy.spatial.distance import squareform, cdist
from sklearn.manifold import MDS  # for MDS dimensionality reduction
import scipy
from datetime import datetime
from tqdm import tqdm
from matplotlib import cm
from sklearn.preprocessing import MinMaxScaler
import matplotlib.animation as animation
import imageio.v2 as imageio
import glob
import os
from vectorizer import _get_data, vectorize


def _soft_sanitation(variable, msg='Cannot complete operation; requires previous step.'):
    if variable is None:
        print(msg)
        return


def _normalize(numpy_arr):
    return (numpy_arr - numpy_arr.min(0)) / numpy_arr.ptp(0)


class SOM:
    def __init__(self, neurons, learning_rate, epochs, sigma_0, convergence_threshold, neuron_dim=2, **kwargs):
        self.neurons = float(neurons)
        self.learning_rate = float(learning_rate)
        self.epochs = epochs
        self.tau_rate = float(kwargs.get('tau_rate', epochs))
        self.sigma_0 = float(sigma_0)
        self.tau_neighbourhood = float(kwargs.get('tau_rate', epochs / np.log(sigma_0)))
        self.convergence_threshold = float(convergence_threshold)
        self.neuron_dim = int(neuron_dim)

        self._node_min_value = kwargs.get('NODE_MIN_VALUE', 0)
        self._node_max_value = kwargs.get('NODE_MAX_VALUE', 1)
        self._weight_min = kwargs.get('NODE_WEIGHT_MIN', 0)
        self._weight_max = kwargs.get('NODE_WEIGHT_MAX', 1)

        self.adjustment_history = []
        self.curr_epoch = 0

    def create_feature_map(self, num_features):
        def naivetuple_to_pos(naive_tuple, _single_num_neurons, _node_stepsize):
            return (naive_tuple[0] * _node_stepsize, naive_tuple[1] * _node_stepsize)

        def _random_weight_vector(num_features):
            return np.array([np.random.uniform(self._weight_min, self._weight_max) for _ in range(num_features)])

        # Adjust mapping dimensions, if required
        self.num_features = num_features
        self.weighdim_matches_inputdim = (self.neuron_dim == self.num_features)
        _single_num_neurons = math.ceil(np.float_power(self.neurons, 1 / self.neuron_dim))
        self.adj_neurons = _single_num_neurons ** self.neuron_dim
        print(f"> Initializing {self.adj_neurons} neurons in the {self.neuron_dim}-dimensional feature map...")
        _node_stepsize = (self._node_max_value - self._node_min_value) / _single_num_neurons

        if self.neuron_dim == 2:
            # Finally make the mapping
            G = nx.grid_2d_graph(_single_num_neurons, _single_num_neurons)
            attrs = {node: {'type': 'neuron',
                            'position': naivetuple_to_pos(node, _single_num_neurons, _node_stepsize),
                            'weight_vector': _random_weight_vector(self.num_features),
                            'adjustment_history': []} for node in G.nodes()}
            nx.set_node_attributes(G, attrs)
            self.neuronal_data = G
        elif self.neuron == 3:
            pass    # TODO; Don't know how to do this yet.
        print("> Feature map initialized.\n")

    def _plot_neuronal_grid(self, figsize=(10, 10), only_draw_nodes=False, save=False, **kwargs):
        hist = {k: v for k, v in nx.get_node_attributes(self.neuronal_data, 'adjustment_history').items() if v != []}
        colors = ['green' if k in list(hist.keys()) else 'red' for k in self.neuronal_data.nodes]

        _viridis = cm.get_cmap('viridis', 8)
        colors_by_weight = np.array([np.linalg.norm(wv) for wv in nx.get_node_attributes(self.neuronal_data, 'weight_vector')])
        colors_by_weight = MinMaxScaler(feature_range=(0, 1)).fit_transform(colors_by_weight.reshape(-1, 1))
        colors_by_weight = [_viridis(x) for x in colors_by_weight]

        sizes = np.array([np.linalg.norm(self.neuronal_data[node]) for node in self.neuronal_data.nodes])
        sizes = MinMaxScaler(feature_range=(200, 500)).fit_transform(sizes.reshape(-1, 1))

        position = nx.get_node_attributes(self.neuronal_data, 'position')

        kwargs = kwargs | {'color': colors, 'size': sizes, 'color_weight': colors_by_weight, 'position': position}
        _ = plt.figure(figsize=figsize)
        _ = plt.title(f'Epoch {self.curr_epoch}')
        _ = plt.axis('off')
        if only_draw_nodes:
            _ = nx.draw_networkx_nodes(self.neuronal_data, pos=position,
                                       node_color=kwargs.get('color_weight', 'lightgreen'),
                                       node_size=kwargs.get('sizes', 200))
        else:
            _ = nx.draw(self.neuronal_data, pos=position,
                        node_color=kwargs.get('color_weight', 'lightgreen'),
                        with_labels=False,
                        node_size=kwargs.get('sizes', 200),
                        ax=kwargs.get('ax'))
        if save:
            plt.savefig(kwargs.get('fname', f'pictures/neurons-{self.curr_epoch}.jpeg'))
            plt.close()

    def _plot_neuronal_mds(self, save=False, **kwargs):
        coordinates = list(nx.get_node_attributes(self.neuronal_data, 'weight_vector').values())
        model2d = MDS(n_components=2, metric=True,
                      n_init=4, max_iter=300, verbose=0, eps=0.001,
                      n_jobs=-1, random_state=42, dissimilarity='euclidean',
                      normalized_stress='auto')
        X_trans = model2d.fit_transform(coordinates)
        # plt.figure(figsize=(8, 8))
        _ = plt.figure(figsize=(10, 10))
        _ = plt.title(f'Epoch {self.curr_epoch}')
        _ = plt.axis('off')
        _ = plt.scatter(x=X_trans[:, 0], y=X_trans[:, 1], alpha=0.9, c='black', s=2)
        if save:
            plt.savefig(kwargs.get('fname', f'pictures/neurons-{self.curr_epoch}.jpeg'))
            plt.close()

    def fit(self, data, animate=True, animate_method='mds_neurons', anim_every_n_epochs=250, **kwargs):
        print('> Fitting map...')
        if len(data[0]) != self.num_features:
            print("FATAL: The number of features detected in the data doesn't match what was entered during map creation. Redo map creation.")
            return
        if len(data) == 0:
            print("FATAL: You entered an empty dataset. Retry.")
            return

        def _get_neighbouring_nodes(bmu):
            return [bmu] + list(nx.neighbors(self.neuronal_data, bmu))

        def _get_random_input_vector(data):
            return data[np.random.randint(0, len(data) - 1)]

        def _get_bmu(input_vector):
            bmu_index = np.argmin([np.linalg.norm(x - input_vector) for x in list(nx.get_node_attributes(self.neuronal_data, 'weight_vector').values())],
                                  axis=0)
            # plt.hist([np.linalg.norm(x - input_vector) for x in list(nx.get_node_attributes(self.neuronal_data, 'weight_vector').values())], bins=30)
            # [np.linalg.norm(x - input_vector) for x in list(nx.get_node_attributes(self.neuronal_data, 'weight_vector').values())][bmu_index]
            return list(self.neuronal_data.nodes)[bmu_index]  # BMU

        if animate:
            # Delete existing files in folder
            [os.remove(f) for f in glob.glob('pictures/*')]
            # make folder if it doesn't exist
            os.makedirs(r'pictures') if not os.path.exists(r'pictures') else None
            fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 10)))
        try:
            for epoch in tqdm(range(self.curr_epoch, self.epochs + 1)):
                self.curr_epoch = epoch
                input_vector = _get_random_input_vector(data)
                bmu = _get_bmu(input_vector)
                if animate and epoch % anim_every_n_epochs == 0 and epoch < self.epochs:
                    if animate_method == 'grid_neurons':
                        self._plot_neuronal_grid(only_draw_nodes=kwargs.get('only_draw_nodes'), save=True)
                    elif animate_method == 'mds_neurons':
                        self._plot_neuronal_mds(save=True, **kwargs)
                    else:
                        raise ValueError('Other methods not supported yet.')
                adj_mags = [self.update_weight(epoch, bmu, neighbour, input_vector) for neighbour in _get_neighbouring_nodes(bmu)]
                self.adjustment_history.append(mean_change := np.mean(adj_mags))
                if mean_change <= self.convergence_threshold:
                    print(f"Early stopping at epoch {epoch}. Convergence threshold reached.")
                    break
        except KeyboardInterrupt as e:
            print('Interrupted...working with what we have.')
        print('> Done fitting map.')

        if animate:
            print('> Animating neuronal updates...')
            nums = sorted([int(''.join(filter(str.isdigit, s))) for s in glob.glob('pictures/*.jpeg')])
            img_paths = [f'pictures/neurons-{n}.jpeg' for n in nums]
            if len(img_paths) == 0:
                raise ValueError('Didn\'t find any paths inside "\\picture" folder.')
            
            # ims = [imageio.imread(f) for f in img_paths]
            # imageio.mimwrite(f'animated_file-{str(datetime.now())}.mp4', ims, fps=60)

            # Create an Imageio writer object for the MP4 file
            writer = imageio.get_writer(f'animated_file-{str(datetime.now())}.mp4', fps=60)

            # Loop through each JPEG image and add it to the writer object
            for image_file in img_paths:
                image = imageio.imread(image_file)
                writer.append_data(image)
            # Close the writer object to finalize the MP4 file
            writer.close()

            plt.close()
            print('> Finished.')

    def update_weight(self, epoch, bmu, neighbour, input_vector, **kwargs):
        def adaptive_eta(epoch):
            return self.learning_rate * math.exp(-epoch / self.tau_rate)
            # return self.learning_rate / (1 + epoch / (self.epochs / 2))  # ALT

        def adaptive_sigma(epoch):
            return self.sigma_0 * math.exp(-epoch / self.tau_neighbourhood)

        def topological_neighourhood(epoch, neighbour, bmu):
            # Range (0, 1)
            lateral_distance = np.linalg.norm(self.neuronal_data.nodes[bmu]['weight_vector'] - self.neuronal_data.nodes[neighbour]['weight_vector'])
            # sigma = constrain(self.sigma_0 / (1 + epoch / self.epochs))  # ALT
            return math.exp(-(lateral_distance**2) / (2 * adaptive_sigma(epoch)**2))

        current_weight = self.neuronal_data.nodes[neighbour]['weight_vector']
        adjustment = adaptive_eta(epoch) * topological_neighourhood(epoch, neighbour, bmu) * (input_vector - current_weight)
        self.neuronal_data.nodes[neighbour]['weight_vector'] = current_weight + adjustment
        magnitude = np.sqrt(adjustment.dot(adjustment))  # adjustment magnitude
        self.neuronal_data.nodes[neighbour]['adjustment_history'].append(magnitude)
        if self.weighdim_matches_inputdim:
            self.neuronal_data.nodes[neighbour]['position'] = self.neuronal_data.nodes[neighbour]['weight_vector']
        else:
            pass  # TODO; find a way to update 2d networkx position deterministically
        return magnitude
        # print(f"Cosine Similarity: {1 - scipy.spatial.distance.cosine(current_weight, current_weight + adjustment)}")


""" SKLEARN """
# data = pd.DataFrame(datasets.load_iris()['data'], columns=datasets.load_iris()['feature_names']).to_numpy()
# data = pd.DataFrame(datasets.fetch_covtype()['data'],
#                     columns=datasets.fetch_covtype()['feature_names']).to_numpy()
""" ARTIFICAL CLUSTER – 3 BLOBS """
# data, labels_true = make_blobs(n_samples=1000, centers=[[0, 0], [2.5, 4], [5, 0]], cluster_std=0.8, random_state=0)
# data = _normalize(data)
# # _ = plt.scatter(*zip(*data))
""" ARTIFICAL CLUSTER – SMILY FACE """
# data, labels_true = make_blobs(n_samples=1000, centers=[[0, 4], [4, 4]], cluster_std=0.2, random_state=0)
# data_arc = np.array([[i, -0.5 * np.sin(i)] for i in np.arange(0, np.pi, 0.1)])
# data_arc = [make_blobs(n_samples=80, centers=[c], cluster_std=0.1, random_state=0)[0] for c in data_arc]
# data_arc = np.concatenate(data_arc)
# data = np.vstack((data, data_arc))
# # Add noise
# # data = np.hstack((data, np.random.normal(0, 1, len(data)).reshape(-1, 1)))
# # data = np.hstack((data, np.random.normal(0, 1, len(data)).reshape(-1, 1)))
# data = _normalize(data)
# # _ = plt.scatter(*zip(*data))
""" VECTORIZED TEXT DATA – 4 Categories """
# data = vectorize(_get_data(source='json'), mode='pretrained')
# print("Input Data:\n", data, "\n\n")
# # data = _normalize(data)
raise NotImplementedError
def run_model(data=data):
    # Class/categorical balance is important!
    neurons = 5 * np.sqrt(len(data)) / 2
    learning_rate = 0.4
    epochs = 50000       # Can be determined by likelihood that every sample is seen in the data. (or change algo accordingly)
    sigma_0 = 100       # Should be some function of num_features. Pull harder if there's a lot of complexity.
    convergence_threshold = 1e-4
    # You want to observe the progression of a pattern slowly
    # TODO: Add Threading
    # Use PCA initialization for reproducibility and accuracy

    S = SOM(neurons=neurons, learning_rate=learning_rate, epochs=epochs, sigma_0=sigma_0, convergence_threshold=convergence_threshold, neuron_dim=2)
    S.create_feature_map(len(data[0]))
    # grid_neurons vs mds_neurons
    S.fit(data, animate=True, animate_method='mds_neurons', anim_every_n_epochs=10, only_draw_nodes=False)
    # S._plot_neuronal_mds(fname='json_neuronal_mds.jpeg', save=True)
    S._plot_neuronal_grid(fname='json_neuronal_grid.jpeg', save=True)


# nx.draw(S.neuronal_data, pos=nx.get_node_attributes(S.neuronal_data, 'position'))
# plt.figure(figsize=(50, 20))
# _ = plt.plot(S.adjustment_history)


if __name__ == '__main__':
    run_model()

# EOF
