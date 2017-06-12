import abc
import itertools
import logging
from typing import List, Tuple

import numpy as np
from keras import Input
from keras.layers import Dense, Concatenate
from keras.models import Model, Sequential, load_model
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors


class Classifier(object, metaclass=abc.ABCMeta):

    @abc.abstractclassmethod
    def train(self, training_data):
        pass

    @abc.abstractclassmethod
    def classify(self, unknowns: np.array)->List[str]:
        pass


class KNNClassifier(Classifier, metaclass=abc.ABCMeta):

    @abc.abstractclassmethod
    def train(self, training_data: Tuple[np.array, np.array]):
        pass


class ProjectionClassifier(Classifier, metaclass=abc.ABCMeta):

    @abc.abstractclassmethod
    def train(self, training_data: List[Tuple[np.array, np.array]]):
        pass


class NeuralNetworkClassifier(Classifier, metaclass=abc.ABCMeta):

    @abc.abstractclassmethod
    def train(self, training_data: Tuple[np.array, np.array, List[str]]):
        pass

    @abc.abstractclassmethod
    def save_to_file(self, model_path: str):
        pass


class MostCommonClassClassifier(Classifier):

    def __init__(self):
        self.__common_class = None  # type: str

    def train(self, training_data: List[str]):
        self.__common_class = max(itertools.groupby(sorted(training_data)), key=lambda t: len(list(t[1])))[0]

    def classify(self, unknowns: np.array) -> List[str]:
        n = unknowns.shape[0]
        return [self.__common_class for _ in range(n)]


class DistanceKNNClassifier(KNNClassifier):

    def __init__(self, neighbors: int, n_jobs: int=-1):
        self.__classifier = KNeighborsClassifier(
            algorithm='auto',
            metric='minkowski',
            n_neighbors=neighbors,
            p=2,
            weights='distance',
            n_jobs=n_jobs
        )  # type: KNeighborsClassifier

    def train(self, training_data: Tuple[np.array, np.array]):
        x, y = training_data
        self.__classifier.fit(x, y)

    def classify(self, unknowns: np.array)->List[str]:
        if unknowns.shape[0] == 1:
            return self.__classifier.predict(unknowns.reshape(1, -1)).tolist()
        else:
            return self.__classifier.predict(unknowns).tolist()


class PiecewiseLinearProjectionClassifier(ProjectionClassifier):

    def __init__(self, embedding_size: int, clusters: int, sgd_iter: int=5,
                 n_jobs: int=-1):
        self.__clusters = clusters  # type: int
        self.__embedding_size = embedding_size  # type: int
        # fit predict shape: (_, embedding_size)
        self.__kmeans = MiniBatchKMeans(n_clusters=clusters)  # type: MiniBatchKMeans
        self.__labels = None  # type: List[str]
        # nearest neighbors shape: (_, embedding_size)
        self.__nearest_neighbors_superclasses = NearestNeighbors(
            metric='minkowski',
            n_neighbors=1,
            p=2,
            n_jobs=n_jobs
        )
        self.__sgd_iter = sgd_iter
        # fit predict shape (_, embedding_size+1)
        self.__sgd_regressors = [[SGDRegressor(n_iter=self.__sgd_iter) for _ in range(self.__embedding_size)]
                                 for _ in range(self.__clusters)]

    def train(self, training_data: Tuple[List[Tuple[np.array, np.array]], List[str]]):
        """
        Train the piecewise linear projection on subclass-superclass embeddings pairs.
        :param training_data: (training_samples, superclass_labels). training_samples is a list of subclass-superclass
            embeddings. superclass_labels are the corresponding labels (IDs) of the superclasses.
        :return: PiecewiseLinearProjectionClassifier is trained as side-effect, nothing is returned.
        """
        training_samples, self.__labels = training_data

        n = len(training_samples)

        x, y = list(map(np.array, zip(*training_samples)))
        del training_samples

        mod_x = np.append(x, np.ones((n, 1)), axis=1)

        # fit nearest neighbors on word embeddings
        self.__nearest_neighbors_superclasses.fit(y)
        logging.log(level=logging.INFO, msg='finished fitting all-embeddings nearest neighbor')

        # compute clusters based on x (inputs)
        cluster_labels = self.__kmeans.fit_predict(x)
        logging.log(level=logging.INFO, msg='finished fit and predict of clusters, clusters={}'.format(self.__clusters))

        # fit nearest neighbor on cluster centers
        clustered_x = [list() for _ in range(self.__clusters)]
        clustered_y = [list() for _ in range(self.__clusters)]
        for i, c in enumerate(cluster_labels):
            clustered_x[c].append(mod_x[i])
            clustered_y[c].append(y[i])

        clustered_x = list(map(np.array, clustered_x))
        clustered_y = list(map(np.array, clustered_y))
        logging.log(level=logging.INFO, msg='sorted training samples into clusters')

        for c in range(self.__clusters):
            for target in range(self.__embedding_size):
                self.__sgd_regressors[c][target].fit(clustered_x[c], clustered_y[c][:, target])
            logging.log(level=logging.INFO, msg='sgd regression: cluster={}/{}'.format(c+1, self.__clusters))
        logging.log(level=logging.INFO, msg='finished training projections')

    def classify(self, unknowns: np.array)->List[str]:
        # add 1 row to unknowns matrix for translation
        mod_unknowns = np.append(unknowns, np.ones((unknowns.shape[0], 1)), axis=1)
        # find most appropiate cluster for all unknowns
        clusters = self.__kmeans.predict(unknowns)
        # compute projections for all unknowns
        projections = np.zeros((unknowns.shape[0], self.__embedding_size))
        for idx, cluster in enumerate(clusters):
            for target in range(self.__embedding_size):
                projections[idx][target] = self.__sgd_regressors[cluster][target]\
                    .predict(mod_unknowns[idx].reshape(1, -1))
        # find closest class to each projection => superclass
        _, indexes = self.__nearest_neighbors_superclasses.kneighbors(projections, return_distance=True)
        # retrieve the label of each superclass
        labels = list()
        for i in range(indexes.shape[0]):
            labels.append(self.__labels[indexes[i][0]])

        return labels


class DeepFFRegressionClassifier(NeuralNetworkClassifier):

    def __init__(self, embedding_size: int, n_hidden_neurons: int, n_hidden_layers: int,
                 batch_size: int, epochs: int, n_jobs: int, model_path: str=None):
        self.__batch_size = batch_size
        self.__epochs = epochs
        self.__superclass_embeddings = np.array(list())
        self.__superclass_labels = list()

        self.__nearest_neighbors = NearestNeighbors(
            metric='minkowski',
            n_neighbors=1,
            p=2,
            n_jobs=n_jobs
        )

        if model_path:
            self.__model = load_model(model_path)
            return

        self.__model = Sequential()
        self.__model.add(Dense(input_shape=(embedding_size,), units=embedding_size, activation='linear'))
        for _ in range(n_hidden_layers):
            self.__model.add(Dense(units=n_hidden_neurons, activation='relu'))
        self.__model.add(Dense(units=embedding_size, activation='linear'))
        self.__model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    def train(self, training_data: Tuple[np.array, np.array, List[str]]):
        x, self.__superclass_embeddings, self.__superclass_labels = training_data

        self.__nearest_neighbors.fit(self.__superclass_embeddings)

        self.__model.fit(x, self.__superclass_embeddings, verbose=1, batch_size=self.__batch_size,
                         epochs=self.__epochs, shuffle=True)

    def classify(self, unknowns: np.array)->List[str]:
        predictions = self.__model.predict(unknowns)
        _, indexes = self.__nearest_neighbors.kneighbors(predictions, return_distance=True)
        labels = list()
        for index in indexes:
            labels.append(self.__superclass_labels[index[0]])
        return labels

    def save_to_file(self, model_path: str):
        self.__model.save(model_path)


class ConcatFFRegressionClassifier(NeuralNetworkClassifier):

    def __init__(self, activation: str, embedding_size: int, n_networks: int, n_hidden_layers: int,
                 n_hidden_neurons: int, batch_size: int, epochs: int, n_jobs: int, model_path: str=None):
        # Check that combination n_networks and embedding_size is valid.
        assert embedding_size % n_networks == 0 and embedding_size >= n_networks
        assert n_hidden_layers >= 1 and n_hidden_neurons >= 1 and embedding_size >= 1 and n_networks >= 1
        self.__activation = activation
        self.__n_networks = n_networks
        self.__n_hidden_layers = n_hidden_layers
        self.__n_outputs = int(embedding_size / n_networks)
        self.__batch_size = batch_size
        self.__epochs = epochs
        self.__superclass_embeddings = np.array(list())
        self.__superclass_labels = list()

        self.__nearest_neighbors = NearestNeighbors(
            metric='minkowski',
            n_neighbors=1,
            p=2,
            n_jobs=n_jobs
        )

        if model_path:
            self.__model = load_model(model_path)
            return

        inp = Input(name='input', shape=(embedding_size,))
        # Init first hidden layers on input layer.
        hidden_layers = [Dense(activation=self.__activation, units=n_hidden_neurons)(inp) for _ in range(n_networks)]
        # Init all other hidden layers on previous hidden layer.
        for _ in range(1, n_hidden_layers):
            hidden_layers = [Dense(activation=self.__activation, units=n_hidden_neurons)(hidden_layer)
                             for hidden_layer in hidden_layers]
        concat = Concatenate(name='concat')(hidden_layers)
        linear_output = Dense(name='output', activation='linear', units=embedding_size)(concat)
        self.__model = Model(inputs=inp, outputs=linear_output)
        self.__model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

    def train(self, training_data: Tuple[np.array, np.array, List[str]]):
        x, self.__superclass_embeddings, self.__superclass_labels = training_data

        self.__nearest_neighbors.fit(self.__superclass_embeddings)

        self.__model.fit(x, self.__superclass_embeddings, verbose=1, batch_size=self.__batch_size, epochs=self.__epochs,
                         shuffle=True)

    def classify(self, unknowns: np.array) -> List[str]:
        predictions = self.__model.predict(unknowns)
        _, indexes = self.__nearest_neighbors.kneighbors(predictions, return_distance=True)
        labels = list()
        for index in indexes:
            labels.append(self.__superclass_labels[index[0]])
        return labels

    def save_to_file(self, model_path: str):
        self.__model.save(model_path)
