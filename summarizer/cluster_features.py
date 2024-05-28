import logging
from typing import List, Dict, Union
import numpy as np
from numpy import ndarray
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClusterFeatures:
    """
    Basic handling of clustering features.
    """

    def __init__(
        self,
        features: ndarray,
        algorithm: str = 'kmeans',
        pca_k: int = None,
        random_state: int = 12345
    ):
        """
        Initializes the ClusterFeatures with the specified parameters.

        :param features: The embedding matrix created by BERT parent.
        :param algorithm: Which clustering algorithm to use ('kmeans' or 'gmm').
        :param pca_k: Number of components for PCA, if None, PCA is not applied.
        :param random_state: Random state for reproducibility.
        """
        if not isinstance(features, np.ndarray):
            logger.error("Features must be a numpy array.")
            raise ValueError("Features must be a numpy array.")

        if pca_k:
            self.features = PCA(n_components=pca_k, random_state=random_state).fit_transform(features)
            logger.info("Applied PCA with %d components.", pca_k)
        else:
            self.features = features

        if algorithm not in {'kmeans', 'gmm'}:
            logger.error("Algorithm must be either 'kmeans' or 'gmm'.")
            raise ValueError("Algorithm must be either 'kmeans' or 'gmm'.")

        self.algorithm = algorithm
        self.pca_k = pca_k
        self.random_state = random_state
        logger.info("Initialized ClusterFeatures with algorithm: %s, PCA components: %s.", algorithm, pca_k)

    def __get_model(self, k: int) -> Union[KMeans, GaussianMixture]:
        """
        Retrieve clustering model.

        :param k: Number of clusters.
        :return: Clustering model instance.
        """
        if self.algorithm == 'gmm':
            return GaussianMixture(n_components=k, random_state=self.random_state)
        return KMeans(n_clusters=k, random_state=self.random_state)

    def __get_centroids(self, model: Union[KMeans, GaussianMixture]) -> ndarray:
        """
        Retrieve centroids of the model.

        :param model: Clustering model.
        :return: Centroids of the clusters.
        """
        if self.algorithm == 'gmm':
            return model.means_
        return model.cluster_centers_

    def __find_closest_args(self, centroids: ndarray) -> Dict[int, int]:
        """
        Find the closest arguments to centroids.

        :param centroids: Centroids to find closest arguments.
        :return: Dictionary of centroid index to closest feature index.
        """
        closest_args = {}
        used_indices = set()

        for j, centroid in enumerate(centroids):
            closest_index = -1
            min_distance = float('inf')

            for i, feature in enumerate(self.features):
                if i in used_indices:
                    continue
                distance = np.linalg.norm(feature - centroid)
                if distance < min_distance:
                    closest_index = i
                    min_distance = distance

            closest_args[j] = closest_index
            used_indices.add(closest_index)

        logger.info("Found closest arguments for centroids.")
        return closest_args

    def cluster(self, ratio: float = 0.1) -> List[int]:
        """
        Clusters sentences based on the ratio.

        :param ratio: Ratio of the number of clusters to the number of features.
        :return: List of sentence indices that qualify for summary.
        """
        if not (0 < ratio <= 1):
            logger.error("Ratio must be between 0 and 1.")
            raise ValueError("Ratio must be between 0 and 1.")

        k = max(1, int(len(self.features) * ratio))
        model = self.__get_model(k).fit(self.features)
        centroids = self.__get_centroids(model)
        cluster_args = self.__find_closest_args(centroids)
        sorted_indices = sorted(cluster_args.values())

        logger.info("Clustered features with ratio: %.2f, resulting in %d clusters.", ratio, k)
        return sorted_indices

    def __call__(self, ratio: float = 0.1) -> List[int]:
        """
        Allows the instance to be called as a function to cluster features.

        :param ratio: Ratio of the number of clusters to the number of features.
        :return: List of sentence indices that qualify for summary.
        """
        return self.cluster(ratio)