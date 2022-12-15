import numpy as np
import unittest
import kmeans


class TestKMeans(unittest.TestCase):


    def test_clustering(self):

        # Generate some dummy data
        X = np.array([[1, 2], [1, 4], [1, 0],
                      [4, 2], [4, 4], [4, 0]])

        # Create a KMeans model
        model = kmeans.KMeans(n_clusters=2, random_state=0)

        # Cluster the data using the model
        clusters = model.fit_predict(X)

        # Check that the expected number of clusters was formed
        self.assertEqual(len(np.unique(clusters)), 2)

        # Check that the points in each cluster are closer to their cluster's centroid
        for i in range(2):
            cluster_points = X[clusters == i]
            centroid = np.mean(cluster_points, axis=0)
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            mean_distance = np.mean(distances)
            mean_distance_from_centroid = np.mean(np.linalg.norm(X - centroid, axis=1))
            self.assertTrue(mean_distance <= mean_distance_from_centroid)


if __name__ == '__main__':
    unittest.main()
