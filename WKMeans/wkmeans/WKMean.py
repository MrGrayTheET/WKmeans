from scipy.stats import wasserstein_distance
from WKMeans.wkmeans.WKUtils import *
from sklearn.metrics.pairwise import rbf_kernel


class WKMeans:
    def __init__(self, k, max_iter=50, tol=1e-4, gamma=1.0, mmd_pairs=10, sample_size=50):
        """
        k: Number of clusters
        max_iter: Maximum number of iterations
        tol: Tolerance for convergence based on MMD loss
        gamma: RBF kernel parameter for MMD
        mmd_pairs: Number of random sample pairs to compute MMD
        sample_size: Sample size used in each MMD pair
        """
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.gamma = gamma
        self.mmd_pairs = mmd_pairs
        self.sample_size = sample_size
        self.assignments = None
        self.centroids = None

    def _compute_barycenter(self, distributions):
        sorted_distributions = [np.sort(d) for d in distributions]
        min_len = min(len(d) for d in sorted_distributions)
        trimmed = [d[:min_len] for d in sorted_distributions]
        return np.mean(trimmed, axis=0)

    def _compute_mmd(self, x, y):
        K_xx = rbf_kernel(x, x, gamma=self.gamma)
        K_yy = rbf_kernel(y, y, gamma=self.gamma)
        K_xy = rbf_kernel(x, y, gamma=self.gamma)

        n = x.shape[0]
        m = y.shape[0]

        mmd_squared = (K_xx.sum() - np.trace(K_xx)) / (n * (n - 1)) \
                      + (K_yy.sum() - np.trace(K_yy)) / (m * (m - 1)) \
                      - 2 * K_xy.mean()

        return mmd_squared

    def _cluster_self_similarity(self, cluster_samples):
        mmds = []
        n = len(cluster_samples)
        for _ in range(self.mmd_pairs):
            idx = np.random.choice(n, 2, replace=False)
            x = cluster_samples[idx[0]][:self.sample_size].reshape(-1, 1)
            y = cluster_samples[idx[1]][:self.sample_size].reshape(-1, 1)
            mmds.append(self._compute_mmd(x, y))
        return np.median(mmds)

    def fit(self, distributions):
        n = len(distributions)
        self.centroids = [distributions[i] for i in np.random.choice(n, self.k, replace=False)]
        prev_loss = float('inf')

        for iteration in range(self.max_iter):
            # Assignment step
            self.assignments = []
            for dist in distributions:
                dists = [wasserstein_distance(dist, c) for c in self.centroids]
                self.assignments.append(np.argmin(dists))

            # Group distributions into clusters
            clusters = [[] for _ in range(self.k)]
            for idx, cluster_id in enumerate(self.assignments):
                clusters[cluster_id].append(distributions[idx])

            # Update step
            new_centroids = []
            for cluster in clusters:
                if cluster:
                    new_centroids.append(self._compute_barycenter(cluster))
                else:
                    new_centroids.append(distributions[np.random.randint(n)])

            # Compute MMD loss
            loss = np.mean([
                self._cluster_self_similarity(np.array(cluster))
                for cluster in clusters if len(cluster) >= 2
            ])

            print(f"Iteration {iteration + 1}, MMD Loss: {loss:.6f}")

            if abs(prev_loss - loss) < self.tol:
                break

            self.centroids = new_centroids
            prev_loss = loss

        return self

    def predict(self, distributions):
        if self.centroids is None:
            raise ValueError("Model has not been fit yet.")
        return [ np.argmin([wasserstein_distance(dist, c) for c in self.centroids])  for dist in distributions
 ]


