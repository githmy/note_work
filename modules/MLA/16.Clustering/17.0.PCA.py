from sklearn.utils.testing import (assert_array_almost_equal, assert_less,
                                   assert_equal, assert_not_equal,
                                   assert_raises)
from sklearn.decomposition import PCA, KernelPCA

rng = np.random.RandomState(0)
X_fit = rng.random_sample((5, 4))
X_pred = rng.random_sample((2, 4))

X = []
for eigen_solver in ("auto", "dense", "arpack"):
    for kernel in ("linear", "rbf", "poly", histogram):
        # histogram kernel produces singular matrix inside linalg.solve
        # XXX use a least-squares approximation?
        inv = not callable(kernel)

        # transform fit data
        # 行为样本，列为特征
        kpca = KernelPCA(4, kernel=kernel, eigen_solver=eigen_solver,
                         fit_inverse_transform=inv)
        X_fit_transformed = kpca.fit_transform(X_fit)
        X_fit_transformed2 = kpca.fit(X_fit).transform(X_fit)
        assert_array_almost_equal(np.abs(X_fit_transformed),
                                  np.abs(X_fit_transformed2))

        # non-regression test: previously, gamma would be 0 by default,
        # forcing all eigenvalues to 0 under the poly kernel
        assert_not_equal(X_fit_transformed.size, 0)

        # transform new data
        X_pred_transformed = kpca.transform(X_pred)
        assert_equal(X_pred_transformed.shape[1],
                     X_fit_transformed.shape[1])

        # inverse transform
        if inv:
            X_pred2 = kpca.inverse_transform(X_pred_transformed)
            assert_equal(X_pred2.shape, X_pred.shape)

pca = PCA()
X_pca = pca.fit_transform(X)

from sklearn.datasets import load_digits
from sklearn.decomposition import KernelPCA
X, _ = load_digits(return_X_y=True)
transformer = KernelPCA(n_components=7, kernel='linear')
X_transformed = transformer.fit_transform(X)
X_transformed.shape
# (1797, 7)