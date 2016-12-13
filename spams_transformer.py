# The MIT License (MIT)
# Copyright (c) 2014-2017 University of Bristol
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

import spams


class SpamsTransformer(BaseEstimator, TransformerMixin):
    """
    This class performs dictionary learning on data using the SPArse Modeling Software (SPAMS).
    """
    
    def __init__(self, total_num_bases, l1_dictionary, l1_reconstruct, class_conditional=False,
                 positive_coefficients=True, l0_max=0, verbose=0, num_iterations=100,
                 minibatch_size=512, use_lasso=True):
        """
        
        :param total_num_bases: The total number of atoms to be learnt by the dictionary learning software. No default
                                value.
        :param l1_dictionary: Regularisation parameter (1-norm) for learning the dictionary. No default value.
        :param l1_reconstruct: Regularisation parameter (1-norm) for reconstruction. No default value.
        :param class_conditional: Whether to partition the data by class, and to learn atoms on a per-class basis.
                                  Default: False
                                  Note: the number of atoms per class will be @total_num_bases / num_classes.
        :param positive_coefficients: Constrains the coefficients to be positive in reconstruction. Default: True
        :param l0_max: When using OMP, this parameter sets the maximal number of coefficients to be used
                       Note: (0 \leq @l0_max \leq total_num_bases). Default: 0
        :param verbose: Verbosity level. Default: 0.
        :param num_iterations: Number of iterations for training. Default: 100.
        :param minibatch_size: The sample size for each minibatch. Default: 512.
        :param use_lasso: When `True` uses LASSO during optimisation, when `False` uses OMP.
        """
        
        self.total_num_bases = total_num_bases
        self.l1_dictionary = l1_dictionary
        self.l1_reconstruct = l1_reconstruct
        self.num_iterations = num_iterations
        self.minibatch_size = minibatch_size
        
        self.use_lasso = use_lasso
        self.l0_max = l0_max
        
        self.verbose = verbose
        
        self.class_conditional = class_conditional
        self.positive_coefficients = positive_coefficients
        
        self.dictionary = None
    
    def fit(self, x, y=None):
        """
        Fits the dictionary learning model.
        
        :param x: The input data. $x \in \mathbb{R}^{N \times M}. Here $N$ is the number of instances, and $M$ is the
                  dimensionality of each instance.
        :param y: The labels that are associated with the data. Only used when `self.class_conditional = True`.
        :return: self
        """
        
        def fit(x, num):
            return spams.trainDL(
                K=num,
                numThreads=2,
                X=np.asfortranarray(x.T),
                mode=[4, 2][self.use_lasso],
                lambda1=[self.l0_max, self.l1_dictionary][self.use_lasso],
                iter=self.num_iterations,
                verbose=self.verbose,
                posAlpha=self.positive_coefficients,
                batchsize=self.minibatch_size,
            )
        
        if self.class_conditional:
            unique_labels = np.unique(y)
            num_bases = self.total_num_bases / len(unique_labels)
            
            self.dictionary = np.column_stack(
                [fit(x[y == yy], num_bases) for yy in unique_labels]
            )
        
        else:
            self.dictionary = fit(x, self.total_num_bases)
        
        return self
    
    def transform(self, X, mask=None):
        """
        Transforms data X to coefficients.
        
        :param X:The input data. $x \in \mathbb{R}^{N \times M}. Here $N$ is the number of instances, and $M$ is the
                  dimensionality of each instance.
        :param mask: Allows missing data to be present in `X`. `mask` should be a binary matrix of the same shape as
                    `X`. An element that evaluates to `True` indicates that data is present, and `False` means that data
                    is missing. Set `mask = None` (default value) to do un-masked transformations.
        :return: Returns a sparse matrix
        """
        
        if self.use_lasso:
            return self._transform_lasso(X, mask)
        
        return self._transform_omp(X, mask)
    
    def inverse_transform(self, alphas, y=None):
        """
        Reconstructs input data based on their coefficients.
        
        :param alphas: Sparse coefficient matrix, eg, as returned from the `self.transform` method
        :param y: Unused in every case
        :return: Reconstructed matrix.
        """
        
        acc_hat = alphas.dot(self.dictionary.T)
        
        return np.asarray(acc_hat, dtype=np.float)
    
    def lasso_params(self, X):
        """
        Builds the parameters for the LASSO dictionary learning
        
        :param X: Input data. See `.fit` for more information.
        :return: Dictionary containing relevant parameters for LASSO optimisation
        """
        
        return dict(
            X=np.asfortranarray(X.T),
            D=np.asfortranarray(self.dictionary),
            lambda1=self.l1_reconstruct,
            numThreads=2,
            pos=self.positive_coefficients
        )
    
    def omp_params(self, X):
        """
        Builds a parameter dictionary for OMP dictionary learning
        
        :param X: Input data See `.fit` for more information.
        :return: Dictionary containing relevant parameters for OMP optimisation.
        """
        
        return dict(
            X=np.asfortranarray(X.T),
            D=np.asfortranarray(self.dictionary),
            lambda1=self.l1_reconstruct,
            numThreads=2,
            L=self.l0_max
        )
    
    def _transform_lasso(self, X, mask):
        """
        Performs LASSO transformation
        
        :param X: Input data. See `.fit` for more information.
        :param mask: Mask on input data. See `.fit` for more information.
        :return: Reconstruction parameters
        """
        
        if mask is None:
            return spams.lasso(**self.lasso_params(X)).T
        
        return spams.lassoMask(B=np.asfortranarray(mask.T), **self.lasso_params(X)).T
    
    def _transform_omp(self, X, mask):
        """
        Performs the OMP transformation.
        
        :param X: Input data. See `.fit` for more information.
        :param mask: Mask on input data. See `.fit` for more information.
        :return: Reconstruction parameters
        """
        
        if mask is None:
            return spams.omp(**self.omp_params(X)).T
        
        return spams.ompMask(
            X=np.asfortranarray(X.T),
            D=np.asfortranarray(self.dictionary),
            B=np.asfortranarray(mask.T),
            L=self.l0_max,
            lambda1=self.l1_reconstruct,
            numThreads=2,
        ).T
    
    @staticmethod
    def save(model, file_name):
        """
        Serialise model to file.
        
        :param model: Model
        :param file_name: Filename
        :return:
        """
        
        import cPickle as pickle
        import gzip
        
        with gzip.open(file_name, 'wb') as fil:
            pickle.dump(model, fil, protocol=0)
    
    @staticmethod
    def load(file_name):
        """
        Deserialise model from file
        
        :param file_name: Filename
        :return:
        """
        
        import cPickle as pickle
        import gzip
        
        with gzip.load(file_name, 'rb') as fil:
            return pickle.load(fil)


def main():
    import matplotlib.pyplot as pl
    import seaborn as sns
    
    sns.set_style('darkgrid')
    sns.set_context('poster')
    
    rng = np.random.RandomState(123)
    
    D = 21
    N = 500
    K = 100
    
    # Generate the data
    t = np.linspace(-np.pi * 2, np.pi * 2, D)
    y = np.asarray([rng.choice(2, 1) for _ in xrange(N)])
    x = np.asarray(
        [np.sin(t * (1.0 + y[n]) + rng.normal(0, 0.125, size=D)) for n in xrange(N)]
    )
    
    # Fit the model
    dl = SpamsTransformer(
        total_num_bases=K,
        l1_dictionary=1.2 / np.sqrt(D),
        l1_reconstruct=1.2 / np.sqrt(D),
        num_iterations=100
    )
    dl.fit(x)
    
    # Plot the data and the learnt bases
    fig, axes = pl.subplots(2, 1, sharex=True)
    
    axes[0].plot(t, x.T)
    axes[0].set_ylabel('Original data')
    
    axes[1].plot(t, dl.dictionary)
    axes[1].set_ylabel('Learnt dictionary')
    
    # Reconstruct the data and plot the first datapoint and its reconstruction
    alphas = dl.transform(x)  # Compute the reconstruction coefficients
    x_hat = dl.inverse_transform(alphas)  # Reconstruct the original data
    
    print type(alphas)
    
    fig, axes = pl.subplots(2, 1, sharex=False, sharey=False)
    abs_diff = np.abs(x - x_hat)
    axes[0].plot(t, x[0], label='Original data')
    axes[0].plot(t, x_hat[0], label='Reconstruction (MAE: {:.3f})'.format(
        np.mean(abs_diff)
    ))
    pl.legend()
    
    axes[1].hist(abs_diff.ravel(), bins=np.linspace(abs_diff.min(), abs_diff.max(), 31))
    
    print 'Average number of reconstruction coefficients: {}'.format(
        alphas.nnz / float(N)
    )
    
    pl.show()


if __name__ == '__main__':
    main()
