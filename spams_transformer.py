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
    def __init__(self, total_num_bases, l1_dictionary, l1_reconstruct, class_conditional=False,
                 positive_coefficients=True, l0_max=0, verbose=0, num_iterations=100,
                 minibatch_size=512, use_lasso=True):
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
    
    def transform(self, X, mask=None):
        if self.use_lasso:
            return self._transform_lasso(X, mask)
        
        return self._transform_omp(X, mask)
    
    def inverse_transform(self, alphas, y=None):
        acc_hat = alphas.dot(self.dictionary.T)
        
        return np.asarray(acc_hat, dtype=np.float)
    
    def lasso_params(self, X):
        return dict(
            X=np.asfortranarray(X.T),
            D=np.asfortranarray(self.dictionary),
            lambda1=self.l1_reconstruct,
            numThreads=2,
            pos=self.positive_coefficients
        )
    
    def omp_params(self, X):
        return dict(
            X=np.asfortranarray(X.T),
            D=np.asfortranarray(self.dictionary),
            lambda1=self.l1_reconstruct,
            numThreads=2,
            L=self.l0_max
        )
    
    def _transform_lasso(self, X, mask):
        if mask is None:
            return spams.lasso(**self.lasso_params(X)).T
        
        return spams.lassoMask(B=np.asfortranarray(mask.T), **self.lasso_params(X)).T
    
    def _transform_omp(self, X, mask):
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
        import cPickle as pickle
        import gzip
        
        with gzip.open(file_name, 'wb') as fil:
            pickle.dump(model, fil)
    
    @staticmethod
    def load(file_name):
        import cPickle as pickle
        import gzip
        
        with gzip.load(file_name, 'rb') as fil:
            return pickle.load(fil)


if __name__ == '__main__':
    def main():
        import matplotlib.pyplot as pl
        import seaborn as sns
        
        sns.set_style('darkgrid')
        sns.set_context('poster')
        
        rng = np.random.RandomState(123)
        
        D = 51
        N = 1000
        
        t = np.linspace(-np.pi * 2, np.pi * 2, D)
        x = np.asarray(
            [np.sin(t * (1 + rng.choice(2, 1))) + rng.normal(0, 0.125, size=D) for _ in xrange(N)]
        )
        
        dl = SpamsTransformer(
            total_num_bases=10,
            l1_dictionary=1.0 / np.sqrt(D),
            l1_reconstruct=1.0 / np.sqrt(D),
        )
        
        dl.fit(x)
        
        fig, axes = pl.subplots(2, 1, sharex=True)
        
        axes[0].plot(x.T)
        axes[1].plot(dl.dictionary)
        
        pl.show()
    
    
    main()
