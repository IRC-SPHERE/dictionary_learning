
# Dictionary Learning

This is a loose wrapper of dictionary learning around the [SPArse Modeling Software](http://spams-devel.gforge.inria.fr/) (SPAMS) library.

I have found that SPAMS has been tricky to install in the past. The simplest and most consistent way of installing it for me has been to rely on [anaconda python](https://www.continuum.io/downloads). The following three commands create a virtual environment called `dictionary_learning` and installs `python-spams` to it:

    conda create -n dictionary_learning python=2.7 pip
    conda config --add channels conda-forge
    conda install python-spams

The `python-spams` library is found in the `conda-forge` repo that can be found [here](https://conda-forge.github.io/).

Finally the following two lines execute th `spams_transformer.py` script as expected:

    source activate dictionary_learning
    python spams_transformer.py

to disable the virtual environment type the following into the command line:

    source deactivate dictionary_learning

# Sample usage

The following code generates a simple dataset of a slow and fast sine wave. The SPAMS transformer then learns dictionary bases from this dataset. In this case the resulting bases consist of fast and slow sine waves, as expected.

    import matplotlib.pyplot as pl
    import seaborn as sns

    from spams_transformer import SpamsTransformer

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