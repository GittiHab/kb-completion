# On Predicting Missing Links in Knowledge Graphs

This repository includes implementations of my bachelor thesis.

## Description


## Usage
To commit large files this repository uses [Git LFS](https://git-lfs.github.com/).
Please make sure you have it [installed](https://help.github.com/articles/installing-git-large-file-storage/) and setup,
if you would like to reproduce the results.

You might need to install required python packages if they are not already installed on you machine. What you will
need for sure is `numpy`. Because it's 2018, everything runs with Python 3.

The thesis reports on the results in the `threshold-optimization` notebook.
To run this, you first have to unpack the predictions of the models which were computed using the source provided by the
authors.
To do this head into the `data` directory and for the datasets `fb15k` and `WN`, which are used in the notebook, run
`tar -xvzf predictions.tar.gz`.
Done! Now you're good to go.

## Citation
If you use any of the code please cite the thesis.

```

```