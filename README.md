# Ongaku

## Overview

Ongaku is a method for creating playlists programmatically, using the
content of the song alone. It uses gammatone cepstral analysis to create
unique matrices to represent each song. A gammatone cepstrum is similar
to the more common spectra used for audio analysis. Instead of doing a
Fourier transform, we do a reverse Fourier transform, and then apply a
transformation according to the gammatone function. This function was
designed to mimic the signals sent to the brain through the cochlear
nerve, the nerve which connects the ear to the brain.

The gammatone cepstra are then compiled together into a corpus, which is used for manifold learning (using sklearn). This creates a metric space for the songs in the library. The manifold learning process needs to be tuned to idealize the playlist outputs, but this is something which is difficult to define mathematically. I've had good results with `n_components = 45` but your mileage may vary. I've defined a few rudimentary metrics which can be optimized over as well. We can draw shapes in this metric space, to define playlists.

The code will be documented in full at
[readthedocs](https://ongaku.readthedocs.io/en/latest/)

## Analysis

The `analysis` module is full of pre-processing methods to turn a song or song library into a gammatone cepstrum or gammatone corpus. It does also have tools for constructing a Fourier spectrum corpus, but the main usage is intended for gammatone cepstrum corpora. Corpora production has been parallelized in the `preprocess` function, but the default is to do this with a `pool_size = 2` due to the fact that each gammatone analysis takes about 5GB of RAM to complete. In the event that I get around to building a gammatone function that isn't a MATLAB port, this may change, but for now, only increase `pool_size` if you know your machine can handle it. `preprocess`, as the main workhorse, will put a `corpus.pkl` in your working directory, which will be needed for the learning and construction stages.

## Learning

The `learning` module implements a couple of manifold learning techniques, and is fully compatible with `sklearn`, so it should interact well with any other pipelines. The gammatone cepstra can be compiled into a corpus, and then used for manifold learning using the `cropped_corpus` function, and then the `flatten_corpus` function, whose output is safe to use for generalized `sklearn` operations. The `learning`  module also contains the `generate_m3u` function, which takes a list of tags and generates a `.m3u` file which represents the location of the specified songs on your computer, being a playlist which is compatible with all major music players.

## Metrics

The `metrics` module contains a number of naïve metrics to measure the "compactness" and the "disjointness" of the resultant manifold. They, while useful to compare different manifold generation methods to each other, don't give a great idea as to how to tune metaparameters, since they're really just measures of the degree to which the curse of dimensionality is affecting your data.

## Playlists

The `playlists` module contains some algorithms for generating playlists from input songs and the manifold data frame. They mostly involve drawing geometric shapes on the manifold and sorting the songs within those n-volumes by distance from some object. For example, the distance playlist draws an n-circle around the input song, with radius equal to the distance between the input song and the m-th closest song.
