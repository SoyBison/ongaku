---
author: "Coen D. Needell"
title: "Ongaku Readme"
date: 2020-01-20
---

# Ongaku

## Overview

Ongaku is a method for creating playlists programmatically, using only the content of the song alone. It uses gammatone cepstral analysis to create unique matrices to represent each song. A gammatone cepstrum is similar to the more common spectra used for audio analysis. Instead of doing a Fourier transform, we do a reverse Fourier transform, and then apply a transformation according to the gammatone function. This function was designed to mimic the signals sent to the brain through the cochlear nerve, the nerve which connects the ear to the brain.  

The gammatone cepstra are then compiled together into a corpus, which is used for manifold learning (using sklearn). This creates a metric space for the songs in the library. The manifold learning process needs to be tuned to idealize the playlist outputs, but this is something which is difficult to define mathematically. I've had good results with `n_components = 45` but your mileage may vary. I've defined a few rudimentary metrics which can be optimized over as well. We can draw shapes in this metric space, to define playlists.

The code will be documented in full at https://readthedocs.org/projects/ongaku/.
