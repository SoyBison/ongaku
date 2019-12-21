import soundfile as sf
import os
import scipy
import matplotlib.pyplot as plt
import numpy as np
import gammatone.gtgram as gt
from scipy import signal
import re
import progressbar
from time import time
from functools import partial
import multiprocessing as mp
import pickle
import tqdm
import mutagen


def make_spect(filepath, method='fourier', height=60, interval=1, verbose=False):
    """
    Turns a file containing sound data into a matrix for processing. Two methods are supported,
    fourier spectrum analysis, which returns a spectrogram, and gammatone which returns a gammatone quefrency cepstrum.
    Gammatones take `much` longer, but are ostensibly better for feature analysis, and are smaller. Spectrograms are big
    but don't take very much time to create.
    :param filepath: str path to file
    :param method: str 'fourier' or 'gamma'
    :param height: int for gammatones, how many quefrency bins should be used. default 60.
    :param interval: int for gammatones, the width in seconds of the time bins. default 2.
    :param verbose: bool toggles behavior showing a plot of the returned 'gram.
    :return: np.array a matrix representing (in decibels) the completed analysis.
    """
    data, sr = sf.read(filepath)
    if verbose:
        plt.figure()

    if method == 'fourier':
        f, t, sxx = signal.spectrogram(data[:, 0], sr)

        if verbose:
            plt.pcolormesh(t, f, 10 * np.log10(sxx))
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.show()
    if method == 'gamma':
        sxx = gt.gtgram(data[:, 0], sr, interval, interval, height, 20)
        if verbose:
            plt.pcolormesh(10 * np.log10(sxx))
            plt.show()
    else:
        raise ValueError(f'{method} is not a valid method.')
    return 10 * np.log10(sxx)


class Song:

    def __init__(self, sr, data, name):
        self.sr = sr
        self.data = data
        self._name = name

    @property
    def name(self):
        x = re.sub('^[^a-zA-Z]*', '', self._name)
        x = re.sub('.\\w*$', '', x)
        return x


def song_name_gen(fname: str):
    fname = fname.rsplit('\\', 1)[-1]
    x = re.sub('^[^a-zA-Z]*', '', fname)
    x = re.sub('.\\w*$', '', x)
    return x


def library_addition(library, target, locale='D:\\What.cd\\', read=False):
    try:
        nodes = os.listdir(locale + target)
    except NotADirectoryError:
        return None
    for file in nodes:
        if file.endswith('.flac'):
            if read:
                data, sr = sf.read(locale + target + '\\' + file)
                library.append(Song(sr, data, file))
            else:
                library.append(locale + target + '\\' + file)
        else:
            library_addition(library, file, locale=(locale + target + '\\'))


def gt_and_store(song_loc, locale='cepstra\\'):
    mdata = mutagen.File(song_loc)
    album = mdata['album'][0]
    try:
        albumartist = mdata['albumartist'][0]
    except KeyError:
        albumartist = mdata['artist'][0]
    name = mdata['title'][0].replace('\\', '').replace('/', '')
    filename = f'{locale}{albumartist} - {album} - {name}.pkl'
    if not os.path.exists(filename):
        cepstrum = make_spect(song_loc, method='gamma', height=32)
        with open(filename, 'wb+') as f:
            pickle.dump(cepstrum, f)


if __name__ == '__main__':
    targets = ['Capital Cities', 'Chillhop Music', 'Looking Glass',
               'Ludovico Einaudi', 'Yelle',
               'Watsky',
               'Kristofer Maddigan - Cuphead [Original WEB Release] (2017) - FLAC',
               'CHVRCHΞS', 'John Mayer']

    lib = []
    for song in targets:
        library_addition(lib, song)

    p = mp.Pool(4)
    if not os.path.exists('cepstra'):
        os.mkdir('cepstra')
    strt = time()
    ceps = list(tqdm.tqdm(p.imap(gt_and_store, lib), total=len(lib)))
    print(time() - strt)
