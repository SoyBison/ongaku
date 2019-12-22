import soundfile as sf
import os
import matplotlib.pyplot as plt
import numpy as np
import gammatone.gtgram as gt
from scipy import signal
import re
from time import time
import multiprocessing as mp
import pickle
import tqdm
import mutagen


def make_spect(filepath, method='fourier', height=60, interval=1, verbose=False, max_len=1800):
    """
    Turns a file containing sound data into a matrix for processing. Two methods are supported,
    fourier spectrum analysis, which returns a spectrogram, and gammatone which returns a gammatone quefrency cepstrum.
    Gammatones take `much` longer, but are ostensibly better for feature analysis, and are smaller. Spectrograms are big
    but don't take very much time to create.
    :param filepath: str path to file
    :param method: str 'fourier' or 'gamma'
    :param max_len: int or float the maximum length in seconds of a song to convert. Important for memory management.
    Default is a half-hour
    :param height: int for gammatones, how many quefrency bins should be used. default 60.
    :param interval: int for gammatones, the width in seconds of the time bins. default 2.
    :param verbose: bool toggles behavior showing a plot of the returned 'gram.
    :return: np.array a matrix representing (in decibels) the completed analysis.
    """
    try:
        data, sr = sf.read(filepath)
    except RuntimeError:
        return None

    if len(data) // sr > max_len:
        return None

    if verbose:
        plt.figure()

    if method == 'fourier':
        f, t, sxx = signal.spectrogram(data[:, 0], sr)
        del data

        if verbose:
            plt.pcolormesh(t, f, 10 * np.log10(sxx))
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.show()
    elif method == 'gamma':
        sxx = gt.gtgram(data[:, 0], sr, interval, interval, height, 20)
        del data
        if verbose:
            plt.pcolormesh(10 * np.log10(sxx))
            plt.show()
    else:
        raise ValueError(f'{method} is not a valid method.')
    with np.testing.suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        # This is because log10 will throw a warning when it coerces a 0 to Nan and I find that obnoxious.
        return 10 * np.log10(sxx)


def song_name_gen(fname: str):
    """
    A utility function which takes a file name and strips out all the stuff that's probably not the song title.
    :param fname: str filename to work on
    :return: str a cleaned up title.
    """
    fname = fname.rsplit('\\', 1)[-1]
    x = re.sub('^[^a-zA-Z]*', '', fname)
    x = re.sub('.\\w*$', '', x)
    return x


def library_addition(library, target, locale='D:\\What.cd\\', filetype='.flac'):
    """
    A utility for recursively adding files of type filetype to a list. Stores them as a list of their full location.
    :param library: list the library you're adding to.
    Note that it edits library in-place so don't drop an unnamed object in there.
    :param target: str the top of the tree. do '' if you want this to be your locale.
    :param locale: str the location for the system to start from. Should be the location of your target.
    Default is my music library's location.
    :param filetype: str The file extension you want this to work on.
    :return: NoneType
    """
    try:
        nodes = os.listdir(locale + target)
    except NotADirectoryError:
        return None
    for file in nodes:
        if file.endswith(filetype):
            library.append(locale + target + '\\' + file)
        else:
            library_addition(library, file, locale=(locale + target + '\\'))


def gt_and_store(song_loc, locale='cepstra\\'):
    """
    This calculates the gammatone cepstrum, pickles it, and drops it in a designated folder. Default is a folder called
    cepstra. This acts like a worker function, so it doesn't return anything.
    :param song_loc: str filepath
    :param locale: str folder to drop cepstra in
    :return: NoneType
    """
    mdata = mutagen.File(song_loc)
    album = mdata['album'][0]
    try:
        albumartist = mdata['albumartist'][0]
    except KeyError:
        albumartist = mdata['artist'][0]
    name = re.sub('[\\\\/]', '', mdata['title'][0])
    filename = f'{locale}{albumartist} - {album} - {name}.pkl'
    filename = re.sub('[?*:"<>/|]', "", filename)
    if not os.path.exists(filename):
        cepstrum = make_spect(song_loc, method='gamma', height=16)

        with open(filename, 'wb+') as f:
            pickle.dump(cepstrum, f)


def preprocess(target_regex, library_locale='D:\\What.cd\\'):
    """
    This runs ```gt_and_store()``` on every file which is in a folder that matches with target_regex.
    :param target_regex:
    :param library_locale:
    :return:
    """

    targets = []
    lib = []
    r = target_regex

    targets += list(filter(r.match, os.listdir(library_locale)))
    for song in targets:
        library_addition(lib, song, locale=library_locale)

    p = mp.Pool(4, maxtasksperchild=100)
    if not os.path.exists('cepstra'):
        os.mkdir('cepstra')
    ceps = list(tqdm.tqdm(p.imap(gt_and_store, lib), total=len(lib)))


if __name__ == '__main__':
    reg = re.compile('Toby Fox|Darren|CHV|STRFKR|Starfucker|Presidents|Passion|Panic|VARIOUS|Imagine|Glass|Death Cab'
                     '|Foo|Emanc|Avi|Coldplay|AWOL|Orchest|WALK|Walk|Juke|kingur|Group|Vulf|Finish|Beautiful|Counting')
    preprocess(reg)
