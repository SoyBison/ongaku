import multiprocessing as mp
import os
import pickle
import re

import gammatone.gtgram as gt
import matplotlib.pyplot as plt
import mutagen
import numpy as np
import soundfile as sf
import tqdm
from scipy import signal

TEST_REGEX = re.compile('Toby Fox|Darren|CHV|STRFKR|Starfucker|Presidents|Passion|Panic|VARIOUS|Imagine|Glass|Death Cab'
                        '|Foo|Emanc|Avi|Coldplay|AWOL|Orchest|WALK|Walk|Juke|'
                        'kingur|Group|Vulf|Finish|Beautiful|Counting'
                        '|Letters|Watsky|Gin|Gemini|Future|Fountains|Corinne|Caravan|Owl|Charming|Vampire|Hideki|Elena'
                        '|Plini|Long Winters|Cult|Chicago|2|MG|Odesza|Hozier|Daft|Cage'
                        '|Pity Sex|Macklemore|Poppy|Paul|Say|'
                        'Roots|Gang|Tycho|Glitch|Toad|fujitsu|sleepy fish|Moods'
                        '|The Worst|Reich|Enya|blink|Toto|Zhu|wün|'
                        'Daniel Hope|ABBA|Portugal|Zedd|Choice|Lorne|Work|Love|VA|Arctic|Chaos|Passenger|Madeon|Nico'
                        '|Panama|Vík|Dobrinka|John|Capital|Yelle|Looking|Cuphead|Chillhop|Bobby|That|Tame|Sublime|'
                        'Strong|Solar|Sleep|Steam|Sam|Seiji|Purity|Kelly|R.E.M.|Re|Monsters|Oasis|Nirvana|Nils|Monster'
                        '|Marina|Maybe|Alice|watsky|Kid Cudi|Jungle|Mahler|Various|IRON|Iglu|Joey|Foxes|First|Eyes|'
                        'Roosevelt|dead|Cro|Clean|Childish|Cinedelic|Pearl|Beck|Butthole|Red Hot|The Chainsmokers')


def make_spect(filepath, method='fourier', height=60, interval=1, verbose=False, max_len=1080):
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
    tag = corpus_tag_generator(song_loc)
    filename = f'{locale}{tag}.pkl'
    filename = re.sub('[?*:"<>/|]', "", filename)
    if not os.path.exists(filename):
        cepstrum = make_spect(song_loc, method='gamma', height=16)
        if cepstrum is None:
            return False
        with open(filename, 'wb+') as f:
            pickle.dump(cepstrum, f)
            return tag


def library_from_regex(target_regex, library_locale='D:\\What.cd\\'):
    """
    Takes in a regex, and a pointer to your music library and compiles a list of song locations from it.


    :param target_regex: re.Pattern

    :param library_locale: str

    :return: list
    """

    targets = []
    lib = []
    r = target_regex

    targets += list(filter(r.match, os.listdir(library_locale)))
    for song in targets:
        library_addition(lib, song, locale=library_locale)
    return lib


def preprocess(target_regex, library_locale='D:\\What.cd\\', pool_size=2):
    """
    This runs ```gt_and_store()``` on every file which is in a folder that matches with target_regex. Some notes about
    running this on a personal computer. If you have more than 16 GB of ram, you should be fine. If you have 16 or less,
    Be prepared for the spin-up to lag your computer. It should stabilize after a while once the processes
    get out of sync. Also creates a dictionary that relates corpus tags to their file location.


    :param target_regex: re.compile a regex of the things you want. Might be long and full of pipes.


    :param library_locale: str the location of your music library.


    :return: a list of successes and failures for if something went wrong with a song.
    """

    lib = library_from_regex(target_regex, library_locale=library_locale)
    p = mp.Pool(pool_size, maxtasksperchild=1000)
    if not os.path.exists('cepstra'):
        os.mkdir('cepstra')
    tags = list(tqdm.tqdm(p.imap(gt_and_store, lib), total=len(lib)))
    create_location_dictionary(lib, tags)


def corpus_tag_generator(song_loc):
    """
    This looks at a file's metadata and turns it into a corpus tag for later usage.


    :param song_loc: str the file location

    :return: str the tag as used by the learning parts of the system.
    """
    mdata = mutagen.File(song_loc)
    try:
        album = mdata['album'][0]
    except KeyError:
        album = "Unknown Album"
    try:
        albumartist = mdata['albumartist'][0]
    except KeyError:
        albumartist = mdata['artist'][0]
    try:
        name = re.sub('[\\\\/]', '', mdata['title'][0])
    except KeyError:
        name = 'Unknown Track'

    filename = f'{albumartist} - {album} - {name}'
    filename = re.sub('[?*:"<>/|]', "", filename)
    return filename


def create_location_dictionary(lib, tags=None):
    """
    Takes everything in the library and adds it to a location dictionary stored in locations.pkl.


    :param lib: list of song locations

    :param tags: list or NoneType if you already have the tag, then you don't need to generate it again.


    :return: NoneType
    """
    if os.path.exists('../locations.pkl') and os.path.getsize('../locations.pkl') > 0:
        with open('../locations.pkl', 'rb') as file:
            mdata_dict = pickle.load(file)
        os.remove('../locations.pkl')
    else:
        mdata_dict = {}

    if tags is None:
        tags = [None] * len(lib)

    for song, tag in zip(lib, tags):
        if tag:
            mdata_dict[tag] = song
        else:
            mdata_dict[corpus_tag_generator(song)] = song

    with open('../locations.pkl', 'wb') as file:
        pickle.dump(mdata_dict, file)


if __name__ == '__main__':
    reg = re.compile('')
    preprocess(reg)
