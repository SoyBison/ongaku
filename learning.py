import numpy as np
import pickle
import os
from sklearn import manifold as mnfd
from sklearn import decomposition as dcomp
from sklearn import preprocessing as pre
import pandas as pd
from analysis import TEST_REGEX, library_from_regex
import winsound
from sklearn.pipeline import Pipeline
import re
from analysis import corpus_tag_generator

TEST_REGEX = re.compile(TEST_REGEX.pattern)

musicbee = 'C:\\Users\\Coen D. Needell\\Music\\MusicBee\\Playlists\\'  # Personal playlist location


def load_corpus(loc='cepstra\\', precompiled=False):
    """
    Generates a corpus for machine learning from your preprocessed cepstra. Location should be the same folder you used
    for the analysis.py run. Returns a dict with keys being the 'song code' as made by the analysis.corpus_tag_generator
    function.
    :param loc: str directory where the spectra are.
    :param precompiled: bool triggers whether or not it should load the corpus from a pickle file or the cepstra folder
    :return: dict
    """
    if precompiled:
        with open('corpus.pkl', 'rb') as file:
            return pickle.load(file)
    corpus = {}
    for song in os.listdir(loc):
        with open(f'cepstra\\{song}', 'rb') as file:
            corpus[song.replace('.pkl', '')] = pickle.load(file)
    corpus = {title: song for title, song in corpus.items() if song is not None}
    with open('corpus.pkl', 'wb') as file:
        pickle.dump(corpus, file)
    return corpus


def create_tag_dict(lib, loc='locations.pkl'):
    """
    Makes a dictionary that relates the tags to associated filename.
    :param lib: list contains all of the filenames.
    :param loc: str file to dump the tag dictionary in if you want to avoid doing this more than once.
    :return:
    """
    mdata_dict = {}
    for song in lib:
        mdata_dict[corpus_tag_generator(song)] = song
    with open(loc, 'wb') as file:
        pickle.dump(mdata_dict, file)
    return mdata_dict


def load_tag_dict(loc='locations.pkl'):
    with open(loc, 'rb') as file:
        mdata_dict = pickle.load(file)
    return mdata_dict


def generate_m3u(tags, title, reference=load_tag_dict(), locale='playlists\\'):
    with open(f'{locale}{title}.m3u', 'w+', encoding='utf-8') as file:
        for tag in tags:
            file.write(reference[tag] + '\n')


def padded_corpus(corp):
    lens = [song.shape[1] for _, song in corp.items()]
    longest = np.max(lens)

    new_corp = {}
    for title, song in corp.items():
        song_size = longest - song.shape[1]
        if song_size == 0:
            new_corp[title] = song
        else:
            new_corp[title] = np.pad(song, ((0, 0), (0, song_size)), constant_values=np.log(0))
    return new_corp


def flattened_corpus(corp):
    new_corp = {}
    for title, song in corp.items():
        new_corp[title] = np.nan_to_num(song.flatten())
    return new_corp


def cropped_corpus(corp, tar_len=90, pad_shorts=False):
    """
    tar_len must be EVEN
    """
    new_corp = {}
    for title, song in corp.items():
        s_len = song.shape[1]
        if s_len > tar_len:
            st = (s_len // 2) - (tar_len // 2)
            end = (s_len // 2) + (tar_len // 2)
            assert end - st == tar_len
            new_corp[title] = song[:,st:end]
        else:
            if pad_shorts:
                new_corp[title] = np.pad(song, ((0, 0), (0, tar_len - s_len)), constant_values=np.log(0))

    return new_corp


def make_manifold(processed_corp,
                  pipeline=Pipeline([('reduce_dims', dcomp.PCA()), ('embedding', mnfd.Isomap(n_components=45))])):
    flat_corp = flattened_corpus(processed_corp)
    songs = list(flat_corp.values())
    songs_scaled = np.nan_to_num(pre.RobustScaler().fit_transform(songs))
    songs_scaled = np.clip(songs_scaled, -1000, 5)

    songs_transformed = pipeline.fit_transform(songs_scaled)
    manifold = {}
    for title, song in zip(flat_corp, songs_transformed):
        manifold[title] = song
    manifold_df = pd.DataFrame(manifold)
    winsound.MessageBeep(winsound.MB_ICONHAND)
    return manifold_df


if __name__ == '__main__':
    libr = library_from_regex(re.compile(''))
    cor = load_corpus()
    nc = cropped_corpus(cor, tar_len=120, pad_shorts=True)
    mandf = make_manifold(nc)
    with open('manifold.pkl') as f:
        pickle.dump(mandf, f)
