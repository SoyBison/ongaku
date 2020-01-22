import numpy as np
import pandas as pd
from source.learning import load_corpus
from scipy.spatial.distance import pdist


def corpus_xdsd(mdf):
    """
    I call this the xd_sd, it's a cross-dimensional standard deviation, it's essentially the average of the standard
    deviations in each dimension.

    :param mdf:
    :return:
    """
    xd_sd = np.nanmean(mdf.std(axis=0))
    return xd_sd


def corpus_cohesion(mdf):
    cohesion = np.nanmean(pdist(mdf)) // np.log(mdf.shape[0])
    return cohesion


def artist_metric(mdf):
    new_df = group_artists(mdf)
    dist_df = {}
    for artist in new_df:
        dist = pdist(new_df[artist])
        dist_df[artist] = np.nanmean(np.log(dist)) / np.log(new_df[artist].shape[1])
    return dist_df


def group_artists(mdf):
    names = np.unique([code.split(' - ')[0] for code in mdf])
    new_df = {}
    for artist in names:
        songs = []
        for code in mdf:
            if code.split(' - ')[0] == artist:
                songs.append(mdf[code].values)
        new_df[artist] = np.array(songs)
    return new_df


def album_metric(mdf):
    new_df = group_albums(mdf)
    dist_df = {}
    for album in new_df:
        dist = pdist(new_df[album])
        dist_df[album] = np.nanmean(np.log(dist)) / np.log(new_df[album].shape[1])
    return dist_df


def avg_album_metric(mdf):
    albmet = album_metric(mdf)
    return np.nanmean(list(albmet.values()))


def avg_artist_metric(mdf):
    artmet = artist_metric(mdf)
    return np.nanmean(list(artmet.values()))


def album_xdsd(mdf):
    new_df = group_albums(mdf)
    sd_df = {}
    for album in new_df:
        xd_sd = np.nanmean(new_df[album].std(axis=0))
        sd_df[album] = xd_sd
    return sd_df


def group_albums(mdf):
    names = np.unique([code.split(' - ')[1] for code in mdf])
    new_df = {}
    for album in names:
        songs = []
        for code in mdf:
            if code.split(' - ')[1] == album:
                songs.append(mdf[code].values)
        new_df[album] = np.array(songs)
    return new_df


def artist_xdsd(mdf):
    new_df = group_artists(mdf)
    sd_df = {}
    for artist in new_df:
        xd_sd = np.nanmean(new_df[artist].std(axis=0))
        sd_df[artist] = xd_sd
    return sd_df


def avg_album_xdsd(mdf):
    return np.nanmean(list(album_xdsd(mdf).values()))


def avg_artist_xdsd(mdf):
    return np.nanmean(list(album_xdsd(mdf).values()))


def artist_cohesion_score(xformd_songlist, corp):
    manifold = {}
    for title, song in zip(corp, xformd_songlist):
        manifold[title] = song
    manifold_df = pd.DataFrame(manifold)
    return avg_artist_metric(manifold_df)


def album_cohesion_score(xformd_songlist, corp):
    manifold = {}
    for title, song in zip(corp, xformd_songlist):
        manifold[title] = song
    manifold_df = pd.DataFrame(manifold)
    return avg_album_metric(manifold_df)


def album_scorer(estimator, x, y=None):
    if y:
        x = np.row_stack([x, y])
    corpus = load_corpus(precompiled=True)
    x_formd = estimator.transform(x)
    return album_cohesion_score(x_formd, corpus)


def artist_scorer(estimator, x, y=None):
    if y:
        x = np.row_stack([x, y])
    corpus = load_corpus(precompiled=True)
    x_formd = estimator.transform(x)
    return artist_cohesion_score(x_formd, corpus)


def album_xdsd_score(xformd_songlist, corp):
    manifold = {}
    for title, song in zip(corp, xformd_songlist):
        manifold[title] = song
    manifold_df = pd.DataFrame(manifold)
    return avg_album_xdsd(manifold_df)


def artist_xdsd_score(xformd_songlist, corp):
    manifold = {}
    for title, song in zip(corp, xformd_songlist):
        manifold[title] = song
    manifold_df = pd.DataFrame(manifold)
    return avg_artist_xdsd(manifold_df)


def corpus_xdsd_score(xformd_songlist, corp):
    manifold = {}
    for title, song in zip(corp, xformd_songlist):
        manifold[title] = song
    manifold_df = pd.DataFrame(manifold)
    return corpus_xdsd(manifold_df)


def album_xdsd_scorer(estimator, x, y=None):
    if y:
        x = np.row_stack([x, y])
    corpus = load_corpus(precompiled=True)
    x_formd = estimator.transform(x)
    return album_xdsd_score(x_formd, corpus)


def artist_xdsd_scorer(estimator, x, y=None):
    if y:
        x = np.row_stack([x, y])
    corpus = load_corpus(precompiled=True)
    x_formd = estimator.transform(x)
    return artist_xdsd_score(x_formd, corpus)


def corpus_xdsd_scorer(estimator, x, y=None):
    if y:
        x = np.row_stack([x, y])
    corpus = load_corpus(precompiled=True)
    x_formd = estimator.transform(x)
    return corpus_xdsd_score(x_formd, corpus)
