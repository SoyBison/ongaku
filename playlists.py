import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, cdist, squareform, euclidean
from time import time
from learning import load_tag_dict
import os

here = os.path.dirname(__file__)


def generate_m3u(tags, title, reference, locale='playlists\\'):
    """
    Takes in a corpus, and a list of tags, and generates a playlist file, which gets dumped in the locale.

    :param list tags: The songs you want to target.
    :param str title: the name of the created playlist file.
    :param dict reference: a dictionary that relates tags to locations
    :param str locale: the folder to dump playlists in.
    :return: NoneType
    """
    with open(f'{locale}{title}.m3u', 'w+', encoding='utf-8') as file:
        for tag in tags:
            file.write(reference[tag] + '\n')


def abs_dist_playlist(tag, manifold_df, length=5, metrics=False):
    """
    Takes in two song tags, and the manifold data frame, and creates a playlist of the input song, and the length
    nearest neighbors, in order of distance from the input song.

    :param str tag: The starting song
    :param pd.DataFrame manifold_df: manifold data frame
    :param int length: desired length of playlist
    :param bool metrics: Toggles printing the playlist to the console.
    :return:
    """
    dist_mat = pd.DataFrame(squareform(pdist(manifold_df.transpose())),
                            columns=manifold_df.transpose().index,
                            index=manifold_df.transpose().index)
    if metrics:
        print(dist_mat[tag].nsmallest(length))
    return list(dist_mat[tag].nsmallest(length).index)


def make_dist_playlist(tag, manifold_df, length=5, verbose=False, locale='playlists\\'):
    plist = abs_dist_playlist(tag, manifold_df, length=length)
    if verbose:
        print(*plist, sep='\n')
    generate_m3u(plist, f"{tag.split(' - ')[-1]}_circle{length}", locale=locale, reference=load_tag_dict())


def line_playlist(taga, tagb, manifold_df, line_res=100, metrics=False):
    st = time()
    a = manifold_df[taga].values
    b = manifold_df[tagb].values
    x = np.linspace(a, b, num=line_res)  # Uhh, just put in a big number
    space_made = time()
    d = pd.DataFrame(cdist(x, manifold_df.transpose()), columns=manifold_df.columns)
    distances_calcd = time()
    min_list = d.transpose().idxmin()
    mins = pd.unique(min_list)
    plist_found = time()
    if metrics:
        print(f'{space_made - st:.3} seconds to generate line.')
        print(f'{distances_calcd - space_made:.3} seconds to calculate distances from the line.')
        print(f'{plist_found - distances_calcd:.3} seconds to find a playlist.')
    return mins


def make_line_playlist(taga, tagb, manifold_df, verbose=True, line_res=100, locale='playlists\\'):
    plist = line_playlist(taga, tagb, manifold_df, line_res=line_res)
    if verbose:
        print(*plist, sep='\n')
    generate_m3u(plist, f"{taga.split(' - ')[-1]} to {tagb.split(' - ')[-1]}", locale=locale, reference=load_tag_dict())


def cone_plist(taga, tagb, manifold_df, line_res=100, min_len=15, metrics=False, resolution=1):
    a, b, x, d, ldist, perpdist = space_maker(line_res, manifold_df, taga, tagb)

    def make_list(_r=1):
        if metrics:
            print(f'Trying cone of size {_r}')
        cone_edge = np.pad(np.linspace(0, _r, num=line_res // 2),
                           (0, (line_res // 2)), 'symmetric')
        songlist = {}
        for i in range(len(cone_edge)):
            q_songs = list(ldist[perpdist == i][ldist <= cone_edge[i]].index)
            for s in q_songs:
                songlist[s] = 0
        plist = list(songlist.keys())
        if _r > euclidean(a, b):
            return plist
        elif len(plist) < min_len:
            return make_list(_r=_r + resolution)
        else:
            return plist

    return make_list()


def make_cone_plist(taga, tagb, manifold_df, verbose=True, line_res=100,
                    locale='playlists\\', min_len=15, resolution=1):
    plist = cone_plist(taga, tagb, manifold_df,
                       min_len=min_len, line_res=line_res, resolution=resolution)
    if verbose:
        print(*plist, sep='\n')
    generate_m3u(plist, f"{taga.split(' - ')[-1]} to {tagb.split(' - ')[-1]} Cone", locale=locale)


def cyl_plist(taga, tagb, manifold_df, line_res=100, min_len=15, metrics=False, resolution=1):
    a, b, x, d, ldist, perpdist = space_maker(line_res, manifold_df, taga, tagb)

    def make_list(_r=1):
        if metrics:
            print(f'Trying cylinder of size {_r}')

        songlist = {}
        for i in range(len(x)):
            q_songs = list(ldist[perpdist == i][ldist <= _r].index)
            for s in q_songs:
                songlist[s] = 0
                if s == tagb or s == taga:
                    break

        plist = list(songlist.keys())
        if _r > euclidean(a, b):
            return plist
        elif len(plist) < min_len:
            return make_list(_r=_r + resolution)
        else:
            return plist

    return make_list()


def space_maker(line_res, manifold_df, taga, tagb):
    if line_res % 2:
        raise ValueError('line_res must be even.')
    a = manifold_df[taga].values
    b = manifold_df[tagb].values
    x = np.linspace(a, b, num=line_res)
    d = pd.DataFrame(cdist(x, manifold_df.transpose()), columns=manifold_df.columns)
    ldist = d.min()
    perpdist = d.idxmin()
    return a, b, x, d, ldist, perpdist


def make_cyl_plist(taga, tagb, manifold_df, verbose=True, line_res=100,
                   locale='playlists\\', min_len=15, resolution=1):
    plist = cyl_plist(taga, tagb, manifold_df,
                      min_len=min_len, line_res=line_res, resolution=resolution)
    if verbose:
        print(*plist, sep='\n')
    generate_m3u(plist, f"{taga.split(' - ')[-1]} to {tagb.split(' - ')[-1]} Cylinder", locale=locale)


def icone_plist(taga, tagb, manifold_df, line_res=100, min_len=15, metrics=False, resolution=1):
    a, b, x, d, ldist, perpdist = space_maker(line_res, manifold_df, taga, tagb)

    def make_list(_r=1):
        if metrics:
            print(f'Trying cone of size {_r}')
        cone_edge = np.pad(np.linspace(_r, 0, num=line_res // 2),
                           (0, (line_res // 2)), 'symmetric')
        songlist = {}
        for i in range(len(cone_edge)):
            q_songs = list(ldist[perpdist == i][ldist <= cone_edge[i]].index)
            for s in q_songs:
                songlist[s] = 0
                if s == tagb:
                    break
        plist = list(songlist.keys())
        if _r > euclidean(a, b):
            return plist
        elif len(plist) < min_len:
            return make_list(_r=_r + resolution)
        else:
            return plist

    return make_list()


def make_icone_plist(taga, tagb, manifold_df, verbose=True, line_res=100,
                     locale='playlists\\', min_len=15, resolution=1):
    plist = icone_plist(taga, tagb, manifold_df,
                        min_len=min_len, line_res=line_res, resolution=resolution)
    if verbose:
        print(*plist, sep='\n')
    generate_m3u(plist, f"{taga.split(' - ')[-1]} to {tagb.split(' - ')[-1]} Inverse Cone", locale=locale)
