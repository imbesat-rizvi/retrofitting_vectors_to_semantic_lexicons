import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import normalize


def read_embeddings(filepath, norm=None, sep=None):
    r"""provide norm='l1' or 'l2' in case vectors are not already normalized"""
    embeddings = []
    w2i = dict()
    i2w = list()
    open_fn = open if not filepath.endswith('.gz') else gzip.open
    
    with open_fn(filepath, "r") as f:
        for i, line in enumerate(f):
            
            line = line.strip().split(sep)
            entity = line[0].lower()
            w2i[entity] = len(i2w)
            i2w.append(entity)
            embeddings.append([float(i) for i in line[1:]])
    
    embeddings = np.array(embeddings)
    if norm:
        embeddings = normalize(embeddings, axis=1, norm=norm)
    
    return w2i, i2w, embeddings


def write_embeddings(
    retrofitted_emb, 
    i2w, 
    filename="retrofitted_emb.txt",
    sep=" ",
    round_decimals=4
):
    emb_df = pd.DataFrame(retrofitted_emb.round(decimals=round_decimals))
    entities = pd.Series((i2w[i] for i in range(retrofitted_emb.shape[0])))
    emb_df = pd.concat((entities, emb_df), axis=1)
    emb_df.to_csv(filename, sep=sep, header=False, index=False)


def indexify_neighbours(w2i, neighbours):
    indexed_neighbours = dict()
    for entity, neighs in neighbours.items():
        if entity in w2i:
            indexed_neighbours[w2i[entity]] = [
                w2i[x] for x in neighs if x in w2i
            ]
            
    # filter entity/entity_idx with empty neighbours
    indexed_neighbours = {k:v for k,v in indexed_neighbours.items() if v}
    return indexed_neighbours


def read_neighbours(
    filepath, 
    w2i=None, 
    sep=None, 
    entity_filter=lambda x: x, # no filter by default
):
    
    neighbours = dict()
    
    with open(filepath, "r") as f:
        for line in f:
            entities = line.lower().strip().split(sep)
            if entity_filter(entities[0]):
                neighbours[entities[0]] = [w for w in entities[1:] if entity_filter(w)]
                    
    if w2i:
        neighbours = indexify_neighbours(w2i, neighbours)
    else:
        # filter entity/entity_idx with empty neighbours
        neighbours = {k:v for k,v in neighbours.items() if v}
        
    return neighbours


def retrofit(embeddings, neighbours, n_iter=10):

    # append extra row with all 0s as unk token
    unk_emb = np.zeros((embeddings.shape[1],))
    embeddings = np.vstack((embeddings, unk_emb))
    new_embeddings = np.array(embeddings)
    
    # separate out indices to update and the corresponding
    # neighbour indices
    update_idcs, neigh_idcs = (
        np.array(i, dtype=object) for i in zip(*neighbours.items())
    )
    update_idcs = update_idcs.astype(np.int32)

    # since embedding is a 2D matrix with row indices representing
    # entity indices, check and keep idcs to update <= last entity index
    valid_idcs = update_idcs < len(embeddings)-1
    update_idcs = update_idcs[valid_idcs]

    # pad with idx -1 i.e. last all zeros unk_emb to create 
    # a 2D matrix of neighbouring indices. -1 idx with zero emb
    # values don't effect calculations
    max_len = max(map(len, neigh_idcs))
    neigh_idcs = np.array([arr+[-1]*(max_len-len(arr)) 
                           for arr in neigh_idcs])

    # In addition to checking valid idcs <= last emb for update_idcs, 
    # check and replace invalid idcs > last emb idx in the neighbouring
    # indices too with -1
    neigh_idcs = np.where(neigh_idcs >= len(embeddings)-1, 
                          -1, neigh_idcs)
    
    # filter out any idx to update if it doesn't have a neighbouring
    # index != -1
    valid_idcs = np.any(neigh_idcs != -1, axis=1)
    update_idcs = update_idcs[valid_idcs]
    neigh_idcs = neigh_idcs[valid_idcs] 
    
    neigh_counts = ((neigh_idcs > -1).sum(axis=1)).reshape((-1,1))
    
    for i in range(n_iter):
        # update rule with \alpha_i = 1 and \beta_{ij} = degree(i)^{-1}
        # as given in the source paper
        new_emb = (neigh_counts*embeddings[update_idcs] + 
                   new_embeddings[neigh_idcs].sum(axis=1))
        new_emb /= (2*neigh_counts)

        new_embeddings[update_idcs,:] = new_emb
    
    # remove last appended rows with 0s as unk weights
    return new_embeddings[:-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        help="input embeddings filepath",
    )

    parser.add_argument(
        "-a",
        "--neighbours",
        help="filepath with neighbours info e.g. lexicon file for word embeddings",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="retrofitted_emb.txt",
        help="output filepath for retrofitted embeddings",
    )

    parser.add_argument(
        "-n",
        "--n_iter",
        type=int,
        default=10,
        help="number of update iterations",
    )

    args = parser.parse_args()

    w2i, i2w, embeddings = read_embeddings(args.input, norm=None)
    neighbours = read_neighbours(
        args.neighbours, w2i=w2i, entity_filter=lambda x: x.isalpha()
    )

    retrofitted_emb = retrofit(embeddings, neighbours, n_iter=args.n_iter)

    write_embeddings(retrofitted_emb, i2w, filename=args.output)
