import pyarrow.parquet as pq
import faiss
import numpy as np
import json
from pathlib import Path

from common.constants import *


def read_embeddings(path):
    emb_df = pq.read_table(path).to_pandas()

    id_to_name = {k: v.decode("utf-8") for k, v in enumerate(list(emb_df[IMAGE_NAME_KEY]))}
    name_to_id = {v: k for k, v in id_to_name.items()}

    id_to_sub_dir = {k: v.decode("utf-8") for k, v in enumerate(list(emb_df[IMAGE_SUB_DIR_KEY]))}
    sub_dir_to_id = {v: k for k, v in id_to_sub_dir.items()}

    embgood = np.stack(emb_df["embedding"].to_numpy())

    return id_to_name, name_to_id, id_to_sub_dir, sub_dir_to_id, embgood


def embeddings_to_numpy(input_path, output_path):
    embedding = pq.read_table(input_path).to_pandas()
    Path(output_path).mkdir(parents=True, exist_ok=True)
    id_name = [{"id": k, "name": v.decode("utf-8")} for k, v in enumerate(list(embedding["image_name"]))]
    json.dump(id_name, open(output_path + "/id_name.json", "w"))
    embedding = np.stack(embedding["embedding"].to_numpy())
    np.save(open(output_path + "/embedding.npy", "wb"), embedding)


def build_index(embedding_vector):
    """
    leverage Facebook's FAISS for things such as similarity search
    :param embedding_vector:
    :return:
    """
    d = embedding_vector.shape[1]
    xb = embedding_vector
    index = faiss.IndexFlatIP(d)
    index.add(xb)
    return index


def search(index, id_to_name, id_to_sub_dir, embedding, rank=5):
    """
    this is the main work-horse of the similarity based search using image embeddings
    """
    # TODO: this plus 1 operation is just for removing the top-1 item that is the same as query key...
    D, I = index.search(np.expand_dims(embedding, 0), rank + 1)  # FAISS search
    list_dist_index = list(zip(D[0], [(id_to_name[x], id_to_sub_dir[x]) for x in I[0]]))
    # Note: we start from index of 1 to get rid of the top-1 item which is the same as query key...
    return list_dist_index[1:]
