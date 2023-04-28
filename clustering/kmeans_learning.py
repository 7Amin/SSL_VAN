import logging
import os
import sys

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from clustering.utils.data_utils import get_loader

import joblib
import json


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("learn_kmeans")


def get_km_model(
    n_clusters,
    init,
    max_iter,
    batch_size,
    tol,
    max_no_improvement,
    n_init,
    reassignment_ratio,
):
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        batch_size=batch_size,
        verbose=1,
        compute_labels=False,
        tol=tol,
        max_no_improvement=max_no_improvement,
        init_size=None,
        n_init=n_init,
        reassignment_ratio=reassignment_ratio,
    )


def load_data(args):
    training_loader = get_loader(args)
    res = []
    for idx, batch_data in enumerate(training_loader):
        data = batch_data["image"].numpy()
        logger.info(f"for batch number {idx} the res size is {len(res)}")
        number = 0.1
        if len(res) == 0:
            number = 0.5
        sampled_indices = np.random.choice(len(data), int(len(data) * number), replace=False)
        data = np.squeeze(data)

        sampled_values = data[sampled_indices]
        merged_array = np.reshape(sampled_values,
                                  (sampled_values.shape[0] * sampled_values.shape[1],
                                   sampled_values.shape[2] * sampled_values.shape[3]))

        res.extend(merged_array)

    logger.info(f"data is ready and size is {len(res)}")
    return res


def learn_kmeans(args):
    filename = "cluster_paths.json"
    np.random.seed(args.seed)
    feat = load_data(args)
    logger.info(f"model is defined")
    km_model = get_km_model(
        args.n_clusters,
        args.init,
        args.max_iter,
        args.batch_size,
        args.tol,
        args.max_no_improvement,
        args.n_init,
        args.reassignment_ratio,
    )
    logger.info(f"training is started")
    km_model.fit(feat)
    logger.info(f"training is finished")
    # y_pred = km_model.predict(feat)
    url = args.km_path.format(args.n_clusters, args.max_iter, args.n_init)
    joblib.dump(km_model, url)

    if os.path.isfile(filename):
        with open(filename, 'r') as file:
            data = json.load(file)
    else:
        # Create new file with empty list
        data = []

    # Append new record to data
    data.append({
        "url": url,
    })

    # Write updated data back to file
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)
    inertia = -km_model.score(feat) / len(feat)
    # y_pred = km_model.predict(feat)
    logger.info("total intertia: %.5f", inertia)
    logger.info("finished successfully")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--a_min", default=-1000, type=float, help="a_min in ScaleIntensityRanged")
    parser.add_argument("--a_max", default=1000, type=float, help="a_max in ScaleIntensityRanged")
    parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
    parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
    parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
    parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
    parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
    parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
    parser.add_argument("--num_workers", default=1, type=int, help="num_workers")
    parser.add_argument("--distributed", action="store_true", help="start distributed training")
    parser.add_argument("--mode", default='test', choices=['test', 'server'], type=str)
    parser.add_argument("--num_samples", default=4, type=int)

    parser.add_argument("--km_path", default='../cluster_models_1/cluster_model_1_{}_{}_{}.joblib', type=str)
    parser.add_argument("--n_clusters", default=105, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--init", default="k-means++")
    parser.add_argument("--max_iter", default=2, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--tol", default=0.0, type=float)
    parser.add_argument("--max_no_improvement", default=100, type=int)
    parser.add_argument("--n_init", default=2, type=int)
    parser.add_argument("--reassignment_ratio", default=0.0, type=float)
    args = parser.parse_args()
    logging.info(str(args))

    learn_kmeans(args)


# # load the model from the file
# loaded_model = load('my_model.joblib')
#
# # use the model to make predictions on new data
# X_new = np.random.randn(10, 10)
# y_pred = loaded_model.predict(X_new)
