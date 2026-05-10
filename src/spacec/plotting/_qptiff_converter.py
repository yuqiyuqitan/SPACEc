import matplotlib.pyplot as plt
import warnings

from ..config import SPACEC_CONFIG


def tissue_labels(tissueframe, region=SPACEC_CONFIG.plotting.default_region_label):
    """
    Plot the tissue and region labels of the given DataFrame.

    Parameters
    ----------
    tissueframe : DataFrame
        The DataFrame containing the labels from the segmentation.
    region : str, optional
        The region to group by, by default "region1".

    Returns
    -------
    None
    """
    centroids = tissueframe.groupby("tissue").mean()
    fig, ax = plt.subplots()
    ax.scatter(centroids["x"], centroids["y"])
    ax.invert_yaxis()
    plt.gca().set_aspect("equal", adjustable="box")

    for i, txt in enumerate(centroids.index):
        ax.annotate(txt, (list(centroids["x"])[i], list(centroids["y"])[i]))

    plt.title("Tissue piece labels")
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(centroids["x"], centroids["y"])
    ax.invert_yaxis()
    plt.gca().set_aspect("equal", adjustable="box")

    for i, txt in enumerate(centroids[region]):
        ax.annotate(int(txt), (list(centroids["x"])[i], list(centroids["y"])[i]))

    plt.title("Region labels")
    plt.show()


def tissue_lables(tissueframe, region=SPACEC_CONFIG.plotting.default_region_label):
    warnings.warn(
        "`tissue_lables` is deprecated and will be removed in a future release. "
        "Use `tissue_labels` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return tissue_labels(tissueframe=tissueframe, region=region)
