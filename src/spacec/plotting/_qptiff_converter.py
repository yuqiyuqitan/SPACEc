import matplotlib.pyplot as plt


def tissue_lables(tissueframe, region="region1"):
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
