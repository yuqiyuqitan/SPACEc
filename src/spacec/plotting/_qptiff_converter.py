import matplotlib.pyplot as plt

from ._style import apply_publication_style, save_figure

apply_publication_style()


def tissue_lables(
    tissueframe,
    region="region1",
    savefig=False,
    output_dir="./",
    output_fname="tissue_labels",
    export_formats=("pdf", "svg", "png"),
    dpi=300,
):
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
    if savefig:
        save_figure(
            fig=fig,
            output_dir=output_dir,
            output_fname=f"{output_fname}_tissue",
            formats=export_formats,
            dpi=dpi,
        )
        plt.close(fig)
    else:
        plt.show()


def tissue_labels(*args, **kwargs):
    """Alias for :func:`tissue_lables` with corrected spelling."""
    return tissue_lables(*args, **kwargs)

    fig, ax = plt.subplots()
    ax.scatter(centroids["x"], centroids["y"])
    ax.invert_yaxis()
    plt.gca().set_aspect("equal", adjustable="box")

    for i, txt in enumerate(centroids[region]):
        ax.annotate(int(txt), (list(centroids["x"])[i], list(centroids["y"])[i]))

    plt.title("Region labels")
    if savefig:
        save_figure(
            fig=fig,
            output_dir=output_dir,
            output_fname=f"{output_fname}_region",
            formats=export_formats,
            dpi=dpi,
        )
        plt.close(fig)
    else:
        plt.show()
