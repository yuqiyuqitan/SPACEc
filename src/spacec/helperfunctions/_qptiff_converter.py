import matplotlib.pyplot as plt
import tifffile
from scipy import ndimage as ndi
from skimage.transform import resize


def downscale_tissue(
    file_path,
    DNAslice=0,
    downscale_factor=64,
    sigma=5.0,
    padding=50,
    savefig=False,
    showfig=True,
    output_dir="./",
    output_fname="",
    figsize=(10, 5),  # new parameter for figure size
):
    """Load and downscale a qptiff nuclear channel image.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Path to the input qptiff file.
    DNAslice : int, default=0
        Index of the channel to use as the nuclear image.
    downscale_factor : int, default=64
        Integer factor used to reduce image size in each spatial dimension.
    sigma : float, default=5.0
        Standard deviation for Gaussian smoothing after resizing.
    padding : int, default=50
        Unused legacy argument retained for backward compatibility.
    savefig : bool, default=False
        Whether to save the preview figure to disk.
    showfig : bool, default=True
        Whether to create and display/save the preview figure.
    output_dir : str, default="./"
        Output directory used when ``savefig=True``.
    output_fname : str, default=""
        Prefix used when writing the preview figure.
    figsize : tuple[int, int], default=(10, 5)
        Figure size used for preview plots.

    Returns
    -------
    numpy.ndarray
        The resized and smoothed nuclear image.
    """
    print("Reading in the qptiff file, might take awhile!")
    currim = tifffile.imread(file_path)
    nucim = currim[DNAslice]
    del currim
    print(f"Loaded nuclear image of dimension (Y,X) = {nucim.shape}")
    resized_im = resize(
        nucim,
        (nucim.shape[0] // downscale_factor, nucim.shape[1] // downscale_factor),
        anti_aliasing=True,
    )
    resized_im = ndi.gaussian_filter(resized_im, sigma=sigma)

    if showfig:
        fig, axs = plt.subplots(1, 2, figsize=figsize)  # use figsize parameter here
        axs[0].imshow(nucim)
        axs[0].set_title("Nuclear image")
        axs[1].hist(resized_im)
        axs[1].set_title("Marker expression level histogram")
        if savefig:
            fig.savefig(
                output_dir + output_fname + "_raw_tissue_plot.pdf", bbox_inches="tight"
            )
        else:
            plt.show()
    print("returning scaled down image!")
    return resized_im
