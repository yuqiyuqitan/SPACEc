import pathlib
from tempfile import TemporaryDirectory

TEST_DIR = pathlib.Path(__file__).parent


def test_cell_segmentation():

    import matplotlib

    import spacec as sp
    from spacec.tools._segmentation import _CELLPOSE_SAM, _cellpose_version

    matplotlib.use("Agg")

    data_path = TEST_DIR / "data"

    gpu = sp.hf.check_for_gpu()

    with TemporaryDirectory() as output_dir:
        # output_dir = pathlib.Path("tests/_out")
        # output_dir.mkdir(exist_ok=True, parents=True)

        print("Segmentation CH")

        sp.pl.segmentation_ch(
            # image for segmentation
            file_name=data_path / "raw/tonsil/1/reg010_X01_Y01_Z01.tif",
            # all channels used for staining
            channel_file=data_path / "raw/tonsil/channelnames.txt",
            output_dir=output_dir,
            # channels used for membrane segmentation
            extra_seg_ch_list=["CD45", "betaCatenin"],
            nuclei_channel="DAPI",
            input_format="Multichannel",
        )

        print("Cell Segmentation (default backend)")
        seg_output1 = sp.tl.cell_segmentation(
            file_name=data_path / "raw/tonsil/1/reg010_X01_Y01_Z01.tif",
            channel_file=data_path / "raw/tonsil/channelnames.txt",
            output_dir=output_dir,
            output_fname="tonsil1",
            nuclei_channel="DAPI",
            membrane_channel_list=[
                "CD45",
                "betaCatenin",
            ],  # default is None; if provide more than one channel, then they will be combined
            compartment="whole-cell",
            input_format="Multichannel",  # Phenocycler or codex
            size_cutoff=0,
        )

        print("Cell Tiled Segmentation (legacy mesmer alias)")
        seg_output2 = sp.tl.cell_segmentation(
            file_name=data_path / "raw/tonsil/1/reg010_X01_Y01_Z01.tif",
            channel_file=data_path / "raw/tonsil/channelnames.txt",
            output_dir=output_dir,
            output_fname="tonsil1",
            seg_method="mesmer",
            nuclei_channel="DAPI",
            membrane_channel_list=[
                "CD45",
                "betaCatenin",
            ],  # default is None; if provide more than one channel, then they will be combined
            compartment="whole-cell",
            input_format="Multichannel",  # Phenocycler or codex
            size_cutoff=0,
            tile_size=32,  # Default tile size
            tile_overlap=0,  # Default tile overlap
            tiling_threshold=32,  # Minimum image size to use tiling
        )

        print(
            f"Cell Segmentation Cellpose (legacy model name 'cyto3', cellpose {_cellpose_version})"
        )
        seg_output_cellpose = sp.tl.cell_segmentation(
            file_name=data_path / "raw/tonsil/1/reg010_X01_Y01_Z01.tif",
            channel_file=data_path / "raw/tonsil/channelnames.txt",
            output_dir=output_dir,
            output_fname="tonsil2",
            seg_method="cellpose",  # cellpose or mesmer
            model="cyto3",  # legacy model name; remapped to 'cpsam' with a warning on cellpose>=4
            diameter=28,  # average cell diameter (in pixels). If set to None, it will be automatically estimated.
            nuclei_channel="DAPI",
            membrane_channel_list=[
                "CD45",
                "betaCatenin",
            ],  # default is None #default is None; if provide more than one channel, then they will be combined
            input_format="Multichannel",  # Phenocycler or codex
            resize_factor=1,  # default is 1; if the image is too large, lower the value. Lower values will speed up the segmentation but may reduce the accuracy.
            size_cutoff=0,
            use_gpu=gpu,
        )
        assert seg_output_cellpose["masks"].max() > 0, (
            "Legacy model name 'cyto3' should still produce a non-empty mask "
            f"on cellpose {_cellpose_version}."
        )

        print("Cell Segmentation Cellpose with save_mask_as_png=True")
        sp.tl.cell_segmentation(
            file_name=data_path / "raw/tonsil/1/reg010_X01_Y01_Z01.tif",
            channel_file=data_path / "raw/tonsil/channelnames.txt",
            output_dir=output_dir,
            output_fname="tonsil_png",
            seg_method="cellpose",
            model="cyto3",
            diameter=28,
            nuclei_channel="DAPI",
            membrane_channel_list=[
                "CD45",
                "betaCatenin",
            ],
            input_format="Multichannel",
            resize_factor=1,
            size_cutoff=0,
            use_gpu=gpu,
            save_mask_as_png=True,
        )
        png_files = list(pathlib.Path(output_dir).rglob("cellpose_seg_*_seg.png"))
        assert (
            len(png_files) == 1
        ), f"Expected exactly one Cellpose diagnostic PNG, found {png_files}"
        import skimage.io as _skimage_io

        saved_overlay = _skimage_io.imread(png_files[0])
        assert (
            saved_overlay.ndim == 3 and saved_overlay.shape[-1] == 3
        ), f"Expected an RGB overlay PNG, got array with shape {saved_overlay.shape}"

        print("Cell Segmentation Cellpose with save_mask=True")
        seg_output_mask = sp.tl.cell_segmentation(
            file_name=data_path / "raw/tonsil/1/reg010_X01_Y01_Z01.tif",
            channel_file=data_path / "raw/tonsil/channelnames.txt",
            output_dir=output_dir,
            output_fname="tonsil_mask",
            seg_method="cellpose",
            model="cyto3",
            diameter=28,
            nuclei_channel="DAPI",
            membrane_channel_list=[
                "CD45",
                "betaCatenin",
            ],
            input_format="Multichannel",
            resize_factor=1,
            size_cutoff=0,
            use_gpu=gpu,
            save_mask=True,
        )
        mask_files = list(pathlib.Path(output_dir).rglob("tonsil_mask_mask.tif"))
        assert (
            len(mask_files) == 1
        ), f"Expected exactly one saved raw mask .tif, found {mask_files}"
        import tifffile as _tifffile

        saved_mask = _tifffile.imread(mask_files[0])
        assert (saved_mask == seg_output_mask["masks"]).all(), (
            "Saved mask .tif should round-trip to the same array returned in "
            "seg_output['masks']"
        )

        print("Cell Tiled Segmentation Cellpose")
        seg_output_cellpose = sp.tl.cell_segmentation(
            file_name=data_path / "raw/tonsil/1/reg010_X01_Y01_Z01.tif",
            channel_file=data_path / "raw/tonsil/channelnames.txt",
            output_dir=output_dir,
            output_fname="tonsil2",
            seg_method="cellpose",  # cellpose or mesmer
            model="cyto3",  # cellpose model
            diameter=28,  # average cell diameter (in pixels). If set to None, it will be automatically estimated.
            nuclei_channel="DAPI",
            membrane_channel_list=[
                "CD45",
                "betaCatenin",
            ],  # default is None #default is None; if provide more than one channel, then they will be combined
            input_format="Multichannel",  # Phenocycler or codex
            resize_factor=1,  # default is 1; if the image is too large, lower the value. Lower values will speed up the segmentation but may reduce the accuracy.
            size_cutoff=0,
            use_gpu=gpu,
            tile_size=32,  # Default tile size
            tile_overlap=0,  # Default tile overlap
            tiling_threshold=32,  # Minimum image size to use tiling
        )

        if _CELLPOSE_SAM:
            print(
                f"Skipping custom model test: CP_test.zip was fine-tuned on cellpose "
                f"3.x and cannot be loaded by Cellpose-SAM (cellpose {_cellpose_version})."
            )
        else:
            print("Cell Segmentation Cellpose with custom model")
            seg_output_cellpose = sp.tl.cell_segmentation(
                file_name=data_path / "raw/tonsil/1/reg010_X01_Y01_Z01.tif",
                channel_file=data_path / "raw/tonsil/channelnames.txt",
                output_dir=output_dir,
                output_fname="tonsil2",
                seg_method="cellpose",  # cellpose or mesmer
                model=data_path / "CP_test.zip",  # cellpose model
                diameter=28,  # average cell diameter (in pixels). If set to None, it will be automatically estimated.
                nuclei_channel="DAPI",
                membrane_channel_list=[
                    "CD45",
                    "betaCatenin",
                ],  # default is None #default is None; if provide more than one channel, then they will be combined
                input_format="Multichannel",  # Phenocycler or codex
                resize_factor=1,  # default is 1; if the image is too large, lower the value. Lower values will speed up the segmentation but may reduce the accuracy.
                size_cutoff=0,
                custom_model=True,
            )

        print("Show Masks")
        overlay_data1, rgb_images1 = sp.pl.show_masks(
            seg_output=seg_output1,
            nucleus_channel="DAPI",  # channel used for nuclei segmentation (displayed in blue)
            additional_channels=[
                "CD45",
                "betaCatenin",
            ],  # additional channels to display (displayed in green - channels will be combined into one image)
            show_subsample=True,  # show a random subsample of the image
            n=2,  # need to be at least 2
            tilesize=300,  # number of subsamples and tilesize
            rand_seed=1,
        )

        print("Show Masks Cellpose")
        overlay_data1, rgb_images1 = sp.pl.show_masks(
            seg_output=seg_output2,
            nucleus_channel="DAPI",  # channel used for nuclei segmentation (displayed in blue)
            additional_channels=[
                "CD45",
                "betaCatenin",
            ],  # additional channels to display (displayed in green - channels will be combined into one image)
            show_subsample=True,  # show a random subsample of the image
            n=2,  # need to be at least 2
            tilesize=300,  # number of subsamples and tilesize
            rand_seed=1,
        )

        # Save segmentation output

        import pickle

        with open(output_dir + "seg_output_tonsil1.pickle", "wb") as f:
            pickle.dump(seg_output1, f)

        # #Save the overlay of the data
        # with open(output_dir / 'overlay_tonsil1.pickle', 'wb') as f:
        #     pickle.dump(overlay_data1, f)


if __name__ == "__main__":
    test_cell_segmentation()
