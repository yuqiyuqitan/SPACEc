import pathlib
from tempfile import TemporaryDirectory


TEST_DIR = pathlib.Path(__file__).parent


def test_cell_segmentation():

    data_path = TEST_DIR / 'data/raw'
    
    with TemporaryDirectory() as output_dir:

        print("Segmentation CH")
        from spacec.segmentation import pl_segmentation_ch
        pl_segmentation_ch(
            # image for segmentation
            file_name = data_path / 'tonsil/1/reg010_X01_Y01_Z01_small.tif',
            # all channels used for staining
            channel_file = data_path / 'tonsil/channelnames.txt',
            output_dir = output_dir,
            # channels used for membrane segmentation
            extra_seg_ch_list = ["CD45", "betaCatenin"], 
            nuclei_channel = 'DAPI',
            technology = 'Phenocycler',
        )

        print("Cell Segmentation")
        from spacec.segmentation import tl_cell_segmentation
        # choose between cellpose or mesmer for segmentation
        # first image
        # seg_output contains {'img': img, 'image_dict': image_dict, 'masks': masks}
        seg_output1 = tl_cell_segmentation(
            file_name = data_path / 'tonsil/1/reg010_X01_Y01_Z01_small.tif',
            channel_file = data_path / 'tonsil/channelnames.txt',
            output_dir = output_dir,
            output_fname = 'tonsil1',
            seg_method ='mesmer', # cellpose or mesmer
            nuclei_channel = 'DAPI',
            membrane_channel_list = ["CD45", "betaCatenin"], #default is None; if provide more than one channel, then they will be combined
            compartment = 'whole-cell', # mesmer # segment whole cells or nuclei only
            technology ='Phenocycler', # Phenocycler or codex
            size_cutoff = 0)
        
        print("Show Masks")
        from spacec.segmentation import pl_show_masks
        overlay_data1, rgb_images1 = pl_show_masks(
            seg_output=seg_output1,
            nucleus_channel = 'DAPI', # channel used for nuclei segmentation (displayed in blue)
            additional_channels = ["CD45", "betaCatenin"], # additional channels to display (displayed in green - channels will be combined into one image)
            show_subsample = True, # show a random subsample of the image
            n=2, #need to be at least 2
            tilesize = 300,# number of subsamples and tilesize
            rand_seed = 1)
        
        # #Save segmentation output
        # with open(output_dir / 'seg_output_tonsil1.pickle', 'wb') as f:
        #     pickle.dump(seg_output1, f)

        # #Save the overlay of the data
        # with open(output_dir / 'overlay_tonsil1.pickle', 'wb') as f:
        #     pickle.dump(overlay_data1, f)


if __name__ == '__main__':
    test_cell_segmentation()