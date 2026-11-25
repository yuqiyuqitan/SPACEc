from ._configurator import style_configurator
from ._general import (
    BC_projection,
    catplot,
    classification_report_heatmap,
    cluster_composition_heatmap,
    cluster_qc,
    cn_exp_heatmap,
    cn_map,
    cn_upset,
    compensation_qc,
    confusion_matrix_heatmap,
    coordinates_on_image,
    correlation_heatmap,
    count_patch_proximity_res,
    create_pie_charts,
    distance_graph,
    dumbbell,
    expression_on_masks,
    extended_dotplot,
    feature_outlier_grid,
    marker_dotplot,
    patch_QC,
    plot_masks,
    plot_top_n_distances,
    ppa_res_donut,
    run_patch_QC,
    stacked_bar_plot,
    swarm_box_plot,
    umap,
    zcount_thres,
)
from ._qptiff_converter import tissue_lables
from ._segmentation import segmentation_ch, show_masks
from ._spatial_widget import spatial_cell_type_widget
from ._style import (
    SPACEcStyle,
    _add_scalebar,
    _style_colorbar,
    _style_legend,
    get_active_style,
    preview_style,
    set_style,
)
from ._style import style as style_context

__all__ = [
    # Style system
    "SPACEcStyle",
    "set_style",
    "style_context",
    "get_active_style",
    "preview_style",
    "style_configurator",
    "_style_legend",
    "_style_colorbar",
    "_add_scalebar",
    # segmentation
    "segmentation_ch",
    "show_masks",
    # general
    "coordinates_on_image",
    "catplot",
    "stacked_bar_plot",
    "swarm_box_plot",
    "create_pie_charts",
    "cn_exp_heatmap",
    "cn_map",
    "cn_upset",
    "dumbbell",
    "count_patch_proximity_res",
    "BC_projection",
    "plot_top_n_distances",
    "distance_graph",
    "plot_masks",
    "expression_on_masks",
    "feature_outlier_grid",
    # clustering / annotation
    "umap",
    "marker_dotplot",
    "extended_dotplot",
    "patch_QC",
    "run_patch_QC",
    "cluster_composition_heatmap",
    "correlation_heatmap",
    # diagnostics & QC
    "classification_report_heatmap",
    "confusion_matrix_heatmap",
    "compensation_qc",
    "cluster_qc",
    "zcount_thres",  # deprecated, moved to sp.pp
    # qptiff converter
    "tissue_lables",
    "ppa_res_donut",
    # interactive widgets
    "spatial_cell_type_widget",
]
