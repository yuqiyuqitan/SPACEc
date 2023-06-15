#' Calculate triangulation distances
#' Last Update: 2023-01-18
#' 
#' @description Using delauney triangulation, compute the interactions between cells in a 2D space
#' 
#' @param df_input dataframe containing unique cell id, x position, y position, cell type annotaiton, and region FOV
#' @param id string referring to  name of column containing unique cell id (there should be no duplicates in the dataframe)
#' @param x_pos string referring to name of column containing x location
#' @param y_pos string referring to name of column containing y location
#' @param cell_type string referring to name of column containing cell type annotations
#' @param region string referring to name of column containing region annotations
#' 
#' @return dataframe containing indices, annotation, and XY positions of all 
#' triangulated cell type interactions and their distance

calculate_triangulation_distances <- function(df_input, 
                                              id, 
                                              x_pos, 
                                              y_pos, 
                                              cell_type, 
                                              region) {
  # Compute the rdelaun distances
  vtress <- deldir::deldir(df_input[[x_pos]], df_input[[y_pos]])
  rdelaun_result <- vtress$delsgs
  
  # Get interactions going both directions
  inverse_result <- rdelaun_result 
  colnames(inverse_result) <- c("x2", "y2", "x1", "y1", "ind2", "ind1") 
  inverse_result <- inverse_result %>%
    dplyr::select(x1, y1, x2, y2, ind1, ind2)
  
  # Combine distances and annotate results with cell type and region information
  rdelaun_result <- rbind(rdelaun_result,
                         inverse_result) %>%
    dplyr::mutate(cell1ID = paste0(x1, "_", y1),
           cell2ID = paste0(x2, "_", y2))

  annotated_result <- rdelaun_result %>%
    dplyr::left_join(df_input,
              by = c("cell1ID" = "XYcellID")) %>%
    dplyr::rename(celltype1 = {{ cell_type }}) %>%
    dplyr::select(-{{ x_pos }},
           -{{ y_pos }},
           -{{ region }},
           -uniqueID)

  annotated_result <- annotated_result %>%
    dplyr::left_join(df_input,
              by = c("cell2ID" = "XYcellID")) %>%
    dplyr::rename(celltype2 = {{ cell_type }}) %>%
    dplyr::select(x1, y1, celltype1, !!as.symbol(paste0(id, ".x")),
           x2, y2, celltype2, !!as.symbol(paste0(id, ".y")),
           {{ region }})
  
  # Calculate distance and reorder columns
  annotated_result <- annotated_result%>%
    dplyr::mutate(distance = sqrt((x2-x1)^2 + (y2-y1)^2)) %>%
    dplyr::select(!!as.symbol(region), 
           !!as.symbol(paste0(id, ".x")), celltype1, x1, y1,
           !!as.symbol(paste0(id, ".y")), celltype2, x2, y2,
           distance)
  colnames(annotated_result) <- c(region, 
                                  "celltype1_index", "celltype1", "celltype1_X", "celltype1_Y",
                                  "celltype2_index", "celltype2", "celltype2_X", "celltype2_Y", 
                                  "distance")
  return(annotated_result)
}

#' @description This is a function in R that calculates the triangulation distances between cells of different types in a given dataset. 
#' 
#' @param df_input: The input dataframe that contains the cell information.
#' @param id: The name of the column in the dataframe that corresponds to the ID of the cells.
#' @param x_pos: The name of the column in the dataframe that corresponds to the x-position of the cells.
#' @param y_pos: The name of the column in the dataframe that corresponds to the y-position of the cells.
#' @param cell_type: The name of the column in the dataframe that corresponds to the type of the cells.
#' @param region: The name of the column in the dataframe that corresponds to the region of the cells.
#' @param num_cores: The number of cores to be used for parallel processing. Defaults to half the number of available cores.
#' @param calc_avg_distance: A Boolean that controls whether the function calculates the average distance between cell types and individual cells. 
#' Defaults to TRUE. The results are stored under the directory, which is defined in csv_output.
#' @param csv_output: The file path where the results of calc_avg_distance will be saved in csv format. Defaults to the working directory.
#' 
#' @return The output of the function is the triangulation distances between cells of different types in the input dataset, 
#' for each region in the dataset. The output is a data frame containing the triangulation distances for each region

get_triangulation_distances <- function(df_input, 
                                        id, 
                                        x_pos, 
                                        y_pos, 
                                        cell_type, 
                                        region, 
                                        num_cores = NULL,
                                        calc_avg_distance = TRUE,
                                        csv_output = getwd()) {
  
  if(typeof(df_input[,x_pos]) != "integer"){
    
    warning("This function expects integer values for xy coordinates.")
    warning("Class will be changed to integer. Please check the generated output!")
    
    i <- c(x_pos, y_pos)   
    df_input[ , i] <- apply(df_input[ , i], 2,            # Specify own function within apply
                        function(x) as.integer(x))
  }
  
  library(doSNOW)
  library(foreach)
  library(parallel)
  
  # Get unique regions
  unique_regions <- unique(df_input[[region]])
  
  # Select only necessary columns
  df_input <- df_input %>%
    dplyr::select({{ id }},
           {{ x_pos }},
           {{ y_pos }},
           {{ cell_type }},
           {{ region }})
  
  # Set up parallelization
  if (is.null(num_cores)){
    num_cores <- floor(detectCores()/2) # default to using half of available cores
  }
  cl <- makeCluster(num_cores)
  clusterExport(cl, c("calculate_triangulation_distances"))
  registerDoSNOW(cl)
  
  # Progress bar
  pb <- utils::txtProgressBar(max = length(unique_regions), style = 3)
  progress <- function(n) utils::setTxtProgressBar(pb, n)
  opts <- list(progress = progress)
  
  triangulation_distances <- foreach(reg_index = 1:length(unique_regions), 
                                     .packages = c("deldir", "tidyverse"), 
                                     .combine = "rbind", 
                                     .options.snow = opts)%dopar%{
    # SUBSET DATASET
    subset <- df_input %>%
      dplyr::filter(!!as.symbol(region) == unique_regions[reg_index]) %>%
      dplyr::mutate(uniqueID = paste0(!!as.symbol(id), "-",
                               !!as.symbol(x_pos), "-",
                               !!as.symbol(y_pos)),
             XYcellID = paste0(!!as.symbol(x_pos),"_", !!as.symbol(y_pos)))
    
    result <- calculate_triangulation_distances(df_input = subset,
                                                id = id,
                                                x_pos = x_pos,
                                                y_pos = y_pos,
                                                cell_type = cell_type,
                                                region = region)
    return(result)
    }
  
  close(pb)
  stopCluster(cl)
  
  if(calc_avg_distance == TRUE) {
    calculate_avg_distance(triangulation_distances = triangulation_distances,
                           csv_output = csv_output)
  }
  
  
  return(triangulation_distances)
}

#' @description 
#' 
#' @param df_input: a dataframe containing the original data
#' @param cell_type: a string representing the name of the column containing the cell type annotations
#' @param region: a string representing the name of the column containing the region information
#' @param permutation: an integer representing a specific permutation number to use as a seed for the random number generator
#' 
#' @return The output of this function is a modified version of the input dataframe df_input where the annotations for the cell types are shuffled, 
#' the shuffling is done based on the unique regions


shuffle_annotations <- function(df_input, 
                                cell_type, 
                                region, 
                                permutation) {
  unique_regions <- unique(df_input[[region]])
  
  df_shuffled <- lapply(1:length(unique_regions),
                        function(region_num){
                          # Subset dataframe
                          df_subset <- df_input %>%
                            dplyr::filter(!!as.symbol(region) == unique_regions[region_num])
                          
                          # Shuffle annotaitons
                          shuffled_annotations <- data.frame(annotations = df_subset[[cell_type]])
                          set.seed(permutation + 1234) # change seed with every permutation
                          rows <- sample(nrow(shuffled_annotations))
                          shuffled_annotations <- data.frame(shuffled_annotations[rows,])
                          colnames(shuffled_annotations) <- c("random_annotations")
                          
                          df_subset <- cbind(df_subset, shuffled_annotations) 
                          
                          return(df_subset)
                        })
  df_shuffled <- do.call(rbind, df_shuffled)
  return(df_shuffled)
}

#' @description 
#' 
#' @param df_input: a dataframe containing the original data
#' @param num_iterations: an integer representing the number of iterations to perform (defaults to 1000)
#' @param id: a string representing the name of the column containing the unique IDs of each cell
#' @param x_pos: a string representing the name of the column containing the x-coordinate position of each cell
#' @param y_pos: a string representing the name of the column containing the y-coordinate position of each cell
#' @param cell_type: a string representing the name of the column containing the cell type annotations
#' @param region: a string representing the name of the column containing the region information
#' @param num_cores: an optional integer representing the number of cores to use for parallel computation (defaults to half of available cores)
#' 
#' @return 


iterate_triangulation_distances <- function(df_input,
                                            num_iterations = 1000,
                                            id,
                                            x_pos,
                                            y_pos,
                                            cell_type,
                                            region,
                                            num_cores = NULL) {
  library(doSNOW)
  library(foreach)
  library(parallel)
  
  # Get unique regions
  unique_regions <- unique(df_input[[region]])
  
  # Select only necessary columns to speed up computation time
  df_input <- df_input %>%
    dplyr::select(!!as.symbol(id), 
           !!as.symbol(x_pos), 
           !!as.symbol(y_pos), 
           !!as.symbol(cell_type), 
           !!as.symbol(region))
  
  # Set up parallelization
  if (is.null(num_cores)){
    num_cores <- floor(detectCores()/2) # default to using half of available cores
  }
  cl <- makeCluster(num_cores)
  clusterExport(cl, c("shuffle_annotations", "get_triangulation_distances", "calculate_triangulation_distances"))
  registerDoSNOW(cl)
  
  # Progress bar
  pb <- txtProgressBar(max = (length(unique_regions)*num_iterations), style = 3)
  progress <- function(n) utils::setTxtProgressBar(pb, n)
  opts <- list(progress = progress)
  
  
  iterative_triangulation_distances <- foreach(reg_index = 1:length(unique_regions)) %:%
    foreach(iteration_index = 1:num_iterations,
            .packages = c("deldir", "tidyverse"),
            .combine = "rbind",
            .options.snow = opts)%dopar%{
              subset <- df_input %>%
                dplyr::filter(!!as.symbol(region) == unique_regions[reg_index]) %>%
                dplyr::mutate(uniqueID = paste0(!!as.symbol(id), "-",
                                         !!as.symbol(x_pos), "-",
                                         !!as.symbol(y_pos)),
                       XYcellID = paste0(!!as.symbol(x_pos),"_", !!as.symbol(y_pos)))
              
              df_shuffled <- shuffle_annotations(df_input = subset,
                                                 cell_type = cell_type,
                                                 region = region,
                                                 permutation = iteration_index)
              
              results <- get_triangulation_distances(df_input = df_shuffled,
                                                     id = id,
                                                     x_pos = x_pos,
                                                     y_pos = y_pos,
                                                     cell_type = "random_annotations",
                                                     region = region,
                                                     num_cores = num_cores,
                                                     calc_avg_distance = FALSE)
              
              per_cell_summary <- results %>%
                dplyr::group_by(celltype1_index, celltype1, celltype2) %>%
                dplyr::summarize(per_cell_mean_dist = mean(distance)) %>%
                dplyr::ungroup()
              
              per_celltype_summary <- per_cell_summary %>%
                dplyr::group_by(celltype1, celltype2) %>%
                dplyr::summarize(mean_dist = mean(per_cell_mean_dist)) %>%
                dplyr::ungroup() %>%
                dplyr::mutate(region = unique_regions[reg_index],
                       iteration = iteration_index)
              colnames(per_celltype_summary) <- c("celltype1", "celltype2", "mean_dist", region, "iteration")
              
              return(per_celltype_summary)
            }
  
  iterative_triangulation_distances <- do.call(rbind, iterative_triangulation_distances)
  close(pb)
  stopCluster(cl)
  
  return(iterative_triangulation_distances)
}

#' @description 
#' 
#' @param triangulation_distances: a dataframe containing the triangulation distances between cells
#' @param csv_output: an optional string representing the directory to which the output csv files should be written (defaults to the current working directory)
#' 
#' @return 


calculate_avg_distance <- function(triangulation_distances = triangulation_distances,
                                   csv_output = getwd()) {
  `%>%` <- magrittr::`%>%`
  
  print("Calculateing the average distance to different cell types on a per individual cell level. This can be interpreted as >> For cell #1, the average distance to a cell of type X is ____. <<")
  # Calculate the average distance to different cell types on a per individual cell level
  # This can be interpreted as
  # "For cell #1, the average distance to a cell of type X is ____."
  per_cell_summary <- triangulation_distances %>%
    dplyr::group_by(celltype1_index, celltype1, celltype2, unique_region) %>%
    dplyr::summarize(per_cell_mean_dist = mean(distance)) %>%
    dplyr::ungroup()
  print(head(per_cell_summary))
  readr::write_csv(per_cell_summary, paste0(csv_output, "/", "per_cell_summary.csv"))
  
  print("Calculateing the average distance between different cell types. This can be interpreted as >> The average distance between cell type X and cell type Y is ___. <<")
  # Calculate the average distance between different cell types
  # This can be interpreted as
  # "The average distance between cell type X and cell type Y is ___."
  per_celltype_summary <- per_cell_summary %>%
    dplyr::group_by(celltype1, celltype2, unique_region) %>%
    dplyr::summarize(mean_dist = mean(per_cell_mean_dist)) %>%
    dplyr::ungroup()
  print(head(per_celltype_summary))
  readr::write_csv(per_celltype_summary, paste0(csv_output, "/", "per_celltype_summary.csv"))
}

#' @description 
#' 
#' @param triangulation_distances: containing the observed triangulation distances between cells, and a dataframe 
#' #' @param iterated_triangulation_distances: containing the expected triangulation distances from the iterative shuffling process. 
#' @param distance_threshold: a numeric representing the distance threshold for observed cell-cell interactions
#' @param pair_to: a vector of strings representing the secondary cell types to be plotted in the dumbbell plot
#' @param colors: a vector of strings representing the colors to be used in the dumbbell plot
#' 
#' @return plots a Dumbbell_plot of logfold change in cell type interaction upon treatment


Dumbbell_plot_interactions <- function(triangulation_distances = triangulation_distances,
                                       iterated_triangulation_distances = iterated_triangulation_distances,
                                       distance_threshold = 128,
                                       treatment_condition_1 = treatment_condition_1,
                                       treatment_condition_2 = treatment_condition_2,
                                       pair_to = c("CD4+ Treg_Stromal", "CD8+ T cell_NK", "DC_NK", "NK_CD8+ T cell", "NK_CD8+ T cells", "Stromal_CD4+ Treg", "CD4+ T cell_Neutrophil", "CD8+ T cell PD1+_NK", "CD4+ T cell_CD4+ T cell", "CD4+ T cell_CD8+ T cell", "CD8+ T cell_Tumor PDL1+ MHCI+"),
                                       colors = c("#00BFC4","#F8766D")
) {
  
  `%>%` <- magrittr::`%>%`
  names(metadata)[names(metadata) == treatment_column] <- "treatment"
  # Set distance threshold for observed cell-cell interactions
  # distance_threshold = 128  corresponds to 100um
  
  # Reformat observed dataset
  observed_distances <- triangulation_distances %>%
    # Append metadata
    dplyr::left_join(metadata,
                     by = c("unique_region")) %>%
    dplyr::filter(distance <= distance_threshold) %>%
    # Calculate the average distance to every cell type for each cell
    dplyr::group_by(celltype1_index, celltype1, celltype2, treatment, unique_region) %>%
    dplyr::summarize(mean_per_cell = mean(distance)) %>%
    dplyr::ungroup() %>%
    # Calculate the average distance between cell type to cell type on a per group basis
    dplyr::group_by(celltype1, celltype2, treatment) %>%
    dplyr::summarize(observed = list(mean_per_cell),
                     observed_mean = mean(unlist(observed), na.rm = TRUE)) %>%
    dplyr::ungroup()
  
  # Reformat exepcted dataset
  expected_distances <- iterated_triangulation_distances %>%
    # Append metadata
    dplyr::left_join(metadata,
                     by = c("unique_region")) %>%
    dplyr::filter(mean_dist <= distance_threshold) %>%
    # Calculate expected mean distance and list values
    dplyr::group_by(celltype1, celltype2, treatment) %>%
    dplyr::summarize(expected = list(mean_dist),
                     expected_mean = mean(mean_dist, na.rm = TRUE)) %>%
    dplyr::ungroup() 
  
  # Calculate pvalues and log fold differences
  distance_pvals <- expected_distances %>%
    dplyr::left_join(observed_distances,
                     by = c("celltype1", "celltype2", "treatment")) %>%
    # Calculate wilcoxon test between observed and expected distances
    dplyr::group_by(celltype1, celltype2, treatment) %>%
    dplyr::mutate(pvalue = wilcox.test(unlist(expected), unlist(observed), exact = FALSE)$p.value) %>%
    dplyr::ungroup() %>%
    dplyr::select(-observed, -expected) %>%
    # Calculate log fold enrichment
    dplyr::mutate(logfold_group = log2(observed_mean/expected_mean),
                  interaction = paste0(celltype1, " --> ", celltype2)) 
  
  # Get order of plot by magnitude of logfold differences between groups
  intermed <- distance_pvals %>%
    dplyr::select(interaction, treatment, logfold_group) %>%
    tidyr::spread(key = treatment, value = logfold_group) 
  
  intermed$difference <- (intermed[,treatment_condition_2] - intermed[,treatment_condition_1])
  
  ord <-(intermed %>%
    dplyr::filter(!is.na(difference)) %>%
  dplyr::arrange(treatment_condition_1))$interaction
  
  # Assign interaction order
  distance_pvals$interaction <- factor(distance_pvals$interaction,
                                       levels = ord)
  
  # Dumbbell plot

  
  data = distance_pvals %>%
    dplyr::filter(!is.na(interaction))
  
  distance_pvals$pairs = paste0(distance_pvals$celltype1, "_", distance_pvals$celltype2)
  distance_pvals_sub = distance_pvals[distance_pvals$celltype1 %in%  pair_to, ]
  
  ggplot2::ggplot(data = distance_pvals_sub %>%
           dplyr::filter(!is.na(interaction))) +
    ggplot2::geom_vline(mapping = ggplot2::aes(xintercept = 0), linetype = "dashed") +
    ggplot2::geom_line(mapping = ggplot2::aes(x = logfold_group, y = interaction),
              na.rm = TRUE) +
    ggplot2::geom_point(mapping = aes(x = logfold_group, y = interaction, fill = treatment, shape = treatment), 
               size = 4, stroke = 0.5, na.rm = TRUE) +
    ggplot2::scale_shape_manual(values = c(24, 22)) + ggplot2::scale_fill_manual(values = colors) +
    ggplot2::theme_bw()+
    ggplot2::theme(panel.grid.major.x = element_blank(),
          panel.grid.minor.x = element_blank(),
          axis.text.y = element_text(size = 16),
          axis.text.x = element_text(size = 16, angle = 45, hjust = 1),
          axis.title.y = element_text(size = 16),
          axis.title.x = element_text(size = 16))
}

#' @description 
#' 
#' @param XXXX
#' 
#' @return 


triangulation_distances <- function(df_input = df_input,
                                    id = "index",
                                    x_pos = "x",
                                    y_pos = "y",
                                    cell_type = "Cell.common",
                                    region = "unique_region",
                                    num_iterations = 1,
                                    plot = TRUE,
                                    plot_triangulation_distances = triangulation_distances,
                                    plot_distance_threshold = 128,
                                    plot_celltype_1 = "CD8+ T cell", 
                                    plot_pair_to = pairs_for_comparisson,
                                    plot_colors = c("#00BFC4","#F8766D")){
  
  iterated_triangulation_distances <- iterate_triangulation_distances(df_input = df_input,
                                                                      id = id,
                                                                      x_pos = x_pos,
                                                                      y_pos = y_pos,
                                                                      cell_type = cell_type,
                                                                      region = region,
                                                                      num_iterations = num_iterations)
  head(iterated_triangulation_distances)
  
  if(plot == TRUE){
    Dumbbell_plot_interactions(triangulation_distances = triangulation_distances,
                               iterated_triangulation_distances = iterated_triangulation_distances,
                               distance_threshold = plot_distance_threshold,
                               celltype_1 = plot_celltype_1, 
                               pair_to = plot_pair_to,
                               colors = plot_colors)
  }
  
  
}
