#' Calculate triangulation distances
#' Last Update: 2021-12-01
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
  vtress <- deldir(df_input[[x_pos]], df_input[[y_pos]])
  rdelaun_result <- vtress$delsgs
  
  # Get interactions going both directions
  inverse_result <- rdelaun_result 
  colnames(inverse_result) <- c("x2", "y2", "x1", "y1", "ind2", "ind1") 
  inverse_result <- inverse_result %>%
    select(x1, y1, x2, y2, ind1, ind2)
  
  # Combine distances and annotate results with cell type and region information
  rdelaun_result <- rbind(rdelaun_result,
                         inverse_result) %>%
    mutate(cell1ID = paste0(x1, "_", y1),
           cell2ID = paste0(x2, "_", y2))

  annotated_result <- rdelaun_result %>%
    left_join(df_input,
              by = c("cell1ID" = "XYcellID")) %>%
    rename(celltype1 = {{ cell_type }}) %>%
    select(-{{ x_pos }},
           -{{ y_pos }},
           -{{ region }},
           -uniqueID)

  annotated_result <- annotated_result %>%
    left_join(df_input,
              by = c("cell2ID" = "XYcellID")) %>%
    rename(celltype2 = {{ cell_type }}) %>%
    select(x1, y1, celltype1, !!as.symbol(paste0(id, ".x")),
           x2, y2, celltype2, !!as.symbol(paste0(id, ".y")),
           {{ region }})
  
  # Calculate distance and reorder columns
  annotated_result <- annotated_result%>%
    mutate(distance = sqrt((x2-x1)^2 + (y2-y1)^2)) %>%
    select(!!as.symbol(region), 
           !!as.symbol(paste0(id, ".x")), celltype1, x1, y1,
           !!as.symbol(paste0(id, ".y")), celltype2, x2, y2,
           distance)
  colnames(annotated_result) <- c(region, 
                                  "celltype1_index", "celltype1", "celltype1_X", "celltype1_Y",
                                  "celltype2_index", "celltype2", "celltype2_X", "celltype2_Y", 
                                  "distance")
  return(annotated_result)
}

get_triangulation_distances <- function(df_input, 
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
  
  # Select only necessary columns
  df_input <- df_input %>%
    select({{ id }},
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
  pb <- txtProgressBar(max = length(unique_regions), style = 3)
  progress <- function(n) setTxtProgressBar(pb, n)
  opts <- list(progress = progress)
  
  triangulation_distances <- foreach(reg_index = 1:length(unique_regions), 
                                     .packages = c("deldir", "tidyverse"), 
                                     .combine = "rbind", 
                                     .options.snow = opts)%dopar%{
    # SUBSET DATASET
    subset <- df_input %>%
      dplyr::filter(!!as.symbol(region) == unique_regions[reg_index]) %>%
      mutate(uniqueID = paste0(!!as.symbol(id), "-",
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
  
  return(triangulation_distances)
}

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
    select(!!as.symbol(id), 
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
  progress <- function(n) setTxtProgressBar(pb, n)
  opts <- list(progress = progress)
  
  
  iterative_triangulation_distances <- foreach(reg_index = 1:length(unique_regions)) %:%
    foreach(iteration_index = 1:num_iterations,
            .packages = c("deldir", "tidyverse"),
            .combine = "rbind",
            .options.snow = opts)%dopar%{
              subset <- df_input %>%
                dplyr::filter(!!as.symbol(region) == unique_regions[reg_index]) %>%
                mutate(uniqueID = paste0(!!as.symbol(id), "-",
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
                                                     num_cores = num_cores)
              
              per_cell_summary <- results %>%
                group_by(celltype1_index, celltype1, celltype2) %>%
                summarize(per_cell_mean_dist = mean(distance)) %>%
                ungroup()
              
              per_celltype_summary <- per_cell_summary %>%
                group_by(celltype1, celltype2) %>%
                summarize(mean_dist = mean(per_cell_mean_dist)) %>%
                ungroup() %>%
                mutate(region = unique_regions[reg_index],
                       iteration = iteration_index)
              colnames(per_celltype_summary) <- c("celltype1", "celltype2", "mean_dist", region, "iteration")
              
              return(per_celltype_summary)
            }
  
  iterative_triangulation_distances <- do.call(rbind, iterative_triangulation_distances)
  close(pb)
  stopCluster(cl)
  
  return(iterative_triangulation_distances)
}
