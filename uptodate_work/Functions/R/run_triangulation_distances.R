# run_triangulation_distances
run_triangulation_distances <- function(df,
                                        cell_index_column,
                                        x_position_column,
                                        y_position_column,
                                        cell_type_column,
                                        region_column,
                                        calc_avg_distance = TRUE,
                                        output_path = output_path,
                                        number_of_iterations = 100,
                                        num_cores = NULL) {
  df <- df %>% filter(!is.na(cell_type_column))
  # remove empty columns

  # calculate observed distances - true distances between cells
  triangulation_distances <- get_triangulation_distances(
    df_input = df,
    id = cell_index_column,
    x_pos = x_position_column,
    y_pos = y_position_column,
    cell_type = cell_type_column,
    region = region_column,
    calc_avg_distance = TRUE, # creates table with avg distances for a quick overview
    csv_output = output_path, # creates an easy to load summary of mean distances
    num_cores = NULL
  ) # If set to NULL the function uses half of the available threads (usually number of CPU cores. Set number manually if you know that you need more threads)

  head(triangulation_distances, n = 3) # show first 3 rows

  # save results as csv
  write.csv(triangulation_distances, paste0(output_path, "/", "triangulation_distances", ".csv"))

  # Iterations
  # In the iterated distances, distances are summarized per region for each iteration.
  # Note: you don't need to shuffle the cell annotations yourself, it's done in the iteration for you


  # calculate expected distances - shuffling simulates a statistically random distribution - no "unexpected" interactions
  iterated_triangulation_distances <- iterate_triangulation_distances(
    df_input = df,
    id = cell_index_column,
    x_pos = x_position_column,
    y_pos = y_position_column,
    cell_type = cell_type_column,
    region = region_column,
    num_iterations = number_of_iterations,
    num_cores = NULL
  ) # If set to NULL the function uses half of the available threads (usually number of CPU cores. Set number manually if you know that you need more threads)

  head(iterated_triangulation_distances, n = 3) # show first 3 rows

  # save results as csv
  write.csv(iterated_triangulation_distances, paste0(output_path, "/", "iterated_triangulation_distances", "_", as.character(number_of_iterations), ".csv"))
}
