


df <- data.frame(y = rnorm(300),
                 x = rnorm(300),
                 groups)

df <- data.frame(y = c(rnorm(30, mean = 1), rnorm(30, mean = 3), rnorm(30, mean = 5)),
                 x = rnorm(90),
                 groups)

ggplot(df, aes(x, y)) +
  
  geom_delaunay_tile(alpha = 0.05) + 
  geom_delaunay_segment2(size = 0.4,
                         lineend = 'round') +
  geom_point(aes(color = groups), size = 5) +
  scale_color_manual(values = c('orange', '#ff5964', '#38618c'), name = "Cell type") +
  theme_void() 






randome_g <- sample(groups)
df$random1 <- randome_g

ggplot(df, aes(x, y)) +
  
  geom_delaunay_tile(alpha = 0.05) + 
  geom_delaunay_segment2(size = 0.4,
                         lineend = 'round') +
  geom_point(aes(color = random1), size = 5) +
  scale_color_manual(values = c('orange', '#ff5964', '#38618c'), name = "Cell type") +
  theme_void()  


groups <- c("A", "A", "A","A", "A", "A","A", "A", "A","A",
            "A", "A", "A","A", "A", "A","A", "A", "A","A",
            "A", "A", "A","A", "A", "A","A", "A", "A","A",
          
            
            "B", "B", "B","B", "B", "B","B", "B", "B","B",
            "B", "B", "B","B", "B", "B","B", "B", "B","B",
            "B", "B", "B","B", "B", "B","B", "B", "B","B",
           
            
            "C", "C", "C","C", "C", "C","C", "C", "C","C",
            "C", "C", "C","C", "C", "C","C", "C", "C","C",
            "C", "C", "C","C", "C", "C","C", "C", "C","C")


# visulize distribution 
require(fitdistrplus)

observed_distances_Bcell_CD4T <- observed_distances %>% filter(celltype1 == "DC") %>%  filter(celltype2 == "B cell") %>%  filter(treatment == 3)
Distribution_obs <- as.data.frame(observed_distances_Bcell_CD4T$observed)
colnames(Distribution_obs) <- "observed"

ggplot(Distribution_obs) +
  aes(x = observed) +
  geom_density(adjust = 1L, fill = "#CECECE") +
  theme_void() +
  xlim(-10, 75) + 
  geom_vline(xintercept =  observed_distances_Bcell_CD4T$observed_mean, linetype = "dashed")

ggplot(Distribution_obs) +
  aes(x = observed) +
  geom_histogram(bins = 10, fill = "darkgrey") +
  theme_void() +
  xlim(-10, 75) + geom_vline(xintercept =  observed_distances_Bcell_CD4T$observed_mean, linetype = "dashed")

obs_vec <- unlist(observed_distances_Bcell_CD4T$observed)

descdist(obs_vec, discrete = FALSE)

shuffled_distances_Bcell_CD4T <- expected_distances %>% filter(celltype1 == "DC") %>%  filter(celltype2 == "B cell") %>%  filter(treatment == 3)
Distribution_shuff <- as.data.frame(shuffled_distances_Bcell_CD4T$expected)
colnames(Distribution_shuff) <- "shuffled"

ggplot(Distribution_shuff) +
  aes(x = shuffled) +
  geom_density(adjust = 1L, fill = "#CECECE") +
  theme_void() +
  xlim(-10, 75) + 
  geom_vline(xintercept =  shuffled_distances_Bcell_CD4T$expected_mean, linetype = "dashed")

ggplot(Distribution_shuff) +
  aes(x = shuffled) +
  geom_histogram(bins = 10, fill = "darkgrey") +
  theme_minimal() +
  xlim(-10, 75) + geom_vline(xintercept =  observed_distances_Bcell_CD4T$observed_mean, linetype = "dashed")

shuff_vec <- unlist(shuffled_distances_Bcell_CD4T$expected)

descdist(shuff_vec, discrete = FALSE)

# combine histograms
mydiff <- function(data, diff){
  c(diff(data, lag = diff), rep(NA, diff))
}

c(unlist(Distribution_obs$observed),rep(NA, 63) )

Distribution_comb <- cbind(c(unlist(Distribution_obs$observed),rep(NA, 63) ), Distribution_shuff$shuffled)

ggplot(Distribution_comb) +
  aes(x = observed) +
  geom_density(adjust = 1L, fill = "#CECECE") +
  theme_void() +
  xlim(-10, 75) + 
  geom_vline(xintercept =  observed_distances_Bcell_CD4T$observed_mean, linetype = "dashed")

ggplot(Distribution_shuff) +
  aes(x = shuffled) +
  geom_density(adjust = 1L, fill = "black", alpha = 0.8) +
  geom_density(data = Distribution_obs, aes(x = observed), adjust = 1L, fill = "lightgrey", alpha = 0.8) +
  theme_void() +
  xlim(-10, 75) + 
  geom_vline(xintercept =  shuffled_distances_Bcell_CD4T$expected_mean, linetype = "dashed", color = "#56ebd3") + 
  geom_vline(xintercept =  observed_distances_Bcell_CD4T$observed_mean, linetype = "dashed", color = "#116966")
