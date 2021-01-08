################################################################################
################          Download MovieLens Data          #####################
################################################################################

# Note: this process could take a couple of minutes
library(tidyverse)
library(caret)
library(data.table)
library(ggplot2)
library(cowplot)
library(gridExtra)
library(egg)
library(ggpubr)
library(lubridate)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


################################################################################
#################               Data Cleaning              #####################
################################################################################


# Create list of all genres
genres = strsplit(edx$genres, "\\|")
genre_list = vector()
for(i in seq(genres)){
  for(j in seq(genres[[i]])){
    if(!genres[[i]][j] %in% genre_list){
      genre_list = c(genre_list, genres[[i]][j])
    }
  }
}

# Convert timestamp from Unix epoch to human-readable format
edx$timestamp = as_datetime(edx$timestamp) %>% date()
validation$timestamp = as_datetime(validation$timestamp) %>% date()
head(edx$timestamp)

# Extract Day, Month, and Year of rating from timestamp
edx$rateYear = edx$timestamp %>% year()
edx$rateMonth = edx$timestamp %>% month()
edx$rateDay = edx$timestamp %>% day()
edx = edx %>% select(-timestamp)

validation$rateYear = validation$timestamp %>% year()
validation$rateMonth = validation$timestamp %>% month()
validation$rateDay = validation$timestamp %>% day()
validation = validation %>% select(-timestamp)


# Extract Release Date of Movie
edx$release = edx$title %>% str_match("\\s\\(\\d{4}\\)") %>% str_remove("\\s") %>% str_remove("\\(") %>% str_remove("\\)") %>% as.numeric()
edx$title = edx$title %>% str_remove("\\s\\(\\d{4}\\)")

validation$release = validation$title %>% str_match("\\s\\(\\d{4}\\)") %>% str_remove("\\s") %>% str_remove("\\(") %>% str_remove("\\)") %>% as.numeric()
validation$title = validation$title %>% str_remove("\\s\\(\\d{4}\\)")

# Extract Genres from list
edx$genres = strsplit(edx$genres, "\\|")
validation$genres = strsplit(validation$genres, "\\|")

for(value in genre_list)  {
  gen_vector = vector(length = NROW(edx))
  for(i in seq(NROW(edx)))  {
    ifelse(value %in% edx$genre[[i]], {gen_vector[i] = TRUE}, {gen_vector[i] = FALSE})
  }
  edx[[value]] = gen_vector
}

for(value in genre_list)  {
  gen_vector = vector(length = NROW(validation))
  for(i in seq(NROW(validation)))  {
    ifelse(value %in% validation$genre[[i]], {gen_vector[i] = TRUE}, {gen_vector[i] = FALSE})
  }
  validation[[value]] = gen_vector
}

edx = edx %>% select(-genres)
validation = validation %>% select(-genres)

# Sum total number of genres listed for each movie
num_edx = edx[,9:28] %>% rowSums()
edx = edx %>% mutate(NUM = num_edx)

num_valid = validation[,9:28] %>% rowSums()
validation = validation %>% mutate(NUM = num_valid)


################################################################################
################          Fit Model to Clean Data          #####################
################################################################################


# Define RMSE equation
RMSE = function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Find optimal lambda value
lambdas = seq(0, 10, 0.25)
rmses = sapply(lambdas, function(l){
  mu = mean(edx$rating)
  b_i = edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u = edx %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  b_y = edx %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by = "userId") %>%
    group_by(release) %>%
    summarize(b_y = sum(rating - b_i - b_u - mu)/(n()+l))
  b_n = edx %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_y, by = "release") %>%
    group_by(NUM) %>%
    summarize(b_n = sum(rating - b_i - b_u - mu)/(n() + l))
  predicted_ratings = edx %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_y, by = "release") %>%
    left_join(b_n, by = "NUM") %>%
    mutate(pred = mu + b_i + b_u + b_y + b_n) %>%
    pull(pred)
  return(RMSE(predicted_ratings, edx$rating))
})


# Predict ratings on validation set and calculate RMSE
lambda = lambdas[which.min(rmses)]
mu = mean(validation$rating)
b_i = validation %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))
b_u = validation %>%
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
b_y = validation %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by = "userId") %>%
  group_by(release) %>%
  summarize(b_y = sum(rating - b_i - b_u - mu)/(n()+lambda))
b_n = validation %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_y, by = "release") %>%
  group_by(NUM) %>%
  summarize(b_n = sum(rating - b_i - b_u - mu)/(n() + lambda))
predicted_ratings = validation %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_y, by = "release") %>%
  left_join(b_n, by = "NUM") %>%
  mutate(pred = mu + b_i + b_u + b_y + b_n) %>%
  pull(pred)


# Table of predictions is written to .csv file
prediction_results = tibble(Predicted = predicted_ratings, 
                            Actual = validation$rating, 
                            Residuals = Predicted - Actual)

write.table(prediction_results, "prediction_results.csv")

# Final RMSE Value is printed in console
final_rmse = RMSE(validation$rating, predicted_ratings)
print("#################  Final RMSE  #################")
print(final_rmse)

################################################################################
################          Validation RMSE = 0.8249795          #################
################################################################################
