# Load necessary libraries
library(tidyverse)
library(MASS) # For synthetic data generation


# Generate synthetic data
set.seed(42) # For reproducibility
data <- as.data.frame(mvrnorm(n = 100, mu = c(0,0,0), Sigma = matrix(c(1,0.5,0.2,0.5,1,0.3,0.2,0.3,1), nrow = 3)))
names(data) <- c("Feature1", "Feature2", "Feature3")
data$Score <- with(data, 3 * Feature1 + 2 * Feature2 - Feature3 + rnorm(100, mean = 0, sd = 0.3))

# Split data into training and testing sets
set.seed(42)
train_indices <- sample(1:nrow(data), 0.8 * nrow(data))
train_data <- data[train_indices, ]
test_data <- data[-train_indices, ]

# Train a linear model
model <- lm(Score ~ ., data = train_data)

# Predict scores on the test set
predictions <- predict(model, test_data)

# Add predictions to the test data
test_data$Predicted_Score <- predictions

# Rank the test data by predicted score
test_data <- test_data %>%
  arrange(desc(Predicted_Score)) %>%
  mutate(Rank = row_number())

# Print top 5 ranked observations
head(test_data, 5)
