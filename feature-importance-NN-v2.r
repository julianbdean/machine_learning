# Install and load necessary packages
# install.packages("neuralnet")
library(neuralnet)

# Load the mtcars dataset
data(mtcars)

# Normalize the dataset
normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
mtcars_norm <- as.data.frame(lapply(mtcars, normalize))

# Specify a formula for our target (mpg) and all other columns as predictors
formula <- as.formula(paste("mpg ~", paste(names(mtcars)[!(names(mtcars) %in% "mpg")], collapse = " + ")))

# Train a neural network
set.seed(123) # For reproducibility
nn <- neuralnet(formula, 
                data = mtcars_norm, 
                hidden = 10, 
                linear.output = TRUE,
                threshold = 0.01)

# Plot the network (optional)
plot(nn)

# Predict using the model
predictions <- compute(nn, mtcars_norm[,-1])$net.result

# Convert predictions back to the original scale (optional)
predictions_orig_scale <- predictions * (max(mtcars$mpg) - min(mtcars$mpg)) + min(mtcars$mpg)


#### feature importance


# Create a function to calculate mean squared error
mse <- function(actual, predicted) {
  return(mean((actual - predicted)^2))
}

# Calculate the MSE of the original model on the data
original_mse <- mse(mtcars_norm$mpg, predictions)

# Initialize a vector to store feature importances
feature_importances <- numeric(ncol(mtcars_norm) - 1)

# For each feature
for (i in 2:ncol(mtcars_norm)) {  # Start from 2 to skip the target variable 'mpg'
  # Make a copy of the data
  temp_data <- mtcars_norm
  # Shuffle one column
  temp_data[, i] <- sample(temp_data[, i])
  # Get predictions for the shuffled data
  shuffled_preds <- compute(nn, temp_data[,-1])$net.result
  # Calculate the MSE for the shuffled data
  shuffled_mse <- mse(temp_data$mpg, shuffled_preds)
  # The importance of the feature is the difference in MSE
  feature_importances[i-1] <- shuffled_mse - original_mse
}

# Display feature importances
names(feature_importances) <- names(mtcars_norm)[2:ncol(mtcars_norm)]
feature_importances <- sort(feature_importances, decreasing = TRUE)
print(feature_importances)


#### direction of importance

# Let's probe the 'hp' feature (horsepower) in the mtcars dataset:

# Create a sequence of values for 'hp' spanning its range
probe_values <- seq(min(mtcars_norm$hp), max(mtcars_norm$hp), length.out = 100)

# Store predictions for each probe value
probe_predictions <- numeric(length(probe_values))

# For each probe value:
for (i in 1:length(probe_values)) {
  temp_data <- mtcars_norm
  temp_data$hp <- probe_values[i]
  probe_predictions[i] <- mean(compute(nn, temp_data[,-1])$net.result)
}

# Now, visualize the relationship
plot(probe_values, probe_predictions, type = "l",
     xlab = "Normalized Horsepower", ylab = "Predicted MPG", 
     main = "Effect of Horsepower on Predicted MPG")



