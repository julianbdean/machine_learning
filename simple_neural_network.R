# install.packages(c('neuralnet','keras','tensorflow'),dependencies = T)

library(tidyverse)
library(neuralnet)

iris <- iris %>% mutate_if(is.character, as.factor)

set.seed(245)
data_rows <- floor(0.80 * nrow(iris))
train_indices <- sample(c(1:nrow(iris)), data_rows)
train_data <- iris[train_indices,]
test_data <- iris[-train_indices,]


model = neuralnet(
  Species~Sepal.Length+Sepal.Width+Petal.Length+Petal.Width,
  data=train_data,
  hidden=c(4,2),
  linear.output = FALSE
)


plot(model,rep = "best")


pred <- predict(model, test_data)


labels <- c("setosa", "versicolor", "virginca")




prediction_label <- data.frame(max.col(pred)) %>%     
  mutate(pred=labels[max.col.pred.]) %>%
  dplyr::select(2) %>%
  unlist()

table(test_data$Species, prediction_label)



check = as.numeric(test_data$Species) == max.col(pred)
accuracy = (sum(check)/nrow(test_data))*100
print(accuracy)






###### Comparison to OLS

library(tidyverse)
library(neuralnet)
library(nnet)

iris <- iris %>% mutate_if(is.character, as.factor)

set.seed(245)
data_rows <- floor(0.80 * nrow(iris))
train_indices <- sample(c(1:nrow(iris)), data_rows)
train_data <- iris[train_indices,]
test_data <- iris[-train_indices,]

# Neural Network
model <- neuralnet(
  Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
  data = train_data,
  hidden = c(4, 2),
  linear.output = FALSE
)

plot(model, rep = "best")

pred <- predict(model, test_data)
labels <- c("setosa", "versicolor", "virginica")
prediction_label <- data.frame(max.col(pred)) %>%     
  mutate(pred = labels[max.col(pred)]) %>%
  dplyr::select(2) %>%
  unlist()

table(test_data$Species, prediction_label)

check <- as.numeric(test_data$Species) == max.col(pred)
accuracy <- (sum(check) / nrow(test_data)) * 100
print(accuracy)

# Logistic Regression
model_logistic <- multinom(Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, data = train_data)
pred_logistic <- predict(model_logistic, newdata = test_data, type = "class")
table(test_data$Species, pred_logistic)
accuracy_logistic <- sum(test_data$Species == pred_logistic) / nrow(test_data) * 100
print(accuracy_logistic)


# Compare Neural Network and Logistic Regression accuracies
if (accuracy > accuracy_logistic) {
  message("Neural Network has higher accuracy.")
} else if (accuracy < accuracy_logistic) {
  message("Logistic Regression has higher accuracy.")
} else {
  message("Neural Network and Logistic Regression have the same accuracy.")
}




################################################
library(tidyverse)
library(neuralnet)
library(nnet)
library(pROC)

iris <- iris %>% mutate_if(is.character, as.factor)

set.seed(245)
data_rows <- floor(0.80 * nrow(iris))
train_indices <- sample(c(1:nrow(iris)), data_rows)
train_data <- iris[train_indices,]
test_data <- iris[-train_indices,]

# Neural Network
model <- neuralnet(
  Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
  data = train_data,
  hidden = c(4, 2),
  linear.output = FALSE
)

plot(model, rep = "best")

pred <- predict(model, test_data)
labels <- c("setosa", "versicolor", "virginica")
prediction_label <- data.frame(max.col(pred)) %>%     
  mutate(pred = labels[max.col(pred)]) %>%
  dplyr::select(2) %>%
  unlist()

# Compute confusion matrix
confusion_matrix_nn <- table(test_data$Species, prediction_label)

# Compute evaluation metrics for Neural Network
accuracy_nn <- sum(diag(confusion_matrix_nn)) / sum(confusion_matrix_nn)
precision_nn <- diag(confusion_matrix_nn) / colSums(confusion_matrix_nn)
recall_nn <- diag(confusion_matrix_nn) / rowSums(confusion_matrix_nn)
f1_score_nn <- 2 * precision_nn * recall_nn / (precision_nn + recall_nn)

# Convert predicted probabilities to matrix format
pred_matrix <- matrix(pred, nrow = nrow(test_data), dimnames = list(NULL, labels))

# Compute multiclass ROC curve and AUC for Neural Network
roc_nn <- multiclass.roc(test_data$Species, pred_matrix)
auc_nn <- unlist(roc_nn$auc)

# Print evaluation metrics for Neural Network
cat("Neural Network Evaluation Metrics:\n")
cat(paste0("Accuracy: ", accuracy_nn, "\n"))
cat(paste0("Precision: ", precision_nn, "\n"))
cat(paste0("Recall: ", recall_nn, "\n"))
cat(paste0("F1 Score: ", f1_score_nn, "\n"))
cat(paste0("Average AUC: ", mean(auc_nn), "\n\n"))



# Logistic Regression
model_logistic <- multinom(Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, data = train_data)
pred_logistic <- predict(model_logistic, newdata = test_data, type = "class")

# Compute confusion matrix
confusion_matrix_logistic <- table(test_data$Species, pred_logistic)

# Compute evaluation metrics for Logistic Regression
accuracy_logistic <- sum(diag(confusion_matrix_logistic)) / sum(confusion_matrix_logistic)
precision_logistic <- diag(confusion_matrix_logistic) / colSums(confusion_matrix_logistic)
recall_logistic <- diag(confusion_matrix_logistic) / rowSums(confusion_matrix_logistic)
f1_score_logistic <- 2 * precision_logistic * recall_logistic / (precision_logistic + recall_logistic)

# Compute ROC curve and AUC for Logistic Regression
roc_logistic <- multiclass.roc(test_data$Species, as.numeric(pred_logistic == labels))
auc_logistic <- unlist(roc_logistic$auc)

# Print evaluation metrics for Logistic Regression
cat("Logistic Regression Evaluation Metrics:\n")
cat(paste0("Accuracy: ", accuracy_logistic, "\n"))
cat(paste0("Precision: ", precision_logistic, "\n"))
cat(paste0("Recall: ", recall_logistic, "\n"))
cat(paste0("F1 Score: ", f1_score_logistic, "\n"))
cat(paste0("Average AUC: ", mean(auc_logistic), "\n\n"))

# Compare Neural Network and Logistic Regression
cat("Comparison:\n")
cat(paste0("Neural Network Average AUC: ", mean(auc_nn), "\n"))
cat(paste0("Logistic Regression Average AUC: ", mean(auc_logistic), "\n"))
