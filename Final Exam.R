# Install necessary packages 
#install.packages("tidyverse")
#install.packages("caret")
#install.packages("rpart")
#install.packages("rpart.plot")
#install.packages("pROC")
#install.packages("e1071")
#install.packages("knitr")
#install.packages("kableExtra")

# Load the libraries
#library(tidyverse)
#library(caret)
#library(rpart)
#library(rpart.plot)
#library(pROC)
#library(e1071)
#library(knitr)
#library(kableExtra)

# Load the dataset
bank_data <- read.csv("C:/Users/nihar/OneDrive/Desktop/Bootcamp/SCMA 632/DataSet/bank-additional-full.csv", sep = ";")

# Check for missing values
missing_values <- colSums(is.na(bank_data))
missing_values

# Print missing values
print("Missing values in each column:")
print(missing_values)

# Convert the target variable to a factor
bank_data$y <- as.factor(bank_data$y)

# Convert categorical variables to factors
categorical_vars <- c("job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "poutcome")
bank_data[categorical_vars] <- lapply(bank_data[categorical_vars], as.factor)

# Split the data into training and test sets
set.seed(123)
train_index <- createDataPartition(bank_data$y, p = 0.7, list = FALSE)
train_data <- bank_data[train_index, ]
test_data <- bank_data[-train_index, ]

# Train the Logistic Regression Model
log_model <- glm(y ~ ., data = train_data, family = binomial)

# Predict on the test set
log_pred_prob <- predict(log_model, newdata = test_data, type = "response")
log_pred <- ifelse(log_pred_prob > 0.5, "yes", "no")
log_pred <- factor(log_pred, levels = c("no", "yes"))

# Train the Decision Tree Model
tree_model <- rpart(y ~ ., data = train_data, method = "class")

# Predict on the test set
tree_pred <- predict(tree_model, newdata = test_data, type = "class")

# Confusion Matrix and Metrics for Logistic Regression
conf_matrix_log <- confusionMatrix(log_pred, test_data$y)
log_accuracy <- conf_matrix_log$overall['Accuracy']
log_precision <- conf_matrix_log$byClass['Pos Pred Value']
log_recall <- conf_matrix_log$byClass['Sensitivity']
log_f1 <- 2 * ((log_precision * log_recall) / (log_precision + log_recall))

# Confusion Matrix and Metrics for Decision Tree
conf_matrix_tree <- confusionMatrix(tree_pred, test_data$y)
tree_accuracy <- conf_matrix_tree$overall['Accuracy']
tree_precision <- conf_matrix_tree$byClass['Pos Pred Value']
tree_recall <- conf_matrix_tree$byClass['Sensitivity']
tree_f1 <- 2 * ((tree_precision * tree_recall) / (tree_precision + tree_recall))

# AUC-ROC for Logistic Regression
log_roc <- roc(test_data$y, as.numeric(log_pred_prob))
log_auc <- auc(log_roc)

# AUC-ROC for Decision Tree
tree_pred_prob <- predict(tree_model, newdata = test_data, type = "prob")[,2]
tree_roc <- roc(test_data$y, as.numeric(tree_pred_prob))
tree_auc <- auc(tree_roc)

# Print metrics
metrics <- data.frame(
  Model = c("Logistic Regression", "Decision Tree"),
  Accuracy = c(log_accuracy, tree_accuracy),
  Precision = c(log_precision, tree_precision),
  Recall = c(log_recall, tree_recall),
  F1_Score = c(log_f1, tree_f1),
  AUC = c(log_auc, tree_auc)
)

print(metrics)

# Output metrics as a table
kable(metrics, format = "html") %>%
  kable_styling(full_width = F, bootstrap_options = c("striped", "hover", "condensed"))

# Plot Confusion Matrices
fourfoldplot(conf_matrix_log$table, color = c("#CC6666", "#99CC99"),
             conf.level = 0, margin = 1, main = "Logistic Regression Confusion Matrix")

fourfoldplot(conf_matrix_tree$table, color = c("#CC6666", "#99CC99"),
             conf.level = 0, margin = 1, main = "Decision Tree Confusion Matrix")

# Plot AUC-ROC Curves
plot(log_roc, main = "ROC Curve - Logistic Regression", col = "blue", lwd = 3)
plot(tree_roc, main = "ROC Curve - Decision Tree", col = "green", lwd = 3, add = TRUE)

legend("bottomright", legend = c("Logistic Regression", "Decision Tree"),
       col = c("blue", "green"), lwd = 3)

# Plot Decision Tree
rpart.plot(tree_model, main = "Decision Tree Structure", type = 4, extra = 101, box.palette = "RdYlGn")


# Logistic Regression Coefficients
log_coef <- summary(log_model)$coefficients
log_coef <- as.data.frame(log_coef)
kable(log_coef, format = "html") %>%
  kable_styling(full_width = F, bootstrap_options = c("striped", "hover", "condensed"))


