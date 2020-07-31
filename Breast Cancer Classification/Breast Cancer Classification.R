#Load required packages
library(readr)
library(dplyr)
library(ggplot2)
library(corrplot)
library(DescTools)
library(caTools)
library(pROC)
library(class)
library(randomForest)

#Load the data file
raw_bc_data <- read_csv("Breast Cancer Data.csv")
head(raw_bc_data)

#Remove first column, turn diagnosis into a factor
bc_data <- raw_bc_data[,-1]
bc_data$diagnosis <- as.factor(bc_data$diagnosis)


########################################### Exploratory Analysis ##################################################

#Check for missing values
sapply(bc_data, function(x) sum(is.na(x)))

#Obtain summary statistics
summary(bc_data)

#Bar chart of benign vs. malignant
plot_1 <- ggplot(bc_data, aes(x = diagnosis, fill = diagnosis)) +
  geom_bar()
plot_1

#Examining correlations between factors
M <- cor(bc_data[, -1])
corrplot(M, method = "circle", tl.cex=0.7, order="hclust")

###################################### Logistic Regression #######################################################

#Break data into training and test sets
set.seed(42)
bc_data$spl <- sample.split(bc_data$diagnosis)
bc_data_train <- bc_data %>% filter(spl == "TRUE")
bc_data_test <- bc_data %>% filter(spl == "FALSE")

#Prepare training and test data for PCA. Use all rows but "diagnosis" in wisc_data, set "diagnosis" as a separate vector.
diagnosis_train <- bc_data_train$diagnosis
diagnosis_test <- bc_data_test$diagnosis
wisc_data_train <- bc_data_train[,-c(1,32)]
wisc_data_test <- bc_data_test[,-c(1,32)]

#Run PCA to reduce dimension of the training set and normalize features
wisc_pr_train <- prcomp(wisc_data_train, center = TRUE, scale = TRUE)
summary(wisc_pr_train)

# Scatter plot observations by components 1 and 2
plot(wisc_pr_train$x[, c(1, 2)], col = (diagnosis_train), 
     xlab = "PC1", ylab = "PC2")

# Repeat for components 1 and 3
plot(wisc_pr_train$x[, c(1, 3)], col = (diagnosis_train), 
     xlab = "PC1", ylab = "PC3")

#Getting proportion of variance for scree plot
pr_var <- wisc_pr_train$sdev^2
pve <- pr_var / sum(pr_var)

#Scree plot
par(pty = "m")
plot(pve, xlab = "Principal Component", 
     ylab = "Proportion of Variance Explained", 
     ylim = c(0, 1), type = "o")

# Plot cumulative proportion of variance explained
plot(cumsum(pve), xlab = "Principal Component", 
     ylab = "Cumulative Proportion of Variance Explained", 
     ylim = c(0, 1), type = "o")
#We wil use the first 7 components, because they represent 90% of the variance in the data. This is not the only
#possible choice!

#Transform the test data using the PCA function from training data
wisc_pr_test <- predict(wisc_pr_train, wisc_data_test)
wisc_pr_test <- as_tibble(wisc_pr_test) %>% select(PC1:PC7)
wisc_pr_train <- as_tibble(wisc_pr_train$x) %>% select(PC1:PC7)

#Combine diagnosis vector to principal components matrix
wisc_pr_train$diagnosis <- diagnosis_train
wisc_pr_test$diagnosis <- diagnosis_test

#Logistic regression with remaining variables
summary(log_model <- glm(diagnosis ~ PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7, family = "binomial", data = wisc_pr_train))

#Computing accuracy on the test set
predicted <- predict(log_model, wisc_pr_test[,1:7], type = "response")
predicted <- ifelse(predicted > 0.5, 1, 0)
d <- table(predicted, wisc_pr_test$diagnosis)
d
sum(diag(d))/sum(d)

#Plot ROC curve
par(pty = "s")
roc(wisc_pr_test$diagnosis, predicted, plot = TRUE, legacy.axes = TRUE, percent = TRUE,
    xlab = "False Positive Percentage", ylab = "True Positive Percentage", col = "#377eb8", lwd = 4,
    print.auc = TRUE)

################################## kNN Classification #############################################################

#Run the kNN model
set.seed(42)
wisc_knn <- knn(wisc_pr_train[,1:7], wisc_pr_test[,1:7], cl = diagnosis_train, k = 19)
f <- table(wisc_knn, wisc_pr_test$diagnosis)
f
sum(diag(f))/sum(f)

#Plot ROC curve
par(pty = "s")
roc(wisc_pr_test$diagnosis, as.numeric(wisc_knn), plot = TRUE, legacy.axes = TRUE, percent = TRUE,
    xlab = "False Positive Percentage", ylab = "True Positive Percentage", col = "#4daf4a", lwd = 4,
    print.auc = TRUE)

##################################### Random Forest ###############################################################

#Run the Random Forest model
set.seed(42)
wisc_pr_rf <- randomForest(diagnosis ~ PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7, data = wisc_pr_train, ntree = 300)
wisc_pr_rf

#Make predictions on the test set
rf_predict <- predict(wisc_pr_rf, wisc_pr_test[,1:7], type = "response")
g <- table(rf_predict, wisc_pr_test$diagnosis)
g
sum(diag(g))/sum(g)

#Plot ROC curve
par(pty = "s")
roc(wisc_pr_test$diagnosis, as.numeric(rf_predict), plot = TRUE, legacy.axes = TRUE, percent = TRUE,
    xlab = "False Positive Percentage", ylab = "True Positive Percentage", col = "#e34a33", lwd = 4,
    print.auc = TRUE)

################################## Comparing Classifiers ##########################################################

#Plot ROC curve for logistic regression model
#par(pty = "s")
roc(wisc_pr_test$diagnosis, predicted, plot = TRUE, legacy.axes = TRUE, percent = TRUE,
    xlab = "False Positive Percentage", ylab = "True Positive Percentage", col = "#377eb8", lwd = 4,
    print.auc = TRUE, print.auc.y = 60, print.auc.x = 30)

#Plot ROC curve for kNN classifier
plot.roc(wisc_pr_test$diagnosis, as.numeric(wisc_knn), percent = TRUE, col = "#4daf4a", lwd = 4,
    print.auc = TRUE, add = TRUE, print.auc.y = 50, print.auc.x = 30)

#Plot ROC curve for RF
plot.roc(wisc_pr_test$diagnosis, as.numeric(rf_predict), percent = TRUE, col = "#e34a33", lwd = 4,
    print.auc = TRUE, add = TRUE, print.auc.y = 40, print.auc.x = 30)

#Add legend to plot
legend("bottomright", legend = c("Logistic Regression","kNN","Random Forest"), col = c("#377eb8","#4daf4a","#e34a33"),
       lwd = 4)