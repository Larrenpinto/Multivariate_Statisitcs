library(readxl)
library(dplyr)
data = read_xlsx("C:\\Users\\Larren\\Downloads\\Notes SEM 2\\Multivariate\\Online Retail.xlsx") 
View(data)
#---------------------------------------------------------------------------------
#                     Exploratory Data Analysis

# to check null values in the dataset 
print(sapply(data, function(x) sum(is.na(x))))

# Plotting the null values using heatmap
library(Amelia)
missmap(data, main="Missing Values Heatmap", col=c("yellow", "black"), legend=FALSE)

# to remove null values from the dataset
data = na.omit(data)

print(sapply(data, function(x) sum(is.na(x))))

library(Amelia)
missmap(data, main="Missing Values Heatmap", col=c("yellow", "black"), legend=FALSE)

# Summary of the data
summary(data)

# Convert InvoiceNo, CustomerID , Country to factor since it's categorical
data$InvoiceNo = as.factor(data$InvoiceNo)
data$CustomerID = as.factor(data$CustomerID)
data$Country = as.factor(data$Country)

library(ggplot2)
data %>%  count(Country, sort = TRUE) %>%  top_n(10) %>%
  ggplot(aes(x = reorder(Country, -n), y = n, fill = Country)) + 
  geom_bar(stat="identity") + geom_text(aes(label = n), vjust = -0.3, size = 4) +
  coord_flip() +theme_minimal() +
  labs(title="Top 10 Countries with Most Transactions", x="Country", y="Count")

# Selecting only numeric columns
data_numeric <- data %>% select(Quantity, UnitPrice)
#---------------------------------------------------------------------
#                   Principle Component Analysis
pca_result = prcomp(data_numeric, center = TRUE, scale. = TRUE)
summary(pca_result)

# Visualize PCA
library(factoextra)
fviz_pca_biplot(pca_result, label="var", repel=TRUE)

# -----------------------------------------------------------       
#                        K Means Clustering
library(ggplot2)
library(factoextra)
library(cluster)

#  Grouping data by CustomerID and summarize purchase behavior
data_grouped <- data %>% group_by(CustomerID) %>%
  summarise(
    TotalSpend = sum(Quantity * UnitPrice, na.rm = TRUE),
    TotalOrders = n(),  # Number of orders
    AvgPrice = mean(UnitPrice, na.rm = TRUE)
  ) %>% na.omit() 

#Scaling data fro clustering
data_scaled1 <- scale(data_grouped[, -1]) 

# Plotting using the Elbow Method to find number of clusters
fviz_nbclust(data_scaled1, kmeans, method = "wss")

set.seed(123)
kmeans_result <- kmeans(data_scaled1, centers = 4, nstart = 10)
print(kmeans_result)

data_grouped$Cluster <- as.factor(kmeans_result$cluster)

ggplot(data_grouped, aes(x = TotalOrders, y = TotalSpend, color = Cluster)) +
  geom_point(alpha = 0.7, size = 3) + theme_minimal() +
  labs(title = "Customer Clusters", x = "Total Orders", y = "Total Spend") +
  facet_wrap(~ Cluster)  # Create separate subplots for each cluster

# Acurracy testing
silhouette_score <- silhouette(kmeans_result$cluster, dist(data_scaled1))
avg_silhouette <- mean(silhouette_score[, 3])  # Average silhouette width
print(avg_silhouette)
# ---------------------------------------------------------------------
#                         K Near Neighbors
library(caret)

# Grouping data by CustomerID
data_grouped1 <- data %>% group_by(CustomerID) %>%
  summarise(
    TotalSpending = sum(UnitPrice * Quantity, na.rm = TRUE),
    AvgQuantity = mean(Quantity, na.rm = TRUE),
    AvgUnitPrice = mean(UnitPrice, na.rm = TRUE)
  )

# grouping data based on spending
data_grouped1$HighSpender <- as.factor(ifelse(data_grouped1$TotalSpending > 100, 1, 0))
set.seed(123) # Ensure reproducibility

# Spliting data 
trainIndex <- createDataPartition(data_grouped1$HighSpender, p = 0.8, list = FALSE)
train_data <- data_grouped1[trainIndex, ]
test_data <- data_grouped1[-trainIndex, ]

# Training KNN model 
knn_model <- train(HighSpender ~ TotalSpending + AvgQuantity + AvgUnitPrice, 
                   data = train_data, method = "knn",  tuneLength = 10)
print(knn_model)
# Makeing predictions
predictions <- predict(knn_model, test_data)

# Calculating Metrics
accuracy <- sum(predictions == test_data$HighSpender) / nrow(test_data)
conf_matrix <- confusionMatrix(predictions, as.factor(test_data$HighSpender))
precision <- conf_matrix$byClass["Precision"]
recall <- conf_matrix$byClass["Recall"]
f1_score <- conf_matrix$byClass["F1"]
cat(sprintf("Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1 Score: %.4f", 
            accuracy, precision, recall, f1_score))
# -------------------------------------------------------------------------
#                       Canonical Correlation

data$TotalPrice <- data$Quantity * data$UnitPrice
X <- data %>% select(Quantity, UnitPrice)
Y <- data %>% select(TotalPrice)

# Applying Canonical Correlation Analysis
cca_result <- cancor(X, Y)

# Printing metrics
cat(sprintf("Canonical Correlation: %.4f | X Coefficients: %.6f, %.6f | Y Coefficients: %.6f", 
            cca_result$cor[1], cca_result$xcoef[1,1], cca_result$xcoef[2,1], 
            cca_result$ycoef[1,1]))