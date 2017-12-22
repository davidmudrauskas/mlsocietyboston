library('plyr')

trainingData <- read.csv("~/Downloads/train.tsv", sep = "\t")
categoryHierarchies <- as.data.frame(table(trainingData$category_name))

# Initialize an empty data frame with enough columns
parsedCategoryHierarchies <- data.frame(matrix(ncol = 5, nrow = 0))
colnames(parsedCategoryHierarchies) <- c("Level 1", "Level 2", "Level 3", "Level 4", "Level 5")

# Separate a category and its subcategories into separate columns
for (hierarchy in categoryHierarchies$Var1) { 
  split <- rbind(data.frame(matrix(ncol = 5, nrow = 0)), strsplit(hierarchy, "/")[[1]])
  
  # Find out how many columns there are and name consistently
  columnNames <- colnames(split) 
  newNames <- c()
  i <- 1
  for (name in columnNames) {
    newNames <- c(newNames, paste("Level", i))
    i = i + 1
  }
  
  colnames(split) <- newNames
  parsedCategoryHierarchies <- rbind.fill(split, parsedCategoryHierarchies)
}

# One population, items without a category, is skipped in the 'for' loop. Adding back
parsedCategoryHierarchies <- rbind(c(""),parsedCategoryHierarchies)

# Sort raw summary table and parsed categories to align records
sortedParsedHierarchies <- parsedCategoryHierarchies[order(parsedCategoryHierarchies$`Level 1`, parsedCategoryHierarchies$`Level 2`, parsedCategoryHierarchies$`Level 3`, parsedCategoryHierarchies$`Level 4`, parsedCategoryHierarchies$`Level 5`),]
sortedHierarchies <- categoryHierarchies[order(categoryHierarchies$Var1),]

countBySubcategory <- cbind(sortedParsedHierarchies, sortedHierarchies$Freq)

# Get basic summary statistics for each set of categories/subcategories
statsByHierarchy <- data.frame()
statsByHierarchy <- aggregate(trainingData$price, list(trainingData$category_name), mean)
statsByHierarchy$median <- aggregate(trainingData$price, list(trainingData$category_name), median)$x
statsByHierarchy$standard_deviation <- aggregate(trainingData$price, list(trainingData$category_name), sd)$x
colnames(statsByHierarchy) <- c("Category hierarchy", "Mean", "Median", "Standard deviation")
# Make sure to sort so records align
statsByHierarchy <- statsByHierarchy[order(statsByHierarchy$`Category hierarchy`),]

# Append to parsed category hierarchy and frequency data
statsBySubcategory <- cbind(countBySubcategory, statsByHierarchy$Mean, statsByHierarchy$Median, statsByHierarchy$`Standard deviation`)
colnames(statsBySubcategory) <- c("Level 1", "Level 2", "Level 3", "Level 4", "Level 5", "Count", "Mean", "Median", "Standard deviation")

write.csv(statsBySubcategory, file = "statsBySubcategory.csv", row.names = FALSE)