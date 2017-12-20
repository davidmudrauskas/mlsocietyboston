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

write.csv(parsedCategoryHierarchies, file = "parsedCategoryHierarchies.csv", row.names = FALSE)
