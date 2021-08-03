# Copyright Tom Collins, 31/8/2020
# Analysis for Alex's listening study.

# Here is a sample of cardinality scores when you test 16-beat
# samples from a selection of seven Classical string quartets
# against 16-beat samples from a selection of 64 string Classical
# string quartets.

# The purpose of this script is to test normality and compute a
# 95% confidence interval.
# x <- c(44.39, 40.1, 38.31, 39.43, 45.08,
#        42.32, 42.38, 45.57, 46.04, 42.69,
#        34.89, 42.31, 43.01, 45.07, 39.02,
#        52.57, 37.88,  48.63, 49.95, 34.43,
#        36.16, 36.13, 49.85, 35.2, 39.62,
#        44.97, 39.15, 45.8, 44.65, 36.17,
#        41.85, 41.46, 43.15, 44.35, 46.67,
#        44.71, 42.11, 40.06, 54.48, 38.64,
#        49.13, 36.28, 35.42, 39.52, 37.81,
#        40.31, 42.07, 39.16, 40.44, 43.34,
#        38.36, 43.39, 42.45, 45.28, 34.43,
#        42.39)
x <- c(60.89,40.1,38.31,39.43,45.08,42.32,42.38,45.57,46.04,42.69,34.89,53.55,43.01,45.07,39.02,52.57,37.88,55.51,48.63,49.95,56.74,36.16,36.13,49.85,35.2,44.97,39.15,45.8,44.65,36.17,41.85,41.46,43.15,44.35,46.67,44.71,42.11,40.06,54.48,38.64,49.13,36.28,35.42,39.52,37.81,40.31,42.07,39.16,40.44,43.34,58.15,38.36,43.39,42.45,45.28,34.43,61.61,39.62,42.39,42.31,52.51,44.39)
shapiro.test(x)
# Test for normality fails, due to the outlying 0.789 value.

# Took inspect the plot.
 library("ggpubr")
 ggdensity(x, main = "Density plot of cardinality scores", xlab = "Cardinality score")

# Bootstrap confidence interval
bstrap <- c()
for (i in 1:1000){
  # First take the sample
  bsample <- sample(x,length(x),replace=T)
  # now calculate the bootstrap estimate
  bestimate <- mean(bsample)
  bstrap <- c(bstrap,bestimate)
}
mean(bstrap)
quantile(bstrap, .025)
quantile(bstrap, .975)

# 95% confidence interval assuming normality
me <- qt(.975, length(x) - 1)*sd(x)/sqrt(length(x))
mean(x) - me
mean(x) + me
