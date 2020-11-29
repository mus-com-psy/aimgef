# Copyright Tom Collins, 31/8/2020
# Analysis for Alex's listening study.

# Here is a sample of cardinality scores when you test 16-beat
# samples from a selection of seven Classical string quartets
# against 16-beat samples from a selection of 64 string Classical
# string quartets.

# The purpose of this script is to test normality and compute a
# 95% confidence interval.
x <- c(0.29508196721311475, 0.26666666666666666,  0.3333333333333333,
   0.7894736842105263, 0.21774193548387097, 0.32051282051282054,
  0.21367521367521367,                0.36, 0.26262626262626265,
             0.359375,  0.2857142857142857, 0.32075471698113206,
   0.4222222222222222,  0.3111111111111111, 0.25925925925925924,
  0.26666666666666666, 0.26119402985074625,  0.2358490566037736,
  0.34328358208955223, 0.25609756097560976,  0.2833333333333333,
   0.3880597014925373,  0.2761904761904762, 0.31645569620253167,
   0.4491525423728814, 0.22413793103448276,  0.2692307692307692,
  0.25773195876288657,  0.3611111111111111,  0.3076923076923077,
  0.24210526315789474, 0.20408163265306123,  0.3486238532110092,
  0.20512820512820512,                 0.3,  0.3111111111111111,
   0.2440944881889764,   0.211864406779661,              0.2875,
  0.25316455696202533, 0.23076923076923078, 0.31521739130434784,
  0.22826086956521738, 0.20212765957446807, 0.26851851851851855,
  0.39622641509433965,  0.3333333333333333, 0.36363636363636365,
  0.22556390977443608, 0.38095238095238093)
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
