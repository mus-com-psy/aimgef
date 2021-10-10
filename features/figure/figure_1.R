library("ggplot2")
library("gridExtra")
data <- read.csv("./MTGenerated.csv")

# xlim_left <- 0
# xlim_right <- 10
# breaks <- c(0, 0.01, 0.03, 0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30)

statComp <- ggplot(data, aes(factor(group), group = 1)) +
  labs(x = "Epochs", y = "Statistical Complexity") +
  geom_line(aes(y = as.numeric(statComp)), color = "blue") + geom_point(aes(y = as.numeric(statComp)), color = "blue")

transComp <- ggplot(data, aes(factor(group), group = 1)) +
  labs(x = "Epochs", y = "Translational Complexity") +
  geom_line(aes(y = as.numeric(transComp.Mean)), color = "blue") + geom_point(aes(y = as.numeric(transComp.Mean)), color = "blue") +
  geom_line(aes(y = as.numeric(transComp.Max)), color = "red") + geom_point(aes(y = as.numeric(transComp.Max)), color = "red") +
  geom_line(aes(y = as.numeric(transComp.Min)), color = "green") + geom_point(aes(y = as.numeric(transComp.Min)), color = "green") +
  geom_line(aes(y = as.numeric(transComp.Variance)), color = "purple") + geom_point(aes(y = as.numeric(transComp.Variance)), color = "purple")

arcScore <- ggplot(data, aes(factor(group), group = 1)) +
  labs(x = "Epochs", y = "Arc Score") +
  geom_line(aes(y = as.numeric(arcScore.Mean)), color = "blue") + geom_point(aes(y = as.numeric(arcScore.Mean)), color = "blue") +
  geom_line(aes(y = as.numeric(arcScore.Max)), color = "red") + geom_point(aes(y = as.numeric(arcScore.Max)), color = "red") +
  geom_line(aes(y = as.numeric(arcScore.Min)), color = "green") + geom_point(aes(y = as.numeric(arcScore.Min)), color = "green") +
  geom_line(aes(y = as.numeric(arcScore.Variance)), color = "purple") + geom_point(aes(y = as.numeric(arcScore.Variance)), color = "purple")

tonalAmb <- ggplot(data, aes(factor(group), group = 1)) +
  labs(x = "Epochs", y = "Tonal Ambiguity") +
  geom_line(aes(y = as.numeric(tonalAmb.Mean)), color = "blue") + geom_point(aes(y = as.numeric(tonalAmb.Mean)), color = "blue") +
  geom_line(aes(y = as.numeric(tonalAmb.Max)), color = "red") + geom_point(aes(y = as.numeric(tonalAmb.Max)), color = "red") +
  geom_line(aes(y = as.numeric(tonalAmb.Min)), color = "green") + geom_point(aes(y = as.numeric(tonalAmb.Min)), color = "green") +
  geom_line(aes(y = as.numeric(tonalAmb.Variance)), color = "purple") + geom_point(aes(y = as.numeric(tonalAmb.Variance)), color = "purple")

attInterval <- ggplot(data, aes(factor(group), group = 1)) +
  labs(x = "Epochs", y = "Average Time between Attacks") +
  geom_line(aes(y = as.numeric(attInterval.Mean)), color = "blue") + geom_point(aes(y = as.numeric(attInterval.Mean)), color = "blue") +
  geom_line(aes(y = as.numeric(attInterval.Max)), color = "red") + geom_point(aes(y = as.numeric(attInterval.Max)), color = "red") +
  geom_line(aes(y = as.numeric(attInterval.Min)), color = "green") + geom_point(aes(y = as.numeric(attInterval.Min)), color = "green") +
  geom_line(aes(y = as.numeric(attInterval.Variance)), color = "purple") + geom_point(aes(y = as.numeric(attInterval.Variance)), color = "purple")

rhyDis <- ggplot(data, aes(factor(group), group = 1)) +
  labs(x = "Epochs", y = "Average Jitter of Attacks") +
  geom_line(aes(y = as.numeric(rhyDis.Mean)), color = "blue") + geom_point(aes(y = as.numeric(rhyDis.Mean)), color = "blue") +
  geom_line(aes(y = as.numeric(rhyDis.Max)), color = "red") + geom_point(aes(y = as.numeric(rhyDis.Max)), color = "red") +
  geom_line(aes(y = as.numeric(rhyDis.Min)), color = "green") + geom_point(aes(y = as.numeric(rhyDis.Min)), color = "green") +
  geom_line(aes(y = as.numeric(rhyDis.Variance)), color = "purple") + geom_point(aes(y = as.numeric(rhyDis.Variance)), color = "purple")

# arcScore
grid.arrange(statComp, transComp, arcScore, tonalAmb, attInterval, rhyDis)
# attInterval

# ggarrange(bxp, dp, bp + rremove("x.text"),
#           labels = c("A", "B", "C"),
#           ncol = 2, nrow = 2)
