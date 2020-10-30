library("ggpubr")

loss <- read.csv("./loss.csv")
loss <- ggline(loss, x = "Step", y = "Value", linetype = "Split", shape = "Split", xlab = "Epoch", ylab = "Loss")
loss <- ggpar(loss, xticks.by = 1, legend.title = "", font.legend = c(12, "plain", "black"))
acc <- read.csv("./acc.csv")
acc <- ggline(acc, x = "Step", y = "Value", linetype = "Split", shape = "Split", xlab = "Epoch", ylab = "Accuracy (%)")
acc <- ggpar(acc, xticks.by = 1, legend.title = "", font.legend = c(12, "plain", "black"))

os <- read.csv("./os_wo-16-8.csv")
#os_mean <- colMeans(os, na.rm = TRUE)
#os_mean <- data.frame(epoch = 0:9, value = os_train_mean)
os_mean <- ggline(os, x = "Epoch", y = "Mean", linetype = "Split", shape = "Split", xlab = "Epoch", ylab = "Originality Score")
os_mean <- ggpar(os_mean, xticks.by = 1, ylim = c(0, 1)) +
  geom_hline(yintercept = 1 - 0.3009752, linetype = "solid") +
  geom_hline(yintercept = 1 - 0.2749859, linetype = "dashed") +
  geom_hline(yintercept = 1 - 0.3276547, linetype = "dashed")
#os_train_min <- sapply(os_train, my_min)
#os_train_min <- data.frame(epoch = 0:9, value = os_train_min)
os_min <- ggline(os, x = "Epoch", y = "Min", linetype = "Split", shape = "Split", xlab = "Epoch", ylab = "Originality Score")
os_min <- ggpar(os_min, xticks.by = 1, ylim = c(0, 1)) +
  geom_hline(yintercept = 1 - 0.3009752, linetype = "solid") +
  geom_hline(yintercept = 1 - 0.2749859, linetype = "dashed") +
  geom_hline(yintercept = 1 - 0.3276547, linetype = "dashed")
ggarrange(loss, acc, os_mean, os_min, hjust = -0.3, common.legend = TRUE, labels = c("a", "b", "c", "d")) %>%
  ggexport(width = 3200, height = 2000, res = 300, filename = "./originality_decreases.png")

os_maia <- read.csv("./os_maia.csv")
os_maia <- ggline(os_maia, x = "Epoch", y = "Min", xlab = "16-Beat", ylab = "Originality Score")
os_maia <- ggpar(os_maia, xticks.by = 1, ylim = c(0, 1)) +
  geom_hline(yintercept = 1 - 0.3009752, linetype = "solid") +
  geom_hline(yintercept = 1 - 0.2749859, linetype = "dashed") +
  geom_hline(yintercept = 1 - 0.3276547, linetype = "dashed")
ggarrange(os_maia) %>%
  ggexport(width = 3200, height = 2000, res = 300, filename = "./originality_maia.png")