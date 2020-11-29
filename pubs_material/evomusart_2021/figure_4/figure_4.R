library("ggpubr")

loss <- read.csv("./loss.csv")
loss <- ggline(loss, x = "Step", y = "Value", linetype = "Split", xlab = "Epoch", ylab = "Loss")
loss <- ggpar(loss, xticks.by = 1, legend.title = "", font.legend = c(12, "plain", "black")) +
  scale_linetype_manual("", values = c("solid", "longdash"), labels = c("Train", "Validation"))
acc <- read.csv("./acc.csv")
acc <- ggline(acc, x = "Step", y = "Value", linetype = "Split", xlab = "Epoch", ylab = "Accuracy (%)")
acc <- ggpar(acc, xticks.by = 1, legend.title = "", font.legend = c(12, "plain", "black")) +
  scale_linetype_manual("", values = c("solid", "longdash"), labels = c("Train", "Validation"))

os <- read.csv("./os_wo-16-8.csv")

os_mean <- ggline(os, x = "Epoch", y = "Mean", linetype = "Split", xlab = "Epoch", ylab = "Originality Score")
os_mean <- ggpar(os_mean, xticks.by = 1, ylim = c(0, 1)) +
  geom_hline(aes(yintercept = 1 - 0.3009752, linetype = "solid")) +
  geom_hline(aes(yintercept = 1 - 0.2749859, linetype = "dashed")) +
  geom_hline(yintercept = 1 - 0.3276547, linetype = "dotted") +
  scale_linetype_manual("", values = c("dotted", "twodash", "solid", "longdash"), labels = c("Baseline .95-CI", "Baseline Mean", "64-target", "7-target"))

os_min <- ggline(os, x = "Epoch", y = "Min", linetype = "Split", xlab = "Epoch", ylab = "Originality Score")
os_min <- ggpar(os_min, xticks.by = 1, ylim = c(0, 1)) +
  geom_hline(aes(yintercept = 1 - 0.3009752, linetype = "solid")) +
  geom_hline(aes(yintercept = 1 - 0.2749859, linetype = "dashed")) +
  geom_hline(yintercept = 1 - 0.3276547, linetype = "dotted") +
  scale_linetype_manual("", values = c("dotted", "twodash", "solid", "longdash"), labels = c("Baseline .95-CI", "Baseline Mean", "64-target", "7-target"))

ggarrange(loss, acc, os_mean, os_min, hjust = -0.3, common.legend = FALSE, labels = c("a", "b", "c", "d")) %>%
  ggexport(width = 3200, height = 2000, res = 300, filename = "./originality_decreases.png")
