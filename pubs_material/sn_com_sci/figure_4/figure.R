library("ggpubr")

binS_0.1 <- read.csv("./simBinScale-binS_0.1.csv_cleaned.csv")
binS_0.1 <- ggerrorplot(binS_0.1, x="Scale", y="Value", desc_stat = "mean_ci", error.plot = "errorbar",
            color = "black", add = "jitter", add.params = list(color = "darkgray"),
            xlab = "Scale", ylab = "Similarity", ylim = c(0, 1))

binS_0.5 <- read.csv("./simBinScale-binS_0.5.csv_cleaned.csv")
binS_0.5 <- ggerrorplot(binS_0.5, x="Scale", y="Value", desc_stat = "mean_ci", error.plot = "errorbar",
            color = "black", add = "jitter", add.params = list(color = "darkgray"),
            xlab = "Scale", ylab = "Similarity", ylim = c(0, 1))

binS_1 <- read.csv("./simBinScale-binS_1.csv_cleaned.csv")
binS_1 <- ggerrorplot(binS_1, x="Scale", y="Value", desc_stat = "mean_ci", error.plot = "errorbar",
            color = "black", add = "jitter", add.params = list(color = "darkgray"),
            xlab = "Scale", ylab = "Similarity", ylim = c(0, 1))

binS_5 <- read.csv("./simBinScale-binS_5.csv_cleaned.csv")
binS_5 <- ggerrorplot(binS_5, x="Scale", y="Value", desc_stat = "mean_ci", error.plot = "errorbar",
            color = "black", add = "jitter", add.params = list(color = "darkgray"),
            xlab = "Scale", ylab = "Similarity", ylim = c(0, 1))
ggarrange(
  binS_0.1, binS_0.5, binS_1, binS_5,
  labels = c("Bin Size: 0.1", "Bin Size: 0.5", "Bin Size: 1", "Bin Size: 5"),
  label.x = 0.1, label.y = 1.015,
  font.label = list(size = 11, face = "bold")) %>%
  ggexport(width = 3200, height = 2000, res = 350, filename = "./BinScale.png")
