library("ggplot2")
library("ggpubr")
library("ggdist")

dataRaw <- subset(read.csv("./MTGeneratedRaw.csv"), group >= 0.3)
transCompRaw <- ggplot(dataRaw, aes(factor(group), transComp)) +
  labs(x = "Epochs", y = "Translational Complexity") +
  ggdist::stat_halfeye(adjust = 0.5, width = 0.6, .width = c(.5, .95)) +
  ggdist::stat_dots(aes(fill = factor(group)), side = "left", dotsize = .4, justification = 1.05, binwidth = .01)
# transCompRaw <- ggplot(iris, aes(Species, Sepal.Width)) +
#   ggdist::stat_halfeye(adjust = .5, width = .3, .width = c(0.5, 1)) +
#   ggdist::stat_dots(side = "left", dotsize = .4, justification = 1.05, binwidth = .1)

ggarrange(transCompRaw) %>%
  ggexport(width = 3200, height = 2000, res = 300, filename = "./MTGeneratedRaw.png")