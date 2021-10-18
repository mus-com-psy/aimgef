library("ggplot2")
library("ggpubr")
library("ggdist")

colors <- c("Orig" = "#E69F00",
            "BeAf" = "#56B4E9",
            "LiTr" = "#009E73",
            "MuTr" = "#F0E442",
            "MaMa" = "#0072B2",
            "MVAE" = "#D55E00",
            "CoRe" = "#CC79A7")

draw <- function(data) {
  transComp <- ggplot(data, aes(factor(Rating), transComp)) +
    labs(x = "Rating", y = "Translational Complexity") +
    ggdist::stat_halfeye(adjust = .5, width = .5, .width = c(.5, .95)) +
    ggdist::stat_dots(aes(color = Category, fill = Category), side = "left", dotsize = .8, binwidth = .01)
  # ggarrange(transComp) %>%
  #   ggexport(width = 8000, height = 2000, res = 300, filename = "./LS transComp.png")

  arcScore <- ggplot(data, aes(factor(Rating), arcScore)) +
    labs(x = "Rating", y = "Arc Score") +
    ggdist::stat_halfeye(adjust = .5, width = .5, .width = c(.5, .95)) +
    ggdist::stat_dots(aes(color = Category, fill = Category), side = "left", dotsize = .3, binwidth = .01)
  # ggarrange(arcScore) %>%
  #   ggexport(width = 8000, height = 2000, res = 300, filename = "./LS arcScore.png")

  tonalAmb <- ggplot(data, aes(factor(Rating), tonalAmb)) +
    labs(x = "Rating", y = "Tonal Ambiguity") +
    ggdist::stat_halfeye(adjust = .5, width = .5, .width = c(.5, .95)) +
    ggdist::stat_dots(aes(color = Category, fill = Category), side = "left", dotsize = .5, binwidth = .01)
  # ggarrange(tonalAmb) %>%
  #   ggexport(width = 8000, height = 2000, res = 300, filename = "./LS tonalAmb.png")

  attInterval <- ggplot(data, aes(factor(Rating), attInterval)) +
    labs(x = "Rating", y = "Average Time between Attacks") +
    ggdist::stat_halfeye(adjust = .5, width = .5, .width = c(.5, .95)) +
    ggdist::stat_dots(aes(color = Category, fill = Category), side = "left", dotsize = 1, binwidth = .01)
  # ggarrange(attInterval) %>%
  #   ggexport(width = 8000, height = 2000, res = 300, filename = "./LS attInterval.png")

  rhyDis <- ggplot(data, aes(factor(Rating), rhyDis)) +
    labs(x = "Rating", y = "Average Jitter of Attacks") +
    ggdist::stat_halfeye(adjust = .5, width = .5, .width = c(.5, .95)) +
    ggdist::stat_dots(aes(color = Category, fill = Category), side = "left", dotsize = .9, binwidth = .01)
  # ggarrange(rhyDis) %>%
  #   ggexport(width = 8000, height = 2000, res = 300, filename = "./LS rhyDis.png")
  plot <- ggarrange(transComp, arcScore, tonalAmb, attInterval, rhyDis, common.legend = T, ncol = 1)

  return(plot)
}

data.ss <- subset(read.csv("./LSRatings.csv"), Aspect == "Ss")
data.ap <- subset(read.csv("./LSRatings.csv"), Aspect == "Ap")
data.re <- subset(read.csv("./LSRatings.csv"), Aspect == "Re")
data.me <- subset(read.csv("./LSRatings.csv"), Aspect == "Me")
data.ha <- subset(read.csv("./LSRatings.csv"), Aspect == "Ha")
data.rh <- subset(read.csv("./LSRatings.csv"), Aspect == "Rh")

ggarrange(draw(data.ss), draw(data.ap), draw(data.re), draw(data.me), draw(data.ha), draw(data.rh),
          common.legend = T, ncol = 6,
          labels = c("Stylistic Success", "Aesthetic Pleasure", "Repetitive Pattern", "Melody", "Harmony", "Rhythm")) %>%
  ggexport(width = 28000, height = 12000, res = 600, filename = "./LS-Aspect.png")