source("R_rainclouds.R")
library(hash)

pal <- "Dark2"
alp <- .8
data <- read_csv("../../LSRatings.csv", col_types = cols(
  Rating = col_factor(levels = c("1", "2", "3", "4", "5", "6", "7")),
  Category = col_factor(levels = c("Orig", "BeAf", "MaMa", "CoRe", "MuTr", "MVAE", "LiTr"))
))

features <- hash()
features[["transComp"]] <- "Translational Complexity"
features[["arcScore"]] <- "Arc score"
features[["tonalAmb"]] <- "Tonal ambiguity"
features[["ioi_mean"]] <- "IOI mean"
features[["ioi_variance"]] <- "IOI variance"
features[["ioi2_mean"]] <- "IOI2 mean"
features[["ioi2_variance"]] <- "IOI2 variance"
features[["kot_mean"]] <- "KOT mean"
features[["kot_variance"]] <- "KOT variance"
features[["kdt_mean"]] <- "KDT mean"
features[["kdt_variance"]] <- "KDT variance"
features[["jitter_mean"]] <- "Jitter mean"
features[["jitter_variance"]] <- "Jitter variance"

ratings <- hash()
ratings[["Ss"]] <- "Stylistic success"
ratings[["Ap"]] <- "Aesthetic pleasure"
ratings[["Re"]] <- "Repetition"
ratings[["Me"]] <- "Melody"
ratings[["Ha"]] <- "Harmony"
ratings[["Rh"]] <- "Rhythm"

myplotf <- function(part, rating, .feature) {
  feature <- sym(.feature)
  d <- subset(data, Part == part & Aspect == rating)
  p <- ggplot(d, aes(x = Rating, y = !!feature)) +
    geom_bar(aes(x = Rating, y = (..count..) / max(table(d$Rating)), fill = Category), width = .15, position = position_stack(), alpha = alp) +
    geom_flat_violin(position = position_nudge(x = .2, y = 0),
                     adjust = 1, trim = FALSE, colour = NA, fill = "gray10", alpha = alp, scale = "count") +
    geom_point(aes(x = as.numeric(Rating) - .3, y = !!feature, colour = Category), position = position_jitter(width = .1), size = 1.3, shape = 20) +
    geom_boxplot(aes(x = Rating, y = !!feature, fill = Category), position = position_dodge(width = .3), outlier.shape = NA, lwd = .4, alpha = 1, width = .3, colour = "black") +
    scale_colour_brewer(palette = pal) +
    scale_fill_brewer(palette = pal) +
    scale_y_continuous(limits = c(0, 1), expand = c(.01, .01)) +
    labs(x = ratings[[rating]], y = features[[.feature]]) +
    guides(colour = guide_legend(nrow = 1)) +
    theme(legend.position = "top") +
    theme_pubr()
  ggsave(paste0(paste(part, .feature, rating, sep = "-"), ".png"),
         plot = p, width = 9, height = 6, scale = 1, dpi = 300)
}

for (part in c("CSQ", "CPI")) {
  for (rating in keys(ratings)) {
    for (feature in keys(features)) {
      myplotf(part, rating, feature)
    }
  }
}