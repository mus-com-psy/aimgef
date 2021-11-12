source("R_rainclouds.R")

pal <- "Dark2"
alp <- .8
data <- read_csv("../LSRatings.csv", col_types = cols(Rating = col_factor(levels = c("1", "2", "3", "4", "5", "6", "7"))))
data.ss <- subset(data, Aspect == "Ss" & Part == "CSQ" & Category != "MaMa")
data.ap <- subset(data, Aspect == "Ap")
data.re <- subset(data, Aspect == "Re")
data.me <- subset(data, Aspect == "Me")
data.ha <- subset(data, Aspect == "Ha")
data.rh <- subset(data, Aspect == "Rh")
# sumrepdat <- summarySE(data.ss, measurevar = "transComp", groupvars = c("Rating", "Category"))

p <- ggplot(data.ss, aes(x = Rating, y = transComp)) +
  geom_bar(aes(x = Rating, y = (..count..) / max(table(data.ss$Rating)), fill = Category), width = .1, position = position_stack(), alpha = alp) +
  geom_flat_violin(position = position_nudge(x = .2, y = 0),
                   adjust = 1, trim = FALSE, colour = NA, fill = "gray10", alpha = alp, scale = "count") +
  geom_point(aes(x = as.numeric(Rating) - .3, y = transComp, colour = Category), position = position_jitter(width = .1), size = 1.3, shape = 20) +
  geom_boxplot(aes(x = Rating, y = transComp, fill = Category), position = position_dodge(width = .3), outlier.shape = NA, lwd = .4, alpha = 1, width = .3, colour = "black") +
  # geom_histogram(aes(x = Rating, fill = Category)) +
  # scale_x_discrete(expand=expansion(add=1)) +
  # geom_line(data = sumrepdat, aes(x = as.numeric(Rating) + .1, y = transComp_mean,
  #                                 group = Category, colour = Category), linetype = 3) +
  # geom_point(data = sumrepdat, aes(x = as.numeric(group) + .1, y = score_mean,
  #                                  group = time, colour = time), shape = 18) +
  # geom_errorbar(data = sumrepdat, aes(x = as.numeric(Rating) + .1, y = transComp_mean,
  #                                     group = Category, colour = Category, ymin = transComp_mean - se, ymax = transComp_mean + se), width
  #                 = .05) +
  scale_colour_brewer(palette = pal) +
  scale_fill_brewer(palette = pal) +
  scale_y_continuous(limits = c(0, 1), expand = c(.01, .01)) +
  # ggtitle("Figure R12: Repeated Measures - Factorial (Extended)") +
  # coord_flip() +
  theme(legend.position = "top") + theme_pubr()

ggsave("plot.png", width = 9, height = 6, scale = 1, dpi = 300)
p