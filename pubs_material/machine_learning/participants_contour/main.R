library("ggpubr")

p <- read.csv("data.csv")
# age <- gghistogram(age, bins = 60, x = "age", y = "..count..", xlab = "Age") +
#   scale_x_continuous(breaks = c(10, 20, 30, 40, 50, 60, 70))
# mt <- gghistogram(mt, bins = 20, x = "mt", y = "..count..", xlab = "Years of musical training") +
#   scale_x_continuous(breaks = c(0, 10, 20)) +
#   scale_y_continuous(breaks = c(0, 2, 4, 6, 8, 10))
# p <- add_summary(p, fun = "median_q1q3", size = 0.3)

p  <- ggscatter(p, x = "mt", y = "age", xlab = "Years of musical training", ylab = "Age") +
  scale_x_continuous(breaks = c(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20)) +
  scale_y_continuous(breaks = c(10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75)) +
  grids(linetype = "dashed")



ggarrange(p) %>%
  ggexport(width = 3200, height = 2000, res = 400, filename = "./participants_contour.png")
