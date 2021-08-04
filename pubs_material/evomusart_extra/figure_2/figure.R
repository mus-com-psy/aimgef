library("ggpubr")

transformer <- read.csv("./oriGen.csv")
transformer <- ggline(transformer, x = "Step", y = "Mean", xlab = "Step (Seconds)", ylab = "Originality Score")
transformer <- ggpar(transformer, xticks.by = 1, ylim = c(0, 1), font.legend = c(11, "plain", "black")) +
  geom_errorbar(aes(ymin = Min, ymax = Max), width=.2) +
  geom_hline(aes(yintercept = 1 - 0.4379057, linetype = "solid")) +
  geom_hline(aes(yintercept = 1 - 0.4212908, linetype = "dashed")) +
  geom_hline(yintercept = 1 - 0.4540512, linetype = "dotted") +
  scale_linetype_manual("", values = c("dotted", "solid"), labels = c("Baseline .95-CI", "Baseline Mean"))
ggarrange(transformer) %>%
  ggexport(width = 3200, height = 2000, res = 600, filename = "./transformer_step.png")
