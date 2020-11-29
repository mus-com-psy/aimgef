library("ggpubr")


maia_markov <- read.csv("./os_maia.csv")
transformer <- read.csv("./os_transformer.csv")

maia_markov <- ggline(maia_markov, x = "Step", y = "Mean", linetype = "WO", xlab = "Step (Bars)", ylab = "Originality Score")
maia_markov <- ggpar(maia_markov, xticks.by = 1, ylim = c(0, 1), font.legend = c(11, "plain", "black")) +
  geom_errorbar(aes(ymin = Min, ymax = Max), width=.2) +
  geom_hline(aes(yintercept = 1 - 0.3009752, linetype = "solid")) +
  geom_hline(aes(yintercept = 1 - 0.2749859, linetype = "dashed")) +
  geom_hline(yintercept = 1 - 0.3276547, linetype = "dotted") +
  scale_linetype_manual("", values = c("dotted", "solid", "longdash", "twodash"), labels = c("Baseline .95-CI", "4-bar", "2-bar", "Baseline Mean"))
ggarrange(maia_markov) %>%
  ggexport(width = 3200, height = 2000, res = 600, filename = "./maia_step.png")

transformer <- ggline(transformer, x = "Step", y = "Mean", linetype = "WO", xlab = "Step (Bars)", ylab = "Originality Score")
transformer <- ggpar(transformer, xticks.by = 1, ylim = c(0, 1), font.legend = c(11, "plain", "black")) +
  geom_errorbar(aes(ymin = Min, ymax = Max), width=.2) +
  geom_hline(aes(yintercept = 1 - 0.3009752, linetype = "solid")) +
  geom_hline(aes(yintercept = 1 - 0.2749859, linetype = "dashed")) +
  geom_hline(yintercept = 1 - 0.3276547, linetype = "dotted") +
  scale_linetype_manual("", values = c("dotted", "solid", "longdash", "twodash"), labels = c("Baseline .95-CI", "4-bar", "2-bar", "Baseline Mean"))
ggarrange(transformer) %>%
  ggexport(width = 3200, height = 2000, res = 600, filename = "./transformer_step.png")
