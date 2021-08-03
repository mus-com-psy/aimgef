library("ggpubr")

jit <- read.csv("./simJit_all.csv")
jit <- ggerrorplot(jit, x="Jitter", y="Value", desc_stat = "mean_ci", error.plot = "errorbar",
            color = "black", add = "jitter", add.params = list(color = "darkgray"),
            xlab = "Jitter (Seconds)", ylab = "Similarity")
ggarrange(jit) %>%
  ggexport(width = 3200, height = 2000, res = 500, filename = "./jitter.png")
