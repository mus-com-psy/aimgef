library("ggpubr")

rt <- read.csv("./runtime.csv")
rt_note <- ggline(rt, x="numNotes", y="runtime", numeric.x.axis=TRUE,
            xlab = "Number of notes (Million)", ylab = "Runtime (Seconds)")
rt_entry <- ggline(rt, x="numEntries", y="runtime", numeric.x.axis=TRUE,
            xlab = "Number of entries (Million)", ylab = "Runtime (Seconds)")

ggarrange(
  rt_note, rt_entry,
  label.x = 0.1, label.y = 1.015,
  font.label = list(size = 11, face = "bold")) %>%
  ggexport(width = 3200, height = 2000, res = 350, filename = "./runtime.png")
