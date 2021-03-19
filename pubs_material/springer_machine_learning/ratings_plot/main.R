library("ggpubr")

# csq_ss <- read.csv("csq_ss.csv")
# csq_ap <- read.csv("csq_ap.csv")
# csq_re <- read.csv("csq_re.csv")
# csq_me <- read.csv("csq_me.csv")
# csq_ha <- read.csv("csq_ha.csv")
# csq_rh <- read.csv("csq_rh.csv")
#
# cpi_ss <- read.csv("cpi_ss.csv")
# cpi_ap <- read.csv("cpi_ap.csv")
# cpi_re <- read.csv("cpi_re.csv")
# cpi_me <- read.csv("cpi_me.csv")
# cpi_ha <- read.csv("cpi_ha.csv")
# cpi_rh <- read.csv("cpi_rh.csv")

# csq <- read.csv("csq.csv")
# cpi <- read.csv("cpi.csv")
p <- read.csv("all.csv")
p$Category <- factor(p$Category, levels = c("LiTr", "BeAf", "CoRe", "MaMa", "MVAE", "MuTr", "Orig"))
p <- ggviolin(p, x = "Category", y = "Rating", orientation = "horiz", add = "median_q1q3")
p <- facet(p +
             theme_pubclean() +
             geom_hline(yintercept = 2, linetype = "dotted", size = 0.5, alpha = 0.4) +
             geom_hline(yintercept = 4, linetype = "dotted", size = 0.5) +
             geom_hline(yintercept = 6, linetype = "dotted", size = 0.5, alpha = 0.4) +
             scale_y_discrete(limits = c("1", "2", "3", "4", "5", "6", "7")),
           scales = "free",
           panel.labs = list(Aspect = c("Ss", "Ap", "Re", "Me", "Ha", "Rh"), Part = c("CPI", "CSQ")),
           facet.by = c("Part", "Aspect"))
p
ggarrange(p) %>%
  ggexport(width = 3200, height = 2000, res = 300, filename = "ratings.png")
