# Set working directory
setwd("./")

# Source the functions with the rankSumGibbsSampler and signRankGibbsSampler
source('signRankSampler.R')
source('rankSumSampler.R')

# install.packages(c("effsize", "BayesFactor", "sn", "parallel", "logspline", "foreach", "doMC"))
library(effsize)
library(BayesFactor)
library(sn)
library(parallel)
library(logspline)
library(foreach)
library(doMC)

no_cores <- detectCores() - 1

#nIter <- 1e4
#nBurnin <- 5e3
#nReps <- nRuns <- 1e3

nIter <- 3e2
nBurnin <- 1e2
nReps <- nRuns <- 10

CSQ_MuTr <- read.csv("CSQ-MuTr.csv")
CSQ_CoRe <- read.csv("CSQ-CoRe.csv")
CSQ_MaMa <- read.csv("CSQ-MaMa.csv")
CSQ_MVAE <- read.csv("CSQ-MVAE.csv")
CSQ_Orig <- read.csv("CSQ-Orig.csv")
CSQ_BeAf <- read.csv("CSQ-BeAf.csv")

CPI_MuTr <- read.csv("CPI-MuTr.csv")
CPI_MVAE <- read.csv("CPI-MVAE.csv")
CPI_Orig <- read.csv("CPI-Orig.csv")
CPI_LiTr <- read.csv("CPI-LiTr.csv")

allScenarios <- c(
  "CPI-LiTr-MuTr-Ss",
  "CPI-LiTr-MVAE-Ss",
  "CPI-LiTr-Orig-Ss"
  #"CSQ-MaMa-MuTr-Ss",
  #"CSQ-MaMa-MuTr-Re",
  #"CSQ-MaMa-MVAE-Re",
  #"CSQ-MaMa-CoRe-Re",
  #"CSQ-MaMa-CoRe-Ss",
  #"CSQ-MuTr-CoRe-Ss",
  #"Orig-CSQ-CPI-Ss",
  #"MuTr-CSQ-CPI-Ss",
  #"MVAE-CSQ-CPI-Ss",
  # "Ap-Ss",
  #"CSQ-MuTr-MVAE-Ss",
  #"CPI-MuTr-MVAE-Ss",
  #"CSQ-MuTr-MVAE-Me",
  #"CSQ-MuTr-MaMa-Me",
  #"CPI-MuTr-MVAE-Me",
  #"CSQ-MuTr-MVAE-Ha",
  #"CSQ-MuTr-MaMa-Ha",
  #"CPI-MuTr-MVAE-Ha"
)

myColnames <- c("obsMeanX", "obsMeanY", "obsMedianY", "obsMedianY", "obsMedianDiff",
                "obsVarX", "obsVarY", "myBF", "moreyBF", "W", "pval_W", "pval_T",
                "cohenD", paste0(100 * c(0.025, 0.25, 0.5, 0.75, 0.975), "% Q"))
nColsResults <- length(myColnames)
registerDoMC(core = no_cores)

analyzeSamples <- function(nIter, nBurnin, myCase, progBar = TRUE) {
  switch(myCase,

         "CPI-LiTr-MuTr-Ss" = {
           x <- CPI_LiTr$Ss
           y <- CPI_MuTr$Ss
           paired <- FALSE
           oneSided <- FALSE
         },
         "CPI-LiTr-MVAE-Ss" = {
           x <- CPI_LiTr$Ss
           y <- CPI_MVAE$Ss
           paired <- FALSE
           oneSided <- FALSE
         },
         "CPI-LiTr-Orig-Ss" = {
           x <- CPI_LiTr$Ss
           y <- CPI_Orig$Ss
           paired <- FALSE
           oneSided <- FALSE
         },
         "CSQ-MaMa-MuTr-Ss" = {
           x <- CSQ_MaMa$Ss
           y <- CSQ_MuTr$Ss
           paired <- FALSE
           oneSided <- FALSE
         },
         "CSQ-MaMa-MuTr-Re" = {
           x <- CSQ_MaMa$Re
           y <- CSQ_MuTr$Re
           paired <- FALSE
           oneSided <- "right"
         },
         "CSQ-MaMa-MVAE-Re" = {
           x <- CSQ_MaMa$Re
           y <- CSQ_MVAE$Re
           paired <- FALSE
           oneSided <- "right"
         },
         "CSQ-MaMa-CoRe-Re" = {
           x <- CSQ_MaMa$Re
           y <- CSQ_CoRe$Re
           paired <- FALSE
           oneSided <- "right"
         },
         "CSQ-MaMa-CoRe-Ss" = {
           x <- CSQ_MaMa$Ss
           y <- CSQ_CoRe$Ss
           paired <- FALSE
           oneSided <- "right"
         },
         "CSQ-MuTr-CoRe-Ss" = {
           x <- CSQ_MuTr$Ss
           y <- CSQ_CoRe$Ss
           paired <- FALSE
           oneSided <- "right"
         },
         "Orig-CSQ-CPI-Ss" = {
           x <- CSQ_Orig$Ss
           y <- CPI_Orig$Ss
           paired <- FALSE
           oneSided <- "right"
         },
         "MuTr-CSQ-CPI-Ss" = {
           x <- CSQ_MuTr$Ss
           y <- CPI_MuTr$Ss
           paired <- FALSE
           oneSided <- "right"
         },
         "MVAE-CSQ-CPI-Ss" = {
           x <- CSQ_MVAE$Ss
           y <- CPI_MVAE$Ss
           paired <- FALSE
           oneSided <- "right"
         },
         "Ap-Ss" = {
           x <- c(
             CSQ_Orig$Ap,
             CSQ_CoRe$Ap,
             CSQ_MaMa$Ap,
             CSQ_MuTr$Ap,
             CSQ_BeAf$Ap,
             CSQ_MVAE$Ap,
             CPI_Orig$Ap,
             CPI_MuTr$Ap,
             CPI_MVAE$Ap,
             CPI_LiTr$Ap
           )
           y <- c(
             CSQ_Orig$Ss,
             CSQ_CoRe$Ss,
             CSQ_MaMa$Ss,
             CSQ_MuTr$Ss,
             CSQ_BeAf$Ss,
             CSQ_MVAE$Ss,
             CPI_Orig$Ss,
             CPI_MuTr$Ss,
             CPI_MVAE$Ss,
             CPI_LiTr$Ss
           )
           paired <- FALSE
           oneSided <- "right"
         },
         "CSQ-MuTr-MVAE-Ss" = {
           x <- CSQ_MuTr$Ss
           y <- CSQ_MVAE$Ss
           paired <- FALSE
         },
         "CPI-MuTr-MVAE-Ss" = {
           x <- CPI_MuTr$Ss
           y <- CPI_MVAE$Ss
           paired <- FALSE
         },
         "CSQ-MuTr-MVAE-Me" = {
           x <- CSQ_MuTr$Me
           y <- CSQ_MVAE$Me
           paired <- FALSE
         },
         "CSQ-MuTr-MaMa-Me" = {
           x <- CSQ_MuTr$Me
           y <- CSQ_MaMa$Me
           paired <- FALSE
         },
         "CPI-MuTr-MVAE-Me" = {
           x <- CPI_MuTr$Me
           y <- CPI_MVAE$Me
           paired <- FALSE
         },
         "CSQ-MuTr-MVAE-Ha" = {
           x <- CSQ_MuTr$Ha
           y <- CSQ_MVAE$Ha
           paired <- FALSE
         },
         "CSQ-MuTr-MaMa-Ha" = {
           x <- CSQ_MuTr$Ha
           y <- CSQ_MaMa$Ha
           paired <- FALSE
         },
         "CPI-MuTr-MVAE-Ha" = {
           x <- CPI_MuTr$Ha
           y <- CPI_MVAE$Ha
           paired <- FALSE
         })
  thisRow <- try(expr = {
    if (paired) {
      mySamples <- signRankGibbsSampler(x, y, nSamples = nIter + nBurnin, progBar = progBar)$deltaSamples[nBurnin:(nIter + nBurnin)]
    } else {
      mySamples <- rankSumGibbsSampler(x, y, nSamples = nIter + nBurnin, progBar = progBar)$deltaSamples[nBurnin:(nIter + nBurnin)]
    }
    rsQuants <- quantile(mySamples, probs = c(0.025, 0.25, 0.5, 0.75, 0.975))
    bf10 <- computeBayesFactorOneZero(mySamples, priorParameter = 1 / sqrt(2), oneSided = oneSided)
    moreyBf <- 1 / exp(ttestBF(x, y, rscale = 1 / sqrt(2), paired = paired)@bayesFactor$bf)

    freqRes <- wilcox.test(x, y, paired = paired)
    testStat <- unname(freqRes$statistic)
    pVal <- unname(freqRes$p.value)
    pValT <- t.test(x, y, paired = paired)$p.value
    cohenD <- cohen.d(x, y, paired = paired)$estimate

    obsMeanX <- mean(x)
    obsMeanY <- mean(y)
    obsMedianX <- median(x)
    obsMedianY <- median(y)
    obsMedianDiff <- median(x - y)
    obsVarX <- var(x)
    obsVarY <- var(y)

    return(unname(c(obsMeanX, obsMeanY, obsMedianX, obsMedianY, obsMedianDiff, obsVarX,
                    obsVarY, bf10, moreyBf, testStat, pVal, pValT, cohenD, rsQuants)))
  }, silent = FALSE)
  if (!is.numeric(thisRow)) {
    thisRow <- rep(0, (nColsResults - 2))
  }

  return(thisRow)
}

#########################
######## Run It! ########
#########################
for (thisScenario in allScenarios) {
  myFilename <- paste0(thisScenario, ".Rdata")
  results <- matrix(ncol = nColsResults, nrow = 0, dimnames = list(NULL, myColnames))
  print(thisScenario)
  myResult <- foreach(k = 1:nRuns, .combine = 'rbind') %dopar% {
    analyzeSamples(myCase = thisScenario, nIter = nIter, nBurnin = nBurnin)
  }
  results <- rbind(results, myResult)
  rownames(results) <- NULL
  save(results, file = myFilename)
  print(mean(results[, "myBF"]))
}
