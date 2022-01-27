# Set working directory
setwd("./")

# Source the functions with the rankSumGibbsSampler and signRankGibbsSampler
source('spearmanSampler.R')

library(effsize)
library(BayesFactor)
library(sn)
library(logspline)
library(parallel)
library(foreach)
library(doMC)
library(copula)
library(hypergeo)

CSQ_MuTr <- read.csv("CSQ-MuTr.csv")
CSQ_CoRe <- read.csv("CSQ-CoRe.csv")
CSQ_MaMa <- read.csv("CSQ-MaMa.csv")
CSQ_MVAE <- read.csv("CSQ-MVAE.csv")
CSQ_Orig <- read.csv("CSQ-Orig.csv")
CSQ_BeAf <- read.csv("CSQ-BeAf.csv")

CPI_MuTr <- read.csv("CPI-MuTr.csv")
CPI_VAE <- read.csv("CPI-MVAE.csv")
CPI_Orig <- read.csv("CPI-Orig.csv")
CPI_LiTr <- read.csv("CPI-LiTr.csv")
year_ss <- read.csv("year_ss.csv")

pearsonBayesFactor10 <- function(n, r, kappa = 1) {
  logHyperTerm <- log(hypergeo::genhypergeo(U = c((n - 1) / 2, (n - 1) / 2), L = ((n + 2 / kappa) / 2), z = r^2))
  logResult <- log(2^(1 - 2 / kappa)) + 0.5 * log(pi) - lbeta(1 / kappa, 1 / kappa) +
    lgamma((n + 2 / kappa - 1) / 2) - lgamma((n + 2 / kappa) / 2) + logHyperTerm
  realResult <- exp(Re(logResult))
  return(realResult)
}

no_cores <- detectCores() - 1


#nIter <- 1e4
#nBurnin <- 5e3
#nReps <- nRuns <- 1e3

nIter <- 50
nBurnin <- 10
nReps <- nRuns <- 5


allScenarios <- c("Rh-Ap")


nQuants <- 41
allQuants <- seq(0, 1, length.out = nQuants)

myColnames <- c("obsPearson", "obsSpearman", "myBF",
                "pVal", "pValPara", paste0(100 * allQuants, "% Q"))
nColsResults <- length(myColnames)
results <- matrix(ncol = nColsResults, nrow = 0, dimnames = list(NULL, myColnames))

registerDoMC(core = no_cores)

allResults <- list()

analyzeSamples <- function(nIter = 1e3, nBurnin = 1e3, myCase, nColsResults, allQuants) {

  switch(myCase,
         "Ha-Ap" = {
           x <- c(
             CSQ_Orig$Ha,
             CSQ_CoRe$Ha,
             CSQ_MaMa$Ha,
             CSQ_MuTr$Ha,
             CSQ_BeAf$Ha,
             CSQ_MVAE$Ha,
             CPI_Orig$Ha,
             CPI_MuTr$Ha,
             CPI_VAE$Ha,
             CPI_LiTr$Ha
           )
           y <- c(
             CSQ_Orig$Ap,
             CSQ_CoRe$Ap,
             CSQ_MaMa$Ap,
             CSQ_MuTr$Ap,
             CSQ_BeAf$Ap,
             CSQ_MVAE$Ap,
             CPI_Orig$Ap,
             CPI_MuTr$Ap,
             CPI_VAE$Ap,
             CPI_LiTr$Ap
           )
           oneSided <- "right"
         },
         "Me-Ap" = {
           x <- c(
             CSQ_Orig$Me,
             CSQ_CoRe$Me,
             CSQ_MaMa$Me,
             CSQ_MuTr$Me,
             CSQ_BeAf$Me,
             CSQ_MVAE$Me,
             CPI_Orig$Me,
             CPI_MuTr$Me,
             CPI_VAE$Me,
             CPI_LiTr$Me
           )
           y <- c(
             CSQ_Orig$Ap,
             CSQ_CoRe$Ap,
             CSQ_MaMa$Ap,
             CSQ_MuTr$Ap,
             CSQ_BeAf$Ap,
             CSQ_MVAE$Ap,
             CPI_Orig$Ap,
             CPI_MuTr$Ap,
             CPI_VAE$Ap,
             CPI_LiTr$Ap
           )
           oneSided <- "right"
         },
         "Rh-Ap" = {
           x <- c(
             CSQ_Orig$Rh,
             CSQ_CoRe$Rh,
             CSQ_MaMa$Rh,
             CSQ_MuTr$Rh,
             CSQ_BeAf$Rh,
             CSQ_MVAE$Rh,
             CPI_Orig$Rh,
             CPI_MuTr$Rh,
             CPI_VAE$Rh,
             CPI_LiTr$Rh
           )
           y <- c(
             CSQ_Orig$Ap,
             CSQ_CoRe$Ap,
             CSQ_MaMa$Ap,
             CSQ_MuTr$Ap,
             CSQ_BeAf$Ap,
             CSQ_MVAE$Ap,
             CPI_Orig$Ap,
             CPI_MuTr$Ap,
             CPI_VAE$Ap,
             CPI_LiTr$Ap
           )
           oneSided <- "right"
         },
         "year-ss" = {
           x <- year_ss$year
           y <- year_ss$rating
           oneSided <- FALSE
         }
  )

  thisRow <- try(expr = {

    mySamples <- spearmanGibbsSampler(x, y, nSamples = nIter + nBurnin, progBar = TRUE)$rhoSamples[nBurnin:(nIter + nBurnin)]
    mySamples <- pearsonToSpearman(mySamples)

    rsQuants <- quantile(mySamples, probs = allQuants)
    bf10 <- computeBayesFactorOneZero(mySamples, priorParameter = 1, oneSided = oneSided, whichTest = "Spearman")

    #myLogspline <- logspline(mySamples)
    #bf01 <- dlogspline(0, fit = myLogspline) / 0.5
    #paraBf <- pearsonBayesFactor10(r = cor(x, y), n = thisN)

    freqRes <- cor.test(x, y, method = "spearman")
    testStat <- unname(freqRes$estimate)
    pVal <- unname(freqRes$p.value)
    pValPara <- cor.test(x, y)$p.value

    obsPearson <- cor(x, y)
    obsSpearman <- cor(x, y, method = "spearman")

    return((c(obsPearson, obsSpearman,
              bf10, pVal, pValPara, rsQuants)))
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
    analyzeSamples(myCase = thisScenario, nIter = nIter, nBurnin = nBurnin, nColsResults = nColsResults,
                   allQuants = allQuants)
  }
  results <- rbind(results, myResult)
  rownames(results) <- NULL
  print(mean(results[, "myBF"]))
  save(results, file = myFilename)
}
  

