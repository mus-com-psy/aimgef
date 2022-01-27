# Make sure to also source the functions in rankBasedCommonFunctions.R
source("rankBasedCommonFunctions.R")

signRankGibbsSampler <- function(xVals, yVals = NULL, nSamples = 1e3, cauchyPriorParameter = 1/sqrt(2), testValue = 0, 
                                 progBar = TRUE, nBurnin = 1, nGibbsIterations = 10, nChains = 10) {
  
  if (progBar) {
    myBar <- txtProgressBar(min = 1, max = nSamples*nChains, initial = 1, char = "*",style=3,width=50)
  }
  
  n <- length(xVals)
  
  if (!is.null(yVals)) {
    differenceScores <- xVals - yVals
  } else {
    differenceScores <- xVals - testValue
  }
  
  differenceSigns <- (sign(differenceScores))
  absDifferenceRanked <- rank(abs(differenceScores))
  prodSignAbsRank <- differenceSigns * absDifferenceRanked
  
  diffSamples <- numeric(n)
  
  
  deltaSamples <- numeric(nSamples)
  deltaSamplesMatrix <- matrix(ncol = nChains, nrow = nSamples-nBurnin)
  oldDeltaProp <- 0
  
  for(thisChain in 1:nChains) {
    
    initDiffSamples <- sort(abs(rnorm(n)))[absDifferenceRanked]
    sampledDiffsAbs <- abs(initDiffSamples)
    
    for (j in 1:nSamples) {
      print(j)
      
      for (i in sample(1:n)) {
        print(i)
        currentRank <- absDifferenceRanked[i]
        
        currentBounds <- upperLowerTruncation(ranks=absDifferenceRanked, values=sampledDiffsAbs, currentRank=currentRank)
        if (is.infinite(currentBounds[["under"]])) {currentBounds[["under"]] <- 0}

        sampledDiffsAbs[i] <- truncNormSample(currentBounds[["under"]], currentBounds[["upper"]], mu = abs(oldDeltaProp), sd=1)
      }
      diffSamples <- sampledDiffsAbs * differenceSigns
      
      if (any(differenceSigns == 0)) {
        nullSamples <- sampledDiffsAbs[differenceSigns == 0] * sample(c(-1,1), size = sum(differenceSigns == 0), replace = TRUE)
        diffSamples[which(differenceSigns == 0)] <- nullSamples
      }
      
      gibbsOutput <- sampleGibbsOneSampleWilcoxon(diffScores = diffSamples, nIter = nGibbsIterations, rscale = cauchyPriorParameter)
      
      deltaSamples[j] <- oldDeltaProp <- gibbsOutput
      if (progBar) setTxtProgressBar(myBar, j + ( (thisChain-1) * nSamples))
      
    }
    
    if (nBurnin > 0) {
      deltaSamples <- deltaSamples[-(1:nBurnin)]
    } else {
      deltaSamples <- deltaSamples
    }
    deltaSamplesMatrix[, thisChain] <- deltaSamples
  }
  
  betweenChainVar <- (nSamples / (nChains - 1)) * sum((apply(deltaSamplesMatrix, 2, mean)  - mean(deltaSamplesMatrix))^2)
  withinChainVar <- (1/ nChains) * sum(apply(deltaSamplesMatrix, 2, var))
  
  fullVar <- ((nSamples - 1) / nSamples) * withinChainVar + (betweenChainVar / nSamples)
  rHat <- sqrt(fullVar/withinChainVar)
  
  return(list(deltaSamples = as.vector(deltaSamplesMatrix), rHat = rHat))
}

sampleGibbsOneSampleWilcoxon <- function(diffScores, nIter = 10, rscale = 1/sqrt(2)){
  ybar <- mean(diffScores)
  n <- length(diffScores)
  sigmaSq <- 1
  mu <- ybar
  g <- ybar^2 / sigmaSq + 1
  
  for(i in 1:nIter){   
    #sample mu
    varMu  <- sigmaSq / (n + (1 / g))
    meanMu <- (n * ybar) / (n + (1 / g))
    mu <- rnorm(1, meanMu, sqrt(varMu) )
    
    # sample g
    scaleg <- (mu^2 + sigmaSq * rscale^2) / (2*sigmaSq)
    g = 1 / rgamma(1, 1, scaleg )
    
    delta <- mu / sqrt(sigmaSq)
  }
  return(delta)
}

