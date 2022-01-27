# Make sure to also source the functions in rankBasedCommonFunctions.R

rankSumGibbsSampler <- function(xVals, yVals, nSamples = 1e3, cauchyPriorParameter = 1/sqrt(2),  progBar = TRUE, 
                                nBurnin = 1, nGibbsIterations = 10, nChains = 10) {
  
  if (progBar) {
    myBar <- txtProgressBar(min = 1, max = nSamples*nChains, initial = 1, char = "*",style=3,width=50)
  }
  
  n1 <- length(xVals)
  n2 <- length(yVals)
  
  allRanks <- rank(c(xVals,yVals))
  xRanks <- allRanks[1:n1]
  yRanks <- allRanks[(n1+1):(n1+n2)]
  
  deltaSamples <- numeric(nSamples)
  deltaSamplesMatrix <- matrix(ncol = nChains, nrow = nSamples-nBurnin)
  totalIterCount <- 0
  
  for(thisChain in 1:nChains) {
    
    currentVals <- sort(rnorm((n1+n2)))[allRanks] # initial values
    
    oldDeltaProp <- 0
    
    for (j in 1:nSamples) {
      
      for (i in sample(1:(n1+n2))) {
        
        currentRank <- allRanks[i]
        
        currentBounds <- upperLowerTruncation(ranks=allRanks, values=currentVals, currentRank=currentRank)
        if (i <= n1) {
          oldDeltaProp <- -0.5*oldDeltaProp
        } else {
          oldDeltaProp <- 0.5*oldDeltaProp
        }
        
        currentVals[i] <- truncNormSample(currentBounds[["under"]], currentBounds[["upper"]], mu=oldDeltaProp, sd=1)
        
      }
      
      xVals <- currentVals[1:n1]
      yVals <- currentVals[(n1+1):(n1+n2)]
      
      gibbsResult <- sampleGibbsTwoSampleWilcoxon(x = xVals, y = yVals, nIter = nGibbsIterations,
                                                  rscale = cauchyPriorParameter)
      
      deltaSamples[j] <- oldDeltaProp <- gibbsResult
      if (progBar) setTxtProgressBar(myBar, j + ( (thisChain-1) * nSamples)) 
      
    }
    
    if (nBurnin > 0) {
      deltaSamples <- -deltaSamples[-(1:nBurnin)]
    } else {
      deltaSamples <- -deltaSamples
    }
    
    deltaSamplesMatrix[, thisChain] <- deltaSamples
    
  }
  
  betweenChainVar <- (nSamples / (nChains - 1)) * sum((apply(deltaSamplesMatrix, 2, mean)  - mean(deltaSamplesMatrix))^2)
  withinChainVar <- (1/ nChains) * sum(apply(deltaSamplesMatrix, 2, var))
  
  fullVar <- ((nSamples - 1) / nSamples) * withinChainVar + (betweenChainVar / nSamples)
  rHat <- sqrt(fullVar/withinChainVar)
  
  return(list(deltaSamples = as.vector(deltaSamplesMatrix), rHat = rHat))
}

sampleGibbsTwoSampleWilcoxon <- function(x, y, nIter = 10, rscale = 1/sqrt(2)) {
  meanx <- mean(x)
  meany <- mean(y)
  n1 <- length(x)
  n2 <- length(y)
  sigmaSq <- 1 # Arbitrary number for sigma
  g <- 1
  for(i in 1:nIter){
    #sample mu
    varMu <- (4 * g * sigmaSq) / ( 4 + g * (n1 + n2) )
    meanMu <- (2 * g * (n2 * meany - n1 * meanx)) / ((g * (n1 + n2) + 4))
    mu <- rnorm(1, meanMu, sqrt(varMu))
    # sample g
    betaG <- (mu^2 + sigmaSq * rscale^2) / (2*sigmaSq)
    g <- 1/rgamma(1, 1, betaG)
    # convert to delta
    delta <- mu / sqrt(sigmaSq)
  }
  return(delta)
}



