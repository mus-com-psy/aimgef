# Make sure to also source the functions in rankBasedCommonFunctions.R
source("rankBasedCommonFunctions.R")

spearmanGibbsSampler <- function(xVals, yVals, nSamples = 1e3, progBar = TRUE, kappaPriorParameter = 1, 
                                 nBurnin = 1, nChains = 5) {
  
  if (progBar) {
    myBar <- txtProgressBar(min = 1, max = nSamples*nChains, initial = 1, char = "*", style=3, width=50)
  }
  
  n <- length(xVals)
  xRanks <- rank(xVals)
  yRanks <- rank(yVals)
  mySigma <- diag(2)
  
  # Target: posterior samples of rho
  rhoSamples <- numeric(nSamples)
  rhoSamplesMatrix <- matrix(ncol = nChains, nrow = nSamples-nBurnin)
  
  for(thisChain in 1:nChains) {
    
    # intitialise latent variables
    # intialise rho that is compatible with xVals, yVals
    currentXVals <- sort(rnorm((n)))[xRanks] # initial values
    currentYVals <- sort(rnorm((n)))[yRanks] # initial values
    
    currentRho <- cor(currentXVals, currentYVals)
    chanceMechanism <- runif(nSamples)
    
    for (j in 1:nSamples) {
      # Metropolis sampler: 
      # currentXVals and currentYVals first get realigned with the underlying current rho
      #
      for (i in sample(1:n)) {
        # Gibbs step go through pairs of z^{x}, z^{y} with current rho fixed
        #   Thus, current is respect to the Gibbs step. Furthermore, 
        #   here align latent variables to the rank
        #
        currentXRank <- xRanks[i]
        currentYRank <- yRanks[i]
  
        regressXOnY <- mean(currentYVals[yRanks==currentYRank])
        regressYOnX <- mean(currentXVals[xRanks==currentXRank])
        
        xBounds <- upperLowerTruncation(ranks=xRanks, values=currentXVals, currentRank=currentXRank)
        currentXVals[i] <- truncNormSample(xBounds[["under"]], xBounds[["upper"]], mu=(currentRho*regressXOnY), sd=sqrt(1-currentRho^2))
        
        yBounds <- upperLowerTruncation(ranks=yRanks, values=currentYVals, currentRank=currentYRank)
        currentYVals[i] <- truncNormSample(yBounds[["under"]], yBounds[["upper"]], mu=(currentRho*regressYOnX), sd=sqrt(1-currentRho^2))
      }    
      
      currentXVals <- (currentXVals-mean(currentXVals))/sd(currentXVals)
      currentYVals <- (currentYVals-mean(currentYVals))/sd(currentYVals)
      
      # This is the sufficient statistic to evaluate the likelihood part of the MH
      rObs <- cor(currentXVals, currentYVals)
      
      # Do Metropolis step here
      rhoNew <- metropolisOneStep(rhoCurrent=currentRho, rObs=rObs,n=n, alpha=1/kappaPriorParameter, chanceMechanism[j])
      
      # Store MH update
      rhoSamples[j] <- rhoNew # add proposal to samples if accepted
      currentRho <- rhoNew # add proposal to samples if accepted
      
      if (progBar) setTxtProgressBar(myBar, j + ( (thisChain-1) * nSamples)) 
    }
    
    if (nBurnin > 0) {
      rhoSamples <- rhoSamples[-(1:nBurnin)]
    } else {
      rhoSamples <- rhoSamples
    }
    
    rhoSamplesMatrix[, thisChain] <- rhoSamples
    
  }

  betweenChainVar <- (nSamples / (nChains - 1)) * sum((apply(rhoSamplesMatrix, 2, mean)  - mean(rhoSamplesMatrix))^2)
  withinChainVar <- (1/ nChains) * sum(apply(rhoSamplesMatrix, 2, var))
  
  fullVar <- ((nSamples - 1) / nSamples) * withinChainVar + (betweenChainVar / nSamples)
  rHat <- sqrt(fullVar/withinChainVar)

  rhoSamples <- pearsonToSpearman(as.vector(rhoSamplesMatrix)) # Transform Pearson's rho to Spearman's rho

  return(list(rhoSamples = rhoSamples, rHat = rHat))
}

metropolisOneStep <- function (rhoCurrent, rObs, n, alpha=1, chanceMechanism) {
  # chanceMechanism is runif(1) vectorised
  # 
  zCurrent <- atanh(rhoCurrent)
  zCandidate <- rnorm(1, mean=atanh(rhoCurrent), sd=1/sqrt(n-3))
  rhoCandidate <- tanh(zCandidate)
  
  logAcceptance <- (alpha-n/2)*(log(1-rhoCandidate^2)-log(1-rhoCurrent^2))+
    n*((1-rhoCurrent*rObs)/(1-rhoCurrent^2)-(1-rhoCandidate*rObs)/(1-rhoCandidate^2))
  
  if (chanceMechanism <= exp(logAcceptance)) {
    return(rhoCandidate)
  } else {
    return(rhoCurrent)
  }
}

pearsonToSpearman <- function(rho){
  
  mySpear <- (6/pi)*asin(rho/2)
  
  return(mySpear)
}

