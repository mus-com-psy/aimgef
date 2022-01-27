
truncNormSample <- function(lBound = -Inf, uBound = Inf, mu = 0, sd = 1) {
  
  lBoundUni <- pnorm(lBound, mean = mu, sd = sd)
  uBoundUni <- pnorm(uBound, mean = mu, sd = sd)
  mySample <- qnorm(runif(1, lBoundUni, uBoundUni), mean = mu, sd = sd)
  
  return(mySample)
}

upperLowerTruncation <- function(ranks, values, currentRank) {
  
  if (currentRank == min(ranks)) {
    under <- -Inf
  } else {
    under <- max(values[ranks < currentRank])
  }
  
  if (currentRank == max(ranks)) {
    upper <- Inf
  } else {
    upper <- min(values[ranks > currentRank])
  }
  
  return(list(under=under, upper=upper))
}


# this function computes BF10 for both Wilcoxon tests and Spearman's rho, as specified in whichTest. Recommended values
# for Wilcoxon are 1/sqrt(2) and 1 for Spearman. These should be the same as specified in the Gibbs sampler function call.
# The oneSided argument can be FALSE (for two-sided tests), "right" for positive one-sided tests, and "left" for negative
# one-sided tests.
computeBayesFactorOneZero <- function(posteriorSamples, priorParameter = 1, oneSided = "right", whichTest = "Wilcoxon") {
  
  postDens <- logspline::logspline(posteriorSamples)
  densZeroPoint <- logspline::dlogspline(0, postDens)
  
  corFactorPosterior <- logspline::plogspline(0, postDens)
  if (oneSided == "right")
    corFactorPosterior <- 1 - corFactorPosterior
  
  if (whichTest == "Wilcoxon") {
    # priorParameter should be the Cauchy scale parameter
    priorDensZeroPoint <- dcauchy(0, scale = priorParameter)
    corFactorPrior <-  pcauchy(0, scale = priorParameter, lower.tail = (oneSided != "right" ))
  } else if (whichTest == "Spearman") {
    # priorParameter should be kappa
    priorDensZeroPoint <- dbeta(0.5, 1/priorParameter, 1/priorParameter) / 2
    corFactorPrior <-  pbeta(0.5, 1/priorParameter, 1/priorParameter, lower.tail = (oneSided != "right" ))
  }
  
  if (isFALSE(oneSided)) {
    bf10 <- priorDensZeroPoint / densZeroPoint
  } else {
    bf10 <- (priorDensZeroPoint / corFactorPrior) / (densZeroPoint / corFactorPosterior)
  }
  
  return(bf10)
}