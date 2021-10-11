xy.data <- read.csv("./arcreport/15-0.796.csv", header=TRUE)
attach(xy.data)
xy.data$ontime2 <- xy.data$ontime^2
head(xy.data)

plot(ontime, pitch, pch=16)

lm1 <- lm(formula = pitch ~ ontime + ontime2, data = xy.data)
summary(lm1)

mse <- mean(resid(lm1)^2)


# The value of interest here is -0.018300.
# It's signifixant (see *** in final column), but not the value you got in Node.js it would seem...

# Call:
# lm(formula = pitch ~ ontime + ontime2, data = xy.data)

# Residuals:
   # Min     1Q Median     3Q    Max 
# -9.676 -2.499 -0.784  2.174 11.595 

# Coefficients:
             # Estimate Std. Error t value Pr(>|t|)    
# (Intercept) 71.048830   1.085949  65.426  < 2e-16 ***
# ontime       0.693592   0.090397   7.673 1.71e-12 ***
# ontime2     -0.018300   0.001595 -11.473  < 2e-16 ***
# ---
# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

# Residual standard error: 4.212 on 156 degrees of freedom
# Multiple R-squared:  0.6788,	Adjusted R-squared:  0.6747 
# F-statistic: 164.8 on 2 and 156 DF,  p-value: < 2.2e-16