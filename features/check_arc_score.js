// Following this: http://reliawiki.org/index.php/Multiple_Linear_Regression_Analysis
// Still can't get it to agree with R yet though!

const fs = require("fs")
const PolynomialRegression = require("ml-regression-polynomial")
const distributions = require("distributions")
const {abs, sum, mean, matrix, multiply, inv, transpose} = require("mathjs")
const mu = require("maia-util")

const rows = fs.readFileSync("./arcreport/15-0.796.csv", "utf8")
.split("\n").slice(1, -1)
// console.log("rows:", rows)
let x = [], y = [], X = []
rows.forEach(function(r){
  const s = r.split(",")
  x.push(parseFloat(s[0]))
  y.push(parseInt(s[1]))
  X.push([1, parseFloat(s[0]), Math.pow(parseFloat(s[0]), 2)])
})
console.log("x.length:", x.length)
console.log("y.length:", y.length)
console.log("x.slice(0, 10):", x.slice(0, 10))
console.log("y.slice(0, 10):", y.slice(0, 10))



// Do the regression.
const degree = 2; // setup the maximum degree of the polynomial
const regression = new PolynomialRegression(x, y, degree);
console.log("regression:", regression)

// console.log(regression._predict(80)); // Apply the model to some x value. Prints 2.6.
// console.log(regression.coefficients); // Prints the coefficients in increasing order of power (from 0 to degree).
// console.log(regression.toString(3)); // Prints a human-readable version of the function.
// console.log(regression.toLaTeX());
// console.log(regression.score(x, y));

// beta = 0 under null hypothesis
// beta hat is the estimated coefficient
// RSS = sum((y_hat - y_predicted)^2)
// SS_X = sum((x - mean(x))^2)
// n = number of data sample (x.length)
const betaHat = regression.coefficients[degree]
console.log("betaHat:", betaHat)
const predicted = x.map(n => {
  return regression._predict(n)
})
const RSS = sum(mu.subtract_two_arrays(y, predicted).map(n => {
  return Math.pow(n, 2)
}))
console.log("RSS:", RSS)
const MSE = RSS/x.length
console.log("MSE:", MSE)

let C = inv(multiply(transpose(matrix(X)), matrix(X)))
// Just grab the array property.
C = C["_data"]
console.log("C:", C)
console.log("C[2][2]:", C[2][2])

const v = betaHat / (MSE*C[2][2])
console.log("t-value:", v)
// df = n - (k + 1), where k is number of variables in model.
const df = x.length - regression.coefficients.length
const t = distributions.Studentt(df);
console.log("t.cdf(abs(v)):", t.cdf(abs(v))) // greater than 0.975
