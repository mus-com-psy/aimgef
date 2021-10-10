const feat = require("./features")
const util = require("./util");
const PolynomialRegression = require("ml-regression-polynomial");
const {sum, mean, abs} = require("mathjs");
const mu = require("maia-util");
const distributions = require("distributions");

function arcScore(file) {
  const points = util.getPoints(file, 'top notes')
  const x = points.map(n => {
    return n[1]
  })
  const y = points.map(n => {
    return n[0]
  })
  const degree = 2; // setup the maximum degree of the polynomial
  const regression = new PolynomialRegression(x, y, degree);

  // console.log(regression._predict(80)); // Apply the model to some x value. Prints 2.6.
  // console.log(regression.coefficients); // Prints the coefficients in increasing order of power (from 0 to degree).
  // console.log(regression.toString(3)); // Prints a human-readable version of the function.
  // console.log(regression.toLaTeX());
  // console.log(regression.score(x, y));

  // beta = 0
  // beta hat = coefficient
  // RSS = sum((y_hat - y_predicted)^2)
  // SS_X = sum((x - mean(x))^2)
  // n = number of data sample (x.length)
  // n - 3
  const beta_hat = regression.coefficients[degree]
  const predicted = x.map(n => {
    return regression._predict(n)
  })
  const rss = sum(mu.subtract_two_arrays(y, predicted).map(n => {
    return Math.pow(n, 2)
  }))
  const x_mean = mean(x)
  const ss_x = sum(x.map(n => {
    return Math.pow(n - x_mean, 2)
  }))
  const v = beta_hat / Math.sqrt(rss / (points.length - 2) / ss_x)
  console.log(v)
  const t = distributions.Studentt(points.length - 2);
  console.log(t.cdf(abs(v))) // greater than 0.975
  return [t.cdf(abs(v)), points]
}

const pieces = [15, 104, 116, 236] // ^, v, /, -
const piece = 236
const csvWriter = require('csv-writer').createObjectCsvWriter({
  path: `${piece}.csv`,
  header: [
    {id: 'ontime', title: 'ontime'},
    {id: 'pitch', title: 'pitch'}
  ]
})
const filename = `/home/zongyu/Projects/listening study/midi/${piece}.mid`
const out = arcScore(filename)
const data = out[1].map(n => {
  return {'ontime': n[1], 'pitch': n[0]}
})
csvWriter
  .writeRecords(data)
  .then(()=> console.log('The CSV file was written successfully: ', out[0]));

