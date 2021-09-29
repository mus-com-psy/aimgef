const mu = require("maia-util")
const fs = require("fs");
const util = require("./util")
const cp = require("child_process");
const PolynomialRegression = require("ml-regression-polynomial");
const distributions = require("distributions");
const {abs, sum, mean} = require("mathjs");
const {getPoints} = require("./util");


function statComp(CSSR, name = 'data') {
  const files = fs.readdirSync(CSSR.midi).filter(f => f.endsWith(".mid"))
  const sequences = []
  for (const file of files) {
    sequences.push(util.points2events(util.getPoints(CSSR.midi + file, true), "mode-2"))
  }
  let data = sequences.join("\n")
  fs.writeFileSync(CSSR.data + name, data)
  // let stdout = cp.execFileSync(
  //   CSSR.executable, [
  //     CSSR.alphabet,
  //     CSSR.data + name,
  //     CSSR.pastSize,
  //   ]
  // )
  let stdout = cp.execFileSync(
    CSSR.executable, [
      CSSR.data + name,
      CSSR.pastSize,
      CSSR.futureSize
    ]
  )
  return parseFloat(stdout.toString().split("\n")[2].split(": ")[1])
}


function transComp(file) {
  const points = util.getPoints(file, 'pitch and ontime')
  const len = points.length
  // Calculate the difference array, but leave it as a vector.
  const maxVecSize = len * (len - 1) / 2
  const maxVec = new Array(maxVecSize)
  let inc = 0 // Increment to populate V.
  for (let i = 0; i < len - 1; i++) {
    for (let j = i + 1; j < len; j++) {
      maxVec[inc++] = mu.subtract_two_arrays(points[j], points[i])
    }
  }
  // console.log("maxVec:", maxVec)
  const uniqueCount = mu.count_rows(maxVec)
  // console.log("uniqueCount:", uniqueCount)
  // const ans = uniqueCount[0].length / maxVecSize
  // The smallest value one could get for uniqueCount[0].length is len - 1 (e.g., an
  // isochronous repeated pitch, or an isochronous ascending/descending scale).
  // The largest value one could get for for uniqueCount[0].length is maxVecSize (e.g.,
  // each entry in maxVec is unique, which is possible to handcraft for small
  // point sets, but actually we'd never encounter in a classical excerpt
  // because it is "structured").
  // So answerBounded01 provides the ratio bounded to [0, 1], with 0 for "least
  // complex" and 1 for "most complex".
  // norm = (x - min) / (max - min)
  return (uniqueCount[0].length - (len - 1)) / (maxVecSize - (len - 1))
}

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
  const t = distributions.Studentt(1000);
  console.log(t.cdf(abs(v))) // greater than 0.975
  return t.cdf(abs(v))
}

// TODO: For maia-features, both CompObj and Model contain emotional features.
// TODO: For arcScore, adjusting degree of freedom does not changing the result value. Also arcScore does not handle when excerpts having multiple arcs.
// TODO: Adding verbose flag for MidiImport? So to not log timeSigs, fsm and comp.notes.length.

function tonalScore(file) {
  const co = util.getCompObj(file, "mf")
  return co.tonal_ambiguity()
}

function attInterval(file) {
  const co = util.getCompObj(file, "mf")
  return co.average_time_between_attacks() // In the unit of beat.
}

function rhyDis(file, grid=0.25, fineness = 0.01) {
  // TODO: Tom mentioned his beat tracking method. Or tempo changes, but the method in pretty_midi does not work
  const ons = getPoints(file, "pitch and ontime", false, undefined).map(n => {
    return n[1]
  })
  const results = {"df": 0, "err": grid}
  for (let df = 0; df < grid; df += fineness) {
    const err = mean(ons.map(n => {
      return n % grid // Distance to the grid lines.
    }))
    if (err < results.err) {
      results.df = df
      results.err = err
    }
  }
  return results // TODO: It cannot handle when excerpts are supposed to have different tempi in various places.
}

module.exports = {
  statComp,
  transComp,
  arcScore,
  tonalScore,
  attInterval,
  rhyDis,
}

