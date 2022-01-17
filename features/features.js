const mu = require("maia-util")
const fs = require("fs");
const util = require("./util")
const cp = require("child_process");
const path = require("path")
const PolynomialRegression = require("ml-regression-polynomial");
const distributions = require("distributions");
const {abs, sum, mean, min, max, variance} = require("mathjs");
const {getPoints} = require("./util");
const math = require("mathjs");


function statComp(CSSR, midiDir, rawData, name) {
  let files
  const sequences = []
  if (rawData) {
    files = fs.readdirSync(midiDir).filter(f => f.endsWith(".json"))
    for (const file of files) {
      let seqStr = ""
      let seq = require(`${midiDir}/${file}`)
      for (const s of seq) {
        if (s === 0) {

        } else if (s >= 1 && s <= 128) {
          seqStr += String.fromCharCode(((s - 1) % 12) + 97)
        } else if (s >= 129 && s <= 256) {
          seqStr += String.fromCharCode(((s - 129) % 12) + 65)
        } else if (s >= 257 && s <= 356) {
          seqStr += String.fromCharCode(Math.floor((s - 257) / 10) + 109)
        } else if (s >= 357 && s <= 388) {
          seqStr += String.fromCharCode(Math.floor((s - 357) / 4) + 77)
        } else {
          console.log("Invalid events index: ", s)
        }
      }
      sequences.push(seqStr)
    }
  } else {
    files = fs.readdirSync(midiDir).filter(f => f.endsWith(".mid"))
    for (const file of files) {
      sequences.push(util.points2events(util.getPoints(path.join(midiDir, file), "default", false, null), "mode-2"))
    }
  }
  let sequenceSize = 0
  for (const seq of sequences) {
    sequenceSize += seq.length
  }
  sequenceSize /= sequences.length
  const alphabetSize = fs.readFileSync(CSSR.alphabet, 'utf8').length
  const l = Math.log2(sequenceSize) / Math.log2(alphabetSize)
  let data = sequences.join("\n")
  fs.writeFileSync(path.join(CSSR.data + name), data)
  try {
    cp.execFileSync(
      CSSR.executable, [
        CSSR.alphabet,
        path.join(CSSR.data + name),
        l,
      ]
    )
  } catch (err) {
  }

  // let stdout = cp.execFileSync(
  //   CSSR.executable, [
  //     CSSR.data + name,
  //     CSSR.pastSize,
  //     CSSR.futureSize
  //   ]
  // )
  // return parseFloat(stdout.toString().split("\n")[2].split(": ")[1])
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
  const t = distributions.Studentt(points.length - 2);
  console.log(t.cdf(abs(v))) // greater than 0.975
  return t.cdf(abs(v))
}

// TODO: For maia-features, both CompObj and Model contain emotional features.
// TODO: For arcScore, adjusting degree of freedom does not changing the result value. Also arcScore does not handle when excerpts having multiple arcs.
// TODO: Adding verbose flag for MidiImport? So to not log timeSigs, fsm and comp.notes.length.

function tonalAmb(file) {
  const co = util.getCompObj(file, "mf")
  return co.tonal_ambiguity()
}

function timeInterval(file) {
  // const co = util.getCompObj(file, "mf")
  // return co.average_time_between_attacks() // In the unit of beat.
  const points = getPoints(file, "default", false, null)
    .sort((x, y) => {
      return x[1] - y[1] || x[0] - y[0] || x[2] - y[2]
    })
  const ioi = []
  const kot = []
  const kdt = []
  for (let i = 1; i < points.length; i++) {
    ioi.push(points[i][1] - points[i - 1][1])
    const t = points[i][1] - points[i - 1][2]
    if (t > 0) {
      kdt.push(t)
    } else if (t < 0) {
      kot.push(abs(t))
    }
  }
  if (kdt.length === 0) {
    console.log("\n\n", file, "\n\n")
  }
  const ioi2 = finDiff(ioi)
  return {
    "firstIOI": {
      "mean": ioi.length === 0 ? "" : mean(ioi),
      "variance": ioi.length === 0 ? "" : variance(ioi)
    },
    "secondIOI": {
      "mean": ioi2.length === 0 ? "" : mean(ioi2),
      "variance": ioi2.length === 0 ? "" : variance(ioi2)
    },
    "KOT": {
      "mean": kot.length === 0 ? "" : mean(kot),
      "variance": kot.length === 0 ? "" : variance(kot)
    },
    "KDT": {
      "mean": kdt.length === 0 ? "" : mean(kdt),
      "variance": kdt.length === 0 ? "" : variance(kdt)
    }
  }
}

function jitter(file, grid = 0.25) {
  // Expressiveness
  // TODO: Tom mentioned his beat tracking method. Or tempo changes, but the method in pretty_midi does not work
  let ons = getPoints(file, "pitch and ontime", false, null).map(n => {
    return n[1]
  })
  const head = ons[0]
  ons = ons.map(n => {
    return n - head
  })
  let err = ons.map(n => {
    return min(abs(n - math.floor(n / grid) * grid), abs(n - math.ceil(n / grid) * grid)) / 0.125
  })

  return {
    'array': err,
    'mean': mean(err),
    'variance': variance(err),
    'min': min(err),
    'max': max(err)
  } // TODO: It cannot handle when excerpts are supposed to have different tempi in various places.
}

function finDiff(arr, absDiff = false) {
  const res = []
  let pre = arr[0]
  for (const x of arr.slice(1)) {
    res.push(absDiff ? abs(x - pre) : x - pre)
    pre = x
  }
  return res
}

module.exports = {
  statComp,
  transComp,
  arcScore,
  tonalAmb,
  timeInterval,
  jitter,
}
const arr1 = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
const arr2 = [2, 3, 5, 6, 2, 3, 5, 6, 2, 3, 5, 6]
const arr3 = [2, 2, 2, 3, 3, 3, 5, 5, 5, 6, 6, 6]
const arr4 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

console.log("arr1:",
  {
    "firstIOI": {
      "mean": mean(arr1),
      "variance": variance(arr1)
    },
    "secondIOI": {
      "mean": mean(finDiff(arr1)),
      "variance": variance(finDiff(arr1))
    }
  })
console.log("arr2:",
  {
    "firstIOI": {
      "mean": mean(arr2),
      "variance": variance(arr2)
    },
    "secondIOI": {
      "mean": mean(finDiff(arr2)),
      "variance": variance(finDiff(arr2))
    }
  })
console.log("arr3:",
  {
    "firstIOI": {
      "mean": mean(arr3),
      "variance": variance(arr3)
    },
    "secondIOI": {
      "mean": mean(finDiff(arr3)),
      "variance": variance(finDiff(arr3))
    }
  })

console.log("arr4:",
  {
    "firstIOI": {
      "mean": mean(arr4),
      "variance": variance(arr4)
    },
    "secondIOI": {
      "mean": mean(finDiff(arr4)),
      "variance": variance(finDiff(arr4))
    }
  })

const maxVec = new Array(4 * 3 / 2)
let inc = 0
// const points = [[0, 64], [1, 62], [3, 67],[4, 69]]
const points = [[0, 60, 2], [0, 72, 2], [1, 64, 1], [1, 67, 1]]
// for (let i = 0; i < 4 - 1; i++) {
//   for (let j = i + 1; j < 4; j++) {
//     maxVec[inc++] = mu.subtract_two_arrays(points[j], points[i])
//   }
// }
console.log(mu.fifth_steps_mode(
  points,
  mu.krumhansl_and_kessler_key_profiles));
// console.log(mu.count_rows(maxVec))