const mu = require("maia-util")
function translational_complexity(P){
  const m = P.length
  let numerator, denominator

  // Calculate the difference array, but leave it as a vector.
  const k = P[0].length
  const bigM = m*(m-1)/2
  let bigV = new Array(bigM)
  let bigL = 0 // Increment to populate V.
  for (let i = 0; i < m - 1; i++){
    for (let j = i + 1; j < m; j++){
      bigV[bigL] = mu.subtract_two_arrays(P[j], P[i])
      bigL++
    }
  }
  // console.log("bigV:", bigV)
  const bigV2 = mu.count_rows(bigV)
  // console.log("bigV2:", bigV2)
  const ans = bigV2[0].length/bigM
  // The smallest value one could get for bigV2[0].length is m - 1 (e.g., an
  // isochronous repeated pitch, or an isochronous ascending/descending scale).
  // The largest value one could get for for bigV2[0].length is bigM (e.g.,
  // each entry in bigV is unique, which is possible to handcraft for small
  // point sets, but actually we'd never encounter in a classical excerpt
  // because it is "structured").
  // So answerBounded01 provides the ratio bounded to [0, 1], with 0 for "least
  // complex" and 1 for "most complex".
  const answerBounded01 = (bigV2[0].length - (m - 1))/(bigM - (m - 1))
  return answerBounded01
}

const ps1 = [[0, 60], [1, 60], [2, 60], [3, 60], [4, 60], [5, 60]]
const ans1 = translational_complexity(ps1)
console.log("ans1:", ans1)

const ps2 = [[0, 60], [1, 62], [3, 57], [3.5, 45], [4, 70], [4, 72]]
const ans2 = translational_complexity(ps2)
console.log("ans2:", ans2)

const ps3 = [[0, 60], [1, 61], [2, 62], [2, 63], [4, 64], [4, 65]]
const ans3 = translational_complexity(ps3)
console.log("ans3:", ans3)

const ps4 = [[0, 60], [0, 64], [1, 60], [1, 64], [2, 60], [2, 64]]
const ans4 = translational_complexity(ps4)
console.log("ans4:", ans4)
