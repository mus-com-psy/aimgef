const mu = require("maia-util")

module.exports.translational_complexity = function (P) {
  const len = P.length
  // Calculate the difference array, but leave it as a vector.
  const maxVecSize = len * (len - 1) / 2
  const maxVec = new Array(maxVecSize)
  let inc = 0 // Increment to populate V.
  for (let i = 0; i < len - 1; i++) {
    for (let j = i + 1; j < len; j++) {
      maxVec[inc++] = mu.subtract_two_arrays(P[j], P[i])
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


// const ps1 = [[0, 60], [1, 60], [2, 60], [3, 60], [4, 60], [5, 60]]
// const ans1 = translational_complexity(ps1)
// console.log("ans1:", ans1)
//
// const ps2 = [[0, 60], [1, 62], [3, 57], [3.5, 45], [4, 70], [4, 72]]
// const ans2 = translational_complexity(ps2)
// console.log("ans2:", ans2)
//
// const ps3 = [[0, 60], [1, 61], [2, 62], [2, 63], [4, 64], [4, 65]]
// const ans3 = translational_complexity(ps3)
// console.log("ans3:", ans3)
//
// const ps4 = [[0, 60], [0, 64], [1, 60], [1, 64], [2, 60], [2, 64]]
// const ans4 = translational_complexity(ps4)
// console.log("ans4:", ans4)
