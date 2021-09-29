const feat = require("./features")

const mainPaths = {
  "alex": {
    "midiFile": "/home/zongyu/Projects/aimgef/features/data/LeeSH01M.mid",
    "CSSR": {
      "midi": "/home/zongyu/Projects/MusicTransformer-pytorch/generated/",
      "executable": "/home/zongyu/Projects/decisional_states-1.0/examples/SymbolicSeries",
      // "executable": "/home/zongyu/Projects/CSSR/CSSR",
      "alphabet": "/home/zongyu/Projects/CSSR/alphabet-mode-1",
      "data": "/home/zongyu/Projects/aimgef/features/data/",
      "pastSize": 1,
      "futureSize": 1
    },
  }
}

const argv = require('minimist')(process.argv.slice(2))
const mainPath = mainPaths[argv.u];
// let pts = util.getPoints(mainPath.midi + fs.readdirSync(mainPath.midi)[0])
// console.log("pts: ", pts)
// let events1 = util.points2events(pts, "mode-1")
// console.log("events with mode-1: ", events1)
// let events2 = util.points2events(pts, "mode-2")
// console.log("events with mode-2: ", events2)


// let groups = []
// for (let i = 0; i < 80; i++) {
//   groups.push(i.toString())
// }
//
// const out = sc.statistical_complexity(mainPath, groups)
// console.log(out)

// feat.translational_complexity()
const mf = require("maia-features")

console.log(feat.arcScore(mainPath.midiFile))
console.log(feat.tonalScore(mainPath.midiFile))