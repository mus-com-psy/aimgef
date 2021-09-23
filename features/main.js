const mm = require("maia-markov");
const sc = require("./statistical_complexity")
const tc = require("./translational_complexity")
const path = require("path");
const mainPaths = {
  "alex": {
    "midi": "C:\\Users\\Zongyu\\Projects\\CSSR\\midi",
    "CSSR": {
      "executable": "C:\\Users\\Zongyu\\Projects\\CSSR\\CSSR.exe",
      "alphabet": "C:\\Users\\Zongyu\\Projects\\CSSR\\data\\alphabet",
      "data": "C:\\Users\\Zongyu\\Projects\\CSSR\\data\\data_10",
      "maxlength" : 1.37
    },
  }
}

const argv = require('minimist')(process.argv.slice(2))
const mainPath = mainPaths[argv.u];

function getPoints(filename, gran=[0, 0.25, 0.33, 0.5, 0.67, 0.75, 1]) {
  let points = []
  const co = new mm.MidiImport(filename, gran).compObj
  points = co.notes.map(n => {
    return [n.ontime, n.MNN]
  })
  return points
}

const points = getPoints("./Lee01M.mid")
console.log()

// const ps1 = [[0, 60], [1, 60], [2, 60], [3, 60], [4, 60], [5, 60]]
// const ans1 = tc(ps1)
// console.log("ans1:", ans1)
//
// const ans2 = sc(mainPath)
// console.log("ans2:", ans2)
