const path = require("path");
const mm = require("maia-markov");
const fs = require("fs");
const util = require("./util")
const cp = require("child_process");


module.exports.statistical_complexity = function (mainPath, groups) {
  const results = {}
  const files = fs.readdirSync(mainPath.midi).filter(f => f.endsWith(".mid"))
  for (const group of groups) {
    const sequences = []
    let midiFiles = files.filter(f => f.startsWith(group + "-"))
    for (const midiFile of midiFiles) {
      sequences.push(util.points2events(util.getPoints(mainPath.midi + midiFile)))
    }
    let data = sequences.join("\n")
    fs.writeFileSync(mainPath.CSSR.data + `group-${group}`, data)
    // let stdout = cp.execFileSync(
    //   mainPath.CSSR.executable, [
    //     mainPath.CSSR.alphabet,
    //     mainPath.CSSR.data + `group-${group}`,
    //     mainPath.CSSR.pastSize,
    //   ]
    // )
    let stdout = cp.execFileSync(
      mainPath.CSSR.executable, [
        mainPath.CSSR.data + `group-${group}`,
        mainPath.CSSR.pastSize,
        mainPath.CSSR.futureSize
      ]
    )
    console.log(stdout.toString())
  }
}
