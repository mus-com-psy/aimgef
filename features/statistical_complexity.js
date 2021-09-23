const path = require("path");
const mm = require("maia-markov");
const fs = require("fs");
const {execSync} = require("child_process");


function getPoints(filename) {
  let points = []
  const co = new mm.MidiImport(filename).compObj
  points = co.notes.map(n => {
    return [n.ontime, n.MNN]
  })
  return points
}

function statistical_complexity(mainPath) {
  const results = {}
  const files = fs.readdirSync(mainPath["midi"])
  let content = ""
  for (const f of files) {
    if (f.slice(-4) === ".mid") {
      const points = getPoints(path.join(mainPath.midi, f))
        .slice(0, 400)
        .map(x => {
          return String.fromCharCode((x[1] % 12) + 97)
        })
      content += points.join("") + "\n"
    }
  }
  fs.writeFileSync(mainPath.CSSR.data, content)
  const cmd = `${mainPath.CSSR.executable} ${mainPath.CSSR.alphabet} ${mainPath.CSSR.data} ${mainPath.CSSR.maxlength}`
  console.log("cmd: ", cmd)
  execSync(cmd)

  const data = fs.readFileSync(mainPath.CSSR.data + "_info", 'utf8').split("\r\n").slice(2, -1)
  for (const row of data) {
    const item = row.split(": ")
    results[item[0]] = item[1]
  }
  return results
}

module.exports = statistical_complexity
