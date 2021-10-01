const util = require("./util")

const mainPaths = {
  "alex": {
    "midiFile": "/home/zongyu/Projects/listening study/midi/151.mid",
  },
  "tom": {
    "midiFile": "./151.mid",
  }
}

const argv = require('minimist')(process.argv.slice(2))
const mainPath = mainPaths[argv.u]

const pts = util.getPoints(mainPath.midiFile, "pitch and ontime", true, null)
console.log("pts.slice(0, 10):", pts.slice(0, 10))
