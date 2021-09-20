const path = require("path");
const mm = require("maia-markov");
const fs = require("fs");
const {exec} = require("child_process");

const mainPaths = {
  "alex": {
    "CSSR": path.join(
      "/Users", "zongyu", "MusicFeatures", "CSSR"
    ),
    "midi": path.join(
      "/Users", "zongyu", "MusicFeatures", "CSSR", "test_midi"
    ),
    "outName": "data_10"
  }
}

let nextU = false
let mainPath;
process.argv.forEach(function (arg, ind) {
  if (arg === "-u") {
    nextU = true
  } else if (nextU) {
    mainPath = mainPaths[arg]
    nextU = false
  }
})

function getPoints(filename) {
  let points = []
  const co = new mm.MidiImport(filename).compObj
  points = co.notes.map(n => {
    return [n.ontime, n.MNN]
  })
  return points
}

const files = fs.readdirSync(mainPath["midi"])
const logger = fs.createWriteStream(path.join(mainPath["CSSR"], mainPath["outName"]), {flags: "a"})
for (const f of files) {
  if (f.slice(-4) === ".mid") {
    const points = getPoints(path.join(mainPath["midi"], f))
      .slice(0, 400)
      .map(x => {
        return String.fromCharCode((x[1] % 12) + 97)
      })
    logger.write(points.join("") + "\n")
  }
}
logger.end()

// exec(mainPath["CSSR"] + "/CSSR alphabet " + mainPath["outName"] + " 400", (error, stdout, stderr) => {
//   if (error) {
//     console.log(`error: ${error.message}`);
//     return;
//   }
//   if (stderr) {
//     console.log(`stderr: ${stderr}`);
//     return;
//   }
//   console.log(`stdout: ${stdout}`);
// });
