const fs = require("fs")
const path = require("path")
const Hasher = require("./../Hasher.js").default
const {performance} = require('perf_hooks');

const selfTest = false;

const param = {
  "tMin": 0.5,
  "tMax": 2,
  "pMin": 1,
  "pMax": 6,
  "sampleSize": 50,
  "numDiv": 10,
  "timeBuffer": 50,
  "winSize": 8,
  "jitter": selfTest ? [0] : [0, 0.001, 0.005, 0.01, 0.05, 0.1],
  "binSize": selfTest ? [0.1] : [0.1, 0.5, 1, 5],
  "scale": selfTest ? [1] : [0.75, 0.9, 1, 1.1, 1.25]
}

// Individual user paths.
const mainPaths = {
  "alex": {
    "inputDir": path.join(
      "/Users", "zongyu", "WebstormProjects",
      "aimgef", "pubs_material", "evomusart_extra", "process", "maestro", "data", "maestro_points_train"
    ),
    "outputDir": path.join(
      "/Users", "zongyu", "WebstormProjects",
      "aimgef", "pubs_material", "evomusart_extra", "process", "maestro", "data", "build_hashtable_out"
    ),
  },
  "server": {
    "inputDir": path.join(__dirname, "out", "maestro_points_train"),
    "outputDir": path.join(__dirname, "out", "runtime_test_out"),
  }
}

// Set mainPath according to the given parameter.
const argv = require('minimist')(process.argv.slice(2))
const mainPath = mainPaths[argv.u];

function shuffle(array) {
  let currentIndex = array.length, temporaryValue, randomIndex;
  // While there remain elements to shuffle...
  while (0 !== currentIndex) {
    // Pick a remaining element...
    randomIndex = Math.floor(Math.random() * currentIndex);
    currentIndex -= 1;
    // And swap it with the current element.
    temporaryValue = array[currentIndex];
    array[currentIndex] = array[randomIndex];
    array[randomIndex] = temporaryValue;
  }
  return array
}


let h = new Hasher()
const data = fs.readdirSync(mainPath["inputDir"])
  .filter(function (filename) {
    return path.extname(filename) === ".json"
  })
for (let i = 1; i <= param.numDiv; i++) {
  let report = {}
  let numNotes = 0
  let numEntries = 0
  const cumuTimes = []
  let cumuTime = 0
  let testFiles = shuffle(data).slice(0, Math.ceil(data.length * i / param.numDiv))
  fs.mkdirSync(path.join(mainPath["outputDir"], `${i}-${param.numDiv}`, "fp"), { recursive: true })
  const startTime = performance.now();
  for (const file of testFiles) {
    const points = require(path.join(mainPath["inputDir"], file))
    numNotes += points.length
    const nh = h.create_hash_entries(
      points, cumuTime, path.basename(file, ".json"), "triples", "increment and file",
      param.tMin, param.tMax, param.pMin, param.pMax,
      path.join(mainPath["outputDir"], `${i}-${param.numDiv}`, "fp")
    )
    numEntries += nh
    cumuTimes.push(cumuTime + points[0][0])
    cumuTime += Math.ceil(points.slice(-1)[0][0] + param.timeBuffer)
  }
  cumuTimes.push(cumuTime)
  fs.writeFileSync(
    path.join(mainPath["outputDir"], `${i}-${param.numDiv}`, "fnamTimes.json"),
    JSON.stringify({"filenames": testFiles, "cumuTimes": cumuTimes}, null, 2)
  )
  const endTime = performance.now();

  report["numTestFiles"] = testFiles.length
  report["numNotes"] = numNotes
  report["numEntries"] = numEntries
  report["runtime"] = (endTime - startTime) / 1000
  report["testFiles"] = testFiles
  fs.writeFileSync(
    path.join(mainPath["outputDir"], `${i}-${param.numDiv}`, "report.json"),
    JSON.stringify(report, null, 2)
  )
}
