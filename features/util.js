const mm = require("maia-markov")
const mf = require("maia-features")
const tone = require("@tonejs/midi")

function getTimeEvents(time, gran) {
  let results = ""
  const granTime = Object.keys(gran).sort((a, b) => {
    return b - a
  })
  let quotient = Math.floor(time / granTime[0])
  let remainder = time % granTime[0]
  let closest = granTime.reduce((a, b) => {
    return Math.abs(parseFloat(b) - remainder) < Math.abs(parseFloat(a) - remainder) ? b : a;
  })
  for (let i = 0; i < quotient; i++) {
    results += gran[granTime[0]]
  }
  results += gran[closest]
  return results
}

function getCompObj(file, mode = "mm", gran = [0, 0.25, 0.33, 0.5, 0.67, 0.75, 1]) {
  const co = new mm.MidiImport(file, gran).compObj
  switch (mode) {
    case "mm":
      return co
    case "mf":
      return new mf.CompObj(co)
  }
}

function getPoints(filename,
                   mode,
                   clean = true,
                   gran = [0, 0.25, 0.33, 0.5, 0.67, 0.75, 1]) {
  const co = new mm.MidiImport(filename, gran).compObj
  let points = co.notes.map(n => {
    return [n.MNN, n.ontime, n.offtime]
  })
  points = Array.from(new Set(points.map(n => {
    return JSON.stringify(n)
  }))).map(n => {
    return JSON.parse(n)
  }).sort((x, y) => {
    return x[1] - y[1] || x[0] - y[0] || x[2] - y[2]
  })
  if (clean) {
    const results = []
    results.push(points[0])
    let prev = points[0]
    for (const pt of points.slice(1)) {
      if (!(pt[1] >= prev[1] && pt[1] < prev[2] && pt[0] === prev[0])) {
        results.push(pt)
        prev = pt
      }
    }
    points = results
  }
  switch (mode) {
    case "pitch and ontime":
      console.log('Getting points in "pitch and ontime" mode.')
      points = points.map(n => {
        return [n[0], n[1]]
      })
      break
    case "top notes":
      console.log('Getting points in "top notes" mode.')
      points = new mf.CompObj(co, "top new MNN").melodyPoints.map(n => {
        return [n[1], n[0]]
      })
      break
    default:
      console.log("Getting points in default mode.")
  }
  return points
}

function points2events(points, mode = "mode-1", gran = {
  0: "",
  0.25: "1",
  0.33: "2",
  0.5: "3",
  0.67: "4",
  0.75: "5",
  1: "6"
}) {
  let results = ""
  switch (mode) {
    case "mode-0":
      for (const pt of points) {
        results += String.fromCharCode((pt[0] % 12) + 97)
      }
      break
    case "mode-1":
      let timestamp = points[0][1]
      for (const pt of points) {
        let td = pt[1] - timestamp
        if (td > 0) {
          results += getTimeEvents(td, gran)
        }
        results += String.fromCharCode((pt[0] % 12) + 97)
        timestamp = pt[1]
      }
      break
    case "mode-2":
      const decomposed = []
      for (const pt of points) {
        decomposed.push([pt[0], pt[1], "on"])
        decomposed.push([pt[0], pt[2], "off"])
      }
      decomposed.sort((x, y) => {
        return x[1] - y[1] || x[0] - y[0]
      })
      let ts = decomposed[0][1]
      for (const pt of decomposed) {
        let td = pt[1] - ts
        if (td > 0) {
          results += getTimeEvents(td, gran)
        }
        if (pt[2] === "on") {
          results += String.fromCharCode((pt[0] % 12) + 97)
        } else if (pt[2] === "off") {
          results += String.fromCharCode((pt[0] % 12) + 65)
        }
        ts = pt[1]
      }
      break
    default:
      console.log(`Wrong mode (${mode})!`)
  }
  console.log("Events sequence length: ", results.length)
  return results
}

module.exports = {
  getPoints,
  getCompObj,
  points2events,
}