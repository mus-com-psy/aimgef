const mm = require("maia-markov");

function getPoints(filename) {
  let points = []
  const co = new mm.MidiImport(filename).compObj
  points = co.notes.map(n => {
    return [n.ontime, n.MNN]
  })
  return points
}

// y = a + bx + cx^2
// maia features to take out top notes
// CompObj(_data, _melodyMode = "top new MNN")
function arc() {

}

