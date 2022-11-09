const path = require("path")
const fs = require("fs")
const mm = require("maia-markov")
const {Midi} = require('@tonejs/midi')

const dir = path.join(__dirname, "original", "total")

let tiFiles = fs.readdirSync(dir)
tiFiles = tiFiles.filter(function (tiFile) {
    return tiFile.split(".")[1] === "mid" || tiFile.split(".")[1] === "midi"
})
let npts = 0
tiFiles.forEach(function (file) {
    let points = getPoints(path.join(dir, file), "mm")
    npts += points.length
})
console.log(npts)

function getPoints(filename, mode = "mm") {
    let points = []
    switch (mode) {
        case "mm":
            const co = new mm.MidiImport(filename).compObj
            let staffNos = []
            co.notes.forEach(function (n) {
                if (n.staffNo > 0) {
                    console.log("n:", n)
                }
                if (staffNos[n.staffNo] === undefined) {
                    staffNos[n.staffNo] = 1
                } else {
                    staffNos[n.staffNo]++
                }
                if (n.staffNo > 0) {
                    console.log("staffNos:", staffNos)
                }

            })
            console.log("staffNos:", staffNos)
            points = co.notes.map(n => {
                return [n.ontime, n.MNN]
            })
            break
        case "tone":
            const midiData = fs.readFileSync(filename)
            const midi = new Midi(midiData)
            midi.tracks.forEach(function (track) {
                track.notes.forEach(function (n) {
                    points.push([
                        n.ticks / midi.header.ppq,
                        n.midi,
                        // n.durationTicks / midi.header.ppq,
                        // track.channel,
                        // Math.round(1000 * n.velocity) / 1000
                    ])
                })
            })
            break
    }
    return points
}