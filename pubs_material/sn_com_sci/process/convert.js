const path = require("path")
const fs = require("fs")
const mm = require("maia-markov")
const {Midi} = require('@tonejs/midi')


const ori_dir = path.join(__dirname, "original", "train")
const can_dir = path.join(__dirname, "original", "validation")

let dirs = [ori_dir, can_dir]
dirs.forEach(function (dir) {
    let files = fs.readdirSync(dir)
    files = files.filter(function (file) {
        return file.split(".")[1] === "mid" || file.split(".")[1] === "midi"
    })
    for (const file of files) {
        let points = getPoints(path.join(dir, file), "mm")
        fs.writeFileSync(
            path.join(dir, file.split(".")[0] + ".json"),
            JSON.stringify(points)
        )
    }
})


function getPoints(filename, mode = "mm") {
    let points = []
    switch (mode) {
        case "mm":
            const co = new mm.MidiImport(filename).compObj
            let staffNos = []
            co.notes.forEach(function (n) {
                if (n.staffNo > 0) {
                    // console.log("n:", n)
                }
                if (staffNos[n.staffNo] === undefined) {
                    staffNos[n.staffNo] = 1
                } else {
                    staffNos[n.staffNo]++
                }
                if (n.staffNo > 0) {
                    // console.log("staffNos:", staffNos)
                }

            })
            // console.log("staffNos:", staffNos)
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