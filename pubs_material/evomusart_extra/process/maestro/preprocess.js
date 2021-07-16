const path = require("path")
const fs = require("fs")
const mm = require("maia-markov")
const {Midi} = require('@tonejs/midi')

fs.readFile('./maestro-v3.0.0/maestro-v3.0.0.json',
    'utf8', (err, jsonString) => {
        if (err) {
            console.log("Error reading maestro index file:", err)
            return
        }
        try {
            const index = JSON.parse(jsonString)
            fs.writeFileSync(
                    "./data/index.json",
                    JSON.stringify(index["midi_filename"])
                )
            for (const [i, filename] of Object.entries(index["midi_filename"])) {
                console.log(`[PROCESSING]\t${String(i).padStart(4, '_')}\t${filename}`)
                let points = getPoints(path.join(__dirname, "maestro-v3.0.0", String(filename)), "tone")
                const output = path.join(
                        __dirname,
                        "maestro-v3.0.0",
                        String(filename).split(".").slice(0, -1).join('.') + ".json")
                fs.writeFileSync(
                    output,
                    JSON.stringify(points)
                )
                console.log(`[DONE]\t${output}\n`)
            }
        } catch (err) {
            console.log('Error parsing JSON string:', err)
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
                        // n.ticks / midi.header.ppq,
                        n.time,
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