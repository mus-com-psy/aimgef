const path = require("path")
const fs = require("fs")
const Hasher = require("./Hasher")
const mm = require("maia-markov")
const {Midi} = require('@tonejs/midi')


const dir = path.join(__dirname, "original", "train")
const can = path.join(__dirname, "candidates", "transformer_train")
let h = new Hasher("./out/lookup.json")
// build(h, dir, "duples")


let tiFiles = fs.readdirSync(dir)
tiFiles = tiFiles.filter(function (tiFile) {
    return tiFile.split(".")[1] === "mid" || tiFile.split(".")[1] === "midi"
})

let canFiles = fs.readdirSync(can)
canFiles = canFiles.filter(function (canFile) {
    return canFile.split(".")[1] === "mid" || canFile.split(".")[1] === "midi"
})

const windowOverlapSizes = {"winSize": 16, "overlap": 8}


for (const file of canFiles) {
    let out = {"nh": [], "ctimes": {}}
    tiFiles.forEach(function (file) {
        out.ctimes[file] = []
    })
    let points = getPoints(path.join(can, file), "mm")
    let win = windowOverlapSizes.winSize
    let overlap = windowOverlapSizes.overlap
    for (let i = 0; i < Math.floor((points[points.length - 1][0] - win) / (win - overlap)); i++) {
        let sub = points.filter(p => {
            return (i * (win - overlap)) <= p[0] && p[0] < (i * (win - overlap) + win)
        })
        let matches = h.match_hash_entries(sub)
        out.nh.push(matches.nosHashes)
        for (const [key, value] of Object.entries(matches.results)) {
            out.ctimes[key].push(value)
        }

        // const hist = h.histogram(
        //     matches,
        //     cumulativeTimes,
        //     pieceNames,
        //     4000, "duples"
        // )
        // let b = 0
    }
    fs.mkdir(`./out/results/${file.split(".")[0]}`, { recursive: true }, (err) => {
        if (err) {
            throw err
        } else {
            fs.writeFileSync(
                `./out/results/${file.split(".")[0]}/nh.json`,
                JSON.stringify(out.nh)
            )
            for (const [key, value] of Object.entries(out.ctimes)) {
                let c = 0
                for (const v of value) {
                    fs.writeFileSync(
                        `./out/results/${file.split(".")[0]}/${key}-${c}.json`,
                        JSON.stringify(value)
                    )
                    c += 1
                }

            }
        }
    });

    // fs.writeFileSync(
    //     path.join(__dirname, "out", "transformer", file.split(".")[0] + ".json"),
    //     JSON.stringify(out)
    // )
}
//
//
// let filename = path.join(__dirname, "candidates", "transformer_train", "0.mid")

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

function build(h, dir, mode = "triples") {
    const filenames = fs.readdirSync(dir)
        .filter(function (filename) {
            return /\.mid$/.test(filename)
        })
    console.log("filenames", filenames)
    filenames.forEach(function (filename) {
        console.log("-- Hashing ", filename)
        const co = new mm.MidiImport(
            path.join(dir, filename)
        ).compObj
        // Count up the staffNos and filter on the one that is most numerous.
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
        const points = co.notes.map(n => {
            return [n.ontime, n.MNN]
        })
        h.create_hash_entries(points, path.basename(filename), mode)
    })
    fs.writeFileSync(
        path.join(__dirname, "out", "lookup.json"),
        JSON.stringify(h.map)
    )
}