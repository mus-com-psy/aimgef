const path = require("path")
const fs = require("fs")
const Hasher = require("./Hasher")
const mu = require("maia-util")
const mm = require("maia-markov")
const {Midi} = require('@tonejs/midi')

const dir = path.join(__dirname, "original", "validation")
// let h = new Hasher("./out/lookup_train.json")
let h = new Hasher()
let b = build(h, dir)
// let tiFiles = fs.readdirSync(dir)
// tiFiles = tiFiles.filter(function (tiFile) {
//     return tiFile.split(".")[1] === "mid" || tiFile.split(".")[1] === "midi"
// })
// let a = getPoints(path.join(__dirname, "original", "train", "1207.mid"))
// const windowOverlapSizes = [
//     {"winSize": 16, "overlap": 8},
//     // {"winSize": 8, "overlap": 4},
//     // {"winSize": 4, "overlap": 2}
// ]
//
// tiFiles.forEach(function (file) {
//     let points = getPoints(path.join(dir, file), "mm")
//     windowOverlapSizes.forEach(function (wo) {
//         let win = wo.winSize
//         let overlap = wo.overlap
//         for (let i = 0; i < Math.floor((points[points.length - 1][0] - win) / (win - overlap)); i++) {
//             let sub = points.filter(p => {
//                 return (i * (win - overlap)) <= p[0] && p[0] < (i * (win - overlap) + win)
//             })
//             let matches = h.match_hash_entries(sub)
//             const hist = h.histogram(
//                 matches,
//                 cumulativeTimes,
//                 pieceNames,
//                 4000, "duples"
//             )
//         }
//
//     })
// })
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

function build(h, dir, mode="triples") {
    // Flat approach
    const filenames = fs.readdirSync(dir)
        .filter(function (fnam) {
            return /\.mid$/.test(fnam)
        })
    // Recursive approach
    // const filenames = []
    // fromDir(dir, /\.xml$/, function(filename){
    //   filenames.push(filename)
    // })
    console.log("filenames", filenames)
    // const cumuTimes = []
    // let cumuTime = 0
    filenames.forEach(function (fname) {
        console.log("-- Hashing ", fname)
        const co = new mm.MidiImport(
            path.join(dir, fname)
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
        console.log("staffNos:", staffNos)
        const points = co.notes.map(n => {
            return [n.ontime, n.MNN]
        })
        h.create_hash_entries(points, path.basename(fname), mode)
        // cumuTimes.push(cumuTime)
        // cumuTime += Math.ceil(points.slice(-1)[0][0] + 50)
    })
    fs.writeFileSync(
        path.join(__dirname, "out", "lookup.json"),
        JSON.stringify(h.map)
    )
    // return {"filenames": filenames, "cumuTimes": cumuTimes}
    return h.map
}