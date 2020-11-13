// Copyright Tom Collins, 20.10.2020

// Requires.
const path = require("path")
const fs = require("fs")
const {Midi} = require('@tonejs/midi')
const mu = require("maia-util")

// Parameters
// Individual user paths
const mainPaths = {
    "midi": path.join(__dirname, "compare"),
    "outDir": path.join(__dirname, "results"),
}

let md = [
    {
        "id": "",
        "ontimeWindow": [0, 0]
    },
    {
        "id": "",
        "ontimeWindow": [0, 0]
    },
]

const windowOverlapSizes = [
    {"winSize": 16, "overlap": 8},
    // { "winSize": 8, "overlap": 4 },
    // { "winSize": 4, "overlap": 2 }
]

// Parameter to try to ensure we're handling integers.
const sf = 24

// Grab command line args to set path to data.
let nextTgt = false
let nextSrc = false
process.argv.forEach(function (arg) {
    if (arg === "-src") {
        nextTgt = false
        nextSrc = true
    } else if (nextSrc && !(arg === "-tgt")) {
        let src = arg.split("-")
        md[0]["id"] = src[0]
        md[0]["ontimeWindow"] = [Number(src[1]), Number(src[2])]
    }
    if (arg === "-tgt") {
        nextSrc = false
        nextTgt = true
    } else if (nextTgt && !(arg === "-src")) {
        let tgt = arg.split("-")
        md[1]["id"] = tgt[0]
        md[1]["ontimeWindow"] = [Number(tgt[1]), Number(tgt[2])]
    }
})

// Import and the MIDI files, do the usual conversions, and write to JS
// variables.
md.forEach(function (mdEntry, iFile) {
    console.log("mdEntry.id:", mdEntry.id)
    if (iFile % 10 === 0) {
        console.log("!!! PFILE " + (iFile + 1) + " OF " + md.length + " !!!")
    }
    try {
        const midiData = fs.readFileSync(path.join(mainPaths["midi"], mdEntry.id + ".mid"))
        const midi = new Midi(midiData)
        const timeSigs = [midi.header.timeSignatures.map(function (ts) {
            return {
                "barNo": ts.measures + 1,
                "topNo": ts.timeSignature[0],
                "bottomNo": ts.timeSignature[1],
                "ontime": ts.ticks / midi.header.ppq
            }
        })[0]] // SUPER HACKY. REVISE LATER!
        console.log("timeSigs:", timeSigs)
        let points = []
        midi.tracks.forEach(function (track) {
            // if (track.channel == targetChannel){
            track.notes.forEach(function (n) {
                points.push([
                    n.ticks / midi.header.ppq,
                    n.midi,
                    n.durationTicks / midi.header.ppq,
                    track.channel,
                    Math.round(1000 * n.velocity) / 1000
                ])
            })
            // }
        })
        points = mu.sort_rows(points)[0]
        // console.log("points.slice(0, 50):", points.slice(0, 50))
        const fsm = mu.fifth_steps_mode(points, mu.krumhansl_and_kessler_key_profiles)
        console.log("fsm:", fsm)
        points.forEach(function (p) {
            p.splice(2, 0, mu.guess_morphetic(p[1], fsm[2], fsm[3]))
        })
        mdEntry["allPoints"] = points
        mdEntry["roi"] = mu.points_belonging_to_interval(
            points, mdEntry.ontimeWindow[0], mdEntry.ontimeWindow[1]
        )
    } catch (e) {
        console.log(e)
    }
})
console.log("md:", md)


// windowOverlapSizes.forEach(function(wo){
//   let win = wo.winSize
//   let overlap = wo.overlap

let comparisons = []
md.forEach(function (xcrt0, i) {
    md.forEach(function (xcrt1, j) {
        if (j > i) {
            let obj = {
                "id0": xcrt0.id,
                "id1": xcrt1.id,
            }

            const a = mu.unique_rows(
                xcrt0.roi.map(function (p) {
                    return [Math.round(sf * p[0]), p[2]]
                })
            )[0]
            const b = mu.unique_rows(
                xcrt1.roi.map(function (p) {
                    return [Math.round(sf * p[0]), p[2]]
                })
            )[0]
            const cs = mu.cardinality_score(a, b)
            obj.cs = cs[0]
            obj.maxVec = [cs[1][0] / sf, cs[1][1]]

            obj.p0 = mu.unique_rows(
                xcrt0.roi.map(function (p) {
                    return [p[0], p[2]]
                })
            )[0]
            obj.p1 = mu.unique_rows(
                xcrt1.roi.map(function (p) {
                    return [p[0], p[2]]
                })
            )[0]

            comparisons.push(obj)
        }
    })
})
console.log("comparisons:", comparisons)

// Write the point sets to file so we can make a figure.
comparisons.forEach(function (c) {
    fs.writeFileSync(
        path.join(mainPaths["outDir"], "point_set_roi_" + c.id0 + ".js"),
        JSON.stringify(c.p0, null, 2)
    )
    fs.writeFileSync(
        path.join(mainPaths["outDir"], "point_set_roi_" + c.id1 + ".js"),
        JSON.stringify(c.p1, null, 2)
    )
})
