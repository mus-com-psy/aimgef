// Copyright Tom Collins, 20.8.2020

// Requires.
const path = require("path")
const fs = require("fs")
// const uu = require("uuid/v4")
const {Midi} = require('@tonejs/midi')
const mu = require("maia-util")
// const an = require("./analyze")
// const plotlib = require('nodeplotlib')

// Parameters
const mainPaths = {
    "midi": path.join(__dirname, "original"),
    "midiDirs": [],
    "testItems": "maia_markov",
    "outputDir": path.join(__dirname, "results"),
}
// const targetChannel = 0 // Analyzing all channels here.
const windowOverlapSizes = [
    {"winSize": 16, "overlap": 8},
    {"winSize": 8, "overlap": 4},
    {"winSize": 4, "overlap": 2}
]

// Grab user name from command line to set path to data.
let nextDirs = false
let nextItems = false
let mainPath = mainPaths
process.argv.forEach(function (arg) {
    if (arg === "-midiDirs") {
        nextItems = false
        nextDirs = true
    } else if (nextDirs && !(arg === "-testItems")) {
        mainPath["midiDirs"].push(arg)
    }
    if (arg === "-testItems") {
        nextDirs = false
        nextItems = true
    } else if (nextItems && !(arg === "-midiDirs")) {
        mainPath["testItems"] = arg
        nextItems = true
    }
})
// fs.mkdir(outputDir)


// Import and the MIDI files that act as the comparison set.
let pointSets = []
let midiDirs = fs.readdirSync(mainPath["midi"])
// console.log("midiDirs:", midiDirs)
midiDirs = midiDirs.filter(function (midiDir) {
    return mainPath["midiDirs"].indexOf(midiDir) >= 0
})
console.log("midiDirs:", midiDirs)
midiDirs.forEach(function (midiDir, jDir) {
    console.log("Working on midiDir:", midiDir, "jDir:", jDir)
    let pFiles = fs.readdirSync(path.join(mainPath["midi"], midiDir))
    pFiles = pFiles.filter(function (pFile) {
        return pFile.split(".")[1] === "mid"
        // && pFile.split(".")[0] == "2327"
    })
    console.log("pFiles.length:", pFiles.length)

    pFiles.forEach(function (pFile, iFile) {
        // console.log("pFile:", pFile)
        // if (iFile % 10 === 0) {
        //     console.log("!!! PFILE " + (iFile + 1) + " OF " + pFiles.length + " !!!")
        // }
        try {
            const midiData = fs.readFileSync(path.join(mainPath["midi"], midiDir, pFile))
            const midi = new Midi(midiData)
            // const timeSigs = [midi.header.timeSignatures.map(function (ts) {
            //     return {
            //         "barNo": ts.measures + 1,
            //         "topNo": ts.timeSignature[0],
            //         "bottomNo": ts.timeSignature[1],
            //         "ontime": ts.ticks / midi.header.ppq
            //     }
            // })[0]] // SUPER HACKY. REVISE LATER!
            // console.log("timeSigs:", timeSigs)
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
            // console.log("fsm:", fsm)
            points.forEach(function (p) {
                p.splice(2, 0, mu.guess_morphetic(p[1], fsm[2], fsm[3]))
            })
            // let comp = an.note_point_set2comp_obj(
            //   points, timeSigs, false, [0, 1/4, 1/2, 3/4, 1]//, [0, 1/6, 1/4, 1/3, 1/2, 2/3, 3/4, 5/6, 1]
            // )
            // console.log("comp:", comp)
            // Strip off file extension.
            pFile = pFile.split(".")[0]
            // comp["id"] = pFile
            // comp["id"] = uu()
            // comp["idGmd"] = pFile
            // comp["name"] = pFile
            // comp["name"] = midi.header.name || mFile.split(".")[0] // "_new"
            // comp["composers"] = [{"id": "default_composer", "name": "none", "displayName": "None"}]
            pointSets.push({"id": pFile, "points": points})
        } catch (e) {
            console.log(e)
        }
    })
}) // midiDirs.forEach()


// Load the test items.
let tiFiles = fs.readdirSync(path.join(__dirname, "candidates", mainPath["testItems"]))
tiFiles = tiFiles.filter(function (tiFile) {
    return tiFile.split(".")[1] === "mid" || tiFile.split(".")[1] === "midi"
    // && pFile.split(".")[0] == "2327"
})
let segs = {}
windowOverlapSizes.forEach(function (wo) {
    let win = wo.winSize
    let overlap = wo.overlap
    segs[`${win}-${overlap}`] = []
    // pointSets.slice(0, 5).forEach(function(c){
    pointSets.forEach(function (c) {
        const points = c.points
        // const points = mu.comp_obj2note_point_set(c)
        let ontimeInSrc = 0
        let lastOntime = points[points.length - 1][0]
        while (ontimeInSrc <= lastOntime - win) {
            let obj = {
                "pieceId": c.id,
                "ontimeInSrc": ontimeInSrc,
                "points": mu.points_belonging_to_interval(points, ontimeInSrc, ontimeInSrc + win)
            }
            segs[`${win}-${overlap}`].push(obj)
            ontimeInSrc += overlap
        }
    })
})
console.log("tiFiles.length:", tiFiles.length)
tiFiles.forEach(function (tiFile, tiFIdx) {
    // console.log("Working on " + tiFile.split(".")[0] + ", test item " + (tiFIdx + 1) + " of " + tiFiles.length + ".")
    const tiMidiData = fs.readFileSync(path.join(__dirname, "candidates", mainPath["testItems"], `${tiFile}`))
    const tiMidi = new Midi(tiMidiData)
    let tiPoints = []
    tiMidi.tracks.forEach(function (track) {
        // if (track.channel == targetChannel){
        track.notes.forEach(function (n) {
            tiPoints.push([
                n.ticks / tiMidi.header.ppq,
                n.midi,
                n.durationTicks / tiMidi.header.ppq,
                track.channel,
                Math.round(1000 * n.velocity) / 1000
            ])
        })
    })
    tiPoints = mu.sort_rows(tiPoints)[0]
    // console.log("tiPoints.slice(0, 50):", tiPoints.slice(0, 50))
    const tiFsm = mu.fifth_steps_mode(tiPoints, mu.krumhansl_and_kessler_key_profiles)
    // console.log("tiFsm:", tiFsm)
    tiPoints.forEach(function (p) {
        p.splice(2, 0, mu.guess_morphetic(p[1], tiFsm[2], tiFsm[3]))
    })
    // console.log("tiPoints.slice(0, 5):", tiPoints.slice(0, 5))


    // Compute similarities to the comparison set.
    let results = {}
    windowOverlapSizes.forEach(function (wo) {
        // let segs = []
        // pointSets.slice(0, 5).forEach(function(c){
        // pointSets.forEach(function (c) {
        //     const points = c.points
        //     // const points = mu.comp_obj2note_point_set(c)
        //     let ontimeInSrc = 0
        //     let win = wo.winSize
        //     let overlap = wo.overlap
        //     let lastOntime = points[points.length - 1][0]
        //
        //     while (ontimeInSrc <= lastOntime - win) {
        //         let obj = {
        //             "pieceId": c.id,
        //             "ontimeInSrc": ontimeInSrc,
        //             "points": mu.points_belonging_to_interval(points, ontimeInSrc, ontimeInSrc + win)
        //         }
        //         segs.push(obj)
        //         ontimeInSrc += overlap
        //     }
        // })
        // console.log("segs.length:", segs.length)

        let ontimeInGen = 0
        // let genSegmentOntimes = []
        let win = wo.winSize
        let overlap = wo.overlap
        let lastOntime = tiPoints[tiPoints.length - 1][0]
        let maxSimilarities = []
        while (ontimeInGen <= lastOntime - win) {
            console.log(`ontimeInGen: ${ontimeInGen} + ${win} / ${lastOntime} ... FileIndex: ${tiFIdx + 1} / ${tiFiles.length}`)
            // genSegmentOntimes.push(ontimeInGen)
            let obj = {
                "ontimeInGen": ontimeInGen,
                "maxSimilarity": null,
                "maxPieceId": null,
                "maxPoints": null
            }
            let tiPointsSegment = mu.points_belonging_to_interval(tiPoints, ontimeInGen, ontimeInGen + win)
            // console.log("tiPointsSegment:", tiPointsSegment)

            // Calculate the similarities.
            let src = segs[`${win}-${overlap}`].map(function (seg) {
                return seg.pieceId
            })
            let cardScores = segs[`${win}-${overlap}`].map(function (seg) {
                let cs = 0
                if (seg.points.length > 0 && tiPointsSegment.length > 0) {
                    const a = mu.unique_rows(
                        seg.points.map(function (p) {
                            return [Math.round(24 * p[0]), p[2]]
                        })
                    )[0]
                    const b = mu.unique_rows(
                        tiPointsSegment.map(function (p) {
                            return [Math.round(24 * p[0]), p[2]]
                        })
                    )[0]
                    cs = mu.cardinality_score(a, b)
                }
                return cs[0]
            })
            const ma = mu.max_argmax(cardScores)
            obj.maxSimilarity = ma[0]
            obj.maxPieceId = src[ma[1]]
            obj.maxPoints = segs[`${win}-${overlap}`][ma[1]]
            maxSimilarities.push(obj)

            ontimeInGen += overlap
        }
        // console.log("maxSimilarities:", maxSimilarities)
        console.log(
            "mu.mean():",
            mu.mean(maxSimilarities.map(function (ms) {
                return ms.maxSimilarity
            }))
        )
        // console.log(tiFile.split(".")[0] + `,${mu.mean(maxSimilarities.map(function (ms) {
        //     return ms.maxSimilarity
        // }))}`)
        // console.log(
        //   "mu.count_rows():",
        //   mu.count_rows(maxSimilarities.map(function(ms){ return [ms.maxPieceId] }))
        // )

        // Plot it.
        // let data = [{
        //     "x": genSegmentOntimes,
        //     "y": maxSimilarities.map(function (ms) {
        //         return ms.maxSimilarity
        //     }),
        //     "type": "line",
        // }]
        // const layout = {
        //     "yaxis": {"range": [0, 1]},
        //     "title": {"text": tiFile}
        // }

        // console.log("data[0]:", data[0])
        // plotlib.stack(data, layout)
        results[`wo-${wo.winSize}-${wo.overlap}`] = {"maxSimilarities": maxSimilarities}
    }); // windowOverlapSizes.forEach()
    fs.writeFileSync(path.join(mainPath["outputDir"], `${tiFile.split(".")[0]}.json`), JSON.stringify(results));
}) // tiFiles.forEach(

// plotlib.plot()
