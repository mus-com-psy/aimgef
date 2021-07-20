const mu = require("maia-util")

class Hasher {
    constructor(_mapPath) {
        if (_mapPath !== undefined) {
            this.map = require(_mapPath)
        } else {
            this.map = {}
        }
    }


    contains(aKey) {
        return this.map[aKey]
    }


    // The expected format is with time in the first dimension and pitch in the
    // second dimension of pts. It is assumed that pts is sorted
    // lexicographically.
    create_hash_entries(
        pts, fnam, mode = "duples",
        tMin = 0.1, tMax = 10, pMin = 1, pMax = 12
    ) {
        const npts = pts.length
        // console.log("npts:", npts)
        let nh = 0

        switch (mode) {
            case "duples":
                for (let i = 0; i < npts - 1; i++) {
                    const v0 = pts[i]
                    let j = i + 1
                    while (j < npts) {
                        const v1 = pts[j]
                        const td = v1[0] - v0[0]
                        const apd = Math.abs(v1[1] - v0[1])
                        // console.log("i:", i, "j:", j)
                        // Decide whether to make a hash entry.
                        if (td > tMin && td < tMax && apd >= pMin && apd <= pMax) {
                            // Make a hash entry, something like "±pdtd"
                            const he = this.create_hash_entry(
                                [v1[1] - v0[1], td], mode,
                                v0[0], fnam,
                                tMin, tMax
                            )
                            this.insert(he)
                            nh++
                        } // End whether to make a hash entry.
                        if (td >= tMax) {
                            j = npts - 1
                        }
                        j++
                    } // End while.
                } // for (let i = 0;
                break

            case "triples":
                for (let i = 0; i < npts - 2; i++) {
                    const v0 = pts[i]
                    let j = i + 1
                    while (j < npts - 1) {
                        const v1 = pts[j]
                        const td1 = v1[0] - v0[0]
                        const apd1 = Math.abs(v1[1] - v0[1])
                        // console.log("i:", i, "j:", j)
                        // Decide whether to proceed to v1 and v2.
                        if (td1 > tMin && td1 < tMax && apd1 >= pMin && apd1 <= pMax) {
                            let k = j + 1
                            while (k < npts) {
                                const v2 = pts[k]
                                const td2 = v2[0] - v1[0]
                                const apd2 = Math.abs(v2[1] - v1[1])
                                // console.log("j:", j, "k:", k)
                                // Decide whether to make a hash entry.
                                if (td2 > tMin && td2 < tMax && apd2 >= pMin && apd2 <= pMax) {
                                    // Make a hash entry, something like "±pd1±pd2tdr"
                                    const he = this.create_hash_entry(
                                        [v1[1] - v0[1], v2[1] - v1[1], td2 / td1], mode,
                                        cumuTime + v0[0], fnam,
                                        tMin, tMax
                                    )
                                    this.insert(he)
                                    nh++
                                } // End whether to make a hash entry.
                                if (td2 >= tMax) {
                                    k = npts - 1
                                }
                                k++
                            } // End k while.
                        }
                        if (td1 >= tMax) {
                            j = npts - 2
                        }
                        j++
                    } // End j while.
                } // for (let i = 0;
                break
            default:
                console.log("Should not get to default in create_hash_entries() switch.")
        }

        return nh
    }


    create_hash_entry(vals, mode, ctime, fnam, tMin, tMax) {
        let str = "", isInteger
        switch (mode) {
            case "duples":
                // Sense-check pitch difference.
                const apd = Math.abs(vals[0])
                if (apd >= 100 || Math.round(vals[0]) !== vals[0]) {
                    console.log("Unexpected pitch difference:", vals[0])
                    console.log("Returning.")
                    return
                }
                if (vals[0] >= 0) {
                    str += "+"
                } else {
                    str += "-"
                }
                if (apd < 10) {
                    str += "0"
                }
                str += apd
                // Sense-check time difference.
                isInteger = Math.round(vals[1]) === vals[1]
                if (vals[1] >= tMax || vals[1] < tMin) {
                    console.log("Unexpected time difference:", vals[1])
                    console.log("Returning.")
                    return
                }
                // Round time difference to 1 d.p. and append to str.
                str += Math.round(10 * vals[1]) / 10
                if (isInteger) {
                    str += ".0"
                }
                break

            case "triples":
                // Sense-check pitch difference.
                vals.slice(0, 2).forEach(function (v, idx) {
                    const apd = Math.abs(v)
                    if (apd >= 100 || Math.round(v) !== v) {
                        console.log("Unexpected pitch difference:", v, idx)
                        console.log("Returning.")
                        return
                    }
                    if (v >= 0) {
                        str += "+"
                    } else {
                        str += "-"
                    }
                    if (apd < 10) {
                        str += "0"
                    }
                    str += apd
                })
                // Sense-check time difference ratio.
                isInteger = Math.round(vals[2]) === vals[2]
                if (vals[2] >= tMax || vals[2] < tMin) {
                    console.log("Unexpected time difference:", vals[2])
                    console.log("Returning.")
                    return
                }
                // Round time difference ratio to 1 d.p. and append to str.
                str += Math.round(10 * vals[2]) / 10
                if (isInteger) {
                    str += ".0"
                }
                break

            default:
                console.log("Should not get to default in create_hash_entry() switch.")
        }
        return {
            "hash": str,
            "ctimes": ctime,
            "fnams": fnam
        }
    }


    // Calculate a histogram for a transformation of the matching hash entries,
    // find the maximum count in this histogram, then work out and return the name
    // of the piece to which this maximum corresponds.
    histogram(parallelTimes, ctimes, fnams, nosBins = 100, mode = "duples") {
        if (parallelTimes.timesInDataset.length !== parallelTimes.timesInQuery.length) {
            console.log(
                "Times in dataset and times in query are not of the same length. " +
                "Returning."
            )
            return
        }

        // Apply transformation.
        let transf
        switch (mode) {
            case "duples":
                transf = parallelTimes.timesInDataset.map(function (td, idx) {
                    return td - parallelTimes.timesInQuery[idx]
                })
                break
            case "triples":

                //////////////////////
                // NEEDS FINISHING! //
                //////////////////////

                break
            default:
                console.log("Should not get to default in histogram() switch.")
        }
        // console.log("transf:", transf)
        // Get min/max data for histogram edges.
        const min = Math.min(...transf)
        const max = Math.max(...transf)
        const binWidth = (max - min) / nosBins
        let edges = new Array(nosBins + 1)
        let count = new Array(nosBins)
        for (let i = 0; i < nosBins; i++) {
            edges[i] = min + binWidth * i
            count[i] = 0
        }
        edges[nosBins] = max
        // console.log("edges:", edges)
        // Go over transformed data and update counts accordingly.
        transf.forEach(function (t) {
            let i = 0
            while (i < nosBins) {
                if (t >= edges[i] && t < edges[i + 1]) {
                    count[i]++
                    i = nosBins - 1
                }
                i++
            }
        })
        // console.log("count:", count)

        const edgeCount = count.map(function (c, idx) {
            return {"edge": edges[idx], "count": c}
        })
        edgeCount.sort(function (a, b) {
            return b.count - a.count
        })
        // console.log("edgeCount.slice(0, 10):", edgeCount.slice(0, 10))
        // Work out which pieces these correspond to.
        const results = edgeCount.slice(0, 10).map(function (ec) {
            const winningSample = ec.edge + binWidth / 2
            const ma = mu.min_argmin(ctimes.map(function (ct) {
                return Math.abs(ct - winningSample)
            }))
            ec.winningPiece = fnams[ma[1]]
            ec.winningPieceIdx = ma[1]
            // let winningPieceIdx = 0
            // let i = ctimes.length - 1
            // while (i >= 0){
            //   if (winningSample >= ctimes[i]){
            //     winningPieceIdx = i
            //     i = 0
            //   }
            //   i--
            // }
            // ec.winningPiece = fnams[winningPieceIdx]
            // ec.winningPieceIdx = winningPieceIdx
            return ec
        })

        return results

        // Find bin with maximum count.
        // const ma = mu.max_argmax(count)
        // console.log("ma:", ma)
        // // Work out which piece this corresponds to.
        // const winningSample = edges[ma[1]] + binWidth/2
        // let winningPieceIdx = 0
        // let i = ctimes.length - 1
        // while (i >= 0){
        //   if (winningSample >= ctimes[i]){
        //     winningPieceIdx = i
        //     i = 0
        //   }
        //   i--
        // }
        // return fnams[winningPieceIdx]

    }


    insert(hashEntry) {
        const hash = hashEntry.hash
        const ctimes = hashEntry.ctimes
        const fnams = hashEntry.fnams

        const lookup = this.contains(hash)
        if (lookup !== undefined) {
            if (lookup[fnams] !== undefined) {
                lookup[fnams].push(ctimes)
            } else {
                lookup[fnams] = [ctimes]
            }

        } else {
            this.map[hash] = {}
            this.map[hash][fnams] = [ctimes]
        }
    }


    // The expected format is with time in the first dimension and pitch in the
    // second dimension of pts. It is assumed that pts is sorted
    // lexicographically.
    match_hash_entries(
        pts, mode = "duples", tMin = 0.1, tMax = 10, pMin = 1, pMax = 12
    ) {
        let tInDset = []
        let tInQuery = []
        let results = {}
        const npts = pts.length
        // console.log("npts:", npts)
        let nh = 0

        switch (mode) {
            case "duples":
                for (let i = 0; i < npts - 1; i++) {
                    const v0 = pts[i]
                    let j = i + 1
                    while (j < npts) {
                        const v1 = pts[j]
                        const td = v1[0] - v0[0]
                        const apd = Math.abs(v1[1] - v0[1])
                        // console.log("i:", i, "j:", j)
                        // Decide whether to make a hash entry.
                        if (td > tMin && td < tMax && apd >= pMin && apd <= pMax) {
                            // Make a hash entry, something like "±pdtd"
                            const he = this.create_hash_entry(
                                [v1[1] - v0[1], td], mode, v0[0]
                            )
                            // console.log("he:", he)
                            // Is there a match?
                            const lookup = this.contains(he.hash)
                            if (lookup !== undefined) {
                                for (const [key, value] of Object.entries(lookup)) {
                                    if (results[key] === undefined) {
                                        results[key] = []
                                    }
                                    value.forEach(function (ctime) {
                                        results[key].push([he.ctimes, ctime])
                                    })
                                }
                            }
                            nh++
                        } // End whether to make a hash entry.
                        if (td >= tMax) {
                            j = npts - 1
                        }
                        j++
                    } // End while.
                } // for (let i = 0;
                break

            case "triples":
                for (let i = 0; i < npts - 2; i++) {
                    const v0 = pts[i]
                    let j = i + 1
                    while (j < npts - 1) {
                        const v1 = pts[j]
                        const td1 = v1[0] - v0[0]
                        const apd1 = Math.abs(v1[1] - v0[1])
                        // console.log("i:", i, "j:", j)
                        // Decide whether to proceed to v1 and v2.
                        if (td1 > tMin && td1 < tMax && apd1 >= pMin && apd1 <= pMax) {
                            let k = j + 1
                            while (k < npts) {
                                const v2 = pts[k]
                                const td2 = v2[0] - v1[0]
                                const apd2 = Math.abs(v2[1] - v1[1])
                                // console.log("j:", j, "k:", k)
                                // Decide whether to make a hash entry.
                                if (td2 > tMin && td2 < tMax && apd2 >= pMin && apd2 <= pMax) {
                                    // Make a hash entry, something like "±pd1±pd2tdr"
                                    const he = this.create_hash_entry(
                                        [v1[1] - v0[1], v2[1] - v1[1], td2 / td1], mode, v0[0]
                                    )
                                    // Is there a match?
                                    const lookup = this.map.contains(he.hash)
                                    if (lookup !== undefined) {
                                        // There's a match!
                                        lookup.ctimes.forEach(function (ctime) {
                                            tInDset.push(ctime)
                                            tInQuery.push(he.ctimes[0])
                                        })
                                    }
                                    nh++
                                } // End whether to make a hash entry.
                                if (td2 >= tMax) {
                                    k = npts - 1
                                }
                                k++
                            } // End k while.
                        }
                        if (td1 >= tMax) {
                            j = npts - 2
                        }
                        j++
                    } // End j while.
                } // for (let i = 0;
                break
            default:
                console.log("Should not get to default in match_hash_entries() switch.")
        }
        // return {
        //     "nosHashes": nh,
        //     "timesInDataset": tInDset,
        //     "timesInQuery": tInQuery
        // }
        return {
            "nosHashes": nh,
            "results": results,
        }
    }


}

module.exports = Hasher