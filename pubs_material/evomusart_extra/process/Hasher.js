const fs = require("fs")
const path = require("path")
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
    pts, cumuTime, fnam, mode = "duples", insert_mode = "increment and file",
    tMin = 0.1, tMax = 10, pMin = 1, pMax = 12, folder = __dirname
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
                cumuTime + v0[0], fnam,
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
                  this.insert(he, insert_mode, folder)
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
        if (vals[2] >= tMax / tMin || vals[2] < tMin / tMax) {
          console.log("Unexpected time difference:", vals[2])
          console.log("Returning.")
          return
        }
        // If ratio less than 1, invert and give it a negative sign so that such
        // values are as accurately represented as positive values.
        // console.log("vals[2] before inversion:", vals[2])
        let sign = "+"
        if (vals[2] < 1) {
          vals[2] = 1 / vals[2]
          sign = "-"
        }
        // console.log("vals[2] after inversion:", vals[2])
        str += sign
        // Round time difference ratio to 1 d.p. and append to str.
        const dp1 = Math.round(10 * vals[2]) / 10
        isInteger = Math.round(dp1) === dp1
        // console.log("isInteger:", isInteger)
        str += dp1
        if (isInteger) {
          str += ".0"
        }
        // console.log("str:", str)
        break

      default:
        console.log("Should not get to default in create_hash_entry() switch.")
    }
    return {
      "hash": str,
      "ctimes": [ctime],
      "fnams": [fnam]
    }
  }


  // Calculate a histogram for a transformation of the matching hash entries,
  // find the maximum count in this histogram, then work out and return the name
  // of the piece to which this maximum corresponds.
  histogram(countBins, ctimes, fnams, size) {
    const out = [];
    // "out" contains the index of bin,
    // and it is sorted based on the corresponding number of hash entries contained in "hist".
    for (let i = 0; i < countBins.length; i++) {
      out.push(i);
    }
    out.sort(function (a, b) {
      return countBins[b] - countBins[a];
    })

    return out.map((idx) => {
      for (let i = 0; i < ctimes.length; i++) {
        if (idx * size <= ctimes[i]) {
          return {"winPiece": fnams[i - 1], "edge": idx * size, "count": countBins[idx]}
        }
      }
    })
  }


  histogram2(hist, ctimes, fnams, size) {
    const out = []
    for (let i = 0; i < hist.length; i++) {
      out.push(i)
    }
    out.sort(function (a, b) {
      return hist[b] - hist[a]
    })
    // console.log("out:", out)
    return out.slice(0, 10)
      .map((idx) => {
        for (let i = 0; i < ctimes.length; i++) {
          if (idx * size <= ctimes[i]) {
            return {"winPiece": fnams[i - 1], "edge": idx * size, "count": hist[idx]}
          }
        }
      })
  }


  insert(hashEntry, method = "hash and lookup", dir) {
    const key = hashEntry.hash
    const lookup = this.contains(key)
    switch (method) {
      case "hash and lookup":
        if (lookup !== undefined) {
          // Extend ctimes and fnams arrays.
          lookup.ctimes.push(hashEntry.ctimes[0])
          lookup.fnams.push(hashEntry.fnams[0])
        } else {
          delete hashEntry.hash
          this.map[key] = hashEntry
        }
        break
      case "increment and file":
        if (lookup !== undefined) {
          this.map[key].increment++
        } else {
          this.map[key] = {
            "increment": 1,
            "log": fs.openSync(
              path.join(dir, key + ".json"), "a"
              // {"flags": "a"}
            )
          }
        }
        const content = JSON.stringify(Math.round(100 * hashEntry.ctimes[0]) / 100) + "," // 82.3MB
        // const content = JSON.stringify(Math.round(10 * hashEntry.ctimes[0]) / 10) + "," // 72.MB
        // const content = JSON.stringify(hashEntry.ctimes[0]) + "," // 162.9MB
        fs.writeSync(this.map[key].log, content)
        // this.map[key].log.write(content)

        // fs.writeFileSync(
        //   path.join(dir, key + ".json"),
        //   JSON.stringify(
        //     [
        //       Math.round(10*hashEntry.ctimes[0])/10,
        //       hashEntry.fnams[0]
        //     ]
        //   )
        //   + ",",
        //   { "flag": "a" }
        // )
        break
      default:
        console.log("Should not get to default in insert()!")
    }

  }


  // The expected format is with time in the first dimension and pitch in the
  // second dimension of pts. It is assumed that pts is sorted
  // lexicographically.
  match_hash_entries(
    pts, mode = "duples", tMin, tMax, pMin, pMax, ctimes, binSize, folder = __dirname
  ) {
    let uninh = new Set()
    const bins = Math.ceil(ctimes[ctimes.length - 1] / binSize);
    let countBins = new Array(bins).fill(0).map(() => {
      return new Set()
    })
    pts = pts.slice(0, 80)
    const npts = pts.length
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
                // There's a match!
                lookup.ctimes.forEach(function (ctime) {
                  tInDset.push(ctime)
                  tInQuery.push(he.ctimes[0])
                })
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
        loop1:
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
                    const he = this.create_hash_entry(
                      [v1[1] - v0[1], v2[1] - v1[1], td2 / td1], mode, v0[0]
                    )
                    if (fs.existsSync(path.join(folder, he.hash + ".json"))) {
                      const lookupStr = fs.readFileSync(
                        path.join(folder, he.hash + ".json"), "utf8"
                      ).slice(0, -1)
                      let lookup = JSON.parse("[" + lookupStr + "]")
                      lookup.forEach((value) => {
                        let dif = value - he.ctimes[0]
                        if (dif >= 0 && dif <= ctimes[ctimes.length - 1]) {
                          countBins[Math.floor(dif / binSize)].add(he.hash)
                        }
                      })
                    }
                    uninh.add(he.hash)
                    nh++
                    if (nh > 5000) {
                      break loop1
                    }
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

    return {
      "nosHashes": nh,
      "uninosHashes": uninh.size,
      "countBins": countBins.map((value => {
        return value.size
      }))
    }
  }
}

exports.default = Hasher
