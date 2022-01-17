const feat = require("./features")
const fs = require("fs")
const path = require("path")
const {mean, min, max, variance} = require("mathjs");

const mainPaths = {
  "alex": {
    "midiFile": "/home/zongyu/Projects/listening_study/midi/151.mid",
    "midiDir": "/home/zongyu/Projects/aimgef/aimgef-assets/stimuli/",
    "MTDir": "/home/zongyu/Projects/MT-generated/",
    "CSSR": {
      // "executable": "/home/zongyu/Projects/decisional_states-1.0/examples/SymbolicSeries",
      "executable": "/home/zongyu/Projects/CSSR/CSSR",
      "alphabet": "/home/zongyu/Projects/CSSR/alphabet-mode-3",
      "data": "/home/zongyu/Projects/aimgef/features/data/",
      "pastSize": 1.855, // log_2(1025) / log_2(42)
      "futureSize": 1.855
    },
  }
}

const argv = require('minimist')(process.argv.slice(2))
const mainPath = mainPaths[argv.u];

function infoReader(filename) {
  const res = {}
  fs.readFileSync(filename).toString().split('\n').slice(7, 12).forEach((n) => {
    const item = n.split(': ')
    const key = item[0].replace(/ /g, '_')
    res[key] = Number(item[1])
  })
  return res
}

function reportMTGenerated() {
  const csvWriter = require('csv-writer').createObjectCsvWriter({
    path: 'MTGenerated.csv',
    header: [
      {id: 'group', title: 'group'},

      {id: 'statComp', title: 'statComp'},

      {id: 'transComp-Mean', title: 'transComp-Mean'},
      {id: 'transComp-Variance', title: 'transComp-Variance'},
      {id: 'transComp-Min', title: 'transComp-Min'},
      {id: 'transComp-Max', title: 'transComp-Max'},

      {id: 'arcScore-Mean', title: 'arcScore-Mean'},
      {id: 'arcScore-Variance', title: 'arcScore-Variance'},
      {id: 'arcScore-Min', title: 'arcScore-Min'},
      {id: 'arcScore-Max', title: 'arcScore-Max'},

      {id: 'tonalAmb-Mean', title: 'tonalAmb-Mean'},
      {id: 'tonalAmb-Variance', title: 'tonalAmb-Variance'},
      {id: 'tonalAmb-Min', title: 'tonalAmb-Min'},
      {id: 'tonalAmb-Max', title: 'tonalAmb-Max'},

      {id: 'attInterval-Mean', title: 'attInterval-Mean'},
      {id: 'attInterval-Variance', title: 'attInterval-Variance'},
      {id: 'attInterval-Min', title: 'attInterval-Min'},
      {id: 'attInterval-Max', title: 'attInterval-Max'},

      {id: 'rhyDis-Mean', title: 'rhyDis-Mean'},
      {id: 'rhyDis-Variance', title: 'rhyDis-Variance'},
      {id: 'rhyDis-Min', title: 'rhyDis-Min'},
      {id: 'rhyDis-Max', title: 'rhyDis-Max'},
    ]
  })
  const csvWriterRaw = require('csv-writer').createObjectCsvWriter({
    path: 'MTGeneratedRaw.csv',
    header: [
      {id: 'group', title: 'group'},
      {id: 'transComp', title: 'transComp'},
      {id: 'arcScore', title: 'arcScore'},
      {id: 'tonalAmb', title: 'tonalAmb'},
      {id: 'attInterval', title: 'attInterval'},
      {id: 'rhyDis', title: 'rhyDis'},

    ]
  })
  const dirs = fs.readdirSync(mainPath.MTDir)
  const data = []
  const raw = []
  for (const dir of dirs) {
    const midiFiles = fs.readdirSync(mainPath.MTDir + dir).filter(n => {
      return n.endsWith('.mid')
    })
    // const jsonFiles = fs.readdirSync(mainPath.MTDir + dir).filter(n => {return n.endsWith('.json')})
    const transComp = [], arcScore = [], tonalAmb = [], attInterval = [], rhyDis = []
    for (const midiFile of midiFiles) {
      const path = mainPath.MTDir + dir + '/' + midiFile
      const trans = feat.transComp(path)
      const arc = feat.arcScore(path)
      const tonal = feat.tonalAmb(path)
      const att = feat.attInterval(path)
      const rhy = feat.rhyDis(path).mean
      transComp.push(trans)
      arcScore.push(arc)
      tonalAmb.push(tonal)
      attInterval.push(att)
      rhyDis.push(rhy)
      raw.push({
        'group': dir,
        'transComp': trans,
        'arcScore': arc,
        'tonalAmb': tonal,
        'attInterval': att,
        'rhyDis': rhy,
      })
    }
    data.push({
      'group': dir,
      'statComp': infoReader(`./data/data_${dir}_info`)['Statistical_Complexity'],
      'transComp-Mean': mean(transComp),
      'transComp-Variance': variance(transComp),
      'transComp-Min': min(transComp),
      'transComp-Max': max(transComp),

      'arcScore-Mean': mean(arcScore),
      'arcScore-Variance': variance(arcScore),
      'arcScore-Min': min(arcScore),
      'arcScore-Max': max(arcScore),

      'tonalAmb-Mean': mean(tonalAmb),
      'tonalAmb-Variance': variance(tonalAmb),
      'tonalAmb-Min': min(tonalAmb),
      'tonalAmb-Max': max(tonalAmb),


      'attInterval-Mean': mean(attInterval),
      'attInterval-Variance': variance(attInterval),
      'attInterval-Min': min(attInterval),
      'attInterval-Max': max(attInterval),

      'rhyDis-Mean': mean(rhyDis),
      'rhyDis-Variance': variance(rhyDis),
      'rhyDis-Min': min(rhyDis),
      'rhyDis-Max': max(rhyDis),
    })
  }
  csvWriter
    .writeRecords(data)
    .then(() => console.log('The CSV file was written successfully.'));
  csvWriterRaw
    .writeRecords(raw)
    .then(() => console.log('The CSV file was written successfully.'));
}

function reportRatings() {
  const header = [
    "ID", "Rating", "Category", "Aspect", "Part",
    "transComp", "arcScore", "tonalAmb", "ioi_mean", "ioi_variance", "ioi2_mean", "ioi2_variance",
    "kot_mean", "kot_variance", "kdt_mean", "kdt_variance", "jitter_mean", "jitter_variance"
  ]
  const res = {}
  for (let i = 1; i <= 250; i++) {
    res[i] = {}
  }
  const isDirectory = fileName => {
    return fs.lstatSync(fileName).isDirectory()
  }
  const dirs = fs.readdirSync(mainPath.midiDir).map(fileName => {
    return path.join(mainPath.midiDir, fileName)
  }).filter(isDirectory)
  for (const dir of dirs) {
    const midiFiles = fs.readdirSync(dir).filter(n => {
      return n.endsWith('.mid')
    }).map(fileName => {return path.join(dir, fileName)})
    for (const file of midiFiles) {
      const id = file.split("/").slice(-1)[0].split(".")[0]
      res[id]["transComp"] = feat.transComp(file)
      res[id]["arcScore"] = feat.arcScore(file)
      res[id]["tonalAmb"] = feat.tonalAmb(file)
      res[id]["ioi_mean"] = feat.timeInterval(file)["firstIOI"]["mean"]
      res[id]["ioi_variance"] = feat.timeInterval(file)["firstIOI"]["variance"]
      res[id]["ioi2_mean"] = feat.timeInterval(file)["secondIOI"]["mean"]
      res[id]["ioi2_variance"] = feat.timeInterval(file)["secondIOI"]["variance"]
      res[id]["kot_mean"] = feat.timeInterval(file)["KOT"]["mean"]
      res[id]["kot_variance"] = feat.timeInterval(file)["KOT"]["variance"]
      res[id]["kdt_mean"] = feat.timeInterval(file)["KDT"]["mean"]
      res[id]["kdt_variance"] = feat.timeInterval(file)["KDT"]["variance"]
      res[id]["jitter_mean"] = feat.jitter(file)["mean"]
      res[id]["jitter_variance"] = feat.jitter(file)["variance"]
    }
  }
  const rows = fs.readFileSync('./ratings.csv').toString().split('\n').slice(1, -1).map(n => {
    return n.split(',')
  })
  for (let i = 0; i < rows.length; i++) {
    const id = rows[i][0]
    try {
      const featScores = []
      for (const f of header.slice(5)) {
        featScores.push(res[id][f])
      }
      rows[i] = rows[i].concat(featScores)
    } catch (e) {
      console.log("Reading midi file failed.")
    }
  }

  const writeStream = fs.createWriteStream('LSRatings.csv');
  writeStream.write(header.join(',') + '\n')
  for (const row of rows) {
    writeStream.write(row.join(',') + '\n')
  }
  // console.log("statComp: ", feat.statComp(mainPath.CSSR, mainPath.midiDir, true))
  // console.log("transComp: ", feat.transComp(mainPath.midiFile))
  // console.log("arcScore: ", feat.arcScore(mainPath.midiFile))
  // console.log("tonalAmb: ", feat.tonalAmb(mainPath.midiFile))
  // console.log("attInterval: ", feat.attInterval(mainPath.midiFile))
  // console.log("rhyDis: ", feat.rhyDis(mainPath.midiFile))
}

function getCategoriesStatComp() {
  const res = {}
  const isDirectory = fileName => {
    return fs.lstatSync(fileName).isDirectory()
  }
  const dirs = fs.readdirSync(mainPath.midiDir).map(fileName => {
    return path.join(mainPath.midiDir, fileName)
  }).filter(isDirectory)
  for (const dir of dirs) {
    const category = dir.split('/').slice(-1)[0]
    feat.statComp(mainPath.CSSR, dir, false, category)
    res[category] = infoReader(path.join(mainPath.CSSR.data, category + "_info"))["Statistical_Complexity"]
  }
  console.log(res)
}

// reportMTGenerated()
reportRatings()
// getCategoriesStatComp()
// const mm = require("maia-markov")
// a = new mm.MidiImport("/home/zongyu/Projects/aimgef/aimgef-assets/stimuli/CSQ-MaMa/26.mid")
// console.log()