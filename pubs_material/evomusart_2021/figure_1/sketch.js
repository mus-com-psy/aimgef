// Export png?
const exportPng = true
// let tokens = "NOTE_ON<63>, NOTE_ON<68>, TIME_SHIFT<1/16>, NOTE_OFF<68>, NOTE_ON<70>, TIME_SHIFT<1/16>, NOTE_OFF<70>, NOTE_ON<72>, TIME_SHIFT<1/16>, NOTE_OFF<72>, NOTE_ON<70>, TIME_SHIFT<1/16>, NOTE_OFF<63>, NOTE_OFF<70>, NOTE_ON<66>, NOTE_ON<69>, TIME_SHIFT<1/16>, NOTE_OFF<69>, NOTE_ON<71>, TIME_SHIFT<1/16>, NOTE_OFF<66>, NOTE_OFF<71>, NOTE_ON<64>, NOTE_ON<73>, TIME_SHIFT<1/16>, NOTE_OFF<64>, NOTE_OFF<73>, TIME_SHIFT<1/16>, NOTE_ON<62>, NOTE_ON<71>, TIME_SHIFT<1/16>, NOTE_OFF<62>, NOTE_OFF<71>"
let tokens = "NOTE_ON<66>,\nNOTE_ON<74>,\nTIME_SHIFT<1/16>,\nNOTE_OFF<74>,\nNOTE_ON<78>,\nTIME_SHIFT<1/16>,\nNOTE_OFF<78>,\nNOTE_ON<81>,\nTIME_SHIFT<1/16>,\nNOTE_OFF<81>,\n...\nNOTE_ON<64>,\nNOTE_ON<79>,\nTIME_SHIFT<1/16>,\nNOTE_OFF<64>,\nNOTE_OFF<79>"
let notes = "[\n" +
    "\t{ ontime: 0,\tMNN: 66,\tMPN: 63,\tduration: 1/4 },\n" +
    "\t{ ontime: 0,\tMNN: 74,\tMPN: 68,\tduration: 1/16 },\n" +
    "\t{ ontime: 1/4,\tMNN: 78,\tMPN: 70,\tduration: 1/16 },\n" +
    "\t{ ontime: 1/2,\tMNN: 81,\tMPN: 72,\tduration: 1/16 },\n" +
    "\t{ ontime: 3/4,\tMNN: 78,\tMPN: 70,\tduration: 1/16 },\n" +
    "\t{ ontime: 1,\tMNN: 71,\tMPN: 66,\tduration: 1/8 },\n" +
    "\t{ ontime: 1,\tMNN: 76,\tMPN: 69,\tduration: 1/16 },\n" +
    "\t{ ontime: 5/4,\tMNN: 79,\tMPN: 71,\tduration: 1/16 },\n" +
    "\t{ ontime: 3/2,\tMNN: 67,\tMPN: 64,\tduration: 1/16 },\n" +
    "\t{ ontime: 3/2,\tMNN: 83,\tMPN: 73,\tduration: 1/16 },\n" +
    "\t{ ontime: 7/4,\tMNN: 64,\tMPN: 62,\tduration: 1/16 },\n" +
    "\t{ ontime: 7/4,\tMNN: 79,\tMPN: 71,\tduration: 1/16 },\n" +
    "]"
let maia = "[\n" +
    "\t{ state: [1, [-8, 0]], context: {...} },\n" +
    "\t{ state: [1, [-8, 4]], context: {...} },\n" +
    "\t{ state: [1, [-8, 7]], context: {...} },\n" +
    "\t{ state: [1, [-8, 4]], context: {...} },\n" +
    "\t{ state: [1, [-3, 2]], context: {...} },\n" +
    "\t{ state: [1, [-8, 5]], context: {...} },\n" +
    "\t{ state: [1, [-7, 9]], context: {...} },\n" +
    "\t{ state: [1, [-10, 5]], context: {...} },\n" +
    "]"
// Point sets to plot.
let p2322 = {
  "pointSet": [
    [
      0,
      66,
      0.25
    ],
    [
      0,
      74,
      0.0625,
    ],
    [
      0.25,
      78,
      0.0625
    ],
    [
      0.5,
      81,
      0.0625
    ],
    [
      0.75,
      78,
      0.0625
    ],
    [
      1,
      71,
      0.125
    ],
    [
      1,
      76,
      0.0625
    ],
    [
      1.25,
      79,
      0.0625
    ],
    [
      1.5,
      67,
      0.0625
    ],
    [
      1.5,
      83,
      0.0625
    ],
    [
      1.75,
      79,
      0.0625
    ],
    [
      1.75,
      64,
      0.0625
    ],
  ],
  "pixelSet": null,
  "notationPng": "2322.png"
}
const transVec = [-144, 1]
// y = 760
let param = {
  "cvWidth": 3200,
  "cvHeight": 1600,
  "p2322": {
    "x": 1750, "y": 450, "width": 1200, "height": 350,
    "notationWidth": 1600,
    "card": p2322.pointSet.length,
    "minX": mu.min_argmin(p2322.pointSet.map(function(p){ return p[0] }))[0],
    "maxX": mu.max_argmax(p2322.pointSet.map(function(p){ return p[0] }))[0],
    "minY": mu.min_argmin(p2322.pointSet.map(function(p){ return p[1] }))[0],
    "maxY": mu.max_argmax(p2322.pointSet.map(function(p){ return p[1] }))[0],
   },
  
}

function preload(){
  p2322.notationImg = loadImage(p2322.notationPng)
}

function setup(){
  let c = createCanvas(param.cvWidth, param.cvHeight)
  background(255)

  // Add the notation PNG files.
  let aspectRatio = p2322.notationImg.height/p2322.notationImg.width
  image(
    p2322.notationImg, 20, 20,
    param.p2322.notationWidth, aspectRatio*param.p2322.notationWidth
  )
  push()
  textStyle(BOLD)
  textSize(80)
  text("a", 20, 60)
  text("b", 1580, 60)
  text("c", 20, 700)
  text("d", 1400, 700)
  text("e", 2000, 700)
  pop()
  let pianoRollHeight = aspectRatio*param.p2322.width - 50
  let pianoRollWidth = param.p2322.notationWidth + 150
  // Plot p2322.
  param.p2322.ppb = param.p2322.width/(param.p2322.maxX - param.p2322.minX)*4
  param.p2322.ppp = param.p2322.height/(param.p2322.maxY - param.p2322.minY)
  p2322.pixelSet = p2322.pointSet.map(function(p){
    const propX = (p[0] - param.p2322.minX)/(param.p2322.maxX - param.p2322.minX)
    const propY = (p[1] - param.p2322.minY)/(param.p2322.maxY - param.p2322.minY)
    const pixlX = param.p2322.x + propX*param.p2322.width
    const pixlY = param.p2322.y - propY*param.p2322.height
    fill(200, 200, 200)
    rect(
      pixlX, pixlY, param.p2322.ppb*p[2], param.p2322.ppp
    )
    fill(0, 0, 0)
    return [pixlX, pixlY]
  })
  stroke(0, 50)
  textStyle(NORMAL)
  textSize(50)
  for (let i = 0; i < 21; i++) {
    line(param.p2322.x - 30, param.p2322.y + param.p2322.ppp - param.p2322.ppp * i, param.p2322.x + param.p2322.width + param.p2322.ppb * 0.0625, param.p2322.y + param.p2322.ppp - param.p2322.ppp * i);
  }
  for (let i = 0; i < 21; i++) {
    if (i % 5 == 1) {
      text(`${64 + i}`, param.p2322.x - 60, param.p2322.y + param.p2322.ppp - param.p2322.ppp * i);
    }
  }
  for (let i = 0; i < 9; i++) {
    line(param.p2322.x + param.p2322.ppb * 0.0625 * i, param.p2322.y + param.p2322.ppp, param.p2322.x + param.p2322.ppb * 0.0625 * i, param.p2322.y - param.p2322.height);
  }
  for (let i = 0; i < 4; i++) {
    text(`${468}.${i%4 + 1}`, param.p2322.x + param.p2322.ppb * 0.0625 * i - 50, param.p2322.y + param.p2322.ppp + 40);
  }
  for (let i = 4; i < 8; i++) {
    text(`${469}.${i%4 + 1}`, param.p2322.x + param.p2322.ppb * 0.0625 * i - 50, param.p2322.y + param.p2322.ppp + 40);
  }
  
  tokens = tokens.split(', ');
  textSize(50)
  text(notes, 100, 750)
  textSize(42)
  text(tokens, 1500, 750)
  textSize(50)
  text(maia, 2100, 750)

  // for (let i = 0; i < tokens.length; i++) {
  //   if (i < 9) {
  //     text(tokens[i], 100 + i * 300, 700)
  //   } else if (i < 18) {
  //     text(tokens[i], 100 + (i - 9) * 300, 850)
  //   } else if (i < 27) {
  //     text(tokens[i], 100 + (i - 18) * 300, 1000)
  //   } else if (i < 34) {
  //     text(tokens[i], 100 + (i - 27) * 300, 1150)
  //   }
  // }
  textSize(50)
  text("Ontime (crotchet beat)", 2200, 580)
  translate(1780, 300)
  rotate(PI * 1.5)
  text("Pitch (MNN)", -100, -120)
  
  if (exportPng){
    saveCanvas(c, 'myCanvas', 'png')
  }
}