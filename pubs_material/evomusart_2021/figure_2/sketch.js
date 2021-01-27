// Export png?
const exportPng = true
// Point sets to plot.
let p1999 = {
    "pointSet": [
        [
            612,
            53
        ],
        [
            612,
            67
        ],
        [
            612,
            69
        ],
        [
            612.25,
            55
        ],
        [
            612.5,
            57
        ],
        [
            612.75,
            59
        ],
        [
            613,
            60
        ],
        [
            613.25,
            62
        ],
        [
            613.5,
            64
        ],
        [
            613.75,
            62
        ],
        [
            614,
            50
        ],
        [
            614,
            63
        ],
        [
            614,
            68
        ],
        [
            614,
            73
        ],
        [
            614.25,
            52
        ],
        [
            614.5,
            54
        ],
        [
            614.75,
            56
        ],
        [
            615,
            57
        ],
        [
            615.25,
            59
        ],
        [
            615.5,
            61
        ],
        [
            615.75,
            63
        ],
        [
            616,
            60
        ],
        [
            616,
            62
        ],
        [
            616,
            67
        ],
        [
            616,
            69
        ],
        [
            616,
            74
        ],
        [
            617,
            46
        ],
        [
            617,
            53
        ],
        [
            617,
            57
        ],
        [
            617,
            62
        ],
        [
            617,
            64
        ],
        [
            617,
            67
        ],
        [
            617,
            69
        ],
        [
            618,
            46
        ],
        [
            618,
            53
        ],
        [
            618,
            57
        ],
        [
            618,
            62
        ],
        [
            618,
            64
        ],
        [
            618,
            67
        ],
        [
            618,
            69
        ]
    ],
    "pixelSet": null,
    "notationPng": "1999_excerpt.png"
}
let p2322 = {
    "pointSet": [
        [
            468,
            54
        ],
        [
            468,
            61
        ],
        [
            468,
            63
        ],
        [
            468,
            68
        ],
        [
            468.25,
            70
        ],
        [
            468.5,
            54
        ],
        [
            468.5,
            72
        ],
        [
            468.75,
            70
        ],
        [
            469,
            57
        ],
        [
            469,
            61
        ],
        [
            469,
            66
        ],
        [
            469,
            69
        ],
        [
            469.25,
            71
        ],
        [
            469.5,
            57
        ],
        [
            469.5,
            59
        ],
        [
            469.5,
            64
        ],
        [
            469.5,
            73
        ],
        [
            469.75,
            62
        ],
        [
            469.75,
            71
        ],
        [
            470,
            58
        ],
        [
            470,
            61
        ],
        [
            470,
            70
        ],
        [
            470.25,
            72
        ],
        [
            470.5,
            58
        ],
        [
            470.5,
            70
        ],
        [
            470.75,
            68
        ],
        [
            471,
            51
        ],
        [
            471,
            62
        ],
        [
            471,
            64
        ],
        [
            471,
            69
        ],
        [
            471.25,
            71
        ],
        [
            471.5,
            51
        ],
        [
            471.5,
            69
        ],
        [
            471.75,
            67
        ],
        [
            472,
            54
        ],
        [
            472,
            61
        ],
        [
            472,
            63
        ],
        [
            472,
            68
        ],
        [
            473,
            54
        ],
        [
            473,
            58
        ],
        [
            473,
            61
        ],
        [
            473,
            63
        ],
        [
            473,
            65
        ],
        [
            473,
            68
        ],
        [
            473,
            70
        ],
        [
            474,
            54
        ],
        [
            474,
            58
        ],
        [
            474,
            61
        ],
        [
            474,
            63
        ],
        [
            474,
            65
        ],
        [
            474,
            68
        ],
        [
            474,
            70
        ]
    ],
    "pixelSet": null,
    "notationPng": "2322_excerpt.png"
}
const transVec = [-144, 1]
// y = 760
let param = {
    "cvWidth": 3200,
    "cvHeight": 2000,
    "p1999": {
        "x": 1860, "y": 1800, "width": 1000, "height": 1000,
        "notationWidth": 1400,
        "card": p1999.pointSet.length,
        "minX": mu.min_argmin(p1999.pointSet.map(function (p) {
            return p[0]
        }))[0],
        "maxX": mu.max_argmax(p1999.pointSet.map(function (p) {
            return p[0]
        }))[0],
        "minY": mu.min_argmin(p1999.pointSet.map(function (p) {
            return p[1]
        }).concat(p2322.pointSet.map(function (p) {
            return p[1]
        })))[0],
        "maxY": mu.max_argmax(p1999.pointSet.map(function (p) {
            return p[1]
        }).concat(p2322.pointSet.map(function (p) {
            return p[1]
        })))[0],

    },
    "p2322": {
        "x": 240, "y": 1800, "width": 1000, "height": 1000,
        "notationWidth": 1400,
        "card": p2322.pointSet.length,
        "minX": mu.min_argmin(p2322.pointSet.map(function (p) {
            return p[0]
        }))[0],
        "maxX": mu.max_argmax(p2322.pointSet.map(function (p) {
            return p[0]
        }))[0],
        "minY": mu.min_argmin(p2322.pointSet.map(function (p) {
            return p[1]
        }).concat(p1999.pointSet.map(function (p) {
            return p[1]
        })))[0],
        "maxY": mu.max_argmax(p2322.pointSet.map(function (p) {
            return p[1]
        }).concat(p1999.pointSet.map(function (p) {
            return p[1]
        })))[0],

    },

}
// We want the plots to fit in whatever the width and height are, but we also want
// them to be at the same scale.
if (param.p1999.maxX - param.p1999.minX >= param.p2322.maxX - param.p2322.minX) {
    param.p1999.rngX = param.p1999.maxX - param.p1999.minX;
    param.p2322.rngX = param.p1999.maxX - param.p1999.minX;
} else {
    param.p1999.rngX = param.p2322.maxX - param.p2322.minX
    param.p2322.rngX = param.p2322.maxX - param.p2322.minX
}
param.p1999.rngY = param.p1999.maxY - param.p1999.minY
param.p2322.rngY = param.p2322.maxY - param.p2322.minY
param.p1999.sfX = 1
param.p1999.sfY = 1
param.p2322.sfX = 1
param.p2322.sfY = 1
if (param.p1999.rngX >= param.p2322.rngX) {
    param.p2322.sfX = param.p2322.rngX / param.p1999.rngX
} else {
    param.p1999.sfX = param.p1999.rngX / param.p2322.rngX
}
if (param.p1999.rngY >= param.p2322.rngY) {
    param.p2322.sfY = param.p2322.rngY / param.p1999.rngY
} else {
    param.p1999.sfY = param.p1999.rngY / param.p2322.rngY
}


param.p1999.ppp = 1 / param.p1999.rngY * param.p1999.height * param.p1999.sfY
param.p2322.ppp = 1 / param.p2322.rngY * param.p2322.height * param.p2322.sfY
param.p1999.ppb = 0.25 / param.p1999.rngX * param.p1999.width * param.p1999.sfX
param.p2322.ppb = 0.25 / param.p2322.rngX * param.p2322.width * param.p2322.sfX

function preload() {
    p1999.notationImg = loadImage(p1999.notationPng)
    p2322.notationImg = loadImage(p2322.notationPng)
}

function setup() {
    let c = createCanvas(param.cvWidth, param.cvHeight)
    // background(255)

    // Add the notation PNG files.
    let aspectRatio = p1999.notationImg.height / p1999.notationImg.width
    image(
        p1999.notationImg, 1630, 80,
        param.p1999.notationWidth, aspectRatio * param.p1999.notationWidth
    )
    aspectRatio = p2322.notationImg.height / p2322.notationImg.width
    image(
        p2322.notationImg, 30, 80,
        param.p2322.notationWidth, aspectRatio * param.p2322.notationWidth
    )

    // Plot p1999.
    p1999.pixelSet = p1999.pointSet.map(function (p) {
        const propX = (p[0] - param.p1999.minX) / param.p1999.rngX
        const propY = (p[1] - param.p1999.minY) / param.p1999.rngY
        const pixlX = param.p1999.x + propX * param.p1999.width * param.p1999.sfX
        const pixlY = param.p1999.y - propY * param.p1999.height * param.p1999.sfY
        // ellipse(
        //   pixlX, pixlY, 10, 10
        // )
        return [pixlX, pixlY]
    })
    // rect(240, 760, 1000, 1000)


    // Plot p2322.
    p2322.pixelSet = p2322.pointSet.map(function (p) {
        const propX = (p[0] - param.p2322.minX) / (param.p2322.maxX - param.p2322.minX)
        const propY = (p[1] - param.p2322.minY) / (param.p2322.maxY - param.p2322.minY)
        const pixlX = param.p2322.x + propX * param.p2322.width * param.p2322.sfX
        const pixlY = param.p2322.y - propY * param.p2322.height * param.p2322.sfY
        // ellipse(
        //   pixlX, pixlY, 10, 10
        // )
        return [pixlX, pixlY]
    })
    push()
    strokeWeight(4);
    // Draw axis
    line(param.p1999.x - param.p1999.ppb, param.p1999.y + param.p1999.ppp, param.p1999.x - param.p1999.ppb, param.p1999.y - param.p1999.height * param.p1999.sfY - param.p1999.ppp);
    line(param.p1999.x - param.p1999.ppb, param.p1999.y + param.p1999.ppp, param.p1999.x + param.p1999.width * param.p1999.sfX + param.p1999.ppb, param.p1999.y + param.p1999.ppp);
    line(param.p2322.x - param.p2322.ppb, param.p2322.y + param.p2322.ppp, param.p2322.x - param.p2322.ppb, param.p2322.y - param.p2322.height * param.p2322.sfY - param.p2322.ppp);
    line(param.p2322.x - param.p2322.ppb, param.p2322.y + param.p2322.ppp, param.p2322.x + param.p2322.width * param.p2322.sfX + param.p2322.ppb, param.p2322.y + param.p2322.ppp);
    pop()
    // Draw scale
    textSize(50)
    for (let i = 0; i < 6; i++) {
        push()
        strokeWeight(4);
        line(param.p1999.x - param.p1999.ppb, param.p1999.y + param.p1999.ppp - param.p1999.ppp * 5 * (i), param.p1999.x - param.p1999.ppb - 10, param.p1999.y + param.p1999.ppp - param.p1999.ppp * 5 * (i));
        line(param.p2322.x - param.p2322.ppb, param.p2322.y + param.p2322.ppp - param.p2322.ppp * 5 * (i), param.p2322.x - param.p2322.ppb - 10, param.p2322.y + param.p2322.ppp - param.p2322.ppp * 5 * (i));
        pop()
        text(`${45 + 5 * i}`, param.p1999.x - param.p1999.ppb - 80, param.p1999.y + param.p1999.ppp - param.p1999.ppp * 5 * (i) + 15);
        text(`${45 + 5 * i}`, param.p2322.x - param.p2322.ppb - 80, param.p2322.y + param.p2322.ppp - param.p2322.ppp * 5 * (i) + 15);
    }
    for (let i = 0; i < 7; i++) {
        push()
        strokeWeight(4);
        line(param.p1999.x + param.p1999.ppb * 4 * (i), param.p1999.y + param.p1999.ppp, param.p1999.x + param.p1999.ppb * 4 * (i), param.p1999.y + param.p1999.ppp + 10);
        line(param.p2322.x + param.p2322.ppb * 4 * (i), param.p2322.y + param.p2322.ppp, param.p2322.x + param.p2322.ppb * 4 * (i), param.p2322.y + param.p2322.ppp + 10);
        pop()
        text(`${612 + i}`, param.p1999.x + param.p1999.ppb * 4 * (i) - 40, param.p1999.y + param.p1999.ppp + 55);
        text(`${468 + i}`, param.p2322.x + param.p2322.ppb * 4 * (i) - 40, param.p2322.y + param.p2322.ppp + 55)
    }
    textSize(50)
    push()
    translate(param.p1999.x - param.p1999.ppb - 75, param.p1999.y - 350)
    rotate(3 * Math.PI / 2)
    text("Pitch (MPN)", 0, -20)
    pop()
    push()
    translate(param.p2322.x - param.p2322.ppb - 75, param.p2322.y - 350)
    rotate(3 * Math.PI / 2)
    text("Pitch (MPN)", 0, -20)
    pop()

    text("Ontime (Crotchet Beats)", param.p1999.x + param.p1999.width * param.p1999.sfX / 2 - 200, param.p1999.y + param.p1999.ppb + 100);
    text("Ontime (Crotchet Beats)", param.p2322.x + param.p2322.width * param.p2322.sfX / 2 - 200, param.p2322.y + param.p2322.ppb + 100);
    push()
    textStyle(BOLD)
    textSize(80)
    text("a", 20, 80)
    text("b", 1620, 80)
    text("c", 20, 800)
    text("d", 1620, 800)
    pop()

    // Draw arrows in one colour if there is a match, and in a different colour if there is not.
    p2322.idxIn1999 = p2322.pointSet.map(function (p) {
        const pTrans = mu.subtract_two_arrays(p, transVec)
        // console.log("pTrans:", pTrans)
        const relIdx = p1999.pointSet.findIndex(function (q) {
            return pTrans.equals(q)
        })
        return relIdx
    })

    p2322.idxIn1999.forEach(function (idx, jdx) {
        if (idx >= 0) {
            stroke(1)
            strokeWeight(2)
            line(
                p1999.pixelSet[idx][0], p1999.pixelSet[idx][1],
                p2322.pixelSet[jdx][0], p2322.pixelSet[jdx][1]
            )
        } else {
            stroke(0, 5)
        }

    })

    // Plot p1999.
    fill(200)
    p1999.pixelSet.forEach(function (p) {
        if (p[0] <= 2485) {
            ellipse(p[0], p[1], 20, 20)
        } else {
            triangle(p[0] - 10, p[1] + 10, p[0], p[1] - 10, p[0] + 10, p[1] + 10)
        }

    })
    // rect(240, 760, 1000, 1000)


    // Plot p2322.
    p2322.pixelSet.forEach(function (p) {
        if (p[0] <= 865) {
            ellipse(p[0], p[1], 20, 20)
        } else {
            triangle(p[0] - 10, p[1] + 10, p[0], p[1] - 10, p[0] + 10, p[1] + 10)
        }
    })


    // console.log("p2322:", p2322)

    if (exportPng) {
        saveCanvas(c, 'myCanvas', 'png')
    }
    console.log(1 / param.p1999.rngY * param.p1999.height * param.p1999.sfY)
    console.log("param:", param)
    console.log(param.p1999.minY, param.p1999.maxY)

}


// function Arrow(_x1, _y1, _x2, _y2, _headPos){
//   this.x1 = _x1
//   this.y1 = _y1
//   this.x2 = _x2
//   this.y2 = _y2
//   this.headPos = _headPos
//   this.headWidth = 5
//   this.headHeight = 7
//   this.xdir = Math.sign(this.x2 - this.x1)
//   this.ydir = Math.sign(this.y2 - this.y1)

//   this.draw = function(){
//     fill(0)
//     line(this.x1, this.y1, this.x2, this.y2)
//     if (this.headPos == "bgn" || this.headPos == "both"){
//       triangle(
//         this.x1, this.y1,
//         this.x1 - this.headWidth, this.y1 + this.ydir*this.headHeight,
//         this.x1 + this.headWidth, this.y1 + this.ydir*this.headHeight
//       )
//     }
//     if (this.headPos == "end" || this.headPos == "both"){
//       triangle(
//         this.x2, this.y2,
//         this.x2 - this.headWidth, this.y2 - this.ydir*this.headHeight,
//         this.x2 + this.headWidth, this.y2 - this.ydir*this.headHeight
//       )
//     }
//   }
// }


