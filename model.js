// How to read allNeurons: allNeurons[layer][neuronOfLayer]
//.location = [layer,neuronOfLayer] .value = (activation of neuron) .bias = (bias of neuron) .grad = (gradient of neuron)
//
// How to read allWeights: allWeights[layer][neuronFrom][neuronTo]
//.location = [layer,neuronFrom,neuronTo] .value = (value of weight) .grad = (gradient of weight)
const getPixels = require('get-pixels')
const fs = require('fs')
const util = require('util')
const readdir = util.promisify(fs.readdir)
const {GPU} = require('gpu.js');
const gpu = new GPU({mode: 'cpu'});
let trainData = []
let validData = []

function filesPromise(path) {
  return new Promise((resolve, reject) => {
    fs.readFile(path, 'utf8', (err, fileData) => err ? reject(err) : resolve(fileData))
  })
}

async function getModel(fileName, dir) {
  const result = await filesPromise(`${dir}/${fileName}.txt`)
  return JSON.parse(result)
}

function save(fileName, dir, content) {
  fs.writeFile(`${dir}/${fileName}.txt`, JSON.stringify(content), (err => {
  }))
}

function getPixelsAsync(url) {
  return new Promise((resolve, reject) => {
    getPixels(url, (err, pixels) => err ? reject(err) : resolve(pixels))
  })
}

function sig(num) {
  return 1 / (1 + Math.exp(-1 * num))
}

const dir = '/mnt/c/Users/Jacob Kirmayer/WebstormProjects/fat-ai-network'


function sigSlope(num) {
  const sigNum = sig(num)
  return sigNum * (1 - sigNum)
}

async function getFiles() {
  for (let type of [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) {
    const fileList = await readdir(`./trainingSet/trainingSet/${type}`);
    let fileNum = 1
    const len = 100
    for (let file of fileList.slice(0,100)) {
      try {
        const pixels = await getPixelsAsync(`./trainingSet/trainingSet/${type}/${file}`)
        const processedPixels = pixels.data.filter((elem, index) => index % 4 === 1)
        const finalPixels = Array.from(processedPixels).map(elem => 2 * (elem / 255) - 1)
        let answers = new Float32Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        answers[type] = 1
        if (fileNum > 0.1 * len) {
          trainData.push({input: finalPixels, answers: answers})
        } else {
          validData.push({input: finalPixels, answers: answers})
        }
        fileNum++
      } catch (e) {
        console.log(e)
      }
    }
    console.log(type)
  }
  console.log('data loaded')
  learn([784, 20, 20, 10], 10, 64, 0.0001, (preds, answers) => {
    const maxPred = Math.max(...preds)
    const maxAnswer = Math.max(...answers)
    return preds.indexOf(maxPred) === answers.indexOf(maxAnswer)
  }, trainData, validData)
}

getFiles()

function predict(trained, inputs) {
  const {model, weights} = trained
  updateNet(inputs, model, weights)
  return model[model.length - 1].map(elem => elem.value)
}


function updateValue(location, model, weights) {
  let maxValue = model[location[0]][location[1]].bias
  model[location[0]][location[1]].gradient = 0
  for (let neuron = 0; neuron < model[location[0] - 1].length; neuron++) {
    const prevValue = model[location[0] - 1][neuron].value
    const weight = weights[location[0] - 1][neuron][location[1]]
    maxValue += prevValue * (2 * sig(weight.value) - 1)
  }
  model[location[0]][location[1]].value = sig(maxValue)
}

function updateLayer(layer, model, weights) {
  layer.forEach(neuron => updateValue(neuron.location, model, weights))
}

function updateLayer1(inputArr, model) {
  for (let node of model[0]) {
    node.value = inputArr[node.location[1]]
  }
}

function updateNet(input, model, weights) {
  let timesRun = 0
  updateLayer1(input, model)
  model.forEach(layer => {
    if (timesRun > 0) {
      updateLayer(layer, model, weights)
    }
    timesRun++
  })
}

function learn(architecture, epochs, bs, lr, accuracyFunc, tset, vset) {
  let allBiases = []
  let allWeights = []
  let allWeightGrads = []
  let allNeuronGrads = []
  let epochLoss = 0

  function smallChange(arr, index) {
    const newArr = arr
    newArr[index] += 0.0001
    return newArr
  }

  const lastLayer = gpu.createKernelMap({
    grads: function lastLayerGrads(preds, answers, predsLen) {
      let currentLoss = 0
      let newLoss = 0
      for (let idx = 0; idx < predsLen; idx++) {
        currentLoss += ((preds[idx] - answers[idx]) ** 2) / predsLen
      }
      for (let idx2 = 0; idx2 < predsLen; idx2++) {
        idx2 === this.thread.x ? newLoss += ((preds[idx2] - answers[idx2] + 0.0001) ** 2) / predsLen : newLoss += ((preds[idx2] - answers[idx2]) ** 2) / predsLen
      }
      return 10000 * (newLoss - currentLoss)
    }
  }, function (preds, biasLayer, answers, predsLen, lr) {
    return biasLayer[this.thread.x] - lastLayerGrads(preds, answers, predsLen) * lr
  }).setOutput([architecture[architecture.length - 1]]).setPipeline(true).setFunctions([smallChange]).setImmutable(true)

  const biasInitArr = architecture.map((elem) => gpu.createKernel(function () {
    return 2 * Math.random() - 1
  }).setOutput([elem]).setPipeline(true))

  architecture.forEach((elem, idx) => {
    allBiases.push(biasInitArr[idx]())
  })

  const neuronGradInit = architecture.map((elem) => gpu.createKernel(function () {
    return 0
  }).setOutput([elem]).setPipeline(true))

  architecture.forEach((elem, idx) => {
    allNeuronGrads.push(neuronGradInit[idx]())
  })

  const weightInitArr = architecture.map((elem, index) => {
    if (index < architecture.length - 1) {
      return gpu.createKernel(function (idx) {
          return 2 * Math.random() - 1
        }
      ).setOutput([architecture[index + 1], elem]).setPipeline(true)
    }
  })


  architecture.forEach((elem, idx) => {
    idx < architecture.length - 1 ? allWeights.push(weightInitArr[idx](idx)) : null
  })

  const weightGradInit = architecture.map((elem, index) => {
    if (index < architecture.length - 1) {
      return gpu.createKernel(function (idx) {
          return 0
        }
      ).setOutput([architecture[index + 1], elem]).setPipeline(true)
    }
  })


  architecture.forEach((elem, idx) => {
    idx < architecture.length - 1 ? allWeightGrads.push(weightGradInit[idx](idx)) : null
  })


  const neuronsArr = architecture.map((elem, index) => gpu.createKernelMap({
    grads: function grad(thisBiasLayer, weightLayer, nextGradLayer, nextRowLength) {
      const xVal = this.thread.x

      function calculateNeuronGrad(thisBias, weightLayer, nextRowLength, nextGradLayer, xVal) {
        let res = 0
        for (let nextNeuron = 0; nextNeuron < nextRowLength; nextNeuron++) {
          res += 2 * sigSlope(thisBias) * (2 * sig(weightLayer[xVal][nextNeuron]) - 1) * nextGradLayer[nextNeuron]
        }
        return res
      }

      return calculateNeuronGrad(thisBiasLayer[this.thread.x], weightLayer, nextRowLength, nextGradLayer, xVal)
    }
  }, function (thisBiasLayer, weightLayer, nextGradLayer, lr, nextRowLength) {
    // this will return an array of the new biases
    const gradient = grad(thisBiasLayer, weightLayer, nextGradLayer, nextRowLength)
    return thisBiasLayer[this.thread.x] - gradient * lr
  }).setOutput([elem]).setPipeline(true).setFunctions([sig, sigSlope]).setImmutable(true))

  const weightsArr = architecture.map((elem, index) => gpu.createKernelMap({
      grads: function weightGrads(thisLayer, weightLayer, nextLayer) {
        const a = thisLayer[this.thread.y]
        const b = sigSlope(weightLayer[this.thread.y][this.thread.x])
        const c = nextLayer[this.thread.x]
        return 2 * a * b * c
      }
    },
    function (thisLayer, weightLayer, nextLayer, lr) {
      const gradient = weightGrads(thisLayer, weightLayer, nextLayer)
      return weightLayer[this.thread.y][this.thread.x] - gradient * lr
    }
  ).setOutput([architecture[index + 1], elem]).setPipeline(true).setFunctions([sigSlope, sig]).setImmutable(true))

  architecture.forEach((elem, idx) => {
    idx < architecture.length - 1 ? allWeights.push(weightInitArr[idx](idx)) : null
  })

  const validKernel = gpu.createKernelMap({
      loss: function calcLoss(preds, answers, answersLength) {
        let loss = 0
        for (let idx = 0; idx < answersLength; idx++) {
          loss += (preds[idx] - answers[this.thread.x][idx]) ** 2
        }
        return loss
      }
    },
    function (inputs, answers, answersLength) {
      const preds = forward(inputs[this.thread.x])
      const loss = calcLoss(preds, answers, answersLength)
      return accuracyFunc(preds, answers)
    }).setOutput([vset.length]).setImmutable(true).setFunctions([forward])

  const forwardArr = architecture.map((elem, index) => gpu.createKernel(
    function (prevLayerValues, weightLayer, prevLayerLength) {
      let sum = 0
      for (let prevNeuron = 0; prevNeuron < prevLayerLength; prevNeuron++) {
        sum += prevLayerValues[prevNeuron] * weightLayer[prevNeuron][this.thread.x]
      }
      return sig(sum)
    }
  ).setOutput([elem]).setImmutable(true).setPipeline(true).setFunctions([sig]))

  function loss(predictions, answers) {
    let loss = 0
    for (let idx = 0; idx < predictions.length; idx++) {
      loss += ((predictions[idx] - answers[idx]) ** 2) / predictions.length
    }
    return loss
  }


  function backProp(preds, answers) {
    const last = lastLayer(preds, allBiases[allBiases.length - 1], answers, answers.length, lr)
    allBiases[allBiases.length - 1] = last.result
    allNeuronGrads[architecture.length - 1] = last.grads
    for (let layer = allBiases.length - 2; layer > -1; layer--) {
      const weightInfo = weightsArr[layer](allBiases[layer], allWeights[layer], allBiases[layer + 1], lr)
      allWeights[layer] = weightInfo.result
      allWeightGrads[layer] = weightInfo.grads
      //thisLayer, weightLayer, nextLayer, lr, nextRowLength
      const neuronInfo = neuronsArr[layer](allBiases[layer], allWeights[layer], allNeuronGrads[layer + 1], lr, architecture[layer + 1])
      allBiases[layer] = neuronInfo.grads
      allNeuronGrads[layer] = neuronInfo.result
    }
  }

  function feed(inputs, layer) {
    if (layer === 0) {
      return inputs
    } else {
      const returned = forwardArr[layer](feed(inputs, layer - 1), allWeights[layer - 1], architecture[layer - 1])
      return returned
    }
  }

  function forward(inputs) {
    return feed(inputs, architecture.length - 1)
  }

  function trainOnce(inputs, answers) {
    const preds = forward(inputs)
    backProp(preds, answers)
  }

  function miniBatches(batch, size) {
    let res = []
    for (let n = 0; n < Math.ceil(batch.length / size); n++) {
      res.push(batch.slice(n * size, (n + 1) * size))
    }
    return res
  }

  function trainMiniBatch(miniBatch) {
    for (let situation of miniBatch) {
      trainOnce(situation.input, situation.answers)
    }
  }

  function trainEpoch(batch, size) {
    for (let miniBatch of miniBatches(batch, size)) {
      trainMiniBatch(miniBatch)
    }
  }

  function validateEpoch(dataSet) {
    // const inputs = dataSet.map(elem=>elem.input)
    // const answers = dataSet.map(elem=>elem.answers)
    // const valid = validKernel(inputs,answers,answers.length)
    // const acc= valid.result.reduce((a,b)=>a+b)
    // const lossVal = valid.loss.reduce((a,b)=>a+b)
    return {accuracy: 1, lossVal: 1}
  }

  console.log('----------------------------------------------------------------------------------------------------')
  for (let n = 0; n < epochs; n++) {
    console.time()
    trainEpoch(tset, bs)
    console.timeEnd()
    const {accuracy, lossVal} = validateEpoch(vset)
    console.log('| epoch', n, 'training loss', epochLoss, 'validation loss', lossVal, 'accuracy', accuracy)
    epochLoss = 0
  }
  console.log('-----------------------------------------------------------------------------------------------------')

  return {model: allBiases, weights: allWeights}
}
