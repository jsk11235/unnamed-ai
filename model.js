// How to read allNeurons: allNeurons[layer][neuronOfLayer]
//.location = [layer,neuronOfLayer] .value = (activation of neuron) .bias = (bias of neuron) .grad = (gradient of neuron)
//
// How to read allWeights: allWeights[layer][neuronFrom][neuronTo]
//.location = [layer,neuronFrom,neuronTo] .value = (value of weight) .grad = (gradient of weight)
const getPixels = require('get-pixels')
const fs = require('fs')
const util = require('util')
const readdir = util.promisify(fs.readdir)
let trainData = []
let validData = []
const files = require('fileTesting')

function filesPromise(path) {
  return new Promise((resolve, reject) => {
    fs.readFile(path, 'utf8',(err, fileData) => err ? reject(err) : resolve(fileData))
  })
}

async function getModel(fileName, dir) {
  const result = await filesPromise(`${dir}/${fileName}.txt`)
  return JSON.parse(result)
}

function save(fileName,dir,content){
  fs.writeFile(`${dir}/${fileName}.txt`,JSON.stringify(content),(err => {}))
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
    const len = fileList.length
    for (let file of fileList) {
      try {
        const pixels = await getPixelsAsync(`./trainingSet/trainingSet/${type}/${file}`)
        const processedPixels = pixels.data.filter((elem, index) => index % 4 === 1)
        const finalPixels = Array.from(processedPixels).map(elem => 2 * (elem / 255) - 1)
        let answers = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
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
  learn([784, 20, 20, 10], 120, 64, 0.0001, (preds, answers) => {
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

function learn(architecture, epochs, bs, lr, accuracyFunc, tset, vset,trained) {
  let allNeurons
  let allWeights
  let currentItem = 0
  let epochLoss = 0
  if (!trained) {
    allNeurons = []
    for (let a = 0; a < architecture.length; a++) {
      allNeurons.push([])
      for (let b = 0; b < architecture[a]; b++) {
        allNeurons[a].push({value: 2 * Math.random() - 1, location: [a, b], gradient: 0, bias: 2 * Math.random() - 1})
      }
    }

    allWeights = []
    for (let a = 0; a < architecture.length - 1; a++) {
      allWeights.push([])
      for (let b = 0; b < architecture[a]; b++) {
        allWeights[a].push([])
        for (let c = 0; c < architecture[a + 1]; c++) {
          allWeights[a][b].push({
            value: 2 * Math.random() - 1,
            location: [a, b, c],
            gradient: 0
          })
        }
      }
    }
  } else {
    allNeurons = trained.model
    allWeights = trained.weights
  }

  function loss(predictions, answers) {
    let loss = 0
    for (let idx = 0; idx < predictions.length; idx++) {
      loss += ((predictions[idx] - answers[idx]) ** 2) / predictions.length
    }
    return loss
  }

  function gradLastLayer(answers) {
    currentItem += 1
    const layer = allNeurons[allNeurons.length - 1]
    const mappedLayer = layer.map(elem => elem.value)
    const currentLoss = loss(mappedLayer, answers)
    epochLoss = currentItem * epochLoss / (currentItem + 1) + currentLoss / (currentItem + 1)
    for (let neuron of layer) {
      neuron.value += 0.0001
      const newMap = layer.map(elem => elem.value)
      const newLoss = loss(newMap, answers)
      neuron.value -= 0.0001
      neuron.gradient = 10000 * (newLoss - currentLoss)
    }
  }

  function backOnce(layer) {
    for (let neuron of layer) {
      let grad = 0
      const loc = neuron.location
      const nextLayer = allNeurons[loc[0] + 1]
      const weightLayer = allWeights[loc[0]]
      for (let nextNeuron = 0; nextNeuron < nextLayer.length; nextNeuron++) {
        weightLayer[loc[1]][nextNeuron].gradient = neuron.value * 2 * sigSlope(weightLayer[loc[1]][nextNeuron].value) * nextLayer[nextNeuron].gradient
        grad += 2 * sigSlope(neuron.value) * (2 * sig(weightLayer[loc[1]][nextNeuron].value) - 1) * nextLayer[nextNeuron].gradient
      }

      neuron.gradient = grad
    }
  }


  function backProp(answers) {
    gradLastLayer(answers)
    for (let layer = allNeurons.length - 2; layer > -1; layer--) {
      backOnce(allNeurons[layer])
    }
  }


  function trainOnce(inputs, answers) {
    updateNet(inputs, allNeurons, allWeights)
    backProp(answers)
    for (let a = 0; a < allNeurons.length; a++) {
      for (let b = 0; b < allNeurons[a].length; b++) {
        allNeurons[a][b].bias -= allNeurons[a][b].gradient * lr
        allNeurons[a][b].gradient = 0
      }
    }
    for (let a = 0; a < allWeights.length; a++) {
      for (let b = 0; b < allWeights[a].length; b++) {
        for (let c = 0; c < allWeights[a][b].length; c++) {
          allWeights[a][b][c].value -= allWeights[a][b][c].gradient * lr
          allWeights[a][b][c].gradient = 0
        }
      }
    }
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
    let lossVal = 0
    let acc = 0
    for (let situation of dataSet) {
      const prediction = predict({model: allNeurons, weights: allWeights}, situation.input)
      lossVal += loss(prediction, situation.answers)
      accuracyFunc(prediction, situation.answers) ? acc++ : null
    }
    return {accuracy: acc / dataSet.length, lossVal: lossVal / dataSet.length}
  }

  console.log('----------------------------------------------------------------------------------------------------')
  for (let n = 0; n < epochs; n++) {
    trainEpoch(tset, bs)
    const {accuracy, lossVal} = validateEpoch(vset)
    console.log('| epoch', n, 'training loss', epochLoss, 'validation loss', lossVal, 'accuracy', accuracy)
    epochLoss = 0
  }
  console.log('-----------------------------------------------------------------------------------------------------')

  return {model: allNeurons, weights: allWeights}
}
