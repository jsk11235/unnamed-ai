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

function getPixelsAsync(url) {
  return new Promise((resolve, reject) => {
    getPixels(url, (err, pixels) => err ? reject(err) : resolve(pixels))
  })
}


async function getFiles() {
  for (let trainOrValid of ['train', 'valid']) {
    for (let type of [3, 7]) {
      const fileList = await readdir(`./digits/mnist_sample/${trainOrValid}/${type}`);
      for (let file of fileList) {
        try {
          const pixels = await getPixelsAsync(`./digits/mnist_sample/${trainOrValid}/${type}/${file}`)
          const processedPixels = pixels.data.filter((elem, index) => index % 4 === 1)
          const finalPixels = Array.from(processedPixels).map(elem=>elem/255)
          const answers = type === 3 ? [1, 0] : [0, 1]
          if (trainOrValid === 'train') {
            trainData.push({input: finalPixels, answers: answers})
          } else {
            validData.push({input: finalPixels, answers: answers})
          }
        } catch (e){console.log(e)}
      }
    }
  }
  console.log('data loaded')
  learn([784,20,2],100,64,0.001,(preds,answers)=>{
    const maxPred = Math.max(...preds)
    const maxAnswer = Math.max(...answers)
    return preds.indexOf(maxPred) === answers.indexOf(maxAnswer)
  },trainData,validData,4.5)
}

// for (let dataItem = 0; dataItem < 9099; dataItem++) {
//   const a = Math.random()
//   const b = Math.random()
//   const c = Math.random()
//   trainData.push({input: [a, b, c], answers: [a + b + c]})
// }
//
// for (let dataItem = 0; dataItem < 9099; dataItem++) {
//   const a = Math.random()
//   const b = Math.random()
//   const c = Math.random()
//   validData.push({input: [a, b, c], answers: [a + b + c]})
// }
//
// function test(){
//   learn([3,400,1],150,32,0.1,(preds,answers)=>
//     Math.abs(preds-answers)<0.001
//   ,trainData,validData)
// }

getFiles()

function predict(trained, inputs) {
  const {model, weights} = trained
  updateNet(inputs, model, weights)
  return model[model.length - 1].map(elem => elem.value)
}

const architecture = [784, 20, 1]
const lr = 0.01

function updateValue(location, model, weights) {
  let maxValue = model[location[0]][location[1]].bias
  for (let neuron = 0; neuron < model[location[0] - 1].length; neuron++) {
    const prevValue = model[location[0] - 1][neuron].value
    const weight = weights[location[0] - 1][neuron][location[1]]
    maxValue += prevValue * weight.value
  }
  model[location[0]][location[1]].value = Math.max(maxValue, 0)/model[location[0]-1].length
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

function learn(architecture, epochs, bs, lr, accuracyFunc, tset, vset,decayRate) {
  let allNeurons
  let allWeights
  let currentItem = 0
  let epochLoss = 0
  allNeurons = []
  for (let a = 0; a < architecture.length; a++) {
    allNeurons.push([])
    for (let b = 0; b < architecture[a]; b++) {
      allNeurons[a].push({value: Math.random(), location: [a, b], gradient: 0, bias: 0})
    }
  }

  allWeights = []
  for (let a = 0; a < architecture.length - 1; a++) {
    allWeights.push([])
    for (let b = 0; b < architecture[a]; b++) {
      allWeights[a].push([])
      for (let c = 0; c < architecture[a + 1]; c++) {
        allWeights[a][b].push({value: Math.random(), location: [a, b, c], gradient: 0, bias: 0})
      }
    }
  }

  function loss(predictions, answers) {
    let loss = 0
    for (let idx = 0; idx < predictions.length; idx++) {
      loss += Math.abs(predictions[idx] - answers[idx])
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
      neuron.gradient = 10000 * (newLoss - currentLoss) * currentLoss**decayRate
    }
  }

  function backOnce(layer) {
    for (let neuron of layer) {
      let grad = 0
      const loc = neuron.location
      if (neuron.value > 0) {
        const nextLayer = allNeurons[loc[0] + 1]
        const weightLayer = allWeights[loc[0]]
        for (let nextNeuron = 0; nextNeuron < nextLayer.length; nextNeuron++) {
          weightLayer[loc[1]][nextNeuron].gradient = neuron.value * nextLayer[nextNeuron].gradient
          grad += weightLayer[loc[1]][nextNeuron].value * nextLayer[nextNeuron].gradient
        }
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
          if (allWeights[a][b][c].value>1){
            allWeights[a][b][c].value = 5
          } else if (allWeights[a][b][c].value<-5){
            allWeights[a][b][c].value = -5
          }
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
