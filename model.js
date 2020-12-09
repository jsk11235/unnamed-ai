// How to read allNeurons: allNeurons[layer][neuronOfLayer]
//.location = [layer,neuronOfLayer] .value = (activation of neuron) .bias = (bias of neuron) .grad = (gradient of neuron)
//
// How to read allWeights: allWeights[layer][neuronFrom][neuronTo]
//.location = [layer,neuronFrom,neuronTo] .value = (value of weight) .grad = (gradient of weight)

let data = []

for (let dataItem = 0; dataItem < 999; dataItem++) {
  const a = Math.random()
  const b = Math.random()
  const c = Math.random()
  data.push({inputs: [a, b, c], answers: [(a + b + c) / 3]})
}

const architecture = [3,2,2,1]
const lr = 0.001



function updateValue(location,model,weights) {
  let maxValue = model[location[0]][location[1]].bias
  for (let neuron = 0; neuron < model[location[0] - 1].length; neuron++) {
    const prevValue = model[location[0] - 1][neuron].value
    const weight = weights[location[0] - 1][neuron][location[1]]
    maxValue += prevValue * weight.value
  }
  model[location[0]][location[1]].value = Math.max(maxValue, 0)
}

function updateLayer(layer,model,weights) {
  layer.forEach(neuron => updateValue(neuron.location,model,weights))
}

function updateLayer1(inputArr,model) {
  for (let node of model[0]) {
    node.value = inputArr[node.location[1]]
  }
}

function updateNet(input,model,weights) {
  let timesRun = 0
  updateLayer1(input,model)
  model.forEach(layer => {
    if (timesRun > 0) {
      updateLayer(layer,model,weights)
    }
    timesRun++
  })
}

function learn(data, architecture,epochs,bs,lr) {
  let allNeurons
  let allWeights
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
    const layer = allNeurons[allNeurons.length - 1]
    const mappedLayer = layer.map(elem => elem.value)
    const currentLoss = loss(mappedLayer, answers)
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
    updateNet(inputs,allNeurons,allWeights)
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
      trainOnce(situation.inputs, situation.answers)
    }
  }

  function trainEpoch(batch, size) {
    for (let miniBatch of miniBatches(batch, size)) {
      trainMiniBatch(miniBatch)
    }
  }

  for (let n = 0; n < epochs; n++) {
    trainEpoch(data, bs)
  }

  return {model:allNeurons,weights:allWeights}
}

const network = learn(data,architecture,100,30,0.001)

function predict(trained,inputs){
  const {model,weights} = trained
  updateNet(inputs,model,weights)
  return model[model.length - 1].map(elem => elem.value)
}

console.log(predict(network,[0.1,0.2,0.3]))