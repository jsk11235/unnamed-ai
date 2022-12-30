const getPixels = require('get-pixels')
const fs = require('fs')
const util = require('util')
const readdir = util.promisify(fs.readdir)

class UnnamedAI {
  filesPromise(path) {
    return new Promise((resolve, reject) => {
      fs.readFile(path, 'utf8', (err, fileData) => err ? reject(err) : resolve(fileData))
    })
  }

  async getModel(filePath) {
    const result = await this.filesPromise(filePath)
    return JSON.parse(result)
  }

  save(filePath, content) {
    fs.writeFile(filePath, JSON.stringify(content), (err => {
    }))
  }


  getPixelsAsync(url) {
    return new Promise((resolve, reject) => {
      getPixels(url, (err, pixels) => err ? reject(err) : resolve(pixels))
    })
  }

  sig(num) {
    return 1 / (1 + Math.exp(-1 * num))
  }

  sigSlope(num) {
    const s = this.sig(num)
    return s*(1-s)
  }

  async loadDataAndTrain(dir, folderList, epochs, lr, fromModel, toModel) {
    let trainData = []
    let validData = []
    for (let type of folderList) {
      const fileList = await readdir(`${dir}/${type}`);
      let fileNum = 1
      const len = fileList.length
      for (let file of fileList) {
        try {
          const pixels = await this.getPixelsAsync(`${dir}/${type}/${file}`)
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
    // const m = null
    const m = await this.getModel(fromModel)
    const model = this.learn([784, 20, 20, 10], epochs, 64, lr, (preds, answers) => {
      const maxPred = Math.max(...preds)
      const maxAnswer = Math.max(...answers)
      return preds.indexOf(maxPred) === answers.indexOf(maxAnswer)
    }, trainData, validData, m)
    await this.save(toModel, model)
  }

  predict(trained, inputs) {
    const {model, weights} = trained
    this.updateNet(inputs, model, weights)
    return model[model.length - 1].map(elem => elem.value)
  }

  updateValue(location, model, weights) {
    let maxValue = model[location[0]][location[1]].bias
    model[location[0]][location[1]].gradient = 0
    for (let neuron = 0; neuron < model[location[0] - 1].length; neuron++) {
      const prevValue = model[location[0] - 1][neuron].value
      const weight = weights[location[0] - 1][neuron][location[1]]
      maxValue += prevValue * (2 * this.sig(weight.value) - 1)
    }
    model[location[0]][location[1]].value = this.sig(maxValue)
  }

  updateLayer(layer, model, weights) {
    layer.forEach(neuron => this.updateValue(neuron.location, model, weights))
  }

  updateLayer1(inputArr, model) {
    for (let node of model[0]) {
      node.value = inputArr[node.location[1]]
    }
  }


   updateNet(input, model, weights) {
    let timesRun = 0
    this.updateLayer1(input, model)
    model.forEach(layer => {
      if (timesRun > 0) {
        this.updateLayer(layer, model, weights)
      }
      timesRun++
    })
  }

  learn(architecture, epochs, bs, lr, accuracyFunc, tset, vset, trained) {
    const instance = this
    let currentItem = 0
    let epochLoss = 0
    if (!trained) {
      this.allNeurons = []
      for (let a = 0; a < architecture.length; a++) {
        this.allNeurons.push([])
        for (let b = 0; b < architecture[a]; b++) {
          this.allNeurons[a].push({
            value: 2 * Math.random() - 1,
            location: [a, b],
            gradient: 0,
            bias: 2 * Math.random() - 1,
            velocity: 0
          })
        }
      }

      this.allWeights
        = []
      for (let a = 0; a < architecture.length - 1; a++) {
        this.allWeights
          .push([])
        for (let b = 0; b < architecture[a]; b++) {
          this.allWeights
            [a].push([])
          for (let c = 0; c < architecture[a + 1]; c++) {
            this.allWeights
              [a][b].push({
              value: 2 * Math.random() - 1,
              location: [a, b, c],
              gradient: 0,
              velocity: 0
            })
          }
        }
      }
    } else {
      this.allNeurons = trained.model
      this.allWeights
        = trained.weights
    }

    function loss(predictions, answers) {
      let loss = 0
      for (let idx = 0; idx < predictions.length; idx++) {
        loss += ((predictions[idx] - answers[idx]) ** 2) / predictions.length
      }
      return loss
    }

    function gradLastLayer(answers) {
      const layer = instance.allNeurons[instance.allNeurons.length - 1]
      const mappedLayer = layer.map(elem => elem.value)
      const currentLoss = loss(mappedLayer, answers)
      epochLoss = (currentItem * epochLoss + currentLoss) / (currentItem + 1)
      currentItem += 1
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
        const nextLayer = instance.allNeurons[loc[0] + 1]
        const weightLayer = instance.allWeights
          [loc[0]]
        for (let nextNeuron = 0; nextNeuron < nextLayer.length; nextNeuron++) {
          const v = nextLayer[nextNeuron].value
          const roc = v * (1 - v)
          weightLayer[loc[1]][nextNeuron].gradient = roc * neuron.value * 2 * instance.sigSlope(weightLayer[loc[1]][nextNeuron].value) * nextLayer[nextNeuron].gradient
          grad += roc * (2 * instance.sig(weightLayer[loc[1]][nextNeuron].value) - 1) * nextLayer[nextNeuron].gradient
        }

        neuron.gradient = grad
      }
    }


    function backProp(answers) {
      gradLastLayer(answers)
      for (let layer = instance.allNeurons.length - 2; layer > -1; layer--) {
        backOnce(instance.allNeurons[layer])
      }
    }

    function trainOnce(inputs, answers) {
      instance.updateNet(inputs, instance.allNeurons, instance.allWeights)
      backProp(answers)
      for (let a = 0; a < instance.allNeurons.length; a++) {
        for (let b = 0; b < instance.allNeurons[a].length; b++) {
          const roc = instance.allNeurons[a][b].value * (1 - instance.allNeurons[a][b].value)
          instance.allNeurons[a][b].bias -= instance.allNeurons[a][b].gradient * roc * lr
          instance.allNeurons[a][b].gradient = 0
        }
      }

      for (let a = 0; a < instance.allWeights.length; a++) {
        for (let b = 0; b < instance.allWeights[a].length; b++) {
          for (let c = 0; c < instance.allWeights[a][b].length; c++) {
            instance.allWeights[a][b][c].value -= instance.allWeights[a][b][c].gradient * lr
            instance.allWeights[a][b][c].gradient = 0
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
        const prediction = instance.predict({model: instance.allNeurons, weights: instance.allWeights}, situation.input)
        lossVal += loss(prediction, situation.answers)
        accuracyFunc(prediction, situation.answers) ? acc++ : null
      }
      return {accuracy: acc / dataSet.length, lossVal: lossVal / dataSet.length}
    }

    console.log('----------------------------------------------------------------------------------------------------')
    for (let n = 0; n < epochs; n++) {
      console.time('epoch time')
      trainEpoch(tset, bs)
      currentItem = 0
      const {accuracy, lossVal} = validateEpoch(vset)
      console.log('| epoch', n, 'training loss', epochLoss, 'validation loss', lossVal, 'accuracy', accuracy)
      console.timeEnd('epoch time')
      epochLoss = 0
    }
    console.log('-----------------------------------------------------------------------------------------------------')

    return {model: this.allNeurons, weights: this.allWeights}
  }

  async predictFile(filePath, modelPath) {
    const model = await this.getModel(modelPath)
    const pixels = await this.getPixelsAsync(filePath)
    const processedPixels = pixels.data.filter((elem, index) => index % 4 === 1)
    const finalPixels = Array.from(processedPixels).map(elem => 2 * (elem / 255) - 1)
    this.predict(model, finalPixels).map((elem, idx) => {
      console.log(`${Math.floor(elem * 10000) / 100}% confidence that this is a ${idx}`)
    })
  }
}

module.exports = new UnnamedAI()
