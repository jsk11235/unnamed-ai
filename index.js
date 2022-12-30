const uai = require('./model.js')
const dir = './archive/trainingSet/trainingSet'
const folderList = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
const epochs = 2
const lr = 0.002
const fromModel = `./${dir}/model1.txt`
// uai.loadDataAndTrain(dir,folderList,epochs,lr,fromModel,fromModel)
uai.predictFile(`./${dir}/4/img_49.jpg`,fromModel)
