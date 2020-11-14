const express = require('express')
const multer = require('multer')
const path = require('path')
const fs = require('fs')
const jpeg = require('jpeg-js')
const cors = require('cors')
const tf = require('@tensorflow/tfjs-node')
const mobilenet = require('@tensorflow-models/mobilenet')
const emojiFromText = require('emoji-from-text')
const http = require('http')

const app = express()
app.use(cors())

const server = http.createServer(app)

const port = process.env.PORT || 3001

const handleError = (err, res) => {
  res
      .status(500)
      .contentType('text/plain')
      .end('Oops! Something went wrong!')
}

const upload = multer({
  dest: './uploads',
  // you might also want to set some limits: https://github.com/expressjs/multer#limits
})

const readImage = path => {
  const buf = fs.readFileSync(path)
  const pixels = jpeg.decode(buf, true)
  return pixels
}

const imageByteArray = (image, numChannels) => {
  const pixels = image.data
  const numPixels = image.width * image.height
  const values = new Int32Array(numPixels * numChannels)


  for (let i = 0; i < numPixels; i++) {
    for (let channel = 0; channel < numChannels; ++channel) {
      values[i * numChannels + channel] = pixels[i * 4 + channel]
    }
  }


  return values
}

const imageToInput = (image, numChannels) => {
  const values = imageByteArray(image, numChannels)
  const outShape = [image.height, image.width, numChannels]
  const input = tf.tensor3d(values, outShape, 'int32')


  return input
}

const onlyUnique = (value, index, self) => {
  return self.indexOf(value) === index
}

const ascBy = (property) => {
  return (a, b) => (a[property] > b[property]) ? 1 : -1
}

const descBy = (property) => {
  return (a, b) => (a[property] < b[property]) ? 1 : -1
}

const NUMBER_OF_CHANNELS = 3

app.get('/', (req, res) => {
  res.send('The server is up!')
})

app.post('/get-emoji',
    upload.single('image' /* name attribute of <file> element in your form */),
    (req, res) => {
      console.log(req.file)
      const tempPath = req.file.path
      const targetPath = path.join(__dirname, req.file.path + req.file.originalname)

      if (path.extname(req.file.originalname).toLowerCase() === '.jpg') {
        fs.rename(tempPath, targetPath, async err => {
          if (err) return handleError(err, res)

          const image = readImage(targetPath)
          const input = imageToInput(image, NUMBER_OF_CHANNELS)

          // Load the model.
          const model = await mobilenet.load()

          // Classify the image.
          let predictions = await model.classify(input)
          predictions = predictions.sort(descBy('probability'))

          for (const prediction of predictions) {

            const predictions = prediction.className.match(/\b(\w+)\b/g)
            const unique = predictions.filter(onlyUnique)
            const emoji = []

            unique.forEach(predName => {
              const emojiGuess = emojiFromText(predName, true)

              emoji.push(emojiGuess.match)
            })

            prediction.emoji = emoji.sort(descBy('score'))
          }

          console.log(predictions)

          res
              .status(200)
              .json(predictions)

          fs.unlink(targetPath, err => {
            if (err) return handleError(err, res)
          })
        })
      } else {
        fs.unlink(tempPath, err => {
          if (err) return handleError(err, res)

          res
              .status(403)
              .contentType('text/plain')
              .end('Only .jpg files are allowed!')
        })
      }
    },
)

server.listen(port, () => console.log(`Example app listening at http://0.0.0.0:${port}`))
