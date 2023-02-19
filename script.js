import '@marcellejs/core/dist/marcelle.css';
import { Howl } from 'https://cdn.skypack.dev/howler';
import {
  batchPrediction,
  datasetBrowser,
  button,
  confusionMatrix,
  dashboard,
  dataset,
  dataStore,
  mlpClassifier,
  mlpRegressor,
  mobileNet,
  modelParameters,
  confidencePlot,
  trainingProgress,
  imageDisplay,
  text,
  textInput,
  toggle,
  trainingPlot,
  webcam,
} from '@marcellejs/core';

import { gradcam, imageClassifier } from './components';

// -----------------------------------------------------------
// INPUT PIPELINE & DATA CAPTURE
// -----------------------------------------------------------

const input = webcam();
const featureExtractor = mobileNet();

const labelInput = textInput();
labelInput.title = 'Instance label';
const capture = button('Hold to record instances');
capture.title = 'Capture instances to the training set';

const store = dataStore('localStorage');
const trainingSet = dataset('TrainingSet-move2audio', store);
const trainingSetBrowser = datasetBrowser(trainingSet);

input.$images
  .filter(() => capture.$pressed.get())
  .map(async (img) => ({
    x: await featureExtractor.process(img),
    y: labelInput.$value.get(),
    thumbnail: input.$thumbnails.get(),
  }))
  .awaitPromises()
  .subscribe(trainingSet.create);

// -----------------------------------------------------------
// TRAINING
// -----------------------------------------------------------
const b = button('Train');

/*const classifier = mlpRegressor({
  units: [64, 32],
  epochs: 20,
  batchSize: 8,
});*/

/*const classifier = mlpClassifier({ layers: [64, 32], epochs: 20 }).sync(
  store,
  'move2audio-classifier',
);*/

const classifier = imageClassifier({ layers: [10] }).sync(store, 'custom-classifier');

b.$click.subscribe(() => classifier.train(trainingSet));

const params = modelParameters(classifier);
const prog = trainingProgress(classifier);
const plotTraining = trainingPlot(classifier);

// -----------------------------------------------------------
// BATCH PREDICTION
// -----------------------------------------------------------

const batchMLP = batchPrediction('mlp', store);
const confMat = confusionMatrix(batchMLP);

const predictButton = button('Update predictions');
predictButton.$click.subscribe(async () => {
  await batchMLP.clear();
  await batchMLP.predict(classifier, trainingSet);
});

const gc = gradcam();

classifier.$training.subscribe(({ status }) => {
  if (status === 'loaded') {
    gc.setModel(classifier.model);
    gc.selectLayer();
  }
});

const $instances = trainingSetBrowser.$selected
  .filter((sel) => sel.length === 1)
  .map(([id]) => trainingSet.get(id))
  .awaitPromises()
  .map(({ x }) => x)
  .merge(input.$images.throttle(500));

const gcDisplay = [
  imageDisplay($instances),
  imageDisplay(
    $instances
      .combine((className, img) => [img, className])
      .map(([img, className]) => gc.explain(img, classifier.labels.indexOf(className)))
      .awaitPromises(),
  ),
];

// -----------------------------------------------------------
// REAL-TIME PREDICTION
// -----------------------------------------------------------

const tog = toggle('toggle prediction');

const $predictions = $instances.map(async (img) => classifier.predict(img)).awaitPromises();

const plotResults = confidencePlot($predictions);

// -----------------------------------------------------------
// DASHBOARDS
// -----------------------------------------------------------

const dash = dashboard({
  title: 'Marcelle Example - Wizard',
  author: 'Marcelle Pirates Crew',
  closable: true,
});

dash
  .page('Data Management')
  .sidebar(input, featureExtractor)
  .use([labelInput, capture], trainingSetBrowser);
dash.page('Training').use(params, b, prog, plotTraining);
dash.page('Batch Prediction').use(predictButton, confMat);
dash.page('Real-time Prediction').sidebar(input).use(tog, plotResults, gcDisplay);
dash.settings.dataStores(store).datasets(trainingSet).models(classifier);

// -----------------------------------------------------------
// MAIN APP
// -----------------------------------------------------------

// Setup the webcam
input.$mediastream.subscribe((s) => {
  document.querySelector('#my-webcam').srcObject = s;
});

setTimeout(() => {
  input.$active.set(true);
}, 200);

// Load audio files
let numLoaded = 0;
const sounds = [
  'Trap Percussion FX Loop.mp3',
  'Trap Loop Minimal 2.mp3',
  'Trap Melody Full.mp3',
].map((x) => new Howl({ src: [x], loop: true, volume: 0 }));
const onload = () => {
  numLoaded += 1;
  if (numLoaded === 3) {
    for (const x of sounds) {
      x.play();
    }
  }
};
for (const s of sounds) {
  s.once('load', onload);
}

dash.show();
