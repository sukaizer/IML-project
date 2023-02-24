import '@marcellejs/core/dist/marcelle.css';
import { Howl } from 'https://cdn.skypack.dev/howler';
import {
  dashboard,
  confidencePlot,
  webcam,
  textInput,
  number,
  button,
  dataStore,
  dataset,
  datasetBrowser,
  modelParameters,
  trainingProgress,
  trainingPlot,
  select,
  imageDisplay,
  text,
  mobileNet,
  mlpRegressor,
} from '@marcellejs/core';
import { gradcam, imageClassifier } from './components';

// -----------------------------------------------------------
// INPUT PIPELINE & CLASSIFICATION
// -----------------------------------------------------------

const hintData = text('Activate the webcam and start taking snapshots to create your dataset');
hintData.title = 'Hint';

const input = webcam();
// for reg uses numeric values
const label = number();
label.title = 'Instance label';
const capture = button('Hold to record instances');
capture.title = 'Capture instances to the training set';

const store = dataStore('localStorage');
const trainingSet = dataset('training-set-dashboard', store);
const trainingSetBrowser = datasetBrowser(trainingSet);
//second trainingset for regression
const trainingSetRegressor = dataset('training-set-reg-dashboard', store);

//const featureExtractor = mobileNet();

input.$images
  .filter(() => capture.$pressed.get())
  .map((x) => ({ x, y: label.$value.get(), thumbnail: input.$thumbnails.get() }))
  .subscribe(trainingSet.create);

/*
input.$images
  .filter(() => capture.$pressed.get())
  .map(async (img) => ({
    x: await featureExtractor.process(img),
    thumbnail: input.$thumbnails.get(),
    y: label.$value.get(),
  }))
  .awaitPromises()
  .subscribe(trainingSetRegressor.create);
  */

// -----------------------------------------------------------
// MODEL & TRAINING
// -----------------------------------------------------------

const hintTrain = text(
  'Train the model with your data here. You can change the parameters to better fit your data',
);
hintTrain.title = 'Hint';

const b = button('Train');
b.title = 'Training Launcher';

const classifier = imageClassifier({ layers: [10] }).sync(store, 'custom-classifier');

const regressionMLP = mlpRegressor({ units: [64, 32], epochs: 20 }).sync(store, 'reg');
regressionMLP.$training.subscribe(console.log);

b.$click.subscribe(() => {
  classifier.train(trainingSet);
  //regressionMLP.train(trainingSetRegressor);
});

const paramsImage = modelParameters(classifier);
paramsImage.title = 'Image classifier : Parameters';
const progImage = trainingProgress(classifier);
progImage.title = 'Image classifier : Training Progress';

const paramsRegressor = modelParameters(regressionMLP);
paramsRegressor.title = 'MLP Regression : Parameters';
const progressRegressor = trainingProgress(regressionMLP);
paramsRegressor.title = 'MLP Regression: Training Progress';
const plotTrainingReg = trainingPlot(regressionMLP, {
  Loss: ['loss', 'lossVal'],
  'Mean Absolute Error': ['meanAbsoluteError2', 'meanAbsoluteError2Val'],
});

// -----------------------------------------------------------
// SINGLE IMAGE PREDICTION
// -----------------------------------------------------------

const gc = gradcam();

classifier.$training.subscribe(({ status }) => {
  if (['loaded', 'success'].includes(status)) {
    gc.setModel(classifier.model);
    gc.selectLayer();
  }
});

const hint = text(
  'Select an image in the dataset browser to inspect predictions and Grad-CAM. Click on play to play music and see what happens',
);
hint.title = 'Hint';
const selectClass = select([]);
selectClass.title = 'Select the class to inspect';
classifier.$training.subscribe(({ status }) => {
  if (['success', 'loaded'].includes(status)) {
    selectClass.$options.set(classifier.labels);
  }
});

const wc = webcam();

const $instances = trainingSetBrowser.$selected
  .filter((sel) => sel.length === 1)
  .map(([id]) => trainingSet.get(id))
  .awaitPromises()
  .map(({ x }) => x)
  .merge(wc.$images.throttle(300));

const $predictions = $instances.map(async (img) => classifier.predict(img)).awaitPromises();

$predictions.subscribe(({ label }) => {
  selectClass.$value.set(label);
});

const plotResults = confidencePlot($predictions);

const gcDisplay = [
  imageDisplay($instances),
  imageDisplay(
    $instances
      .combine((className, img) => [img, className], selectClass.$value)
      .map(([img, className]) => gc.explain(img, classifier.labels.indexOf(className)))
      .awaitPromises(),
  ),
];
/*
const $predictionsRegressor = wc.$images
  .throttle(500)
  .map(async (img) => regressionMLP.predict(await featureExtractor.process(img)))
  .awaitPromises();

$predictionsRegressor.subscribe(async ({ label }) => {
  console.log(label);
});

const plotResultsReg = confidencePlot($predictionsRegressor);
*/
// -----------------------------------------------------------
// SOUNDS
// -----------------------------------------------------------
let isPlaying = null;

const sound = new Howl({
  src: ['src/sound.m4a'],
});
const sound2 = new Howl({
  src: ['src/sound2.mp3'],
});

$predictions.subscribe(async ({ label }) => {
  //console.log(label);
  if (label === classifier.labels[0]) {
    if (isPlaying === 0) {
      //sound.play();
    } else {
      sound.play();
      sound2.pause();
      isPlaying = 0;
    }
  }
  if (label === classifier.labels[1]) {
    if (isPlaying === 1) {
      //sound.play();
    } else {
      sound.pause();
      sound2.play();
      isPlaying = 1;
    }
  }
});

let playing = false;
const musicButton = button('Push me');
musicButton.title = 'play music';
musicButton.$click.subscribe(() => {
  if (playing) {
    sound.pause();
    playing = false;
  } else {
    sound.play();
    playing = true;
  }
});

// -----------------------------------------------------------
// DASHBOARDS
// -----------------------------------------------------------

const dash = dashboard({
  title: 'FaceMusic',
  author: 'Aki, Alexandre, David, Alegria',
});

dash.page('Data Management').sidebar(hintData, input).use([label, capture], trainingSetBrowser);
dash
  .page('Training')
  .sidebar(hintTrain)
  .use(
    b,
    'Image Classifier',
    paramsImage,
    progImage,
    'MLP Regressor',
    paramsRegressor,
    progressRegressor,
    'Regression plots',
    plotTrainingReg,
  );
dash
  .page('Inspect Predictions')
  .sidebar(hint, wc, musicButton)
  .use('Dataset', trainingSetBrowser, selectClass, 'Visualisation', gcDisplay, plotResults);
dash.settings.dataStores(store).datasets(trainingSet, trainingSetRegressor).models(classifier);

$('#dash').click(function () {
  dash.show();
});
