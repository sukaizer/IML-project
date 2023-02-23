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
  mlpClassifier,
  mlpRegressor,
} from '@marcellejs/core';
import { gradcam, imageClassifier } from './components';

// -----------------------------------------------------------
// INPUT PIPELINE & CLASSIFICATION
// -----------------------------------------------------------

const input = webcam();

const label = number();
label.title = 'Instance label';
const capture = button('Hold to record instances');
capture.title = 'Capture instances to the training set';

const store = dataStore('localStorage');
const trainingSet = dataset('training-set-dashboard', store);
const trainingSetBrowser = datasetBrowser(trainingSet);

const trainingSetMLP = dataset('training-set-mlp-dashboard', store);
const trainingSetRegressor = dataset('training-set-reg-dashboard', store);

const featureExtractor = mobileNet();

input.$images
  .filter(() => capture.$pressed.get())
  .map((x) => ({ x, y: label.$value.get(), thumbnail: input.$thumbnails.get() }))
  .subscribe(trainingSet.create);

input.$images
  .filter(() => capture.$pressed.get())
  .map(async (img) => ({
    x: await featureExtractor.process(img),
    thumbnail: input.$thumbnails.get(),
    y: label.$value.get(),
  }))
  .awaitPromises()
  .subscribe(trainingSetMLP.create);

input.$images
  .filter(() => capture.$pressed.get())
  .map(async (img) => ({
    x: await featureExtractor.process(img),
    thumbnail: input.$thumbnails.get(),
    y: label.$value.get(),
  }))
  .awaitPromises()
  .subscribe(trainingSetRegressor.create);

// -----------------------------------------------------------
// MODEL & TRAINING
// -----------------------------------------------------------

const b = button('Train');
b.title = 'Training Launcher';

const classifier = imageClassifier({ layers: [10] }).sync(store, 'custom-classifier');

const classifierMLP = mlpClassifier({ layers: [128, 64], epochs: 30 }).sync(store, 'mlp');

const classifierRegressor = mlpRegressor({ units: [64, 32], epochs: 50 });

b.$click.subscribe(() => {
  classifier.train(trainingSet);
  classifierMLP.train(trainingSetMLP);
  classifierRegressor.train(trainingSetRegressor);
});

const paramsImage = modelParameters(classifier);
paramsImage.title = 'Image classifier : Parameters';
const progImage = trainingProgress(classifier);
progImage.title = 'Image classifier : Training Progress';
const plotTrainingImage = trainingPlot(classifier);
plotTrainingImage.title = 'Image classifier : Plot';

const paramsMLP = modelParameters(classifierMLP);
paramsMLP.title = 'MLP : Parameters';
const progressMLP = trainingProgress(classifierMLP);
progressMLP.title = 'MLP: Training Progress';

const paramsRegressor = modelParameters(classifierRegressor);
paramsMLP.title = 'MLP Regression : Parameters';
const progressRegressor = trainingProgress(classifierRegressor);
progressMLP.title = 'MLP Regression: Training Progress';

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

const hint = text('Select an image in the dataset browser to inspect predictions and Grad-CAM');
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
  .merge(wc.$images.throttle(500));

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
  title: 'Marcelle: Grad-CAM Example',
  author: 'Marcelle Pirates Crew',
});

dash.page('Data Management').sidebar(input).use([label, capture], trainingSetBrowser);
dash
  .page('Training')
  .use(
    b,
    'Image Classifier',
    paramsImage,
    progImage,
    'MLP Regressor',
    paramsRegressor,
    progressRegressor,
    'MLP',
    paramsMLP,
    progressMLP,
    'Plots',
    plotTrainingImage,
  );
dash
  .page('Inspect Predictions')
  .sidebar(hint, wc)
  .use(musicButton, trainingSetBrowser, selectClass, gcDisplay, plotResults);
dash.settings
  .dataStores(store)
  .datasets(trainingSet, trainingSetRegressor, trainingSetMLP)
  .models(classifier);

dash.show();
