import '@marcellejs/core/dist/marcelle.css';
import {
  dashboard,
  confidencePlot,
  webcam,
  textInput,
  button,
  dataStore,
  dataset,
  datasetBrowser,
  mlpClassifier,
  modelParameters,
  trainingProgress,
  trainingPlot,
  select,
  imageDisplay,
  mobileNet,
  text,
  number,
  mlpRegressor,
} from '@marcellejs/core';
import { gradcam, imageClassifier } from './components';

// -----------------------------------------------------------
// INPUT PIPELINE & CLASSIFICATION
// -----------------------------------------------------------

const input = webcam();

const label = textInput();
label.title = 'Instance label';

const labelN = number();
labelN.title = 'Instance label';

const capture = button('Hold to record instances');
capture.title = 'Capture instances to the training set';

const store = dataStore('localStorage');
const trainingSet = dataset('training-set-dashboard', store);
const trainingSetBrowser = datasetBrowser(trainingSet);

const featureExtractor = mobileNet();

input.$images
  .filter(() => capture.$pressed.get())
  .map(async (img) => ({
    x: await featureExtractor.process(img),
    thumbnail: input.$thumbnails.get(),
    y: labelN.$value.get(),
  }))
  .awaitPromises()
  .subscribe(trainingSet.create);

// -----------------------------------------------------------
// MODEL & TRAINING
// -----------------------------------------------------------

const b = button('Train');
b.title = 'Training Launcher';
const classifier = mlpClassifier({ layers: [64, 32], epochs: 20 }).sync(store, 'classifierMLP');
//const classifier = mlpRegressor({ layers: [64, 32], epochs: 20 }).sync(store, 'MLPRegressor');

//const classifier = imageClassifier({ layers: [10] }).sync(store, 'custom-classifier');

b.$click.subscribe(() => classifier.train(trainingSet));

const params = modelParameters(classifier);
const prog = trainingProgress(classifier);
const plotTraining = trainingPlot(classifier);

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

$predictions.subscribe(({ labelN }) => {
  selectClass.$value.set(labelN);
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
// DASHBOARDS
// -----------------------------------------------------------

const dash = dashboard({
  title: 'Marcelle: Grad-CAM Example',
  author: 'Marcelle Pirates Crew',
});

dash.page('Data Management').sidebar(input).use([labelN, capture], trainingSetBrowser);
dash.page('Training').use(params, b, prog, plotTraining);
dash
  .page('Inspect Predictions')
  .sidebar(hint, wc)
  .use(trainingSetBrowser, selectClass, gcDisplay, plotResults);
dash.settings.dataStores(store).datasets(trainingSet).models(classifier);

dash.show();
