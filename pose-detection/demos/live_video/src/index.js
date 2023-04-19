/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import '@tensorflow/tfjs-backend-webgl';
import '@tensorflow/tfjs-backend-webgpu';

import * as tf from '@tensorflow/tfjs-core';
import * as posedetection from '@tensorflow-models/pose-detection';

import { Camera, Context } from './camera';
import { RendererCanvas2d } from './renderer_canvas2d';
import { setupDatGui } from './option_panel';
import { STATE } from './params';
import { setupStats } from './stats_panel';
import { setBackendAndEnvFlags } from './util';
import OSCTransport from './osc_transport';
import MIDITransport from './midi_transport';
import { PoseClassifierHelper } from './pose_classifier_helper';

let detector, camera, stats;
let startInferenceTime, numInferences = 0;
let inferenceTimeSum = 0, lastPanelUpdate = 0;
let rafId;
let renderer = null;

async function createDetector() {
  switch (STATE.model) {
    case posedetection.SupportedModels.MoveNet:
      let modelType;
      if (STATE.modelConfig.type == 'lightning') {
        modelType = posedetection.movenet.modelType.SINGLEPOSE_LIGHTNING;
      } else if (STATE.modelConfig.type == 'thunder') {
        modelType = posedetection.movenet.modelType.SINGLEPOSE_THUNDER;
      } else if (STATE.modelConfig.type == 'multipose') {
        modelType = posedetection.movenet.modelType.MULTIPOSE_LIGHTNING;
      }
      const modelConfig = { modelType };

      if (STATE.modelConfig.customModel !== '') {
        modelConfig.modelUrl = STATE.modelConfig.customModel;
      }
      if (STATE.modelConfig.type === 'multipose') {
        modelConfig.enableTracking = STATE.modelConfig.enableTracking;
      }
      console.warn(`loading model from custom url; model switching will not work.`)
      return posedetection.createDetector(STATE.model, {
        ...modelConfig,
        modelUrl: `${window.location.protocol}//${window.location.hostname}:${window.location.port}/models/my-model.json`
      });
  }
}

async function checkGuiUpdate() {
  if (STATE.isTargetFPSChanged || STATE.isSizeOptionChanged) {
    camera = await Camera.setup(STATE.camera);
    STATE.isTargetFPSChanged = false;
    STATE.isSizeOptionChanged = false;
  }

  if (STATE.isModelChanged || STATE.isFlagChanged || STATE.isBackendChanged) {
    STATE.isModelChanged = true;

    window.cancelAnimationFrame(rafId);

    if (detector != null) {
      detector.dispose();
    }

    if (STATE.isFlagChanged || STATE.isBackendChanged) {
      await setBackendAndEnvFlags(STATE.flags, STATE.backend);
    }

    try {
      detector = await createDetector(STATE.model);
    } catch (error) {
      detector = null;
      alert(error);
    }

    STATE.isFlagChanged = false;
    STATE.isBackendChanged = false;
    STATE.isModelChanged = false;
  }
}

function beginEstimatePosesStats() {
  startInferenceTime = (performance || Date).now();
}

function endEstimatePosesStats() {
  const endInferenceTime = (performance || Date).now();
  inferenceTimeSum += endInferenceTime - startInferenceTime;
  ++numInferences;

  const panelUpdateMilliseconds = 1000;
  if (endInferenceTime - lastPanelUpdate >= panelUpdateMilliseconds) {
    const averageInferenceTime = inferenceTimeSum / numInferences;
    inferenceTimeSum = 0;
    numInferences = 0;
    stats.customFpsPanel.update(
      1000.0 / averageInferenceTime, 120 /* maxValue */);
    lastPanelUpdate = endInferenceTime;
  }
}

async function renderResult() {
  if (!camera ||camera.video.readyState < 2) {
    return;
  }

  let poses = null;

  // Detector can be null if initialization failed (for example when loading
  // from a URL that does not exist).
  if (detector != null) {
    // FPS only counts the time it takes to finish estimatePoses.
    beginEstimatePosesStats();

    // Detectors can throw errors, for example when using custom URLs that
    // contain a model that doesn't provide the expected output.
    try {
      poses = await detector.estimatePoses(
        camera.video,
        {
          maxPoses: STATE.modelConfig.maxPoses,
          flipHorizontal: false
        }
      );
    } catch (error) {
      detector.dispose();
      detector = null;
      alert(error);
    }

    if (poses.length) {
      STATE.lastKeyPoints = posedetection.calculators.keypointsToNormalizedKeypoints(poses[0].keypoints, { width: camera.video.width, height: camera.video.height });
    } else {
      STATE.lastKeyPoints = null;
    }

    endEstimatePosesStats();
  }

  const rendererParams = [camera.video, poses, STATE.isModelChanged];

  renderer.draw(rendererParams);

  osc.transmitPoses(poses, { width: camera.video.width, height: camera.video.height }, {
    scoreThreshold: STATE.modelConfig.scoreThreshold,
    scalePose: STATE.modelConfig.scalePose || false,
    outputDebug: output_debug
  });
}

async function runFrame() {
  await checkGuiUpdate();

  if (STATE.camera.enabled) {

  } else {
    // loop video
    if (video && video.paused) {
      video.play();
    }
  }

  if (!STATE.isModelChanged) {
    await renderResult();
  }

  rafId = requestAnimationFrame(runFrame);
}

async function app() {
  STATE.camera.runCamera = () => { runCamera() };

  const gui = await setupDatGui();
  stats = setupStats();

  if (poseClassifierHelper) {
    poseClassifierHelper.addGuiElements(gui);
  }

  await setBackendAndEnvFlags(STATE.flags, STATE.backend);
  detector = await createDetector();

  const runButton = document.getElementById('submit');
  runButton.onclick = runVideo;

  const uploadButton = document.getElementById('videofile');
  uploadButton.onchange = updateVideo;

  await runCamera();
  runFrame();
};

async function runCamera() {
  // Clear reference to any previous uploaded video.
  if (camera?.video) {
    URL.revokeObjectURL(camera.video.currentSrc);
    camera.video.src = '';
  }

  camera = await Camera.setup(STATE.camera);
  const canvas = document.getElementById('output');
  canvas.width = camera.video.width;
  canvas.height = camera.video.height;
  renderer = new RendererCanvas2d(canvas, document.querySelector('#scatter-gl-container'));
  STATE.camera.enabled = true;
}

async function updateVideo(event) {
  // Clear reference to any previous uploaded video.
  if (camera?.video?.currentSrc) {
    URL.revokeObjectURL(camera.video.currentSrc);
  }

  STATE.video.file = event.target.files[0];
}

let video;

async function runVideo() {
  camera = new Context();

  if (camera?.video) {
    camera.video.srcObject = null;
  }

  camera.video.src = URL.createObjectURL(STATE.video.file);

  // Wait for video to be loaded.
  camera.video.load();
  await new Promise((resolve) => {
    camera.video.onloadeddata = () => {
      resolve(video);
    };
  });

  const videoWidth = camera.video.videoWidth;
  const videoHeight = camera.video.videoHeight;
  // Must set below two lines, otherwise video element doesn't show.
  camera.video.width = videoWidth;
  camera.video.height = videoHeight;
  camera.canvas.width = videoWidth;
  camera.canvas.height = videoHeight;
  const canvasContainer = document.querySelector('.canvas-wrapper');
  canvasContainer.style = `width: ${videoWidth}px; height: ${videoHeight}px`;
  renderer = new RendererCanvas2d(camera.canvas);

  STATE.camera.enabled = false;

  // Warming up pipeline.
  const [runtime, $backend] = STATE.backend.split('-');

  if (runtime === 'tfjs') {
    const warmUpTensor =
      tf.fill([camera.video.height, camera.video.width, 3], 0, 'float32');
    await detector.estimatePoses(
      warmUpTensor,
      { maxPoses: STATE.modelConfig.maxPoses, flipHorizontal: false });
    warmUpTensor.dispose();
  }

  video = camera.video;
  video.pause();
  video.currentTime = 0;
  video.play();
}


const osc = new OSCTransport();
const midi = new MIDITransport;
const poseClassifierHelper = new PoseClassifierHelper(STATE);

const output_debug = new RendererCanvas2d(document.getElementById('output_debug'));

app();
