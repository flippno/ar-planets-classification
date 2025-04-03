import * as THREE from 'three';
import { MindARThree } from 'mindar-image-three';
import { loadGLTF } from './loader.js';

// Initialize MindAR and Three.js
const mindarThree = new MindARThree({
  container: document.querySelector("#container"),
  imageTargetSrc: "./assets/targets/planets.mind"
});

const planetButton = document.getElementById("planet");
planetButton.style.display = "none";

// Destructure renderer, scene, and camera from MindAR instance
const { renderer, scene, camera } = mindarThree;

// Add a light to the scene
const light = new THREE.HemisphereLight(0xffffff, 0xbbbbff, 1);
scene.add(light);

// Load 3D assets and set up anchors
async function loadAssets() {
  const planets = {};

  const earth = await loadGLTF('../../assets/models/earth/scene.gltf');
  earth.scene.scale.set(0.3, 0.3, 0.3);
  earth.scene.position.set(0, -0.4, 0);
  earth.scene.visible = false;
  planets.earth = earth.scene;

  const earthAnchor = mindarThree.addAnchor(0);
  earthAnchor.group.add(planets.earth);

  const jupiter = await loadGLTF('../../assets/models/jupiter/scene.gltf');
  jupiter.scene.scale.set(0.3, 0.3, 0.3);
  jupiter.scene.position.set(0, -0.4, 0);
  jupiter.scene.visible = false;
  planets.jupiter = jupiter.scene;

  const jupiterAnchor = mindarThree.addAnchor(1);
  jupiterAnchor.group.add(planets.jupiter);

  const mars = await loadGLTF('../../assets/models/mars/scene.gltf');
  mars.scene.scale.set(0.3, 0.3, 0.3);
  mars.scene.position.set(0, -0.4, 0);
  mars.scene.visible = false;
  planets.mars = mars.scene;

  const marsAnchor = mindarThree.addAnchor(2);
  marsAnchor.group.add(planets.mars);

  const mercury = await loadGLTF('../../assets/models/mercury/scene.gltf');
  mercury.scene.scale.set(0.3, 0.3, 0.3);
  mercury.scene.position.set(0, -0.4, 0);
  mercury.scene.visible = false;
  planets.mercury = mercury.scene;

  const mercuryAnchor = mindarThree.addAnchor(3);
  mercuryAnchor.group.add(planets.mercury);

  const neptune = await loadGLTF('../../assets/models/neptune/scene.gltf');
  neptune.scene.scale.set(0.3, 0.3, 0.3);
  neptune.scene.position.set(0, -0.4, 0);
  neptune.scene.visible = false;
  planets.neptune = neptune.scene;

  const neptuneAnchor = mindarThree.addAnchor(4);
  neptuneAnchor.group.add(planets.neptune);

  const saturn = await loadGLTF('../../assets/models/saturn/scene.gltf');
  saturn.scene.scale.set(0.3, 0.3, 0.3);
  saturn.scene.position.set(0, -0.4, 0);
  saturn.scene.visible = false;
  planets.saturn = saturn.scene;

  const saturnAnchor = mindarThree.addAnchor(5);
  saturnAnchor.group.add(planets.saturn);

  const uranus = await loadGLTF('../../assets/models/uranus/scene.gltf');
  uranus.scene.scale.set(0.3, 0.3, 0.3);
  uranus.scene.position.set(0, -0.4, 0);
  uranus.scene.visible = false;
  planets.uranus = uranus.scene;

  const uranusAnchor = mindarThree.addAnchor(6);
  uranusAnchor.group.add(planets.uranus);

  const venus = await loadGLTF('../../assets/models/venus/scene.gltf');
  venus.scene.scale.set(0.3, 0.3, 0.3);
  venus.scene.position.set(0, -0.4, 0);
  venus.scene.visible = false;
  planets.venus = venus.scene;

  const venusAnchor = mindarThree.addAnchor(7);
  venusAnchor.group.add(planets.venus);

  return planets;
}
const planets = await loadAssets();

// --- Load your own TensorFlow.js model ---
// Adjust modelURL to the correct path where your model files reside.
const modelURL = "./my-planet-model1.json";
const model = await tf.loadLayersModel(modelURL);
console.log("Custom model loaded successfully");

// Specify the input dimensions expected by your model
const MODEL_INPUT_WIDTH = 224;  // change as needed
const MODEL_INPUT_HEIGHT = 224; // change as needed

// Function to preprocess video frames for prediction
function preprocessFrame(video) {
  return tf.tidy(() => {
    const frame = tf.browser.fromPixels(video);
    // Resize to the model's input dimensions
    const resized = tf.image.resizeBilinear(frame, [MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH]);
    // Normalize pixel values (if required by your model)
    const normalized = resized.div(255);
    // Expand dims to create a batch of 1
    return normalized.expandDims(0);
  });
}

let running = false; // Flag to control detection loop

// Arrays to record FPS and latency measurements
let fpsRecords = [];
let latencyRecords = [];

// Start function: starts MindAR, renders the scene, and begins prediction loop
const start = async () => {
  running = true; // Enable detection loop

  // Clear any previous records
  fpsRecords = [];
  latencyRecords = [];

  await mindarThree.start();
  renderer.setAnimationLoop(() => {
    renderer.render(scene, camera);
  });

  const video = mindarThree.video;
  let skipCount = 0;

  const fpsDisplay = document.getElementById("fpsDisplay"); // Add a div in HTML for FPS display
  const latencyDisplay = document.getElementById("latencyDisplay"); // Add a div in HTML for latency display

  let lastFrameTime = performance.now(); // Initialize last frame timestamp

  const detect = async () => {
    // Skip frames to reduce processing load
    if (skipCount < 10) {
      skipCount += 1;
      window.requestAnimationFrame(detect);
      return;
    }
    skipCount = 0;

    const startTime = performance.now(); // Start timing

    // Preprocess the video frame and predict
    const inputTensor = preprocessFrame(video);
    const predictionTensor = model.predict(inputTensor);
    // Assuming your model outputs a probability vector.
    // Convert the tensor to a JavaScript array.
    const predictions = await predictionTensor.data();
    // Clean up the tensors created in this frame
    inputTensor.dispose();
    predictionTensor.dispose();

    const endTime = performance.now(); // End timing

    // Calculate latency in ms for this prediction
    const latency = endTime - startTime;
    latencyRecords.push(latency);

    // Calculate FPS based on the time between frames
    const currentTime = performance.now();
    const fps = 1000 / (currentTime - lastFrameTime);
    lastFrameTime = currentTime;
    fpsRecords.push(fps);

    // Update UI
    fpsDisplay.innerText = `FPS: ${fps.toFixed(2)}`;
    latencyDisplay.innerText = `Latency: ${latency.toFixed(2)} ms`;

    // Map predictions to corresponding class names.
    // Adjust this part based on how your model’s output is ordered.
    const classNames = ['earth', 'uranus', 'venus', 'neptune', 'jupiter', 'mercury', 'saturn', 'mars'];
    for (let i = 0; i < predictions.length; i++) {
      console.log(`${classNames[i]}: ${(predictions[i] * 100).toFixed(2)}%`);
    }
    let detectedPlanet = null;
    predictions.forEach((probability, index) => {
      // Adjust threshold values as needed.
      if (probability >= 0.75) {
        detectedPlanet = classNames[index];
      }
    });

    // Update the planet button based on prediction
    if (detectedPlanet) {
      planetButton.style.display = "block";
      planetButton.innerHTML = detectedPlanet.charAt(0).toUpperCase() + detectedPlanet.slice(1);
    } else {
      planetButton.style.display = "none";
    }

    if (running) {
      window.requestAnimationFrame(detect);
    }
  };

  window.requestAnimationFrame(detect);
};

// Handle planet button clicks to show the corresponding 3D model
planetButton.addEventListener("click", () => {
  const planetName = planetButton.innerText.toLowerCase();
  // Hide all models first
  Object.values(planets).forEach((mesh) => {
    mesh.visible = false;
  });
  // Show only the selected planet
  if (planets[planetName]) {
    planets[planetName].visible = true;
  }
});

// Start/Stop buttons
const startButton = document.querySelector("#startButton");
const stopButton = document.querySelector("#stopButton");

startButton.addEventListener("click", () => {
  start();
});

// Function to compute the average of an array
const calculateAverage = (array) => {
  if (array.length === 0) return 0;
  const sum = array.reduce((acc, curr) => acc + curr, 0);
  return sum / array.length;
};

stopButton.addEventListener("click", () => {
  running = false; // Stop detection loop
  mindarThree.stop();
  renderer.setAnimationLoop(null);

  // Calculate average FPS and latency
  const avgFPS = calculateAverage(fpsRecords);
  const avgLatency = calculateAverage(latencyRecords);

  // Display the averages in the UI
  const fpsDisplay = document.getElementById("fpsDisplay");
  const latencyDisplay = document.getElementById("latencyDisplay");

  fpsDisplay.innerText = `Avg FPS: ${avgFPS.toFixed(2)}`;
  latencyDisplay.innerText = `Avg Latency: ${avgLatency.toFixed(2)} ms`;
});
