(function () {
  // Set up canvas and WebGL context (existing code)
  const canvas = document.getElementById('glcanvas');
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  const gl = canvas.getContext('webgl');
  if (!gl) {
    alert("WebGL not supported.");
    return;
  }
  data_server = "http://10.204.100.101:5000/";

  // Adjust layout for narrow controls on the right
  const mainContainer = document.createElement('div');
  mainContainer.classList.add('container-fluid', 'd-flex', 'p-0');

  const canvasContainer = document.createElement('div');
  canvasContainer.classList.add('flex-grow-1', 'p-0');
  canvasContainer.appendChild(canvas);

  const controlsContainer = document.createElement('div');
  controlsContainer.classList.add('controls-container', 'p-3', 'bg-light');
  controlsContainer.style.width = '600px'; // Set a fixed width for the controls
  controlsContainer.appendChild(document.getElementById('controls'));

  mainContainer.appendChild(canvasContainer);
  mainContainer.appendChild(controlsContainer);

  document.body.appendChild(mainContainer);

  // Function to resize the canvas and update WebGL viewport
  function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    gl.viewport(0, 0, canvas.width, canvas.height);
  }

  // Attach resize event listener
  window.addEventListener('resize', resizeCanvas);

  // Call resizeCanvas initially to set up the correct size
  resizeCanvas();

  // UI elements
  const expSelect = document.getElementById('expSelect');
  const fileNameDOM = document.getElementById('fileName');
  const epochSlider = document.getElementById('epochSlider');
  const dpmSlider = document.getElementById('dpmSlider');
  const epochValue = document.getElementById('epochValue');
  const dpmValue = document.getElementById('dpmValue');
  const frameIdInput = document.getElementById('frameId');
  //update cdValue
  const cdValue = document.getElementById("cdValue");
  const dpmCdValue = document.getElementById("dpmCdValue");

  let sortedEpochs = []; // Declare sortedEpochs in a higher scope

  // Apply Bootstrap classes to UI elements
  document.getElementById('controls').classList.add('container', 'mt-3');
  expSelect.classList.add('form-select', 'mb-3');
  fileNameDOM.classList.add('badge', 'bg-primary', 'ms-2');
  epochSlider.classList.add('form-range', 'mb-3');
  dpmSlider.classList.add('form-range', 'mb-3');
  epochValue.classList.add('badge', 'bg-primary', 'ms-2');
  dpmValue.classList.add('badge', 'bg-primary', 'ms-2');

  cdValue.classList.add('badge', 'bg-primary', 'ms-2');
  dpmCdValue.classList.add('badge', 'bg-primary', 'ms-2');
  frameIdInput.classList.add('form-control', 'mb-3');

  // // Add Bootstrap grid layout

  function constructFileName(epoch, set, frameId) {

    // Zero-pad epoch to 6 digits
    const paddedEpoch = epoch.toString().padStart(6, '0');
    return `sample_ep_${paddedEpoch}_gt0_${set}-idx-${frameId}.json`;
  }

  epochSlider.addEventListener('input', () => {
    epochValue.textContent = epochSlider.value;

    // Find the nearest value in sortedEpochs
    const sliderValue = parseInt(epochSlider.value, 10);
    const nearestEpoch = sortedEpochs.reduce((prev, curr) =>
      Math.abs(curr - sliderValue) < Math.abs(prev - sliderValue) ? curr : prev
    );

    // Update the slider value to the nearest epoch
    epochSlider.value = nearestEpoch;
    epochValue.textContent = nearestEpoch;

    // Zero-pad nearestEpoch to 6 digits
    const paddedEpoch = nearestEpoch.toString().padStart(6, '0');

    // Update the fileName span
    fileName.textContent = constructFileName(paddedEpoch, document.querySelector('input[name="set"]:checked').value, frameIdInput.value);
    // Load the JSON file
    loadJSON(expSelect.value + "/" + fileName.textContent);
    // console.log("v  epochSlider dmpInverseIndex", dmpInverseIndex);
  });

  dpmSlider.addEventListener('input', () => {
    dpmValue.textContent = dpmSlider.value;

    // Find the nearest value in sortedDPMStep
    // console.log("currentData.steps", currentData.steps);
    var sortedDPMStep = currentData.steps.slice().sort((a, b) => a - b);
    // console.log("currentData.steps", currentData.steps);
    const sliderValue = parseInt(dpmSlider.value, 10);
    const nearestDPM = sortedDPMStep.reduce((prev, curr) =>
      Math.abs(curr - sliderValue) < Math.abs(prev - sliderValue) ? curr : prev
    );
    dpmSlider.value = nearestDPM;
    dpmValue.textContent = nearestDPM;
  });
  function makeInverseIndex(steps) {
    const inverseIndex = {};
    steps.forEach((step, index) => {
      inverseIndex[step] = index;
    });
    return inverseIndex;
  }

  // Cache for loaded JSON data from files
  const jsonCache = {};
  let currentData = null;
  function updateUI() {
    const dpmTicks = document.getElementById("dpmTicks");
    dpmTicks.innerHTML = "";
    var sortedDPMStep = currentData.steps.slice().sort((a, b) => a - b);
    // console.log("sortedDPMStep", sortedDPMStep);
    // Add each step to the datalist
    sortedDPMStep.forEach(step => {
      const option = document.createElement("option");
      option.value = step;
      option.label = step;
      dpmTicks.appendChild(option);
    });
    // Set the dpmSlider to the first value
    dpmSlider.min = sortedDPMStep[0];
    dpmSlider.max = sortedDPMStep[sortedDPMStep.length - 1];
    dpmSlider.value = sortedDPMStep[0];
    dpmValue.textContent = sortedDPMStep[0];
    // Update the fileName span
  }
  function loadJSON(filename) {
    // Adjust base path as needed.
    const url = data_server + filename;
    if (jsonCache[filename]) {
      currentData = jsonCache[filename];
      updateUI()
      // console.log("Loaded from cache:", filename);
    } else {
      fetch(url)
        .then(response => response.json())
        .then(data => {
          jsonCache[filename] = data;
          currentData = data;
          // console.log("Fetched:", filename);
          updateUI()
        })
        .catch(err => console.error("Error loading JSON:", err));
    }
    //make inverse index lookup for currentData.steps
    // dmpInverseIndex = makeInverseIndex(currentData.steps);
    // console.log("dmpInverseIndex", dmpInverseIndex);
    //setup dpmTicks

  }

  // --- EXPERIMENT and FILE DROPDOWNS ---
  // Populate experiment select by fetching the directory listing from http://127.0.0.1:5000/
  fetch(data_server, { mode: 'no-cors' })
    .then(response => response.text())
    .then(htmlText => {
      const parser = new DOMParser();
      const doc = parser.parseFromString(htmlText, "text/html");
      const anchorNodes = doc.querySelectorAll("ul li a");
      anchorNodes.forEach(anchor => {
        const exp = anchor.textContent.trim();
        const option = document.createElement("option");
        option.value = exp;
        option.textContent = exp;
        expSelect.appendChild(option);
      });
      // Optionally trigger a change event on expSelect if you want to load files immediately.
      expSelect.dispatchEvent(new Event("change"));
    })
    .catch(err => console.error("Error fetching experiments", err));

  // When an experiment is selected, fetch its directory and build fileSelect options.// When an experiment is selected, fetch its directory and build fileSelect options.
  expSelect.addEventListener("change", () => {
    // Clear the fileSelect dropdown and reset UI collections.
    fileName.innerHTML = "";
    const selectedExp = expSelect.value;
    const url = data_server + `${selectedExp}`;
    fetch(url)
      .then(response => response.text())
      .then(htmlText => {
        const parser = new DOMParser();
        const doc = parser.parseFromString(htmlText, "text/html");
        const anchorNodes = doc.querySelectorAll("ul li a");

        // Collections to deduplicate values.
        const availableSets = new Set();
        const epochs = new Set();
        const frameIds = new Set();

        anchorNodes.forEach(anchor => {
          const fileName = anchor.textContent.trim();
          // Only consider JSON files.
          if (fileName.endsWith('.json')) {

            // Regex explanation:
            // - Group 1: epoch number (e.g. "000999")
            // - Group 2: the set ("train" or "val")
            // - Group 3: exactly 3 characters immediately before ".json" (e.g. "32d")
            const match = fileName.match(/sample_ep_(\d+)_gt\d+_(train|val)-idx-(.{3})\.json/);
            if (match) {
              const epoch = parseInt(match[1], 10);
              const setStr = match[2];
              const frameId = match[3]; // exactly 3 characters (e.g. "32d")
              epochs.add(epoch);
              availableSets.add(setStr);
              frameIds.add(frameId);
            }
          }
        });
        console.log("Available sets:", availableSets);
        console.log("Epochs:", epochs);
        console.log("Frame IDs:", frameIds);
        //availableSets.has("val")
        console.log("Available sets has val:", availableSets.has("val"));

        // Update the epoch slider and datalist using the epochs Set.
        if (epochs.size > 0) {
          const epochArr = Array.from(epochs);
          const minEpoch = Math.min(...epochArr);
          const maxEpoch = Math.max(...epochArr);
          epochSlider.min = minEpoch;
          epochSlider.max = maxEpoch;
          epochSlider.value = maxEpoch;
          epochValue.textContent = maxEpoch;

          // Update sortedEpochs
          sortedEpochs = epochArr.sort((a, b) => a - b);

          // Clear and update the datalist with each unique epoch found.
          const tickDatalist = document.getElementById("epochTicks");
          tickDatalist.innerHTML = "";
          sortedEpochs.forEach(epoch => {
            const option = document.createElement("option");
            option.value = epoch;
            option.label = epoch;
            tickDatalist.appendChild(option);
          });
        }

        // Update the frameId input.
        // Since these are three-character strings, you might want to show them as radio buttons.
        // For simplicity, here we just pick the first value.
        if (frameIds.size > 0) {
          const sortedFrameIds = Array.from(frameIds).sort(); // lexicographic sort
          //update frameId input with drop down
          // Clear existing options
          frameId.innerHTML = "";
          sortedFrameIds.forEach(fid => {
            const option = document.createElement("option");
            option.value = fid;
            option.textContent = fid;
            frameId.appendChild(option);

          });
        }

        // Update the set radio buttons.
        // For example, enable/disable the radio buttons if a set is missing.
        const trainRadio = document.getElementById("trainRadio");
        const valRadio = document.getElementById("valRadio");
        if (!availableSets.has("train")) {
          trainRadio.disabled = true;
        } else {
          trainRadio.disabled = false;
        }
        if (!availableSets.has("val")) {
          valRadio.disabled = true;
        } else {
          valRadio.disabled = false;
        }

        //construct first file name from first elements in epoch, set, frameId
        // epoch is 0-paddde 6 digits
        const firstEpoch = sortedEpochs[0].toString().padStart(6, '0');
        fist_file_name = `sample_ep_${firstEpoch}_gt0_${availableSets.values().next().value}-idx-${frameIds.values().next().value}.json`;
        fileName.textContent = fist_file_name;
        loadJSON(selectedExp + "/" + fist_file_name);
      })
      .catch(err => console.error("Error fetching file list for experiment", err));
  });

  // Helper function to update radio buttons
  function updateRadioButtons(containerId, values, name) {
    const container = document.getElementById(containerId);
    container.innerHTML = ""; // Clear existing buttons
    values.forEach(value => {
      const label = document.createElement("label");
      const radio = document.createElement("input");
      radio.type = "radio";
      radio.name = name;
      radio.value = value;
      label.appendChild(radio);
      label.appendChild(document.createTextNode(value));
      container.appendChild(label);
    });
  }

  // Helper function to update slider
  function updateSlider(sliderId, values) {
    const slider = document.getElementById(sliderId);
    const min = Math.min(...values);
    const max = Math.max(...values);
    slider.min = min;
    slider.max = max;
    slider.value = min; // Set to minimum by default
    const valueDisplay = document.getElementById(sliderId + "Value");
    valueDisplay.textContent = min; // Update display
    slider.addEventListener("input", () => {
      valueDisplay.textContent = slider.value;
    });
  }


  // --- WebGL Setup for Rendering Points (existing shader and render code) ---
  const vertexShaderSource = `
  attribute vec3 aPosition;
  uniform mat4 uModelViewMatrix;
  uniform mat4 uProjectionMatrix;
  void main(void) {
    gl_PointSize = 5.0;
    gl_Position = uProjectionMatrix * uModelViewMatrix * vec4(aPosition, 1.0);
  }
`;
const fragmentShaderSource = `
  precision mediump float;
  uniform vec4 uColor;
  void main(void) {
    gl_FragColor = uColor;
  }
`;

  function createShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.error("Shader error:", gl.getShaderInfoLog(shader));
      gl.deleteShader(shader);
      return null;
    }
    return shader;
  }
  const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
  const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);

  const shaderProgram = gl.createProgram();
  gl.attachShader(shaderProgram, vertexShader);
  gl.attachShader(shaderProgram, fragmentShader);
  gl.linkProgram(shaderProgram);
  if (!gl.getProgramParameter(shaderProgram, gl.LINK_STATUS)) {
    console.error("Unable to initialize the shader program:", gl.getProgramInfoLog(shaderProgram));
  }

  const programInfo = {
    program: shaderProgram,
    attribLocations: {
      aPosition: gl.getAttribLocation(shaderProgram, "aPosition")
    },
    uniformLocations: {
      uProjectionMatrix: gl.getUniformLocation(shaderProgram, "uProjectionMatrix"),
      uModelViewMatrix: gl.getUniformLocation(shaderProgram, "uModelViewMatrix"),
      uColor: gl.getUniformLocation(shaderProgram, "uColor")
    }
  };

  const pointBuffer = gl.createBuffer();

  // --- CAMERA/ORBIT CONTROLS ---
  let cameraAngleX = 0, cameraAngleY = 0;
  let cameraDistance = 10;
  let cameraPanX = 0, cameraPanY = 0;
  let lastMouseX, lastMouseY;
  let isDragging = false;

  canvas.addEventListener('mousedown', (e) => {
    isDragging = true;
    lastMouseX = e.clientX;
    lastMouseY = e.clientY;
  });
  window.addEventListener('mouseup', () => { isDragging = false; });
  canvas.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    const dx = e.clientX - lastMouseX;
    const dy = e.clientY - lastMouseY;
    // Left-button drag: orbit
    cameraAngleY += dx * 0.5;
    cameraAngleX += dy * 0.5;
    lastMouseX = e.clientX;
    lastMouseY = e.clientY;
  });
  canvas.addEventListener('wheel', (e) => {
    cameraDistance += e.deltaY * 0.01;
    cameraDistance = Math.max(2, cameraDistance); // Minimum distance
    cameraDistance = Math.min(cameraDistance, 200); // Maximum distance to prevent zooming too far out
  });

  // --- SIMPLE MATRIX UTILITIES ---
  const mat4 = {
    create: function () {
      return new Float32Array([1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1]);
    },
    perspective: function (out, fovy, aspect, near, far) {
      const f = 1.0 / Math.tan(fovy / 2);
      out[0] = f / aspect;
      out[1] = 0;
      out[2] = 0;
      out[3] = 0;
      out[4] = 0;
      out[5] = f;
      out[6] = 0;
      out[7] = 0;
      out[8] = 0;
      out[9] = 0;
      out[10] = (far + near) / (near - far);
      out[11] = -1;
      out[12] = 0;
      out[13] = 0;
      out[14] = (2 * far * near) / (near - far);
      out[15] = 0;
      return out;
    },
    identity: function (out) {
      out[0] = 1; out[1] = 0; out[2] = 0; out[3] = 0;
      out[4] = 0; out[5] = 1; out[6] = 0; out[7] = 0;
      out[8] = 0; out[9] = 0; out[10] = 1; out[11] = 0;
      out[12] = 0; out[13] = 0; out[14] = 0; out[15] = 1;
      return out;
    },
    translate: function (out, a, v) {
      const x = v[0], y = v[1], z = v[2];
      if (a === out) {
        out[12] = a[0] * x + a[4] * y + a[8] * z + a[12];
        out[13] = a[1] * x + a[5] * y + a[9] * z + a[13];
        out[14] = a[2] * x + a[6] * y + a[10] * z + a[14];
      } else {
        const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
        const a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
        const a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
        out[0] = a00; out[1] = a01; out[2] = a02; out[3] = a03;
        out[4] = a10; out[5] = a11; out[6] = a12; out[7] = a13;
        out[8] = a20; out[9] = a21; out[10] = a22; out[11] = a23;
        out[12] = a00 * x + a10 * y + a20 * z + a[12];
        out[13] = a01 * x + a11 * y + a21 * z + a[13];
        out[14] = a02 * x + a12 * y + a22 * z + a[14];
        out[15] = a03 * x + a13 * y + a23 * z + a[15];
      }
      return out;
    },
    rotate: function (out, a, rad, axis) {
      let x = axis[0], y = axis[1], z = axis[2];
      let len = Math.hypot(x, y, z);
      if (len < 0.000001) return null;
      len = 1 / len;
      x *= len; y *= len; z *= len;
      const s = Math.sin(rad);
      const c = Math.cos(rad);
      const t = 1 - c;
      const a00 = a[0], a01 = a[1], a02 = a[2], a03 = a[3];
      const a10 = a[4], a11 = a[5], a12 = a[6], a13 = a[7];
      const a20 = a[8], a21 = a[9], a22 = a[10], a23 = a[11];
      const b00 = x * x * t + c, b01 = y * x * t + z * s, b02 = z * x * t - y * s;
      const b10 = x * y * t - z * s, b11 = y * y * t + c, b12 = z * y * t + x * s;
      const b20 = x * z * t + y * s, b21 = y * z * t - x * s, b22 = z * z * t + c;
      out[0] = a00 * b00 + a10 * b01 + a20 * b02;
      out[1] = a01 * b00 + a11 * b01 + a21 * b02;
      out[2] = a02 * b00 + a12 * b01 + a22 * b02;
      out[3] = a03 * b00 + a13 * b01 + a23 * b02;
      out[4] = a00 * b10 + a10 * b11 + a20 * b12;
      out[5] = a01 * b10 + a11 * b11 + a21 * b12;
      out[6] = a02 * b10 + a12 * b11 + a22 * b12;
      out[7] = a03 * b10 + a13 * b11 + a23 * b12;
      out[8] = a00 * b20 + a10 * b21 + a20 * b22;
      out[9] = a01 * b20 + a11 * b21 + a21 * b22;
      out[10] = a02 * b20 + a12 * b21 + a22 * b22;
      out[11] = a03 * b20 + a13 * b21 + a23 * b22;
      out[12] = a[12];
      out[13] = a[13];
      out[14] = a[14];
      out[15] = a[15];
      return out;
    }
  };
  
  function renderAxes() {
    if (fontTextureLoaded) {
      renderAxisLabels();
    }
    const axisVertices = [
      // X-axis (red)
      0, 0, 0, 10, 0, 0,
      // Y-axis (green)
      0, 0, 0, 0, 10, 0,
      // Z-axis (blue)
      0, 0, 0, 0, 0, 10
    ];
  
    const axisColors = [
      [1.0, 0.0, 0.0, 1.0], // Red for X-axis
      [0.0, 1.0, 0.0, 1.0], // Green for Y-axis
      [0.0, 0.0, 1.0, 1.0]  // Blue for Z-axis
    ];
  
    // Create and bind the buffer for axis vertices
    const axisBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, axisBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(axisVertices), gl.STATIC_DRAW);
  
    // Draw each axis separately
    for (let i = 0; i < 3; i++) {
      gl.vertexAttribPointer(programInfo.attribLocations.aPosition, 3, gl.FLOAT, false, 0, i * 2 * 3 * Float32Array.BYTES_PER_ELEMENT);
      gl.enableVertexAttribArray(programInfo.attribLocations.aPosition);
      gl.uniform4fv(programInfo.uniformLocations.uColor, axisColors[i]);
      gl.drawArrays(gl.LINES, 0, 2); // Draw 2 vertices per axis
    }
  }
  var fontInfo = { //singularity_logs/plots/mlp_blob/vizr/8x8-font.png
    letterHeight: 8,
    spaceWidth: 8,
    spacing: -1,
    textureWidth: 64,
    textureHeight: 40,
    glyphInfos: {
      'a': { x:  0, y:  0, width: 8, },
      'b': { x:  8, y:  0, width: 8, },
      'c': { x: 16, y:  0, width: 8, },
      'd': { x: 24, y:  0, width: 8, },
      'e': { x: 32, y:  0, width: 8, },
      'f': { x: 40, y:  0, width: 8, },
      'g': { x: 48, y:  0, width: 8, },
      'h': { x: 56, y:  0, width: 8, },
      'i': { x:  0, y:  8, width: 8, },
      'j': { x:  8, y:  8, width: 8, },
      'k': { x: 16, y:  8, width: 8, },
      'l': { x: 24, y:  8, width: 8, },
      'm': { x: 32, y:  8, width: 8, },
      'n': { x: 40, y:  8, width: 8, },
      'o': { x: 48, y:  8, width: 8, },
      'p': { x: 56, y:  8, width: 8, },
      'q': { x:  0, y: 16, width: 8, },
      'r': { x:  8, y: 16, width: 8, },
      's': { x: 16, y: 16, width: 8, },
      't': { x: 24, y: 16, width: 8, },
      'u': { x: 32, y: 16, width: 8, },
      'v': { x: 40, y: 16, width: 8, },
      'w': { x: 48, y: 16, width: 8, },
      'x': { x: 56, y: 16, width: 8, },
      'y': { x:  0, y: 24, width: 8, },
      'z': { x:  8, y: 24, width: 8, },
      '0': { x: 16, y: 24, width: 8, },
      '1': { x: 24, y: 24, width: 8, },
      '2': { x: 32, y: 24, width: 8, },
      '3': { x: 40, y: 24, width: 8, },
      '4': { x: 48, y: 24, width: 8, },
      '5': { x: 56, y: 24, width: 8, },
      '6': { x:  0, y: 32, width: 8, },
      '7': { x:  8, y: 32, width: 8, },
      '8': { x: 16, y: 32, width: 8, },
      '9': { x: 24, y: 32, width: 8, },
      '-': { x: 32, y: 32, width: 8, },
      '*': { x: 40, y: 32, width: 8, },
      '!': { x: 48, y: 32, width: 8, },
      '?': { x: 56, y: 32, width: 8, },
    },
  };
  function makeVerticesForString(fontInfo, s) {
    const len = s.length;
    const numVertices = len * 6;
    const positions = new Float32Array(numVertices * 2);
    const texcoords = new Float32Array(numVertices * 2);
    let offset = 0;
    let x = 0;
    const maxX = fontInfo.textureWidth;
    const maxY = fontInfo.textureHeight;
  
    for (let i = 0; i < len; ++i) {
      const letter = s[i];
      const glyphInfo = fontInfo.glyphInfos[letter];
      if (glyphInfo) {
        // console.log("glyphInfo", glyphInfo);
        const x2 = x + glyphInfo.width;
        const u1 = glyphInfo.x / maxX;
        const v1 = (glyphInfo.y + fontInfo.letterHeight) / maxY;
        const u2 = (glyphInfo.x + glyphInfo.width) / maxX;
        const v2 = glyphInfo.y / maxY;
  
        // 6 vertices per letter
        positions.set([x, 0, x2, 0, x, fontInfo.letterHeight, x, fontInfo.letterHeight, x2, 0, x2, fontInfo.letterHeight], offset);
        texcoords.set([u1, v1, u2, v1, u1, v2, u1, v2, u2, v1, u2, v2], offset);
  
        x += glyphInfo.width + fontInfo.spacing;
        offset += 12;
      } else {
        // console.warn(`Glyph for character "${letter}" not found.`);
        x += fontInfo.spaceWidth;
      }
    }
  
    return {
      arrays: {
        position: new Float32Array(positions.buffer, 0, offset),
        texcoord: new Float32Array(texcoords.buffer, 0, offset),
      },
      numVertices: offset / 2,
    };
  }
  let fontTextureLoaded = false; // Flag to track if the texture is loaded

  const fontTexture = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, fontTexture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 255, 255])); // Placeholder

  const fontImage = new Image();
  fontImage.src = '8x8-font.png'; // Replace with the actual path to your font texture
  fontImage.onload = () => {
    console.log("8x8-font.png Font image loaded");
    gl.bindTexture(gl.TEXTURE_2D, fontTexture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, fontImage);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

    fontTextureLoaded = true; // Set the flag to true
  };
  function renderTextWithGlyphs(text, position) {
  }
  function renderAxisLabels() {
    renderTextWithGlyphs('x', [10, 0, 0]); // X-axis
  }
  function reshapeTo3D(array) {
    const reshaped = [];
    for (let i = 0; i < array.length; i += 3) {
      reshaped.push(array.slice(i, i + 3));
    }
    return reshaped;
  }

  function calculateCD(xs, gt) {
    // reshape xs and gt to 3D arrays. from 1D aray of length 3N to N x 3

    // Calculate the Chamfer Distance between two sets of points
    let totalDistance = 0;
    for (let i = 0; i < xs.length; i++) {
      const x = xs[i];
      const g = gt[i];
      const distance = Math.sqrt(Math.pow(x[0] - g[0], 2) + Math.pow(x[1] - g[1], 2) + Math.pow(x[2] - g[2], 2));
      totalDistance += distance;
    }
    const cd = totalDistance / xs.length;
    return cd;
  }
  // --- RENDER LOOP ---
  function render() {
    // Clear with a sky-blue background
    gl.clearColor(0.53, 0.81, 0.92, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);

    const aspect = canvas.width / canvas.height;
    const projectionMatrix = mat4.create();
    mat4.perspective(projectionMatrix, 60 * Math.PI / 180, aspect, 0.1, 500.0);

    const modelViewMatrix = mat4.create();
    mat4.identity(modelViewMatrix);

    // Apply camera transformations: translate, then orbit rotations and pan
    mat4.translate(modelViewMatrix, modelViewMatrix, [0, 0, -cameraDistance]);
    mat4.rotate(modelViewMatrix, modelViewMatrix, cameraAngleX * Math.PI / 180, [1, 0, 0]);
    mat4.rotate(modelViewMatrix, modelViewMatrix, cameraAngleY * Math.PI / 180, [0, 1, 0]);
    mat4.translate(modelViewMatrix, modelViewMatrix, [cameraPanX, cameraPanY, 0]);

    gl.useProgram(programInfo.program);
    gl.uniformMatrix4fv(programInfo.uniformLocations.uProjectionMatrix, false, projectionMatrix);
    gl.uniformMatrix4fv(programInfo.uniformLocations.uModelViewMatrix, false, modelViewMatrix);

    // Render axes
    renderAxes();

    // If JSON data is loaded, extract and draw points from GT and xs arrays.
    if (currentData) {
      // Disable aTexCoord for points
      // const texcoordLocation = gl.getAttribLocation(shaderProgram, "aTexCoord");
      // if (texcoordLocation !== -1) {
      //   gl.disableVertexAttribArray(texcoordLocation);
      // }

      // Draw GT points in red
      const gtPoints = currentData.gt[0]; // assume first set
      //if unnormRadio is True --> unnorm the poitn frist
      // console.log("unnormed", document.getElementById("unnormRadio").checked);
      // console.log("std", currentData.data_std);
      // console.log("mean", currentData.data_mean);

      // "data_mean": [
      //   78.09962868690491,
      //   11.083830393850803,
      //   0.10256138234399259
      // ],
      // "data_std": [
      //     34.22457043749478,
      //     27.583163497404275,
      //     1.6358030562888937
      // ],
      let gtArray = [];
      gtPoints.forEach(pt => {
        if (document.getElementById("unnormRadio").checked)
          gtArray.push(pt[0] * currentData.data_std[0] + currentData.data_mean[0], pt[1] * currentData.data_std[1] + currentData.data_mean[1], pt[2] * currentData.data_std[2] + currentData.data_mean[2]);
        else gtArray.push(pt[0], pt[1], pt[2])
      }
      );
      //space-padded 2 significant digit, with comma
      cdValue.textContent = currentData.cd.toFixed(2).padStart(6, ' ');
      gl.bindBuffer(gl.ARRAY_BUFFER, pointBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(gtArray), gl.STATIC_DRAW);
      gl.vertexAttribPointer(programInfo.attribLocations.aPosition, 3, gl.FLOAT, false, 0, 0);
      gl.enableVertexAttribArray(programInfo.attribLocations.aPosition);
      gl.uniform4fv(programInfo.uniformLocations.uColor, [1.0, 0.0, 0.0, 1.0]); // Set color to red
      gl.drawArrays(gl.POINTS, 0, gtArray.length / 3);

      // Draw xs points in blue (if available)
      if (currentData.xts && currentData.xts.length > 0) {
        //get dpmSlider value
        const sliderValue = parseInt(dpmSlider.value);
        var sortedDPMStep = currentData.steps.slice().sort((a, b) => a - b);
        const nearestDPM = sortedDPMStep.reduce((prev, curr) =>
          Math.abs(curr - sliderValue) < Math.abs(prev - sliderValue) ? curr : prev
        );
        // console.log("currentData.steps", currentData.steps);
        // console.log("sortedDPMStep", sortedDPMStep);  
        // console.log("nearestDPM", nearestDPM);
        const dpmIndex = currentData.steps.indexOf(nearestDPM);
        // console.log("dpmValue",nearestDPM,"dpmIndex", dpmIndex);
        // //last element of xts

        const xsPoints = currentData.xts[dpmIndex];
        // const xsPoints = currentData.xts[currentData.xts.length - 1];
        // console.log( "xsPoints", xsPoints);
        let xsArray = [];

        xsPoints.forEach(pt => {
          if (document.getElementById("unnormRadio").checked)
            xsArray.push(pt[0] * currentData.data_std[0] + currentData.data_mean[0], pt[1] * currentData.data_std[1] + currentData.data_mean[1], pt[2] * currentData.data_std[2] + currentData.data_mean[2]);
          else xsArray.push(pt[0], pt[1], pt[2])
        }
        );
        // console.log("xsPoints", xsPoints);
        // console.log("xsArray", xsArray);


        calculatedCD = calculateCD(reshapeTo3D(xsArray), reshapeTo3D(gtArray));
        //update dpmCdValue
        dpmCdValue.textContent = calculatedCD.toFixed(2);
        gl.bindBuffer(gl.ARRAY_BUFFER, pointBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(xsArray), gl.STATIC_DRAW);
        gl.vertexAttribPointer(programInfo.attribLocations.aPosition, 3, gl.FLOAT, false, 0, 0);
        gl.enableVertexAttribArray(programInfo.attribLocations.aPosition);
        gl.uniform4fv(programInfo.uniformLocations.uColor, [0.0, 0.0, 1.0, 1.0]); // Set color to blue
        gl.drawArrays(gl.POINTS, 0, xsArray.length / 3);
      }
    }
    requestAnimationFrame(render);
  }

  render();
})();