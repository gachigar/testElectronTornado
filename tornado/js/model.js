const netSize = 20;
const colours = 3;
const latentDim = 12;
//                x   y   r
const inputSize = 1 + 1 + 1 + latentDim;

let audioCtx = new AudioContext();
let destination = audioCtx.destination;
let analyser1 = audioCtx.createAnalyser();
let analyser2 = audioCtx.createAnalyser();
let analyser3 = audioCtx.createAnalyser();
//let analyserPCM = audioCtx.createAnalyser();
let gainNode = audioCtx.createGain();
gainNode.gain.setValueAtTime(0.5, audioCtx.currentTime);
let pl1 = document.querySelector('audio');
let source = audioCtx.createMediaElementSource(pl1);
source.connect(analyser1);
source.connect(analyser2);
source.connect(analyser3);
//source.connect(analyserPCM);
analyser1.connect(gainNode);
gainNode.connect(destination);
analyser1.fftSize = 8192;
analyser2.fftSize = 4096;
analyser3.fftSize = 2048;
//analyserPCM.fftSize = 4096;
analyser1.smoothingTimeConstant = 0;
analyser1.minDecibels = -100;
analyser1.maxDecibels = 0;

analyser1.smoothingTimeConstant = 0.1;
analyser1.minDecibels = -100;
analyser1.maxDecibels = 0;

analyser1.smoothingTimeConstant = 0.2;
analyser1.minDecibels = -100;
analyser1.maxDecibels = 0;

let frequencyData1 = new Uint8Array(analyser1.frequencyBinCount);
let frequencyData2 = new Uint8Array(analyser2.frequencyBinCount);
let frequencyData3 = new Uint8Array(analyser3.frequencyBinCount);
let frequencyDataFinal = new Uint8Array(12);
//let PCMData = new Float32Array(analyserPCM.frequencyBinCount);

let icanvas = document.getElementById('infa');
let ictx = icanvas.getContext('2d');

/*function averagePoolFFT(fftarray) {
    for (let i = 0; i < fftarray.length)
}*/

function touchStarted() {
  audioCtx.resume();
  console.log("S!");
}

function buildModel(numDense, activationFunction, canvas) {
    const model = tf.sequential();
    const init = tf.initializers.randomNormal({mean: 0, stddev: 1});

    model.add(tf.layers.conv2d(
        {
            inputShape: [canvas.width, canvas.height, inputSize],
            kernelSize: 3,
            filters: 12,//12
            strides: 1,
            padding: "same",
            activation: activationFunction,
            kernelInitializer: init,
            biasIntializer: init,
            useBias: true
        }));

    for (k = 0; k < numDense; k++) {
        model.add(tf.layers.conv2d(
            {
                kernelSize: 3,
                filters: 12,//12
                strides: 1,
                padding: "same",
                activation: activationFunction,
                kernelInitializer: init,
                biasIntializer: init,
                useBias: true
            }
        ));
    }

    model.add(tf.layers.conv2d({
        kernelSize: 1,
        filters: colours,
        strides: 1,
        activation: 'sigmoid',
        kernelInitializer: init,
        useBias: true
    }));

    return model;
}

function getInputTensor(imageWidth, imageHeight, inputSizeExcludingLatent) {
    // NOTE: Height probably has to equal width
    const coords = new Float32Array(imageWidth * imageHeight * inputSizeExcludingLatent);
    let dst = 0;

    for (let i = 0; i < imageWidth * imageHeight; i++) {

        const x = i % imageWidth;
        const y = Math.floor(i / imageWidth);
        const coord = imagePixelToNormalisedCoord(x, y, imageWidth, imageHeight);

        for (let d = 0; d < inputSizeExcludingLatent; d++) {
            coords[dst++] = coord[d];
        }
    }

    return tf.tensor2d(coords, [imageWidth * imageHeight, inputSizeExcludingLatent]);
}


function imagePixelToNormalisedCoord(x, y, imageWidth, imageHeight) {
    const normX = (x - (imageWidth / 2)) / imageWidth;
    const normY = (y - (imageHeight / 2)) / imageHeight;

    // TODO: Make the norm configurable
    const r = Math.sqrt(normX * normX + normY * normY);

    const result = [normX, normY, r];

    return result;
}


async function runInferenceLoop(canvas, model, z1, z2, currentStep) {

    const steps = 1000;
    const inputSizeExcludingLatent = inputSize - latentDim;


    tf.tidy(() => {
        const t = currentStep / steps;

        // Work out the new z:
        // z = z_1 * (1-t) + t * z_2
        const a = tf.mul(z1, tf.scalar(1 - t)).mul(tf.scalar(0.12));
        const b = tf.mul(z2, tf.scalar(t)).mul(tf.scalar(0.12));

        analyser1.getByteFrequencyData(frequencyData1);
        analyser2.getByteFrequencyData(frequencyData2);
        analyser3.getByteFrequencyData(frequencyData3);
        for (let i = 0; i < 4; i++) {
            let a = 0;
            for (let j = 0; j < 8; j++) {
                if (frequencyData1[2+i*8+j] > 96) {
                    a += Math.pow(frequencyData1[2+i*8+j] - 96, 1.2);
                }
            }
            frequencyDataFinal[i] = a/8;
        }
        for (let i = 0; i < 4; i++) {
            let a = 0;
            for (let j = 0; j < 64; j++) {
                if (frequencyData2[16+i*64+j] > 72) {
                    a += Math.pow(frequencyData2[16+i*64+j] - 72, 1.05);
                }
            }
            frequencyDataFinal[i+4] = a/64;
        }
        for (let i = 0; i < 4; i++) {
            let a = 0;
            for (let j = 0; j < 128; j++) {
                if (frequencyData3[136+i*128+j] > 64) {
                    a +=  Math.pow(frequencyData3[136+i*128+j] - 64, 1.1);
                }
            }
            frequencyDataFinal[i+8] = a/128;
        }

        const c = tf.tensor(frequencyDataFinal.slice(0, latentDim));

        const z = tf.add(tf.add(a, b), c.mul(tf.scalar(0.0008)));

        let xs = getInputTensor(canvas.width, canvas.height, inputSizeExcludingLatent);

        const ones = tf.ones([xs.shape[0], 1]);
        const axis = 1;
        xs = tf.concat([xs, tf.mul(z, ones)], axis).reshape([1, canvas.width, canvas.height, inputSize]);

        ys = model.predict(xs);
        renderToCanvas(ys, canvas);
    });


    if (currentStep == steps) {
        currentStep = -1; // So that +1 takes us to 0.
        z1 = z2; // Start where we ended up
        z2 = tf.randomNormal([latentDim], 0, 1);
    }

    await tf.nextFrame();
    runInferenceLoop(canvas, model, z1, z2, currentStep + 1);
}


async function animateCppn(canvasId) {
    const canvas = document.getElementById(canvasId);
    const layers = 2;
    const model = buildModel(layers, "tanh", canvas);

    const z1 = tf.randomNormal([latentDim], 0, 1);
    const z2 = tf.randomNormal([latentDim], 0, 1);

    runInferenceLoop(canvas, model, z1, z2, 0);
}


function renderToCanvas(a, canvas) {
    const height = canvas.height;
    const width = canvas.width;
    const ctx = canvas.getContext('2d');
    const imageData = new ImageData(width, height);
    const data = a.dataSync();


    for (let i = 0; i < height * width; ++i) {
        const j = i * 4;
        const k = i * 3;

        imageData.data[j] = Math.round(255 * data[k]);
        imageData.data[j + 1] = 0;
        imageData.data[j + 2] = Math.round(255 * data[k + 2]);
        imageData.data[j + 3] = 255 - Math.round(255 * data[k + 1]);
    }

    ictx.clearRect(0, 0, icanvas.width, icanvas.height);

    for (let i = 0; i < frequencyDataFinal.length; i++) {
        ictx.beginPath();
        ictx.moveTo(100 + i*10, 500);
        ictx.lineTo(100 + i*10, 500 - frequencyDataFinal[i] / 2);
        ictx.stroke();
        ictx.closePath();
    }
    ctx.putImageData(imageData, 0, 0);
}
