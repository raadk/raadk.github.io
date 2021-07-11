// https://github.com/tensorflow/tfjs-examples/blob/master/mobilenet


// const MOBILENET_MODEL_PATH =
//     // tslint:disable-next-line:max-line-length
//     'https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v2_100_224/classification/3/default/1';

const MOBILENET_MODEL_PATH = "http://46.101.78.204/like_model/model.json"

const IMAGE_SIZE = 224;

let model;
const loadModel = async () => {
    status('Loading model...');

    // model = await tf.loadGraphModel(MOBILENET_MODEL_PATH, {fromTFHub: true});
    // model = await tf.loadLayersModel(MOBILENET_MODEL_PATH);
    console.log('layers')
    model = await tf.loadLayersModel(MOBILENET_MODEL_PATH);
    status('Loaded model');

    // Warmup the model. This isn't necessary, but makes the first prediction
    // faster. Call `dispose` to release the WebGL memory allocated for the return
    // value of `predict`.
    model.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();

    document.getElementById('loading-text').style.display = 'none';
    document.getElementById('file-container').style.display = '';
};


/**
 * Given an image element, makes a prediction through mobilenet returning the
 * probabilities of the top K classes.
 */
async function predict(imgElement, imgPreviewElement) {
    status('Predicting...');

    // The first start time includes the time it takes to extract the image
    // from the HTML and preprocess it, in additon to the predict() call.
    const startTime1 = performance.now();
    // The second start time excludes the extraction and preprocessing and
    // includes only the predict() call.
    let startTime2;
    const logits = tf.tidy(() => {
        const img = tf.cast(tf.browser.fromPixels(imgElement), 'float32');

        const batched = img.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);
        /*
        const offset = tf.scalar(127.5);
        // Normalize the image from [0, 255] to [-1, 1].
        const normalized = img.sub(offset).div(offset);

        // Reshape to a single-element batch so we can pass it to predict.
        const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);
         */

        startTime2 = performance.now();
        // Make a prediction through mobilenet.
        return model.predict(batched);
    });

    // Convert logits to probabilities and class names.
    const pred = await getPredFromLogit(logits);

    const totalTime1 = performance.now() - startTime1;
    const totalTime2 = performance.now() - startTime2;
    status(`Done in ${Math.floor(totalTime1)} ms ` +
        `(not including preprocessing: ${Math.floor(totalTime2)} ms)`);

    // Show the classes in the DOM.
    showResults(imgElement, imgPreviewElement, pred * 100);
}

async function getPredFromLogit(logits) {
    const values = await logits.data();
    return values[0];
}

//
// UI
//

function createProgressBarHtml(proba) {
    return `
    <br>
    
    <div class="row">
        <div class="col-2 col-xl-3 ">
            <span class="text-right text-danger"><h4><i class="bi bi-heart"></i></h4></span>
        </div> 
    
        <div class="col-8 col-xl-6 ">
            <div class="progress" style="height: 30px;">
                <div class="progress-bar bg-danger" role="progressbar" style="width: ${proba.toFixed(0)}%;"
                     aria-valuenow="${proba.toFixed(0)}" 
                     aria-valuemin="0" aria-valuemax="100">
                    ${proba.toFixed(0)}%
                </div>
            </div>
            <!--progress-->
        </div>
        <!--col-->
        
        <div class="col-2 col-xl-3 ">
            <span class="text-left text-danger"><h4><i class="bi bi-heart-fill"></i></h4></span>
        </div> 
    </div>
    <br>
    <!--row-->
    `
}

function showResults(imgElement, imgPreviewElement, proba) {
    const predictionContainer = document.createElement('div');
    predictionContainer.className = 'pred-container';

    const imgContainer = document.createElement('div');
    imgContainer.className = 'col-12 uploaded-image'
    // imgContainer.appendChild(imgElement);
    imgContainer.appendChild(imgPreviewElement);
    predictionContainer.appendChild(imgContainer);

    const probsContainer = document.createElement('div');
    probsContainer.innerHTML = createProgressBarHtml(proba)

    probsContainer.className = 'col-12 probability-score'
    predictionContainer.appendChild(probsContainer);

    predictionsElement.insertBefore(
        predictionContainer, predictionsElement.firstChild);
}

const filesElement = document.getElementById('files');
filesElement.addEventListener('change', evt => {
    let files = evt.target.files;
    // Display thumbnails & issue call to predict each image.
    for (let i = 0, f; f = files[i]; i++) {
        // Only process image files (skip non image files)
        if (!f.type.match('image.*')) {
            continue;
        }
        let reader = new FileReader();
        reader.onload = e => {
            // Fill the image & call predict.
            let img_preview = document.createElement('img');
            img_preview.className = 'img-fluid'
            img_preview.src = e.target.result;
            img_preview.width = 660;
            img_preview.height = 660;

            let img = document.createElement('img');
            img.src = e.target.result;
            img.width = IMAGE_SIZE;
            img.height = IMAGE_SIZE;
            img.onload = () => predict(img, img_preview);
        };

        // Read in the image file as a data URL.
        reader.readAsDataURL(f);
    }
});

const demoStatusElement = document.getElementById('status');
// const status = msg => demoStatusElement.innerText = msg;
const status = msg => console.log(msg);

const predictionsElement = document.getElementById('predictions');

loadModel();