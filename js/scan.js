let model;

async function loadModel() {
    model = await tf.loadLayersModel('model/mohamedbenhasan_ResNet50_model.json');  // Load the model
    console.log("Model loaded successfully");
}

async function startScan() {
    const fileInput = document.getElementById('imageUpload');
    const file = fileInput.files[0];
    
    if (!file) {
        alert("Please upload an image");
        return;
    }

    const image = await loadImage(file);
    const tensor = preprocessImage(image);

    const prediction = await model.predict(tensor).data();
    displayResult(prediction);
}

function loadImage(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
            const img = new Image();
            img.src = reader.result;
            img.onload = () => resolve(img);
        };
        reader.onerror = reject;
        reader.readAsDataURL(file);
    });
}

function preprocessImage(image) {
    const tensor = tf.browser.fromPixels(image)
        .resizeNearestNeighbor([224, 224])  // Adjust this based on your model input size
        .toFloat()
        .expandDims();
    return tensor;
}

function displayResult(prediction) {
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = `Prediction: ${prediction}`;
}

// Load the model when the page loads
loadModel();
