const fileInputContainer = document.getElementById('fileInputContainer');
const webcamContainer = document.getElementById('webcamContainer');
const fileInput = document.getElementById('fileInput');
const webcam = document.getElementById('webcam');
const captureCanvas = document.getElementById('captureCanvas');
const takePictureButton = document.getElementById('takePictureButton');
const predictButton = document.getElementById('predictButton');
const modelSelect = document.getElementById('modelSelect');
const loading = document.getElementById('loading');
const resultDiv = document.querySelector('.result');
const resultText = document.getElementById('resultText');
const switchToWebcam = document.getElementById('switchToWebcam');
const switchToFile = document.getElementById('switchToFile');

let selectedFile = null; // Store selected file or webcam snapshot
let webcamStream = null;

// Toggle input methods
switchToFile.addEventListener('click', () => {
    fileInputContainer.style.display = 'block';
    webcamContainer.style.display = 'none';
    selectedFile = null; // Reset selection
    stopWebcam();
});

switchToWebcam.addEventListener('click', () => {
    fileInputContainer.style.display = 'none';
    webcamContainer.style.display = 'block';
    selectedFile = null; // Reset selection
    startWebcam();
});

// Start the webcam
function startWebcam() {
    navigator.mediaDevices.getUserMedia({ video: { width: 240, height: 240 } })
        .then(stream => {
            webcamStream = stream;
            webcam.srcObject = stream;
            webcam.play();
        })
        .catch(error => console.error('Webcam error:', error));
}

// Stop the webcam
function stopWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
}

// Capture snapshot from webcam
takePictureButton.addEventListener('click', () => {
    const context = captureCanvas.getContext('2d');
    captureCanvas.width = 240;
    captureCanvas.height = 240;
    context.drawImage(webcam, 0, 0, 240, 240);
    captureCanvas.toBlob((blob) => {
        selectedFile = new File([blob], 'webcam_image.jpg', { type: 'image/jpeg' });
        resultText.textContent = 'Picture taken, ready to predict!';
        stopWebcam();
    }, 'image/jpeg');
});

// Predict
predictButton.addEventListener('click', async () => {
    const model = modelSelect.value;
    const formData = new FormData();

    if (selectedFile) {
        formData.append('file', selectedFile);
    } else if (fileInput.files.length > 0) {
        formData.append('file', fileInput.files[0]);
    } else {
        alert('Please select an image or capture one using the webcam!');
        return;
    }

    // Show loading spinner
    loading.style.display = 'block';
    resultDiv.style.display = 'none';

    try {
        const response = await fetch(`https://sharktide-tf-misc-model-server.hf.space/predict/${model}`, {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();
        loading.style.display = 'none';

        if (response.ok) {
            resultText.textContent = data.prediction;
            resultDiv.style.display = 'block';
        } else {
            resultText.textContent = `Error: ${data.error}`;
            resultDiv.style.display = 'block';
        }
    } catch (error) {
        loading.style.display = 'none';
        resultText.textContent = `Error: ${error.message}`;
        resultDiv.style.display = 'block';
    }
});