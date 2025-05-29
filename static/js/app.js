let mediaRecorder;
let audioChunks = [];
let recordingStartTime;
let timerInterval;
let currentTimestamp;

// DOM Elements
const recordButton = document.getElementById('recordButton');
const recordingTimer = document.getElementById('recordingTimer');
const transcriptionDiv = document.getElementById('transcription');
const refinedTranscription = document.getElementById('refinedTranscription');
const saveRefinedButton = document.getElementById('saveRefined');
const currentImage = document.getElementById('currentImage');

// Initialize
async function init() {
    // Load initial image
    const response = await fetch('/api/get-image');
    const data = await response.json();
    currentImage.src = `/static/images/${data.image}`;
    
    // Set up click handlers for image segmentation
    currentImage.addEventListener('click', handleImageClick);
}

// Recording functions
async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        
        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };
        
        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            await sendAudioForTranscription(audioBlob);
        };
        
        audioChunks = [];
        mediaRecorder.start();
        recordingStartTime = Date.now();
        updateRecordingTimer();
        
        recordButton.textContent = 'Stop Recording';
        recordButton.classList.add('recording');
        recordingTimer.classList.remove('hidden');
    } catch (err) {
        console.error('Error accessing microphone:', err);
        alert('Error accessing microphone. Please ensure you have granted microphone permissions.');
    }
}

function stopRecording() {
    mediaRecorder.stop();
    clearInterval(timerInterval);
    
    recordButton.textContent = 'Start Recording';
    recordButton.classList.remove('recording');
    recordingTimer.classList.add('hidden');
}

function updateRecordingTimer() {
    timerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - recordingStartTime) / 1000);
        const minutes = Math.floor(elapsed / 60).toString().padStart(2, '0');
        const seconds = (elapsed % 60).toString().padStart(2, '0');
        recordingTimer.textContent = `${minutes}:${seconds}`;
        
        // Stop recording after 90 seconds
        if (elapsed >= 90) {
            stopRecording();
        }
    }, 1000);
}

// Transcription functions
async function sendAudioForTranscription(audioBlob) {
    const formData = new FormData();
    formData.append('audio', audioBlob);
    formData.append('image_filename', currentImage.src.split('/').pop());
    
    try {
        const response = await fetch('/api/transcribe', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        currentTimestamp = data.timestamp;
        transcriptionDiv.textContent = data.transcription;
        refinedTranscription.value = data.transcription;
    } catch (err) {
        console.error('Error sending audio for transcription:', err);
        alert('Error processing audio. Please try again.');
    }
}

// Image segmentation functions
async function handleImageClick(event) {
    // Only allow segmentation clicks during recording
    if (!mediaRecorder || mediaRecorder.state !== 'recording') {
        console.log('Not recording, segmentation click ignored.');
        return;
    }

    const rect = currentImage.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    // Convert click coordinates to image coordinates
    const imageX = (x / rect.width) * currentImage.naturalWidth;
    const imageY = (y / rect.height) * currentImage.naturalHeight;
    
    // Get image as base64
    const imageBlob = await fetch(currentImage.src).then(res => res.blob());
    const reader = new FileReader();
    reader.readAsDataURL(imageBlob);
    
    reader.onloadend = async () => {
        const base64data = reader.result;
        
        try {
            // Remove any existing mask overlays
            const existingMask = document.getElementById('segmentationMask');
            if (existingMask) {
                existingMask.remove();
            }

            const response = await fetch('/api/segment', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    image: base64data,
                    points: [[imageX, imageY]]
                })
            });
            
            const data = await response.json();
            
            // Display the segmentation mask
            const maskImage = document.createElement('img');
            maskImage.id = 'segmentationMask';
            maskImage.src = data.mask;
            maskImage.style.position = 'absolute';
            maskImage.style.top = rect.top + 'px';
            maskImage.style.left = rect.left + 'px';
            maskImage.style.width = rect.width + 'px';
            maskImage.style.height = rect.height + 'px';
            maskImage.style.opacity = '0.5'; // Make the mask semi-transparent
            maskImage.style.pointerEvents = 'none'; // Allow clicks to go through to the image
            document.body.appendChild(maskImage);

            // Display segmentation score (as a placeholder for object name)
            let objectInfo = document.getElementById('objectInfo');
            if (!objectInfo) {
                objectInfo = document.createElement('div');
                objectInfo.id = 'objectInfo';
                objectInfo.style.position = 'absolute';
                objectInfo.style.top = (rect.bottom + 10) + 'px';
                objectInfo.style.left = rect.left + 'px';
                objectInfo.style.backgroundColor = 'white';
                objectInfo.style.padding = '5px';
                objectInfo.style.borderRadius = '5px';
                objectInfo.style.zIndex = '100';
                document.body.appendChild(objectInfo);
            }
            objectInfo.textContent = `Segmentation Score: ${data.score.toFixed(2)}`;

        } catch (err) {
            console.error('Error processing image segmentation:', err);
        }
    };
}

// Add an element to display object info
const objectInfoDiv = document.createElement('div');
objectInfoDiv.id = 'objectInfo';
objectInfoDiv.style.position = 'absolute';
objectInfoDiv.style.zIndex = '100'; // Ensure it's above other content
document.body.appendChild(objectInfoDiv);

// Event Listeners
recordButton.addEventListener('click', () => {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        stopRecording();
    } else {
        startRecording();
    }
});

saveRefinedButton.addEventListener('click', async () => {
    try {
        const response = await fetch('/api/save-refined', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                timestamp: currentTimestamp,
                refined_text: refinedTranscription.value
            })
        });
        
        const data = await response.json();
        if (data.success) {
            alert('Refined transcription saved successfully!');
        }
    } catch (err) {
        console.error('Error saving refined transcription:', err);
        alert('Error saving refined transcription. Please try again.');
    }
});

// Initialize the application
init(); 