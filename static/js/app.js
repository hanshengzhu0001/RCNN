let mediaRecorder;
let audioChunks = [];
let recordingStartTime;
let timerInterval;
let currentTimestamp;
let isMarkingMode = false;
let markedObjects = [];
let tempMarkerPosition = null;

// DOM Elements
const recordButton = document.getElementById('recordButton');
const markObjectsButton = document.getElementById('markObjectsButton');
const recordingTimer = document.getElementById('recordingTimer');
const transcriptionDiv = document.getElementById('transcription');
const refinedTranscription = document.getElementById('refinedTranscription');
const saveRefinedButton = document.getElementById('saveRefined');
const currentImage = document.getElementById('currentImage');
const objectDialog = document.getElementById('objectDialog');
const objectName = document.getElementById('objectName');
const saveMarkObject = document.getElementById('saveMarkObject');
const cancelMarkObject = document.getElementById('cancelMarkObject');
const objectList = document.getElementById('objectList');
const markersContainer = document.getElementById('markersContainer');

// Initialize
async function init() {
    // Load initial image
    const response = await fetch('/api/get-image');
    const data = await response.json();
    const imageFilename = data.image;
    currentImage.src = `/static/images/${imageFilename}`;
    
    // Load existing marked objects
    const objectsResponse = await fetch(`/api/get-objects/${imageFilename}`);
    const objectsData = await objectsResponse.json();
    markedObjects = objectsData.objects || [];
    
    // Display existing markers
    markedObjects.forEach(object => {
        addMarkerToImage(object);
    });
    updateObjectList();
    
    // Set up click handlers for image segmentation and object marking
    currentImage.addEventListener('click', handleImageClick);
    
    // Set up object marking handlers
    markObjectsButton.addEventListener('click', toggleMarkingMode);
    saveMarkObject.addEventListener('click', saveObject);
    cancelMarkObject.addEventListener('click', cancelObjectMarking);
    objectName.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            saveObject();
        }
    });
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
        
        // Display original transcriptions
        if (data.segmented_transcriptions && data.segmented_transcriptions.length > 0) {
            const fullTranscription = data.segmented_transcriptions
                .map(segment => segment.transcription)
                .join('\n');
            
            transcriptionDiv.textContent = fullTranscription;
        } else {
            transcriptionDiv.textContent = "No transcription available.";
        }

        // Display refined transcription if available
        if (data.refined_transcription) {
            refinedTranscription.value = data.refined_transcription;
        } else {
            refinedTranscription.value = "GPT refinement not available. Please try again.";
        }
    } catch (err) {
        console.error('Error sending audio for transcription:', err);
        alert('Error processing audio. Please try again.');
    }
}

// Object Marking Functions
function toggleMarkingMode() {
    isMarkingMode = !isMarkingMode;
    markObjectsButton.textContent = isMarkingMode ? 'Cancel Marking' : 'Mark Objects';
    markObjectsButton.classList.toggle('bg-red-500');
    markObjectsButton.classList.toggle('bg-green-500');
    markObjectsButton.classList.toggle('hover:bg-red-600');
    markObjectsButton.classList.toggle('hover:bg-green-600');
    
    // Disable recording while marking objects
    recordButton.disabled = isMarkingMode;
    recordButton.classList.toggle('opacity-50');
}

function showObjectDialog(x, y) {
    objectDialog.classList.remove('hidden');
    objectName.value = '';
    tempMarkerPosition = { x, y };
    objectName.focus();
}

function hideObjectDialog() {
    objectDialog.classList.add('hidden');
    tempMarkerPosition = null;
}

function cancelObjectMarking() {
    hideObjectDialog();
}

function saveObject() {
    const name = objectName.value.trim();
    if (name && tempMarkerPosition) {
        const object = {
            id: Date.now(),
            name: name,
            x: tempMarkerPosition.x,
            y: tempMarkerPosition.y
        };
        
        markedObjects.push(object);
        addMarkerToImage(object);
        updateObjectList();
        saveMarkedObjects();
        
        hideObjectDialog();
    }
}

function addMarkerToImage(object) {
    const marker = document.createElement('div');
    marker.className = 'marker';
    marker.style.left = `${object.x}px`;
    marker.style.top = `${object.y}px`;
    marker.setAttribute('data-object-id', object.id);
    
    const label = document.createElement('div');
    label.className = 'marker-label';
    label.textContent = object.name;
    marker.appendChild(label);
    
    markersContainer.appendChild(marker);
}

function updateObjectList() {
    objectList.innerHTML = markedObjects.map(obj => `
        <div class="flex justify-between items-center py-1">
            <span>${obj.name}</span>
            <button onclick="removeObject(${obj.id})" class="text-red-500 hover:text-red-700">
                ×
            </button>
        </div>
    `).join('');
}

function removeObject(id) {
    markedObjects = markedObjects.filter(obj => obj.id !== id);
    const marker = markersContainer.querySelector(`[data-object-id="${id}"]`);
    if (marker) {
        marker.remove();
    }
    updateObjectList();
    saveMarkedObjects();
}

async function saveMarkedObjects() {
    try {
        const response = await fetch('/api/save-objects', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image_filename: currentImage.src.split('/').pop(),
                objects: markedObjects
            })
        });
        
        if (!response.ok) {
            console.error('Failed to save marked objects');
        }
    } catch (err) {
        console.error('Error saving marked objects:', err);
    }
}

// Modified Image Click Handler
async function handleImageClick(event) {
    const rect = currentImage.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    
    if (isMarkingMode) {
        showObjectDialog(x, y);
        return;
    }
    
    // Only allow segmentation clicks during recording
    if (!mediaRecorder || mediaRecorder.state !== 'recording') {
        console.log('Not recording, segmentation click ignored.');
        return;
    }

    // Original segmentation code...
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
            maskImage.style.opacity = '0.5';
            maskImage.style.pointerEvents = 'none';
            document.body.appendChild(maskImage);

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

// Event Listeners
recordButton.addEventListener('click', () => {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        stopRecording();
    } else {
        startRecording();
    }
});

saveRefinedButton.addEventListener('click', async () => {
    if (!currentTimestamp) {
        alert('No transcription has been created yet.');
        return;
    }
    
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
        } else {
            alert('Error saving refined transcription.');
        }
    } catch (err) {
        console.error('Error saving refined transcription:', err);
        alert('Error saving refined transcription. Please try again.');
    }
});

// Initialize the application
init(); 