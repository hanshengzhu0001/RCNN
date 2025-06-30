let mediaRecorder;
let audioChunks = [];
let recordingStartTime;
let timerInterval;
let currentTimestamp;
let currentDetections = []; // Store current YOLO detections
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
const analyzeScientificButton = document.getElementById('analyzeScientificButton');

// Add D3.js library loading
const script = document.createElement('script');
script.src = 'https://d3js.org/d3.v7.min.js';
script.onload = () => {
    console.log('D3.js loaded successfully');
    init();
};
document.head.appendChild(script);
const objectDialog = document.getElementById('objectDialog');
const objectName = document.getElementById('objectName');
const saveMarkObject = document.getElementById('saveMarkObject');
const cancelMarkObject = document.getElementById('cancelMarkObject');
const objectList = document.getElementById('objectList');
const markersContainer = document.getElementById('markersContainer');
const clickedBoxInfo = document.getElementById('clickedBoxInfo'); // Get reference to the new div

// Initialize
async function init() {
    try {
        console.log('Starting initialization...');
        
        // Add event listeners
        recordButton.addEventListener('click', () => {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                stopRecording();
            } else {
                startRecording();
            }
        });
        
        markObjectsButton.addEventListener('click', toggleMarkingMode);
        analyzeScientificButton.addEventListener('click', analyzeScientificImage);
        
        // Create loading indicator first
        const loadingDiv = document.createElement('div');
        loadingDiv.id = 'loadingIndicator';
        loadingDiv.textContent = 'Loading image...';
        loadingDiv.style.position = 'absolute';
        loadingDiv.style.top = '50%';
        loadingDiv.style.left = '50%';
        loadingDiv.style.transform = 'translate(-50%, -50%)';
        loadingDiv.style.backgroundColor = 'rgba(255, 255, 255, 0.8)';
        loadingDiv.style.padding = '10px';
        loadingDiv.style.borderRadius = '5px';
        document.querySelector('.image-container').appendChild(loadingDiv);
        console.log('Loading indicator created');

        // Load initial image
        console.log('Fetching image from server...');
        const response = await fetch('/api/get-image');
        const data = await response.json();
        console.log('Received image data:', data);
        
        // Set up image load handler before setting src
        currentImage.onload = async () => {
            console.log('Image loaded successfully');
            // Change loading text to indicate processing
            loadingDiv.textContent = 'Processing image...';
            
            try {
                console.log('Starting background processing...');
                // Comment out object detection and pre-segmentation for now
                // await runObjectDetection();
                // console.log('Object detection completed');
                
                // await runPreSegmentation();
                // console.log('Pre-segmentation completed');
            } catch (error) {
                console.error('Error processing image:', error);
            } finally {
                // Remove loading indicator
                loadingDiv.remove();
                console.log('Loading indicator removed');
                // Automatically run scientific analysis after image loads
                await analyzeScientificImage();
            }
        };

        // Set image source after setting up onload handler
        console.log('Setting image source:', `/static/images/${data.image}`);
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
        console.log('Initialization completed');
    } catch (error) {
        console.error('Error initializing application:', error);
    }
}

// Run object detection on current image
async function runObjectDetection() {
    try {
        // Get image as base64
        const imageBlob = await fetch(currentImage.src).then(res => res.blob());
        const reader = new FileReader();
        
        const base64data = await new Promise((resolve) => {
            reader.onloadend = () => resolve(reader.result);
            reader.readAsDataURL(imageBlob);
        });
        
        const response = await fetch('/api/detect', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: base64data
            })
        });
        
        if (!response.ok) {
            throw new Error('Object detection request failed');
        }
        
        const data = await response.json();
        if (!data.detections) {
            throw new Error('No detections in response');
        }
        
        currentDetections = data.detections;
        console.log('Object detections:', currentDetections);
    } catch (err) {
        console.error('Error running object detection:', err);
        currentDetections = []; // Reset detections on error
    }
}

// Check if a point is inside a bounding box
function isPointInBox(point, box) {
    const [x1, y1, x2, y2] = box;
    const [x, y] = point;
    const isInside = x >= x1 && x <= x2 && y >= y1 && y <= y2;
    console.log(`Checking point (${x}, ${y}) against box [${x1}, ${y1}, ${x2}, ${y2}]: ${isInside}`);
    return isInside;
}

// Get object name at click point
function getObjectAtPoint(point) {
    console.log('Checking detections at point:', point);
    console.log('Current detections:', currentDetections);
    
    for (const detection of currentDetections) {
        if (isPointInBox(point, detection.box)) {
            console.log('Found object:', detection);
            return {
                name: detection.class,
                confidence: detection.confidence
            };
        }
    }
    console.log('No object found at point');
    return null;
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

    // Check for scientific bounding box click first (highest priority)
    const scientificBox = getScientificBoxAtPoint([x, y]);
    if (scientificBox) {
        displayClickedBoxInfo(scientificBox.label, scientificBox.confidence, scientificBox.caption);
        return; // Exit early if scientific box was clicked
    }

    // Check for object detection as fallback
    const object = getObjectAtPoint([x, y]);
    const tooltip = d3.select(".segment-tooltip");
    
    if (object) {
        // Display object detection result with confidence score
        tooltip.html(`${object.name} (${(object.confidence * 100).toFixed(1)}%)`)
            .style("left", (event.pageX + 10) + "px")
            .style("top", (event.pageY + 10) + "px")
            .style("display", "block");
    } else {
        // Display "No object detected" with 0% confidence
        tooltip.html(`No object detected (0.0%)`)
            .style("left", (event.pageX + 10) + "px")
            .style("top", (event.pageY + 10) + "px")
            .style("display", "block");
    }
    
    // Hide tooltip after 2 seconds
    setTimeout(() => {
        tooltip.style("display", "none");
    }, 2000);
    
    // Only proceed with segmentation if recording
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
            const response = await fetch('/api/segment', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    image: base64data,
                    x: imageX,
                    y: imageY,
                    regions: window.currentRegions || []  // Pass current regions for background check
                })
            });
            
            if (!response.ok) {
                const error = await response.json();
                console.log('Segment request failed:', error.error);
                return;
            }
            
            const data = await response.json();
            
            // Add the new segment to the visualization
            const overlay = document.getElementById('segmentationOverlay');
            if (overlay) {
                const svg = d3.select(overlay).select('svg');
                
                // If this is a background segment, remove previous background segments
                if (data.class === 'background') {
                    // Remove previous background segments from SVG
                    svg.selectAll(".segment")
                        .filter(function() {
                            return d3.select(this).attr("data-class") === "background";
                        })
                        .remove();
                    
                    // Remove previous background segments from currentRegions
                    if (window.currentRegions) {
                        window.currentRegions = window.currentRegions.filter(region => region.class !== 'background');
                    }
                }
                
                // Create a new group for the segment
                const group = svg.append("g")
                    .attr("class", "segment")
                    .attr("data-id", data.id)
                    .attr("data-class", data.class)
                    .attr("data-score", data.score)
                    .style("opacity", 0.7)
                    .style("pointer-events", "auto")
                    .style("cursor", "pointer");
                
                // Draw polygons for the segment
                data.polygons.forEach(polygon => {
                    // Ensure polygon is closed
                    if (polygon.length > 0 && polygon[0] !== polygon[polygon.length - 1]) {
                        polygon.push(polygon[0]);
                    }
                    
                    // Create polygon with explicit styling
                    group.append("polygon")
                        .attr("points", polygon.map(p => `${p[0]},${p[1]}`).join(" "))
                        .style("fill", `rgba(${data.color.join(",")}, 0.5)`)
                        .style("stroke", `rgb(${data.color.join(",")})`)
                        .style("stroke-width", "2px");
                });
                
                // Add click timestamp for segmentation
                const timestamp = Date.now() - recordingStartTime;
                group.attr('data-timestamp', timestamp);
                
                // Store in global variable for background checking
                if (!window.currentRegions) {
                    window.currentRegions = [];
                }
                window.currentRegions.push(data);
            }
        } catch (error) {
            console.error('Error in segmentation:', error);
        }
    };
}

// Add function to check for scientific bounding box clicks
function getScientificBoxAtPoint(point) {
    const [clickX, clickY] = point;
    
    // Get all detection boxes from the SVG
    const detectionBoxes = d3.selectAll('.detection-box');
    
    let clickedBox = null;
    detectionBoxes.each(function() {
        const group = d3.select(this);
        const rect = group.select('rect');
        
        if (!rect.empty()) {
            // Get the scaled coordinates directly from the rect attributes
            const x = parseFloat(rect.attr('x'));
            const y = parseFloat(rect.attr('y'));
            const width = parseFloat(rect.attr('width'));
            const height = parseFloat(rect.attr('height'));
            
            // Check if click is within this bounding box using simple coordinate comparison
            if (clickX >= x && clickX <= x + width && clickY >= y && clickY <= y + height) {
                clickedBox = {
                    label: group.attr('data-label') || 'Unknown',
                    confidence: parseFloat(group.attr('data-confidence')) || 0,
                    caption: group.attr('data-caption') || '',
                    text: group.attr('data-text') || ''
                };
                return false; // Break out of each loop
        }
        }
    });
    
    return clickedBox;
}

// Event Listeners
recordButton.addEventListener('click', () => {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
        stopRecording();
        recordButton.textContent = 'Start Recording';
        recordButton.classList.remove('recording');
    } else {
        startRecording();
        recordButton.textContent = 'Stop Recording';
        recordButton.classList.add('recording');
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

// Add pre-segmentation function
async function runPreSegmentation() {
    try {
        console.log('Starting pre-segmentation...');
        // Get image as base64
        const imageBlob = await fetch(currentImage.src).then(res => res.blob());
        const reader = new FileReader();
        
        // Convert blob to base64
        const base64data = await new Promise((resolve) => {
            reader.onloadend = () => resolve(reader.result);
            reader.readAsDataURL(imageBlob);
        });
        
        console.log('Sending pre-segmentation request...');
        const response = await fetch('/api/pre-segment', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: base64data
            })
        });
        
        const data = await response.json();
        console.log('Received segmentation data:', data.regions ? data.regions.length : 0, 'regions');
        
        if (data.regions) {
            // Wait for next frame to ensure DOM is ready
            await new Promise(resolve => requestAnimationFrame(resolve));
            
            // Create or update segmentation overlay
            let overlay = document.getElementById('segmentationOverlay');
            if (!overlay) {
                overlay = document.createElement('div');
                overlay.id = 'segmentationOverlay';
                overlay.className = 'segmentation-overlay';
                overlay.style.position = 'absolute';
                overlay.style.top = '0';
                overlay.style.left = '0';
                overlay.style.width = '100%';
                overlay.style.height = '100%';
                overlay.style.pointerEvents = 'none';
                overlay.style.zIndex = '1000';
                document.querySelector('.image-container').appendChild(overlay);
            }
            
            // Clear previous visualization
            overlay.innerHTML = '';
            
            // Get image dimensions and position
            const rect = currentImage.getBoundingClientRect();
            const imgWidth = rect.width;
            const imgHeight = rect.height;
            const naturalWidth = currentImage.naturalWidth;
            const naturalHeight = currentImage.naturalHeight;

            console.log('Image dimensions:', {
                rendered: { width: imgWidth, height: imgHeight },
                natural: { width: naturalWidth, height: naturalHeight },
                scaleX: imgWidth / naturalWidth,
                scaleY: imgHeight / naturalHeight
            });
            
            // Create SVG with explicit dimensions and position
            const svg = d3.select(overlay)
                .append("svg")
                .attr("width", imgWidth)
                .attr("height", imgHeight)
                .attr("viewBox", `0 0 ${imgWidth} ${imgHeight}`)
                .style("position", "absolute")
                .style("top", "0")
                .style("left", "0")
                .style("width", "100%")
                .style("height", "100%")
                .style("pointer-events", "auto");
            
            console.log('Drawing segments...');
            // Create a group for each region
            const regionGroups = svg.selectAll("g")
                .data(data.regions)
                .enter()
                .append("g")
                .attr("class", "segment")
                .attr("data-id", d => d.id)
                .attr("data-class", d => d.class)
                .attr("data-score", d => d.score)
                .style("opacity", 0.7)  // Make segments more visible
                .style("pointer-events", "auto")
                .style("cursor", "pointer");
            
            // Draw polygons for each region
            regionGroups.each(function(region) {
                console.log('Drawing region:', region);
                const group = d3.select(this);
                // Draw mask pixels as rectangles
                const mask = region.mask;
                const color = region.color;

                for (let y = 0; y < mask.length; y++) {
                    for (let x = 0; x < mask[0].length; x++) {
                        if (mask[y][x] === 1) {
                            group.append("rect")
                                .attr("x", x)
                                .attr("y", y)
                                .attr("width", 1)
                                .attr("height", 1)
                                .style("fill", `rgb(${color.join(",")})`)
                                .style("opacity", 0.7) // Adjust opacity as needed
                                .style("pointer-events", "auto");
                        }
                    }
                }
            });
            
            // Add hover effects
            regionGroups
                .on("mouseover", function(event, d) {
                    // Highlight the hovered segment using a filter
                    d3.select(this)
                        .style("filter", "brightness(1.2)"); // Apply a brightness filter on hover
                    
                    svg.selectAll(".segment")
                        .filter(other => other.id !== d.id)
                        .style("opacity", 0.4); // Dim other segments
                    
                    // Find the center of the mask (optional - could use bounding box or just the event point)
                    const mask = d.mask;
                    let sumX = 0;
                    let sumY = 0;
                    let count = 0;
                    for (let y = 0; y < mask.length; y++) {
                        for (let x = 0; x < mask[0].length; x++) {
                            if (mask[y][x] === 1) {
                                sumX += x;
                                sumY += y;
                                count++;
                            }
                        }
                    }
                    
                    let bestMatch = null;
                    if (count > 0) {
                        const maskCenterX = sumX / count;
                        const maskCenterY = sumY / count;
                        
                        // Find overlapping detection with highest confidence
                        for (const detection of currentDetections) {
                            // Check if mask center is within detection bounding box
                            if (isPointInBox([maskCenterX, maskCenterY], detection.box)) {
                                if (!bestMatch || detection.confidence > bestMatch.confidence) {
                                    bestMatch = detection;
                                }
                            }
                        }
                    }
                    
                    let tooltipText;
                    if (bestMatch) {
                        // If object detection found, show its class and confidence
                        tooltipText = `${bestMatch.class} (${(bestMatch.confidence * 100).toFixed(1)}%)`;
                    } else {
                        // If no object detection, show segmentation class without score
                         tooltipText = `${d.class}`;
                    }

                    const tooltip = d3.select(".segment-tooltip");
                    tooltip.html(tooltipText)
                        .style("display", "block")
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY + 10) + "px");
                })
                .on("mousemove", function(event) {
                    const tooltip = d3.select(".segment-tooltip");
                    tooltip
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY + 10) + "px");
                })
                .on("mouseout", function() {
                    svg.selectAll(".segment")
                        .style("opacity", 0.7); // Restore opacity of all segments
                    
                    // Remove the filter from the hovered segment
                    d3.select(this)
                        .style("filter", null); // Remove the filter on mouseout
                    
                    const tooltip = d3.select(".segment-tooltip");
                    tooltip.style("display", "none");
                });
            
            console.log('Segmentation visualization complete');
        }
    } catch (err) {
        console.error('Error running pre-segmentation:', err);
    }
}

// Add function to handle overlapping boxes
function handleOverlappingBoxes(detections) {
    // Sort detections by confidence score (highest first)
    const sortedDetections = [...detections].sort((a, b) => b.confidence - a.confidence);
    const processedBoxes = []; // Store processed boxes in [x1, y1, x2, y2] format

    // Helper function to calculate IoU (Intersection over Union)
    function calculateIoU(box1, box2) {
        const [x1_1, y1_1, x2_1, y2_1] = box1;
        const [x1_2, y1_2, x2_2, y2_2] = box2;

        const x_overlap = Math.max(0, Math.min(x2_1, x2_2) - Math.max(x1_1, x1_2));
        const y_overlap = Math.max(0, Math.min(y2_1, y2_2) - Math.max(y1_1, y1_1)); // Corrected y1_1 typo
        
        const intersection = x_overlap * y_overlap;
        
        const box1_area = (x2_1 - x1_1) * (y2_1 - y1_1);
        const box2_area = (x2_2 - x1_2) * (y2_2 - y1_2);
        
        const union = box1_area + box2_area - intersection;
        
        return union === 0 ? 0 : intersection / union;
    }

    for (const detection of sortedDetections) {
        if (!detection.bbox) continue;

        let [x1, y1, x2, y2] = detection.bbox; // Get coordinates in [x1, y1, x2, y2] format
        let currentBox = [x1, y1, x2, y2];

        let hasOverlap = true;
        let attempts = 0;
        const maxAttempts = 20;
        const minBoxSize = 10; // Minimum dimension for a box to be considered valid

        while (hasOverlap && attempts < maxAttempts) {
            hasOverlap = false;
            let conflictBox = null;

            // Check overlap with all processed boxes
            for (const processed of processedBoxes) {
                if (calculateIoU(currentBox, processed.bbox) > 0.1) { // Use IoU for overlap check
                    hasOverlap = true;
                    conflictBox = processed.bbox;
                    break; // Handle one significant conflict at a time
                }
            }

            // If we found a conflict, try to resolve it
            if (conflictBox) {
                 const [px1, py1, px2, py2] = conflictBox; // Get conflict box in [x1, y1, x2, y2]

                // Calculate overlap dimensions
                const overlapX1 = Math.max(x1, px1);
                const overlapY1 = Math.max(y1, py1);
                const overlapX2 = Math.min(x2, px2);
                const overlapY2 = Math.min(y2, py2);

                const overlapWidth = overlapX2 - overlapX1;
                const overlapHeight = overlapY2 - overlapY1;

                // Simple resolution: try to push or shrink the current box out of the overlap
                // Prioritize shrinking along the smaller overlap dimension
                if (overlapWidth > 0 && overlapHeight > 0) {
                     if (overlapWidth < overlapHeight) {
                        // Overlap is narrower horizontally, try to shift or shrink horizontally
                        if (x1 < px1) { // Current box is to the left of conflict
                            x2 = overlapX1; // Shrink right side to the start of overlap
                        } else { // Current box is to the right of conflict
                            x1 = overlapX2; // Shift left side to the end of overlap
                        }
                     } else {
                        // Overlap is narrower vertically, try to shift or shrink vertically
                        if (y1 < py1) { // Current box is above conflict
                            y2 = overlapY1; // Shrink bottom side to the start of overlap
                        } else { // Current box is below conflict
                            y1 = overlapY2; // Shift top side to the end of overlap
                        }
                     }
                }

                // Update the current box coordinates after potential adjustment
                currentBox = [x1, y1, x2, y2];

                // Re-check if the box is still valid after adjustment
                if (x2 - x1 <= minBoxSize || y2 - y1 <= minBoxSize) {
                    hasOverlap = false; // Treat as resolved (by effectively removing)
                    // Note: We are not adding this tiny box to processedBoxes
                }
            }

            attempts++;
        }

        // Add the box if it's valid after attempts to resolve overlap
        if (currentBox[2] - currentBox[0] > minBoxSize && currentBox[3] - currentBox[1] > minBoxSize) {
            processedBoxes.push({
                ...detection,
                bbox: currentBox // Use the potentially adjusted box dimensions in [x1, y1, x2, y2] format
            });
        }
    }

    return processedBoxes;
}

async function analyzeScientificImage() {
    try {
        // Get image as base64
        const imageBlob = await fetch(currentImage.src).then(res => res.blob());
        const reader = new FileReader();
        
        const base64data = await new Promise((resolve) => {
            reader.onloadend = () => resolve(reader.result);
            reader.readAsDataURL(imageBlob);
        });
        
        const response = await fetch('/api/analyze-scientific', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: base64data
            })
        });
        
        if (!response.ok) {
            throw new Error('Scientific image analysis request failed');
        }
        
        const data = await response.json();
        console.log('Raw analysis data:', data); // Debug: Print raw data

        // Find the overall best y-tick and x-tick among all detections (for potential later visualization)
        let overallBestYTick = null;
        let overallBestXTick = null;
        (data.text_elements || []).forEach((det) => {
             if (det.bbox) {
                 const lowerCaseClass = (det.element_type || det.type || '').toLowerCase();
                 if (lowerCaseClass === 'y-tick') {
                     if (!overallBestYTick || det.confidence > overallBestYTick.confidence) {
                         overallBestYTick = det;
                     }
                 } else if (lowerCaseClass === 'x-tick') {
                     if (!overallBestXTick || det.confidence > overallBestXTick.confidence) {
                         overallBestXTick = det;
                     }
                 }
             }
        });
        console.log('Overall Best Y-Tick (before threshold):', overallBestYTick); // Debug: Print overall best
        console.log('Overall Best X-Tick (before threshold):', overallBestXTick); // Debug: Print overall best

        // Apply 20% confidence threshold to match backend
        const threshold = 0.2;
        const thresholdedChartDetections = (data.chart_elements || []).filter(det => det.confidence >= threshold);
        const thresholdedTextDetections = (data.text_elements || []).filter(det => det.confidence >= threshold);
        console.log('Thresholded chart detections:', thresholdedChartDetections); // Debug: Print thresholded chart data
        console.log('Thresholded text detections:', thresholdedTextDetections); // Debug: Print thresholded text data

        // Track if a y-tick or x-tick has been added from chart detections
        let yTickFoundInChart = null;
        let xTickFoundInChart = null;

        // Find the first y-tick and x-tick in chart detections above the threshold
        for (const det of thresholdedChartDetections) {
            if (det.bbox) {
                const lowerCaseClass = (det.element_type || '').toLowerCase();
                if (lowerCaseClass === 'y-tick' && !yTickFoundInChart) {
                    yTickFoundInChart = det;
                } else if (lowerCaseClass === 'x-tick' && !xTickFoundInChart) {
                    xTickFoundInChart = det;
                }
                // Stop looking once both are found (optimization)
                if (yTickFoundInChart && xTickFoundInChart) {
                    break;
                }
            }
        }

        // Create a new array for chart detections excluding the found ticks
        const otherChartDetections = thresholdedChartDetections.filter(det => {
            const lowerCaseClass = (det.element_type || '').toLowerCase();
            return lowerCaseClass !== 'y-tick' && lowerCaseClass !== 'x-tick';
        });

        // Filter thresholded text detections to keep other non-axis text
        const otherTextDetections = [];
        const axisLabelPattern = /^[-\d.,()\s]+$/; // Pattern for numbers, commas, parentheses, spaces, and hyphen

        // Create the final list of text detections to visualize and display
        const textDetectionsToProcessAndVisualize = [];

        thresholdedTextDetections.forEach((det) => { // Iterate over thresholded text detections
            if (det.bbox) {
                const lowerCaseClass = (det.element_type || det.type || '').toLowerCase(); // Convert class to lowercase for comparison

                // Keep only detections that are NOT y-tick or x-tick and not generic axis labels
                if (lowerCaseClass !== 'y-tick' && lowerCaseClass !== 'x-tick' && !axisLabelPattern.test((det.text || '').trim())) { // Also handle undefined text
                    textDetectionsToProcessAndVisualize.push(det);
                }
            }
        });

        // Add the found y-tick and x-tick from chart detections to the visualization list
        if (yTickFoundInChart) {
            textDetectionsToProcessAndVisualize.push(yTickFoundInChart);
        }
        if (xTickFoundInChart) {
            textDetectionsToProcessAndVisualize.push(xTickFoundInChart);
        }

        console.log('Text detections for processing and visualization (pre-overlap):', textDetectionsToProcessAndVisualize); // Debug: Print list before overlap processing

        // Process overlapping boxes (now applied to the filtered/selected lists)
        const processedChartDetections = handleOverlappingBoxes(otherChartDetections);
        const processedTextDetections = handleOverlappingBoxes(textDetectionsToProcessAndVisualize); // Process the filtered list

        console.log('Processed chart detections (post-overlap):', processedChartDetections); // Debug: Print processed chart data

        // Create a combined list for displaying Chart Elements, including the selected ticks
        const chartElementsForDisplay = [...processedChartDetections];
        if (yTickFoundInChart) {
            chartElementsForDisplay.push(yTickFoundInChart);
        }
        if (xTickFoundInChart) {
            chartElementsForDisplay.push(xTickFoundInChart);
        }

        // Create or get the overlay for bounding boxes
        let overlay = document.getElementById('segmentationOverlay');
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.id = 'segmentationOverlay';
            overlay.className = 'segmentation-overlay';
            document.querySelector('.image-container').appendChild(overlay);
        }
        
        // Clear previous visualization and clicked info
        overlay.innerHTML = '';
        clickedBoxInfo.textContent = 'Click on a bounding box on the image to see details here.';
        
        // Get image dimensions - Use current rendered size only
        const imgRect = currentImage.getBoundingClientRect();
        const imgWidth = imgRect.width;
        const imgHeight = imgRect.height;
        const naturalWidth = currentImage.naturalWidth;
        const naturalHeight = currentImage.naturalHeight;

        // Calculate simple scale factors
        const scaleX = imgWidth / naturalWidth;
        const scaleY = imgHeight / naturalHeight;

        console.log('Image scaling factors:', {
            rendered: { width: imgWidth, height: imgHeight },
            natural: { width: naturalWidth, height: naturalHeight },
            scaleX: scaleX,
            scaleY: scaleY
        });

        // Create SVG overlay with exact image dimensions
        const svg = d3.select(overlay)
            .append("svg")
            .attr("width", imgWidth)
            .attr("height", imgHeight)
            .style("position", "absolute")
            .style("top", "0")
            .style("left", "0")
            .style("pointer-events", "auto");

        // Create tooltip if it doesn't exist
        let tooltip = d3.select(".segment-tooltip");
        if (tooltip.empty()) {
             tooltip = d3.select("body")
                .append("div")
                .attr("class", "segment-tooltip")
                .style("position", "absolute")
                .style("background-color", "rgba(0, 0, 0, 0.9)")
                .style("color", "white")
                .style("padding", "6px 8px")
                .style("border-radius", "4px")
                .style("font-size", "11px")
                .style("pointer-events", "none")
                .style("z-index", "1000")
                .style("display", "none");
        }

        // Add bounding boxes for chart elements - initially hidden
        processedChartDetections.forEach((det, index) => {
            if (det.bbox) {
                const [x1, y1, x2, y2] = det.bbox;
                
                // Simple proportional scaling
                const scaledX1 = x1 * scaleX;
                const scaledY1 = y1 * scaleY;
                const scaledX2 = x2 * scaleX;
                const scaledY2 = y2 * scaleY;
                const scaledWidth = scaledX2 - scaledX1;
                const scaledHeight = scaledY2 - scaledY1;

                const group = svg.append("g")
                    .attr("class", `detection-box chart-detection`)
                    .attr("id", `chart-element-${index}`)
                    .attr("data-label", (det.element_type || 'Unknown Class'))
                    .attr("data-confidence", det.confidence)
                    .attr("data-caption", det.caption || '')
                    .style("display", "block"); // Show chart elements by default

                // Add event listeners
                group.on("click", function(event) {
                    event.stopPropagation();
                    displayClickedBoxInfo((det.element_type || 'Unknown Class'), det.confidence, det.caption);
                })
                .on("mouseover", function(event) {
                    d3.select(this).style("filter", "brightness(1.2)");
                    const tooltipText = `${(det.element_type || 'Unknown Class')} (${(det.confidence * 100).toFixed(1)}%)`;
                    d3.select(".segment-tooltip")
                        .html(tooltipText)
                        .style("display", "block")
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY + 10) + "px");
                })
                .on("mousemove", function(event) {
                     d3.select(".segment-tooltip")
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY + 10) + "px");
                })
                .on("mouseout", function() {
                    d3.select(this).style("filter", null);
                    d3.select(".segment-tooltip").style("display", "none");
                });

                // Add rectangle with consistent sizing
                group.append("rect")
                    .attr("x", scaledX1)
                    .attr("y", scaledY1)
                    .attr("width", scaledWidth)
                    .attr("height", scaledHeight)
                    .attr("fill", "red")
                    .attr("stroke", "red")
                    .attr("stroke-width", 2)
                    .attr("fill-opacity", 0.1);

                // Add simple fixed-size label
                const labelText = `${(det.element_type || 'Unknown Class')} ${(det.confidence * 100).toFixed(0)}%`;
                const fixedFontSize = 9; // Fixed font size
                const fixedTextHeight = 14; // Fixed text height

                // Text background with fixed size
                group.append("rect")
                    .attr("x", scaledX1)
                    .attr("y", scaledY1 - fixedTextHeight)
                    .attr("width", labelText.length * 5.5) // Fixed character width
                    .attr("height", fixedTextHeight)
                    .attr("fill", "black")
                    .attr("fill-opacity", 0.8)
                    .style("pointer-events", "none");

                // Text label with fixed size
                group.append("text")
                    .attr("x", scaledX1 + 2)
                    .attr("y", scaledY1 - 3)
                    .text(labelText)
                    .attr("fill", "white")
                    .style("font-size", `${fixedFontSize}px`)
                    .style("font-weight", "bold")
                    .style("font-family", "Arial, sans-serif")
                    .style("pointer-events", "none");
            }
        });
        
        // Add bounding boxes for text elements - initially hidden
        processedTextDetections.forEach((det, index) => {
            if (det.bbox) {
                const [x1, y1, x2, y2] = det.bbox;
                
                // Simple proportional scaling
                const scaledX1 = x1 * scaleX;
                const scaledY1 = y1 * scaleY;
                const scaledX2 = x2 * scaleX;
                const scaledY2 = y2 * scaleY;
                const scaledWidth = scaledX2 - scaledX1;
                const scaledHeight = scaledY2 - scaledY1;

                const group = svg.append("g")
                    .attr("class", `detection-box text-detection`)
                    .attr("id", `text-element-${index}`)
                    .attr("data-label", (det.text || det.element_type || det.type || 'Unknown Text'))
                    .attr("data-confidence", det.confidence)
                    .attr("data-text", det.text || '')
                    .attr("data-caption", det.caption || '')
                    .style("display", "block"); // Show text elements by default

                // Add event listeners
                 group.on("click", function(event) {
                    event.stopPropagation();
                    displayClickedBoxInfo((det.text || det.element_type || det.type || 'Unknown Text'), det.confidence, det.caption);
                })
                .on("mouseover", function(event) {
                    d3.select(this).style("filter", "brightness(1.2)");
                    const tooltipText = `${(det.text || det.element_type || det.type || 'Unknown Text')} (${(det.confidence * 100).toFixed(1)}%)`;
                     d3.select(".segment-tooltip")
                        .html(tooltipText)
                        .style("display", "block")
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY + 10) + "px");
                })
                .on("mousemove", function(event) {
                     d3.select(".segment-tooltip")
                        .style("left", (event.pageX + 10) + "px")
                        .style("top", (event.pageY + 10) + "px");
                })
                .on("mouseout", function() {
                    d3.select(this).style("filter", null);
                    d3.select(".segment-tooltip").style("display", "none");
                });

                // Add rectangle with consistent sizing
                group.append("rect")
                    .attr("x", scaledX1)
                    .attr("y", scaledY1)
                    .attr("width", scaledWidth)
                    .attr("height", scaledHeight)
                    .attr("fill", "blue")
                    .attr("stroke", "blue")
                    .attr("stroke-width", 2)
                    .attr("fill-opacity", 0.1);

                // Add simple fixed-size label
                const labelText = `${(det.text || det.element_type || det.type || 'Unknown Text')} ${(det.confidence * 100).toFixed(0)}%`;
                const fixedFontSize = 9; // Fixed font size
                const fixedTextHeight = 14; // Fixed text height

                // Text background with fixed size
                group.append("rect")
                    .attr("x", scaledX1)
                    .attr("y", scaledY1 - fixedTextHeight)
                    .attr("width", labelText.length * 5.5) // Fixed character width
                    .attr("height", fixedTextHeight)
                    .attr("fill", "black")
                    .attr("fill-opacity", 0.8)
                    .style("pointer-events", "none");

                // Text label with fixed size
                group.append("text")
                    .attr("x", scaledX1 + 2)
                    .attr("y", scaledY1 - 3)
                    .text(labelText)
                    .attr("fill", "white")
                    .style("font-size", `${fixedFontSize}px`)
                    .style("font-weight", "bold")
                    .style("font-family", "Arial, sans-serif")
                    .style("pointer-events", "none");
            }
        });
        
        // Display the analysis results with interactive buttons grouped by element type
        const analysisResults = document.createElement('div');
        analysisResults.className = 'analysis-results bg-white p-4 rounded-lg shadow-lg mt-4';

        // Create caption section
        const captionSection = document.createElement('div');
        captionSection.className = 'caption-section mb-4';
        captionSection.innerHTML = `
            <h3 class="text-lg font-semibold mb-2">Auto Caption</h3>
            <p class="text-gray-700">${data.caption}</p>
        `;
        analysisResults.appendChild(captionSection);

        // Group elements by type
        const chartElementsByType = {};
        const textElementsByType = {};
        
        processedChartDetections.forEach((det, index) => {
            const elementType = det.element_type || 'Unknown Class';
            if (!chartElementsByType[elementType]) {
                chartElementsByType[elementType] = [];
            }
            chartElementsByType[elementType].push({...det, index});
        });
        
        processedTextDetections.forEach((det, index) => {
            const elementType = det.element_type || det.type || 'Unknown Text';
            if (!textElementsByType[elementType]) {
                textElementsByType[elementType] = [];
            }
            textElementsByType[elementType].push({...det, index});
        });

        // Create interactive detections section with grouped buttons
        const detectionsSection = document.createElement('div');
        detectionsSection.className = 'detections-section';
        
        // Chart Elements table rows
        const chartElementsHtml = Object.keys(chartElementsByType).map(elementType => {
            const elements = chartElementsByType[elementType];
            const elementIds = elements.map(el => `chart-element-${el.index}`).join(',');
            const avgConfidence = elements.reduce((sum, el) => sum + el.confidence, 0) / elements.length;
            
            return `
                <tr class="border-b border-gray-200 hover:bg-gray-50">
                    <td class="py-2 px-3">
                        <button onclick="toggleElementGroup('${elementIds}')" 
                                class="bg-red-100 hover:bg-red-200 text-red-800 px-3 py-1 rounded text-sm font-medium transition-colors">
                            ${elementType}
                        </button>
                    </td>
                    <td class="py-2 px-3 text-sm text-gray-600">
                        ${elements.length} items, avg: ${(avgConfidence * 100).toFixed(1)}%
                    </td>
                    <td class="py-2 px-3 text-center">
                        <input type="checkbox" id="chart-${elementType.replace(/\s+/g, '-')}" 
                               class="w-4 h-4 text-red-600 rounded focus:ring-red-500" />
                    </td>
                </tr>
            `;
        }).join('');

        // Text Elements table rows
        const textElementsHtml = Object.keys(textElementsByType).map(elementType => {
            const elements = textElementsByType[elementType];
            const elementIds = elements.map(el => `text-element-${el.index}`).join(',');
            const avgConfidence = elements.reduce((sum, el) => sum + el.confidence, 0) / elements.length;
            
            return `
                <tr class="border-b border-gray-200 hover:bg-gray-50">
                    <td class="py-2 px-3">
                        <button onclick="toggleElementGroup('${elementIds}')" 
                                class="bg-blue-100 hover:bg-blue-200 text-blue-800 px-3 py-1 rounded text-sm font-medium transition-colors">
                            ${elementType}
                        </button>
                    </td>
                    <td class="py-2 px-3 text-sm text-gray-600">
                        ${elements.length} items, avg: ${(avgConfidence * 100).toFixed(1)}%
                    </td>
                    <td class="py-2 px-3 text-center">
                        <input type="checkbox" id="text-${elementType.replace(/\s+/g, '-')}" 
                               class="w-4 h-4 text-blue-600 rounded focus:ring-blue-500" />
                    </td>
                </tr>
            `;
        }).join('');

        detectionsSection.innerHTML = `
            <div class="mb-6">
                <div class="flex justify-between items-center mb-3">
                    <h3 class="text-lg font-semibold">Chart Elements</h3>
                    <div class="space-x-2">
                        <button onclick="showAllBoundingBoxes('chart')" 
                                class="bg-red-500 hover:bg-red-600 text-white px-3 py-1 rounded text-xs">
                            Show All Chart
                        </button>
                        <button onclick="hideAllBoundingBoxes('chart')" 
                                class="bg-gray-500 hover:bg-gray-600 text-white px-3 py-1 rounded text-xs">
                            Hide All Chart
                        </button>
                    </div>
                </div>
                <div class="overflow-x-auto">
                    <table class="min-w-full bg-white border border-gray-200 rounded-lg">
                        <thead class="bg-gray-50">
                            <tr>
                                <th class="py-3 px-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Element Type</th>
                                <th class="py-3 px-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Details</th>
                                <th class="py-3 px-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">Completed</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-gray-200">
                            ${chartElementsHtml}
                        </tbody>
                    </table>
                </div>
            </div>
            <div class="mb-6">
                <div class="flex justify-between items-center mb-3">
                    <h3 class="text-lg font-semibold">Text Elements</h3>
                    <div class="space-x-2">
                        <button onclick="showAllBoundingBoxes('text')" 
                                class="bg-blue-500 hover:bg-blue-600 text-white px-3 py-1 rounded text-xs">
                            Show All Text
                        </button>
                        <button onclick="hideAllBoundingBoxes('text')" 
                                class="bg-gray-500 hover:bg-gray-600 text-white px-3 py-1 rounded text-xs">
                            Hide All Text
                        </button>
                    </div>
                </div>
                <div class="overflow-x-auto">
                    <table class="min-w-full bg-white border border-gray-200 rounded-lg">
                        <thead class="bg-gray-50">
                            <tr>
                                <th class="py-3 px-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Element Type</th>
                                <th class="py-3 px-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Details</th>
                                <th class="py-3 px-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider">Completed</th>
                            </tr>
                        </thead>
                        <tbody class="divide-y divide-gray-200">
                            ${textElementsHtml}
                        </tbody>
                    </table>
                </div>
            </div>
            <div class="mt-4 pt-4 border-t border-gray-200">
                <div class="flex justify-center space-x-2">
                    <button onclick="showAllBoundingBoxes('all')" 
                            class="bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded text-sm font-medium">
                        Show All Elements
                    </button>
                    <button onclick="hideAllBoundingBoxes('all')" 
                            class="bg-gray-500 hover:bg-gray-600 text-white px-4 py-2 rounded text-sm font-medium">
                        Hide All Elements
                    </button>
                </div>
            </div>
        `;
        analysisResults.appendChild(detectionsSection);

        // Add to the UI
        const existingResults = document.querySelector('.analysis-results');
        if (existingResults) {
            existingResults.remove();
        }
        const imageContainer = document.querySelector('.image-container');
        if (imageContainer) {
            imageContainer.appendChild(analysisResults);
        } else {
            console.error('Error: Could not find .image-container to display analysis results.');
        }

    } catch (err) {
        console.error('Error analyzing scientific image:', err);
        alert('Error analyzing scientific image. Please try again.');
    }
}

// Add functions to handle bounding box visibility - show element groups together
function toggleBoundingBox(elementId) {
    const element = d3.select(`#${elementId}`);
    if (!element.empty()) {
        const currentDisplay = element.style("display");
        
        if (currentDisplay === "none") {
            // Hide all other bounding boxes first
            d3.selectAll('.detection-box').style("display", "none");
            // Show only this one
            element.style("display", "block");
        } else {
            // Hide this one
            element.style("display", "none");
        }
    }
}

function toggleElementGroup(elementIds) {
    const ids = elementIds.split(',');
    const firstElement = d3.select(`#${ids[0]}`);
    
    if (!firstElement.empty()) {
        const currentDisplay = firstElement.style("display");
        
        if (currentDisplay === "none") {
            // Hide all other bounding boxes first
            d3.selectAll('.detection-box').style("display", "none");
            // Show all elements in this group
            ids.forEach(id => {
                d3.select(`#${id}`).style("display", "block");
            });
        } else {
            // Hide all elements in this group
            ids.forEach(id => {
                d3.select(`#${id}`).style("display", "none");
            });
        }
    }
}

function showAllBoundingBoxes(type) {
    let selector;
    switch(type) {
        case 'chart':
            selector = '.chart-detection';
            break;
        case 'text':
            selector = '.text-detection';
            break;
        case 'all':
        default:
            selector = '.detection-box';
            break;
    }
    d3.selectAll(selector).style("display", "block");
}

function hideAllBoundingBoxes(type) {
    let selector;
    switch(type) {
        case 'chart':
            selector = '.chart-detection';
            break;
        case 'text':
            selector = '.text-detection';
            break;
        case 'all':
        default:
            selector = '.detection-box';
            break;
    }
    d3.selectAll(selector).style("display", "none");
}

// Function to display clicked box info with enhanced formatting
function displayClickedBoxInfo(label, confidence, caption) {
    let infoText = `${label}`;
    
    // Add confidence score
    if (confidence !== undefined) {
        infoText += ` (${(confidence * 100).toFixed(1)}% confidence)`;
    }
    
    // Add caption if available
    if (caption && caption.trim() !== '') {
        infoText += ` - ${caption}`;
    }
    
    // Update the info display with enhanced styling
    clickedBoxInfo.innerHTML = `
        <div class="clicked-box-details">
            <span class="element-label font-semibold text-blue-600">${label}</span>
            ${confidence !== undefined ? `<span class="confidence-score text-sm text-gray-600">(${(confidence * 100).toFixed(1)}%)</span>` : ''}
            ${caption && caption.trim() !== '' ? `<div class="element-caption text-sm text-gray-700 mt-1">${caption}</div>` : ''}
        </div>
    `;
    
    // Add visual feedback
    clickedBoxInfo.style.borderLeft = '4px solid #3B82F6';
    clickedBoxInfo.style.backgroundColor = '#F8FAFC';
    
    // Auto-clear after 10 seconds
    setTimeout(() => {
        clickedBoxInfo.innerHTML = 'Click on a bounding box on the image to see details here.';
        clickedBoxInfo.style.borderLeft = '';
        clickedBoxInfo.style.backgroundColor = '';
    }, 10000);
}

// Add button to the UI
const analyzeButton = document.createElement('button');
analyzeButton.id = 'analyzeButton';
analyzeButton.textContent = 'Analyze Scientific Image';
analyzeButton.className = 'button';
analyzeButton.onclick = analyzeScientificImage;
document.querySelector('.button-container').appendChild(analyzeButton);