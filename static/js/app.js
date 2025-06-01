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

// Initialize
async function init() {
    try {
        console.log('Starting initialization...');
        
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
                // Run object detection first
                await runObjectDetection();
                console.log('Object detection completed');
                
                // Then run pre-segmentation
                await runPreSegmentation();
                console.log('Pre-segmentation completed');
            } catch (error) {
                console.error('Error processing image:', error);
            } finally {
                // Remove loading indicator
                loadingDiv.remove();
                console.log('Loading indicator removed');
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

    // Check for object detection first
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
                        .style("stroke-width", "2px")
                        .style("vector-effect", "non-scaling-stroke")
                        .style("pointer-events", "auto");
                });
                
                // Add hover effects
                group.on("mouseover", function(event) {
                    // Highlight the hovered segment by increasing opacity of its pixels
                    d3.select(this).selectAll("rect")
                        .style("opacity", 0.9); // Adjust opacity for hover
                    
                    svg.selectAll(".segment")
                        .filter(other => other.id !== data.id)
                        .style("opacity", 0.4); // Dim other segments
                    
                    let tooltipText = `${data.class}`;
                    if (data.score) {
                        tooltipText += ` (${(data.score * 100).toFixed(1)}%)`;
                    }
                    
                    const tooltip = d3.select(".segment-tooltip");
                    tooltip.html(tooltipText)
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
                    
                    // Restore opacity of pixels in the hovered segment
                    d3.select(this).selectAll("rect")
                        .style("opacity", 0.7); // Adjust opacity back to normal
                    
                    const tooltip = d3.select(".segment-tooltip");
                    tooltip.style("display", "none");
                });
                
                // Store the new region
                if (!window.currentRegions) {
                    window.currentRegions = [];
                }
                window.currentRegions.push(data);
            }
        } catch (err) {
            console.error('Error processing image segmentation:', err);
        }
    };
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
            const imgWidth = currentImage.naturalWidth;
            const imgHeight = currentImage.naturalHeight;
            console.log('Image dimensions:', imgWidth, 'x', imgHeight);
            console.log('Image position:', rect);
            
            // Create SVG with explicit dimensions and position
            const svg = d3.select(overlay)
                .append("svg")
                .attr("width", rect.width)
                .attr("height", rect.height)
                .attr("viewBox", `0 0 ${imgWidth} ${imgHeight}`)
                .style("position", "absolute")
                .style("top", "0")
                .style("left", "0")
                .style("width", "100%")
                .style("height", "100%")
                .style("pointer-events", "none");
            
            // Create tooltip div
            const tooltip = d3.select("body")
                .append("div")
                .attr("class", "segment-tooltip")
                .style("position", "absolute")
                .style("background-color", "rgba(0, 0, 0, 0.8)")
                .style("color", "white")
                .style("padding", "8px")
                .style("border-radius", "4px")
                .style("font-size", "14px")
                .style("pointer-events", "none")
                .style("z-index", "1000")
                .style("display", "none");
            
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