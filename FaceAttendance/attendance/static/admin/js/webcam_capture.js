document.addEventListener('DOMContentLoaded', function() {
    // Check if we're on the employee add/change page
    const employeeForm = document.getElementById('employee_form');
    if (!employeeForm) return;

    // Create modal HTML
    const modalHtml = `
    <div id="webcam-modal" style="display:none; position:fixed; z-index:9999; 
        left:0; top:0; width:100%; height:100%; 
        background-color:rgba(0,0,0,0.5); 
        display:flex; align-items:center; justify-content:center;">
        <div style="background:white; padding:20px; border-radius:10px; width:500px;">
            <h3>Capture Employee Image</h3>
            <div id="webcam-error" style="color:red; display:none;">Webcam access not available</div>
            <video id="webcam-video" width="100%" autoplay style="display:none;"></video>
            <div style="margin-top:10px;">
                <button id="capture-btn" class="button" style="display:none;">Capture</button>
                <button id="close-modal-btn" class="button">Cancel</button>
            </div>
            <canvas id="capture-canvas" style="display:none;"></canvas>
        </div>
    </div>
    `;

    // Append modal to body
    const modalContainer = document.createElement('div');
    modalContainer.innerHTML = modalHtml;
    document.body.appendChild(modalContainer);

    // Get modal elements
    const modal = document.getElementById('webcam-modal');
    const videoElement = document.getElementById('webcam-video');
    const captureButton = document.getElementById('capture-btn');
    const closeModalButton = document.getElementById('close-modal-btn');
    const captureCanvas = document.getElementById('capture-canvas');
    const errorElement = document.getElementById('webcam-error');
    const imageInput = document.getElementById('id_image');

    // Find capture method radios
    const captureMethodRadios = document.querySelectorAll('input[name="capture_method"]');

    // Webcam stream reference
    let webcamStream = null;

    // Toggle webcam capture
    function toggleWebcamCapture() {
        const webcamRadio = document.querySelector('input[name="capture_method"][value="webcam"]');
        
        if (webcamRadio && webcamRadio.checked) {
            modal.style.display = 'flex';
            startWebcam();
        } else {
            modal.style.display = 'none';
            stopWebcam();
        }
    }

    // Start webcam
    function startWebcam() {
        // Reset UI
        videoElement.style.display = 'none';
        captureButton.style.display = 'none';
        errorElement.style.display = 'none';

        // Check for getUserMedia support
        const getUserMedia = (
            navigator.mediaDevices && navigator.mediaDevices.getUserMedia
        ) || 
        (navigator.getUserMedia) || 
        (navigator.webkitGetUserMedia) || 
        (navigator.mozGetUserMedia);

        if (!getUserMedia) {
            showWebcamError('Webcam access not supported by this browser');
            return;
        }

        // Attempt to get user media
        const constraints = { video: true };
        
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia(constraints)
                .then(function(stream) {
                    webcamStream = stream;
                    videoElement.srcObject = stream;
                    videoElement.play()
                        .then(() => {
                            videoElement.style.display = 'block';
                            captureButton.style.display = 'block';
                        })
                        .catch(err => {
                            showWebcamError('Error playing video: ' + err.message);
                        });
                })
                .catch(function(err) {
                    showWebcamError('Webcam access error: ' + err.message);
                });
        } else if (getUserMedia) {
            // Fallback for older browsers
            getUserMedia.call(navigator, constraints, 
                function(stream) {
                    webcamStream = stream;
                    videoElement.srcObject = stream;
                    videoElement.play();
                    videoElement.style.display = 'block';
                    captureButton.style.display = 'block';
                }, 
                function(err) {
                    showWebcamError('Webcam access error: ' + err.message);
                }
            );
        } else {
            showWebcamError('getUserMedia not supported in this browser');
        }
    }

    // Show webcam error
    function showWebcamError(message) {
        errorElement.textContent = message;
        errorElement.style.display = 'block';
        videoElement.style.display = 'none';
        captureButton.style.display = 'none';
        
        // Revert to file upload
        setTimeout(() => {
            document.querySelector('input[name="capture_method"][value="file"]').checked = true;
        }, 100);
    }

    // Stop webcam
    function stopWebcam() {
        if (webcamStream) {
            webcamStream.getTracks().forEach(track => track.stop());
            webcamStream = null;
            videoElement.srcObject = null;
            videoElement.style.display = 'none';
            captureButton.style.display = 'none';
        }
    }

    // Capture image
    captureButton.addEventListener('click', function() {
        captureCanvas.width = videoElement.videoWidth;
        captureCanvas.height = videoElement.videoHeight;
        
        const context = captureCanvas.getContext('2d');
        context.drawImage(videoElement, 0, 0);
        
        captureCanvas.toBlob(function(blob) {
            const file = new File([blob], 'webcam_capture.jpg', { type: 'image/jpeg' });
            
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            imageInput.files = dataTransfer.files;

            modal.style.display = 'none';
            stopWebcam();
        }, 'image/jpeg');
    });

    // Close modal
    closeModalButton.addEventListener('click', function() {
        modal.style.display = 'none';
        stopWebcam();
        document.querySelector('input[name="capture_method"][value="file"]').checked = true;
    });

    // Add event listeners to capture method radios
    captureMethodRadios.forEach(radio => {
        radio.addEventListener('change', toggleWebcamCapture);
    });
});
