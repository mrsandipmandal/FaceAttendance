document.addEventListener('DOMContentLoaded', function() {
    // Check if we're on the employee add/change page
    const employeeForm = document.getElementById('employee_form');
    if (!employeeForm) return;

    // Create modal HTML
    const modalHtml = `
    <div id="webcam-modal" style="display:none; position:fixed; z-index:9999; 
        left:0; top:0; width:100%; height:100%; 
        background-color:rgba(0,0,0,0.5); 
        align-items:center; justify-content:center;">
        <div style="background:white; padding:20px; border-radius:10px; width:500px;">
            <h3>Capture Employee Image</h3>
            <video id="webcam-video" width="100%" autoplay></video>
            <div style="margin-top:10px;">
                <button id="capture-btn" class="button">Capture</button>
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
        navigator.mediaDevices.getUserMedia({ video: true })
            .then(function(stream) {
                webcamStream = stream;
                videoElement.srcObject = stream;
                videoElement.play();
            })
            .catch(function(err) {
                console.error("Webcam access error:", err);
                alert("Unable to access webcam. Please check permissions.");
                // Revert to file upload
                document.querySelector('input[name="capture_method"][value="file"]').checked = true;
            });
    }

    // Stop webcam
    function stopWebcam() {
        if (webcamStream) {
            webcamStream.getTracks().forEach(track => track.stop());
            webcamStream = null;
            videoElement.srcObject = null;
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
