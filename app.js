
const fileInput = document.querySelector('input[type="file"]');
const uploadButton = document.querySelector('#upload-button');
const output = document.querySelector('#output');
const loadingIndicator = document.querySelector('#loading-indicator');

uploadButton.addEventListener('click', async (e) => {
    e.preventDefault();
    const file = fileInput.files[0];

    //Validate the file
    if (!file) {
        return alert('Please select an image file.');
    }

    if (!file.type.match(/image\/*/)) {
        return alert('Invalid image file');
    }

    if (file.size > 5000000) {
        return alert('File size too big');
    }

    // Show the loading indicator
    loadingIndicator.style.display = 'block';

    // Create a new FormData object to send the file data to the server
    const formData = new FormData();
    formData.append('image', file);

    try {
        // Use the fetch API to send the image data to the server
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        // Parse the response as json
        const data = await response.json();

        // Update the UI with the predicted emotion
        output.innerHTML = `<p>Emotion detected: ${data.emotion}</p>`;
    } catch (error) {
        // Handle any errors that may occur
        console.error(error);
        alert('An error occurred while analyzing the image. Please try again.');
    } finally {
        // Hide the loading indicator
        loadingIndicator.style.display = 'none';
    }
});
