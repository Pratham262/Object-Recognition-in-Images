document.getElementById('classifyBtn').addEventListener('click', function () {
    let fileInput = document.getElementById('imageUpload');
    let resultDisplay = document.getElementById('result');
    let classifyBtn = document.getElementById('classifyBtn');

    if (fileInput.files.length === 0) {
        alert("Please upload an image first.");
        return;
    }

    let formData = new FormData();
    formData.append("image", fileInput.files[0]);

    classifyBtn.disabled = true;
    classifyBtn.textContent = "Processing...";
    resultDisplay.textContent = "Analyzing image...";

    fetch("http://127.0.0.1:5000/classify", { 
        method: "POST",
        body: formData
    })
    .then(response => response.json())  // Ensure response is converted to JSON
    .then(data => {
        if (data.prediction) {  // Check if "prediction" exists in response
            resultDisplay.textContent = `Prediction: ${data.prediction}`;
        } else {
            throw new Error("Invalid response format");
        }
    })
    .catch(error => {
        console.error("Error:", error);
        resultDisplay.textContent = "Error: Could not classify image.";
    })
    .finally(() => {
        classifyBtn.disabled = false;
        classifyBtn.textContent = "Classify";
    });
});
