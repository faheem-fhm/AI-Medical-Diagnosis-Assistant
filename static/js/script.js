const uploadBox = document.getElementById("uploadBox");
const fileInput = document.getElementById("fileInput");
const uploadContent = document.getElementById("uploadContent");
const previewContainer = document.getElementById("previewContainer");
const previewImage = document.getElementById("previewImage");
const fileName = document.getElementById("fileName");
const removeBtn = document.getElementById("removeBtn");
const uploadForm = document.getElementById("uploadForm");

// =======================
// Show Image Preview
// =======================

function showPreview(file) {

    const reader = new FileReader();

    reader.onload = function (e) {
        previewImage.src = e.target.result;
    };

    reader.readAsDataURL(file);

    fileName.innerText = file.name;

    uploadContent.style.display = "none";
    previewContainer.style.display = "block";
}

// =======================
// File Selection
// =======================

fileInput.addEventListener("change", function () {

    if (fileInput.files.length > 0) {
        showPreview(fileInput.files[0]);
    }

});

// =======================
// Drag & Drop
// =======================

uploadBox.addEventListener("dragover", function (e) {

    e.preventDefault();
    uploadBox.classList.add("dragover");

});

uploadBox.addEventListener("dragleave", function () {

    uploadBox.classList.remove("dragover");

});

uploadBox.addEventListener("drop", function (e) {

    e.preventDefault();

    uploadBox.classList.remove("dragover");

    const file = e.dataTransfer.files[0];

    if (file) {

        fileInput.files = e.dataTransfer.files;
        showPreview(file);

    }

});

// =======================
// Remove Image
// =======================

removeBtn.addEventListener("click", function () {

    fileInput.value = "";
    previewImage.src = "";

    uploadContent.style.display = "block";
    previewContainer.style.display = "none";

});

const analyzeBtn = document.getElementById("analyzeBtn");

analyzeBtn.addEventListener("click", function () {

    if (fileInput.files.length === 0) {

        alert(
            "Please select a Chest X-ray image.\n\nSupported Formats:\n• JPG\n• JPEG\n• PNG"
        );

        return;
    }

    uploadForm.submit();

});