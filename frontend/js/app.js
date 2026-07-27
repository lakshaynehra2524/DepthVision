const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("imageInput");
const fileNamePreview = document.getElementById("fileNamePreview");
const previewThumb = document.getElementById("previewThumb");
const analyzeBtn = document.getElementById("analyzeBtn");
const loading = document.getElementById("loading");
const results = document.getElementById("results");
const featuresSection = document.getElementById("featuresSection");

function showPreview(file) {

    if (!file) return;

    fileNamePreview.textContent = file.name;
    fileNamePreview.classList.remove("hidden");

    const reader = new FileReader();
    reader.onload = (e) => {
        previewThumb.src = e.target.result;
        previewThumb.classList.remove("hidden");
    };
    reader.readAsDataURL(file);
}

fileInput.addEventListener("change", () => {
    showPreview(fileInput.files[0]);
});

["dragenter", "dragover"].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add("drag-over");
    });
});

["dragleave", "drop"].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove("drag-over");
    });
});

dropZone.addEventListener("drop", (e) => {
    const droppedFiles = e.dataTransfer.files;
    if (droppedFiles.length) {
        fileInput.files = droppedFiles;
        showPreview(droppedFiles[0]);
    }
});

function buildStaggeredRows(rows) {
    return rows
        .map((rowHTML, index) => rowHTML.replace(
            "<tr>",
            `<tr style="animation-delay:${index * 0.05}s">`
        ))
        .join("");
}

analyzeBtn.addEventListener(
    "click",
    async () => {

        const file = fileInput.files[0];

        if (!file) {
            alert("Please select an image.");
            return;
        }

        loading.classList.remove("hidden");
        results.classList.add("hidden");
        featuresSection.style.display = "none";
        analyzeBtn.disabled = true;

        const formData = new FormData();
        formData.append("file", file);

        try {

            const response = await fetch(
                "/predict",
                {
                    method: "POST",
                    body: formData
                }
            );

            if (!response.ok) {
                throw new Error(`Server responded with ${response.status}`);
            }

            const data = await response.json();

            // Display image

            const imageElement = document.getElementById("resultImage");
            imageElement.src = "data:image/jpeg;base64," + data.annotated_image;

            // Risk Table

            const riskTable = document.getElementById("riskTable");

            const riskRows = data.high_risk_pairs.map(pair => `
                    <tr>
                        <td>${pair.Object_1}</td>
                        <td>${pair.Object_2}</td>
                        <td>${Number(pair.Distance).toFixed(2)}</td>
                    </tr>
                    `);

            riskTable.innerHTML = `
            <tr>
                <th>Object 1</th>
                <th>Object 2</th>
                <th>Distance</th>
            </tr>
            ` + buildStaggeredRows(riskRows);

            // Objects Table

            const objectsTable = document.getElementById("objectsTable");

            const objectRows = data.objects.map(obj => `
                    <tr>
                        <td>${obj.class}</td>
                        <td>${obj.confidence}</td>
                        <td>${obj.depth}</td>
                    </tr>
                    `);

            objectsTable.innerHTML = `
            <tr>
                <th>Class</th>
                <th>Confidence</th>
                <th>Depth</th>
            </tr>
            ` + buildStaggeredRows(objectRows);

            loading.classList.add("hidden");
            results.classList.remove("hidden");

        }

        catch (error) {

            console.error(error);

            loading.classList.add("hidden");

            alert("Error connecting to server.");
        }

        finally {
            analyzeBtn.disabled = false;
        }
    }
);
