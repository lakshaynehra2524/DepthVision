document
    .getElementById(
        "analyzeBtn"
    )
    .addEventListener(
        "click",
        async () => {

            const fileInput =
                document.getElementById(
                    "imageInput"
                );

            const file =
                fileInput.files[0];

            if (!file) {

                alert(
                    "Please select an image."
                );

                return;
            }

            const loading =
                document.getElementById(
                    "loading"
                );

            const results =
                document.getElementById(
                    "results"
                );

            const featuresSection =
                document.getElementById(
                    "featuresSection"
                );

            loading.classList.remove(
                "hidden"
            );

            results.classList.add(
                "hidden"
            );
            featuresSection.style.display =
                "none";

            const formData =
                new FormData();

            formData.append(
                "file",
                file
            );

            try {

                const response =
                    await fetch(
                        "/predict",
                        {
                            method: "POST",
                            body: formData
                        }
                    );

                const data =
                    await response.json();

                // Display image

                const imageElement =
                    document.getElementById(
                        "resultImage"
                    );

                imageElement.src =
                    "data:image/jpeg;base64," +
                    data.annotated_image;

                // Risk Table

                const riskTable =
                    document.getElementById(
                        "riskTable"
                    );

                riskTable.innerHTML = "";


                let riskHTML = `
            <tr>
                <th>Object 1</th>
                <th>Object 2</th>
                <th>Distance</th>
            </tr>
            `;

                data.high_risk_pairs.forEach(
                    pair => {

                        riskHTML += `
                    <tr>
                        <td>${pair.Object_1}</td>
                        <td>${pair.Object_2}</td>
                        <td>${Number(pair.Distance).toFixed(2)}</td>
                    </tr>
                    `;
                    }
                );

                riskTable.innerHTML =
                    riskHTML;

                // Objects Table

                const objectsTable =
                    document.getElementById(
                        "objectsTable"
                    );

                objectsTable.innerHTML = "";

                let objectsHTML = `
            <tr>
                <th>Class</th>
                <th>Confidence</th>
                <th>Depth</th>
            </tr>
            `;

                data.objects.forEach(
                    obj => {

                        objectsHTML += `
                    <tr>
                        <td>${obj.class}</td>
                        <td>${obj.confidence}</td>
                        <td>${obj.depth}</td>
                    </tr>
                    `;
                    }
                );

                objectsTable.innerHTML =
                    objectsHTML;

                loading.classList.add(
                    "hidden"
                );

                results.classList.remove(
                    "hidden"
                );

            }

            catch (error) {

                console.error(
                    error
                );

                loading.classList.add(
                    "hidden"
                );

                alert(
                    "Error connecting to server."
                );
            }
        }
    );