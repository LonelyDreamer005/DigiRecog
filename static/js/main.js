/**
 * Smart Handwritten Digit Recognition - Client Logic with In-Browser 5-Finger Air Canvas
 */

document.addEventListener("DOMContentLoaded", () => {
    // --- Navigation Tabs ---
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetId = btn.getAttribute("data-tab");
            tabBtns.forEach(b => b.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));

            btn.classList.add("active");
            document.getElementById(targetId).classList.add("active");
        });
    });

    // --- Input Sub Tabs ---
    const subTabBtns = document.querySelectorAll(".sub-tab-btn");
    const inputViews = {
        "air-canvas": document.getElementById("air-canvas-input-view"),
        "canvas": document.getElementById("canvas-input-view"),
        "upload": document.getElementById("upload-input-view")
    };

    subTabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const mode = btn.getAttribute("data-input");
            subTabBtns.forEach(b => b.classList.remove("active"));
            Object.values(inputViews).forEach(v => {
                if (v) v.classList.remove("active");
            });

            btn.classList.add("active");
            if (inputViews[mode]) {
                inputViews[mode].classList.add("active");
            }

            if (mode === "canvas") {
                initTouchCanvas();
            }
        });
    });

    // --- Model Switcher ---
    const modelRadios = document.querySelectorAll('input[name="model_choice"]');
    const softmaxCard = document.getElementById("softmaxResultCard");
    const cnnCard = document.getElementById("cnnResultCard");

    function updateModelVisibility() {
        const choice = document.querySelector('input[name="model_choice"]:checked').value;
        if (choice === "both") {
            softmaxCard.style.display = "flex";
            cnnCard.style.display = "flex";
        } else if (choice === "softmax") {
            softmaxCard.style.display = "flex";
            cnnCard.style.display = "none";
        } else if (choice === "cnn") {
            softmaxCard.style.display = "none";
            cnnCard.style.display = "flex";
        }
    }
    modelRadios.forEach(r => r.addEventListener("change", updateModelVisibility));

    // =========================================================================
    // 🖐️ 5-FINGER IN-BROWSER AIR CANVAS LOGIC (MediaPipe Hands)
    // =========================================================================
    const airVideo = document.getElementById("airVideo");
    const airCanvas = document.getElementById("airDrawCanvas");
    const airCtx = airCanvas.getContext("2d");
    const toggleAirBtn = document.getElementById("toggleAirCameraBtn");
    const predictAirBtn = document.getElementById("predictAirBtn");
    const clearAirBtn = document.getElementById("clearAirBtn");
    const airBadge = document.getElementById("airGestureBadge");
    const airBadgeText = document.getElementById("airGestureText");

    let camera = null;
    let handsDetector = null;
    let isAirRunning = false;
    let prevAirX = null, prevAirY = null;
    let airHasDrawn = false;
    let isAutoPredictDebounced = false;

    // Initialize blank dark air canvas
    function resetAirCanvas() {
        airCtx.clearRect(0, 0, airCanvas.width, airCanvas.height);
        prevAirX = null;
        prevAirY = null;
        airHasDrawn = false;
        predictAirBtn.disabled = true;
    }

    function setGestureBadge(text, color = "#10b981", icon = "🖐️") {
        airBadgeText.textContent = text;
        airBadge.style.borderColor = color;
        airBadge.querySelector(".gesture-icon").textContent = icon;
    }

    // MediaPipe Hand Landmark processing
    function onHandResults(results) {
        if (!isAirRunning) return;

        if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
            const landmarks = results.multiHandLandmarks[0];

            // Invert X coordinate for mirror view
            const getPt = (idx) => ({
                x: (1 - landmarks[idx].x) * airCanvas.width,
                y: landmarks[idx].y * airCanvas.height
            });

            const indexTip = getPt(8);
            const indexPip = getPt(6);
            const middleTip = getPt(12);
            const middlePip = getPt(10);
            const ringTip = getPt(16);
            const ringPip = getPt(14);
            const pinkyTip = getPt(20);
            const pinkyPip = getPt(18);

            // Determine if fingers are UP (Tip Y is lower/higher than PIP)
            const isIndexUp = indexTip.y < indexPip.y;
            const isMiddleUp = middleTip.y < middlePip.y;
            const isRingUp = ringTip.y < ringPip.y;
            const isPinkyUp = pinkyTip.y < pinkyPip.y;

            const totalUp = (isIndexUp ? 1 : 0) + (isMiddleUp ? 1 : 0) + (isRingUp ? 1 : 0) + (isPinkyUp ? 1 : 0);

            // --- GESTURE 1: DRAW MODE (Index UP only: ☝️) ---
            if (isIndexUp && !isMiddleUp && !isRingUp && !isPinkyUp) {
                setGestureBadge("DRAWING ✍️", "#10b981", "☝️");

                if (prevAirX !== null && prevAirY !== null) {
                    airCtx.lineWidth = 22;
                    airCtx.lineCap = "round";
                    airCtx.lineJoin = "round";
                    airCtx.strokeStyle = "#FFFFFF";
                    airCtx.shadowColor = "#38bdf8";
                    airCtx.shadowBlur = 6;

                    airCtx.beginPath();
                    airCtx.moveTo(prevAirX, prevAirY);
                    airCtx.lineTo(indexTip.x, indexTip.y);
                    airCtx.stroke();
                }

                prevAirX = indexTip.x;
                prevAirY = indexTip.y;
                airHasDrawn = true;
                predictAirBtn.disabled = false;
            }
            // --- GESTURE 2: HOVER MODE (Index + Middle UP: ✌️) ---
            else if (isIndexUp && isMiddleUp && !isRingUp && !isPinkyUp) {
                setGestureBadge("HOVER / LIFT PEN ✌️", "#f59e0b", "✌️");
                prevAirX = null;
                prevAirY = null;
            }
            // --- GESTURE 3: CLEAR CANVAS (4 or 5 Fingers OPEN: ✋) ---
            else if (totalUp >= 3) {
                setGestureBadge("CLEAR CANVAS ✋", "#ef4444", "✋");
                resetAirCanvas();
                resetResults();
            }
            // --- GESTURE 4: PREDICT DIGIT (Fist / 0 Fingers: ✊) ---
            else if (totalUp === 0 && airHasDrawn && !isAutoPredictDebounced) {
                setGestureBadge("PREDICTING 🎯", "#8b5cf6", "✊");
                prevAirX = null;
                prevAirY = null;
                isAutoPredictDebounced = true;
                triggerAirPrediction();
                setTimeout(() => { isAutoPredictDebounced = false; }, 1500);
            }
            else {
                prevAirX = null;
                prevAirY = null;
            }

        } else {
            setGestureBadge("Show Hand to Draw", "#64748b", "🖐️");
            prevAirX = null;
            prevAirY = null;
        }
    }

    async function initAirMediaPipe() {
        if (typeof Hands === "undefined") {
            alert("MediaPipe Hands library is loading. Please wait 2 seconds and try again.");
            return;
        }

        handsDetector = new Hands({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
        });

        handsDetector.setOptions({
            maxNumHands: 1,
            modelComplexity: 1,
            minDetectionConfidence: 0.7,
            minTrackingConfidence: 0.7
        });

        handsDetector.onResults(onHandResults);

        camera = new Camera(airVideo, {
            onFrame: async () => {
                if (isAirRunning && handsDetector) {
                    await handsDetector.send({ image: airVideo });
                }
            },
            width: 480,
            height: 360
        });

        await camera.start();
        isAirRunning = true;
        toggleAirBtn.textContent = "Stop Air Camera";
        toggleAirBtn.className = "btn btn-secondary";
        setGestureBadge("Ready: Raise Index Finger to Draw", "#10b981", "☝️");
    }

    toggleAirBtn.addEventListener("click", async () => {
        if (isAirRunning) {
            if (camera) {
                camera.stop();
                camera = null;
            }
            isAirRunning = false;
            toggleAirBtn.textContent = "Start Air Camera";
            toggleAirBtn.className = "btn btn-primary";
            setGestureBadge("Camera Stopped", "#64748b", "📷");
        } else {
            try {
                await initAirMediaPipe();
            } catch (err) {
                alert("Could not access camera: " + err.message);
            }
        }
    });

    clearAirBtn.addEventListener("click", () => {
        resetAirCanvas();
        resetResults();
    });

    function triggerAirPrediction() {
        if (!airHasDrawn) return;

        // Render black background + white strokes to extract clean image
        const exportCanvas = document.createElement("canvas");
        exportCanvas.width = airCanvas.width;
        exportCanvas.height = airCanvas.height;
        const expCtx = exportCanvas.getContext("2d");

        expCtx.fillStyle = "#000000";
        expCtx.fillRect(0, 0, exportCanvas.width, exportCanvas.height);
        expCtx.drawImage(airCanvas, 0, 0);

        const dataUrl = exportCanvas.toDataURL("image/png");
        sendPredictionRequest(dataUrl);
    }

    predictAirBtn.addEventListener("click", triggerAirPrediction);

    // =========================================================================
    // ✍️ TOUCH / MOUSE DRAWING CANVAS LOGIC
    // =========================================================================
    const canvas = document.getElementById("drawCanvas");
    const ctx = canvas.getContext("2d");
    let isDrawing = false;
    let hasDrawn = false;

    function initTouchCanvas() {
        ctx.fillStyle = "#000000";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.lineWidth = 20;
        ctx.lineCap = "round";
        ctx.lineJoin = "round";
        ctx.strokeStyle = "#FFFFFF";
    }
    initTouchCanvas();

    function getCanvasPos(e) {
        const rect = canvas.getBoundingClientRect();
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        return {
            x: (clientX - rect.left) * (canvas.width / rect.width),
            y: (clientY - rect.top) * (canvas.height / rect.height)
        };
    }

    function startDraw(e) {
        e.preventDefault();
        isDrawing = true;
        hasDrawn = true;
        const pos = getCanvasPos(e);
        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);
    }

    function draw(e) {
        if (!isDrawing) return;
        e.preventDefault();
        const pos = getCanvasPos(e);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
    }

    function stopDraw(e) {
        if (!isDrawing) return;
        e.preventDefault();
        isDrawing = false;
        ctx.closePath();
    }

    canvas.addEventListener("mousedown", startDraw);
    canvas.addEventListener("mousemove", draw);
    canvas.addEventListener("mouseup", stopDraw);
    canvas.addEventListener("mouseleave", stopDraw);

    canvas.addEventListener("touchstart", startDraw, { passive: false });
    canvas.addEventListener("touchmove", draw, { passive: false });
    canvas.addEventListener("touchend", stopDraw, { passive: false });

    document.getElementById("clearCanvasBtn").addEventListener("click", () => {
        initTouchCanvas();
        hasDrawn = false;
        resetResults();
    });

    document.getElementById("predictCanvasBtn").addEventListener("click", () => {
        if (!hasDrawn) return;
        const dataUrl = canvas.toDataURL("image/png");
        sendPredictionRequest(dataUrl);
    });

    // =========================================================================
    // 📁 FILE UPLOAD LOGIC
    // =========================================================================
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const uploadPreview = document.getElementById("uploadPreview");
    const predictUploadBtn = document.getElementById("predictUploadBtn");
    let uploadedDataUrl = null;

    dropZone.addEventListener("click", () => fileInput.click());

    fileInput.addEventListener("change", (e) => {
        const file = e.target.files[0];
        if (file) handleUploadFile(file);
    });

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.style.borderColor = "var(--accent-blue)";
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.style.borderColor = "var(--border-color)";
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.style.borderColor = "var(--border-color)";
        if (e.dataTransfer.files.length > 0) {
            handleUploadFile(e.dataTransfer.files[0]);
        }
    });

    function handleUploadFile(file) {
        const reader = new FileReader();
        reader.onload = (ev) => {
            uploadedDataUrl = ev.target.result;
            uploadPreview.src = uploadedDataUrl;
            uploadPreview.style.display = "block";
            predictUploadBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    predictUploadBtn.addEventListener("click", () => {
        if (uploadedDataUrl) {
            sendPredictionRequest(uploadedDataUrl);
        }
    });

    // =========================================================================
    // 🧠 BACKEND PREDICTION API (Softmax & CNN)
    // =========================================================================
    async function sendPredictionRequest(imageDataUrl) {
        const modelChoice = document.querySelector('input[name="model_choice"]:checked').value;
        const t0 = performance.now();

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    image: imageDataUrl,
                    model: modelChoice
                })
            });

            const data = await response.json();
            const latency = (performance.now() - t0).toFixed(1);
            document.getElementById("inferTimeBadge").textContent = `Latency: ${latency}ms`;

            if (data.error) {
                alert("Prediction error: " + data.error);
                return;
            }

            // Update preprocessed 28x28 image view
            if (data.preprocessed_image) {
                document.getElementById("preprocessedImg").src = data.preprocessed_image;
            }

            // Render Softmax & CNN results
            if (data.predictions.softmax) {
                renderModelOutput("softmax", data.predictions.softmax);
            }
            if (data.predictions.cnn) {
                renderModelOutput("cnn", data.predictions.cnn);
            }

        } catch (err) {
            console.error("API Error:", err);
            alert("Failed to communicate with prediction backend.");
        }
    }

    function renderModelOutput(prefix, predData) {
        const digitEl = document.getElementById(`${prefix}Digit`);
        const confEl = document.getElementById(`${prefix}Conf`);
        const probsEl = document.getElementById(`${prefix}Probs`);

        digitEl.textContent = predData.predicted_digit;
        confEl.textContent = `${predData.confidence}%`;

        // Render 10 probability bars
        probsEl.innerHTML = "";
        predData.probabilities.forEach((prob, digit) => {
            const isPredicted = digit === predData.predicted_digit;
            const row = document.createElement("div");
            row.className = `prob-row ${isPredicted ? "active" : ""}`;

            const fillClass = prefix === "cnn" ? "prob-bar-fill cnn-fill" : "prob-bar-fill";
            row.innerHTML = `
                <span class="prob-digit">${digit}</span>
                <div class="prob-bar-bg">
                    <div class="${fillClass}" style="width: ${prob}%"></div>
                </div>
                <span class="prob-percent">${prob.toFixed(1)}%</span>
            `;
            probsEl.appendChild(row);
        });
    }

    function resetResults() {
        document.getElementById("softmaxDigit").textContent = "-";
        document.getElementById("softmaxConf").textContent = "0.0%";
        document.getElementById("softmaxProbs").innerHTML = "";

        document.getElementById("cnnDigit").textContent = "-";
        document.getElementById("cnnConf").textContent = "0.0%";
        document.getElementById("cnnProbs").innerHTML = "";
    }
});
