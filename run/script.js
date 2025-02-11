document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const modelTypeSelect = document.getElementById('modelTypeSelect');
    const modelSelect = document.getElementById('modelSelect');
    const digitInput = document.getElementById('digitInput');
    const generateBtn = document.getElementById('generateBtn');
    const errorDiv = document.getElementById('error');
    const loadingDiv = document.getElementById('loading');
    const timingInfo = document.getElementById('timingInfo');
    const audioSection = document.getElementById('audioSection');
    const spectrogramSection = document.getElementById('spectrogramSection');
    
    // Time display elements
    const indicesTime = document.getElementById('indicesTime');
    const spectrogramTime = document.getElementById('spectrogramTime');
    const audioTime = document.getElementById('audioTime');
    const totalTime = document.getElementById('totalTime');
    
    // Audio and image elements
    const audioPlayer = document.getElementById('audioPlayer');
    const spectrogramImage = document.getElementById('spectrogramImage');

    // Input validation and button enable/disable
    function validateInputs() {
        const modelType = modelTypeSelect.value;
        const model = modelSelect.value;
        const digit = digitInput.value;
        
        let isValid = modelType !== '' && model !== '';
        
        if (modelType === 'conditional') {
            isValid = isValid && digit !== '' && 
                     !isNaN(digit) && 
                     parseInt(digit) >= 0 && 
                     parseInt(digit) <= 9;
        }
        
        generateBtn.disabled = !isValid;
        return isValid;
    }

    // Reset display elements
    function resetDisplay() {
        errorDiv.style.display = 'none';
        loadingDiv.style.display = 'none';
        timingInfo.style.display = 'none';
        audioSection.style.display = 'none';
        spectrogramSection.style.display = 'none';
    }

    // Model type selection handler
    modelTypeSelect.addEventListener('change', function() {
        const modelType = this.value;
        modelSelect.disabled = modelType === '';
        digitInput.disabled = modelType !== 'conditional';
        
        if (modelType !== 'conditional') {
            digitInput.value = '';
        }
        
        validateInputs();
    });

    // Model selection handler
    modelSelect.addEventListener('change', validateInputs);

    // Digit input handler
    digitInput.addEventListener('input', function(e) {
        // Only allow single digits
        if (e.target.value.length > 0) {
            const value = e.target.value.slice(-1);
            if (!isNaN(value) && parseInt(value) >= 0 && parseInt(value) <= 9) {
                e.target.value = value;
            } else {
                e.target.value = '';
            }
        }
        validateInputs();
    });

    // Prevent form submission on Enter key
    digitInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            e.preventDefault();
            if (validateInputs()) {
                generateBtn.click();
            }
        }
    });

    // Generate button handler
    generateBtn.addEventListener('click', async function(e) {
        e.preventDefault();
        
        if (!validateInputs()) return;

        resetDisplay();
        loadingDiv.style.display = 'block';
        
        // Disable all inputs during generation
        modelTypeSelect.disabled = true;
        modelSelect.disabled = true;
        digitInput.disabled = true;
        generateBtn.disabled = true;

        try {
            const response = await fetch('http://localhost:8956/api/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    digit: modelTypeSelect.value === 'conditional' ? parseInt(digitInput.value) : null,
                    model: modelSelect.value,
                    model_type: modelTypeSelect.value
                }),
            });

            if (!response.ok) {
                throw new Error('Generation failed');
            }

            const data = await response.json();

            // Display timing information
            indicesTime.textContent = `${data.times.indices.toFixed(2)}s`;
            spectrogramTime.textContent = `${data.times.spectrogram.toFixed(2)}s`;
            audioTime.textContent = `${data.times.audio.toFixed(2)}s`;
            totalTime.textContent = `${data.times.total.toFixed(2)}s`;
            timingInfo.style.display = 'block';

            // Update audio player
            audioPlayer.src = `http://localhost:8956${data.audioUrl}`;
            audioSection.style.display = 'block';

            // Update spectrogram image
            spectrogramImage.src = `http://localhost:8956${data.spectrogramUrl}`;
            spectrogramSection.style.display = 'block';

        } catch (err) {
            console.error('Error:', err);
            errorDiv.textContent = 'Failed to generate audio. Please try again.';
            errorDiv.style.display = 'block';
        } finally {
            // Re-enable inputs
            modelTypeSelect.disabled = false;
            modelSelect.disabled = modelTypeSelect.value === '';
            digitInput.disabled = modelTypeSelect.value !== 'conditional';
            validateInputs();
            loadingDiv.style.display = 'none';
        }
    });
});