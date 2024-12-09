function validateForm() {
    const totalVariableInput = document.getElementById('totalVariable');
    const totalVariable = totalVariableInput.value;
    const trainInput = document.getElementById('train');
    const testInput = document.getElementById('test');
    const inputFile = document.getElementById('inputFile').files[0];
    const columnHeader = document.getElementById('columnHeader')

    if (totalVariable <= 0) {
        alert("Jumlah Variabel x harus lebih dari 0."); // Alert if invalid
        return false; // Prevent submission
    }

    if (!inputFile) {
        alert("Please upload a CSV or XLSX file."); // Alert if no file is selected
        return false; // Prevent submission
    }

    // Prepare form data to send to the backend
    const formData = new FormData();
    formData.append('file', inputFile);
    formData.append('train', trainInput.value);
    formData.append('test', testInput.value);
    formData.append('totalVariables', totalVariable);
    formData.append('columnHeader', columnHeader.checked ? 'on' : 'off');

    // Send data to the backend using AJAX
    $.ajax({
        url: '/save', // Update this URL to your backend endpoint
        type: 'POST',
        data: formData,
        contentType: false, // Tell jQuery not to set contentType
        processData: false, // Tell jQuery not to process the data
        success: function(response) {
            // Handle success response from server
            console.log(response);
            window.location.href = '/results'; // Redirect to result page
        },
        error: function(xhr, status, error) {
            // Handle error response from server
            console.error(error);
            alert("An error occurred while processing your request.");
        }
    });

    return false; // Prevent default form submission
}

function updateTestValue() {
    const trainInput = document.getElementById('train');
    const testInput = document.getElementById('test');
    const feedbackMessage = document.getElementById('feedbackMessage');
    let trainValue = parseFloat(trainInput.value) || 0; // Default to 0 if empty

    // Ensure trainValue does not exceed 100
    if (trainValue > 100) {
        trainValue = 100; // Cap the value at 100
        trainInput.value = trainValue; // Update the input field
        feedbackMessage.textContent = "Value tidak dapat lebih dari 100"; // Show feedback
    } else {
        feedbackMessage.textContent = ""; // Clear feedback
    }

    const testValue = 100 - trainValue;
    testInput.value = testValue >= 0 ? testValue : 0; // Ensure test value is not negative
}

function filterInput(event) {
    const trainInput = document.getElementById('train');

    // Get the current value
    const currentValue = trainInput.value;

    // Replace invalid characters (anything that is not a digit)
    const filteredValue = currentValue.replace(/[^0-9]/g, ''); // Allow only digits

    // Update the input field with the filtered value
    trainInput.value = filteredValue;

    // Call updateTestValue to recalculate the test value
    updateTestValue();
}

// Attach the input event listener to the train input field
document.getElementById('train').addEventListener('input', filterInput);

// Attach the form submission event listener
document.getElementById('trainModelForm').addEventListener('submit', validateForm);