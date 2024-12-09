// Get the number of variables from local storage
const numVariables = localStorage.getItem('numVariables');

function showLagrangePopup() {
    showTab('tab7');
}

function hideLagrangePopup() {
    document.getElementById('lagrangePopup').style.display = 'none';
}

function showKernelPopup() {
    document.getElementById('kernelPopup').style.display = 'block';
    document.getElementById('helpPopup').style.display = 'none';
    
}

function hideKernelPopup() {
    document.getElementById('kernelPopup').style.display = 'none';
}

function showHelpPopup(){
    document.getElementById('helpPopup').style.display = 'block';
    document.getElementById('kernelPopup').style.display = 'none';
}

function hideHelpPopup(){
    document.getElementById('helpPopup').style.display = 'none';
}

// Function untuk menunjukkan tab spesifik
function showTab(tabId) {
    // Hide all tab contents
    const contents = document.querySelectorAll('.tab-content');
    contents.forEach(content => {
        content.classList.remove('active');
        content.style.display = 'none';
    });

    // Remove active class from all tabs
    const tabs = document.querySelectorAll('.tab');
    tabs.forEach(tab => {
        tab.classList.remove('active');
    });

    // Set active tab
    const activeTab = document.querySelector(`.tab[onclick="showTab('${tabId}')"]`);
    if (activeTab) {
        activeTab.classList.add('active');
    }

    // Menampilkan content
    const activeContent = document.getElementById(tabId);
    activeContent.classList.add('active');
    activeContent.style.display = 'block';
}

// Membuat tab1 sebagai default
document.addEventListener('DOMContentLoaded', () => {
    showTab('tab1');
});
