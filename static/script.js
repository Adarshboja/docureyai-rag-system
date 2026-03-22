



// ===== Tab Switching =====
document.getElementById('summarizer-tab').addEventListener('click', function () {
    this.classList.add('tab-active');
    document.getElementById('chatbot-tab').classList.remove('tab-active');
    document.getElementById('summarizer-section').classList.remove('hidden');
    document.getElementById('chatbot-section').classList.add('hidden');
});

document.getElementById('chatbot-tab').addEventListener('click', function () {
    this.classList.add('tab-active');
    document.getElementById('summarizer-tab').classList.remove('tab-active');
    document.getElementById('chatbot-section').classList.remove('hidden');
    document.getElementById('summarizer-section').classList.add('hidden');
});

// ===== PDF Upload for Summarizer =====
const pdfDropzone = document.getElementById('pdf-dropzone');
const pdfUpload = document.getElementById('pdf-upload');

pdfDropzone.addEventListener('click', () => pdfUpload.click());

pdfUpload.addEventListener('change', function (e) {
    if (e.target.files.length > 0) {
        const file = e.target.files[0];
        if (file.type === 'application/pdf') {
            updateDropzone(file);
        } else {
            alert('Please upload a PDF file');
            pdfUpload.value = '';
            resetDropzone();
        }
    }
});

pdfDropzone.addEventListener('drop', function (e) {
    preventDefaults(e);
    const file = e.dataTransfer.files[0];
    if (file && file.type === 'application/pdf') {
        updateDropzone(file);
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        pdfUpload.files = dataTransfer.files;
    } else {
        alert('Invalid PDF');
        resetDropzone();
    }
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function updateDropzone(file) {
    pdfDropzone.innerHTML = `
        <div class="text-center">
            <i class="fas fa-file-pdf text-4xl text-red-500 mb-2"></i>
            <p class="font-medium text-gray-800">${file.name}</p>
            <p class="text-sm text-gray-500 mt-1">${(file.size / (1024 * 1024)).toFixed(2)} MB</p>
        </div>`;
}

function resetDropzone() {
    pdfDropzone.innerHTML = `
        <i class="fas fa-cloud-upload-alt text-4xl text-blue-500 mb-3"></i>
        <p class="text-gray-600 mb-2">Drag & drop your PDF file here</p>
        <p class="text-sm text-gray-500 mb-4">or</p>
        <label for="pdf-upload" class="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg cursor-pointer transition-all">
            Browse Files
        </label>
        <p class="text-xs text-gray-500 mt-3">Max file size: 10MB</p>`;
}

// ===== Generate Summary =====
document.getElementById('generate-summary').addEventListener('click', function () {
    if (!pdfUpload.files.length) {
        alert('Please upload a PDF file first');
        return;
    }

    const button = this;
    const summaryLength = document.getElementById('summary-length').value;
    const summaryStyle = document.getElementById('summary-style').value;
    const summaryArea = document.getElementById('summary-results');

    button.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i> Processing...';
    button.disabled = true;

    const formData = new FormData();
    formData.append('pdf', pdfUpload.files[0]);
    formData.append('summary_length', summaryLength);
    formData.append('summary_style', summaryStyle);

    fetch('/result', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.summary) {
            summaryArea.innerHTML = data.summary.replace(/\n/g, '<br>');  // ✅ Your requested change
        } else {
            alert(data.error || "Unexpected error: No summary returned.");
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error generating summary.');
    })
    .finally(() => {
        button.innerHTML = '<i class="fas fa-magic mr-2"></i> Generate Summary';
        button.disabled = false;
    });
});

// ===== Chatbot PDF Upload =====
const chatPdfUpload = document.getElementById('chat-pdf-upload');
const chatbotFileName = document.getElementById('chatbot-file-name');

chatPdfUpload.addEventListener('change', function () {
    if (chatPdfUpload.files.length > 0) {
        const file = chatPdfUpload.files[0];
        if (file.type === 'application/pdf') {
            chatbotFileName.innerText = file.name;

            const formData = new FormData();
            formData.append('pdf', file);

            fetch('/result', {
                method: 'POST',
                body: formData
            })
            .then(res => res.json())
            .then(data => {
                addMessage("Document loaded! Ask anything about it.");
            })
            .catch(err => {
                console.error(err);
                addMessage("Failed to load the document.");
            });

        } else {
            alert('Please upload a PDF file');
            chatbotFileName.innerText = "No document loaded";
            chatPdfUpload.value = '';
        }
    }
});

// ===== Chatbot Interaction =====
const chatInput = document.getElementById('chat-input');
const sendMessageBtn = document.getElementById('send-message');
const chatMessages = document.getElementById('chat-messages');

function addMessage(content, isUser = false) {
    const div = document.createElement('div');
    div.className = `flex ${isUser ? 'justify-end' : 'justify-start'}`;
    div.innerHTML = `
        <div class="${isUser ? 'user-bubble' : 'bot-bubble'} px-4 py-3 max-w-[80%]">
            <p>${content}</p>
        </div>`;
    chatMessages.appendChild(div);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

sendMessageBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('keypress', function (e) {
    if (e.key === 'Enter') sendMessage();
});

function sendMessage() {
    const message = chatInput.value.trim();
    if (!message) return;
    addMessage(message, true);
    chatInput.value = '';

    const typing = document.createElement('div');
    typing.className = 'flex justify-start';
    typing.innerHTML = `
        <div class="bot-bubble px-4 py-3 max-w-[80%]">
            <div class="flex space-x-2">
                <div class="w-2 h-2 rounded-full bg-gray-400 animate-pulse"></div>
                <div class="w-2 h-2 rounded-full bg-gray-400 animate-pulse" style="animation-delay: 0.2s"></div>
                <div class="w-2 h-2 rounded-full bg-gray-400 animate-pulse" style="animation-delay: 0.4s"></div>
            </div>
        </div>`;
    chatMessages.appendChild(typing);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message })
    })
    .then(res => res.json())
    .then(data => {
        chatMessages.removeChild(typing);
       addMessage((data.answer || "Sorry, I couldn't understand that.").replace(/\n/g, '<br>'));

    })
    .catch(err => {
        chatMessages.removeChild(typing);
        addMessage("Error contacting server.");
        console.error(err);
    });
}
