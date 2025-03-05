// scripts/main.js

// DOM Elements
const chatHistory = document.getElementById('chat-history');
const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const newChatBtn = document.getElementById('new-chat-btn');
const uploadDocBtn = document.getElementById('upload-doc-btn');
const uploadModal = document.getElementById('upload-modal');
const closeModal = document.querySelector('.close');
const fileInput = document.getElementById('file-input');
const submitFileBtn = document.getElementById('submit-file');

// Open Upload Modal
uploadDocBtn.addEventListener('click', () => {
  uploadModal.style.display = 'flex';
});

// Close Upload Modal
closeModal.addEventListener('click', () => {
  uploadModal.style.display = 'none';
});

// Handle File Upload
submitFileBtn.addEventListener('click', () => {
  const file = fileInput.files[0];
  if (file) {
    // Simulate file upload (replace with actual API call)
    addMessage('bot', 'Document uploaded successfully!');
    uploadModal.style.display = 'none';
  } else {
    alert('Please select a file to upload.');
  }
});

// Handle Send Message
sendBtn.addEventListener('click', () => {
  const query = chatInput.value.trim();
  if (query) {
    addMessage('user', query);
    chatInput.value = '';
    // Simulate bot response (replace with actual API call)
    setTimeout(() => {
      addMessage('bot', 'This is a response to your query.');
    }, 1000);
  }
});

// Add Message to Chat
function addMessage(sender, text) {
  const messageDiv = document.createElement('div');
  messageDiv.classList.add('message', sender);
  messageDiv.textContent = text;
  chatMessages.appendChild(messageDiv);
  chatMessages.scrollTop = chatMessages.scrollHeight; // Auto-scroll to bottom
}

// Handle New Chat
newChatBtn.addEventListener('click', () => {
  chatMessages.innerHTML = ''; // Clear chat messages
  addMessage('bot', 'Welcome to a new chat!');
});

// Add Chat History Items (Example)
const historyItems = ['Chat 1', 'Chat 2', 'Chat 3'];
historyItems.forEach((item) => {
  const li = document.createElement('li');
  li.textContent = item;
  chatHistory.appendChild(li);
});