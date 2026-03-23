const express = require('express');
const cors = require('cors');
const multer = require('multer');
const pdfParse = require('pdf-parse');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 10000;

// Paths
const STATIC_DIR = path.join(__dirname, 'static');
const TEMPLATE_DIR = path.join(__dirname, 'templates');
const UPLOAD_DIR = path.join(__dirname, 'uploads');

// Ensure upload directory exists
if (!fs.existsSync(UPLOAD_DIR)) {
    fs.mkdirSync(UPLOAD_DIR, { recursive: true });
}

// Middlewares
app.use(cors());
app.use(express.json());
app.use('/static', express.static(STATIC_DIR, { fallthrough: true }));

// Multer setup for in-memory PDF upload
const upload = multer({
    storage: multer.memoryStorage(),
    limits: { fileSize: 10 * 1024 * 1024 }, // 10MB
    fileFilter: (req, file, cb) => {
        if (file.mimetype !== 'application/pdf') {
            return cb(new Error('Only PDF files are allowed'));
        }
        cb(null, true);
    }
});

// In-memory chat context
let chatContext = { text: '' };

// Utility: basic heuristic summarizer
function summarizeText(text, length = 'medium', style = 'concise') {
    const clean = text.replace(/\s+/g, ' ').trim();
    if (!clean) return 'No text detected in this PDF.';

    const sentences = clean.split(/(?<=[.!?])\s+/).filter(Boolean);
    const lengthMap = { short: 2, medium: 4, detailed: 8 };
    const take = lengthMap[length] || lengthMap.medium;
    const selected = sentences.slice(0, take);

    if (style === 'bullet') {
        return selected.map(s => `• ${s.trim()}`).join('\n');
    }
    if (style === 'paragraph') {
        return selected.join(' ');
    }
    return selected.join(' ');
}

// Utility: lightweight Q&A using keyword matching
function answerQuestion(text, question) {
    const clean = text.replace(/\s+/g, ' ').trim();
    if (!clean) return 'Please upload a PDF before asking questions.';

    const sentences = clean.split(/(?<=[.!?])\s+/).filter(Boolean);
    const terms = question.toLowerCase().split(/\W+/).filter(t => t.length > 3);

    const matches = sentences.filter(s => {
        const lower = s.toLowerCase();
        return terms.some(t => lower.includes(t));
    });

    if (matches.length) {
        return matches.slice(0, 3).join(' ');
    }

    return 'I could not find a specific answer in the document, but here is a quick recap:\n' +
           summarizeText(clean, 'short', 'paragraph');
}

// Routes
app.get('/', (req, res) => {
    res.sendFile(path.join(TEMPLATE_DIR, 'index.html'));
});

app.post('/result', upload.single('pdf'), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: 'PDF file is required' });
        }

        const { summary_length = 'medium', summary_style = 'concise' } = req.body;
        const data = await pdfParse(req.file.buffer);

        chatContext.text = data.text || '';
        const summary = summarizeText(chatContext.text, summary_length, summary_style);

        res.json({ summary });
    } catch (err) {
        console.error(err);
        res.status(500).json({ error: err.message || 'Failed to generate summary' });
    }
});

app.post('/chat', express.json(), (req, res) => {
    try {
        const message = (req.body && req.body.message) || '';
        if (!message) {
            return res.status(400).json({ answer: 'Message is required.' });
        }

        const answer = answerQuestion(chatContext.text, message);
        res.json({ answer });
    } catch (err) {
        console.error(err);
        res.status(500).json({ answer: 'Error: ' + (err.message || 'Unable to answer.') });
    }
});

// Fallback to serve index for any non-API route
app.get('*', (req, res, next) => {
    if (req.path.startsWith('/result') || req.path.startsWith('/chat')) return next();
    res.sendFile(path.join(TEMPLATE_DIR, 'index.html'));
});

app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
