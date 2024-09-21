const express = require('express');
const multer = require('multer');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');
const app = express();
const port = 3000;

// Set up storage for uploaded files
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, 'uploads/')
  },
  filename: function (req, file, cb) {
    cb(null, Date.now() + path.extname(file.originalname))
  }
});

const upload = multer({ storage: storage });

// Serve static files from the 'public' directory
app.use(express.static('public'));
app.use('/uploads', express.static('uploads'));

// Handle file upload
app.post('/upload', upload.single('video'), (req, res) => {
  if (req.file) {
    const videoPath = req.file.path;
    
    // Spawn the Python process to analyze the video
    const pythonProcess = spawn('python', ['backend/main.py', videoPath]);
    
    pythonProcess.stdout.on('data', (data) => {
      console.log(`Python stdout: ${data}`);
    });

    pythonProcess.stderr.on('data', (data) => {
      console.error(`Python stderr: ${data}`);
    });

    pythonProcess.on('close', (code) => {
      console.log(`Python process exited with code ${code}`);
    });

    res.json({ message: 'File uploaded and analysis started', videoPath: videoPath });
  } else {
    res.status(400).json({ message: 'No file uploaded' });
  }
});

// New endpoint to analyze video
app.get('/analyze/:videoPath', (req, res) => {
  const videoPath = path.join('uploads', req.params.videoPath);
  
  if (!fs.existsSync(videoPath)) {
    return res.status(404).json({ error: 'Video file not found' });
  }

  res.writeHead(200, {
    'Content-Type': 'text/plain',
    'Transfer-Encoding': 'chunked'
  });

  const pythonProcess = spawn('python', ['backend/main.py', videoPath]);
  
  pythonProcess.stdout.on('data', (data) => {
    res.write(data);
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`Python stderr: ${data}`);
    res.write(`Error: ${data}\n`);
  });

  pythonProcess.on('close', (code) => {
    console.log(`Python process exited with code ${code}`);
    res.end();
  });
});

// Start the server
app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});