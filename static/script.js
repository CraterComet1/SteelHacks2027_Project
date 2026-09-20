```javascript
const frame = document.getElementById("detectionFrame");

// Ask for the newest detection frame repeatedly
setInterval(() => {
    frame.src = "/frame?t=" + Date.now();
}, 10500);
```
