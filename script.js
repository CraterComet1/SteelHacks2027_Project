```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Video Frame Capture</title>

    <style>
        .container {
            display: flex;
            gap: 20px;
        }

        .box {
            width: 500px;
            height: 350px;
            border: 2px solid black;
        }

        video, canvas {
            width: 100%;
            height: 100%;
            object-fit: contain;
        }

        button {
            margin-top: 20px;
            padding: 10px 20px;
            font-size: 16px;
        }
    </style>
</head>

<body>

    <h1>Video Frame Capture</h1>

    <div class="container">

        <!-- Left: Video -->
        <div class="box">
            <!-- Live camera --><div class="box"> <video id="video" autoplay playsinline></video> 
        </div>

        <!-- Right: Captured Frame -->
        <div class="box"> 
            <img id="detectionFrame" src="LanternFly_Snapshot_Box.png">
        </div>

    </div>

    <button onclick="captureFrame()">Capture Frame</button>

    <script> 
    const frame = document.getElementById("detectionFrame");
    
    // Ask for the newest frame repeatedly 
    setInterval(() => { 
        frame.src = "/frame?t=" + Date.now(); 
    }, 10500); 
    
    </script>

</body>
</html>
```
