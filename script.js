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
            <video id="video" controls>
                <source src="video.mp4" type="video/mp4">
                Your browser does not support video.
            </video>
        </div>

        <!-- Right: Captured Frame -->
        <div class="box">
            <canvas id="frame"></canvas>
        </div>

    </div>

    <button onclick="captureFrame()">Capture Frame</button>

    <script>
        const video = document.getElementById("video");
        const canvas = document.getElementById("frame");
        const context = canvas.getContext("2d");

        function captureFrame() {
            // Make the canvas the same size as the video
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;

            // Copy the current video frame onto the canvas
            context.drawImage(
                video,
                0,
                0,
                canvas.width,
                canvas.height
            );
        }
    </script>

</body>
</html>
```
