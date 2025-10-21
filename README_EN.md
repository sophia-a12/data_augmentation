# Video/image data enhancement tools
## Program Introduction
This tool provides an operational interface and supports two core functions: video frame extraction and image enhancement. It can quickly generate diverse image data, suitable for dataset expansion during model training.<br>
## Core Features
1. Supports multiple file upload formats: video files (MP4, AVI, MOV, MKV), image files (JPG, JPEG, PNG, BMP)<br>
2. Video frame extraction: Customize the number of frames to extract, and automatically and evenly sample video frames<br>
3. Image enhancement: Supports eight image enhancement methods: brightness enhancement, brightness reduction, Gaussian blur, Gaussian noise, simulated occlusion, rotation (45 degrees counterclockwise), horizontal flip, and vertical flip<br>
4. Real-time preview and download: The interface displays the video frame extraction and image enhancement results in real time, and provides a download button<br>

##  Environmental requirements
Python 3.7 and above<br>
Dependent libraries: streamlit, opencv-python, numpy, pillow<br>

### Video Processing
1. Select the video file in the upload area.<br>
2. Use the slider to set the number of frames to extract and click Extract Frames.<br>
3. Preview the extracted frames and click Download.<br>
4. Select the frames to be enhanced. The subsequent steps are the same as for image processing.<br>

### Image processing
1. Select the image file in the upload area.<br>
2. Select the desired image enhancement method (multiple selections are allowed).<br>
3. If you selected Simulate Occlusion, adjust the occlusion block size using the panel.<br>
4. Click Execute Image Enhancement. The enhanced results will be displayed after processing is complete and can be downloaded individually or in batches.
