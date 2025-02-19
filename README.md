# Audio Generation MNIST Setup Guide

## Prerequisites
- Docker installed on your system
- Web browser
- Internet connection

## Installation Steps

1. Navigate to the run folder:
```bash
cd /run
```

2. Download the required models:
- Download the `saved_models` folder from [Google Drive](https://drive.google.com/drive/folders/1vg-l1cmQJveYuNxIf3T3YID1GBKD2TPH?usp=sharing)
- Place the downloaded folder in your current directory

3. Build the Docker image:
```bash
docker build -t audio-generation-mnist .
```

4. Run the Docker container:
```bash
docker run -d -p 8956:8956 audio-mnist-generation
```

> **Note**: If you need to use a different port, make sure to update both the backend and frontend configurations accordingly.

## Accessing the Application

1. Once the Docker container is running, open the `frontend.html` file in your web browser.

2. The application should now be accessible and ready to use.

## Port Configuration

The default configuration uses port 8956. If you need to change this:
- Update the port in the Docker run command
- Modify the corresponding port settings in both backend and frontend code

## Troubleshooting

If you encounter any issues:
1. Ensure Docker is running properly
2. Verify that port 8956 is not being used by another application
3. Check that all files from the saved_models folder were downloaded successfully