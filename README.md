# Activity Detection using MediaPipe and Deep Learning

This project utilizes MediaPipe for pose detection and a deep learning model for recognizing various activities in real-time from video input.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Folder Structure](#folder-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

To run this project, ensure you have Docker installed on your machine. Then, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/PattadonTh/HUMAN-ACTIVITY-RECOGNITION.git
   cd yourrepository
   ```

2. Build the Docker image:

   ```bash
   docker build -t your_image_name .
   ```

3. Run the Docker container:

   Run the following command to execute the application:

   ```bash
   docker run --rm -it your_image_name
   ```

   Replace `your_image_name` with the name you chose for your Docker image.

## Usage

The application processes video input from the specified video file located in the `assets/` directory and detects activities such as walking, clapping, and raising hands based on the pose landmarks.

1. Ensure your video file is placed in the `assets/` directory.
2. Run the application as described in the [Installation](#installation) section.

## Folder Structure

```
your-project/
│
├── assets/
│   └── video_test.mp4       # Sample video file used in the application
├── src/
│   ├── app.py                # Main application script
├── Dockerfile                 # Dockerfile for building the container
└── requirements.txt           # Python dependencies for the project
```

## Contributing

Contributions are welcome! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
