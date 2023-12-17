# Real-Time Emotion Detection Using Convolutional Neural Network (CNN)

## Methodology
This project employs a Convolutional Neural Network (CNN) for real-time emotion detection through a webcam. The methodology involves the following key steps:

1. **Dataset:** The model is trained on the FER-2013 dataset, consisting of grayscale face images categorized into seven emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral).

2. **Model Architecture:** A CNN architecture is utilized for its effectiveness in image classification tasks. The model includes convolutional layers for feature extraction, batch normalization to improve convergence, max-pooling for down-sampling, and dense layers for classification.

3. **Data Preprocessing:** Images are preprocessed by resizing to a standardized input size, normalizing pixel values, and augmenting the dataset through horizontal and vertical flips.

4. **Training:** The model is trained on the preprocessed dataset using the Adam optimizer and categorical crossentropy loss function. Training involves multiple epochs to learn the patterns and features associated with different emotions.

5. **Real-Time Detection:** The trained model is then employed for real-time emotion detection through a webcam. Each frame from the webcam feed is processed, converted to grayscale, preprocessed, and fed into the model for prediction.

## Algorithm Used
The algorithm is based on deep learning principles, specifically using a Convolutional Neural Network. CNNs are well-suited for image-related tasks due to their ability to automatically learn hierarchical features from data. The layers of the CNN detect low-level features (edges, textures) and progressively combine them to recognize higher-level patterns (facial features, emotions).

The FER-2013 dataset provides a diverse set of facial expressions, allowing the CNN to generalize and accurately predict the emotion displayed in real-time webcam images.

This project combines the robustness of CNNs with real-time webcam usage, making it applicable for scenarios such as emotion-aware applications, user experience testing, and interactive systems.
# Repository Contents

1. **Accuracy.jpg:**
   - Description: Graph depicting the model's accuracy during training.
   - Purpose: Visual representation of how accurately the model is learning from the training data.

2. **EMDEC_Webcam.ipynb:**
   - Description: Jupyter notebook containing code for real-time emotion detection using a webcam.
   - Purpose: Allows users to run the real-time emotion detection system using their webcam.

3. **EMDec.ipynb:**
   - Description: Jupyter notebook containing code for training the emotion detection model.
   - Purpose: Provides the code and details for training the Convolutional Neural Network (CNN) model on the FER-2013 dataset.

4. **Fear.jpg:**
   - Description: Screenshot showcasing the real-time emotion detection system identifying the emotion "Fear."
   - Purpose: Visual confirmation of the model's performance in detecting fear emotions.

5. **Loss.jpg:**
   - Description: Graph illustrating the model's loss during training.
   - Purpose: Provides insights into how well the model is minimizing errors during the training process.

6. **angry.jpg:**
   - Description: Screenshot demonstrating the real-time emotion detection system identifying the emotion "Angry."
   - Purpose: Visual confirmation of the model's performance in detecting angry emotions.

7. **disgust.jpg:**
   - Description: Screenshot illustrating the real-time emotion detection system identifying the emotion "Disgust."
   - Purpose: Visual confirmation of the model's performance in detecting disgust emotions.

8. **happy.jpg:**
   - Description: Screenshot displaying the real-time emotion detection system identifying the emotion "Happy."
   - Purpose: Visual confirmation of the model's performance in detecting happy emotions.

9. **neutral.jpg:**
   - Description: Screenshot presenting the real-time emotion detection system identifying the emotion "Neutral."
   - Purpose: Visual confirmation of the model's performance in detecting neutral emotions.

10. **sad.jpg:**
    - Description: Screenshot showcasing the real-time emotion detection system identifying the emotion "Sad."
    - Purpose: Visual confirmation of the model's performance in detecting sad emotions.

11. **surprise.jpg:**
    - Description: Screenshot depicting the real-time emotion detection system identifying the emotion "Surprise."
    - Purpose: Visual confirmation of the model's performance in detecting surprise emotions.

This repository contains the necessary components for training the model, evaluating its performance, and utilizing it for real-time emotion detection using a webcam. The screenshots provide visual validation of the model's ability to recognize various emotions.
