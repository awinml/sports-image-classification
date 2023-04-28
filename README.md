# Sports Image Classification

<div align='center'>
   <img src="sports_img_clf_cover.png"  width="700" height="500">
<br/>
</div>

- Built a **image classification model** to predict the sport that is being represented in the image.
- Compared five different CNN architectures ie. a **custom CNN, InceptionV3, ResNet50V2, MobileNetV2 and EfficientNetB3** based on model performance on this dataset.
- A fine-tuned **EfficientNetB3** model was used for generating the final predictions which yielded an **Accuracy of 92.72%, F1-Score of 91.76% and an ROC-AUC Score of 96.92%.**
- A **web app was made using Streamlit** to make predict the sport for new images using the best model.

## Streamlit App

The live app can be viewed [here](https://huggingface.co/spaces/awinml/sports_classification).

## Data

The dataset is a collection of images representing **100 different types of Sports and activities**. The sports range from traditional sports like "archery", "arm wrestling", "bowling", "football", "water polo", "weightlifting" to non-traditional ones like "wingsuit flying" and "nascar racing". The goal is the predict the correct sport based on the image.

- Each image is of size 224 x 224 pixels.
- The images have been segregated into train, test and valid directories.

- There are 13572 train, 500 test, 500 validation images.

The dataset can be downloaded from [here](https://www.kaggle.com/datasets/gpiosenka/sports-classification).

## Experiments:

#### Custom CNN:

- A baseline was created using a custom CNN model with 3 convolution layers and 3 dense layers. A **kernel of size 3 x 3** was used for all the convolution layers. Training the model with an **Adam optimizer with learning rate of 0.001** for **47 epochs** yielded an **Accuracy of 56.44%, F1-Score of 48.48% and an ROC-AUC Score of 49.46%.**

#### InceptionV3:

- The **InceptionV3** model was initialized with pre-trained ImageNet weights. Only the Dense layers were fine-tuned. Training the model with an **Adam optimizer with learning rate of 0.001** for **22 epochs** yielded an **Accuracy of 68.92%, F1-Score of 64.92% and an ROC-AUC Score of 66.64.**

#### ResNet50V2:

- The **ResNet50V2** model was initialized with pre-trained ImageNet weights. Only the Dense layers were fine-tuned. Training the model with an **Adam optimizer with learning rate of 0.001** for **16 epochs** yielded an **Accuracy of 72.88%, F1-Score of 70.67% and an ROC-AUC Score of 69.72.**

#### MobileNetV2:

- The **MobileNetV2** model was initialized with pre-trained ImageNet weights and all the layers were fine-tuned. Training the model with an **Adam optimizer with learning rate of 0.001** for **8 epochs** yielded an **Accuracy of 86.68%, F1-Score of 86.79% and an ROC-AUC Score of 88.36.**

#### EfficientNetB3:

- The **EfficientNetB3** model was initialized with pre-trained ImageNet weights and all the layers were fine-tuned. Training the model with an **Adam optimizer with learning rate of 0.001** for **18 epochs** yielded an **Accuracy of 92.72%, F1-Score of 91.76% and an ROC-AUC Score of 96.92%.**

For all the models that were fine-tuned:

- Two hidden layers of 256 and 128 neurons respectively were added with leaky-relu activations.
- Dropout Layers with p=0.1 were added to prevent overfitting.
- Number of epochs was 50 with Early stopping with patience parameter as 2 epochs.
- Batch size of 32 was used for training.
- Sparse Categorical Cross-Entropy was used as the loss function.

## Results:

The best results were obtained using a fine-tuned EfficientNetB3 model. It was used for generating the final predictions. It achieved an **Accuracy of 92.72%, F1-Score of 91.76% and an ROC-AUC Score of 96.92%.**

The results from all the models have been summarized below:

|              **Model**              | **Accuracy** | **F1-Score** | **ROC\-AUC Score** |
| :---------------------------------: | :----------: | :----------: | ------------------ |
|           **Custom CNN**            |    56.44     |    48.48     | 49.46              |
|  **InceptionV3** _\(fine-tuned\)_   |    68.92     |    64.92     | 66.64              |
|   **ResNet50V2** _\(fine-tuned\)_   |    72.88     |    70.67     | 69.72              |
|  **MobileNetV2** _\(fine-tuned\)_   |    86.68     |    86.79     | 88.36              |
| **EfficientNetB3** _\(fine-tuned\)_ |    92.72     |    91.76     | 96.92              |

## Run Locally

1. Install required libraries:
   ```bash
     pip install -r streamlit/requirements.txt
   ```
2. Fine-tune models:
   ```bash
     python sports-clf-custom-cnn.py
     python sports-clf-inception.py
     python sports-clf-resnet.py
     python sports-clf-mobilenet.py
     python sports-clf-efficientnet.py
   ```
3. Generate predictions:
   ```bash
     python sports-clf-final-predictions.py
   ```

## License &nbsp;&nbsp; [![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

Author: [@awinml](https://www.github.com/awinml)

## Feedback

If you have any feedback, please reach out to me at: &nbsp; &nbsp;
<a href="https://www.linkedin.com/in/ashwin-mathur-ds/"><img src="https://img.shields.io/badge/LinkedIn-Ashwin-blue" alt="LinkedIn" href="https://www.linkedin.com/in/ashwin-mathur-ds/"></a>
