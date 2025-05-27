# Plant Leaf Disease Detection

## Project Overview

This project implements a plant leaf disease detection system using deep learning with PyTorch and EfficientNet-B3. The system is capable of training a model on a dataset of plant leaf images and then using the trained model to predict diseases on new images through a simple web-based user interface built with Streamlit.

## Dataset

This project uses the PlantVillage dataset, which is available on Kaggle. The dataset contains images of healthy and diseased plant leaves from various plant species. The dataset includes 38 different classes, covering various plant diseases and healthy plant leaves.

Dataset Link: [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

The dataset includes the following plant species and their diseases:
- Apple (Apple scab, Black rot, Cedar apple rust, Healthy)
- Blueberry (Healthy)
- Cherry (Powdery mildew, Healthy)
- Corn (Cercospora leaf spot, Common rust, Northern Leaf Blight, Healthy)
- Grape (Black rot, Esca, Leaf blight, Healthy)
- Orange (Haunglongbing/Citrus greening)
- Peach (Bacterial spot, Healthy)
- Pepper (Bacterial spot, Healthy)
- Potato (Early blight, Late blight, Healthy)
- Raspberry (Healthy)
- Soybean (Healthy)
- Squash (Powdery mildew)
- Strawberry (Leaf scorch, Healthy)
- Tomato (Bacterial spot, Early blight, Late blight, Leaf Mold, Septoria leaf spot, Spider mites, Target Spot, Yellow Leaf Curl Virus, Mosaic virus, Healthy)

## Files in this Project

- `train_model.py` / `train_model.ipynb`: Script/Notebook for training the deep learning model.
- `test_model.py` / `test_model.ipynb`: Script/Notebook for evaluating the trained model on a test dataset.
- `ui.py`: Streamlit application for a user interface to predict diseases on uploaded images.
- `best_model.pth`: Trained model weights (generated after running training script/notebook).
- `class_names.json`: List of class names in the correct order (generated after running training script/notebook).
- `README.md`: This file.

## Setup and Installation

1.  **Clone the repository (if applicable):**

    ```bash
    # If your project is in a Git repository, clone it first.
    # git clone <repository_url>
    # cd <repository_directory>
    ```

2.  **Create a Python Virtual Environment:**

    It's highly recommended to use a virtual environment.

    ```bash
    python -m venv myplantenv
    ```

3.  **Activate the Virtual Environment:**

    - On Windows:
      ```bash
      myplantenv\Scripts\activate
      ```
    - On macOS and Linux:
      ```bash
      source myplantenv/bin/activate
      ```

4.  **Install Dependencies:**

    Navigate to the project directory and install the required libraries. You can create a `requirements.txt` file with the following content:

    ```
    torch
    torchvision
    Pillow
    numpy
    scikit-learn
    streamlit
    requests
    jupyter
    notebook
    # geopy # Uncomment if you still need GPS functionality in other parts
    # transformers # Uncomment if you still need GPT-2 functionality in other parts
    ```

    Then install:

    ```bash
    pip install -r requirements.txt
    ```

    *Note: Ensure you have the correct CUDA drivers and PyTorch version installed if you plan to use GPU for training. Refer to the official PyTorch documentation for installation instructions specific to your system.*

## Data Preparation

1. **Download the Dataset:**
   - Download the PlantVillage dataset from Kaggle
   - Extract the dataset to your project directory
   - The dataset should be organized in the following structure:

```
/path/to/your/data/
├── train/
│   ├── Apple___Apple_scab/
│   │   ├── image1.jpg
│   │   ├── image2.png
│   │   └── ...
│   ├── Apple___Black_rot/
│   │   ├── image1.jpg
│   │   └── ...
│   └── ...
├── valid/
│   ├── Apple___Apple_scab/
│   │   ├── image1.jpg
│   │   └── ...
│   └── ...
└── test/
    ├── Apple___Apple_scab/
    │   ├── image1.jpg
    │   └── ...
    └── ...
```

2. **Dataset Split:**
   - The dataset should be split into training (70%), validation (15%), and testing (15%) sets
   - Each class should maintain its proportion across all splits
   - Make sure you have `train`, `valid`, and `test` directories in your project root or update the data loading paths in `train_model.py` and `test_model.py` accordingly

## Training the Model

You can train the model using either the Python script or Jupyter notebook:

### Using Python Script:
```bash
python train_model.py
```

### Using Jupyter Notebook:
1. Start Jupyter Notebook:
```bash
jupyter notebook
```
2. Open `train_model.ipynb` in your browser
3. Run all cells in the notebook

Both methods will save the trained model weights as `best_model.pth` and the list of class names as `class_names.json` in the project directory upon completion or early stopping.

## Testing the Model

After training, you can evaluate the model's performance using either the Python script or Jupyter notebook. Ensure `best_model.pth` and `class_names.json` are present.

### Using Python Script:
```bash
python test_model.py
```

### Using Jupyter Notebook:
1. Start Jupyter Notebook (if not already running):
```bash
jupyter notebook
```
2. Open `test_model.ipynb` in your browser
3. Run all cells in the notebook

Both methods will load the saved model and class names, perform batch inference on the test set, and print the confusion matrix and classification report.

## Running the User Interface

To use the interactive Streamlit application, make sure `best_model.pth` and `class_names.json` are in the same directory as `ui.py`. Then run:

```bash
streamlit run ui.py
```

This will start the Streamlit server and open the application in your web browser. You can then upload plant leaf images to get predictions.

## Customization

- **Hyperparameters:** Modify learning rates, batch size, epochs, scheduler parameters, etc., in `train_model.py` to tune the model.
- **Model Architecture:** Experiment with different pre-trained models or modify the classifier head further in `train_model.py`.
- **Data Augmentation:** Adjust the parameters or add more transformations in the `train_transform` in `train_model.py`.
- **Evaluation Metrics:** Add more evaluation metrics or visualizations in `test_model.py`.
- **UI Features:** Enhance the `ui.py` with additional features like displaying top-k predictions, visualizing heatmaps, or providing disease information.

---

Feel free to explore and modify the code to suit your specific needs! 