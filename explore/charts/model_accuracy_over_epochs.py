import os

import matplotlib.pyplot as plt

project_directory = os.path.dirname(os.path.abspath(__file__))  # Get current script's directory
project_directory = os.path.join(project_directory, "..", "..")  # Go up two levels to reach the root (e.g., if this is inside 'notebooks')

submissions_path = os.path.join(project_directory, "submissions")

for submission in os.listdir(submissions_path):
    submission_approach = submission.split('.csv')[0]
    submission_path_image = os.path.join(project_directory, "charts", submission_approach)
    submission_path_image_file = os.path.join(submission_path_image, "accuracy_over_epochs.png")

    os.makedirs(submission_path_image, exist_ok=True)

    history = {
        'accuracy': [0.55, 0.62, 0.63, 0.65, 0.68],
        'val_accuracy': [0.53, 0.57, 0.60, 0.63, 0.64]
    }

    epochs = range(1, len(history['accuracy']) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['accuracy'], label='Training Accuracy', color='blue', marker='o')
    plt.plot(epochs, history['val_accuracy'], label='Validation Accuracy', color='green', marker='o')
    plt.title('Model Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()

    # Save the plot
    plt.savefig(submission_path_image_file)  # Save as an image for LaTeX
    plt.show()
