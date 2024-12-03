import os

import matplotlib.pyplot as plt
import pandas as pd

project_directory = os.path.dirname(os.path.abspath(__file__))  # Get current script's directory
project_directory = os.path.join(project_directory, "..", "..")  # Go up two levels to reach the root (e.g., if this is inside 'notebooks')

submissions_path_dir = os.path.join(project_directory, "submissions")

for submission in os.listdir(submissions_path_dir):
    submission_path = os.path.join(project_directory, "submissions", submission)

    submission_approach = submission.split('.csv')[0]
    submission_path_image = os.path.join(project_directory, "charts", submission_approach, "sentiment_vs_phraseid.png")

    # Sample data (replace with your Kaggle submission data)
    kaggle_submission = pd.read_csv(submission_path)  # Adjust the file path

    # Plot Sentiment vs PhraseId
    plt.figure(figsize=(10, 6))
    plt.scatter(kaggle_submission['PhraseId'], kaggle_submission['Sentiment'], c=kaggle_submission['Sentiment'], cmap='viridis', alpha=0.7)
    plt.title('Sentiment vs PhraseId')
    plt.xlabel('PhraseId')
    plt.ylabel('Sentiment')
    plt.colorbar(label='Sentiment')
    plt.tight_layout()

    # Save the plot
    plt.savefig(submission_path_image)  # Save as an image for LaTeX
    plt.show()
