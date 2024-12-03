import os

import matplotlib.pyplot as plt
import pandas as pd

project_directory = os.path.dirname(os.path.abspath(__file__))  # Get current script's directory
project_directory = os.path.join(project_directory, "..", "..")  # Go up two levels to reach the root (e.g., if this is inside 'notebooks')

submissions_path_dir = os.path.join(project_directory, "submissions")

for submission in os.listdir(submissions_path_dir):
    submission_path = os.path.join(project_directory, "submissions", submission)

    submission_approach = submission.split('.csv')[0]
    submission_path_image = os.path.join(project_directory, "charts", submission_approach, "sentiment_distribution.png")

    # Sample data (replace with your Kaggle submission data)
    kaggle_submission = pd.read_csv(submission_path)  # Adjust the file path
    sentiment_counts = kaggle_submission['Sentiment'].value_counts().sort_index()

    # Plot sentiment distribution
    plt.figure(figsize=(8, 6))
    sentiment_counts.plot(kind='bar', color=['red', 'orange', 'yellow', 'lightgreen', 'green'])
    plt.title('Sentiment Distribution in Kaggle Submission')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.xticks(ticks=range(5), labels=['Negative', 'Somewhat Negative', 'Neutral', 'Somewhat Positive', 'Positive'], rotation=0)
    plt.tight_layout()

    # Save the plot
    plt.savefig(submission_path_image)  # Save as an image for LaTeX
    plt.show()
