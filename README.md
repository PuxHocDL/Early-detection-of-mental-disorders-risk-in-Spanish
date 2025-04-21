# MentalRiskES 2025 - Early Detection of Mental Disorders in Spanish

This repository contains the code and resources for participating in the **MentalRiskES@IberLEF2025** shared task, focusing on early detection of mental disorder risks in Spanish social media comments. The project addresses two tasks: **Risk Detection of Gambling Disorders** (Task 1) and **Type of Addiction Detection** (Task 2). The tasks are designed as online problems, emphasizing both accuracy and speed of detection in a continuous stream of user messages.

## Table of Contents
- [Overview](#overview)
- [Tasks](#tasks)
  - [Task 1: Risk Detection of Gambling Disorders](#task-1-risk-detection-of-gambling-disorders)
  - [Task 2: Type of Addiction Detection](#task-2-type-of-addiction-detection)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Training and Evaluation](#training-and-evaluation)
- [Submission](#submission)
- [Results and Error Analysis](#results-and-error-analysis)
- [Contributing](#contributing)
- [License](#license)

## Overview
The **MentalRiskES@IberLEF2025** shared task aims to identify mental disorder risks, specifically gambling-related disorders, from Spanish social media comments. Early detection is critical, as mental disorders affect 1 in 8 people globally, with a significant rise in anxiety and depression post-COVID-19. This project implements machine learning models to process streaming user messages, enabling timely risk identification.

The repository provides scripts for data preprocessing, model training, prediction, and submission to the MentalRiskES API. It includes a baseline notebook for interacting with the submission API and code for both tasks.

## Tasks

### Task 1: Risk Detection of Gambling Disorders
- **Type**: Binary classification
- **Objective**: Predict whether a user is at high risk (label = 1) or low risk (label = 0) of developing a gambling-related disorder based on their messages.
- **Goal**: Enable early detection for timely interventions.
- **Evaluation**: Accuracy, macro F1-score, and earliness (time to correct prediction) are key metrics, as the task simulates an online streaming scenario.

### Task 2: Type of Addiction Detection
- **Type**: Multiclass classification
- **Objective**: For all users (regardless of high or low risk), identify the specific type of addiction associated with their gambling disorder. The categories are:
  - **Betting**: Placing bets on sports-related events (e.g., football betting).
  - **Online Gaming**: Participating in games of chance (e.g., roulette, blackjack, slot machines).
  - **Trading**: Speculative investments, especially in cryptocurrencies.
  - **Lootboxes**: Purchasing virtual items with randomized content in video games.
- **Notes**:
  - Each user is associated with exactly one addiction type.
  - The prediction submitted in the last round is used for evaluation.
- **Evaluation**: Macro F1-score, accuracy, and earliness.

## Dataset
The dataset consists of Spanish social media comments provided by the MentalRiskES organizers. Due to privacy and competition rules, the dataset is not included in this repository. Participants can access the data through the **MentalRiskES@IberLEF2025** submission API after registration.

- **Trial Data**: Available for testing systems before the evaluation phase.
- **Test Data**: Accessed via the API during the evaluation period.
- **Format**: Each user has a sequence of messages, processed in a streaming fashion.
- **Preprocessing**:
  - Emojis are mapped to Spanish text (e.g., "🔝" → "arriba") using a predefined dictionary.
  - Messages are cleaned and tokenized for model input.

Example messages:
- Betting: “yo de frees no hable” / “pues de pago si hay buenos , pero a mi me ha llevado mi tiempo emplea tu el tuyo”
- Online Gaming: “24 céntimos con los free spins , Roma no se construyó en un dia”
- Trading: “Siii fue cuestión de rapidez porqje subio a 60 y rapidio se desinflo”
- Lootboxes: “y al que le toco sirve para intercambio”

## Model Architecture

### Task 1: Risk Detection of Gambling Disorders
- **Model**: LSTM-based classifier with attention mechanism.
- **Backbone**: Pre-trained `pysentimiento/robertuito-sentiment-analysis` for generating contextual embeddings.
- **Architecture**:
  - **Preprocessing**: Emojis are replaced with Spanish text, and messages are tokenized.
  - **Feature Extraction**: RoBERTuito generates CLS embeddings for each message.
  - **LSTM Layer**: Processes sequential embeddings with bidirectional LSTM (hidden sizes: 96, 128).
  - **Attention Mechanism**: Computes weighted context vector for classification.
  - **Classifier**: Linear layer for binary output (high/low risk).
- **Data Augmentation**: Neutral samples are split, and positive samples are augmented with neutral messages to balance earliness and accuracy.
- **Training**:
  - Optimizer: Adam (learning rate = 1e-4).
  - Loss: Cross-entropy with soft labels ([0.95, 0.05] for low risk, [0.05, 0.95] for high risk).
  - GDRO (Group Distributionally Robust Optimization) to handle class imbalance.
  - Batch sizes: 2, 4, 8.
  - Epochs: 150.
- **Output**: Binary prediction (0 or 1) for each user.

### Task 2: Type of Addiction Detection
- **Model**: Fine-tuned RoBERTa-based model (`PlanTL-GOB-ES/roberta-base-bne`).
- **Architecture**:
  - **Preprocessing**: Emojis are replaced with Spanish text, and messages are combined into a single text per user.
  - **Feature Extraction**: RoBERTa-base-bne processes tokenized text.
  - **Classifier**: Linear layer for 4-class output (Betting, Online Gaming, Trading, Lootboxes).
- **Data Augmentation**: Back-translation (Spanish → English → French → Spanish) for the minority class (Lootboxes) to address class imbalance.
- **Training**:
  - Optimizer: Adam with Optuna-optimized hyperparameters (learning rate: 3e-4 to 5e-4, batch size: 2 or 4, epochs: 4-10, weight decay: 0.05-0.1).
  - Loss: Weighted cross-entropy with class weights ([1.0, 1.0, 1.0, 1.5] or [1.0, 1.0, 1.0, 2.0]).
  - 5-fold stratified cross-validation.
  - Gradient accumulation and mixed precision (FP16) for efficiency.
- **Output**: Multiclass prediction (0: Betting, 1: Online Gaming, 2: Trading, 3: Lootboxes).

## Usage
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/mentalriskes-2025.git
   cd mentalriskes-2025
   ```
2. **Run Notebooks**:
   Task 1:
   ```bash
   jupyter notebook task1.ipynb
   ```
   Execute cells to preprocess data, train the LSTM model, and save parameters to   pre_trained_models/.
   Task 2:
   ```bash
   jupyter notebook task2.ipynb
   ```
   Execute cells to preprocess data, optimize hyperparameters with Optuna, train the RoBERTa model, and save to ./best_model_best
   Submission:
   ```bash
   jupyter notebook ClientServer.ipynb
   ```
   Use this notebook to interact with the MentalRiskES API, retrieve messages, generate predictions, and submit results.
