# Description
This repository is maintained by the CTR-Ninjas team for the Kaggle CTR dataset competition, as part of the Y-DATA 2024-2025 course.

<p align="center">
  <img src="images/CTR_Ninjas_Logo.webp" alt="" width="400" />
</p>

# Installation Instructions

To set up the project environment and run the pipeline, follow these steps:

1. **Clone the repository**  
   Clone the main repository to your local machine:
   ```bash
   git clone https://github.com/OphirTuretz/Y-DATA-Ninjas.git
   ```

2. **Create a virtual environment**  
   Within the project directory, create a new virtual environment:
   ```bash
   python3 -m venv venv
   ```

3. **Activate the virtual environment**  
   Activate the virtual environment with the following command:
   ```bash
   source venv/bin/activate
   ```

4. **Install required Python packages**  
   Install the necessary dependencies by running:
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up environment variables**  
   Create a `.env` file and insert your WANDB API key by copying the template:
   ```bash
   cp .env_template .env
   ```
   Replace `$YOUR_API_KEY` in `.env` with your actual WANDB API key.

6. **Create the data directory**  
   Set up the `data` directory to store your datasets:
   ```bash
   mkdir data
   ```

7. **Upload datasets**  
   Upload the following datasets to the `data` folder:  
   - `train_dataset_full.csv` (for training)  
   - `X_test_1st.csv` (for inference)

8. **Update project name**  
   Change the `WANDB_PROJECT` name in the `app/const.py` file to match your project.

9. **Run the pipeline**  
   Finally, execute the pipeline by running:
   ```bash
   inv pipeline
   ```
