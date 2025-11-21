🧠 Soft Skills Recommendation System for IT Jobs

🎯 Overview

This project implements a Soft Skills Recommendation System that predicts which soft skills are required for different IT job roles using Bayesian Networks and Deep Learning.
It combines probabilistic reasoning (Bayesian Networks) and neural learning (MLP) to analyze relationships between job title, qualification, and experience to suggest key soft skills such as teamwork, leadership, problem-solving, and creativity.

🧩 Features

✅ Two Bayesian Network models:

• Msme (SME-based): Expert-defined model using manually assigned conditional probabilities

• Madv (Advertisement-based): Data-driven model learned using Bayesian Estimator

✅ One Deep Learning model:

• Multi-Layer Perceptron (MLP): Learns soft skill patterns from training data

✅ Flask web interface:

• Job input form (job title, qualification, experience)

• Model selection (Msme / Madv / DL)

• Real-time soft skill probability predictions

• Accuracy comparison charts by Job Title and Soft Skills

• Global model accuracy summary

✅ Visualization:

• Blue → Msme

• Red → Madv

• Green → DL

Results -
<img width="1000" height="400" alt="accuracy_by_softskills" src="https://github.com/user-attachments/assets/68489ad7-c57c-45a3-b3de-445e9a4544e4" />

<img width="1100" height="500" alt="accuracy_by_job_title" src="https://github.com/user-attachments/assets/4648fc92-25d1-459f-afd0-50910e5e8f1f" />

<img width="900" height="400" alt="dl_accuracy_by_softskills" src="https://github.com/user-attachments/assets/2cf30bf9-cb09-4501-9e7c-9a45b7ee7b99" />

<img width="1000" height="500" alt="dl_accuracy_by_job_title" src="https://github.com/user-attachments/assets/1f755b56-f43d-4979-92de-63a115e43516" />

