# Recommendation Engine 1
## Content-Based Hadith Recommendation System
## Project Overview
This project is a content-based recommendation system that recommends relevant Hadiths based on a user's input. The user enters a topic or keyword, and the system retrieves the most relevant Hadiths using TF-IDF vectorization and cosine similarity.
## Dataset Information
Source: Hugging Face
Total Hadiths: 33,738
Features: 9
## Exploratory Data Analysis
No missing values
No duplicate records
Combined the relevant text columns for recommendation
## Recommendation Workflow
TF-IDF vectorization to convert Hadith text into numerical vectors
Transform user input into the same TF-IDF vector space
Calculate cosine similarity between the user's query and all Hadiths
Rank Hadiths based on similarity scores
Retrieve the top matching recommendations
## Technologies Used
Python
Pandas
Scikit-learn
Streamlit
## Future Improvements
Auto-suggestions while typing
Better handling of empty or irrelevant queries
Collaborative filtering
Hybrid recommendation system
Semantic recommendations using transformer-based embeddings
## Author
Nubula H
Data Science & Machine Learning Learner
