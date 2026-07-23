import streamlit as st 
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import linear_kernel

with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)

with open("hadhith.pkl", "rb") as f:
    hadhith = pickle.load(f)

print(type(hadhith))
print(hadhith.head())

st.title("HADHITH RECOMMENDATION SYSTEM")

user_input = st.text_input("Enter a topic",placeholder = "Example: patience")

def get_recommendation(user_input):
        user_input = user_input.lower().strip()
        query_vector = tfidf.transform([user_input])
        cosine_sim1 = linear_kernel(query_vector, tfidf_matrix).flatten()
        if cosine_sim1.max()<0.1:
             return None
        similarity_scores = list(enumerate(cosine_sim1))
        similarity_scores = sorted(similarity_scores, key= lambda x: x[1], reverse = True)
        similarity_scores = similarity_scores[:10]
        similarity_index = [i[0] for i in similarity_scores]
        
        return hadhith.iloc[similarity_index]

if st.button("Recommend"):

    if user_input.strip() ==  "":
        st.warning("Please enter a topic")

    else:
        recommendations = get_recommendation(user_input)

    if recommendations is None:
             st.error("❌ No relevant hadhith found. Please try another topic")

    else:
        st.subheader("Recommended Hadhiths")

    recommendations = get_recommendation(user_input)
    
    if recommendations is None:
         st.error("❌ No relevant hadhith found. Please try another topic")
    
    else:
         st.subheader("ecommended Hadhiths")


    for i in range(len(recommendations)):
        st.write("### Recommendation", i+1)
        st.write("Book", recommendations.iloc[i]["Book"])
        st.write("Chapter", recommendations.iloc[i]["Chapter_Title_English"])
        st.write("English")
    
        st.write(recommendations.iloc[i]["English_Text"])
        st.write("Arabic")
        
        st.write(recommendations.iloc[i]["Arabic_Text"])

        st.markdown("---")


         