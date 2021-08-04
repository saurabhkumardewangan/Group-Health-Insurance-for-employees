import pandas as pd
import streamlit as st 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from PIL import Image


#################### Importing the data
ghi = pd.read_excel("C:/Users/ADMIN/Desktop/Projects_DS/New Data/dataset2L.xlsx")
cs2 = pd.read_excel("C:/Users/ADMIN/Desktop/Projects_DS/New Data/op.xlsx")

#################### Loading the Title image
image = Image.open("C:/Users/ADMIN/Desktop/Projects_DS/logo.png")
st.image(image)

#################### Loading the Text Tilte
st.title("Group Health Insurance Policy")

#################### Creating Sidebar data
st.sidebar.subheader("INFORMATION")

st.sidebar.write(" ")

st.subheader("Policy cover of 2 Lacs Sum Insured")

st.write(" ")

check_data = st.sidebar.checkbox("View sample data")
if check_data:
    st.sidebar.write(ghi.head())
    
st.sidebar.write(" ")

check_data1 = st.sidebar.checkbox("View sample data for recommendation")
if check_data1:
    st.sidebar.write(cs2.head())
    
st.sidebar.write(" ")

check_data2 = st.sidebar.checkbox("About the app")
if check_data2:
    st.sidebar.write("This app is for recommending Group Health Insurance policy similar to which the customer already have. ")
    
st.write(" ")

#################### Creating Main data 

# Input the premium
Policy_select = st.selectbox("Select your current policy",cs2.Providerproduct)

st.write(" ")

# Number of recommendation
radio = [1,2,3,4,5]
topN = st.radio("Select Top N policy to see",radio)


# Recommendation

# Creating a Tfidf Vectorizer to remove all stop words
tfidf = TfidfVectorizer(stop_words = "english")    # taking stop words from tfid vectorizer 

# Preparing the Tfidf matrix by fitting and transforming
tfidf_matrix = tfidf.fit_transform(cs2.Providerproduct)

# Computing the cosine similarity on Tfidf matrix
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

# creating a mapping of policy name to index number 
policy_index = pd.Series(cs2.index, index = cs2['Providerproduct']).drop_duplicates()

def get_recommendations(Name, topN): 
    global result
    # topN = 10
    # Getting the policy  category index using its title 
    policy_id = policy_index[Name]
    
    # Getting the pair wise similarity score for all the policy with that 
    # policy
    cosine_scores = list(enumerate(cosine_sim_matrix[policy_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar movies 
    cosine_scores_N = cosine_scores[0: topN+1]
    
    # Getting the policy category index 
    policy_idx  =  [i[0] for i in cosine_scores_N]
    policy_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar policy category and scores
    policy_similar_category = pd.DataFrame(columns=["Providerproduct", "Score"])
    policy_similar_category["Providerproduct"] = cs2.loc[policy_idx, "Providerproduct"]
    policy_similar_category["Score"] = policy_scores
    policy_similar_category.reset_index(inplace = True)  
    # policy_similar_category.drop(["index"], axis=1, inplace=True)
   
    return policy_similar_category
    

result = get_recommendations(Policy_select,topN)
recommended_policy = result.iloc[1:,:]

st.write(" ")

################# Creating Recommendation 

if st.button("Recommendation"):
    st.write("Recommended policies with their index and similarity score")
    st.dataframe(recommended_policy)
    

