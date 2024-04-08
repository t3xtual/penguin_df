# Importing the necessary libraries.
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load the DataFrame
csv_file = 'penguin.csv'
df = pd.read_csv(csv_file)

# Display the first five rows of the DataFrame
df.head()

# Drop the NAN values
df = df.dropna()

species_map = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo':2}

# Add numeric column 'label' to resemble non numeric column 'species'
df['label'] = df['species'].map(species_map)

sex_map = {'Male':0, 'Female':1}
# Convert the non-numeric column 'sex' to numeric in the DataFrame
df['sex'] = df['sex'].map(sex_map)

island_map = {'Biscoe': 0, 'Dream': 1, 'Torgersen':2}
# Convert the non-numeric column 'island' to numeric in the DataFrame
df['island'] = df['island'].map(island_map)


# Create X and y variables
X = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# Build a SVC model using the 'sklearn' module.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)

# Build a LogisticRegression model using the 'sklearn' module.
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Build a RandomForestClassifier model using the 'sklearn' module.
rf_clf = RandomForestClassifier(n_jobs = -1)
rf_clf.fit(X_train, y_train)

def prediction(model, island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex):
	label = model.predict(np.array([island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex]).reshape((1,-1)))[0]
	return list(species_map.keys())[list(species_map.values()).index(label)]

st.title("Penguin Prediction Models")

bill_length_mm = st.sidebar.slider("Bill Length (In mm)", df['bill_length_mm'].min(),df['bill_length_mm'].max())
bill_depth_mm = st.sidebar.slider("Bill Depth (In mm)", df['bill_depth_mm'].min(),df['bill_depth_mm'].max())
flipper_length_mm = st.sidebar.slider("Flipper Length (In mm)", df['flipper_length_mm'].min(),df['flipper_length_mm'].max())
body_mass_g = st.sidebar.slider("Body Mass (In grams)", df['body_mass_g'].min(), df['body_mass_g'].max())
island = island_map[st.sidebar.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))]
sex = sex_map[st.sidebar.selectbox('Gender', ('Male', 'Female'))]
classifier = st.sidebar.selectbox('Classifier', ('Support Vector Machine', 'Logistic Regression', 'Random Forest Classifier'))

if st.sidebar.button("Predict"):
	if classifier == "Support Vector Machine":
		species_type = prediction(svc_model, island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex)
		score = svc_model.score(X_train, y_train)
		st.write("Species predicted:", species_type)
		st.write("Accuracy score of this model is:", score)
	elif classifier == "Logistic Regression":
		species_type = prediction(log_reg, island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex)
		score = log_reg.score(X_train, y_train)
		st.write("Species predicted:", species_type)
		st.write("Accuracy score of this model is:", score)
	elif classifier == 'Random Forest Classifier':
		species_type = prediction(rf_clf, island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex)
		score = rf_clf.score(X_train, y_train)
		st.write("Species predicted:", species_type)
		st.write("Accuracy score of this model is:", score)
