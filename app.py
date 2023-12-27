import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
dataset = pd.read_excel('D:\Downloads\dataset_dashboard\greenMobility.xlsx')

# Function to generate sampling distribution for a specific variable
def generate_sampling_distribution(data, variable, sample_size, num_samples):
    sampling_distribution = []
    
    # Drop NaN values and check if the resulting array is empty or too small
    valid_data = data[variable].dropna()
    
    if len(valid_data) < sample_size:
        print(f"Warning: Not enough valid data for variable '{variable}' to generate samples. Skipping.")
        return []  # Return an empty list if there's not enough data
    
    for _ in range(num_samples):
        sample = np.random.choice(valid_data, sample_size, replace=True)
        sample_mean = np.mean(sample)
        sampling_distribution.append(sample_mean)
    
    return sampling_distribution

# Parameters for sampling distribution
sample_size = 30
num_samples = 1000

# Set page title
st.title('Green Mobility Survey Dashboard')

# Sidebar Image
sidebar_image_path = 'tram_green.jpg'  # Replace with the actual path to your image
st.sidebar.image(sidebar_image_path, use_column_width=True)

# Sidebar Greeting
st.sidebar.title('Welcome to Green Mobility Dashboard!')
st.sidebar.markdown('Explore the survey results and sampling distributions.')

# About Section
st.sidebar.header('About')
st.sidebar.markdown('This web interface is part of a data analysis project to understand how our fellow students use transportation, '
                    'in order to find areas to explore and understand their behavior for a greener future.',
         )


# Section 1: Age Distribution Pie Chart and Transport Mode Pie Chart
st.header('👥 Age Distribution and')

# Age Distribution Pie Chart
age_counts = dataset['What is your age category?  '].value_counts()
fig_age = plt.figure(figsize=(6, 6))
plt.pie(age_counts, labels=age_counts.index, autopct='%0.0f%%', startangle=90, colors=plt.cm.Paired.colors, textprops={'fontsize': 10})
plt.title('Age Distribution')
st.pyplot(fig_age)

# Add padding
st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)

# Transport Mode Pie Chart
st.header(' 🚗 Transport Mode Distribution')

transport_counts = dataset['What is your primary mode of transportation for commuting to work or school?'].value_counts()
fig_transport = plt.figure(figsize=(6, 6))
sns.set_palette("pastel")
plt.pie(transport_counts, labels=transport_counts.index, autopct='%1.1f%%', startangle=50, counterclock=False)
plt.title('Distribution of Principal Means of Transport')
st.pyplot(fig_transport)

# Add padding
st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)

# Section 2: Knowledge about Eco-Friendly Transport Options Bar Plot and Time Spent on Daily Commute Histogram
st.header('♻️ Eco-Friendly Knowledge Distribution')

# Knowledge about Eco-Friendly Transport Options Bar Plot
response_counts = dataset['Are you aware of eco-friendly transportation options available in your area?  '].value_counts()
fig_response = plt.figure(figsize=(10, 8))
sns.set_palette("pastel")
sns.barplot(x=response_counts.index, y=response_counts.values, palette=['red', 'green'])
plt.title('Distribution of Knowledge about Eco-Friendly Transport Options')
plt.xlabel('Response')
plt.ylabel('Count')
total_responses = len(dataset)
for i, count in enumerate(response_counts):
    plt.text(i, count + 0.1, f'{count / total_responses * 100:.1f}%', ha='center', va='bottom')
st.pyplot(fig_response)

# Add padding
st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)

# Time Spent on Daily Commute Histogram
st.header('🕒 Time Spent on Daily Commute Histogram')

filtered_dataset = dataset[dataset['On average, how much time you spend on your daily commute?'] != '30 - 60 , 1h +']
fig_commute_time = plt.figure(figsize=(10, 8))
sns.histplot(filtered_dataset, x="On average, how much time you spend on your daily commute?")
plt.xticks(rotation=45, fontsize=12)
st.pyplot(fig_commute_time)

# Add padding
st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)

# Section 3: Environmental Concern Bar Chart
st.header('🌍 Environmental Concern')

plt.figure(figsize=(8, 6))
dataset['On a scale from 1 to 10, how satisfied are you with your current transportation choices in terms of environmental sustainability? (1 being very dissatisfied, 10 being very satisfied)  '].astype(str).value_counts().sort_index().plot(kind='bar', color='lightgreen')
plt.title('Environmental Concern')
plt.xlabel('Level of Concern')
plt.ylabel('Count')
st.pyplot(plt)

# Add padding
st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)


st.set_option('deprecation.showPyplotGlobalUse', False)

#Stacked Bar Chart of Eco-friendly Transportation Consideration by Awareness
#Preparing the data
stacked_data = dataset.groupby(['Are you aware of eco-friendly transportation options available in your area?  ','How many times have you considered switching to more eco-friendly transportation options (e.g., cycling, public transit) due to environmental concerns in the past year?  ']).size().unstack()

#Plotting the stacked bar chart
fig, ax = plt.subplots(figsize=(10, 6))

stacked_data.plot(kind='bar', stacked=True, colormap='viridis', figsize=(10,6))
plt.title('Consideration for Eco-friendly Transportation by Awareness')
plt.xlabel('Awareness of Eco-friendly Transportation Options')
plt.ylabel('Number of Responses')
plt.xticks(ticks=[0, 1], labels=['Not Aware', 'Aware'], rotation=0)  # Assuming 0 is 'Not Aware' and 1 is 'Aware'
plt.legend(title='Consideration for Eco-friendly Transport')
st.pyplot()

#Bar Chart of Frequency of Public Transportation Usage by Environmental Concern Level
#Preparing the data
bar_data = dataset.groupby(['How concerned are you about the environmental impact of your transportation choices?  ', 'How often do you use public transportation for your daily commute  ']).size().unstack()

#Plotting the bar chart

fig2, ax = plt.subplots(figsize=(12, 8))
bar_data.plot(kind='bar', stacked=True, colormap='plasma', figsize=(12, 8))

plt.title('Frequency of Public Transportation Usage by Environmental Concern Level')
plt.xlabel('Environmental Concern Level')
plt.ylabel('Number of Responses')
plt.xticks(rotation=45)
plt.legend(title='Frequency of Public Transport Usage')
st.pyplot()


# Section 7: Sampling Distribution Histograms
st.header('📉 Sampling Distribution Histograms')

# Generate sampling distributions for the mean of 'age' and 'distance_traveled'
sampling_distribution_switch = generate_sampling_distribution(dataset, 'How many times have you considered switching to more eco-friendly transportation options (e.g., cycling, public transit) due to environmental concerns in the past year?  ', sample_size, num_samples)
sampling_distribution_satisfaction = generate_sampling_distribution(dataset, 'On a scale from 1 to 10, how satisfied are you with your current transportation choices in terms of environmental sustainability? (1 being very dissatisfied, 10 being very satisfied)  ', sample_size, num_samples)

# Plotting the sampling distributions
fig_sampling = plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(sampling_distribution_switch, bins=20, edgecolor='black')
plt.title('Sampling Distribution of Mean Consideration to switch')
plt.xlabel('Mean Consideration to switch')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(sampling_distribution_satisfaction, bins=20, edgecolor='black')
plt.title('Sampling Distribution of Mean Satisfaction')
plt.xlabel('Mean Satisfaction')
plt.ylabel('Frequency')

st.pyplot(fig_sampling)
