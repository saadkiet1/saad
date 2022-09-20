import streamlit
import pandas as pd
import torch
from transformers import pipeline
import streamlit as st

def app():
    st.title("Text Summarization  🤓")

    st.markdown("This is a Web application that Summarizes Text 😎")
    upload_file = st.file_uploader('Upload a file containing Text data')
    button = st.button("Summarize")

    st.cache(allow_output_mutation=True)
    def facebook_bart_model():
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        return summarizer
    summarizer= facebook_bart_model()

    def text_summarizer(text):
        a = summarizer(text, max_length=150, min_length=30, do_sample=False)
        return a[0]['summary_text']


    # Check to see if a file has been uploaded
    if upload_file is not None and button:
        st.success("Summarizing Text, Please wait...")
        # If it has then do the following:

        # Read the file to a dataframe using pandas
        df = pd.read_csv(upload_file)

        # Create a section for the dataframe header

        df1 = df.copy()
        df1['summarized_text'] = df1['Dialog'].apply(text_summarizer)

        df2 = df1[['Name','summarized_text']]
        st.write(df2.head(5))

        @st.cache
        def convert_df(dataframe):
            return dataframe.to_csv().encode('utf-8')

        csv = convert_df(df2)
        st.download_button(label="Download CSV", data=csv, file_name='summarized_output.csv', mime='text/csv')






if __name__ == "__main__":
    app()
