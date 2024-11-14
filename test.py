import streamlit as st
import os

from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods

url = "https://us-south.ml.cloud.ibm.com"

def get_model(model_type,max_tokens,min_tokens,decoding):

    generate_params = {
        GenParams.MAX_NEW_TOKENS: max_tokens,
        GenParams.MIN_NEW_TOKENS: min_tokens,
        GenParams.DECODING_METHOD: decoding
    }

    model = Model(
        model_id=model_type,
        params=generate_params,
        credentials={
            "apikey": st.secrets["api_key"],
            "url": url
        },
        project_id=st.secrets["project_id"]
        )

    return model

def answer_questions():
    st.title('🌠Test watsonx.ai LLM')
    st.info('Created by : Ahmad Maulana Yusuf | GEN AI & ML \n\n sc : https://github.com/WiiAhmad/test-llm')
    user_question = st.text_input('Ask a question, for example: What is IBM?')
    st.info('Masukan pertanyaan lalu enter dan tunggu akan menggenerate jawaban sesuai pertanyaan')
    if user_question.strip():
        #st.info(f'Testing info: {user_question}')
        model_type = ModelTypes.FLAN_UL2
        max_tokens = 100
        min_tokens = 20
        decoding = DecodingMethods.GREEDY

        model = get_model(model_type, max_tokens, min_tokens, decoding)

        generated_response = model.generate(prompt=user_question)
        #print("Answer: " + generated_response)
        formatted_output = f"""
        **Answer to your question:** {user_question} \
        *{generated_response}*</i>
        """
        st.markdown(formatted_output, unsafe_allow_html=True)

if __name__ == "__main__":
    answer_questions()