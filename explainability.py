# This module contains the script to generate explanation to predictions.
import pandas as pd
import shap
import json
from sklearn import *
import openai
import numpy as np
from openai.error import ServiceUnavailableError,RateLimitError,APIConnectionError,APIError,AuthenticationError

np.bool = np.bool_
np.int = np.int64

import os

# Get the absolute path of the "config" folder
config_folder = os.path.join(os.path.dirname(__file__), '..', 'config')
# Construct the absolute path to the JSON file
config_file_path = os.path.join(config_folder, 'prompt_config.json')
model_file_path = os.path.join(config_folder, 'models.json')
# Read the JSON file


def generate_response_using_openai(prompt):
    '''
    The generate_response_using_openai function is used to generate a response using the OpenAI API. It takes a prompt as input, which is the message or question for which a response is desired.
    The function first reads the OpenAI API key from a configuration file. This key is necessary to authenticate and authorize the API request. If the API key is not found in the configuration file, an error message is printed.
    Next, the function sends a request to the OpenAI API using the openai.ChatCompletion.create method. This method creates a chat-based completion using the GPT-3.5-turbo model. The messages parameter is set to a list containing a single dictionary with two keys: role and content. The role is set to "user" and the content is set to the prompt provided as input.
    If the API request is successful, the function retrieves the generated response message from the API response. It accesses the choices list, which contains the generated messages, and retrieves the content of the first choice. This content is then returned as the output of the function.
    The function also handles different types of errors that may occur during the API request. If an AuthenticationError occurs, it means that the provided API key is incorrect or does not exist, and an error message is printed. If a RateLimitError occurs, it means that too many requests have been made and the function suggests trying again after a minute. If a ServiceUnavailableError occurs, it means that the OpenAI service is currently facing trouble handling requests, and the function suggests trying again later. If an APIConnectionError occurs, it means that there was an error establishing a connection with the OpenAI API, and the function suggests trying again or waiting for some time.
    Overall, the generate_response_using_openai function provides a convenient way to generate responses using the OpenAI API, handling errors and returning the generated response message.
    '''
    with open(config_file_path,'r') as file:
        configs = json.load(file)

    try:
        openai.api_key = configs['openai_api_key']
    except KeyError as e:
        print("API Key not found, please update the config file.")
    
    try:
        completions = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],

        )
        message = completions['choices'][0]['message']['content']
        return message
    except AuthenticationError as e:
        print('The Open AI API key provided is incorrect or does not exist. Re-enter the correct API key in the config file.')
    except RateLimitError as e:
        print('Too many requests, please try after a minute.')
    except ServiceUnavailableError as e:
        print('Open AI is currently facing trouble handling requests. Please try some time later.')
    except APIConnectionError as e:
        print('There was an error establishing connection with Open AI. Please try again.\nIf the issue persists, please try some time later.')

def generate_response_using_palm(prompt):
    import vertexai
    from vertexai.language_models import TextGenerationModel

    vertexai.init(project="ust-genai-pa-poc-gcp", location="us-central1")
    parameters = {
        "temperature": 0.2,
        "max_output_tokens": 1024,
        "top_p": 0.8,
        "top_k": 40
    }
    model = TextGenerationModel.from_pretrained("text-bison@001")
    response = model.predict(
        prompt,
        **parameters
    )
    return response.text

def combine_features_with_shap_values(feature_lst,shap_values,exp_type):
    '''Originally shap returns a set of arrays as outputs. There is no linking to the features corresponding to each 
    shapley values. So we will manipulate the features of the dataset and the shap values to put them together in a 
    dictionary using this function.
    
    The combine_features_with_shap_values function is used to combine the features of a dataset with their corresponding SHAP values. It takes in three inputs: feature_lst, shap_values, and exp_type.
    The function first initializes an empty dictionary called shap_dictionary to store the combined features and SHAP values.
    If the explainer_type is "KernelClassifiers", the function iterates over each element in the feature_lst using a for loop. It assigns the corresponding SHAP value from shap_values to the feature in the shap_dictionary using the index i. After each iteration, i is incremented by 1. Finally, the function returns the shap_dictionary.
    If the explainer_type is "TreeClassifiers", the function follows a similar process as above, but with an additional level of indexing. It iterates over each element in the feature_lst using a for loop. It assigns the corresponding SHAP value from shap_values to the feature in the shap_dictionary using the index i and an additional index of 0. After each iteration, i is incremented by 1. Finally, the function returns the shap_dictionary.
    In both cases, the shap_dictionary is a dictionary where the keys are the features and the values are the corresponding SHAP values. This allows for easy access and interpretation of the SHAP values for each feature.
    '''
    shap_dictionary = {}
    if exp_type == "KernelClassifiers":
        i = 0
        for ele in feature_lst:
            shap_dictionary[ele] = shap_values[0][i]
            i = i+1
        return shap_dictionary
    elif exp_type == "TreeClassifiers":
        i = 0
        for ele in feature_lst:
            shap_dictionary[ele] = shap_values[0][0][i]
            i = i+1
        return shap_dictionary


def fetch_explainer_type(algorithm):
    '''
    The fetch_explainer_type function is used to determine the type of explainer associated with a given algorithm. It takes an algorithm name as input and returns the explainer type.

    The function first opens a JSON file called models.json using the open function and reads its contents using the json.load function. This JSON file contains a mapping of explainer types to lists of algorithm names.

    The function then iterates over the keys of the loaded JSON object, which represent the explainer types. For each explainer type, it retrieves the list of algorithm names associated with that explainer type. If the input algorithm name is found in one of these lists, the function sets the algo variable to the corresponding explainer type.

    If the algo variable is still an empty string after iterating over all the explainer types, it means that the input algorithm name is not supported. In this case, the function returns False to indicate that no explainer type was found.

    If the algo variable is not an empty string, it means that an explainer type was found for the input algorithm name. In this case, the function returns the value of the algo variable, which represents the explainer type.

    The fetch_explainer_type function can be used in conjunction with the explain function to determine the explainer type for a given algorithm and generate explanations using that explainer type.
    '''
    with open(model_file_path,'r') as file:
        supported_models = json.load(file)
    algo = ''
    try:
        for ele in list(supported_models.keys()):
            model_names = supported_models[ele]
            if algorithm in model_names:
                algo = ele
    except KeyError as e:
        print("This algorithm is currently not supported.")

    if algo == '':
        return False
    else:
        return algo
    
def generate_shap_values(model,df,exp_type,training_data):
    '''
    The generate_shap_values function is used to generate SHAP (SHapley Additive exPlanations) values for a given machine learning model and dataset. SHAP values provide explanations for individual predictions made by the model.

    The function takes in four parameters:

    model: The machine learning model for which SHAP values need to be generated.

    df: The dataframe containing the input data for which SHAP values need to be generated.

    exp_type: The type of explanation method to be used. Currently, the function supports two types: 'TreeClassifiers' and 'KernelClassifiers'.

    training_data: The training data used to train the machine learning model.

    The function first checks the exp_type parameter to determine the type of explanation method to be used. If exp_type is 'TreeClassifiers', it creates a TreeExplainer object from the SHAP library using the model parameter. If exp_type is 'KernelClassifiers', it creates a KernelExplainer object from the SHAP library using the model.predict_proba function and a sample of the training_data parameter.

    Once the appropriate explainer object is created, the function calls the shap_values method of the explainer object, passing in the df parameter to generate the SHAP values. The SHAP values are then returned by the function.
    To use the generate_shap_values function, you need to have a trained machine learning model (model), a dataframe containing the input data (df), the type of explanation method (exp_type), and the training data used to train the model (training_data). After calling the function with these parameters, it will return the SHAP values for the input data.
    '''
    if exp_type=='TreeClassifiers':
        explainer = shap.TreeExplainer(model)
    elif exp_type=="KernelClassifiers":
        explainer = shap.KernelExplainer(model.predict_proba,shap.sample(training_data,100))
    shap_values = explainer.shap_values(df)
    return shap_values

def explain(model,df,training_data,features_info=None):
    '''
    The explain function is used to generate explanations for machine learning models using SHAP values and two different language models (OpenAI and PALM).
    First, the function reads the configuration file to get the prompt template. It also gets the algorithm name from the model.
    Next, the function calls the fetch_explainer_type function to determine the type of explainer based on the algorithm name. If the explainer type is supported, the function proceeds with generating the explanations. Otherwise, it prints a message stating that the model is not supported.
    If the explainer type is supported, the function calls the generate_shap_values function to generate the SHAP values for the dataset using the appropriate explainer. The function also calls the combine_features_with_shap_values function to combine the features of the dataset with their corresponding SHAP values.
    Then, the function creates a prompt using the prompt template and the algorithm name, feature information, feature values, and SHAP values. This prompt is used to generate explanations using both the OpenAI and PALM language models. The function calls the generate_response_using_palm function to generate an explanation using PALM, and the generate_response_using_openai function to generate an explanation using OpenAI.

    Finally, the function returns the explanations generated by PALM and OpenAI.
    '''
    with open(config_file_path,'r') as file:
        configs = json.load(file)

    try:
        prompt_template = configs['explanation_prompt']
    except KeyError as e:
        print("Prompt template could not be found.")

    algorithm_name = model.__class__.__name__
    
    exp_type = fetch_explainer_type(algorithm_name)

    if exp_type:
        shap_values = generate_shap_values(model,df,exp_type,training_data)
        shap_values_dictionary = combine_features_with_shap_values(list(training_data.columns),shap_values,exp_type)
        feature_values = dict(df)
        prompt = prompt_template.format(algorithm=str(algorithm_name),features=str(features_info),values=str(feature_values),shap_values=str(shap_values_dictionary))

        palm_explanation = generate_response_using_palm(prompt)
        #openai_explanation = generate_response_using_openai(prompt)
        return palm_explanation
        #print(prompt)

    else:
        print("The model is not supported at the moment.")




    