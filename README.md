# Document Summarization with Gemini from Vertex AI
This Accelerator for Machine Learning Projects ("AMP") allows users to summarize documents and text using Google's Gemini models from the Vertex AI Model Garden. It provides two summarization modes: text-based and document-based with document summarization supported through LlamaIndex as the vector store.

![](/assets/readme-header.png)

## Architecture and Overview

### 2 Modes of Summarization
This AMP supports two modes of document summarization: text-based summarization and document-based summarization. For document summarization, upload and remove documents in the 'Manage Vector Store' tab or through the CML Job 'Load Documents in docs folder to LlamaIndex vector store'. As is documented in the Supported Gemini Models below, there are several Gemini models to choose from depending on the desired output. Max Output Tokens and Temperature may be adjusted for lengthier responses and level of randomness in response, respectively.

#### Text-based Summarization
![](/assets/screenshot-summarize-from-text-input.png)

#### Document-based Summarization
![](/assets/screenshot-summarize-from-doc-library.png)

![](/assets/screenshot-manage-vector-store.png)

### Supported Gemini Models
Several Gemini models are tested and supported and more may be added by adjusting the below parameter in the `3_application/app.py` file:
```
ALLOWED_MODELS = [
    "models/gemini-1.0-pro-latest",
    "models/gemini-1.0-pro",
    "models/gemini-pro",
    "models/gemini-1.0-pro-001",
    "models/gemini-1.5-pro-001",
    "models/gemini-1.5-pro-latest"
]
```
![](/assets/screenshot-multiple-models.png)


## Deployment

### AMP Deployment Methods
There are two ways to launch this prototype on CML:

1. **From Prototype Catalog** - Navigate to the Prototype Catalog on a CML workspace, select the "Document Summarization with Gemini from Vertex AI" tile, click "Launch as Project", click "Configure Project".

2. **As ML Prototype** - In a CML workspace, click "New Project", add a Project Name, select "ML Prototype" as the Initial Setup option, copy in the [repo URL](https://github.com/cloudera/CML_AMP_Summarization_with_Vertex_AI_Gemini), click "Create Project", click "Configure Project".

### AMP Deployment
In both cases, you will need to specify the `GOOGLE_API_KEY` *(steps in next section on how to create this)* which enables the connection between Google's Vertex AI API and the Application in CML.

![](/assets/screenshot-setup-amp.png)

![](/assets/screenshot-amp-creation-script.png)

## Requirements

### Setup API Key with Access to Gemini

#### 1. Enable Vertex AI API
If you have not already, navigate to the [Vertex AI Marketplace](https://console.cloud.google.com/marketplace/product/google/aiplatform.googleapis.com) and enable the Vertex AI Platform API from your Marketplace. Once complete, your entry should show like below.

![](/assets/enable-vertex-ai-marketplace.png)

#### 2. Generate API Key
From the marketplace entry above, click "Manage" and navigate to "Credentials". Here you will click "Create Credentials" and save the API key value which appears (we will use this as an environment variable when deploying the AMP). 

![](/assets/select-credentials.png)

![](/assets/select-credentials-dropdown.png)

![](/assets/create-api-key.png)

#### 3. Enable Gemini Model
Gemini will need to be enabled for the Project space you created the API key in above. If this has not been done already, you should do this for the project the API key resides in. The UI will also share an error message with where to enable the model if required.

#### Recommended Runtime
JupyterLab - Python 3.11 - Standard - 2024.05

#### Resource Requirements
This AMP creates the following workloads with resource requirements:
- CML Session: `2 CPU, 8GB MEM`
- CML Jobs: `2 CPU, 8GB MEM`
- CML Application: `2 CPU, 8GB MEM`

#### External Resources
This AMP requires pip packages and models from huggingface. Depending on your CML networking setup, you may need to whitelist some domains:
- pypi.python.org
- pypi.org
- pythonhosted.org
- huggingface.co
Additionally, it will require access to Google's Vertex AI API. Please ensure the endpoint you leverage for Gemini is whitelisted as well.

## Technologies Used
#### Models and Utilities
- [Gemini](https://blog.google/technology/ai/google-gemini-ai/)
     - LLM Model from Google's Vertex AI Model Garden 
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
     - Vector Embeddings Generation Model
- [Hugging Face transformers library](https://pypi.org/project/transformers/)
#### Vector Store
- [Llama Index](https://docs.llamaindex.ai/en/stable/module_guides/indexing/vector_store_index/)
#### Chat Frontend
- [Streamlit](https://github.com/streamlit)

## Deploying on CML
There are two ways to launch this prototype on CML:

1. **From Prototype Catalog** - Navigate to the Prototype Catalog on a CML workspace, select the "Intelligent QA Chatbot with NiFi, Pinecone, and Llama2" tile, click "Launch as Project", click "Configure Project"

2. **As ML Prototype** - In a CML workspace, click "New Project", add a Project Name, select "ML Prototype" as the Initial Setup option, copy in the [repo URL](https://github.com/cloudera/CML_AMP_Summarization_with_Vertex_AI_Gemini), click "Create Project", click "Configure Project"


## The Fine Print

IMPORTANT: Please read the following before proceeding.  This AMP includes or otherwise depends on certain third party software packages.  Information about such third party software packages are made available in the notice file associated with this AMP.  By configuring and launching this AMP, you will cause such third party software packages to be downloaded and installed into your environment, in some instances, from third parties' websites.  For each third party software package, please see the notice file and the applicable websites for more information, including the applicable license terms.

If you do not wish to download and install the third party software packages, do not configure, launch or otherwise use this AMP.  By configuring, launching or otherwise using the AMP, you acknowledge the foregoing statement and agree that Cloudera is not responsible or liable in any way for the third party software packages.


Refer to the Project **NOTICE** and **LICENSE** files in the root directory. Author: Cloudera Inc.
