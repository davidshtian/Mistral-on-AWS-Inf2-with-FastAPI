# Mistral on AWS Inf2 with FastAPI
Use FastAPI to quickly host serving of Mistral model on AWS Inferentia2 instance Inf2 🚀
Support Multimodal input type (input_embeds) 🖼️

![image](https://github.com/davidshtian/Mistral-on-AWS-Inf2-with-FastAPI/assets/14228056/94f8aa15-6851-41d5-b89e-2b8699949fef)


## Environment Setup
Follow the instructions in Neuron docs [Pytorch Neuron Setup](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-setup.html) for basic environment setup. 

## Install Packages
Go to the virtual env and install the extra packages.
```
pip install -r requirements.txt
```

## Run the App
```
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Send the Request
Test via the input_ids (normal prompt) version:
```
python client.py
```

Test via the input_embeds (common multimodal input, skip embedding layer) version:
```
python embeds_client.py
```
