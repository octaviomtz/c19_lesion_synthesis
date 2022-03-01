# streamlit api for lesions synthesis
We create a docker image that can be used in the google cloud platform
![c19_api_preview](/github_images/c19_api_preview.jpeg?raw=true)

## Notes:
To work with docker in windows, we need to change to WSL2 (ubuntu & git at the moment are installed for WSL1). To do this, first open powershell and check the WSL versions and then change the version.
```bash
wsl -l -v # to check versions
wsl -s docker-desktop # to change version
```

## Working method
Based on https://towardsdatascience.com/deploy-a-dockerized-streamlit-app-to-gcp-with-compute-engine-9b82fd2cdd28   
Do not expose a specific PORT in Dockerfile but call the streamlit application like this:
```dockerfile
CMD streamlit run app.py --server.port $PORT
``` 
Then build the image and run the app on port 8080 inside the container and port 80 outside of the container.
```bash
docker build . -t st_c19_tds
docker run -p 80:8080 -e PORT=8080 st_c19_tds
```
Then, in gcloud  console submit the build to the GCP contrainer registry:
```bash
gcloud builds submit --tag gcr.io/<project-name>/my-app
```
Then, on GCP, in compute engine create a new VM instance.   
1. Select the checkbox “Deploy a container image to this VM instance.”   
1. Allow HTTP traffic by clicking the box   

To deploy, go to gcloud console and run 
```bash
docker run -p 80:8080 -e PORT=8080 gcr.io/<project-name>/my-app
```
Back on GCP we select the VM instance we created and open a SSH in the browser window (dropdown menu next SSH)
and we run the the docker container
```bash
# we are in the SSH console open in GCP
docker run -p 80:8080 -e PORT=8080 gcr.io/<project-name>/my-app
```
We can find the external IP of the app in the VM instance information. 


## Previous method

To build and run the docker image run:
```bash
docker build . -t api_name
docker run -p 8501:8501 -d api_name:latest
```

To deploy the image to GCP first tag the image with the corresponding GCP project:
```bash
docker tag <imageID> gcr.io/<GCP-project>/<name-of-the-image:tag>
```
Then, make sure you are in the right GCP project:
```bash
gcloud config get-value project
```
And that you have the right credentials for gcloud and that docker is configured:
```bash
gcloud auth login
gcloud auth configure-docker
```
Then you should be able to push
```bash
docker push gcr.io/<GCP-project>/<name-of-the-image:tag>
# docker push gcr.io/st-lung-lesions-v0/st-lung-lesions-v00:latest #example
```

## Local development
'/content/drive/MyDrive/repositories/streamlit/c19_lesion_synthesis/streamlit_from_colab'

For more info
https://www.youtube.com/watch?v=s46l_PmXBAQ   
https://www.youtube.com/watch?v=3OP-q55hOUI