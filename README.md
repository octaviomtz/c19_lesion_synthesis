# streamlit api for lesions synthesis

## Notes:
To work with docker in windows, we need to change to WSL2 (ubuntu & git at the moment are installed for WSL1). To do this, first open powershell and check the WSL versions and then change the version.
```bash
wsl -l -v # to check versions
wsl -s docker-desktop # to change version
```
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

For more info
https://www.youtube.com/watch?v=s46l_PmXBAQ
https://www.youtube.com/watch?v=3OP-q55hOUI