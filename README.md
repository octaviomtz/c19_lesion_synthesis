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

