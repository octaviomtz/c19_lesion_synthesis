# base image
FROM python:3.7

# streamlit-specific commands
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'
RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'

# expose default port for streamlit
EXPOSE 8501

# copy and install packages
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt

#copy the rest
COPY . .

#run app
CMD stramlit run main.py