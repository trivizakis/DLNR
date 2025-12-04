FROM python:3.9.7

WORKDIR /denoiser

RUN mkdir /denoiser/input_data
RUN mkdir /denoiser/output_data

COPY denoiser.h5 ./denoiser.h5
COPY denoiser.py ./denoiser.py
COPY hypes_handler.py ./hypes_handler.py
COPY model_factory.py ./model_factory.py

ENV DEBIAN_FRONTEND noninteractive
ENV TF_CPP_MIN_LOG_LEVEL 3
ENV KMP_AFFINITY noverbose
ENV GIT_PYTHON_REFRESH quiet

#install requires packages
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN python3 -m pip install --upgrade pip
RUN pip install SimpleITK scikit-image mlflow tensorflow==2.8 protobuf==3.19.4

#absl-py==1.0.0 alabaster==0.7.12 appdirs==1.4.4 arrow==0.13.1 astroid==2.6.6 asttokens==2.0.5 astunparse==1.6.3 atomicwrites==1.4.0 attrs==21.4.0 autopep8==1.6.0 babel==2.9.1 backcall==0.2.0 binaryornot==0.4.4 black==19.10b0 bleach==4.1.0 brotlipy==0.7.0 cachetools==5.0.0 certifi==2021.10.8 cffi==1.15.0 chardet==4.0.0 charset-normalizer==2.0.4 click==8.0.4 cloudpickle==2.0.0 colorama==0.4.4 cookiecutter==1.7.2 cryptography==36.0.0 cycler==0.11.0 cytoolz==0.11.0 dask==2022.2.1 debugpy==1.5.1 decorator==5.1.1 defusedxml==0.7.1 diff-match-patch==20200713 docutils==0.17.1 entrypoints==0.3 executing==0.8.3 flake8==3.9.2 flatbuffers==2.0 fonttools==4.31.1 fsspec==2022.2.0 gast==0.5.3 google-auth-oauthlib==0.4.6 google-auth==2.6.2 google-pasta==0.2.0 grpcio==1.44.0 h5py==3.6.0 idna==3.3 imagecodecs==2021.8.26 imageio==2.9.0 imagesize==1.3.0 importlib-metadata==4.8.2 inflection==0.5.1 intervaltree==3.1.0 ipykernel==6.9.1 ipython-genutils==0.2.0 ipython==8.1.1 isort==5.9.3 jedi==0.18.1 jeepney==0.7.1 jinja2-time==0.2.0 jinja2==2.11.3 joblib==1.1.0 jsonschema==3.2.0 jupyter-client==6.1.12 jupyter-core==4.9.2 jupyterlab-pygments==0.1.2 keras-preprocessing==1.1.2 keras==2.8.0 keyring==23.4.0 kiwisolver==1.4.0 lazy-object-proxy==1.6.0 libclang==13.0.0 locket==0.2.1 markdown==3.3.6 markupsafe==1.1.1 matplotlib-inline==0.1.2 matplotlib==3.5.1 mccabe==0.6.1 mistune==0.8.4 mkl-fft==1.3.1 mkl-random==1.2.2 mkl-service==2.4.0 mypy-extensions==0.4.3 nbclient==0.5.11 nbconvert==6.3.0 nbformat==5.1.3 nest-asyncio==1.5.1 networkx==2.7.1 nibabel==3.2.2 numpy==1.22.3 numpydoc==1.2 oauthlib==3.2.0 opt-einsum==3.3.0 packaging==21.3 pandocfilters==1.5.0 parso==0.8.3 partd==1.2.0 pathspec==0.7.0 pexpect==4.8.0 pickleshare==0.7.5 pillow==9.0.1 pip==21.2.4 pluggy==1.0.0 poyo==0.5.0 prompt-toolkit==3.0.20 protobuf==3.19.4 psutil==5.8.0 ptyprocess==0.7.0 pure-eval==0.2.2 pyasn1-modules==0.2.8 pyasn1==0.4.8 pycodestyle==2.7.0 pycparser==2.21 pydocstyle==6.1.1 pyflakes==2.3.1 pygments==2.11.2 pylint==2.9.6 pyls-spyder==0.4.0 pyopenssl==22.0.0 pyparsing==3.0.4 pyrsistent==0.18.0 pysocks==1.7.1 python-dateutil==2.8.2 python-lsp-black==1.0.0 python-lsp-jsonrpc==1.0.0 python-lsp-server==1.2.4 python-slugify==5.0.2 pytz==2021.3 pywavelets==1.3.0 pyxdg==0.27 pyyaml==6.0 pyzmq==22.3.0 qdarkstyle==3.0.2 qstylizer==0.1.10 qtawesome==1.0.3 qtconsole==5.2.2 qtpy==1.11.2 regex==2021.11.2 requests-oauthlib==1.3.1 requests==2.27.1 rope==0.22.0 rsa==4.8 rtree==0.9.7 scikit-image==0.19.2 scikit-learn==1.0.2 scipy==1.8.0 secretstorage==3.3.1 setuptools==58.0.4 sip==4.19.13 six==1.16.0 sklearn==0.0 snowballstemmer==2.2.0 sortedcontainers==2.4.0 sphinx==4.4.0 sphinxcontrib-applehelp==1.0.2 sphinxcontrib-devhelp==1.0.2 sphinxcontrib-htmlhelp==2.0.0 sphinxcontrib-jsmath==1.0.1 sphinxcontrib-qthelp==1.0.3 sphinxcontrib-serializinghtml==1.1.5 spyder-kernels==2.1.3 spyder==5.1.5 stack-data==0.2.0 tensorboard-data-server==0.6.1 tensorboard-plugin-wit==1.8.1 tensorboard==2.8.0 tensorflow-io-gcs-filesystem==0.24.0 tensorflow==2.8.0 tensorflow-gpu tensorrt termcolor==1.1.0 testpath==0.5.0 text-unidecode==1.3 textdistance==4.2.1 tf-estimator-nightly==2.8.0.dev2021122109 threadpoolctl==3.1.0 three-merge==0.1.1 tifffile==2021.7.2 tinycss==0.4 toml==0.10.2 toolz==0.11.2 tornado==6.1 traitlets==5.1.1 typed-ast==1.4.3 typing-extensions==3.10.0.2 ujson==4.2.0 unidecode==1.2.0 urllib3==1.26.8 watchdog==2.1.6 wcwidth==0.2.5 webencodings==0.5.1 werkzeug==2.0.3 wheel==0.37.1 whichcraft==0.6.1 wrapt==1.12.1 wurlitzer==3.0.2 yapf==0.31.0 zipp==3.7.0

# Run python script
CMD [ "python3", "./denoiser.py"]