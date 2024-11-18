# Building Docker images and GATE Cloud apps

The stance classifiers are deployed on GATE Cloud via a two step process, first the classifier itself is deployed as a container that exposes an ELG-compliant API endpoint using the Flask python HTTP framework, then the GATE Cloud API endpoint is a simple GATE app that contains an ELG client PR configured to call the Python endpoint.

## Building the Python classifier images

The Python-based classifiers can be built using the `./build.sh` script in this directory.  To publish on the public GATE Cloud the images should be pushed to the `elg.docker.gate.ac.uk` registry:

```
TAG=elg.docker.gate.ac.uk/stance-target-oblivious:latest ./build.sh oblivious
TAG=elg.docker.gate.ac.uk/stance-multilingual:latest ./build.sh ensemble
```

Any arguments following the initial `oblivious` or `ensemble` will be passed through to the `docker buildx` command, allowing you to specify things like `--push`, `--platform`, etc. as required.

The `Dockerfile` has been designed so that the two images will share most of their layers including the (large) Python virtual environment, only the model layer and configuration variables will differ.

## Using custom models

By default, the two build modes `oblivious` and `ensemble` will download the default models from huggingface ([GateNLP/stance-twitter-xlm-target-oblivious](https://huggingface.co/GateNLP/stance-twitter-xlm-target-oblivious) for oblivious mode, and the pair of [GateNLP/stance-bertweet-target-oblivious](https://huggingface.co/GateNLP/stance-bertweet-target-oblivious) and [GateNLP/stance-bertweet-target-aware](https://huggingface.co/GateNLP/stance-bertweet-target-aware) for ensemble mode).  There is also an `aware` mode available if you want to build a service that uses _only_ the target-aware model rather than combining it in ensemble with a target-oblivious one.  To use different models you will need to specify `--build-arg` options for the `MODEL` (for the single-model oblivious/aware builds) or `MODEL_TO` and `MODEL_TA` (for the two models in ensemble mode).  These can be the model repository names if the models are public on the huggingface hub, or for models that are not published to huggingface, create a folder `extra-models` under this `docker` folder and put the model files in there.  This folder will be included into the final image and the models therein can be referenced from a `--build-arg` as `./extra-models/<name>` (with the leading `./`).

## Building the GATE Cloud applications

The GATE applications that call these Python endpoints can be created using the `build_zip.sh` script in the `cloud-applications` folder, passing the URL of the ELG-compliant endpoint they will delegate to - in the case of GATE Cloud the Python classifiers are deployed as cluster-internal knative services, so:

```
cd cloud-applications
./build_zip.sh metadata/english http://stance-english.elg.svc.cluster.local/process stance-english.zip
./build_zip.sh metadata/multilingual http://stance-multilingual.elg.svc.cluster.local/process stance-multi.zip
./build_zip.sh metadata/reply-only_english http://stance-english-reply-only.elg.svc.cluster.local/process stance-target-oblivious.zip
```

This will create zip files with the appropriate `application.xgapp`, `metadata` and the ELG client plugin, ready to deploy to GATE Cloud in the usual way.
