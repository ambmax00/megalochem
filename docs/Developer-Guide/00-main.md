# Developer Guide

## Building in Docker

To ease development, you may use the Docker file in /docker. It generates a container which has all the external libraries installed. 
Build it in its directory
```
docker build --tag "megalochem" .
```
You can then launch a container with
```
docker run -it -v ~/path/to/repo/:/megalochem megalochem /bin/bash
```
And then just run cmake and build it (in a separate directory, but using a volume, so you don't loose your build). 

You can also use the development container extension in vscode.
