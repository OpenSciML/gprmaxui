PROJECT=build-with-ai-project
LOCATION=us-central1
REPOSITORY_NAME=build-with-ai-docker-repo
IMAGE_NAME="$LOCATION-docker.pkg.dev/${PROJECT}/$REPOSITORY_NAME/gprmaxui:latest"

echo "Project: $PROJECT"
echo "Location: $LOCATION"
echo "Repository Name: $REPOSITORY_NAME"
echo "Image Name: $IMAGE_NAME"

#gcloud artifacts repositories create $REPOSITORY_NAME \
#    --repository-format=docker \
#    --location=$LOCATION
echo $IMAGE_NAME
#gcloud auth configure-docker $LOCATION-docker.pkg.dev
#docker build . -t $IMAGE_NAME --file ./docker/Dockerfile
#docker push $IMAGE_NAME