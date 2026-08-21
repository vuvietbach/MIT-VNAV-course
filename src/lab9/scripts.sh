docker build -f- -t orbslam:latest . < Dockerfile_ORBSLAM3
docker build -f- -t kimera:latest . < Dockerfile_KIMERA
./run_docker.sh orbslam:latest
./run_docker.sh kimera:latest