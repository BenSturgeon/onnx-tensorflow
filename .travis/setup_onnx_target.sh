export TRAVIS=1

echo "This is a CI test for onnx-tf master with onnx 1.7.0 and tf 1.15.0 release."
docker pull winnietsang/onnx-tensorflow:onnx1.7.0-tf1.15
docker run -t -d --name=test winnietsang/onnx-tensorflow:onnx1.7.0-tf1.15 /bin/bash
