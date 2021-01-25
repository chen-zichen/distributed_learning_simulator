# python3 ./simulator.py --dataset_name CIFAR10 --model_name MobileNet --algorithm sign_SGD  --worker_number 2 --learning_rate 0.0001
# python3 ./simulator.py --dataset_name MNIST --model_name LeNet5 --algorithm fed_quant --worker_number 3 --epoch 2 --local_epoch 2 --optimizer SGD
python3 ./simulator.py --dataset_name MNIST --model_name LeNet5 --algorithm fed_quant --worker_number 3 --epoch 1 --local_epoch 1 --optimizer SGD
