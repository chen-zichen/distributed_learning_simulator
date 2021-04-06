python3 ./simulator.py --dataset_name MNIST --model_name LeNet5 --distributed_algorithm  multiround_shapley_value --worker_number 4 --epoch 2 --round 10 --log_level INFO --learning_rate 0.01
# python3 ./simulator.py --dataset_name MNIST --model_name LeNet5 --distributed_algorithm sign_SGD --worker_number 2 --round 1  --optimizer_name SGD  --epoch 1 --log_level INFO --learning_rate 0.01
